#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn import metrics

from training.deepfake_folder_eval import discover_records, sample_frames
from training.output_paths import default_run_id, run_output_dir, validate_run_id


def _eer(y_true, y_score):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


def calculate_metrics(predictions, labels, video_ids):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    frame_fpr, frame_tpr, _ = metrics.roc_curve(labels, predictions, pos_label=1)
    frame_acc = float(((predictions > 0.5).astype(np.int64) == labels).mean())
    grouped = defaultdict(list)
    for prediction, label, video_id in zip(predictions, labels, video_ids):
        grouped[video_id].append((float(prediction), int(label)))
    video_predictions = []
    video_labels = []
    for frames in grouped.values():
        video_predictions.append(sum(prediction for prediction, _ in frames) / len(frames))
        video_labels.append(int(round(sum(label for _, label in frames) / len(frames))))
    video_predictions = np.asarray(video_predictions, dtype=np.float64)
    video_labels = np.asarray(video_labels, dtype=np.int64)
    video_fpr, video_tpr, _ = metrics.roc_curve(video_labels, video_predictions, pos_label=1)
    video_acc = float(((video_predictions > 0.5).astype(np.int64) == video_labels).mean())
    return {
        'acc': frame_acc,
        'auc': float(metrics.auc(frame_fpr, frame_tpr)),
        'eer': _eer(labels, predictions),
        'ap': float(metrics.average_precision_score(labels, predictions)),
        'video_auc': float(metrics.auc(video_fpr, video_tpr)),
        'video_eer': _eer(video_labels, video_predictions),
        'video_acc': video_acc,
        'num_frames': int(len(labels)),
        'num_videos': int(len(video_labels)),
        'real_videos': int((video_labels == 0).sum()),
        'fake_videos': int((video_labels == 1).sum()),
    }


def load_model(detector_path, test_config_path, checkpoint_path, device):
    import torch
    import yaml
    from training.detectors import DETECTOR

    with Path(detector_path).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    with Path(test_config_path).open(encoding='utf-8') as handle:
        config.update(yaml.safe_load(handle))
    config['frame_num']['test'] = 32
    model = DETECTOR[config['model_name']](config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    normalized = {}
    for key, value in state_dict.items():
        key = key.replace('module.', '')
        key = key.replace('base_model.', 'backbone.')
        key = key.replace('classifier.', 'head.')
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)
    model.eval()
    return model, config


def load_image(path, config, device):
    import cv2
    import torch

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unable to read image: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config['resolution'], config['resolution']), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
    mean = torch.tensor(config['mean'], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config['std'], dtype=tensor.dtype).view(3, 1, 1)
    return ((tensor - mean) / std).to(device)


def run_worker(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the four-GPU evaluator')
    device = torch.device('cuda')
    model, config = load_model(args.detector_path, args.test_config, args.checkpoint, device)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for dataset_name in args.datasets:
        records = discover_records(Path(args.root), dataset_name)
        samples = []
        for record in records:
            for frame in sample_frames(record.frames, args.frame_num):
                samples.append((frame, record.label, record.video_id))
        predictions = []
        labels = []
        video_ids = []
        total_iterations = (len(samples) + args.batch_size - 1) // args.batch_size
        for iteration, start in enumerate(range(0, len(samples), args.batch_size), 1):
            batch = samples[start:start + args.batch_size]
            images = torch.stack([load_image(frame, config, device) for frame, _, _ in batch])
            batch_labels = torch.tensor([label for _, label, _ in batch], dtype=torch.long, device=device)
            with torch.no_grad():
                output = model({'image': images, 'label': batch_labels, 'mask': None, 'landmark': None}, inference=True)
            predictions.extend(output['prob'].detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())
            video_ids.extend(video_id for _, _, video_id in batch)
            if not args.quiet:
                print(f'[{dataset_name}] iter={iteration}/{total_iterations} samples={len(batch)}', flush=True)
        all_results[dataset_name] = calculate_metrics(predictions, labels, video_ids)
        result_path = output_dir / f'{args.run_id}_{dataset_name}.json'
        payload = {
            'run_id': args.run_id,
            'source': 'deepfake',
            'source_type': args.source_type,
            'dataset': dataset_name,
            'dataset_root': str(Path(args.root).resolve()),
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'frame_num': args.frame_num,
            'batch_size': args.batch_size,
            'metrics': all_results[dataset_name],
        }
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def aggregate(args):
    output_dir = resolve_output_dir(args)
    results = {}
    for path in sorted(output_dir.glob(f'{args.run_id}_*.json')):
        if path.name.endswith('_summary.json'):
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        results[payload['dataset']] = payload
    summary = {
        'run_id': args.run_id,
        'source': 'deepfake',
        'dataset_root': str(Path(args.root).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'frame_num': args.frame_num,
        'datasets': results,
    }
    path = output_dir / f'{args.run_id}_summary.json'
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    print(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/hjr/deepfake/testset')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--checkpoint', default='./checkpoints/ckpt_best.pth')
    parser.add_argument('--detector-path', default='./training/config/detector/dfd_hr.yaml')
    parser.add_argument('--test-config', default='./training/config/test_config.yaml')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--frame-num', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--source-type', default='folder')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    args.run_id = validate_run_id(args.run_id or default_run_id())
    return args


def resolve_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)
    return run_output_dir(Path(__file__).resolve().parents[1], 'eval', args.run_id)


if __name__ == '__main__':
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_worker(args)
