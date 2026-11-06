#!/usr/bin/env python3
"""
Diagnose a DFD-HR + PMIL checkpoint: dump per-frame logits of every branch once,
then score any fusion offline (no re-running the network).

  dump     run the model on one dataset, save orig / bag / part logits per frame (.npz)
  analyze  Video/Frame AUC of the original branch, the bag, each part and several fusions,
           plus logit statistics; calibration for z-score fusion comes ONLY from FF++
           (in-domain test split), never from the target test sets.

Use scripts/run_pmil_diagnose.sh to dump all datasets on 4 GPUs and analyze.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['FaceForensics++', 'Celeb-DF-v2', 'FaceShifter', 'DeepFakeDetection', 'DFDC', 'DFDCP', 'WDF']
TARGETS = DATASETS[1:]
SHORT = {'FaceForensics++': 'FF++', 'Celeb-DF-v2': 'CDF-v2', 'FaceShifter': 'FSh',
         'DeepFakeDetection': 'DFD', 'DFDC': 'DFDC', 'DFDCP': 'DFDCP', 'WDF': 'WDF'}
FOLDER_DATASETS = {'WDF'}


# ----------------------------------------------------------------------------- dump
def build_model(detector, test_config, checkpoint, dataset, device, frame_num=32):
    import torch
    import yaml
    from training.config_utils import load_detector_config
    from training.detectors import DETECTOR

    config = load_detector_config(detector)
    with open(test_config, encoding='utf-8') as handle:
        config.update(yaml.safe_load(handle))
    config['frame_num']['test'] = frame_num
    tpl = PROJECT_ROOT / f'training/config/part_template_32_{dataset}.npy'
    if config['model_name'] == 'dfd_hr_pmil' and tpl.is_file():
        config['part_mil']['template_path'] = str(tpl)
        print(f'[{dataset}] registered template {tpl}')
    model = DETECTOR[config['model_name']](config).to(device)
    state = torch.load(checkpoint, map_location=device)
    state = state.get('state_dict', state)
    state = {k.replace('module.', '').replace('base_model.', 'backbone.').replace('classifier.', 'head.'): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def json_loader(config, dataset, batch_size):
    """Exactly the data pipeline of scripts/test.py (same resize / normalisation)."""
    import torch
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

    cfg = dict(config)
    cfg['test_dataset'] = dataset
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode='test')
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=int(cfg.get('workers', 4)),
                                         collate_fn=ds.collate_fn, drop_last=False)
    meta = []
    for path, label in zip(ds.data_dict['image'], ds.data_dict['label']):
        parts = str(path).replace('\\', '/').split('/')
        meta.append((path, int(label != 0), f'{parts[-3]}_{parts[-2]}'))
    return loader, meta


def folder_samples(root, dataset, frame_num):
    from training.deepfake_folder_eval import discover_records, sample_frames

    out = []
    for record in discover_records(Path(root), dataset):
        for frame in sample_frames(record.frames, frame_num):
            out.append((frame, int(record.label), record.video_id))
    return out


def _subsample(samples, max_videos):
    keep = defaultdict(list)
    for s in samples:
        keep[s[2]].append(s)
    vids = sorted(keep)
    vids = [vids[i] for i in np.random.default_rng(0).permutation(len(vids))[:max_videos]]
    return [s for v in vids for s in keep[v]]


def dump(args):
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / 'training'))
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from evaluate_deepfake_testset import load_image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = build_model(args.detector, args.test_config, args.checkpoint, args.dataset, device, args.frame_num)
    def batches():
        if args.dataset in FOLDER_DATASETS:
            samples = folder_samples(args.wdf_root, args.dataset, args.frame_num)
            if args.max_videos:
                samples = _subsample(samples, args.max_videos)
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start:start + args.batch_size]
                yield torch.stack([load_image(p, config, device) for p, _, _ in batch]), batch
        else:
            loader, meta = json_loader(config, args.dataset, args.batch_size)
            pos = 0
            for data_dict in loader:
                n = data_dict['image'].shape[0]
                yield data_dict['image'].to(device), meta[pos:pos + n]
                pos += n
            assert pos == len(meta), (pos, len(meta))

    rows, metas = defaultdict(list), []
    for it, (images, batch) in enumerate(batches()):
        labels = torch.tensor([y for _, y, _ in batch], device=device)
        with torch.no_grad():
            out = model({'image': images, 'label': labels, 'mask': None, 'landmark': None}, inference=True)
        if 'cls_orig' in out:
            orig = out['cls_orig'][:, 1] - out['cls_orig'][:, 0]
            rows['bag'].append(out['bag_logit'].float().cpu().numpy())
            rows['parts'].append(out['part_logits'].float().cpu().numpy())
            att = out['part_attn'].float()
            rows['attn_entropy'].append((-(att.clamp_min(1e-12).log() * att).sum(-1)).cpu().numpy())
        else:
            orig = out['cls'][:, 1] - out['cls'][:, 0]
        rows['orig'].append(orig.float().cpu().numpy())
        metas.extend(batch)
        if it % 50 == 0:
            print(f'[{args.dataset}] {len(metas)} frames', flush=True)
    samples = metas

    out_path = Path(args.out_dir) / f'{args.dataset}.npz'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        labels=np.array([y for _, y, _ in samples]),
                        videos=np.array([v for _, _, v in samples]),
                        **{k: np.concatenate(v) for k, v in rows.items()})
    print(f'saved {out_path}')


# ----------------------------------------------------------------------------- analyze
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float('nan')


def _video(y, s, vids):
    acc_s, acc_y = defaultdict(list), {}
    for yi, si, vi in zip(y, s, vids):
        acc_s[vi].append(si)
        acc_y[vi] = yi
    keys = sorted(acc_s)
    return np.array([acc_y[k] for k in keys]), np.array([np.mean(acc_s[k]) for k in keys])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def analyze(args):
    root = Path(args.out_dir)
    data = {d: dict(np.load(root / f'{d}.npz')) for d in DATASETS if (root / f'{d}.npz').exists()}
    if not data:
        raise SystemExit(f'no dumps in {root}')
    has_parts = 'bag' in next(iter(data.values()))
    if 'FaceForensics++' not in data and has_parts:
        print('[warn] no FF++ dump: z-score fusion is skipped')

    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    # ---- logit statistics
    emit('## Logit statistics (frame level)\n')
    emit('| dataset | orig mean±std (real / fake) | bag mean±std (real / fake) | corr(orig, bag) |')
    emit('|---|---|---|---|')
    for d, z in data.items():
        y = z['labels']
        row = f"| {SHORT[d]} | {z['orig'][y == 0].mean():+.2f}±{z['orig'][y == 0].std():.2f} / " \
              f"{z['orig'][y == 1].mean():+.2f}±{z['orig'][y == 1].std():.2f} |"
        if has_parts:
            row += f" {z['bag'][y == 0].mean():+.2f}±{z['bag'][y == 0].std():.2f} / " \
                   f"{z['bag'][y == 1].mean():+.2f}±{z['bag'][y == 1].std():.2f} | " \
                   f"{np.corrcoef(z['orig'], z['bag'])[0, 1]:.3f} |"
        else:
            row += ' - | - |'
        emit(row)

    # ---- scores to evaluate
    def scorers():
        yield 'orig (DFD-HR branch)', lambda z: sigmoid(z['orig'])
        if not has_parts:
            return
        yield 'bag (PMIL branch)', lambda z: sigmoid(z['bag'])
        yield 'avg logit a=0.5 (= main run)', lambda z: sigmoid(0.5 * z['orig'] + 0.5 * z['bag'])
        for a in (0.1, 0.25):
            yield f'avg logit a={a}', (lambda a: lambda z: sigmoid((1 - a) * z['orig'] + a * z['bag']))(a)
        yield 'avg prob a=0.5', lambda z: 0.5 * sigmoid(z['orig']) + 0.5 * sigmoid(z['bag'])
        if 'FaceForensics++' in data:
            ff = data['FaceForensics++']
            mo, so = ff['orig'].mean(), ff['orig'].std() + 1e-6
            mb, sb = ff['bag'].mean(), ff['bag'].std() + 1e-6
            for a in (0.25, 0.5):
                yield f'z-score (FF++ stats) a={a}', (lambda a: lambda z: sigmoid(
                    (1 - a) * (z['orig'] - mo) / so + a * (z['bag'] - mb) / sb))(a)
        names = ['eyes_brows', 'nose', 'mouth', 'boundary', 'skin']
        k = next(iter(data.values()))['parts'].shape[1]
        names = names[:k] if k <= 5 else [f'part{i}' for i in range(k)]
        for i, n in enumerate(names):
            yield f'part: {n}', (lambda i: lambda z: sigmoid(z['parts'][:, i]))(i)
        yield 'parts mean', lambda z: sigmoid(z['parts'].mean(1))

    for level in ('video', 'frame'):
        emit(f'\n## {level.capitalize()} AUC (%)\n')
        cols = [d for d in DATASETS if d in data]
        emit('| score | ' + ' | '.join(SHORT[d] for d in cols) + ' | Avg(6 targets) |')
        emit('|---|' + '---|' * (len(cols) + 1))
        for name, fn in scorers():
            vals = []
            for d in cols:
                z = data[d]
                s = fn(z)
                if level == 'video':
                    yv, sv = _video(z['labels'], s, z['videos'])
                    vals.append(_auc(yv, sv))
                else:
                    vals.append(_auc(z['labels'], s))
            tgt = [v for d, v in zip(cols, vals) if d in TARGETS]
            avg = np.mean(tgt) if len(tgt) == len(TARGETS) else float('nan')
            emit(f'| {name} | ' + ' | '.join(f'{v * 100:.2f}' for v in vals) + f' | {avg * 100:.2f} |')

    if has_parts:
        emit('\n## Attention entropy per part (nats; uniform over 1024 tokens = 6.93)\n')
        emit('| dataset | ' + ' | '.join(f'q{i}' for i in range(next(iter(data.values()))['attn_entropy'].shape[1])) + ' |')
        emit('|---|' + '---|' * next(iter(data.values()))['attn_entropy'].shape[1])
        for d, z in data.items():
            emit(f'| {SHORT[d]} | ' + ' | '.join(f'{v:.2f}' for v in z['attn_entropy'].mean(0)) + ' |')

    (root / 'diagnosis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nwritten {root / "diagnosis.md"}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--checkpoint', required=True)
    d.add_argument('--detector', default=str(PROJECT_ROOT / 'training/config/detector/dfd_hr_pmil.yaml'))
    d.add_argument('--test_config', default=str(PROJECT_ROOT / 'training/config/test_config.yaml'))
    d.add_argument('--dataset', required=True, choices=DATASETS)
    d.add_argument('--wdf_root', default='/data/hjr/deepfake/testset')
    d.add_argument('--frame_num', type=int, default=32)
    d.add_argument('--max_videos', type=int, default=0, help='folder datasets only; 0 = all videos')
    d.add_argument('--batch_size', type=int, default=32)
    d.add_argument('--out_dir', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    dump(args) if args.cmd == 'dump' else analyze(args)


if __name__ == '__main__':
    main()
