"""
eval pretained model.
"""
import os
import json
from pathlib import Path
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics, summarize_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from output_paths import default_run_id, run_output_dir, validate_run_id
from config_utils import load_detector_config

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/dfd_hr.yaml',
                    help='path to detector YAML file')
parser.add_argument('--test_config', type=str,
                    default='./training/config/test_config.yaml',
                    help='path to testing dataset configuration YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='./logs/dfd_hr/ckpt_best.pth')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory for JSON evaluation results (default: ./outputs/dfbench/eval/<run_id>)')
parser.add_argument('--run_id', type=str, default=None,
                    help='run directory/name shared by logs and result JSON')
parser.add_argument('--output_group', type=str, default='dfbench',
                    help='output namespace (default: dfbench)')
parser.add_argument('--source_type', type=str, default='json')
parser.add_argument('--quiet', action='store_true')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

on_2060 = "2060" in torch.cuda.get_device_name()
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader, quiet=False):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), disable=quiet):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders, quiet=False):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key], quiet=quiet)
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        if not quiet:
            tqdm.write(f"dataset: {key}")
            for k, v in metric_one_dataset.items():
                tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def save_test_results(metrics_all_datasets, config, weights_path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_output_dir(project_root, 'eval', args.run_id, output_group=args.output_group)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = '_'.join(config['test_dataset'])
    output_path = output_dir / f'test_metrics_{args.run_id}_{dataset_slug}.json'
    payload = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'checkpoint': str(Path(weights_path).resolve()) if weights_path else None,
        'source_type': config.get('source_type', 'json'),
        'source': config.get('dataset_json_folder'),
        'datasets': config['test_dataset'],
        'mode': config['mode'],
        'lmdb': config['lmdb'],
        'dataset_root_rgb': config.get('dataset_root_rgb'),
        'dataset_json_folder': config.get('dataset_json_folder'),
        'frame_num_test': config['frame_num']['test'],
        'test_batch_size': config['test_batchSize'],
        'metrics': summarize_test_metrics(metrics_all_datasets),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return output_path


def main():
    # parse options and load config
    config = load_detector_config(args.detector_path)
    with open(args.test_config, 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    args.run_id = validate_run_id(args.run_id or os.environ.get('RUN_ID') or default_run_id())
    config['run_id'] = args.run_id
    config['output_group'] = args.output_group
    config['source_type'] = args.source_type
    config['quiet'] = args.quiet
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
        config['workers'] = 0
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    model.eval()
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')
            if 'base_model.' in new_key:
                new_key = new_key.replace('base_model.', 'backbone.')
            if 'classifier.' in new_key:
                new_key = new_key.replace('classifier.', 'head.')
            new_weights[new_key] = value
        

        model.load_state_dict(new_weights, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, quiet=config['quiet'])
    result_path = save_test_results(best_metric, config, weights_path)
    print(f'===> Saved test metrics to {result_path}')
    print('===> Test Done!')

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python3 scripts/test.py --detector_path ./training/config/detector/dfd_hr.yaml --test_dataset Celeb-DF-v2 FaceShifter DeepFakeDetection DFDC DFDCP test_DFR test_WDF test_FFIW uniface_ff blendface_ff mobileswap_ff e4s_ff facedancer_ff fsgan_ff inswap_ff simswap_ff --weights_path logs/dfd-hr/dfd_hr_2025-10-24-01-19-06/test/avg/ckpt_best.pth
    main()
