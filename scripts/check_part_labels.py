#!/usr/bin/env python3
"""
Sanity-check the mask-derived part labels used by PMIL v2 (CPU only, ~1 min).

For each FF++ training subset it samples frames, loads the manipulation mask the
same way the dataset does (frames -> masks), and reports:
  - how many fake frames have a usable (non-empty) mask
  - the fraction of fake frames in which each part is labelled "fake"

python3 scripts/check_part_labels.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART_NAMES = ['eyes_brows', 'nose', 'mouth', 'boundary', 'skin']


def collect(json_path, split):
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    out = defaultdict(list)
    for dataset in data.values():
        for label_name, node in dataset.items():
            def walk(n, in_split):
                if isinstance(n, dict):
                    for k, v in n.items():
                        if k == 'frames' and isinstance(v, list) and in_split:
                            out[label_name].extend(v)
                        else:
                            walk(v, in_split or k == split)
            walk(node, False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='/data/hjr/dfbench/dataset_json/FaceForensics++.json')
    ap.add_argument('--rgb_root', default='/data/hjr/dfbench/rgb')
    ap.add_argument('--template', default=str(PROJECT_ROOT / 'training/config/part_template_32.npy'))
    ap.add_argument('--split', default='train')
    ap.add_argument('--num', type=int, default=300, help='frames per subset')
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--resolution', type=int, default=448)
    args = ap.parse_args()

    pm = np.load(args.template).reshape(len(PART_NAMES), -1).astype(np.float64)
    grid = int(round(pm.shape[1] ** 0.5))
    frames = collect(args.json, args.split)
    root = Path(args.rgb_root)
    print(f'{"subset":12s} {"frames":>6s} {"mask ok":>8s} {"mean area":>9s}  part-fake rate ({", ".join(PART_NAMES)})')
    for label_name in sorted(frames):
        paths = frames[label_name]
        random.Random(0).shuffle(paths)
        ok, areas, labels = 0, [], []
        for f in paths[:args.num]:
            p = Path(str(f).replace('\\', '/').replace('frames', 'masks'))
            p = p if p.is_absolute() else root / p
            m = cv2.imread(str(p), 0) if p.exists() else None
            if m is None:
                continue
            m = cv2.resize(m, (args.resolution, args.resolution)).astype(np.float64) / 255
            tok = cv2.resize(m, (grid, grid), interpolation=cv2.INTER_AREA).reshape(-1)
            if tok.sum() == 0:
                continue
            ok += 1
            areas.append(m.mean())
            labels.append((tok @ pm.T / pm.sum(1)) > args.thr)
        n = min(len(paths), args.num)
        rates = np.mean(labels, 0) if labels else np.zeros(len(PART_NAMES))
        area = np.mean(areas) if areas else 0.0
        print(f'{label_name:12s} {n:6d} {ok:8d} {area:9.3f}  ' + '  '.join(f'{r:.2f}' for r in rates))
    print('\nReal subsets should show "mask ok" = 0 (no mask -> all parts real).')
    print('Fake subsets need "mask ok" close to "frames"; otherwise their part labels are skipped.')


if __name__ == '__main__':
    main()
