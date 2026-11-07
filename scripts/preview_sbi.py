#!/usr/bin/env python3
"""Render SBI pseudo-fakes from FF++ real training frames (CPU, ~10 s) and report part-label rates.

python3 scripts/preview_sbi.py      -> outputs/sbi_preview/sbi_samples.png
Rows: blended image | mask overlay. Check that the blended region sits on the face part it names.
"""
import argparse
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_part_labels import collect  # noqa: E402

# load the SBI core without importing torch
src = (ROOT / 'training/dataset/sbi_dataset.py').read_text().split('# ' + '-' * 77 + ' dataset')[0]
for line in ('import torch\n', 'from .abstract_dataset import DeepfakeAbstractBaseDataset\n',
             'from .path_utils import resolve_rgb_path\n'):
    src = src.replace(line, '')
SBI = {}
exec(src, SBI)
PARTS = ['eyes_brows', 'nose', 'mouth', 'boundary', 'skin']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='/data/hjr/dfbench/dataset_json/FaceForensics++.json')
    ap.add_argument('--rgb_root', default='/data/hjr/dfbench/rgb')
    ap.add_argument('--template', default=str(ROOT / 'training/config/part_template_32.npy'))
    ap.add_argument('--num', type=int, default=12)
    ap.add_argument('--stats', type=int, default=400, help='samples for the part-label statistics')
    ap.add_argument('--out', default=str(ROOT / 'outputs/sbi_preview/sbi_samples.png'))
    args = ap.parse_args()

    frames = collect(args.json, 'train').get('FF-real', [])
    if not frames:
        raise SystemExit('no FF-real training frames in the manifest')
    random.Random(0).shuffle(frames)
    pm = np.load(args.template).reshape(len(PARTS), -1).astype(np.float64)
    grid = int(round(pm.shape[1] ** 0.5))
    rng = np.random.default_rng(0)
    tiles, regions, labels, missing = [], Counter(), [], 0
    for f in frames:
        if len(labels) >= args.stats:
            break
        p = Path(str(f).replace('\\', '/'))
        p = p if p.is_absolute() else Path(args.rgb_root) / p
        lm_path = Path(str(p).replace('frames', 'landmarks').replace('.png', '.npy'))
        img = cv2.imread(str(p))
        if img is None or not lm_path.exists():
            missing += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm = np.load(lm_path).astype(np.float32).reshape(-1, 2)
        blended, mask, region = SBI['self_blend'](img, lm, rng)
        regions[region] += 1
        m = mask / max(mask.max(), 1e-6)
        tok = cv2.resize(cv2.resize(m, (448, 448)), (grid, grid), interpolation=cv2.INTER_AREA).reshape(-1)
        ratio = tok @ pm.T / pm.sum(1)
        lab = ratio > 0.2
        if not lab.any():
            lab[np.argmax(ratio)] = True
        labels.append((region, lab))
        if len(tiles) < args.num:
            over = img.copy()
            over[..., 1] = np.clip(over[..., 1] * (1 - mask) + 255 * mask, 0, 255).astype(np.uint8)
            t = np.concatenate([blended, over], 0)
            cv2.putText(t, region, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(t)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
    print(f'saved {out}  (frames without landmarks skipped: {missing})')
    print(f'\npart-fake rate per SBI region ({", ".join(PARTS)})')
    for r in sorted(regions):
        rows = np.array([l for rr, l in labels if rr == r], float)
        print(f'  {r:6s} n={len(rows):4d}  ' + '  '.join(f'{v:.2f}' for v in rows.mean(0)))


if __name__ == '__main__':
    main()
