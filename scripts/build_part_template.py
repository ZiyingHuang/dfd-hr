#!/usr/bin/env python3
"""
Build the fixed facial-part template used by DFD-HR + PMIL, and check that it
fits the face crops of every dataset you will test on.

The template is the median 81-point landmark shape of FF++ training frames
(h-flip symmetrised, because horizontal flip is a training augmentation),
rasterised into 5 soft part masks on the 32x32 local-token grid.

Example
-------
python3 scripts/build_part_template.py \
    --landmark_glob "/data/hjr/dfbench/rgb/FaceForensics++/original_sequences/youtube/c23/landmarks/*/*.npy" \
    --check "FF++=/data/hjr/dfbench/rgb/FaceForensics++/original_sequences/youtube/c23/frames/*/*.png" \
    --check "CDF=/data/hjr/dfbench/rgb/Celeb-DF-v2/Celeb-real/frames/*/*.png" \
    --check "WDF=/data/hjr/deepfake/testset/WDF/*/*/*/*/*.png"

Outputs
-------
training/config/part_template_32.npy            (5, 32, 32) masks, order = PART_NAMES
training/config/part_template_32_landmarks.npy  (81, 2) normalised template landmarks
outputs/part_template/masks.png                 the 5 masks
outputs/part_template/check_<name>.png          mean face of each dataset + part outlines
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'training' / 'detectors' / 'utils'))
from part_prior import PART_NAMES, build_part_masks, mirror_landmarks  # noqa: E402

COLORS = [(176, 114, 76), (104, 168, 85), (82, 78, 196), (178, 114, 129), (59, 162, 217)]  # BGR


def frame_size_of(lm_path, default):
    img_path = lm_path.replace('landmarks', 'frames').replace('.npy', '.png')
    img = cv2.imread(img_path) if Path(img_path).exists() else None
    if img is not None:
        return img.shape[1], img.shape[0]
    if default is None:
        raise FileNotFoundError(f'cannot find the frame of {lm_path}; pass --frame_size')
    return default, default


def load_landmarks(pattern, num, frame_size, seed):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'no landmark files match: {pattern}')
    random.Random(seed).shuffle(files)
    shapes, sizes = [], {}
    for path in files[:num]:
        lm = np.load(path).astype(np.float32).reshape(-1, 2)
        if lm.shape[0] != 81 or not np.isfinite(lm).all() or lm.max() <= 0:
            continue
        w, h = frame_size_of(path, frame_size)
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
        shapes.append(lm / np.array([w, h], np.float32))
    if not shapes:
        raise SystemExit('no valid 81-point landmarks found')
    print(f'landmarks used: {len(shapes)} / {len(files)} files, frame sizes: {sizes}')
    return np.stack(shapes)


def mean_face(pattern, num, seed, size=448):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'  [warn] no frames match {pattern}')
        return None, 0
    random.Random(seed).shuffle(files)
    acc, n = np.zeros((size, size, 3), np.float64), 0
    for path in files[:num]:
        img = cv2.imread(path)
        if img is None:
            continue
        acc += cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        n += 1
    return (acc / max(n, 1)).astype(np.uint8), n


def draw_outlines(img, masks):
    out = img.copy()
    size = img.shape[0]
    for k, m in enumerate(masks):
        up = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLORS[k], 2)
    for k, name in enumerate(PART_NAMES):
        cv2.putText(out, name, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[k], 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmark_glob', default='/data/hjr/dfbench/rgb/FaceForensics++/original_sequences/youtube/c23/landmarks/*/*.npy')
    ap.add_argument('--out', default=str(PROJECT_ROOT / 'training/config/part_template_32.npy'))
    ap.add_argument('--preview_dir', default=str(PROJECT_ROOT / 'outputs/part_template'))
    ap.add_argument('--num', type=int, default=5000)
    ap.add_argument('--frame_size', type=int, default=None, help='only needed if frames are not next to landmarks')
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--boundary_width', type=float, default=0.06)
    ap.add_argument('--check', action='append', default=[], metavar='NAME=GLOB',
                    help='frame glob of a dataset to overlay the template on (repeatable)')
    ap.add_argument('--check_num', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    shapes = load_landmarks(args.landmark_glob, args.num, args.frame_size, args.seed)
    template = np.median(shapes, 0)
    template = 0.5 * (template + mirror_landmarks(template))
    spread = np.median(np.abs(shapes - template), 0).mean()
    print(f'median absolute landmark deviation from template: {spread:.4f} (fraction of image size)')

    masks = build_part_masks(template, grid=args.grid, boundary_width=args.boundary_width)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, masks)
    np.save(out.with_name(out.stem + '_landmarks.npy'), template)
    for name, m in zip(PART_NAMES, masks):
        print(f'  {name:12s} coverage={m.mean():.3f}')
    print(f'saved {out}')

    preview = Path(args.preview_dir)
    preview.mkdir(parents=True, exist_ok=True)
    tiles = [cv2.resize((m * 255).astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST) for m in masks]
    cv2.imwrite(str(preview / 'masks.png'), np.concatenate(tiles, 1))
    for item in args.check:
        name, pattern = item.split('=', 1)
        face, n = mean_face(pattern, args.check_num, args.seed)
        if face is None:
            continue
        path = preview / f'check_{name}.png'
        cv2.imwrite(str(path), draw_outlines(face, masks))
        print(f'  check {name}: mean of {n} frames -> {path}')
    print('Open the check_*.png files: eyes / nose / mouth outlines should sit on the mean face of every dataset.')


if __name__ == '__main__':
    main()
