#!/usr/bin/env python3
"""
Build the fixed facial-part template used by DFD-HR + PMIL, check it against the
face crops of every test set, and register it to test sets whose crop protocol
differs from DeepfakeBench (e.g. WDF from another preprocessing pipeline).

The template is the median 81-point landmark shape of FF++ training frames
(h-flip symmetrised, because horizontal flip is a training augmentation),
rasterised into 5 soft part masks on the 32x32 local-token grid.

Registration: test sets without landmarks are aligned at the *dataset* level.
The mean face of N unlabeled frames is matched to the mean face of the FF++
training frames with a scale + translation search; the template landmarks are
moved by that transform and re-rasterised. No labels and no per-image
landmarks are used, and only the PMIL branch sees the result.

Example (the 6 test sets used in the paper)
-------
python3 scripts/build_part_template.py \
    --check_json Celeb-DF-v2 --check_json FaceShifter --check_json DeepFakeDetection \
    --check_json DFDC --check_json DFDCP \
    --check "WDF=/data/hjr/deepfake/testset/WDF/*/*/*/*/*.png"

Outputs
-------
training/config/part_template_32.npy             (5, 32, 32) masks, order = PART_NAMES
training/config/part_template_32_landmarks.npy   (81, 2) normalised template landmarks
training/config/part_template_32_<NAME>.npy      registered masks, only for misaligned test sets
outputs/part_template/masks.png                  the 5 masks
outputs/part_template/check_<NAME>.png           mean face + template (thin) + registered template (thick)
outputs/part_template/registration.json          scale / shift per dataset
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'training' / 'detectors' / 'utils'))
from part_prior import PART_NAMES, build_part_masks, mirror_landmarks  # noqa: E402

COLORS = [(176, 114, 76), (104, 168, 85), (82, 78, 196), (178, 114, 129), (59, 162, 217)]  # BGR
SIZE = 448


# ----------------------------------------------------------------------------- landmarks
def frame_path_of(lm_path):
    return lm_path.replace('landmarks', 'frames').replace('.npy', '.png')


def load_landmarks(pattern, num, frame_size, seed):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'no landmark files match: {pattern}')
    random.Random(seed).shuffle(files)
    shapes, frames, sizes = [], [], {}
    for path in files[:num]:
        lm = np.load(path).astype(np.float32).reshape(-1, 2)
        if lm.shape[0] != 81 or not np.isfinite(lm).all() or lm.max() <= 0:
            continue
        img_path = frame_path_of(path)
        img = cv2.imread(img_path) if Path(img_path).exists() else None
        if img is not None:
            w, h = img.shape[1], img.shape[0]
            frames.append(img_path)
        elif frame_size is not None:
            w = h = frame_size
        else:
            raise SystemExit(f'cannot find the frame of {path}; pass --frame_size')
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
        shapes.append(lm / np.array([w, h], np.float32))
    if not shapes:
        raise SystemExit('no valid 81-point landmarks found')
    print(f'landmarks used: {len(shapes)} / {len(files)} files, frame sizes: {sizes}')
    return np.stack(shapes), frames


# ----------------------------------------------------------------------------- frames
def frames_from_glob(pattern):
    return sorted(glob.glob(pattern))


def frames_from_json(name, json_dir, rgb_root):
    path = Path(json_dir) / f'{name}.json'
    if not path.is_file():
        print(f'  [warn] manifest not found: {path}')
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    found = {'test': [], 'any': []}

    def walk(node, in_test):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'frames' and isinstance(value, list):
                    found['test' if in_test else 'any'].extend(value)
                else:
                    walk(value, in_test or key == 'test')

    walk(data, False)
    frames = found['test'] or found['any']
    root = Path(rgb_root)
    out = []
    for f in frames:
        p = Path(str(f).replace('\\', '/'))
        out.append(str(p if p.is_absolute() else root / p))
    return sorted(set(out))


def mean_face(files, num, seed):
    files = list(files)
    random.Random(seed).shuffle(files)
    acc, n = np.zeros((SIZE, SIZE, 3), np.float64), 0
    for path in files[:num]:
        img = cv2.imread(path)
        if img is None:
            continue
        acc += cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        n += 1
    if n == 0:
        return None, 0
    return (acc / n).astype(np.uint8), n


# ----------------------------------------------------------------------------- registration
def _features(img):
    """High-pass grey image: keeps eyes / nose / mouth, removes skin tone and lighting."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 12)
    return (g - g.mean()) / (g.std() + 1e-6)


def _scale_shift_matrix(s, dx, dy):
    c = SIZE / 2
    return np.float32([[s, 0, (1 - s) * c + dx], [0, s, (1 - s) * c + dy]])


def _ncc(a, b, weight):
    a = (a - (a * weight).sum() / weight.sum()) * weight
    b = (b - (b * weight).sum() / weight.sum()) * weight
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def register(ref_img, tgt_img, scales=np.arange(0.70, 1.45, 0.01)):
    """Find (s, dx, dy) with tgt(x) ~ ref(M^-1 x), M = scale s about the centre + shift (pixels at SIZE)."""
    ref, tgt = _features(ref_img), _features(tgt_img)
    win = cv2.createHanningWindow((SIZE, SIZE), cv2.CV_32F)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    weight = np.exp(-(((xx - SIZE / 2) / (0.30 * SIZE)) ** 2 + ((yy - SIZE / 2) / (0.36 * SIZE)) ** 2))
    weight = weight.astype(np.float32)
    best = (-2.0, 1.0, 0.0, 0.0)
    for s in scales:
        warped = cv2.warpAffine(ref, _scale_shift_matrix(s, 0, 0), (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
        (dx, dy), _ = cv2.phaseCorrelate(warped * win, tgt * win)
        if abs(dx) > SIZE * 0.25 or abs(dy) > SIZE * 0.25:
            continue
        cand = cv2.warpAffine(ref, _scale_shift_matrix(s, dx, dy), (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
        score = _ncc(cand, tgt, weight)
        if score > best[0]:
            best = (score, float(s), float(dx), float(dy))
    return best


def transform_landmarks(lm01, s, dx, dy):
    c = SIZE / 2
    pts = lm01 * SIZE
    pts = s * (pts - c) + c + np.array([dx, dy], np.float32)
    return (pts / SIZE).astype(np.float32)


# ----------------------------------------------------------------------------- drawing
def draw_outlines(img, masks, thickness=2):
    out = img.copy()
    size = img.shape[0]
    for k, m in enumerate(masks):
        up = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLORS[k], thickness)
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
                    help='frames of a dataset given by a glob (repeatable)')
    ap.add_argument('--check_json', action='append', default=[], metavar='NAME',
                    help='frames of a dfbench dataset read from its JSON manifest (test split, repeatable)')
    ap.add_argument('--json_dir', default='/data/hjr/dfbench/dataset_json')
    ap.add_argument('--rgb_root', default='/data/hjr/dfbench/rgb')
    ap.add_argument('--check_num', type=int, default=1000)
    ap.add_argument('--no_register', action='store_true', help='only draw the checks, never write registered templates')
    ap.add_argument('--reg_scale_tol', type=float, default=0.04, help='|s-1| above this -> write a registered template')
    ap.add_argument('--reg_shift_tol', type=float, default=0.5, help='shift (in tokens of the 32-grid) above this -> register')
    ap.add_argument('--min_ncc', type=float, default=0.15, help='below this the registration is not trusted')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # ---- template from FF++ training landmarks
    shapes, ref_frames = load_landmarks(args.landmark_glob, args.num, args.frame_size, args.seed)
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

    # ---- reference mean face = the frames the landmarks belong to
    ref_face, n_ref = mean_face(ref_frames, args.check_num, args.seed)
    if ref_face is None:
        raise SystemExit('could not read the FF++ frames next to the landmarks; registration needs them')
    cv2.imwrite(str(preview / 'check_FF-train.png'), draw_outlines(ref_face, masks))
    print(f'reference: mean of {n_ref} FF++ training frames -> {preview / "check_FF-train.png"}')

    checks = [(n, frames_from_json(n, args.json_dir, args.rgb_root)) for n in args.check_json]
    for item in args.check:
        name, pattern = item.split('=', 1)
        checks.append((name, frames_from_glob(pattern)))

    # remove stale registered templates of the datasets we are re-checking
    for name, _ in checks:
        stale = out.with_name(f'{out.stem}_{name}.npy')
        if stale.exists():
            stale.unlink()

    report = {}
    token = SIZE / args.grid
    print(f'\n{"dataset":20s} {"frames":>7s} {"scale":>6s} {"dx(tok)":>8s} {"dy(tok)":>8s} {"ncc":>6s}  action')
    for name, files in checks:
        face, n = mean_face(files, args.check_num, args.seed)
        if face is None:
            print(f'{name:20s} {"0":>7s}  [warn] no readable frames')
            continue
        score, s, dx, dy = register(ref_face, face)
        misaligned = abs(s - 1) > args.reg_scale_tol or max(abs(dx), abs(dy)) / token > args.reg_shift_tol
        action = 'aligned'
        if misaligned and score < args.min_ncc:
            misaligned, action = False, f'UNRELIABLE (ncc<{args.min_ncc}), kept FF++ template'
        img = draw_outlines(face, masks, thickness=1)
        if misaligned and not args.no_register:
            reg_lm = transform_landmarks(template, s, dx, dy)
            reg_masks = build_part_masks(reg_lm, grid=args.grid, boundary_width=args.boundary_width)
            reg_path = out.with_name(f'{out.stem}_{name}.npy')
            np.save(reg_path, reg_masks)
            img = draw_outlines(img, reg_masks, thickness=3)
            action = f'REGISTERED -> {reg_path.name}'
        elif misaligned:
            action = 'misaligned (not registered: --no_register)'
        cv2.imwrite(str(preview / f'check_{name}.png'), img)
        report[name] = {'frames': n, 'scale': s, 'dx_px448': dx, 'dy_px448': dy,
                        'dx_tokens': dx / token, 'dy_tokens': dy / token, 'ncc': score, 'action': action}
        print(f'{name:20s} {n:7d} {s:6.3f} {dx / token:8.2f} {dy / token:8.2f} {score:6.3f}  {action}')
    (preview / 'registration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nchecks and registration report in {preview}')
    print('Thin outlines = FF++ template, thick outlines = registered template (only for misaligned datasets).')


if __name__ == '__main__':
    main()
