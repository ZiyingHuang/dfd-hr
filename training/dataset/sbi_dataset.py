"""
Self-Blended Images (SBI, Shiohara & Yamasaki, CVPR 2022) as an extra source of fakes.

Every FF++ real training frame additionally yields `ratio` pseudo-fakes per epoch: the frame
is blended with a colour/quality/geometry-perturbed copy of itself inside a deformed facial
region. The blending mask is returned as the sample's manipulation mask, so the part labels
used by PMIL become genuinely region-specific (FF++'s own masks cover the whole face).

Re-implemented with numpy / cv2 only (no external SBI code). The SBI fakes then go through
exactly the same resize, augmentation and normalisation as every other training sample.
"""

import os
import random

import cv2
import numpy as np
import torch

from .abstract_dataset import DeepfakeAbstractBaseDataset
from .path_utils import resolve_rgb_path

# dlib 81-point index groups
JAW = list(range(0, 17))
BROWS = list(range(17, 27))
NOSE = list(range(27, 36))
EYES = list(range(36, 48))
MOUTH = list(range(48, 68))
FOREHEAD = list(range(68, 81))

REGIONS = {
    'face': JAW + FOREHEAD,                         # SBI default: the whole face hull
    'inner': BROWS + EYES + NOSE + MOUTH,           # inner face, no contour
    'lower': JAW[2:15] + NOSE[2:] + MOUTH,          # nose + mouth + lower cheeks
    'mouth': MOUTH + [31, 32, 33, 34, 35],          # mouth area
    'eyes': BROWS + EYES,                           # eyes + brows
}
DEFAULT_REGION_PROBS = {'face': 0.4, 'inner': 0.2, 'lower': 0.15, 'mouth': 0.125, 'eyes': 0.125}


# ----------------------------------------------------------------------------- SBI core
def _hull_mask(lm, idx, shape):
    mask = np.zeros(shape[:2], np.float32)
    hull = cv2.convexHull(np.round(lm[idx]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask


def _elastic(mask, rng, amp=0.015, sigma=0.04):
    """Smooth random displacement field with std = amp * image size (mild, as in SBI)."""
    h, w = mask.shape
    s = max(h, w)

    def field():
        d = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma * s)
        return d / (d.std() + 1e-6) * amp * s

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, xs + field(), ys + field(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _deform_mask(mask, rng):
    h, w = mask.shape
    s = max(h, w)
    mask = _elastic(mask, rng)
    k = int(rng.integers(1, max(2, s // 40)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.dilate(mask, ker) if rng.random() < 0.5 else cv2.erode(mask, ker)
    blur = rng.uniform(0.005, 0.04) * s
    mask = cv2.GaussianBlur(mask, (0, 0), blur)
    m = mask.max()
    return mask / m if m > 1e-6 else mask


def _source_transform(img, rng):
    """Colour + quality perturbation of the source copy (uint8 RGB in, uint8 RGB out)."""
    x = img.astype(np.float32)
    if rng.random() < 0.3:                                              # RGB shift
        x += rng.uniform(-20, 20, 3)
    if rng.random() < 0.3:                                              # hue / saturation / value
        hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.3, 0.3) * 10) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + rng.uniform(-30, 30), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + rng.uniform(-20, 20), 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:                                              # brightness / contrast
        x = (x - 128) * rng.uniform(0.9, 1.1) + 128 + rng.uniform(-0.1, 0.1) * 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    r = rng.random()
    if r < 0.3:                                                         # down-scale (quality gap)
        h, w = x.shape[:2]
        f = rng.uniform(0.3, 0.8)
        small = cv2.resize(x, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_LINEAR)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif r < 0.6:                                                       # sharpen
        blur = cv2.GaussianBlur(x, (0, 0), 1.0)
        x = cv2.addWeighted(x, 1.5, blur, -0.5, 0)
    return x


def _random_affine(img, mask, rng):
    """Small translation / scale of the source (and of the mask, as in SBI)."""
    h, w = img.shape[:2]
    s = rng.uniform(0.95, 1 / 0.95)
    tx, ty = rng.uniform(-0.03, 0.03, 2) * np.array([w, h])
    M = np.float32([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return img, mask


def self_blend(img, landmarks, rng=None, region_probs=None, region=None):
    """
    Args:
        img: (H, W, 3) uint8 RGB real face.
        landmarks: (81, 2) landmarks in pixel coordinates of `img`.
    Returns:
        blended (H, W, 3) uint8, mask (H, W) float32 in [0, 1], region name.
    """
    rng = np.random.default_rng() if rng is None else rng
    if region is None:
        probs = region_probs or DEFAULT_REGION_PROBS
        names = list(probs)
        p = np.array([probs[n] for n in names], np.float64)
        region = names[int(rng.choice(len(names), p=p / p.sum()))]
    mask = _hull_mask(np.asarray(landmarks, np.float32), REGIONS[region], img.shape)
    source = _source_transform(img, rng)
    source, mask = _random_affine(source, mask, rng)
    mask = _deform_mask(mask, rng)
    mask = mask * float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0]))
    m3 = mask[..., None]
    blended = (source.astype(np.float32) * m3 + img.astype(np.float32) * (1 - m3))
    return np.clip(blended, 0, 255).astype(np.uint8), mask.astype(np.float32), region


# ----------------------------------------------------------------------------- dataset
class SBIAugmentedDataset(DeepfakeAbstractBaseDataset):
    """Original training samples + `ratio` x (number of real frames) SBI pseudo-fakes."""

    def __init__(self, config=None, mode='train'):
        super().__init__(config=config, mode=mode)
        scfg = config.get('sbi', {})
        self.sbi_ratio = float(scfg.get('ratio', 1.0))
        self.region_probs = scfg.get('region_probs', DEFAULT_REGION_PROBS)
        self.n_base = len(self.image_list)
        reals = [i for i, y in enumerate(self.label_list) if y == 0]
        n_sbi = int(round(len(reals) * self.sbi_ratio))
        self.sbi_sources = [reals[i % len(reals)] for i in range(n_sbi)] if reals else []
        tpl = scfg.get('fallback_landmarks', './training/config/part_template_32_landmarks.npy')
        self.fallback_lm01 = np.load(tpl).astype(np.float32) if tpl and os.path.exists(tpl) else None
        self.n_fallback = 0
        print(f'[SBI] base samples={self.n_base} real={len(reals)} sbi={len(self.sbi_sources)} '
              f'ratio={self.sbi_ratio} regions={self.region_probs}')

    def __len__(self):
        return self.n_base + len(self.sbi_sources)

    def _landmarks_for(self, image_path, shape):
        lm_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        if not self.lmdb:
            lm_path = resolve_rgb_path(lm_path, self.config['dataset_root_rgb'])
        lm = None
        if os.path.exists(lm_path):
            lm = np.load(lm_path).astype(np.float32).reshape(-1, 2)
            if lm.shape[0] != 81 or not np.isfinite(lm).all() or lm.max() <= 0:
                lm = None
        if lm is None and self.fallback_lm01 is not None:
            self.n_fallback += 1
            lm = self.fallback_lm01 * np.array([shape[1], shape[0]], np.float32)
        return lm

    def __getitem__(self, index, no_norm=False):
        if index < self.n_base:
            return super().__getitem__(index, no_norm=no_norm)
        src = self.sbi_sources[index - self.n_base]
        image_path = self.data_dict['image'][src]
        if not self.lmdb:
            image_path = resolve_rgb_path(image_path, self.config['dataset_root_rgb'])
        img = cv2.imread(image_path)
        lm = self._landmarks_for(image_path, img.shape) if img is not None else None
        if img is None or lm is None:
            return super().__getitem__(src, no_norm=no_norm)   # rare: fall back to the real frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(random.getrandbits(64))
        blended, mask, _ = self_blend(img, lm, rng, self.region_probs)

        size = self.config['resolution']
        blended = cv2.resize(blended, (size, size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)[..., None].astype(np.float32)
        with_mask = self.config.get('with_mask', False)
        if self.config['use_data_augmentation']:
            blended, _, mask = self.data_aug(blended, None, mask if with_mask else None, None)
        if not no_norm:
            blended = self.normalize(self.to_tensor(blended))
            if with_mask:
                mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        return blended, 1, None, (mask if with_mask else None), [image_path]
