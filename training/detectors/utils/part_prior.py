"""Facial part prior: 81-point landmarks -> K soft part masks on the token grid.

numpy/cv2 only, so it can be used (and tested) without torch.
"""

import cv2
import numpy as np

PART_NAMES = ['eyes_brows', 'nose', 'mouth', 'boundary', 'skin']

# dlib 81-point layout: 0-16 jaw, 17-26 brows, 27-35 nose, 36-47 eyes,
# 48-67 mouth, 68-80 forehead.
LEFT_EYE = list(range(17, 22)) + list(range(36, 42))
RIGHT_EYE = list(range(22, 27)) + list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH = list(range(48, 60))
FACE = list(range(0, 17)) + list(range(68, 81))

# Horizontal mirror permutation of the 68 dlib points.
MIRROR_68 = (list(range(16, -1, -1)) + list(range(26, 16, -1)) + [27, 28, 29, 30] +
             [35, 34, 33, 32, 31] + [45, 44, 43, 42, 47, 46] + [39, 38, 37, 36, 41, 40] +
             [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55] + [64, 63, 62, 61, 60, 67, 66, 65])


def _hull_mask(points01, size):
    mask = np.zeros((size, size), np.uint8)
    hull = cv2.convexHull(np.round(points01 * size).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def build_part_masks(landmarks01, grid=32, raster=448, boundary_width=0.06,
                     part_dilate=0.03, blur=0.02):
    """
    Args:
        landmarks01: (81, 2) landmarks normalised to [0, 1] (x, y).
        grid: token grid of the local branch (32 for a 448 input split 2x2 with ViT-L/14).
    Returns:
        (K, grid, grid) float32 masks in [0, 1], order = PART_NAMES.
    """
    lm = np.asarray(landmarks01, dtype=np.float32)
    if lm.shape != (81, 2):
        raise ValueError(f'expected (81, 2) landmarks, got {lm.shape}')

    def kernel(frac):
        k = max(1, int(round(frac * raster)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

    eyes = np.maximum(_hull_mask(lm[LEFT_EYE], raster), _hull_mask(lm[RIGHT_EYE], raster))
    nose = _hull_mask(lm[NOSE], raster)
    mouth = _hull_mask(lm[MOUTH], raster)
    eyes, nose, mouth = [cv2.dilate(m, kernel(part_dilate)) for m in (eyes, nose, mouth)]

    face = _hull_mask(lm[FACE], raster)
    k_band = kernel(boundary_width / 2)
    boundary = cv2.dilate(face, k_band) - cv2.erode(face, k_band)
    inner = cv2.erode(face, k_band)
    skin = np.clip(inner.astype(np.int32) - eyes - nose - mouth, 0, 1).astype(np.uint8)

    masks = []
    sigma = blur * raster
    for m in (eyes, nose, mouth, boundary, skin):
        m = m.astype(np.float32)
        if sigma > 0:
            m = cv2.GaussianBlur(m, (0, 0), sigma)
        m = cv2.resize(m, (grid, grid), interpolation=cv2.INTER_AREA)
        m = m / max(float(m.max()), 1e-6)
        masks.append(m)
    return np.stack(masks, 0).astype(np.float32)


def mirror_landmarks(lm01):
    """Mirror normalised 81-pt landmarks horizontally (keeps semantic point order)."""
    lm01 = np.asarray(lm01, dtype=np.float32)
    out = lm01.copy()
    out[:68] = lm01[MIRROR_68]
    out[:68, 0] = 1.0 - out[:68, 0]
    # forehead points 68-80 have no fixed dlib semantics: mirror and re-sort by x
    fh = lm01[68:].copy()
    fh[:, 0] = 1.0 - fh[:, 0]
    out[68:][np.argsort(lm01[68:, 0])] = fh[np.argsort(fh[:, 0])]
    return out
