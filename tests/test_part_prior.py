import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'detectors' / 'utils'))
from part_prior import MIRROR_68, PART_NAMES, build_part_masks, mirror_landmarks  # noqa: E402


def _synthetic_face():
    L = np.zeros((81, 2))
    t = np.linspace(np.pi * 0.95, np.pi * 0.05, 17)
    L[0:17] = np.c_[0.5 - 0.36 * np.cos(t), 0.45 + 0.42 * np.sin(t)]
    L[17:22] = np.c_[np.linspace(0.22, 0.44, 5), 0.36 * np.ones(5)]
    L[22:27] = np.c_[np.linspace(0.56, 0.78, 5), 0.36 * np.ones(5)]
    L[27:31] = np.c_[0.5 * np.ones(4), np.linspace(0.42, 0.6, 4)]
    L[31:36] = np.c_[np.linspace(0.44, 0.56, 5), 0.63 * np.ones(5)]
    for c, s in [(0.33, 36), (0.67, 42)]:
        a = np.linspace(0, 2 * np.pi, 7)[:6]
        L[s:s + 6] = np.c_[c + 0.06 * np.cos(a), 0.43 + 0.02 * np.sin(a)]
    a = np.linspace(0, 2 * np.pi, 13)[:12]
    L[48:60] = np.c_[0.5 + 0.1 * np.cos(a), 0.75 + 0.04 * np.sin(a)]
    a = np.linspace(0, 2 * np.pi, 9)[:8]
    L[60:68] = np.c_[0.5 + 0.07 * np.cos(a), 0.75 + 0.015 * np.sin(a)]
    t = np.linspace(np.pi * 0.95, np.pi * 0.05, 13)
    L[68:81] = np.c_[0.5 - 0.36 * np.cos(t), 0.4 - 0.3 * np.sin(t)]
    return L.astype(np.float32)


def test_mirror_permutation_is_an_involution():
    assert sorted(MIRROR_68) == list(range(68))
    assert all(MIRROR_68[MIRROR_68[i]] == i for i in range(68))


def test_mirror_twice_is_identity():
    L = _synthetic_face()
    np.testing.assert_allclose(mirror_landmarks(mirror_landmarks(L)), L, atol=1e-6)


def test_masks_shape_range_and_locations():
    masks = build_part_masks(_synthetic_face(), grid=32)
    assert masks.shape == (len(PART_NAMES), 32, 32)
    assert masks.min() >= 0 and abs(masks.max() - 1) < 1e-5
    centers = {n: np.array(np.unravel_index(np.argmax(m), m.shape)) for n, m in zip(PART_NAMES, masks)}
    assert centers['eyes_brows'][0] < centers['nose'][0] < centers['mouth'][0]   # top -> bottom
    eyes, mouth = masks[0], masks[2]
    assert (eyes * mouth).sum() < 0.05 * eyes.sum()                               # parts barely overlap
