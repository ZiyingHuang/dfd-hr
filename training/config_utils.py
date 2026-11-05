"""Detector-config loading with optional `_base_` inheritance.

A detector YAML may contain `_base_: other.yaml` (path relative to the YAML
itself). The base is loaded first and the child is deep-merged on top, so an
ablation config only needs to list the keys it changes.
"""

from copy import deepcopy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_detector_config(path, _seen=None) -> dict:
    path = Path(path).resolve()
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise ValueError(f'circular _base_ reference at {path}')
    _seen.add(path)
    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    base_ref = config.pop('_base_', None)
    if base_ref is None:
        return config
    base = load_detector_config(path.parent / base_ref, _seen)
    return deep_merge(base, config)
