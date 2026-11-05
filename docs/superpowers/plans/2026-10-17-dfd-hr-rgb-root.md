# DFD-HR RGB Root Configuration Implementation Plan

> **For agentic workers:** Execute this plan inline with verification checkpoints. Do not start formal training.

**Goal:** Make DFD-HR resolve normalized JSON frame paths under `/data/hjr/dfbench/rgb` while keeping the official `FaceForensics++` training selection unchanged.

**Architecture:** Set absolute RGB and JSON roots in both DFD-HR YAML configs. Add a small path resolver in the dataset loader that normalizes Windows separators, preserves absolute paths, and resolves relative frame paths against `dataset_root_rgb`; use the resolved path before deriving masks and landmarks. Validate with the current partial dataset and configuration parsing without launching training.

**Tech Stack:** Python, PyYAML, DFD-HR dataset loader, JSON/PNG filesystem checks.

---

### Task 1: Update runtime roots

**Files:**
- Modify: `training/config/train_config.yaml:1-6`
- Modify: `training/config/test_config.yaml:1-5`

- [x] Set `dataset_root_rgb` to `/data/hjr/dfbench/rgb`.
- [x] Set `dataset_json_folder` to `/data/hjr/dfbench/dataset_json`.
- [x] Leave `lmdb: False` and the detector dataset lists unchanged.

### Task 2: Resolve DFD-HR RGB paths

**Files:**
- Modify: `training/dataset/abstract_dataset.py` near `__getitem__` and RGB loading.

- [x] Normalize runtime JSON paths with `replace('\\', '/')`.
- [x] If the normalized path is absolute, use it unchanged.
- [x] Otherwise resolve it under `Path(self.config['dataset_root_rgb'])`.
- [x] Perform existence checks on the resolved path before loading.
- [x] Derive `masks` and `landmarks` from the resolved path so optional auxiliary loading remains consistent.
- [x] Do not change dataset selection or add missing datasets.

### Task 3: Validate without training

**Files:**
- No additional source files.

- [x] Parse both YAML files with PyYAML and assert the configured roots.
- [x] Load `Celeb-DF-v1.json`, normalize one sample frame path, resolve it under the configured root, and assert it exists.
- [x] Run Python compilation/diagnostics on the modified loader.
- [x] Do not launch a training job because the default `FaceForensics++` RGB data is still downloading.
