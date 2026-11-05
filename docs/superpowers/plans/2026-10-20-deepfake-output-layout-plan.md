# Deepfake Output Layout Implementation Plan

> **For agentic workers:** Implement this plan task-by-task with verification checkpoints. Do not commit unless the user explicitly requests a commit.

**Goal:** Organize future DFD-HR artifacts by data source under `outputs/deepfake/{train,eval}/{run_id}/` and `outputs/dfbench/{train,eval}/{run_id}/`, migrate existing flat results, and preserve read-only dataset directories.

**Architecture:** Add a small shared output-path helper for default run IDs, output namespaces, and train/eval run directories. JSON-based DFD-HR/DeepfakeBench workflows use the `dfbench` namespace; folder-based deepfake workflows use `deepfake`. Python entrypoints accept explicit output/run arguments while shell wrappers pass one shared run directory to all workers.

**Tech Stack:** Python 3, pathlib, argparse, Bash, pytest, existing PyTorch/DFD-HR runners.

---

### Task 1: Add shared output path helpers and tests

**Files:**
- Create: `training/output_paths.py`
- Test: `tests/test_output_paths.py`

- [ ] Add `default_run_id()` returning `YYYYMMDD_HHMMSS` and `run_output_dir(project_root, phase, run_id)` returning `outputs/deepfake/<phase>/<run_id>` for `phase in {train, eval}`.
- [ ] Add tests for both phases, both output namespaces, explicit run IDs, and rejection of invalid path components without touching real output directories.
- [ ] Run `PYTHONPATH=.:training pytest -q tests/test_output_paths.py` and verify the new tests pass.

### Task 2: Route training artifacts into timestamped train runs

**Files:**
- Modify: `scripts/train.py` argument parser and config setup
- Modify: `training/trainer/trainer.py` run-directory construction
- Modify: `scripts/run_dfd_hr.sh`

- [ ] Add `--output_dir` and `--run_id` arguments to `train.py`.
- [ ] Default training output to `outputs/deepfake/train/<run_id>`; explicit `--output_dir` remains authoritative.
- [ ] Set a config flag indicating that `config['log_dir']` is already the final run directory, so `Trainer` does not append a second model/timestamp directory.
- [ ] Preserve existing `train/`, `test/`, checkpoint, feature, metric, and TensorBoard subdirectories below the final run directory.
- [ ] Update `run_dfd_hr.sh` to default the FaceForensics++/DeepfakeBench training wrapper to `outputs/dfbench/train/$RUN_ID`, while allowing a validated `OUTPUT_GROUP` override.
- [ ] Run `bash -n scripts/run_dfd_hr.sh` and compile `scripts/train.py`/`training/trainer/trainer.py`.

### Task 3: Route JSON-based evaluation into timestamped dfbench eval runs

**Files:**
- Modify: `scripts/test.py`
- Modify: `scripts/run_paper_eval_4gpu.sh`

- [ ] Add `--run_id` to `test.py` and use the shared helper when `--output_dir` is omitted.
- [ ] Ensure the JSON metrics filename uses the same run ID as its parent directory.
- [ ] Change the paper-eval wrapper to `outputs/dfbench/eval/$RUN_ID`, pass both `--output_dir` and `--run_id`, and keep all four GPU logs in that directory.
- [ ] Preserve explicit `--output_dir` compatibility.
- [ ] Add/adjust tests for deterministic output path and run ID propagation; run `bash -n` and targeted pytest.

### Task 4: Route folder-based deepfake evaluation into timestamped eval runs

**Files:**
- Modify: `scripts/evaluate_deepfake_testset.py`
- Modify: `scripts/run_deepfake_testset_4gpu.sh`

- [ ] Make the evaluator resolve its default output to `outputs/deepfake/eval/<run_id>` while accepting an explicit `--output-dir` unchanged.
- [ ] Ensure worker JSON files and the aggregate summary use the same resolved directory and run ID.
- [ ] Update the four-GPU wrapper to set `OUTPUT_DIR=outputs/deepfake/eval/$RUN_ID` before launching workers.
- [ ] Keep iteration logging behavior unchanged: one line per iteration, with no dataset writes.
- [ ] Run `bash -n`, Python compilation, and focused evaluator/path tests.

### Task 5: Migrate historical flat deepfake and dfbench evaluation files

**Files:**
- Move only existing files under `outputs/deepfake/` into:
  - `outputs/deepfake/eval/20261019_031304/`
  - `outputs/deepfake/eval/20261019_115030/`

- [ ] Create both destination directories.
- [ ] Move files by their original run ID; preserve filenames and file bytes.
- [ ] Do not invent missing JSON files for the incomplete run.
- [ ] Verify destination file counts and hashes against pre-move values.
- [ ] Confirm `outputs/deepfake/` root contains no result files and both `train/` and `eval/` phase directories are present.
- [ ] Move the eight legacy `outputs/dfbench/` files into `outputs/dfbench/eval/20261018_114206/` and verify their hashes.
- [ ] Confirm `outputs/dfbench/` root contains no result files and both `train/` and `eval/` phase directories are present.

### Task 6: Final verification

- [ ] Run `PYTHONPATH=.:training pytest -q`.
- [ ] Run `python -m py_compile` on all modified Python files.
- [ ] Run `bash -n` on all modified shell wrappers.
- [ ] Verify only output directories changed; `/data/hjr/deepfake/testset/` and other datasets remain untouched.
- [ ] Report the final directory tree and any pre-existing `outputs/dfbench/` files left unchanged.
