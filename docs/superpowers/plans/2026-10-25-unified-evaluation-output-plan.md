# Unified Evaluation Output Implementation Plan

> **For agentic workers:** Implement task-by-task with verification checkpoints. Do not commit unless explicitly requested.

**Goal:** Produce one final log and one aggregate JSON per dfbench/deepfake evaluation run, with per-dataset metrics, all-dataset averages, source-specific averages, and explicit seven-subset fallback provenance.

**Architecture:** Keep the existing per-dataset inference implementations, but make their batch output quiet and write temporary worker result JSONs. A shared aggregation module will combine scalar metrics, count fields, errors, and source metadata into one final JSON and one concise final log. The seven requested subsets use JSON manifests for CDF-v2/DFD/DFDC/DFDCP and the existing folder evaluator for DFR/WDF/FFIW.

**Tech Stack:** Python 3, pathlib, argparse, JSON, Bash, PyTorch, pytest.

---

### Task 1: Add aggregation utilities and regression tests

**Files:**
- Create: `training/evaluation_aggregation.py`
- Test: `tests/test_evaluation_aggregation.py`

- [ ] Add helpers to average common numeric metric fields, sum count fields, split averages by `source_type`, preserve errors, and write one JSON plus one concise log.
- [ ] Define the JSON contract: `status`, `run_id`, `namespace`, `datasets`, `averages`, `completed_datasets`, `errors`, and `provenance`.
- [ ] Write tests for seven complete datasets, one failed dataset, mixed `json`/`folder` sources, and missing metric fields.
- [ ] Run `PYTHONPATH=.:training pytest -q tests/test_evaluation_aggregation.py`; verify the tests fail before implementation and pass after it.

### Task 2: Make both evaluators quiet and emit temporary worker results

**Files:**
- Modify: `scripts/test.py`
- Modify: `scripts/evaluate_deepfake_testset.py`

- [ ] Add a quiet flag that disables `tqdm`/per-batch prints while preserving exceptions and final return values.
- [ ] Add an explicit temporary result path or worker result mode so each worker can write one machine-readable dataset result without creating a user-facing final JSON.
- [ ] Add `source_type` and source path metadata to every dataset result (`json` for dfbench manifest runs, `folder` for fallback runs).
- [ ] Preserve existing explicit output-dir/run-id behavior and single-dataset invocation compatibility.
- [ ] Add focused tests for quiet argument parsing and result metadata without loading a checkpoint.

### Task 3: Implement unified seven-subset wrappers and final files

**Files:**
- Modify: `scripts/run_paper_eval_4gpu.sh`
- Modify: `scripts/run_deepfake_testset_4gpu.sh`
- Modify: `training/output_paths.py` if namespace helpers are needed

- [ ] For dfbench, run JSON workers for CDF-v2/DFD/DFDC/DFDCP and folder workers for DFR/WDF/FFIW using explicit `source_type` metadata.
- [ ] Redirect worker stdout/stderr to temporary files outside the final run directory; do not leave per-batch logs in the final output.
- [ ] After all workers finish, call the aggregation utility once to write exactly:
  ```text
  outputs/dfbench/eval/<run_id>/dfbench_<run_id>.json
  outputs/dfbench/eval/<run_id>/dfbench_<run_id>.log
  ```
- [ ] Do the equivalent for deepfake and write exactly `deepfake_<run_id>.json` and `deepfake_<run_id>.log`.
- [ ] Include worker success/failure, missing manifest warnings, per-dataset metrics, all-seven macro averages, and JSON/folder source-specific averages.
- [ ] Remove temporary worker files after aggregation; preserve nonzero exit status if any requested dataset fails.

### Task 4: Verify output contract and seven-subset behavior

- [ ] Run `PYTHONPATH=.:training pytest -q`.
- [ ] Run Python compilation and `bash -n` for all wrappers.
- [ ] Run a no-GPU aggregation fixture test to verify one JSON/one log output without inference.
- [ ] Verify the final JSON contains all seven requested dataset keys when fixture results are complete and marks `source_type` correctly.
- [ ] Verify missing JSON manifests are reported as explicit source/provenance warnings rather than silently treated as JSON results.
- [ ] Confirm no dataset files are modified and no per-batch worker logs remain in the final run directory.
