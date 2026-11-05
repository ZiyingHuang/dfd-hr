# Unified Evaluation Output Design

## Goal

For both `dfbench` and `deepfake` evaluations, write exactly one final log and one aggregate JSON per run. The JSON must include per-dataset metrics, macro averages, source provenance, and errors/status; batch progress must not remain in the final log.

## Seven-subset policy

The requested non-FF++ Protocol-1 subsets are:

```text
CDF-v2, DFD, DFDC, DFDCP, DFR/DFo, WDF, FFIW
```

Available sources are mixed:

- CDF-v2, DFD, DFDC, DFDCP: DFD-HR JSON manifests under `/data/hjr/dfbench/dataset_json`.
- DFR, WDF, FFIW: existing folder-label data under `/data/hjr/deepfake/testset` because corresponding JSON manifests are absent.

Each dataset entry records `source_type` as `json` or `folder`. The aggregate includes all seven only when all seven complete, and also includes source-specific averages so the mixed provenance is explicit.

## Output contract

Each run writes only:

```text
outputs/<namespace>/eval/<run_id>/<namespace>_<run_id>.json
outputs/<namespace>/eval/<run_id>/<namespace>_<run_id>.log
```

The JSON contains:

- `status`, `run_id`, `namespace`, `checkpoint`;
- `datasets` with per-dataset metrics and source metadata;
- `averages.all_completed` for the seven requested datasets;
- `averages.by_source_type` for JSON and folder groups;
- `errors` and `completed_datasets`.

The log contains only run configuration, worker completion/failure summaries, missing-source warnings, final metric averages, output paths, and errors. Worker stdout/stderr is temporary and removed after aggregation.

## Aggregation

All scalar metric fields common to completed datasets are averaged using an unweighted arithmetic mean. Count fields are summed and never averaged. A failed or unavailable dataset is excluded from averages and listed in `errors`; the overall run status is nonzero unless all seven requested datasets complete.

## Compatibility

- Existing explicit `--output_dir` and `--run_id` arguments remain supported.
- Dataset directories and JSON manifests remain read-only.
- Existing historical per-dataset outputs are preserved; the new one-file contract applies to future runs.
- The mixed-source seven-subset result is a local comparison run, not a strict single-loader paper reproduction.
