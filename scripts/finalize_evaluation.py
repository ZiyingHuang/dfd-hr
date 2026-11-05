#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from training.evaluation_aggregation import aggregate_results, write_aggregate_artifacts


def load_results(input_dir: Path) -> dict[str, dict[str, object]]:
    results = {}
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") and payload.get("metrics"):
            results[payload["dataset"]] = {
                "source_type": payload.get("source_type", "unknown"),
                "source": payload.get("source"),
                "provenance": {
                    key: payload.get(key)
                    for key in ("checkpoint", "dataset_root", "dataset_root_rgb", "dataset_json_folder", "source")
                    if payload.get(key) is not None
                },
                "metrics": payload["metrics"],
            }
            continue
        for dataset, metrics in payload.get("metrics", {}).items():
            results[dataset] = {
                "source_type": payload.get("source_type", "unknown"),
                "source": payload.get("source"),
                "provenance": {
                    key: payload.get(key)
                    for key in ("checkpoint", "dataset_root", "dataset_root_rgb", "dataset_json_folder", "source")
                    if payload.get(key) is not None
                },
                "metrics": metrics,
            }
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--worker-log-dir", default=None)
    return parser.parse_args()


def load_worker_errors(log_dir: Path | None) -> list[dict[str, str]]:
    if log_dir is None or not log_dir.exists():
        return []
    errors = []
    for path in sorted(log_dir.glob("*.log")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines and any("Traceback" in line or "Error" in line or "failed" in line.lower() for line in lines):
            errors.append({"worker": path.name, "error": "\n".join(lines[-20:])})
    return errors


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.input_dir))
    aggregate = aggregate_results(results, args.datasets, errors=load_worker_errors(Path(args.worker_log_dir) if args.worker_log_dir else None))
    aggregate.update({
        "namespace": args.namespace,
        "run_id": args.run_id,
    })
    json_path, log_path = write_aggregate_artifacts(
        args.output_dir,
        args.namespace,
        args.run_id,
        aggregate,
    )
    print(json_path)
    print(log_path)
    if aggregate["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
