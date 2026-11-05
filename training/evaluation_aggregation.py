import json
from pathlib import Path
from typing import Any

from training.output_paths import validate_output_group, validate_run_id


COUNT_KEYS = {"num_frames", "num_videos", "real_videos", "fake_videos"}


def _numeric_metrics(results: list[dict[str, Any]], common_only: bool = True) -> dict[str, float]:
    metric_key_sets = [
        {key for key in result.get("metrics", {}) if key not in COUNT_KEYS}
        for result in results
    ]
    common_keys = set.intersection(*metric_key_sets) if metric_key_sets else set()
    values: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.get("metrics", {}).items():
            if key in COUNT_KEYS or (common_only and key not in common_keys) or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in sorted(values.items()) if items}


def aggregate_results(
    results: dict[str, dict[str, Any]],
    requested_datasets: list[str],
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    errors = errors or []
    requested_set = set(requested_datasets)
    ignored = sorted(name for name in results if name not in requested_set)
    results = {name: result for name, result in results.items() if name in requested_set}
    completed = [name for name in requested_datasets if name in results]
    missing = [name for name in requested_datasets if name not in results]
    all_errors = list(errors)
    all_errors.extend({"dataset": name, "error": "no result"} for name in missing)

    totals: dict[str, int] = {}
    for result in results.values():
        for key in COUNT_KEYS:
            value = result.get("metrics", {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + int(value)

    by_source: dict[str, list[dict[str, Any]]] = {}
    for result in results.values():
        by_source.setdefault(result.get("source_type", "unknown"), []).append(result)

    source_averages = {
        source: _numeric_metrics(source_results, common_only=True)
        for source, source_results in sorted(by_source.items())
    }
    status = "complete" if not all_errors and len(completed) == len(requested_datasets) else "partial"
    metric_keys = set()
    for result in results.values():
        metric_keys.update(key for key in result.get("metrics", {}) if key not in COUNT_KEYS)
    missing_metrics = {
        name: sorted(metric_keys - set(result.get("metrics", {})))
        for name, result in results.items()
        if metric_keys - set(result.get("metrics", {}))
    }
    return {
        "status": status,
        "datasets": results,
        "completed_datasets": completed,
        "requested_datasets": requested_datasets,
        "averages": {
            "all_completed": _numeric_metrics(list(results.values()), common_only=True),
            "by_source_type": source_averages,
        },
        "totals": totals,
        "errors": all_errors,
        "ignored_results": ignored,
        "missing_metrics": missing_metrics,
    }


def write_aggregate_artifacts(
    output_dir: str | Path,
    namespace: str,
    run_id: str,
    aggregate: dict[str, Any],
) -> tuple[Path, Path]:
    validate_output_group(namespace)
    validate_run_id(run_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{namespace}_{run_id}.json"
    log_path = output_dir / f"{namespace}_{run_id}.log"
    if json_path.exists() or log_path.exists():
        raise FileExistsError(f"aggregate artifacts already exist for run {run_id}")
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"namespace={namespace}",
        f"run_id={run_id}",
        f"status={aggregate['status']}",
        f"completed_datasets={','.join(aggregate['completed_datasets'])}",
        f"averages={json.dumps(aggregate['averages'], sort_keys=True)}",
        f"totals={json.dumps(aggregate['totals'], sort_keys=True)}",
    ]
    if aggregate["errors"]:
        lines.append(f"errors={json.dumps(aggregate['errors'], sort_keys=True)}")
    if aggregate.get("ignored_results"):
        lines.append(f"ignored_results={json.dumps(aggregate['ignored_results'])}")
    if aggregate.get("missing_metrics"):
        lines.append(f"missing_metrics={json.dumps(aggregate['missing_metrics'], sort_keys=True)}")
    lines.append(f"json={json_path}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, log_path
