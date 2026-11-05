import json

from training.evaluation_aggregation import aggregate_results, write_aggregate_artifacts


def test_aggregate_results_averages_metrics_and_sums_counts() -> None:
    results = {
        "CDF-v2": {
            "source_type": "json",
            "metrics": {
                "auc": 0.8,
                "video_auc": 0.9,
                "num_frames": 10,
                "num_videos": 2,
            },
        },
        "DFR": {
            "source_type": "folder",
            "metrics": {
                "auc": 0.6,
                "video_auc": 0.7,
                "num_frames": 20,
                "num_videos": 4,
            },
        },
    }

    aggregate = aggregate_results(results, requested_datasets=["CDF-v2", "DFR"])

    assert aggregate["status"] == "complete"
    assert aggregate["averages"]["all_completed"] == {
        "auc": 0.7,
        "video_auc": 0.8,
    }
    assert aggregate["averages"]["by_source_type"] == {
        "folder": {"auc": 0.6, "video_auc": 0.7},
        "json": {"auc": 0.8, "video_auc": 0.9},
    }
    assert aggregate["totals"] == {"num_frames": 30, "num_videos": 6}


def test_write_aggregate_artifacts_writes_one_json_and_one_log(tmp_path) -> None:
    aggregate = aggregate_results(
        {
            "CDF-v2": {
                "source_type": "json",
                "metrics": {"video_auc": 0.9},
            }
        },
        requested_datasets=["CDF-v2"],
    )

    json_path, log_path = write_aggregate_artifacts(
        tmp_path,
        namespace="dfbench",
        run_id="20261025_120000",
        aggregate=aggregate,
    )

    assert json_path.name == "dfbench_20261025_120000.json"
    assert log_path.name == "dfbench_20261025_120000.log"
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert "iter=" not in log_path.read_text(encoding="utf-8")


def test_aggregate_ignores_unrequested_results_and_reports_missing_metrics() -> None:
    aggregate = aggregate_results(
        {
            "CDF-v2": {"source_type": "json", "metrics": {"auc": 0.8}},
            "unexpected": {"source_type": "json", "metrics": {"auc": 0.1}},
        },
        requested_datasets=["CDF-v2", "DFR"],
    )

    assert aggregate["status"] == "partial"
    assert aggregate["averages"]["all_completed"] == {"auc": 0.8}
    assert aggregate["ignored_results"] == ["unexpected"]
    assert {error["dataset"] for error in aggregate["errors"]} == {"DFR"}
