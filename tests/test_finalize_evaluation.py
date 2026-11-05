import json

from scripts.finalize_evaluation import load_results


def test_load_results_accepts_dataset_and_multi_dataset_payloads(tmp_path) -> None:
    (tmp_path / "dataset.json").write_text(json.dumps({
        "dataset": "DFR",
        "source_type": "folder",
        "source": "/data/hjr/deepfake/testset",
        "metrics": {"video_auc": 0.9},
    }), encoding="utf-8")
    (tmp_path / "multi.json").write_text(json.dumps({
        "source_type": "json",
        "source": "/data/hjr/dfbench/dataset_json",
        "metrics": {"DFDC": {"video_auc": 0.8}},
    }), encoding="utf-8")

    results = load_results(tmp_path)

    assert results["DFR"]["source_type"] == "folder"
    assert results["DFDC"]["source_type"] == "json"
