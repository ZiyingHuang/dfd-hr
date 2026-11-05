from pathlib import Path

import pytest

from training.dataset.path_utils import resolve_rgb_path, sample_frame_paths
from training.metrics.utils import summarize_test_metrics


def test_resolves_windows_json_path_under_rgb_root(tmp_path: Path) -> None:
    frame = tmp_path / "Celeb-DF-v1" / "YouTube-real" / "frames" / "00000" / "000.png"
    _ = frame.parent.mkdir(parents=True)
    _ = frame.touch()

    resolved = resolve_rgb_path(
        r"Celeb-DF-v1\YouTube-real\frames\00000\000.png",
        str(tmp_path),
    )

    assert resolved == str(frame)
    assert Path(resolved).is_file()


def test_rejects_path_outside_rgb_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = resolve_rgb_path("../outside/frame.png", str(tmp_path))


def test_rejects_absolute_path_outside_rgb_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = resolve_rgb_path("/etc/passwd", str(tmp_path))


def test_rejects_symlink_escape_from_rgb_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.png"
    _ = outside.touch()
    inside = tmp_path / "dataset"
    _ = inside.mkdir()
    _ = (inside / "link.png").symlink_to(outside)

    with pytest.raises(ValueError, match="outside"):
        _ = resolve_rgb_path("link.png", str(inside))


def test_rejects_windows_drive_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = resolve_rgb_path(r"C:\datasets\frame.png", str(tmp_path))


def test_samples_evenly_across_full_manifest_sequence() -> None:
    frames = [f"frame_{index:03d}.png" for index in range(16)]

    selected = sample_frame_paths(frames, 4)

    assert selected == [
        "frame_000.png",
        "frame_004.png",
        "frame_008.png",
        "frame_012.png",
    ]


def test_paper_train_and_test_frame_counts_are_separate() -> None:
    import yaml

    project = Path(__file__).parents[1]
    with (project / "training/config/train_config.yaml").open(encoding="utf-8") as handle:
        train_config = yaml.safe_load(handle)
    with (project / "training/config/test_config.yaml").open(encoding="utf-8") as handle:
        test_config = yaml.safe_load(handle)

    assert train_config["frame_num"]["train"] == 8
    assert test_config["frame_num"]["test"] == 32


def test_result_summary_excludes_prediction_arrays() -> None:
    import numpy as np

    summary = summarize_test_metrics({
        "Celeb-DF-v2": {
            "auc": np.float64(0.9),
            "video_auc": 0.95,
            "pred": np.array([0.1, 0.9]),
            "label": np.array([0, 1]),
        }
    })

    assert summary == {
        "Celeb-DF-v2": {"auc": 0.9, "video_auc": 0.95}
    }
