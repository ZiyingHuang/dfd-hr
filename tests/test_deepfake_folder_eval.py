from pathlib import Path

from training.deepfake_folder_eval import discover_records, infer_label, sample_frames


def test_folder_label_mapping() -> None:
    assert infer_label("CDF", "Celeb-real") == 0
    assert infer_label("CDF", "Celeb-synthesis") == 1
    assert infer_label("DFD", "real") == 0
    assert infer_label("DFD", "fake") == 1
    assert infer_label("WDF", "real") == 0
    assert infer_label("WDF", "fake") == 1


def test_discover_wdf_records_without_json(tmp_path: Path) -> None:
    sequence = tmp_path / "WDF" / "real_test" / "12" / "real" / "34"
    sequence.mkdir(parents=True)
    (sequence / "000.png").touch()
    (sequence / "010.png").touch()

    records = discover_records(tmp_path, "WDF")

    assert len(records) == 1
    assert records[0].label == 0
    assert records[0].video_id == "real_test/12/real/34"
    assert len(records[0].frames) == 2


def test_sample_frames_uses_full_sequence() -> None:
    frames = [f"frame_{index:03d}.png" for index in range(16)]

    assert sample_frames(frames, 4) == [
        "frame_000.png",
        "frame_004.png",
        "frame_008.png",
        "frame_012.png",
    ]
