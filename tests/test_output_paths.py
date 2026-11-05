from pathlib import Path
import re

import pytest

from training.output_paths import (
    default_run_id,
    run_output_dir,
    validate_output_group,
    validate_run_id,
)


def test_default_run_id_uses_second_precision() -> None:
    assert re.fullmatch(r"\d{8}_\d{6}", default_run_id())


def test_run_output_dir_separates_train_and_eval() -> None:
    project_root = Path("/project")

    assert run_output_dir(project_root, "train", "20261020_120000") == (
        project_root / "outputs" / "deepfake" / "train" / "20261020_120000"
    )
    assert run_output_dir(project_root, "eval", "20261020_120001") == (
        project_root / "outputs" / "deepfake" / "eval" / "20261020_120001"
    )
    assert run_output_dir(
        project_root,
        "eval",
        "20261020_120002",
        output_group="dfbench",
    ) == project_root / "outputs" / "dfbench" / "eval" / "20261020_120002"


def test_run_output_dir_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="phase"):
        run_output_dir(Path("/project"), "other", "20261020_120000")


@pytest.mark.parametrize("run_id", [".", "..", "../escape", "run/name", "run name"])
def test_run_id_rejects_path_components(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        validate_run_id(run_id)


def test_output_group_rejects_path_components() -> None:
    with pytest.raises(ValueError, match="output_group"):
        validate_output_group("../escape")
