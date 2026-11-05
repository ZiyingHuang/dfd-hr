from datetime import datetime
from pathlib import Path
import re


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_run_id(run_id: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-'")
    return run_id


def validate_output_group(output_group: str) -> str:
    if not _SAFE_RUN_ID.fullmatch(output_group):
        raise ValueError("output_group must contain only letters, digits, '_' or '-'")
    return output_group


def run_output_dir(
    project_root: str | Path,
    phase: str,
    run_id: str,
    output_group: str = "deepfake",
) -> Path:
    if phase not in {"train", "eval"}:
        raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
    run_id = validate_run_id(run_id)
    output_group = validate_output_group(output_group)

    return Path(project_root) / "outputs" / output_group / phase / run_id
