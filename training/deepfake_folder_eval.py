from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable


SUPPORTED_DATASETS = {"CDF", "DFD", "DFDC", "DFDCP", "DFR", "FFIW", "WDF"}


@dataclass(frozen=True)
class VideoRecord:
    dataset: str
    video_id: str
    label: int
    frames: tuple[str, ...]


def infer_label(dataset_name: str, label_folder: str) -> int:
    if dataset_name == "CDF":
        mapping = {"Celeb-real": 0, "YouTube-real": 0, "Celeb-synthesis": 1}
    elif dataset_name == "WDF":
        mapping = {"real": 0, "fake": 1}
    else:
        mapping = {"real": 0, "fake": 1}
    try:
        return mapping[label_folder]
    except KeyError as exc:
        raise ValueError(f"Unknown label folder {label_folder!r} for dataset {dataset_name!r}") from exc


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def _frame_paths(sequence: Path) -> tuple[str, ...]:
    frames = sorted(
        (path for path in sequence.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=_frame_sort_key,
    )
    return tuple(str(path.resolve()) for path in frames)


def _records_for_label(dataset_root: Path, dataset_name: str, label_folder: Path) -> Iterable[VideoRecord]:
    label = infer_label(dataset_name, label_folder.name)
    for sequence in sorted(path for path in label_folder.iterdir() if path.is_dir()):
        frames = _frame_paths(sequence)
        if frames:
            yield VideoRecord(
                dataset=dataset_name,
                video_id=str(sequence.relative_to(dataset_root)),
                label=label,
                frames=frames,
            )


def discover_records(root: Path, dataset_name: str) -> list[VideoRecord]:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported folder dataset {dataset_name!r}")
    root = root.resolve()
    dataset_root = (root / dataset_name).resolve()
    try:
        dataset_root.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes root: {dataset_name!r}") from exc
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    records: list[VideoRecord] = []
    if dataset_name == "CDF":
        label_folders = [
            dataset_root / "Celeb-real",
            dataset_root / "YouTube-real",
            dataset_root / "Celeb-synthesis",
        ]
        for label_folder in label_folders:
            if label_folder.is_dir():
                records.extend(_records_for_label(dataset_root, dataset_name, label_folder))
    elif dataset_name == "WDF":
        for split in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            for identity in sorted(path for path in split.iterdir() if path.is_dir()):
                for label_folder in sorted(path for path in identity.iterdir() if path.is_dir()):
                    records.extend(_records_for_label(dataset_root, dataset_name, label_folder))
    else:
        for label_folder in (dataset_root / "real", dataset_root / "fake"):
            if label_folder.is_dir():
                records.extend(_records_for_label(dataset_root, dataset_name, label_folder))

    if not records:
        raise ValueError(f"No PNG sequence records found under {dataset_root}")
    return records


def sample_frames(frames: Iterable[str], frame_num: int) -> list[str]:
    paths = list(frames)
    if frame_num <= 0:
        raise ValueError("frame_num must be positive")
    if len(paths) <= frame_num:
        return paths
    step = max(1, len(paths) // frame_num)
    return [paths[index] for index in range(0, len(paths), step)][:frame_num]
