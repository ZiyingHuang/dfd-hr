from pathlib import Path, PureWindowsPath


def resolve_rgb_path(file_path: str, dataset_root_rgb: str) -> str:
    """Resolve a JSON frame path inside the configured RGB dataset root."""
    normalized_path = file_path.replace("\\", "/")
    root = Path(dataset_root_rgb).expanduser().resolve()
    path = Path(normalized_path)
    if PureWindowsPath(normalized_path).is_absolute() and not path.is_absolute():
        raise ValueError(f"Resolved dataset path is outside RGB root: {file_path}")
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        _ = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Resolved dataset path is outside RGB root: {file_path}") from exc
    return str(resolved)


def sample_frame_paths(frame_paths: list[str], frame_num: int) -> list[str]:
    """Select evenly spaced frames from the full manifest sequence."""
    if frame_num <= 0:
        raise ValueError("frame_num must be positive")
    if len(frame_paths) <= frame_num:
        return list(frame_paths)
    step = max(1, len(frame_paths) // frame_num)
    return [frame_paths[index] for index in range(0, len(frame_paths), step)][:frame_num]
