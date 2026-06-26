from __future__ import annotations

"""Filesystem storage helpers for task screenshot artifacts."""

import shutil
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ARTIFACT_DIR = PYTHON_ROOT / "artifacts" / "process"
TASK_ARTIFACT_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def get_task_artifact_dir(task_artifact_dir: str | Path | None = None) -> Path:
    resolved_path = Path(task_artifact_dir or DEFAULT_TASK_ARTIFACT_DIR).expanduser().resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def list_task_artifacts(task_artifact_dir: str | Path | None = None) -> list[Path]:
    artifact_dir = get_task_artifact_dir(task_artifact_dir)
    return sorted(
        (
            path.resolve()
            for path in artifact_dir.iterdir()
            if path.is_file() and path.suffix.lower() in TASK_ARTIFACT_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def count_task_artifacts(task_artifact_dir: str | Path | None = None) -> int:
    return len(list_task_artifacts(task_artifact_dir))


def save_task_artifact(
    source_path: str | Path,
    *,
    task_artifact_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    resolved_source_path = Path(source_path).expanduser().resolve()
    if not resolved_source_path.exists():
        raise FileNotFoundError(f"Task artifact source file does not exist: {resolved_source_path}")
    if not resolved_source_path.is_file():
        raise ValueError(f"Task artifact source path must be a file: {resolved_source_path}")
    if resolved_source_path.suffix.lower() not in TASK_ARTIFACT_SUFFIXES:
        raise ValueError(f"Task artifact source file must use a supported screenshot extension: {resolved_source_path.suffix}")

    artifact_dir = get_task_artifact_dir(task_artifact_dir)
    target_name = filename or resolved_source_path.name
    if Path(target_name).suffix.lower() not in TASK_ARTIFACT_SUFFIXES:
        raise ValueError("Task artifact filename must use a supported screenshot extension.")
    destination_path = artifact_dir / Path(target_name).name
    shutil.copy2(resolved_source_path, destination_path)
    return destination_path.resolve()


def clear_task_artifacts(task_artifact_dir: str | Path | None = None) -> Path:
    artifact_dir = get_task_artifact_dir(task_artifact_dir)
    for artifact_path in artifact_dir.iterdir():
        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
            continue
        artifact_path.unlink()
    return artifact_dir


def is_task_artifact_dir_empty(task_artifact_dir: str | Path | None = None) -> bool:
    return count_task_artifacts(task_artifact_dir) == 0
