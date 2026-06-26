from __future__ import annotations

"""Compute the isolated task state from task artifact storage."""

from dataclasses import dataclass
from pathlib import Path

from modules.task_artifact_store import list_task_artifacts


@dataclass(frozen=True)
class TaskArtifactSnapshot:
    filename: str
    path: str
    size_bytes: int


@dataclass(frozen=True)
class TaskStateSnapshot:
    status: str
    file_count: int
    artifacts: list[TaskArtifactSnapshot]


def get_task_state_snapshot(task_artifact_dir: str | Path | None = None) -> TaskStateSnapshot:
    artifact_paths = list_task_artifacts(task_artifact_dir)
    artifacts = [
        TaskArtifactSnapshot(
            filename=artifact_path.name,
            path=str(artifact_path),
            size_bytes=artifact_path.stat().st_size,
        )
        for artifact_path in artifact_paths
    ]
    return TaskStateSnapshot(
        status="ready" if artifacts else "empty",
        file_count=len(artifacts),
        artifacts=artifacts,
    )
