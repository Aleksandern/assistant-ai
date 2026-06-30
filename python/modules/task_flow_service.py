from __future__ import annotations

"""Use-case orchestration for task screenshot capture, solve, and clear flows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from modules.sqlite_conversation_store import create_code_test, get_latest_code_test
from modules.task_artifact_store import clear_task_artifacts, list_task_artifacts
from modules.task_openai_solver import solve_task_from_screenshots
from modules.task_screenshot_service import ScreenshotAdapter, create_task_screenshot
from modules.task_state_service import TaskStateSnapshot, get_task_state_snapshot


class TaskSnapshotPublisher(Protocol):
    def publish_task_snapshot(
        self,
        *,
        status: str,
        file_count: int,
        artifacts: list[dict[str, object]],
        latest_result: dict[str, object] | None,
        error: str | None,
    ) -> None: ...


@dataclass(frozen=True)
class TaskFlowSnapshot:
    status: str
    file_count: int
    artifacts: list[dict[str, object]]
    latest_result: dict[str, object] | None
    error: str | None


def capture_task_screenshot(
    *,
    screenshot_adapter: ScreenshotAdapter,
    publisher: TaskSnapshotPublisher,
    task_artifact_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> TaskFlowSnapshot:
    try:
        create_task_screenshot(
            screenshot_adapter=screenshot_adapter,
            task_artifact_dir=task_artifact_dir,
        )
    except Exception as exc:
        return publish_current_task_snapshot(
            publisher=publisher,
            task_artifact_dir=task_artifact_dir,
            database_path=database_path,
            status="error",
            latest_result=None,
            error=f"Task screenshot capture failed: {exc}",
        )

    return publish_current_task_snapshot(
        publisher=publisher,
        task_artifact_dir=task_artifact_dir,
        database_path=database_path,
        latest_result=None,
        error=None,
    )


def solve_current_task(
    *,
    publisher: TaskSnapshotPublisher,
    task_artifact_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    client: object | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dotenv_path: str | Path | None = None,
    task_prompt: str | None = None,
) -> TaskFlowSnapshot:
    screenshot_paths = list_task_artifacts(task_artifact_dir)
    if not screenshot_paths:
        return publish_current_task_snapshot(
            publisher=publisher,
            task_artifact_dir=task_artifact_dir,
            database_path=database_path,
            status="error",
            latest_result=None,
            error="No task screenshots are available to solve.",
        )

    try:
        solve_result = solve_task_from_screenshots(
            screenshot_paths,
            client=client,
            api_key=api_key,
            model=model,
            dotenv_path=dotenv_path,
            task_prompt=task_prompt,
        )
    except Exception as exc:
        return publish_current_task_snapshot(
            publisher=publisher,
            task_artifact_dir=task_artifact_dir,
            database_path=database_path,
            status="error",
            latest_result=None,
            error=f"Task solve failed: {exc}",
        )

    try:
        create_code_test(
            solve_result.response_text,
            solve_result.source_file_count,
            database_path=database_path,
        )
    except Exception as exc:
        return publish_current_task_snapshot(
            publisher=publisher,
            task_artifact_dir=task_artifact_dir,
            database_path=database_path,
            status="error",
            latest_result=None,
            error=f"Task result persistence failed: {exc}",
        )

    clear_task_artifacts(task_artifact_dir)
    return publish_current_task_snapshot(
        publisher=publisher,
        task_artifact_dir=task_artifact_dir,
        database_path=database_path,
        status="complete",
        latest_result={
            "name": "code_tests",
            "status": "passed",
            "summary": f"Saved task solve from {solve_result.source_file_count} screenshot(s).",
            "response_text": solve_result.response_text,
        },
        error=None,
    )


def clear_current_task(
    *,
    publisher: TaskSnapshotPublisher,
    task_artifact_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> TaskFlowSnapshot:
    clear_task_artifacts(task_artifact_dir)
    return publish_current_task_snapshot(
        publisher=publisher,
        task_artifact_dir=task_artifact_dir,
        database_path=database_path,
        latest_result=None,
        error=None,
    )


def publish_current_task_snapshot(
    *,
    publisher: TaskSnapshotPublisher,
    task_artifact_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    status: str | None = None,
    latest_result: dict[str, object] | None = None,
    error: str | None,
) -> TaskFlowSnapshot:
    snapshot = build_current_task_snapshot(
        task_artifact_dir=task_artifact_dir,
        database_path=database_path,
        status=status,
        latest_result=latest_result,
        error=error,
    )
    publisher.publish_task_snapshot(
        status=snapshot.status,
        file_count=snapshot.file_count,
        artifacts=snapshot.artifacts,
        latest_result=snapshot.latest_result,
        error=snapshot.error,
    )
    return snapshot


def build_current_task_snapshot(
    *,
    task_artifact_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    status: str | None = None,
    latest_result: dict[str, object] | None = None,
    error: str | None,
) -> TaskFlowSnapshot:
    task_state = get_task_state_snapshot(task_artifact_dir)
    return TaskFlowSnapshot(
        status=status or task_state.status,
        file_count=task_state.file_count,
        artifacts=_build_transport_artifacts(task_state),
        latest_result=_resolve_latest_result(
            latest_result=latest_result,
            database_path=database_path,
        ),
        error=error,
    )


def _resolve_latest_result(
    *,
    latest_result: dict[str, object] | None,
    database_path: str | Path | None,
) -> dict[str, object] | None:
    if latest_result is not None:
        return latest_result

    latest_code_test = get_latest_code_test(database_path=database_path)
    if latest_code_test is None:
        return None

    return {
        "name": "code_tests",
        "status": "passed",
        "summary": f"Saved task solve from {latest_code_test.source_file_count} screenshot(s).",
        "response_text": latest_code_test.response_text,
    }


def _build_transport_artifacts(task_state: TaskStateSnapshot) -> list[dict[str, object]]:
    transport_artifacts: list[dict[str, object]] = []
    for artifact in task_state.artifacts:
        content_type = _resolve_content_type(Path(artifact.path))
        transport_artifact = {
            "id": artifact.filename,
            "kind": "image",
            "label": artifact.filename,
            "path": artifact.path,
        }
        if content_type is not None:
            transport_artifact["content_type"] = content_type
        transport_artifacts.append(transport_artifact)
    return transport_artifacts


def _resolve_content_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return None
