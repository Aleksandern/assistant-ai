from __future__ import annotations

"""Capture a screenshot through an injected adapter and persist it as a task artifact."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from modules.task_artifact_store import save_task_artifact


class ScreenshotAdapter(Protocol):
    def __call__(self) -> str | Path | None: ...


@dataclass(frozen=True)
class TaskScreenshotArtifactRecord:
    path: str
    filename: str
    size_bytes: int
    content_type: str


class TaskScreenshotCaptureError(RuntimeError):
    """Raised when screenshot capture does not produce a usable file."""


_CONTENT_TYPES_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def create_task_screenshot(
    *,
    screenshot_adapter: ScreenshotAdapter,
    task_artifact_dir: str | Path | None = None,
) -> TaskScreenshotArtifactRecord:
    """Capture a screenshot via the injected adapter and persist it through task artifact storage."""
    try:
        captured_path = screenshot_adapter()
    except Exception as exc:
        raise TaskScreenshotCaptureError(f"Screenshot capture failed: {exc}") from exc

    if captured_path is None:
        raise TaskScreenshotCaptureError("Screenshot capture did not produce a screenshot file.")
    if isinstance(captured_path, str) and not captured_path.strip():
        raise TaskScreenshotCaptureError("Screenshot capture did not produce a screenshot file.")

    resolved_captured_path = Path(captured_path).expanduser().resolve()
    if not resolved_captured_path.exists() or not resolved_captured_path.is_file():
        raise TaskScreenshotCaptureError(
            f"Screenshot file was not created by the capture backend: {resolved_captured_path}"
        )

    saved_path = save_task_artifact(
        resolved_captured_path,
        task_artifact_dir=task_artifact_dir,
    )
    _cleanup_captured_screenshot_source(
        source_path=resolved_captured_path,
        saved_path=saved_path,
    )
    return TaskScreenshotArtifactRecord(
        path=str(saved_path),
        filename=saved_path.name,
        size_bytes=saved_path.stat().st_size,
        content_type=_resolve_content_type(saved_path),
    )


def _resolve_content_type(path: Path) -> str:
    return _CONTENT_TYPES_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def _cleanup_captured_screenshot_source(*, source_path: Path, saved_path: Path) -> None:
    resolved_saved_path = saved_path.expanduser().resolve()
    if source_path == resolved_saved_path:
        return

    try:
        source_path.unlink(missing_ok=True)
    except OSError:
        # Cleanup is best-effort and must not mask a successful artifact save.
        return
