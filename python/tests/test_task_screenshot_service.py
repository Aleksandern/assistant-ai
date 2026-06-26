from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_artifact_store import count_task_artifacts
from modules.task_screenshot_service import (
    TaskScreenshotArtifactRecord,
    TaskScreenshotCaptureError,
    create_task_screenshot,
)


class TaskScreenshotServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.task_dir = self.root_dir / "python" / "artifacts" / "process"
        self.capture_dir = self.root_dir / "capture"
        self.capture_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_task_screenshot_captures_and_saves_artifact_via_task_store(self) -> None:
        captured_path = self.capture_dir / "active-app.png"
        captured_bytes = b"fake-png-binary"
        captured_path.write_bytes(captured_bytes)
        adapter = Mock(return_value=captured_path)

        artifact = create_task_screenshot(
            screenshot_adapter=adapter,
            task_artifact_dir=self.task_dir,
        )

        adapter.assert_called_once_with()
        self.assertEqual(
            TaskScreenshotArtifactRecord(
                path=str((self.task_dir / "active-app.png").resolve()),
                filename="active-app.png",
                size_bytes=len(captured_bytes),
                content_type="image/png",
            ),
            artifact,
        )
        self.assertEqual(1, count_task_artifacts(self.task_dir))
        self.assertEqual(captured_bytes, Path(artifact.path).read_bytes())
        self.assertFalse(captured_path.exists())

    def test_create_task_screenshot_does_not_delete_source_when_saved_path_matches_source(self) -> None:
        captured_path = self.capture_dir / "same-path.png"
        captured_bytes = b"fake-png-binary"
        captured_path.write_bytes(captured_bytes)

        with patch(
            "modules.task_screenshot_service.save_task_artifact",
            return_value=captured_path.resolve(),
        ):
            artifact = create_task_screenshot(
                screenshot_adapter=lambda: captured_path,
                task_artifact_dir=self.task_dir,
            )

        self.assertEqual(str(captured_path.resolve()), artifact.path)
        self.assertTrue(captured_path.exists())
        self.assertEqual(captured_bytes, captured_path.read_bytes())

    def test_create_task_screenshot_ignores_cleanup_oserror_after_successful_save(self) -> None:
        captured_path = self.capture_dir / "cleanup-error.png"
        captured_bytes = b"fake-png-binary"
        captured_path.write_bytes(captured_bytes)

        original_unlink = Path.unlink

        def failing_unlink(path: Path, *args: object, **kwargs: object) -> None:
            if path == captured_path.resolve():
                raise OSError("cleanup exploded")
            return original_unlink(path, *args, **kwargs)

        with patch("pathlib.Path.unlink", autospec=True, side_effect=failing_unlink):
            artifact = create_task_screenshot(
                screenshot_adapter=lambda: captured_path,
                task_artifact_dir=self.task_dir,
            )

        self.assertEqual(str((self.task_dir / "cleanup-error.png").resolve()), artifact.path)
        self.assertTrue(Path(artifact.path).exists())
        self.assertTrue(captured_path.exists())
        self.assertEqual(captured_bytes, captured_path.read_bytes())

    def test_create_task_screenshot_returns_predictable_record_for_jpeg_capture(self) -> None:
        captured_path = self.capture_dir / "window-shot.jpeg"
        captured_bytes = b"fake-jpeg-binary"
        captured_path.write_bytes(captured_bytes)

        artifact = create_task_screenshot(
            screenshot_adapter=lambda: captured_path,
            task_artifact_dir=self.task_dir,
        )

        self.assertEqual("window-shot.jpeg", artifact.filename)
        self.assertEqual(str((self.task_dir / "window-shot.jpeg").resolve()), artifact.path)
        self.assertEqual(len(captured_bytes), artifact.size_bytes)
        self.assertEqual("image/jpeg", artifact.content_type)

    def test_create_task_screenshot_raises_service_error_when_adapter_returns_no_file(self) -> None:
        with self.assertRaisesRegex(TaskScreenshotCaptureError, "did not produce a screenshot file"):
            create_task_screenshot(
                screenshot_adapter=lambda: None,
                task_artifact_dir=self.task_dir,
            )

        self.assertEqual(0, count_task_artifacts(self.task_dir))

    def test_create_task_screenshot_raises_service_error_when_adapter_returns_empty_path(self) -> None:
        with self.assertRaisesRegex(TaskScreenshotCaptureError, "did not produce a screenshot file"):
            create_task_screenshot(
                screenshot_adapter=lambda: "   ",
                task_artifact_dir=self.task_dir,
            )

        self.assertEqual(0, count_task_artifacts(self.task_dir))

    def test_create_task_screenshot_raises_service_error_when_adapter_returns_missing_file(self) -> None:
        missing_path = self.capture_dir / "missing.png"

        with self.assertRaisesRegex(TaskScreenshotCaptureError, "Screenshot file was not created"):
            create_task_screenshot(
                screenshot_adapter=lambda: missing_path,
                task_artifact_dir=self.task_dir,
            )

        self.assertEqual(0, count_task_artifacts(self.task_dir))

    def test_create_task_screenshot_maps_adapter_exception_to_service_error(self) -> None:
        with self.assertRaisesRegex(TaskScreenshotCaptureError, "Screenshot capture failed: backend exploded"):
            create_task_screenshot(
                screenshot_adapter=lambda: (_ for _ in ()).throw(RuntimeError("backend exploded")),
                task_artifact_dir=self.task_dir,
            )

        self.assertEqual(0, count_task_artifacts(self.task_dir))

    def test_create_task_screenshot_does_not_swallow_task_store_failures(self) -> None:
        captured_path = self.capture_dir / "active-app.png"
        captured_path.write_bytes(b"fake-png-binary")

        with patch(
            "modules.task_screenshot_service.save_task_artifact",
            side_effect=RuntimeError("store exploded"),
        ):
            with self.assertRaisesRegex(RuntimeError, "store exploded"):
                create_task_screenshot(
                    screenshot_adapter=lambda: captured_path,
                    task_artifact_dir=self.task_dir,
                )


if __name__ == "__main__":
    unittest.main()
