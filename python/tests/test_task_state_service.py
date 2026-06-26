from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_artifact_store import save_task_artifact
from modules.task_state_service import TaskArtifactSnapshot, TaskStateSnapshot, get_task_state_snapshot


class TaskStateServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.task_dir = self.root_dir / "python" / "artifacts" / "process"
        self.source_dir = self.root_dir / "source"
        self.source_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_task_state_snapshot_returns_empty_when_no_task_files_exist(self) -> None:
        snapshot = get_task_state_snapshot(self.task_dir)

        self.assertEqual(
            TaskStateSnapshot(
                status="empty",
                file_count=0,
                artifacts=[],
            ),
            snapshot,
        )

    def test_get_task_state_snapshot_returns_ready_when_screenshot_exists(self) -> None:
        screenshot_path = self.source_dir / "screen-1.png"
        screenshot_path.write_bytes(b"fake-png-binary")
        saved_path = save_task_artifact(screenshot_path, task_artifact_dir=self.task_dir)

        snapshot = get_task_state_snapshot(self.task_dir)

        self.assertEqual("ready", snapshot.status)
        self.assertEqual(1, snapshot.file_count)
        self.assertEqual(
            [
                TaskArtifactSnapshot(
                    filename=saved_path.name,
                    path=str(saved_path),
                    size_bytes=len(b"fake-png-binary"),
                )
            ],
            snapshot.artifacts,
        )

    def test_get_task_state_snapshot_reports_correct_file_count_for_multiple_screenshots(self) -> None:
        first_path = self.source_dir / "screen-1.png"
        first_path.write_bytes(b"first")
        second_path = self.source_dir / "screen-2.jpg"
        second_path.write_bytes(b"second")
        save_task_artifact(first_path, task_artifact_dir=self.task_dir)
        save_task_artifact(second_path, task_artifact_dir=self.task_dir)

        snapshot = get_task_state_snapshot(self.task_dir)

        self.assertEqual("ready", snapshot.status)
        self.assertEqual(2, snapshot.file_count)
        self.assertEqual(["screen-1.png", "screen-2.jpg"], [artifact.filename for artifact in snapshot.artifacts])

    def test_get_task_state_snapshot_has_stable_shape_for_future_ui_consumers(self) -> None:
        snapshot = get_task_state_snapshot(self.task_dir)

        self.assertIsInstance(snapshot, TaskStateSnapshot)
        self.assertIsInstance(snapshot.status, str)
        self.assertIsInstance(snapshot.file_count, int)
        self.assertIsInstance(snapshot.artifacts, list)

    def test_get_task_state_snapshot_does_not_include_non_task_conversation_fields(self) -> None:
        snapshot = get_task_state_snapshot(self.task_dir)

        self.assertFalse(hasattr(snapshot, "conversation_id"))
        self.assertFalse(hasattr(snapshot, "remote_text"))
        self.assertFalse(hasattr(snapshot, "reply_text"))
        self.assertFalse(hasattr(snapshot, "is_active"))


if __name__ == "__main__":
    unittest.main()
