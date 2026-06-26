from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_artifact_store import (
    clear_task_artifacts,
    count_task_artifacts,
    get_task_artifact_dir,
    is_task_artifact_dir_empty,
    list_task_artifacts,
    save_task_artifact,
)


class TaskArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.task_dir = self.root_dir / "python" / "artifacts" / "process"
        self.source_dir = self.root_dir / "source"
        self.source_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_list_task_artifacts_returns_empty_list_for_empty_directory(self) -> None:
        self.assertEqual([], list_task_artifacts(self.task_dir))

    def test_count_task_artifacts_returns_zero_for_empty_directory(self) -> None:
        self.assertEqual(0, count_task_artifacts(self.task_dir))

    def test_is_task_artifact_dir_empty_returns_true_for_empty_directory(self) -> None:
        self.assertTrue(is_task_artifact_dir_empty(self.task_dir))

    def test_get_task_artifact_dir_creates_missing_directory(self) -> None:
        resolved_path = get_task_artifact_dir(self.task_dir)

        self.assertEqual(self.task_dir.resolve(), resolved_path)
        self.assertTrue(resolved_path.is_dir())

    def test_save_task_artifact_copies_screenshot_into_task_directory(self) -> None:
        screenshot_path = self.source_dir / "screen-1.png"
        screenshot_bytes = b"fake-png-binary"
        screenshot_path.write_bytes(screenshot_bytes)

        saved_path = save_task_artifact(screenshot_path, task_artifact_dir=self.task_dir)

        self.assertEqual(self.task_dir.resolve(), saved_path.parent)
        self.assertTrue(saved_path.exists())
        self.assertEqual(screenshot_bytes, saved_path.read_bytes())
        self.assertEqual("screen-1.png", saved_path.name)

    def test_save_task_artifact_rejects_filename_with_non_screenshot_extension(self) -> None:
        screenshot_path = self.source_dir / "screen-1.png"
        screenshot_path.write_bytes(b"fake-png-binary")

        with self.assertRaisesRegex(ValueError, "Task artifact filename must use a supported screenshot extension"):
            save_task_artifact(
                screenshot_path,
                task_artifact_dir=self.task_dir,
                filename="renamed.txt",
            )

    def test_list_and_count_task_artifacts_include_saved_screenshot(self) -> None:
        screenshot_path = self.source_dir / "screen-1.png"
        screenshot_path.write_bytes(b"fake-png-binary")

        saved_path = save_task_artifact(screenshot_path, task_artifact_dir=self.task_dir)

        self.assertEqual([saved_path], list_task_artifacts(self.task_dir))
        self.assertEqual(1, count_task_artifacts(self.task_dir))
        self.assertFalse(is_task_artifact_dir_empty(self.task_dir))

    def test_clear_task_artifacts_removes_files_from_task_directory(self) -> None:
        first_path = self.source_dir / "screen-1.png"
        first_path.write_bytes(b"first")
        second_path = self.source_dir / "screen-2.jpg"
        second_path.write_bytes(b"second")
        save_task_artifact(first_path, task_artifact_dir=self.task_dir)
        save_task_artifact(second_path, task_artifact_dir=self.task_dir)

        clear_task_artifacts(self.task_dir)

        self.assertEqual([], list_task_artifacts(self.task_dir))
        self.assertEqual(0, count_task_artifacts(self.task_dir))
        self.assertTrue(is_task_artifact_dir_empty(self.task_dir))
        self.assertEqual([], list(self.task_dir.iterdir()))

    def test_clear_task_artifacts_is_no_op_for_already_empty_directory(self) -> None:
        clear_task_artifacts(self.task_dir)

        self.assertTrue(self.task_dir.is_dir())
        self.assertEqual([], list(self.task_dir.iterdir()))

    def test_clear_task_artifacts_removes_all_task_owned_contents(self) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = self.task_dir / "screen-1.png"
        screenshot_path.write_bytes(b"png")
        notes_path = self.task_dir / "notes.txt"
        notes_path.write_text("remove", encoding="utf-8")
        metadata_dir = self.task_dir / "metadata"
        metadata_dir.mkdir()
        metadata_file = metadata_dir / "state.json"
        metadata_file.write_text("{}", encoding="utf-8")

        clear_task_artifacts(self.task_dir)

        self.assertFalse(screenshot_path.exists())
        self.assertFalse(notes_path.exists())
        self.assertFalse(metadata_dir.exists())
        self.assertFalse(metadata_file.exists())
        self.assertEqual([], list(self.task_dir.iterdir()))

    def test_list_and_count_task_artifacts_ignore_non_screenshot_files(self) -> None:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = self.task_dir / "screen-1.png"
        screenshot_path.write_bytes(b"png")
        ignored_text_path = self.task_dir / "notes.txt"
        ignored_text_path.write_text("ignore", encoding="utf-8")

        listed_paths = list_task_artifacts(self.task_dir)

        self.assertEqual([screenshot_path.resolve()], listed_paths)
        self.assertEqual(1, count_task_artifacts(self.task_dir))
        self.assertFalse(is_task_artifact_dir_empty(self.task_dir))


if __name__ == "__main__":
    unittest.main()
