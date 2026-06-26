from __future__ import annotations

"""Contract tests for locating and loading source file DOCX files."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.file_docx_loader import find_first_file_docx, load_file_docx_bytes


class FileDocxLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_dir = Path(self.temp_dir.name) / "data" / "file"
        self.file_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_find_first_file_docx_returns_first_sorted_docx_path(self) -> None:
        ignored_txt = self.file_dir / "a-ignore.txt"
        ignored_txt.write_text("ignore", encoding="utf-8")
        later_docx = self.file_dir / "z-file.docx"
        later_docx.write_bytes(b"later")
        first_docx = self.file_dir / "b-file.docx"
        first_docx.write_bytes(b"first")

        result = find_first_file_docx(self.file_dir)

        self.assertEqual(first_docx.resolve(), result)
        self.assertIsInstance(result, Path)

    def test_find_first_file_docx_does_not_depend_on_specific_filename(self) -> None:
        unexpected_name = self.file_dir / "totally-custom-name.docx"
        unexpected_name.write_bytes(b"custom")

        result = find_first_file_docx(self.file_dir)

        self.assertEqual(unexpected_name.resolve(), result)

    def test_find_first_file_docx_selection_is_deterministic(self) -> None:
        first_docx = self.file_dir / "alpha.docx"
        first_docx.write_bytes(b"alpha")
        second_docx = self.file_dir / "beta.docx"
        second_docx.write_bytes(b"beta")

        first_result = find_first_file_docx(self.file_dir)
        second_result = find_first_file_docx(self.file_dir)

        self.assertEqual(first_docx.resolve(), first_result)
        self.assertEqual(first_result, second_result)

    def test_find_first_file_docx_matches_docx_extension_case_insensitively(self) -> None:
        uppercase_docx = self.file_dir / "file.DOCX"
        uppercase_docx.write_bytes(b"uppercase")

        result = find_first_file_docx(self.file_dir)

        self.assertEqual(uppercase_docx.resolve(), result)

    def test_load_file_docx_bytes_returns_binary_contents(self) -> None:
        file_docx = self.file_dir / "file-source.docx"
        expected_bytes = b"fake-docx-binary"
        file_docx.write_bytes(expected_bytes)

        result = load_file_docx_bytes(self.file_dir)

        self.assertEqual(expected_bytes, result)
        self.assertIsInstance(result, bytes)

    def test_find_first_file_docx_raises_clear_error_when_directory_is_missing(self) -> None:
        missing_dir = Path(self.temp_dir.name) / "missing" / "file"

        with self.assertRaisesRegex(FileNotFoundError, "File directory does not exist"):
            find_first_file_docx(missing_dir)

    def test_find_first_file_docx_raises_clear_error_when_no_docx_files_exist(self) -> None:
        (self.file_dir / "file.txt").write_text("not docx", encoding="utf-8")

        with self.assertRaisesRegex(FileNotFoundError, "No \\.docx files found in file directory"):
            find_first_file_docx(self.file_dir)

    def test_find_first_file_docx_raises_clear_error_when_path_is_not_directory(self) -> None:
        not_a_directory = self.file_dir / "file.docx"
        not_a_directory.write_bytes(b"fake-docx")

        with self.assertRaisesRegex(NotADirectoryError, "File path is not a directory"):
            find_first_file_docx(not_a_directory)

    def test_load_file_docx_bytes_raises_clear_error_when_file_cannot_be_read(self) -> None:
        file_docx = self.file_dir / "file.docx"
        file_docx.write_bytes(b"fake-docx")

        with patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied")):
            with self.assertRaisesRegex(RuntimeError, "Failed to read file DOCX file"):
                load_file_docx_bytes(self.file_dir)


if __name__ == "__main__":
    unittest.main()
