from __future__ import annotations

"""Tests for the candidate profile import runner."""

import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run.import_candidate_profile import main


class ImportCandidateProfileRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.file_dir = self.root_dir / "data" / "file"
        self.file_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_main_prints_file_json_path_on_success(self) -> None:
        self._create_docx(self.file_dir / "candidate.docx", ["John Doe", "Python Developer"])
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--repository-root", str(self.root_dir)])

        self.assertEqual(0, exit_code)
        self.assertEqual(str((self.file_dir / "file.json").resolve()), stdout.getvalue().strip())

    def test_main_prints_clear_error_and_returns_non_zero_on_failure(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = main(["--repository-root", str(self.root_dir)])

        self.assertEqual(1, exit_code)
        self.assertIn("Candidate profile import failed", stdout.getvalue())
        self.assertIn("No .docx file found in data/file", stdout.getvalue())

    def _create_docx(self, path: Path, paragraphs: list[str]) -> None:
        paragraph_xml = "".join(
            f"<w:p><w:r><w:t>{self._escape_xml(paragraph)}</w:t></w:r></w:p>"
            for paragraph in paragraphs
        )
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{paragraph_xml}</w:body>"
            "</w:document>"
        )

        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", self._content_types_xml())
            archive.writestr("_rels/.rels", self._rels_xml())
            archive.writestr("word/document.xml", document_xml)

    def _content_types_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )

    def _rels_xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )

    def _escape_xml(self, value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


if __name__ == "__main__":
    unittest.main()
