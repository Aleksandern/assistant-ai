from __future__ import annotations

"""Contract tests for candidate profile import from DOCX into JSON."""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.candidate_profile_importer import import_candidate_profile


class CandidateProfileImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.file_dir = self.root_dir / "data" / "file"
        self.file_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_import_candidate_profile_creates_file_json_from_first_docx(self) -> None:
        self._create_docx(
            self.file_dir / "candidate-file.docx",
            [
                "Jane Doe",
                "Title: Senior Python Developer",
                "Senior Python Developer with 8+ years of experience.",
                "Core Expertise:",
                "Backend Systems:",
                "Python, FastAPI, PostgreSQL",
                "Selected Projects:",
                "Billing Platform - backend system for payment processing",
            ],
        )

        output_path = import_candidate_profile(repository_root=self.root_dir)

        self.assertEqual((self.file_dir / "file.json").resolve(), output_path)
        self.assertTrue(output_path.exists())

        profile = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual("Jane Doe", profile["full_name"])
        self.assertEqual("Senior Python Developer", profile["target_role"])
        self.assertEqual(8, profile["years_experience"])
        self.assertEqual(["Python", "FastAPI", "PostgreSQL"], profile["primary_stack"])
        self.assertEqual([], profile["secondary_stack"])
        self.assertEqual(
            [
                {
                    "name": "Billing Platform",
                    "summary": "",
                    "tech": [],
                }
            ],
            profile["projects"],
        )
        self.assertIn("Jane Doe", profile["raw_text"])
        self.assertIn("Senior Python Developer", profile["raw_text"])

    def test_import_candidate_profile_does_not_depend_on_filename(self) -> None:
        self._create_docx(
            self.file_dir / "z-last-name.docx",
            ["Ignored Candidate", "Title: QA Engineer"],
        )
        self._create_docx(
            self.file_dir / "a-first-name.docx",
            ["Alice Johnson", "Title: Backend Engineer"],
        )

        output_path = import_candidate_profile(repository_root=self.root_dir)
        profile = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("Alice Johnson", profile["full_name"])
        self.assertEqual("Backend Engineer", profile["target_role"])
        self.assertIn("Alice Johnson", profile["raw_text"])
        self.assertNotIn("Ignored Candidate", profile["raw_text"])

    def test_import_candidate_profile_raises_clear_error_when_no_docx_exists(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "No .docx file found in data/file"):
            import_candidate_profile(repository_root=self.root_dir)

    def test_import_candidate_profile_preserves_raw_text(self) -> None:
        self._create_docx(
            self.file_dir / "file-source.docx",
            [
                "Maria Petrova",
                "Platform Engineer",
                "Built internal tooling for deployment automation.",
            ],
        )

        output_path = import_candidate_profile(repository_root=self.root_dir)
        profile = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            "Maria Petrova\nPlatform Engineer\nBuilt internal tooling for deployment automation.",
            profile["raw_text"],
        )

    def test_import_candidate_profile_does_not_invent_missing_fields(self) -> None:
        self._create_docx(
            self.file_dir / "minimal.docx",
            ["Candidate Without Details"],
        )

        output_path = import_candidate_profile(repository_root=self.root_dir)
        profile = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("Candidate Without Details", profile["full_name"])
        self.assertEqual("", profile["target_role"])
        self.assertIsNone(profile["years_experience"])
        self.assertEqual([], profile["primary_stack"])
        self.assertEqual([], profile["secondary_stack"])
        self.assertEqual([], profile["strengths"])
        self.assertEqual([], profile["projects"])
        self.assertEqual([], profile["style_rules"])
        self.assertEqual([], profile["avoid_claims"])

    def test_import_candidate_profile_extracts_structured_fields_from_realistic_sections(self) -> None:
        self._create_docx(
            self.file_dir / "structured.docx",
            [
                "Aleksandr Melnikov",
                "GitHub: https://github.com/aleksandern",
                "Title: Senior Full-Stack Engineer (Web + Mobile + Backend Systems)",
                "Senior Full-Stack Engineer with 15+ years of experience, including 7+ years building production mobile applications with React Native.",
                "Focused on building reliable, scalable mobile systems with real-time data flows, consistent state management, and strong backend integration.",
                "Core Expertise:",
                "Web/Mobile Systems:",
                "React, React Native, Native Modules, Performance Optimization, App Architecture",
                "Backend Systems:",
                "Node.js, NestJS, API Design, Scalable Services, MongoDB, PostgreSQL, MySQL",
                "Real-time & Event-driven Systems:",
                "WebSockets, Push Notifications, State Synchronization",
                "Selected Mobile & Full-Stack Projects:",
                "Puppy Island – Mobile & Backend System (iOS, Android, Web).",
                "Led development of a multi-role platform with complex operational workflows and real-time coordination between staff, customers, and admin systems.",
                "Tech: React Native, Next.js, Node.js (NestJS), MongoDB, AWS",
                "Springshot – Mobile Application (iOS, Android).",
                "Worked on an orchestration platform for airlines that helps mobile teams coordinate and perform time-sensitive operational tasks.",
                "Tech: React Native, Firebase, Socket.IO",
            ],
        )

        output_path = import_candidate_profile(repository_root=self.root_dir)
        profile = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("Senior Full-Stack Engineer (Web + Mobile + Backend Systems)", profile["target_role"])
        self.assertEqual(15, profile["years_experience"])
        self.assertEqual(
            [
                "React",
                "React Native",
                "Native Modules",
                "Performance Optimization",
                "App Architecture",
                "Node.js",
                "NestJS",
                "API Design",
                "Scalable Services",
                "MongoDB",
                "PostgreSQL",
                "MySQL",
            ],
            profile["primary_stack"],
        )
        self.assertEqual(
            ["WebSockets", "Push Notifications", "State Synchronization"],
            profile["secondary_stack"],
        )
        self.assertIn(
            "Focused on building reliable, scalable mobile systems with real-time data flows, consistent state management, and strong backend integration.",
            profile["strengths"],
        )
        self.assertEqual(
            [
                {
                    "name": "Puppy Island",
                    "summary": "Led development of a multi-role platform with complex operational workflows and real-time coordination between staff, customers, and admin systems.",
                    "tech": ["React Native", "Next.js", "Node.js (NestJS)", "MongoDB", "AWS"],
                },
                {
                    "name": "Springshot",
                    "summary": "Worked on an orchestration platform for airlines that helps mobile teams coordinate and perform time-sensitive operational tasks.",
                    "tech": ["React Native", "Firebase", "Socket.IO"],
                },
            ],
            profile["projects"],
        )

    def test_import_candidate_profile_keeps_parenthesized_stack_items_and_skips_project_urls_in_summary(self) -> None:
        self._create_docx(
            self.file_dir / "sections.docx",
            [
                "Nina Example",
                "Title: Staff Engineer",
                "Core Expertise:",
                "Integrations:",
                "Payments (Stripe, transactional flows, failure handling), Messaging, Third-party APIs",
                "Selected Projects:",
                "Checkout Platform – Payment orchestration system.",
                "https://checkout.example.com",
                "Handled retries, failures, and transactional integrity.",
                "Tech: Node.js, PostgreSQL",
            ],
        )

        output_path = import_candidate_profile(repository_root=self.root_dir)
        profile = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [
                "Payments (Stripe, transactional flows, failure handling)",
                "Messaging",
                "Third-party APIs",
            ],
            profile["primary_stack"],
        )
        self.assertEqual(
            [
                {
                    "name": "Checkout Platform",
                    "summary": "Handled retries, failures, and transactional integrity.",
                    "tech": ["Node.js", "PostgreSQL"],
                }
            ],
            profile["projects"],
        )

    def test_import_candidate_profile_raises_clear_error_when_docx_cannot_be_read(self) -> None:
        broken_docx_path = self.file_dir / "broken.docx"
        broken_docx_path.write_text("not a real docx archive", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "Failed to read file DOCX file"):
            import_candidate_profile(repository_root=self.root_dir)

    def test_import_candidate_profile_raises_clear_error_when_text_cannot_be_extracted(self) -> None:
        docx_path = self.file_dir / "missing-document-xml.docx"
        with zipfile.ZipFile(docx_path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types></Types>")

        with self.assertRaisesRegex(RuntimeError, "Failed to extract text from file DOCX file"):
            import_candidate_profile(repository_root=self.root_dir)

    def test_import_candidate_profile_raises_clear_error_when_file_json_cannot_be_written(self) -> None:
        self._create_docx(
            self.file_dir / "candidate.docx",
            ["Nina Example", "Data Engineer"],
        )
        (self.file_dir / "file.json").mkdir()

        with self.assertRaisesRegex(RuntimeError, "Failed to write file JSON file"):
            import_candidate_profile(repository_root=self.root_dir)

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
