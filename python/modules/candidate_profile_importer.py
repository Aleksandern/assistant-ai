from __future__ import annotations

"""Import a candidate file from DOCX and persist a structured JSON profile."""

# TODO: File parsing is still heuristic and incomplete. Return here later to
# improve multi-line project summaries and broader field extraction accuracy.

import json
import re
import zipfile
from pathlib import Path
from typing import TypedDict
from xml.etree import ElementTree


class CandidateProject(TypedDict):
    name: str
    summary: str
    tech: list[str]


class CandidateProfile(TypedDict):
    full_name: str
    target_role: str
    years_experience: int | None
    primary_stack: list[str]
    secondary_stack: list[str]
    strengths: list[str]
    projects: list[CandidateProject] | list[str]
    style_rules: list[str]
    avoid_claims: list[str]
    raw_text: str


PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_FILE_DIRECTORY = REPOSITORY_ROOT / "data" / "file"
DEFAULT_FILE_JSON_PATH = DEFAULT_FILE_DIRECTORY / "file.json"
WORDPROCESSING_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def import_candidate_profile(*, repository_root: str | Path | None = None) -> Path:
    resolved_repository_root = Path(repository_root or REPOSITORY_ROOT).expanduser().resolve()
    file_directory = resolved_repository_root / "data" / "file"
    source_docx_path = _find_first_docx(file_directory)
    raw_text = _extract_raw_text_from_docx(source_docx_path)
    profile = _build_candidate_profile(raw_text=raw_text)
    output_path = file_directory / "file.json"
    _write_candidate_profile(output_path=output_path, profile=profile)
    return output_path.resolve()


def _find_first_docx(file_directory: Path) -> Path:
    docx_paths = sorted(path for path in file_directory.iterdir() if path.is_file() and path.suffix.lower() == ".docx")
    if not docx_paths:
        raise FileNotFoundError("No .docx file found in data/file")
    return docx_paths[0]


def _extract_raw_text_from_docx(docx_path: Path) -> str:
    try:
        with zipfile.ZipFile(docx_path) as archive:
            document_xml = archive.read("word/document.xml")
    except RuntimeError:
        raise
    except KeyError as exc:
        raise RuntimeError(f"Failed to extract text from file DOCX file: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read file DOCX file: {exc}") from exc

    try:
        raw_text = _extract_text_from_document_xml(document_xml)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from file DOCX file: {exc}") from exc

    if not raw_text:
        raise RuntimeError("Failed to extract text from file DOCX file: extracted text is empty.")

    return raw_text


def _extract_text_from_document_xml(document_xml: bytes) -> str:
    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:p", WORDPROCESSING_NAMESPACE):
        fragments: list[str] = []
        for node in paragraph.iter():
            if node.tag == _namespaced_tag("t") and node.text:
                fragments.append(node.text)
            elif node.tag == _namespaced_tag("tab"):
                fragments.append("\t")
            elif node.tag in {_namespaced_tag("br"), _namespaced_tag("cr")}:
                fragments.append("\n")

        normalized_paragraph = "".join(fragments).strip()
        if normalized_paragraph:
            paragraphs.append(normalized_paragraph)

    return "\n".join(paragraphs).strip()


def _namespaced_tag(local_name: str) -> str:
    return f"{{{WORDPROCESSING_NAMESPACE['w']}}}{local_name}"


def _build_candidate_profile(*, raw_text: str) -> CandidateProfile:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    full_name = lines[0] if lines else ""
    target_role = _extract_target_role(lines)
    years_experience = _extract_years_experience(raw_text)
    primary_stack, secondary_stack = _extract_stacks(lines)
    strengths = _extract_strengths(lines)
    projects = _extract_projects(lines)

    return CandidateProfile(
        full_name=full_name,
        target_role=target_role,
        years_experience=years_experience,
        primary_stack=primary_stack,
        secondary_stack=secondary_stack,
        strengths=strengths,
        projects=projects,
        style_rules=[],
        avoid_claims=[],
        raw_text=raw_text,
    )


def _extract_target_role(lines: list[str]) -> str:
    for line in lines[1:5]:
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()

    for line in lines[1:4]:
        if "@" in line:
            continue
        if re.search(r"https?://|www\\.|linkedin|github|phone|email", line, re.IGNORECASE):
            continue
        return line

    return ""


def _extract_years_experience(raw_text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\+?\s+years?\b", raw_text, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _extract_stacks(lines: list[str]) -> tuple[list[str], list[str]]:
    primary_stack: list[str] = []
    secondary_stack: list[str] = []
    section_lines = _extract_section_lines(lines, "Core Expertise:")
    if not section_lines:
        return primary_stack, secondary_stack

    current_subsection = ""
    for line in section_lines:
        if line.endswith(":"):
            current_subsection = line[:-1].strip().lower()
            continue

        items = _split_comma_separated_items(line)
        if not items:
            continue

        if "real-time" in current_subsection or "event-driven" in current_subsection:
            secondary_stack.extend(items)
            continue

        primary_stack.extend(items)

    return _deduplicate(primary_stack), _deduplicate(secondary_stack)


def _extract_strengths(lines: list[str]) -> list[str]:
    strengths: list[str] = []
    for line in lines:
        if line.lower().startswith("focused on "):
            strengths.append(line)
    return strengths


def _extract_projects(lines: list[str]) -> list[CandidateProject] | list[str]:
    project_lines = _extract_section_lines_by_prefix(lines, "Selected")
    if not project_lines:
        return []

    projects: list[CandidateProject] = []
    current_project: CandidateProject | None = None

    for line in project_lines:
        if line.endswith(":"):
            continue

        if _looks_like_project_heading(line):
            if current_project is not None:
                projects.append(current_project)
            current_project = CandidateProject(
                name=_extract_project_name(line),
                summary="",
                tech=[],
            )
            continue

        if current_project is None:
            continue

        if line.startswith("Tech:"):
            current_project["tech"] = _split_comma_separated_items(line.split(":", 1)[1].strip())
            continue

        if _is_url_line(line):
            continue

        if not current_project["summary"]:
            current_project["summary"] = line

    if current_project is not None:
        projects.append(current_project)

    return [project for project in projects if project["name"]]


def _extract_section_lines(lines: list[str], header: str) -> list[str]:
    try:
        start_index = lines.index(header)
    except ValueError:
        return []

    collected: list[str] = []
    for line in lines[start_index + 1 :]:
        if _is_major_section_heading(line):
            break
        collected.append(line)
    return collected


def _extract_section_lines_by_prefix(lines: list[str], header_prefix: str) -> list[str]:
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith(header_prefix) and line.endswith(":"):
            start_index = index
            break

    if start_index is None:
        return []

    collected: list[str] = []
    for line in lines[start_index + 1 :]:
        if _is_major_section_heading(line):
            break
        collected.append(line)
    return collected


def _is_major_section_heading(line: str) -> bool:
    major_headings = {
        "Core Expertise:",
        "Systems & Production Experience",
        "Professional Experience:",
        "Selected Mobile & Full-Stack Projects:",
        "Selected Projects:",
        "Additional Projects",
        "Education:",
        "Additional Focus",
    }
    return line in major_headings


def _split_comma_separated_items(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0

    for character in value:
        if character == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue

        if character == "(":
            depth += 1
        elif character == ")" and depth > 0:
            depth -= 1

        current.append(character)

    tail = "".join(current).strip()
    if tail:
        items.append(tail)

    return items


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _looks_like_project_heading(line: str) -> bool:
    if line.startswith("Tech:"):
        return False
    if " – " in line or " - " in line:
        return True
    return False


def _extract_project_name(line: str) -> str:
    left_part = re.split(r"\s+[–-]\s+", line, maxsplit=1)[0]
    return left_part.strip().rstrip(".")


def _is_url_line(line: str) -> bool:
    return bool(re.match(r"^https?://", line, re.IGNORECASE))


def _write_candidate_profile(*, output_path: Path, profile: CandidateProfile) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write file JSON file: {exc}") from exc
