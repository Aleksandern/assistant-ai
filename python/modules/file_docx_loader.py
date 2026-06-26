from __future__ import annotations

"""Locate and load the source candidate DOCX file."""

from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_FILE_DIRECTORY = REPOSITORY_ROOT / "data" / "file"


def find_first_file_docx(file_dir: str | Path | None = None) -> Path:
    resolved_file_dir = _resolve_file_directory(file_dir)

    if not resolved_file_dir.exists():
        raise FileNotFoundError(f"File directory does not exist: {resolved_file_dir}")

    if not resolved_file_dir.is_dir():
        raise NotADirectoryError(f"File path is not a directory: {resolved_file_dir}")

    docx_paths = sorted(
        path for path in resolved_file_dir.iterdir() if path.is_file() and path.suffix.lower() == ".docx"
    )
    if not docx_paths:
        raise FileNotFoundError(f"No .docx files found in file directory: {resolved_file_dir}")

    return docx_paths[0].resolve()


def load_file_docx_bytes(file_dir: str | Path | None = None) -> bytes:
    file_docx_path = find_first_file_docx(file_dir)

    try:
        return file_docx_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Failed to read file DOCX file: {file_docx_path}") from exc


def _resolve_file_directory(file_dir: str | Path | None) -> Path:
    if file_dir is None:
        return DEFAULT_FILE_DIRECTORY

    return Path(file_dir).expanduser().resolve()
