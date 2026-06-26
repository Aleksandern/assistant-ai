#!/usr/bin/env python3

"""Thin CLI wrapper around candidate profile import from DOCX into JSON."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.candidate_profile_importer import import_candidate_profile


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Import the first candidate file DOCX into data/file/file.json.")
    parser.add_argument(
        "--repository-root",
        default=str(PROJECT_ROOT.parent),
        help="Repository root that contains data/file/.",
    )
    args = parser.parse_args(argv)

    try:
        output_path = import_candidate_profile(repository_root=args.repository_root)
    except Exception as exc:
        print(f"Candidate profile import failed: {exc}")
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
