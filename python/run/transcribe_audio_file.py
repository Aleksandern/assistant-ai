#!/usr/bin/env python3

"""Thin CLI wrapper around the whisper.cpp transcription module.

Usage:
- `python/.venv/bin/python python/run/transcribe_audio_file.py /path/to/audio.wav`
- prints recognized text to stdout
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.whisper_cpp_transcriber import transcribe_audio_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe one audio file to text with whisper.cpp.")
    parser.add_argument("audio_path", help="Path to the source audio file.")
    parser.add_argument("--language", help="Optional Whisper language code such as en or ru.")
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "libs" / "whisper.cpp" / "models" / "ggml-base.bin"),
        help="Path to the ggml Whisper model file.",
    )
    parser.add_argument(
        "--whisper-cli-path",
        default=str(PROJECT_ROOT / "libs" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        help="Path to the whisper.cpp CLI binary.",
    )
    parser.add_argument("--ffmpeg-path", help="Optional explicit path to ffmpeg.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable whisper.cpp GPU inference instead of the safer CPU default.")
    args = parser.parse_args()

    try:
        transcript = transcribe_audio_file(
            args.audio_path,
            whisper_cli_path=args.whisper_cli_path,
            model_path=args.model_path,
            ffmpeg_path=args.ffmpeg_path,
            language=args.language,
            use_gpu=args.use_gpu,
        )
        print(transcript)
        raise SystemExit(0)
    except Exception as exc:
        print(f"Transcription failed: {exc}")
        raise SystemExit(1)
