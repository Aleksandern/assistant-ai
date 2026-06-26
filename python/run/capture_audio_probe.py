#!/usr/bin/env python3
"""Thin entry point for the macOS ffmpeg audio probe module."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ffmpeg_audio_probe_macos import build_parser, ensure_ffmpeg, print_devices, record_wav


if __name__ == "__main__":
    args = build_parser().parse_args()
    ffmpeg = ensure_ffmpeg()
    if args.list_devices:
        raise SystemExit(print_devices(ffmpeg))
    raise SystemExit(
        record_wav(
            ffmpeg=ffmpeg,
            audio_device=args.audio_device,
            duration=args.duration,
            sample_rate=args.sample_rate,
            channels=args.channels,
            output_path=Path(args.output).resolve(),
        )
    )
