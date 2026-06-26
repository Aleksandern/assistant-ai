#!/usr/bin/env python3
"""
Thin entry point for the macOS ScreenCaptureKit audio capture module.

What this script is:
- A small test utility for AssistantAI.
- It captures audio from a display or a specific macOS application such as
  Zoom, Telegram, or a browser call.
- It saves the captured result to WAV so we can verify that system audio can
  be read without a virtual audio device.

Quick start:
1. Create a venv:
   python3 -m venv .venv
2. Install Python dependencies:
   .venv/bin/pip install pyobjc-core pyobjc-framework-ScreenCaptureKit
3. Install ffmpeg:
   brew install ffmpeg
4. In macOS System Settings, allow Screen Recording for the terminal app that
   runs this script, then restart that terminal.
5. List available targets:
   python/.venv/bin/python python/run/capture_system_audio_macos.py --list-targets
6. Record audio from a specific app:
   python/.venv/bin/python python/run/capture_system_audio_macos.py --app-name Zoom --duration 10 --output python/artifacts/zoom.wav

Available options:
- --list-targets
  Shows shareable displays and applications visible to ScreenCaptureKit.
- --app-name <name>
  Captures audio from an application whose visible name matches the given text.
- --bundle-id <id>
  Captures audio from an application by exact macOS bundle identifier.
- --display-index <index>
  Chooses which display context to use when building the capture stream.
- --duration <seconds>
  Controls how long the recording should run.
- --output <path>
  Sets the WAV file path to create.
- --sample-rate <hz>
  Sets the output WAV sample rate.
- --channels <count>
  Sets the output WAV channel count.
- --include-self-audio
  Includes audio from the current process instead of excluding it.
- --timeout <seconds>
  Sets how long the script waits for ScreenCaptureKit async operations.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.macos_audio_listener import capture_segment_to_movie, ensure_ffmpeg, list_targets
from modules.wav_file_writer import convert_movie_to_wav


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture macOS system audio to WAV using ScreenCaptureKit.")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds.")
    parser.add_argument("--display-index", type=int, default=0, help="Display index from shareable content.")
    parser.add_argument("--app-name", help="Capture a specific app by visible name. Example: Zoom.")
    parser.add_argument("--bundle-id", help="Capture a specific app by exact macOS bundle identifier.")
    parser.add_argument("--list-targets", action="store_true", help="List shareable displays and applications.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output WAV sample rate.")
    parser.add_argument("--channels", type=int, default=1, help="Output WAV channel count.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "artifacts/system-audio-test.wav"), help="Output WAV path.")
    parser.add_argument("--include-self-audio", action="store_true", help="Include audio from the current process.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout for ScreenCaptureKit async operations.")
    args = parser.parse_args()

    try:
        if args.list_targets:
            list_targets(timeout=args.timeout)
            raise SystemExit(0)

        ffmpeg = ensure_ffmpeg()
        movie_path, _ = capture_segment_to_movie(
            duration=args.duration,
            display_index=args.display_index,
            app_name=args.app_name,
            bundle_id=args.bundle_id,
            include_self_audio=args.include_self_audio,
            timeout=args.timeout,
        )
        try:
            output_path = Path(args.output).resolve()
            convert_movie_to_wav(ffmpeg, movie_path, output_path, args.sample_rate, args.channels)
            print(f"Saved WAV to: {output_path}")
        finally:
            movie_path.unlink(missing_ok=True)
        raise SystemExit(0)
    except Exception as exc:
        print(f"Capture failed: {exc}")
        print("Most likely causes on macOS:")
        print("  1. Screen Recording permission was not granted to the terminal or Python process.")
        print("  2. ScreenCaptureKit did not expose a display in the current session.")
        print("  3. The API captured no usable audio for the chosen target.")
        raise SystemExit(1)
