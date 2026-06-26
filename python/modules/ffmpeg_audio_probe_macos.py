from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH. Install ffmpeg first.")
    return ffmpeg


DEVICE_LINE_RE = re.compile(r"^\[AVFoundation indev @ .*\] \[(\d+)\] (.+)$")


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def list_audio_devices(ffmpeg: str) -> list[tuple[str, str]]:
    proc = run_ffmpeg([ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""])
    devices: list[tuple[str, str]] = []
    in_audio_section = False

    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio_section = True
            continue
        if "AVFoundation video devices:" in line:
            in_audio_section = False
            continue
        if not in_audio_section:
            continue

        match = DEVICE_LINE_RE.match(line.strip())
        if match:
            devices.append((match.group(1), match.group(2)))

    return devices


def print_devices(ffmpeg: str) -> int:
    devices = list_audio_devices(ffmpeg)
    if not devices:
        print("No AVFoundation audio devices were listed by ffmpeg.")
        print("On macOS this often means the terminal has no microphone permission yet,")
        print("or ffmpeg only sees standard input devices and not system output audio.")
        return 1

    print("Audio devices visible to ffmpeg:")
    for idx, name in devices:
        print(f"  [{idx}] {name}")
    return 0


def record_wav(
    ffmpeg: str,
    audio_device: str,
    duration: int,
    sample_rate: int,
    channels: int,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        f":{audio_device}",
        "-t",
        str(duration),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    print("Recording command:")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        print(f"Saved WAV to: {output_path}")
    else:
        print("ffmpeg failed to record audio.")
        print("Try `--list-devices` first, then choose a visible input such as `default` or `0`.")
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe and record macOS audio devices via ffmpeg.")
    parser.add_argument("--list-devices", action="store_true", help="List AVFoundation audio devices visible to ffmpeg.")
    parser.add_argument("--audio-device", default="default", help="AVFoundation audio device name or index.")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=1, help="Number of output channels.")
    parser.add_argument("--output", default="python/artifacts/capture-test.wav", help="Path to the output WAV file.")
    return parser
