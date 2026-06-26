from __future__ import annotations

import subprocess
import time
import wave
from pathlib import Path

import numpy


def convert_movie_to_wav(ffmpeg: str, input_path: Path, output_path: Path, sample_rate: int, channels: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_timestamped_wav_path(output_dir: Path, prefix: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    millis = int((time.time() % 1) * 1000)
    return output_dir / f"{prefix}-{timestamp}-{millis:03d}.wav"


def merge_wav_files(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise RuntimeError("No WAV files were provided for merging.")

    params = None
    frames: list[bytes] = []

    for path in input_paths:
        with wave.open(str(path), "rb") as wav_file:
            current_params = wav_file.getparams()
            if params is None:
                params = current_params
            elif current_params[:4] != params[:4]:
                raise RuntimeError("All WAV chunks must share the same audio format before merging.")
            frames.append(wav_file.readframes(wav_file.getnframes()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setparams(params)
        for chunk in frames:
            wav_file.writeframes(chunk)


def write_mono_audio_to_wav(audio: numpy.ndarray, output_path: Path, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = numpy.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(numpy.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
