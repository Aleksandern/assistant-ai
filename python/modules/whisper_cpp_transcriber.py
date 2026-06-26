from __future__ import annotations

"""Transcribe one local audio file to plain text via whisper.cpp.

Usage:
- call `transcribe_audio_file("/path/to/audio.wav")`
- call `transcribe_audio_buffer(audio, sample_rate=16000)` for mono float32 audio in memory
- the module converts input audio to 16 kHz mono WAV via ffmpeg
- then runs `whisper-cli` and returns the transcript string
"""

import io
import numbers
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHISPER_CPP_DIR = PROJECT_ROOT / "libs" / "whisper.cpp"
DEFAULT_WHISPER_CPP_BINARY = DEFAULT_WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
DEFAULT_WHISPER_CPP_MODEL = DEFAULT_WHISPER_CPP_DIR / "models" / "ggml-base.bin"
DEFAULT_WHISPER_SAMPLE_RATE = 16000
DEFAULT_WHISPER_CPP_RPATHS = (
    "@executable_path/../src",
    "@executable_path/../ggml/src",
    "@executable_path/../ggml/src/ggml-blas",
    "@executable_path/../ggml/src/ggml-metal",
)
MACHO_MAGIC_PREFIXES = (
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
)


def transcribe_audio_file(
    audio_path: str | Path,
    *,
    whisper_cli_path: str | Path | None = None,
    model_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    language: str | None = None,
    use_gpu: bool = False,
) -> str:
    input_path = _resolve_existing_file(audio_path, label="Audio file")
    whisper_cli, model = _resolve_whisper_runtime(
        whisper_cli_path=whisper_cli_path,
        model_path=model_path,
    )
    ffmpeg = _resolve_ffmpeg_binary(ffmpeg_path)

    with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prepared_audio_path = temp_dir / "input.wav"

        _convert_audio_for_whisper(
            ffmpeg=ffmpeg,
            input_path=input_path,
            output_path=prepared_audio_path,
        )

        return _transcribe_prepared_input(
            whisper_cli=whisper_cli,
            model=model,
            input_source=prepared_audio_path,
            transcript_dir=temp_dir,
            language=language,
            use_gpu=use_gpu,
        )


def transcribe_audio_buffer(
    audio: numpy.ndarray,
    sample_rate: int,
    *,
    whisper_cli_path: str | Path | None = None,
    model_path: str | Path | None = None,
    language: str | None = None,
    use_gpu: bool = False,
) -> str:
    validated_audio = _validate_audio_buffer(audio)
    validated_sample_rate = _validate_sample_rate(sample_rate)
    whisper_cli, model = _resolve_whisper_runtime(
        whisper_cli_path=whisper_cli_path,
        model_path=model_path,
    )

    prepared_audio = _resample_audio_for_whisper(
        validated_audio,
        source_sample_rate=validated_sample_rate,
    )
    wav_bytes = _serialize_audio_buffer_to_wav_bytes(
        prepared_audio,
        sample_rate=DEFAULT_WHISPER_SAMPLE_RATE,
    )

    with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        return _transcribe_prepared_input(
            whisper_cli=whisper_cli,
            model=model,
            input_source="-",
            transcript_dir=temp_dir,
            language=language,
            use_gpu=use_gpu,
            stdin_bytes=wav_bytes,
        )


def _resolve_whisper_runtime(
    *,
    whisper_cli_path: str | Path | None,
    model_path: str | Path | None,
) -> tuple[Path, Path]:
    whisper_cli = _resolve_existing_file(
        whisper_cli_path or DEFAULT_WHISPER_CPP_BINARY,
        label="whisper.cpp binary",
        install_hint="Run `bash python/run/setup_whisper_cpp.sh` first.",
    )
    model = _resolve_existing_file(
        model_path or DEFAULT_WHISPER_CPP_MODEL,
        label="whisper.cpp model",
        install_hint="Run `bash python/run/setup_whisper_cpp.sh` first.",
    )
    _ensure_portable_whisper_runtime(whisper_cli)
    return whisper_cli, model


def _ensure_portable_whisper_runtime(whisper_cli: Path) -> None:
    if os.name != "posix" or not shutil.which("otool") or not shutil.which("install_name_tool"):
        return
    if not _is_macho_binary(whisper_cli):
        return

    existing_rpaths = _read_macho_rpaths(whisper_cli)
    desired_rpaths = set(DEFAULT_WHISPER_CPP_RPATHS)
    needs_update = any(path not in desired_rpaths for path in existing_rpaths) or any(
        path not in existing_rpaths for path in desired_rpaths
    )
    if not needs_update:
        return

    update_command = [shutil.which("install_name_tool") or "install_name_tool"]
    for existing_rpath in existing_rpaths:
        update_command.extend(["-delete_rpath", existing_rpath])
    for desired_rpath in DEFAULT_WHISPER_CPP_RPATHS:
        update_command.extend(["-add_rpath", desired_rpath])
    update_command.append(str(whisper_cli))
    _run_command(update_command, operation="whisper.cpp runtime relink")


def _read_macho_rpaths(binary_path: Path) -> tuple[str, ...]:
    otool_binary = shutil.which("otool") or "otool"
    completed = subprocess.run(
        [otool_binary, "-l", str(binary_path)],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Failed to inspect whisper.cpp runtime paths."
            if not details
            else f"Failed to inspect whisper.cpp runtime paths. stderr: {details}"
        )

    rpaths: list[str] = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path "):
            continue
        path_value = stripped[5:].split(" (offset ", 1)[0].strip()
        if path_value:
            rpaths.append(path_value)
    return tuple(rpaths)


def _is_macho_binary(binary_path: Path) -> bool:
    try:
        with binary_path.open("rb") as file_handle:
            return file_handle.read(4) in MACHO_MAGIC_PREFIXES
    except OSError:
        return False


def _transcribe_prepared_input(
    *,
    whisper_cli: Path,
    model: Path,
    input_source: str | Path,
    transcript_dir: Path,
    language: str | None,
    use_gpu: bool,
    stdin_bytes: bytes | None = None,
) -> str:
    transcript_prefix = transcript_dir / "transcript"
    transcript_path = transcript_prefix.with_suffix(".txt")

    command = _build_whisper_command(
        whisper_cli=whisper_cli,
        model=model,
        input_source=input_source,
        transcript_prefix=transcript_prefix,
        language=language,
        use_gpu=use_gpu,
    )

    _run_command(
        command,
        operation="whisper.cpp transcription",
        stdin_bytes=stdin_bytes,
    )

    if not transcript_path.exists():
        raise RuntimeError(
            f"whisper.cpp transcription completed but did not produce transcript output: {transcript_path}"
        )

    return transcript_path.read_text(encoding="utf-8").strip()


def _build_whisper_command(
    *,
    whisper_cli: Path,
    model: Path,
    input_source: str | Path,
    transcript_prefix: Path,
    language: str | None,
    use_gpu: bool,
) -> list[str]:
    command = [
        str(whisper_cli),
        "-m",
        str(model),
        "-f",
        str(input_source),
        "-otxt",
        "-of",
        str(transcript_prefix),
        "-np",
    ]
    if not use_gpu:
        command.append("-ng")
    if language and language.strip().lower() != "auto":
        command.extend(["-l", language.strip()])
    elif language and language.strip().lower() == "auto":
        command.extend(["-l", "auto"])
    return command


def _resolve_existing_file(
    path: str | Path,
    *,
    label: str,
    install_hint: str | None = None,
) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        message = f"{label} was not found: {resolved_path}"
        if install_hint:
            message = f"{message}. {install_hint}"
        raise FileNotFoundError(message)
    return resolved_path


def _resolve_ffmpeg_binary(ffmpeg_path: str | Path | None) -> Path:
    if ffmpeg_path is not None:
        return _resolve_existing_file(ffmpeg_path, label="ffmpeg binary")

    discovered = shutil.which("ffmpeg")
    if not discovered:
        raise FileNotFoundError("ffmpeg was not found in PATH.")
    return Path(discovered).resolve()


def _convert_audio_for_whisper(ffmpeg: Path, input_path: Path, output_path: Path) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run_command(command, operation="audio conversion for whisper.cpp")


def _validate_audio_buffer(audio: numpy.ndarray) -> numpy.ndarray:
    if not isinstance(audio, numpy.ndarray):
        raise TypeError("Audio buffer must be a numpy.ndarray.")
    if audio.ndim != 1:
        raise ValueError("Audio buffer must be mono and one-dimensional.")
    if audio.dtype != numpy.float32:
        raise TypeError("Audio buffer must use dtype float32.")
    if audio.size == 0:
        raise ValueError("Audio buffer must not be empty.")
    return audio


def _validate_sample_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, numbers.Integral):
        raise TypeError("Sample rate must be a positive integer.")
    sample_rate_value = int(sample_rate)
    if sample_rate_value <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    return sample_rate_value


def _resample_audio_for_whisper(audio: numpy.ndarray, *, source_sample_rate: int) -> numpy.ndarray:
    if source_sample_rate == DEFAULT_WHISPER_SAMPLE_RATE:
        return audio.astype(numpy.float32, copy=False)

    target_length = max(
        1,
        int(round(audio.shape[0] * DEFAULT_WHISPER_SAMPLE_RATE / float(source_sample_rate))),
    )
    source_positions = numpy.linspace(0.0, audio.shape[0] - 1, num=audio.shape[0], dtype=numpy.float64)
    target_positions = numpy.linspace(0.0, audio.shape[0] - 1, num=target_length, dtype=numpy.float64)
    resampled = numpy.interp(target_positions, source_positions, audio.astype(numpy.float64, copy=False))
    return resampled.astype(numpy.float32)


def _serialize_audio_buffer_to_wav_bytes(audio: numpy.ndarray, *, sample_rate: int) -> bytes:
    clipped = numpy.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(numpy.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _run_command(command: list[str], *, operation: str, stdin_bytes: bytes | None = None) -> None:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        input=stdin_bytes,
    )
    if completed.returncode == 0:
        return

    details: list[str] = [f"{operation} failed with exit code {completed.returncode}."]
    stdout_text = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr_text = completed.stderr.decode("utf-8", errors="replace").strip()
    if stdout_text:
        details.append(f"stdout: {stdout_text}")
    if stderr_text:
        details.append(f"stderr: {stderr_text}")
    raise RuntimeError(" ".join(details))
