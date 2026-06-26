from __future__ import annotations

"""Filesystem queue for captured audio utterance WAV files.

This module owns the storage lifecycle for completed audio artifacts:
- publish finished audio into `inbox`
- claim the oldest ready file into `processing`
- mark processed files as `done` or `failed`

It is intended to sit between the voice capture runner and downstream consumers
such as transcription or other processing modules.
"""

import time
from datetime import UTC, datetime
from pathlib import Path
import re

import numpy

from modules.wav_file_writer import write_mono_audio_to_wav


QUEUE_TMP_DIRNAME = ".tmp"
QUEUE_INBOX_DIRNAME = "inbox"
QUEUE_PROCESSING_DIRNAME = "processing"
QUEUE_DONE_DIRNAME = "done"
QUEUE_FAILED_DIRNAME = "failed"
CAPTURED_AUDIO_CONVERSATION_ID_PATTERN = re.compile(r"^(?P<prefix>.+)-conv-(?P<conversation_id>\d+)-\d{8}-\d{6}-\d{6}\.wav$")


def ensure_captured_audio_queue_dirs(storage_dir: str | Path) -> Path:
    root_dir = Path(storage_dir)
    for dirname in (
        QUEUE_TMP_DIRNAME,
        QUEUE_INBOX_DIRNAME,
        QUEUE_PROCESSING_DIRNAME,
        QUEUE_DONE_DIRNAME,
        QUEUE_FAILED_DIRNAME,
    ):
        (root_dir / dirname).mkdir(parents=True, exist_ok=True)
    return root_dir


def make_captured_audio_filename(prefix: str, conversation_id: int) -> str:
    time_ns = time.time_ns()
    timestamp = datetime.fromtimestamp(time_ns / 1_000_000_000, tz=UTC).strftime("%Y%m%d-%H%M%S")
    micros = (time_ns // 1_000) % 1_000_000
    return f"{prefix}-conv-{conversation_id}-{timestamp}-{micros:06d}.wav"


def parse_conversation_id_from_captured_audio_filename(audio_filename: str | Path) -> int:
    filename = Path(audio_filename).name
    match = CAPTURED_AUDIO_CONVERSATION_ID_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(
            "Captured audio filename must contain a conversation id like `prefix-conv-123-YYYYMMDD-HHMMSS-micros.wav`."
        )

    return int(match.group("conversation_id"))


def publish_captured_audio(
    audio: numpy.ndarray,
    sample_rate: int,
    storage_dir: str | Path,
    conversation_id: int,
    prefix: str = "utterance",
) -> Path:
    root_dir = ensure_captured_audio_queue_dirs(storage_dir)
    filename = make_captured_audio_filename(prefix, conversation_id)
    final_path = root_dir / QUEUE_INBOX_DIRNAME / filename
    tmp_path = root_dir / QUEUE_TMP_DIRNAME / filename.replace(".wav", ".partial.wav")

    write_mono_audio_to_wav(audio, tmp_path, sample_rate)
    tmp_path.replace(final_path)
    return final_path


def claim_oldest_ready_audio(storage_dir: str | Path) -> Path | None:
    root_dir = ensure_captured_audio_queue_dirs(storage_dir)
    inbox_dir = root_dir / QUEUE_INBOX_DIRNAME
    for ready_path in sorted(inbox_dir.glob("*.wav")):
        claimed_path = root_dir / QUEUE_PROCESSING_DIRNAME / ready_path.name
        try:
            ready_path.replace(claimed_path)
        except FileNotFoundError:
            # Another worker may have claimed the file after directory listing but before our move.
            continue
        return claimed_path
    return None


def mark_audio_processing_succeeded(processing_path: str | Path) -> Path:
    path = _validate_processing_path(processing_path)
    done_path = path.parent.parent / QUEUE_DONE_DIRNAME / path.name
    path.replace(done_path)
    return done_path


def mark_audio_processing_failed(processing_path: str | Path) -> Path:
    path = _validate_processing_path(processing_path)
    failed_path = path.parent.parent / QUEUE_FAILED_DIRNAME / path.name
    path.replace(failed_path)
    return failed_path


def _validate_processing_path(processing_path: str | Path) -> Path:
    path = Path(processing_path)
    if path.parent.name != QUEUE_PROCESSING_DIRNAME:
        raise ValueError("Expected a file path inside the processing directory.")
    if not path.exists():
        raise ValueError(f"Processing file does not exist: {path}")
    return path
