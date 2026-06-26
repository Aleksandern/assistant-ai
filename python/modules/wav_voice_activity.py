from __future__ import annotations

from dataclasses import dataclass
import wave
from pathlib import Path

import numpy
import torch
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad


_MODEL = None


def get_vad_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_silero_vad()
    return _MODEL


@dataclass(frozen=True)
class StreamingVoiceActivityResult:
    contains_speech: bool
    speech_started: bool
    speech_ended: bool
    speech_active: bool
    processed_samples: int


class StreamingVoiceActivityDetector:
    """Streaming wrapper around Silero's VADIterator for arbitrary chunk sizes."""

    def __init__(
        self,
        *,
        sample_rate: int,
        threshold: float,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("StreamingVoiceActivityDetector supports only 8000 Hz and 16000 Hz audio.")

        self._sample_rate = sample_rate
        self._frame_size = 256 if sample_rate == 8000 else 512
        self._iterator = VADIterator(
            get_vad_model(),
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._pending_samples = numpy.array([], dtype=numpy.float32)

    def process_chunk(self, chunk: numpy.ndarray | torch.Tensor) -> StreamingVoiceActivityResult:
        chunk_array = self._to_numpy_float32(chunk)
        contains_speech = self._iterator.triggered
        speech_started = False
        speech_ended = False
        processed_samples = 0

        if chunk_array.size:
            self._pending_samples = numpy.concatenate([self._pending_samples, chunk_array]).astype(
                numpy.float32,
                copy=False,
            )

        while self._pending_samples.shape[0] >= self._frame_size:
            frame = self._pending_samples[: self._frame_size]
            self._pending_samples = self._pending_samples[self._frame_size :]
            processed_samples += frame.shape[0]

            was_triggered = self._iterator.triggered
            event = self._iterator(torch.from_numpy(frame))
            is_triggered = self._iterator.triggered

            if was_triggered or is_triggered:
                contains_speech = True

            if event is None:
                continue

            if "start" in event:
                speech_started = True
                contains_speech = True
            if "end" in event:
                speech_ended = True

        return StreamingVoiceActivityResult(
            contains_speech=contains_speech,
            speech_started=speech_started,
            speech_ended=speech_ended,
            speech_active=self._iterator.triggered,
            processed_samples=processed_samples,
        )

    def reset(self) -> None:
        self._pending_samples = numpy.array([], dtype=numpy.float32)
        self._iterator.reset_states()

    def pending_sample_count(self) -> int:
        return int(self._pending_samples.shape[0])

    @staticmethod
    def _to_numpy_float32(chunk: numpy.ndarray | torch.Tensor) -> numpy.ndarray:
        if isinstance(chunk, numpy.ndarray):
            return chunk.astype(numpy.float32, copy=False)
        if torch.is_tensor(chunk):
            return chunk.detach().cpu().numpy().astype(numpy.float32, copy=False)
        return numpy.asarray(chunk, dtype=numpy.float32)


def get_speech_segments(
    audio: torch.Tensor,
    *,
    sampling_rate: int,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
):
    model = get_vad_model()
    return get_speech_timestamps(
        audio,
        model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )


def read_wav_as_tensor(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        if sample_width != 2:
            raise RuntimeError(f"Expected 16-bit PCM WAV, got sample width {sample_width}.")
        if channels != 1:
            raise RuntimeError(f"Expected mono WAV, got {channels} channel(s).")

        pcm_bytes = wav_file.readframes(wav_file.getnframes())
        samples = numpy.frombuffer(pcm_bytes, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        return torch.from_numpy(samples), frame_rate


def analyze_wav(
    path: Path,
    *,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> tuple[bool, int, float]:
    audio, sampling_rate = read_wav_as_tensor(path)
    speech_timestamps = get_speech_segments(
        audio,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    speech_samples = sum(segment["end"] - segment["start"] for segment in speech_timestamps)
    speech_ms = int(round((speech_samples / sampling_rate) * 1000)) if sampling_rate else 0
    total_ms = int(round((audio.numel() / sampling_rate) * 1000)) if sampling_rate else 0
    speech_ratio = (speech_ms / total_ms) if total_ms else 0.0
    return bool(speech_timestamps), speech_ms, speech_ratio


def analyze_audio(
    audio: torch.Tensor,
    *,
    sampling_rate: int,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
) -> tuple[bool, int, float]:
    speech_timestamps = get_speech_segments(
        audio,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    speech_samples = sum(segment["end"] - segment["start"] for segment in speech_timestamps)
    speech_ms = int(round((speech_samples / sampling_rate) * 1000)) if sampling_rate else 0
    total_ms = int(round((audio.numel() / sampling_rate) * 1000)) if sampling_rate else 0
    speech_ratio = (speech_ms / total_ms) if total_ms else 0.0
    return bool(speech_timestamps), speech_ms, speech_ratio
