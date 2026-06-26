from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy


@dataclass(frozen=True)
class FinalizedUtterance:
    audio: numpy.ndarray
    speech_chunk_count: int
    utterance_chunk_count: int
    trailing_chunk_count: int
    trailing_pause_ms: int


@dataclass(frozen=True)
class UtteranceAssemblerResult:
    event: str
    prepended_pre_roll_samples: int = 0
    accumulated_pause_ms: int = 0
    finalized_utterance: FinalizedUtterance | None = None


class SlidingAudioWindow:
    def __init__(self, *, max_samples: int) -> None:
        self._max_samples = max(1, max_samples)
        self._chunks: deque[numpy.ndarray] = deque()
        self._samples = 0

    def push_chunk(self, chunk: numpy.ndarray) -> None:
        self._chunks.append(chunk)
        self._samples += chunk.shape[0]

        while self._chunks and self._samples > self._max_samples:
            overflow = self._samples - self._max_samples
            oldest = self._chunks[0]
            if oldest.shape[0] <= overflow:
                self._chunks.popleft()
                self._samples -= oldest.shape[0]
                continue

            self._chunks[0] = oldest[overflow:]
            self._samples -= overflow
            break

    def as_array(self) -> numpy.ndarray:
        return numpy.concatenate(list(self._chunks)).astype(numpy.float32, copy=False)

    def sample_count(self) -> int:
        return self._samples


class UtteranceAssembler:
    def __init__(
        self,
        *,
        sample_rate: int,
        segment_duration: float,
        pre_roll_ms: int,
        max_pause_ms: int,
        post_roll_ms: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_ms = max(1, int(round(segment_duration * 1000)))
        self.max_pre_roll_samples = max(0, int(round(sample_rate * pre_roll_ms / 1000)))
        self.max_post_roll_samples = max(0, int(round(sample_rate * post_roll_ms / 1000)))

        self._max_pause_ms = max_pause_ms
        self._utterance_chunks: list[numpy.ndarray] = []
        self._speech_chunk_count = 0
        self._pending_silence_chunks: list[numpy.ndarray] = []
        self._pending_silence_ms = 0
        self._pre_roll_chunks: deque[numpy.ndarray] = deque()
        self._pre_roll_samples = 0

    def push_chunk(
        self,
        chunk: numpy.ndarray,
        *,
        has_recent_voice: bool,
    ) -> UtteranceAssemblerResult:
        if has_recent_voice:
            prepended_pre_roll_samples = 0
            if not self._utterance_chunks and self._pre_roll_chunks:
                prepended_pre_roll_samples = self._pre_roll_samples
                self._utterance_chunks.extend(
                    self._trim_chunks_to_tail(self._pre_roll_chunks, self.max_pre_roll_samples)
                )
                self._pre_roll_chunks.clear()
                self._pre_roll_samples = 0

            if self._pending_silence_chunks:
                self._utterance_chunks.extend(self._pending_silence_chunks)
                self._pending_silence_chunks = []

            self._utterance_chunks.append(chunk)
            self._speech_chunk_count += 1
            self._pending_silence_ms = 0
            return UtteranceAssemblerResult(
                event="speech",
                prepended_pre_roll_samples=prepended_pre_roll_samples,
            )

        if self._utterance_chunks:
            self._pending_silence_chunks.append(chunk)
            self._pending_silence_ms += self.chunk_ms
            accumulated_pause_ms = self._pending_silence_ms
            finalized_utterance = None
            if self._pending_silence_ms >= self._max_pause_ms:
                finalized_utterance = self.finalize()
            return UtteranceAssemblerResult(
                event="silence",
                accumulated_pause_ms=accumulated_pause_ms,
                finalized_utterance=finalized_utterance,
            )

        self._push_pre_roll_chunk(chunk)
        return UtteranceAssemblerResult(event="discarded_silence")

    def finalize(self) -> FinalizedUtterance | None:
        if not self._utterance_chunks:
            self._pending_silence_chunks = []
            self._pending_silence_ms = 0
            return None

        trailing_pause_ms = self._pending_silence_ms
        trailing_chunks = self._trim_chunks_to_head(
            self._pending_silence_chunks,
            self.max_post_roll_samples,
        )
        merged = numpy.concatenate(self._utterance_chunks + trailing_chunks).astype(
            numpy.float32,
            copy=False,
        )
        finalized = FinalizedUtterance(
            audio=merged,
            speech_chunk_count=self._speech_chunk_count,
            utterance_chunk_count=len(self._utterance_chunks),
            trailing_chunk_count=len(trailing_chunks),
            trailing_pause_ms=trailing_pause_ms,
        )
        self._utterance_chunks = []
        self._speech_chunk_count = 0
        self._pending_silence_chunks = []
        self._pending_silence_ms = 0
        return finalized

    def has_open_utterance(self) -> bool:
        return bool(self._utterance_chunks)

    def debug_state(self) -> dict[str, int]:
        return {
            "utterance_chunks": len(self._utterance_chunks),
            "speech_chunk_count": self._speech_chunk_count,
            "pending_silence_chunks": len(self._pending_silence_chunks),
            "pending_silence_ms": self._pending_silence_ms,
            "pre_roll_chunks": len(self._pre_roll_chunks),
            "pre_roll_samples": self._pre_roll_samples,
        }

    @staticmethod
    def _trim_chunks_to_tail(
        chunks: list[numpy.ndarray] | deque[numpy.ndarray],
        max_samples: int,
    ) -> list[numpy.ndarray]:
        if max_samples <= 0 or not chunks:
            return []

        remaining = max_samples
        kept: list[numpy.ndarray] = []
        for chunk in reversed(list(chunks)):
            if remaining <= 0:
                break
            if chunk.shape[0] <= remaining:
                kept.append(chunk)
                remaining -= chunk.shape[0]
            else:
                kept.append(chunk[-remaining:])
                remaining = 0
        kept.reverse()
        return kept

    @staticmethod
    def _trim_chunks_to_head(
        chunks: list[numpy.ndarray] | deque[numpy.ndarray],
        max_samples: int,
    ) -> list[numpy.ndarray]:
        if max_samples <= 0 or not chunks:
            return []

        remaining = max_samples
        kept: list[numpy.ndarray] = []
        for chunk in list(chunks):
            if remaining <= 0:
                break
            if chunk.shape[0] <= remaining:
                kept.append(chunk)
                remaining -= chunk.shape[0]
            else:
                kept.append(chunk[:remaining])
                remaining = 0
        return kept

    def _push_pre_roll_chunk(self, chunk: numpy.ndarray) -> None:
        if self.max_pre_roll_samples <= 0:
            self._pre_roll_chunks.clear()
            self._pre_roll_samples = 0
            return

        self._pre_roll_chunks.append(chunk)
        self._pre_roll_samples += chunk.shape[0]

        while self._pre_roll_chunks and self._pre_roll_samples > self.max_pre_roll_samples:
            overflow = self._pre_roll_samples - self.max_pre_roll_samples
            oldest = self._pre_roll_chunks[0]
            if oldest.shape[0] <= overflow:
                self._pre_roll_chunks.popleft()
                self._pre_roll_samples -= oldest.shape[0]
                continue

            self._pre_roll_chunks[0] = oldest[overflow:]
            self._pre_roll_samples -= overflow
            break
