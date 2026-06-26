from __future__ import annotations

"""Sequential background worker for completed in-memory utterances."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import queue
import threading
from typing import Generic, TypeVar

import numpy


ProcessedResult = TypeVar("ProcessedResult")
_SENTINEL = object()


@dataclass(frozen=True)
class QueuedUtterance:
    audio: numpy.ndarray
    sample_rate: int
    audio_recorded_at: str | datetime
    utterance_chunk_count: int
    trailing_chunk_count: int
    trailing_pause_ms: int


class InMemoryUtteranceProcessorWorker(Generic[ProcessedResult]):
    def __init__(
        self,
        *,
        process_utterance: Callable[[QueuedUtterance], ProcessedResult],
        on_processed: Callable[[QueuedUtterance, ProcessedResult], None] | None = None,
        on_processing_error: Callable[[QueuedUtterance, Exception], None] | None = None,
    ) -> None:
        self._process_utterance = process_utterance
        self._on_processed = on_processed
        self._on_processing_error = on_processing_error
        self._queue: queue.Queue[QueuedUtterance | object] = queue.Queue()
        self._started = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="in-memory-utterance-worker", daemon=True)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Worker cannot be started after it has been closed.")
        if self._started:
            return
        self._started = True
        self._thread.start()

    def submit(self, utterance: QueuedUtterance) -> None:
        if not self._started:
            raise RuntimeError("Worker must be started before submitting utterances.")
        if self._closed:
            raise RuntimeError("Worker is already closed.")
        self._queue.put(utterance)

    def stop(self, *, process_pending: bool = True) -> None:
        if not self._started or self._closed:
            return

        self._closed = True
        if not process_pending:
            self._discard_pending_utterances()
        self._queue.put(_SENTINEL)
        self._thread.join()

    def _discard_pending_utterances(self) -> None:
        while True:
            try:
                queued_item = self._queue.get_nowait()
            except queue.Empty:
                return
            if queued_item is _SENTINEL:
                self._queue.put(_SENTINEL)
                return

    def _run(self) -> None:
        while True:
            queued_item = self._queue.get()
            if queued_item is _SENTINEL:
                return

            assert isinstance(queued_item, QueuedUtterance)
            try:
                processed = self._process_utterance(queued_item)
            except Exception as exc:
                if self._on_processing_error is not None:
                    self._on_processing_error(queued_item, exc)
                continue

            if self._on_processed is not None:
                self._on_processed(queued_item, processed)
