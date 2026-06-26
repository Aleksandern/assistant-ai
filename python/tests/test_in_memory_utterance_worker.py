from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.in_memory_utterance_worker import InMemoryUtteranceProcessorWorker, QueuedUtterance


class InMemoryUtteranceProcessorWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio = numpy.array([0.1, -0.1], dtype=numpy.float32)

    def test_worker_processes_utterances_in_submission_order(self) -> None:
        processed_order: list[str] = []
        completed_order: list[str] = []

        worker = InMemoryUtteranceProcessorWorker(
            process_utterance=lambda utterance: processed_order.append(str(utterance.audio_recorded_at)) or str(utterance.audio_recorded_at),
            on_processed=lambda utterance, result: completed_order.append(f"{utterance.audio_recorded_at}:{result}"),
        )

        worker.start()
        worker.submit(self._queued_utterance("first"))
        worker.submit(self._queued_utterance("second"))
        worker.stop(process_pending=True)

        self.assertEqual(["first", "second"], processed_order)
        self.assertEqual(["first:first", "second:second"], completed_order)

    def test_worker_reports_processing_errors_and_continues(self) -> None:
        processed_order: list[str] = []
        failures: list[str] = []

        def process_utterance(utterance: QueuedUtterance) -> str:
            processed_order.append(str(utterance.audio_recorded_at))
            if utterance.audio_recorded_at == "first":
                raise RuntimeError("boom")
            return "ok"

        worker = InMemoryUtteranceProcessorWorker(
            process_utterance=process_utterance,
            on_processing_error=lambda utterance, exc: failures.append(f"{utterance.audio_recorded_at}:{exc}"),
        )

        worker.start()
        worker.submit(self._queued_utterance("first"))
        worker.submit(self._queued_utterance("second"))
        worker.stop(process_pending=True)

        self.assertEqual(["first", "second"], processed_order)
        self.assertEqual(["first:boom"], failures)

    def test_worker_can_drop_pending_utterances_on_shutdown(self) -> None:
        processed_order: list[str] = []

        worker = InMemoryUtteranceProcessorWorker(
            process_utterance=lambda utterance: processed_order.append(str(utterance.audio_recorded_at)),
        )

        worker.start()
        worker.submit(self._queued_utterance("first"))
        worker.submit(self._queued_utterance("second"))
        worker.stop(process_pending=False)

        self.assertLessEqual(len(processed_order), 1)
        self.assertNotIn("second", processed_order)

    def test_worker_rejects_submit_after_shutdown(self) -> None:
        worker = InMemoryUtteranceProcessorWorker(process_utterance=lambda utterance: None)

        worker.start()
        worker.stop(process_pending=True)

        with self.assertRaises(RuntimeError):
            worker.submit(self._queued_utterance("late"))

    def _queued_utterance(self, audio_recorded_at: str) -> QueuedUtterance:
        return QueuedUtterance(
            audio=self.audio,
            sample_rate=16000,
            audio_recorded_at=audio_recorded_at,
            utterance_chunk_count=1,
            trailing_chunk_count=0,
            trailing_pause_ms=0,
        )


if __name__ == "__main__":
    unittest.main()
