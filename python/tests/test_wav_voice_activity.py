from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.wav_voice_activity import StreamingVoiceActivityDetector


class FakeVADIterator:
    def __init__(self, _model, *, threshold, sampling_rate, min_silence_duration_ms, speech_pad_ms):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.triggered = False
        self.frames: list[numpy.ndarray] = []
        self.events = [
            {"start": 0},
            None,
            {"end": 0},
        ]
        self.reset_called = False

    def __call__(self, frame):
        frame_array = frame.detach().cpu().numpy()
        self.frames.append(frame_array)
        event = self.events.pop(0) if self.events else None
        if event is not None and "start" in event:
            self.triggered = True
        elif event is not None and "end" in event:
            self.triggered = False
        return event

    def reset_states(self):
        self.reset_called = True
        self.triggered = False


class StreamingVoiceActivityDetectorTests(unittest.TestCase):
    def test_process_chunk_buffers_until_full_silero_frames(self) -> None:
        fake_iterator = FakeVADIterator(
            object(),
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=800,
            speech_pad_ms=30,
        )

        with (
            patch("modules.wav_voice_activity.get_vad_model", return_value=object()),
            patch("modules.wav_voice_activity.VADIterator", return_value=fake_iterator),
        ):
            detector = StreamingVoiceActivityDetector(
                sample_rate=16000,
                threshold=0.5,
                min_silence_duration_ms=800,
                speech_pad_ms=30,
            )

            first = detector.process_chunk(numpy.ones(700, dtype=numpy.float32))
            first_pending_samples = detector.pending_sample_count()
            second = detector.process_chunk(numpy.ones(700, dtype=numpy.float32))
            second_pending_samples = detector.pending_sample_count()
            third = detector.process_chunk(numpy.ones(200, dtype=numpy.float32))
            third_pending_samples = detector.pending_sample_count()

        self.assertTrue(first.contains_speech)
        self.assertTrue(first.speech_started)
        self.assertEqual(512, first.processed_samples)
        self.assertEqual(188, first_pending_samples)

        self.assertTrue(second.contains_speech)
        self.assertFalse(second.speech_started)
        self.assertFalse(second.speech_ended)
        self.assertTrue(second.speech_active)
        self.assertEqual(512, second.processed_samples)
        self.assertEqual(376, second_pending_samples)

        self.assertTrue(third.contains_speech)
        self.assertTrue(third.speech_ended)
        self.assertFalse(third.speech_active)
        self.assertEqual(512, third.processed_samples)
        self.assertEqual(64, third_pending_samples)

        self.assertEqual(3, len(fake_iterator.frames))
        self.assertEqual([512, 512, 512], [frame.shape[0] for frame in fake_iterator.frames])

    def test_reset_clears_buffer_and_iterator_state(self) -> None:
        fake_iterator = FakeVADIterator(
            object(),
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=800,
            speech_pad_ms=30,
        )

        with (
            patch("modules.wav_voice_activity.get_vad_model", return_value=object()),
            patch("modules.wav_voice_activity.VADIterator", return_value=fake_iterator),
        ):
            detector = StreamingVoiceActivityDetector(
                sample_rate=16000,
                threshold=0.5,
                min_silence_duration_ms=800,
                speech_pad_ms=30,
            )
            detector.process_chunk(numpy.ones(300, dtype=numpy.float32))
            detector.reset()

        self.assertEqual(0, detector.pending_sample_count())
        self.assertTrue(fake_iterator.reset_called)


if __name__ == "__main__":
    unittest.main()
