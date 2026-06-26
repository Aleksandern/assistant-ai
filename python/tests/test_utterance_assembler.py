from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.utterance_assembler import SlidingAudioWindow, UtteranceAssembler


def make_chunk(value: float, samples: int = 200) -> numpy.ndarray:
    return numpy.full(samples, value, dtype=numpy.float32)


class UtteranceAssemblerTests(unittest.TestCase):
    def build_assembler(
        self,
        *,
        pre_roll_ms: int = 400,
        max_pause_ms: int = 800,
        post_roll_ms: int = 300,
        segment_duration: float = 0.2,
    ) -> UtteranceAssembler:
        return UtteranceAssembler(
            sample_rate=1000,
            segment_duration=segment_duration,
            pre_roll_ms=pre_roll_ms,
            max_pause_ms=max_pause_ms,
            post_roll_ms=post_roll_ms,
        )

    def test_speech_starts_after_silent_chunks_with_pre_roll(self) -> None:
        assembler = self.build_assembler(pre_roll_ms=400, post_roll_ms=0)
        silent_1 = make_chunk(0.1)
        silent_2 = make_chunk(0.2)
        silent_3 = make_chunk(0.3)
        speech = make_chunk(1.0)

        assembler.push_chunk(silent_1, has_recent_voice=False)
        assembler.push_chunk(silent_2, has_recent_voice=False)
        assembler.push_chunk(silent_3, has_recent_voice=False)
        result = assembler.push_chunk(speech, has_recent_voice=True)
        finalized = assembler.finalize()

        self.assertEqual("speech", result.event)
        self.assertEqual(400, result.prepended_pre_roll_samples)
        self.assertIsNotNone(finalized)
        self.assertEqual(1, finalized.speech_chunk_count)
        numpy.testing.assert_array_equal(
            numpy.concatenate([silent_2, silent_3, speech]).astype(numpy.float32),
            finalized.audio,
        )

    def test_short_pause_does_not_close_utterance(self) -> None:
        assembler = self.build_assembler(post_roll_ms=0)
        speech_1 = make_chunk(1.0)
        silence = make_chunk(0.0)
        speech_2 = make_chunk(2.0)

        assembler.push_chunk(speech_1, has_recent_voice=True)
        silence_result = assembler.push_chunk(silence, has_recent_voice=False)
        assembler.push_chunk(speech_2, has_recent_voice=True)
        finalized = assembler.finalize()

        self.assertEqual("silence", silence_result.event)
        self.assertIsNone(silence_result.finalized_utterance)
        self.assertEqual(200, silence_result.accumulated_pause_ms)
        self.assertIsNotNone(finalized)
        self.assertEqual(2, finalized.speech_chunk_count)
        numpy.testing.assert_array_equal(
            numpy.concatenate([speech_1, silence, speech_2]).astype(numpy.float32),
            finalized.audio,
        )

    def test_long_pause_closes_utterance(self) -> None:
        assembler = self.build_assembler(post_roll_ms=300)
        speech = make_chunk(1.0)
        silences = [make_chunk(float(index)) for index in range(4)]

        assembler.push_chunk(speech, has_recent_voice=True)
        results = [assembler.push_chunk(chunk, has_recent_voice=False) for chunk in silences]

        finalized = results[-1].finalized_utterance

        self.assertIsNotNone(finalized)
        self.assertEqual(1, finalized.speech_chunk_count)
        self.assertEqual(800, finalized.trailing_pause_ms)
        self.assertFalse(assembler.has_open_utterance())

    def test_post_roll_is_limited_to_configured_size(self) -> None:
        assembler = self.build_assembler(post_roll_ms=300)
        speech = make_chunk(1.0)
        silence_1 = make_chunk(0.0)
        silence_2 = make_chunk(0.5)
        silence_3 = make_chunk(0.75)
        silence_4 = make_chunk(0.9)

        assembler.push_chunk(speech, has_recent_voice=True)
        assembler.push_chunk(silence_1, has_recent_voice=False)
        assembler.push_chunk(silence_2, has_recent_voice=False)
        assembler.push_chunk(silence_3, has_recent_voice=False)
        result = assembler.push_chunk(silence_4, has_recent_voice=False)
        finalized = result.finalized_utterance

        self.assertIsNotNone(finalized)
        self.assertEqual(1, finalized.speech_chunk_count)
        expected = numpy.concatenate([speech, silence_1, silence_2[:100]]).astype(numpy.float32)
        numpy.testing.assert_array_equal(expected, finalized.audio)

    def test_silent_chunks_before_start_do_not_form_utterance(self) -> None:
        assembler = self.build_assembler()

        result = assembler.push_chunk(make_chunk(0.0), has_recent_voice=False)

        self.assertEqual("discarded_silence", result.event)
        self.assertFalse(assembler.has_open_utterance())
        self.assertIsNone(assembler.finalize())

    def test_state_resets_after_finalize(self) -> None:
        assembler = self.build_assembler(post_roll_ms=0, max_pause_ms=200)
        first_speech = make_chunk(1.0)
        second_speech = make_chunk(2.0)
        silence = make_chunk(0.0)

        assembler.push_chunk(first_speech, has_recent_voice=True)
        first_result = assembler.push_chunk(silence, has_recent_voice=False)
        assembler.push_chunk(second_speech, has_recent_voice=True)
        second_finalized = assembler.finalize()

        self.assertIsNotNone(first_result.finalized_utterance)
        self.assertEqual(1, first_result.finalized_utterance.speech_chunk_count)
        numpy.testing.assert_array_equal(first_speech, first_result.finalized_utterance.audio)
        self.assertIsNotNone(second_finalized)
        self.assertEqual(1, second_finalized.speech_chunk_count)
        numpy.testing.assert_array_equal(second_speech, second_finalized.audio)

    def test_zero_length_chunk_is_accepted(self) -> None:
        assembler = self.build_assembler()
        empty_chunk = numpy.array([], dtype=numpy.float32)

        result = assembler.push_chunk(empty_chunk, has_recent_voice=False)

        self.assertEqual("discarded_silence", result.event)
        self.assertIsNone(assembler.finalize())

    def test_very_short_chunk_is_preserved(self) -> None:
        assembler = self.build_assembler(post_roll_ms=0)
        short_chunk = numpy.array([0.25, -0.25, 0.1], dtype=numpy.float32)

        assembler.push_chunk(short_chunk, has_recent_voice=True)
        finalized = assembler.finalize()

        self.assertIsNotNone(finalized)
        self.assertEqual(1, finalized.speech_chunk_count)
        numpy.testing.assert_array_equal(short_chunk, finalized.audio)

    def test_pre_roll_zero_discards_leading_silence(self) -> None:
        assembler = self.build_assembler(pre_roll_ms=0, post_roll_ms=0)
        silence = make_chunk(0.0)
        speech = make_chunk(1.0)

        assembler.push_chunk(silence, has_recent_voice=False)
        assembler.push_chunk(speech, has_recent_voice=True)
        finalized = assembler.finalize()

        self.assertIsNotNone(finalized)
        self.assertEqual(1, finalized.speech_chunk_count)
        numpy.testing.assert_array_equal(speech, finalized.audio)

    def test_post_roll_zero_excludes_trailing_silence(self) -> None:
        assembler = self.build_assembler(post_roll_ms=0, max_pause_ms=200)
        speech = make_chunk(1.0)
        silence = make_chunk(0.0)

        assembler.push_chunk(speech, has_recent_voice=True)
        result = assembler.push_chunk(silence, has_recent_voice=False)

        self.assertIsNotNone(result.finalized_utterance)
        self.assertEqual(1, result.finalized_utterance.speech_chunk_count)
        numpy.testing.assert_array_equal(speech, result.finalized_utterance.audio)

    def test_max_pause_zero_finalizes_on_first_silent_chunk(self) -> None:
        assembler = self.build_assembler(post_roll_ms=0, max_pause_ms=0)
        speech = make_chunk(1.0)
        silence = make_chunk(0.0)

        assembler.push_chunk(speech, has_recent_voice=True)
        result = assembler.push_chunk(silence, has_recent_voice=False)

        self.assertEqual("silence", result.event)
        self.assertIsNotNone(result.finalized_utterance)
        self.assertEqual(1, result.finalized_utterance.speech_chunk_count)
        numpy.testing.assert_array_equal(speech, result.finalized_utterance.audio)


class SlidingAudioWindowTests(unittest.TestCase):
    def test_push_chunk_keeps_only_recent_tail_with_sample_limit(self) -> None:
        window = SlidingAudioWindow(max_samples=4)

        window.push_chunk(numpy.array([1.0, 2.0], dtype=numpy.float32))
        window.push_chunk(numpy.array([3.0, 4.0, 5.0], dtype=numpy.float32))

        self.assertEqual(4, window.sample_count())
        numpy.testing.assert_array_equal(
            numpy.array([2.0, 3.0, 4.0, 5.0], dtype=numpy.float32),
            window.as_array(),
        )

    def test_push_chunk_handles_window_smaller_than_single_chunk(self) -> None:
        window = SlidingAudioWindow(max_samples=3)

        window.push_chunk(numpy.array([1.0, 2.0, 3.0, 4.0], dtype=numpy.float32))

        self.assertEqual(3, window.sample_count())
        numpy.testing.assert_array_equal(
            numpy.array([2.0, 3.0, 4.0], dtype=numpy.float32),
            window.as_array(),
        )

    def test_as_array_returns_float32_audio(self) -> None:
        window = SlidingAudioWindow(max_samples=10)

        window.push_chunk(numpy.array([1, 2, 3], dtype=numpy.int16))

        self.assertEqual(numpy.float32, window.as_array().dtype)


if __name__ == "__main__":
    unittest.main()
