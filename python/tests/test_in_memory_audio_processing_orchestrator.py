from __future__ import annotations

"""Contract tests for the in-memory utterance processing orchestrator."""

from datetime import UTC, datetime
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.in_memory_audio_processing_orchestrator import (
    InMemoryAudioProcessingError,
    _build_timings,
    process_completed_utterance,
)
from modules.openai_conversation_reply_generator import OpenAIConversationReplyRecord
from modules.openai_conversation_reply_generator import OpenAIConversationReplyTimings
from modules.sqlite_conversation_store import create_conversation, list_conversation_turns


class InMemoryAudioProcessingOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.database_path = self.root_dir / "database" / "test.sqlite3"
        self.audio = numpy.array([0.0, 0.25, -0.25, 0.5], dtype=numpy.float32)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_process_completed_utterance_transcribes_generates_reply_and_persists_turn(self) -> None:
        conversation = create_conversation("Practice conversation", database_path=self.database_path)

        processed = process_completed_utterance(
            conversation_id=conversation.conversation_id,
            audio=self.audio,
            sample_rate=16000,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            database_path=self.database_path,
            transcribe_utterance_audio=lambda audio, sample_rate: f"transcript {sample_rate}Hz {len(audio)} samples",
            generate_reply_for_text=lambda conversation_id, transcript: f"reply for {transcript}",
        )

        self.assertEqual("transcript 16000Hz 4 samples", processed.transcript)
        self.assertEqual("reply for transcript 16000Hz 4 samples", processed.suggested_reply)
        self.assertEqual(conversation.conversation_id, processed.conversation_id)
        self.assertIsNone(processed.audio_filename)
        self.assertEqual("2026-04-24T10:11:12+00:00", processed.audio_recorded_at)
        self.assertIsNone(processed.timings)

        turns = list_conversation_turns(conversation.conversation_id, database_path=self.database_path)
        self.assertEqual(1, len(turns))
        self.assertIsNone(turns[0].audio_filename)
        self.assertEqual("2026-04-24T10:11:12+00:00", turns[0].audio_recorded_at)
        self.assertEqual("transcript 16000Hz 4 samples", turns[0].remote_text)
        self.assertEqual("reply for transcript 16000Hz 4 samples", turns[0].reply_text_suggest)

    def test_process_completed_utterance_passes_openai_options_and_stream_callbacks(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        request_started_calls: list[str] = []
        started_transcripts: list[str] = []
        streamed_chunks: list[str] = []

        with patch(
            "modules.in_memory_audio_processing_orchestrator.generate_reply_in_openai_conversation",
            side_effect=lambda conversation_id, transcript, **kwargs: (
                kwargs["on_text_delta"]("Reply "),
                kwargs["on_text_delta"]("streamed."),
                OpenAIConversationReplyRecord(
                    conversation_id=conversation_id,
                    openai_conversation_id="conv_123",
                    reply_text="Reply streamed.",
                    response_id="resp_stream",
                ),
            )[-1],
        ) as generate_reply_mock:
            processed = process_completed_utterance(
                conversation_id=conversation.conversation_id,
                audio=self.audio,
                sample_rate=16000,
                audio_recorded_at="2026-04-24T10:11:12+00:00",
                database_path=self.database_path,
                transcribe_utterance_audio=lambda audio, sample_rate: "transcript text",
                openai_api_key="test-key",
                openai_model="gpt-5-mini",
                openai_instructions="Keep it short.",
                openai_debug=True,
                openai_fast_mode=True,
                openai_fast_model="gpt-5-nano",
                openai_use_conversation=False,
                openai_disable_instructions=True,
                openai_minimal_instructions="One sentence.",
                openai_stream=True,
                openai_request_started_callback=lambda: request_started_calls.append("started"),
                openai_reply_started_callback=started_transcripts.append,
                openai_text_delta_callback=streamed_chunks.append,
            )

        self.assertEqual("Reply streamed.", processed.suggested_reply)
        self.assertEqual(["started"], request_started_calls)
        self.assertEqual(["transcript text"], started_transcripts)
        self.assertEqual(["Reply ", "streamed."], streamed_chunks)
        generate_reply_mock.assert_called_once_with(
            conversation.conversation_id,
            "transcript text",
            api_key="test-key",
            model="gpt-5-mini",
            instructions="Keep it short.",
            dotenv_path=None,
            database_path=self.database_path,
            debug=True,
            fast_mode=True,
            fast_model="gpt-5-nano",
            service_tier=None,
            use_conversation=False,
            disable_instructions=True,
            minimal_instructions="One sentence.",
            stream=True,
            on_text_delta=streamed_chunks.append,
        )

    def test_process_completed_utterance_passes_openai_service_tier_option(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )

        with patch(
            "modules.in_memory_audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
            ),
        ) as generate_reply_mock:
            processed = process_completed_utterance(
                conversation_id=conversation.conversation_id,
                audio=self.audio,
                sample_rate=16000,
                audio_recorded_at="2026-04-24T10:11:12+00:00",
                database_path=self.database_path,
                transcribe_utterance_audio=lambda audio, sample_rate: "transcript text",
                openai_service_tier="priority",
            )

        self.assertEqual("reply from conversation", processed.suggested_reply)
        generate_reply_mock.assert_called_once_with(
            conversation.conversation_id,
            "transcript text",
            api_key=None,
            model=None,
            instructions=None,
            service_tier="priority",
            dotenv_path=None,
            database_path=self.database_path,
            debug=False,
            fast_mode=False,
            fast_model=None,
            use_conversation=True,
            disable_instructions=False,
            minimal_instructions=None,
            stream=False,
            on_text_delta=None,
        )

    def test_process_completed_utterance_passes_openai_dotenv_path_to_shared_generator_for_prompt_cache_config(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        dotenv_path = self.root_dir / ".env.prompt-cache"

        with patch(
            "modules.in_memory_audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
            ),
        ) as generate_reply_mock:
            process_completed_utterance(
                conversation_id=conversation.conversation_id,
                audio=self.audio,
                sample_rate=16000,
                audio_recorded_at="2026-04-24T10:11:12+00:00",
                database_path=self.database_path,
                transcribe_utterance_audio=lambda audio, sample_rate: "transcript text",
                openai_dotenv_path=dotenv_path,
            )

        self.assertEqual(dotenv_path, generate_reply_mock.call_args.kwargs["dotenv_path"])

    def test_process_completed_utterance_accepts_datetime_audio_recorded_at(self) -> None:
        conversation = create_conversation("Practice conversation", database_path=self.database_path)

        processed = process_completed_utterance(
            conversation_id=conversation.conversation_id,
            audio=self.audio,
            sample_rate=16000,
            audio_recorded_at=datetime(2026, 4, 24, 10, 11, 12, tzinfo=UTC),
            database_path=self.database_path,
            transcribe_utterance_audio=lambda audio, sample_rate: "transcript",
            generate_reply_for_text=lambda conversation_id, transcript: "reply",
        )

        self.assertEqual("2026-04-24T10:11:12+00:00", processed.audio_recorded_at)

    def test_process_completed_utterance_wraps_transcription_failures(self) -> None:
        create_conversation("Practice conversation", database_path=self.database_path)

        with self.assertRaises(InMemoryAudioProcessingError) as error:
            process_completed_utterance(
                conversation_id=1,
                audio=self.audio,
                sample_rate=16000,
                audio_recorded_at="2026-04-24T10:11:12+00:00",
                database_path=self.database_path,
                transcribe_utterance_audio=lambda audio, sample_rate: (_ for _ in ()).throw(RuntimeError("stt failed")),
                generate_reply_for_text=lambda conversation_id, transcript: "reply",
            )

        self.assertEqual("transcription", error.exception.stage)
        self.assertEqual(1, error.exception.conversation_id)
        self.assertIn("stt failed", str(error.exception))

    def test_process_completed_utterance_wraps_openai_failures(self) -> None:
        create_conversation("Practice conversation", database_path=self.database_path)

        with self.assertRaises(InMemoryAudioProcessingError) as error:
            process_completed_utterance(
                conversation_id=1,
                audio=self.audio,
                sample_rate=16000,
                audio_recorded_at="2026-04-24T10:11:12+00:00",
                database_path=self.database_path,
                transcribe_utterance_audio=lambda audio, sample_rate: "transcript",
                generate_reply_for_text=lambda conversation_id, transcript: (_ for _ in ()).throw(
                    RuntimeError("openai failed")
                ),
            )

        self.assertEqual("openai", error.exception.stage)
        self.assertIn("openai failed", str(error.exception))

    def test_process_completed_utterance_wraps_sqlite_failures(self) -> None:
        create_conversation("Practice conversation", database_path=self.database_path)

        with patch(
            "modules.in_memory_audio_processing_orchestrator.add_conversation_turn",
            side_effect=RuntimeError("sqlite failed"),
        ):
            with self.assertRaises(InMemoryAudioProcessingError) as error:
                process_completed_utterance(
                    conversation_id=1,
                    audio=self.audio,
                    sample_rate=16000,
                    audio_recorded_at="2026-04-24T10:11:12+00:00",
                    database_path=self.database_path,
                    transcribe_utterance_audio=lambda audio, sample_rate: "transcript",
                    generate_reply_for_text=lambda conversation_id, transcript: "reply",
                )

        self.assertEqual("sqlite", error.exception.stage)
        self.assertIn("sqlite failed", str(error.exception))

    def test_process_completed_utterance_collects_timings_only_when_enabled(self) -> None:
        conversation = create_conversation("Practice conversation", database_path=self.database_path)

        processed = process_completed_utterance(
            conversation_id=conversation.conversation_id,
            audio=self.audio,
            sample_rate=16000,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            database_path=self.database_path,
            transcribe_utterance_audio=lambda audio, sample_rate: "transcript",
            generate_reply_for_text=lambda conversation_id, transcript: "reply",
            measure_timings=True,
        )

        self.assertIsNotNone(processed.timings)
        assert processed.timings is not None
        self.assertGreaterEqual(processed.timings.transcription_ms, 0)
        self.assertGreaterEqual(processed.timings.openai_full_ms, 0)
        self.assertGreaterEqual(processed.timings.post_openai_local_ms, 0)
        self.assertGreaterEqual(processed.timings.sqlite_ms, 0)
        self.assertGreaterEqual(processed.timings.processing_ms, 0)
        self.assertIsNone(processed.timings.openai_ttft_ms)
        self.assertIsNone(processed.timings.openai_ttfut_ms)
        self.assertIsNone(processed.timings.end_to_end_first_useful_ms)
        self.assertIsNotNone(processed.timings.end_to_end_full_reply_ms)

        processed_without_timings = process_completed_utterance(
            conversation_id=conversation.conversation_id,
            audio=self.audio,
            sample_rate=16000,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            database_path=self.database_path,
            transcribe_utterance_audio=lambda audio, sample_rate: "transcript",
            generate_reply_for_text=lambda conversation_id, transcript: "reply",
            measure_timings=False,
        )

        self.assertIsNone(processed_without_timings.timings)

    def test_build_timings_records_explicit_post_openai_local_segment(self) -> None:
        openai_timings = OpenAIConversationReplyTimings(
            ttft_ms=110,
            ttfut_ms=180,
            full_ms=320,
            cached_tokens=64,
        )

        with patch("modules.in_memory_audio_processing_orchestrator._elapsed_ms", return_value=455):
            timings = _build_timings(
                measure_timings=True,
                transcription_ms=90,
                openai_reply_timings=openai_timings,
                openai_full_ms=333,
                post_openai_local_ms=41,
                sqlite_ms=17,
                processing_started_at_monotonic=1.0,
            )

        self.assertIsNotNone(timings)
        assert timings is not None
        self.assertEqual(90, timings.transcription_ms)
        self.assertEqual(110, timings.openai_ttft_ms)
        self.assertEqual(180, timings.openai_ttfut_ms)
        self.assertEqual(320, timings.openai_full_ms)
        self.assertEqual(41, timings.post_openai_local_ms)
        self.assertEqual(17, timings.sqlite_ms)
        self.assertEqual(455, timings.processing_ms)
        self.assertEqual(270, timings.end_to_end_first_useful_ms)
        self.assertEqual(410, timings.end_to_end_full_reply_ms)

    def test_build_timings_defaults_post_openai_local_to_zero_and_keeps_partial_fields_none(self) -> None:
        with patch("modules.in_memory_audio_processing_orchestrator._elapsed_ms", return_value=0):
            timings = _build_timings(
                measure_timings=True,
                transcription_ms=None,
                openai_reply_timings=None,
                openai_full_ms=None,
                post_openai_local_ms=None,
                sqlite_ms=None,
                processing_started_at_monotonic=1.0,
            )

        self.assertIsNotNone(timings)
        assert timings is not None
        self.assertEqual(0, timings.transcription_ms)
        self.assertIsNone(timings.openai_ttft_ms)
        self.assertIsNone(timings.openai_ttfut_ms)
        self.assertEqual(0, timings.openai_full_ms)
        self.assertEqual(0, timings.post_openai_local_ms)
        self.assertEqual(0, timings.sqlite_ms)
        self.assertEqual(0, timings.processing_ms)
        self.assertIsNone(timings.end_to_end_first_useful_ms)
        self.assertEqual(0, timings.end_to_end_full_reply_ms)

    def test_process_completed_utterance_uses_reply_generator_latency_metadata_for_end_to_end_timings(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )

        with patch(
            "modules.in_memory_audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
                timings=OpenAIConversationReplyTimings(
                    ttft_ms=110,
                    ttfut_ms=180,
                    full_ms=320,
                    cached_tokens=64,
                ),
            ),
        ):
            processed = process_completed_utterance(
                conversation_id=conversation.conversation_id,
                audio=self.audio,
                sample_rate=16000,
                audio_recorded_at="2026-04-24T10:11:12+00:00",
                database_path=self.database_path,
                transcribe_utterance_audio=lambda audio, sample_rate: "transcript text",
                measure_timings=True,
            )

        self.assertIsNotNone(processed.timings)
        assert processed.timings is not None
        self.assertEqual(110, processed.timings.openai_ttft_ms)
        self.assertEqual(180, processed.timings.openai_ttfut_ms)
        self.assertEqual(320, processed.timings.openai_full_ms)
        self.assertIsNotNone(processed.timings.end_to_end_first_useful_ms)
        self.assertIsNotNone(processed.timings.end_to_end_full_reply_ms)
        assert processed.timings.end_to_end_first_useful_ms is not None
        assert processed.timings.end_to_end_full_reply_ms is not None
        self.assertGreaterEqual(
            processed.timings.end_to_end_first_useful_ms,
            processed.timings.openai_ttfut_ms or 0,
        )
        self.assertGreaterEqual(
            processed.timings.end_to_end_full_reply_ms,
            processed.timings.openai_full_ms,
        )


if __name__ == "__main__":
    unittest.main()
