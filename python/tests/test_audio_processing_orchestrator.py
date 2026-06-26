from __future__ import annotations

"""Contract tests for queued audio processing orchestration."""

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

from modules.audio_processing_orchestrator import AudioProcessingFailedError, _build_timings, process_one_ready_audio
from modules.captured_audio_queue import publish_captured_audio
from modules.openai_conversation_reply_generator import OpenAIConversationReplyRecord, OpenAIConversationReplyTimings
from modules.sqlite_conversation_store import create_conversation, get_active_conversation, list_conversation_turns


class AudioProcessingOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.queue_dir = self.root_dir / "voice-queue"
        self.database_path = self.root_dir / "database" / "test.sqlite3"
        self.audio = numpy.array([0.0, 0.25, -0.25, 0.5], dtype=numpy.float32)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_process_one_ready_audio_claims_transcribes_persists_and_marks_done(self) -> None:
        create_conversation("Practice conversation", is_active=True, database_path=self.database_path)
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=1,
            prefix="utterance",
        )

        processed = process_one_ready_audio(
            self.queue_dir,
            database_path=self.database_path,
            transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
            generate_reply_for_text=lambda conversation_id, transcript: f"reply for {transcript}",
        )

        self.assertIsNotNone(processed)
        self.assertEqual("done", processed.final_path.parent.name)
        self.assertEqual(published_path.name, processed.audio_filename)
        self.assertTrue(processed.final_path.exists())
        self.assertFalse((self.queue_dir / "processing" / published_path.name).exists())

        active_conversation = get_active_conversation(database_path=self.database_path)
        self.assertIsNotNone(active_conversation)
        self.assertEqual("Practice conversation", active_conversation.topic_hint)

        turns = list_conversation_turns(active_conversation.conversation_id, database_path=self.database_path)
        self.assertEqual(1, len(turns))
        self.assertEqual(f"transcript for {published_path.name}", turns[0].remote_text)
        self.assertEqual(
            f"reply for transcript for {published_path.name}",
            turns[0].reply_text_suggest,
        )
        self.assertIsNone(processed.timings)

    def test_process_one_ready_audio_uses_conversation_id_embedded_in_filename_instead_of_active_conversation(self) -> None:
        target_conversation = create_conversation("Target conversation", database_path=self.database_path)
        create_conversation("Different active conversation", is_active=True, database_path=self.database_path)
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=target_conversation.conversation_id,
            prefix="utterance",
        )

        processed = process_one_ready_audio(
            self.queue_dir,
            database_path=self.database_path,
            transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
            generate_reply_for_text=lambda conversation_id, transcript: f"reply for {transcript}",
        )

        self.assertIsNotNone(processed)
        self.assertEqual(target_conversation.conversation_id, processed.conversation_id)
        self.assertEqual(
            [published_path.name],
            [
                turn.audio_filename
                for turn in list_conversation_turns(target_conversation.conversation_id, database_path=self.database_path)
            ],
        )

        active_conversation = get_active_conversation(database_path=self.database_path)
        self.assertIsNotNone(active_conversation)
        self.assertNotEqual(target_conversation.conversation_id, active_conversation.conversation_id)

    def test_process_one_ready_audio_fails_when_filename_conversation_id_does_not_exist(self) -> None:
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=999,
            prefix="utterance",
        )

        with self.assertRaises(AudioProcessingFailedError) as error:
            process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                generate_reply_for_text=lambda conversation_id, transcript: f"reply for {transcript}",
            )

        failure = error.exception
        self.assertIsNotNone(failure.failed_path)
        self.assertEqual("failed", failure.failed_path.parent.name)
        self.assertTrue(failure.failed_path.exists())
        self.assertEqual(published_path.name, failure.failed_path.name)
        self.assertIn("foreign key", str(failure).lower())
        self.assertIsNone(get_active_conversation(database_path=self.database_path))

    def test_process_one_ready_audio_fails_when_filename_does_not_embed_conversation_id(self) -> None:
        create_conversation("Practice conversation", is_active=True, database_path=self.database_path)
        queue_root = self.queue_dir
        (queue_root / ".tmp").mkdir(parents=True, exist_ok=True)
        (queue_root / "inbox").mkdir(parents=True, exist_ok=True)
        invalid_path = queue_root / "inbox" / "utterance-20260423-123456-654321.wav"
        invalid_path.write_bytes(b"not-a-real-wav")

        with self.assertRaises(AudioProcessingFailedError) as error:
            process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                generate_reply_for_text=lambda conversation_id, transcript: f"reply for {transcript}",
            )

        failure = error.exception
        self.assertIsNotNone(failure.failed_path)
        self.assertEqual("failed", failure.failed_path.parent.name)
        self.assertTrue(failure.failed_path.exists())
        self.assertEqual(invalid_path.name, failure.failed_path.name)
        self.assertIn("conversation id", str(failure).lower())

    def test_process_one_ready_audio_marks_failed_and_raises_when_transcription_fails(self) -> None:
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=999,
            prefix="utterance",
        )

        with self.assertRaises(AudioProcessingFailedError) as error:
            process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda _audio_path: (_ for _ in ()).throw(RuntimeError("stt failed")),
                generate_reply_for_text=lambda conversation_id, transcript: transcript,
            )

        failure = error.exception
        self.assertIsNotNone(failure.failed_path)
        self.assertEqual("failed", failure.failed_path.parent.name)
        self.assertTrue(failure.failed_path.exists())
        self.assertEqual(published_path.name, failure.failed_path.name)
        self.assertFalse((self.queue_dir / "processing" / published_path.name).exists())
        self.assertIsNone(get_active_conversation(database_path=self.database_path))

    def test_process_one_ready_audio_returns_none_when_queue_is_empty(self) -> None:
        self.assertIsNone(
            process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: audio_path.name,
                generate_reply_for_text=lambda conversation_id, transcript: transcript,
            )
        )

    def test_process_one_ready_audio_collects_timings_only_when_enabled(self) -> None:
        create_conversation("Practice conversation", is_active=True, database_path=self.database_path)
        publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=1,
            prefix="utterance",
        )

        processed = process_one_ready_audio(
            self.queue_dir,
            database_path=self.database_path,
            transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
            generate_reply_for_text=lambda conversation_id, transcript: f"reply for {transcript}",
            measure_timings=True,
        )

        self.assertIsNotNone(processed)
        self.assertIsNotNone(processed.timings)
        assert processed.timings is not None
        self.assertGreaterEqual(processed.timings.queue_lookup_ms, 0)
        self.assertGreaterEqual(processed.timings.transcription_ms, 0)
        self.assertGreaterEqual(processed.timings.openai_full_ms, 0)
        self.assertGreaterEqual(processed.timings.post_openai_local_ms, 0)
        self.assertGreaterEqual(processed.timings.sqlite_ms, 0)
        self.assertGreaterEqual(processed.timings.finalize_ms, 0)
        self.assertGreaterEqual(processed.timings.processing_ms, 0)
        self.assertIsNone(processed.timings.openai_ttft_ms)
        self.assertIsNone(processed.timings.openai_ttfut_ms)
        self.assertIsNone(processed.timings.end_to_end_first_useful_ms)
        self.assertIsNotNone(processed.timings.queue_wait_ms)
        self.assertIsNotNone(processed.timings.end_to_end_full_reply_ms)

    def test_process_one_ready_audio_uses_reply_generator_latency_metadata_for_end_to_end_timings(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=conversation.conversation_id,
            prefix="utterance",
        )

        with patch(
            "modules.audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
                timings=OpenAIConversationReplyTimings(
                    ttft_ms=150,
                    ttfut_ms=240,
                    full_ms=500,
                    cached_tokens=128,
                ),
            ),
        ):
            processed = process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                openai_model="gpt-5-mini",
                measure_timings=True,
            )

        self.assertIsNotNone(processed)
        self.assertIsNotNone(processed.timings)
        assert processed.timings is not None
        self.assertEqual(150, processed.timings.openai_ttft_ms)
        self.assertEqual(240, processed.timings.openai_ttfut_ms)
        self.assertEqual(500, processed.timings.openai_full_ms)
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

    def test_build_timings_records_post_openai_local_segment_consistently_for_queue_path(self) -> None:
        openai_timings = OpenAIConversationReplyTimings(
            ttft_ms=150,
            ttfut_ms=240,
            full_ms=500,
            cached_tokens=128,
        )

        with patch("modules.audio_processing_orchestrator._elapsed_ms", return_value=777):
            timings = _build_timings(
                measure_timings=True,
                queue_lookup_ms=4,
                recorded_at=datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC),
                claimed_at=datetime(2026, 5, 7, 12, 0, 1, 500000, tzinfo=UTC),
                transcription_ms=850,
                openai_reply_timings=openai_timings,
                openai_full_ms=610,
                post_openai_local_ms=19,
                sqlite_ms=8,
                finalize_ms=1,
                processing_started_at_monotonic=1.0,
                finished_at=datetime(2026, 5, 7, 12, 0, 3, tzinfo=UTC),
            )

        self.assertIsNotNone(timings)
        assert timings is not None
        self.assertEqual(4, timings.queue_lookup_ms)
        self.assertEqual(1500, timings.queue_wait_ms)
        self.assertEqual(850, timings.transcription_ms)
        self.assertEqual(150, timings.openai_ttft_ms)
        self.assertEqual(240, timings.openai_ttfut_ms)
        self.assertEqual(500, timings.openai_full_ms)
        self.assertEqual(19, timings.post_openai_local_ms)
        self.assertEqual(8, timings.sqlite_ms)
        self.assertEqual(1, timings.finalize_ms)
        self.assertEqual(777, timings.processing_ms)
        self.assertEqual(2590, timings.end_to_end_first_useful_ms)
        self.assertEqual(2850, timings.end_to_end_full_reply_ms)

    def test_build_timings_defaults_post_openai_local_to_zero_without_openai_metadata(self) -> None:
        with patch("modules.audio_processing_orchestrator._elapsed_ms", return_value=0):
            timings = _build_timings(
                measure_timings=True,
                queue_lookup_ms=None,
                recorded_at=None,
                claimed_at=None,
                transcription_ms=None,
                openai_reply_timings=None,
                openai_full_ms=None,
                post_openai_local_ms=None,
                sqlite_ms=None,
                finalize_ms=None,
                processing_started_at_monotonic=1.0,
                finished_at=None,
            )

        self.assertIsNotNone(timings)
        assert timings is not None
        self.assertEqual(0, timings.queue_lookup_ms)
        self.assertIsNone(timings.queue_wait_ms)
        self.assertEqual(0, timings.transcription_ms)
        self.assertIsNone(timings.openai_ttft_ms)
        self.assertIsNone(timings.openai_ttfut_ms)
        self.assertEqual(0, timings.openai_full_ms)
        self.assertEqual(0, timings.post_openai_local_ms)
        self.assertEqual(0, timings.sqlite_ms)
        self.assertEqual(0, timings.finalize_ms)
        self.assertEqual(0, timings.processing_ms)
        self.assertIsNone(timings.end_to_end_first_useful_ms)
        self.assertIsNone(timings.end_to_end_full_reply_ms)

    def test_process_one_ready_audio_uses_openai_conversation_reply_generator_by_default(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=conversation.conversation_id,
            prefix="utterance",
        )

        with patch(
            "modules.audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
            ),
        ) as generate_reply_mock:
            processed = process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                openai_model="gpt-5-mini",
            )

        self.assertIsNotNone(processed)
        self.assertEqual("reply from conversation", processed.suggested_reply)
        generate_reply_mock.assert_called_once_with(
            conversation.conversation_id,
            f"transcript for {published_path.name}",
            api_key=None,
            model="gpt-5-mini",
            instructions=None,
            dotenv_path=None,
            database_path=self.database_path,
            debug=False,
            fast_mode=False,
            fast_model=None,
            service_tier=None,
            use_conversation=True,
            disable_instructions=False,
            minimal_instructions=None,
            stream=False,
            on_text_delta=None,
        )

    def test_process_one_ready_audio_passes_openai_dotenv_path_to_shared_generator_for_prompt_cache_config(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=conversation.conversation_id,
            prefix="utterance",
        )
        dotenv_path = self.root_dir / ".env.prompt-cache"

        with patch(
            "modules.audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
            ),
        ) as generate_reply_mock:
            process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                openai_dotenv_path=dotenv_path,
            )

        self.assertEqual(dotenv_path, generate_reply_mock.call_args.kwargs["dotenv_path"])

    def test_process_one_ready_audio_passes_openai_debug_and_fast_mode_options(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=conversation.conversation_id,
            prefix="utterance",
        )

        with patch(
            "modules.audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
            ),
        ) as generate_reply_mock:
            processed = process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                openai_model="gpt-5-mini",
                openai_instructions="Keep it short.",
                openai_debug=True,
                openai_fast_mode=True,
                openai_fast_model="gpt-5-nano",
                openai_use_conversation=False,
                openai_disable_instructions=True,
                openai_minimal_instructions="One sentence.",
            )

        self.assertIsNotNone(processed)
        self.assertEqual("reply from conversation", processed.suggested_reply)
        generate_reply_mock.assert_called_once_with(
            conversation.conversation_id,
            f"transcript for {published_path.name}",
            api_key=None,
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
            stream=False,
            on_text_delta=None,
        )

    def test_process_one_ready_audio_passes_openai_service_tier_option(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=conversation.conversation_id,
            prefix="utterance",
        )

        with patch(
            "modules.audio_processing_orchestrator.generate_reply_in_openai_conversation",
            return_value=OpenAIConversationReplyRecord(
                conversation_id=conversation.conversation_id,
                openai_conversation_id="conv_123",
                reply_text="reply from conversation",
                response_id="resp_123",
            ),
        ) as generate_reply_mock:
            processed = process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                openai_service_tier="priority",
            )

        self.assertIsNotNone(processed)
        self.assertEqual("reply from conversation", processed.suggested_reply)
        generate_reply_mock.assert_called_once_with(
            conversation.conversation_id,
            f"transcript for {published_path.name}",
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

    def test_process_one_ready_audio_passes_streaming_callbacks_without_changing_persistence(self) -> None:
        conversation = create_conversation(
            "Practice conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=16000,
            storage_dir=self.queue_dir,
            conversation_id=conversation.conversation_id,
            prefix="utterance",
        )
        started_transcripts: list[str] = []
        streamed_chunks: list[str] = []

        with patch(
            "modules.audio_processing_orchestrator.generate_reply_in_openai_conversation",
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
            processed = process_one_ready_audio(
                self.queue_dir,
                database_path=self.database_path,
                transcribe_ready_audio=lambda audio_path: f"transcript for {audio_path.name}",
                openai_stream=True,
                openai_reply_started_callback=started_transcripts.append,
                openai_text_delta_callback=streamed_chunks.append,
            )

        self.assertIsNotNone(processed)
        self.assertEqual(
            [f"transcript for {published_path.name}"],
            started_transcripts,
        )
        self.assertEqual(["Reply ", "streamed."], streamed_chunks)
        self.assertEqual("Reply streamed.", processed.suggested_reply)
        generate_reply_mock.assert_called_once_with(
            conversation.conversation_id,
            f"transcript for {published_path.name}",
            api_key=None,
            model=None,
            instructions=None,
            dotenv_path=None,
            database_path=self.database_path,
            debug=False,
            fast_mode=False,
            fast_model=None,
            service_tier=None,
            use_conversation=True,
            disable_instructions=False,
            minimal_instructions=None,
            stream=True,
            on_text_delta=streamed_chunks.append,
        )


if __name__ == "__main__":
    unittest.main()
