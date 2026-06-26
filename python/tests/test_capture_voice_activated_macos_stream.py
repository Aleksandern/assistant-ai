from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.in_memory_audio_processing_orchestrator import InMemoryAudioProcessingTimings, ProcessedUtteranceRecord
from modules.openai_conversation_initializer import InitializedConversationRecord
from run.capture_voice_activated_macos_stream import main


class FakeStreamingVoiceActivityDetector:
    def __init__(self, results: list[object] | None = None) -> None:
        self._results = list(results or [])

    def process_chunk(self, _chunk):
        if self._results:
            return self._results.pop(0)
        return _vad_result(False)

    def pending_sample_count(self) -> int:
        return 0


def _vad_result(
    contains_speech: bool,
    *,
    speech_started: bool = False,
    speech_ended: bool = False,
    speech_active: bool | None = None,
    processed_samples: int = 0,
):
    return type(
        "FakeStreamingVoiceActivityResult",
        (),
        {
            "contains_speech": contains_speech,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "speech_active": contains_speech if speech_active is None else speech_active,
            "processed_samples": processed_samples,
        },
    )()


class CaptureVoiceActivatedMacosStreamRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[str] = []

    def test_main_lists_targets_without_startup_side_effects(self) -> None:
        stdout = io.StringIO()

        with (
            patch("run.capture_voice_activated_macos_stream.list_targets", side_effect=self._record_list_targets),
            patch("run.capture_voice_activated_macos_stream.initialize_openai_conversation") as initialize_mock,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--list-targets"])

        self.assertEqual(0, exit_code)
        self.assertEqual(["list_targets"], self.events)
        initialize_mock.assert_not_called()

    def test_main_initializes_conversation_before_listener_starts(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        initialized_calls: list[dict[str, object]] = []

        def record_initialize(topic_hint: str, **kwargs):
            self.events.append("initialize_openai_conversation")
            initialized_calls.append({"topic_hint": topic_hint, **kwargs})
            return InitializedConversationRecord(
                conversation_id=1,
                openai_conversation_id="conv_123",
                topic_hint="",
                file_name="file.docx",
            )

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch("run.capture_voice_activated_macos_stream.initialize_openai_conversation", side_effect=record_initialize),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--openai-api-key",
                    "test-key",
                    "--openai-model",
                    "gpt-5-mini",
                    "--openai-instructions",
                    "Keep it short.",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [
                "resolve_target",
                "initialize_openai_conversation",
                "listener.start",
                "listener.read_chunk",
                "listener.stop",
            ],
            self.events,
        )
        self.assertEqual("", initialized_calls[0]["topic_hint"])
        self.assertEqual("test-key", initialized_calls[0]["api_key"])
        self.assertEqual("gpt-5-mini", initialized_calls[0]["model"])
        self.assertEqual("Keep it short.", initialized_calls[0]["instructions"])
        self.assertIn("Conversation initialized:", stdout.getvalue())
        self.assertIn("Voice-activated stream capture started.", stdout.getvalue())

    def test_main_reuses_existing_conversation_when_conv_id_is_provided(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        orchestrator_calls: list[dict[str, object]] = []

        def record_process_completed_utterance(**kwargs):
            orchestrator_calls.append(kwargs)
            return self._processed_record()

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch("run.capture_voice_activated_macos_stream.get_conversation", side_effect=self._record_existing_conversation),
            patch("run.capture_voice_activated_macos_stream.initialize_openai_conversation") as initialize_mock,
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--conv-id",
                    "7",
                    "--max-pause-ms",
                    "200",
                    "--segment-duration",
                    "0.2",
                    "--min-speech-ms",
                    "0",
                    "--stream-openai=false",
                ]
            )

        self.assertEqual(0, exit_code)
        initialize_mock.assert_not_called()
        self.assertEqual(
            [
                "resolve_target",
                "get_conversation",
                "listener.start",
                "listener.read_chunk",
                "listener.read_chunk",
                "listener.read_chunk",
                "listener.stop",
            ],
            self.events,
        )
        self.assertEqual(1, len(orchestrator_calls))
        self.assertEqual(7, orchestrator_calls[0]["conversation_id"])
        self.assertIn("Conversation reused:", stdout.getvalue())
        self.assertIn("conversation_id=7", stdout.getvalue())

    def test_main_fails_fast_when_conv_id_is_missing(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch("run.capture_voice_activated_macos_stream.get_conversation", side_effect=self._record_missing_conversation),
            patch("run.capture_voice_activated_macos_stream.initialize_openai_conversation") as initialize_mock,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--conv-id", "999"])

        self.assertEqual(1, exit_code)
        initialize_mock.assert_not_called()
        self.assertEqual(["resolve_target", "get_conversation"], self.events)
        self.assertIn("Voice-activated stream capture failed: Conversation with id=999 was not found.", stdout.getvalue())

    def test_main_fails_fast_when_conv_id_has_no_openai_conversation_id(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.get_conversation",
                side_effect=self._record_existing_conversation_without_openai_id,
            ),
            patch("run.capture_voice_activated_macos_stream.initialize_openai_conversation") as initialize_mock,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--conv-id", "8"])

        self.assertEqual(1, exit_code)
        initialize_mock.assert_not_called()
        self.assertEqual(["resolve_target", "get_conversation"], self.events)
        self.assertIn(
            "Voice-activated stream capture failed: Conversation with id=8 does not have openai_conversation_id and cannot be reused.",
            stdout.getvalue(),
        )

    def test_main_sends_finalized_utterance_to_in_memory_orchestrator(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        orchestrator_calls: list[dict[str, object]] = []

        def record_process_completed_utterance(**kwargs):
            orchestrator_calls.append(kwargs)
            return self._processed_record()

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(orchestrator_calls))
        self.assertEqual(1, orchestrator_calls[0]["conversation_id"])
        self.assertIsInstance(orchestrator_calls[0]["audio_recorded_at"], object)
        self.assertNotIn("publish_captured_audio", stdout.getvalue())
        self.assertIn("remote transcript", stdout.getvalue())
        self.assertIn("suggested reply", stdout.getvalue())
        self.assertNotIn("Detected speech chunk", stdout.getvalue())
        self.assertNotIn("Silence inside utterance", stdout.getvalue())
        self.assertNotIn("Discarded silent chunk", stdout.getvalue())

    def test_main_passes_auto_language_to_whisper_only_when_lang_auto_flag_is_set(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        orchestrator_calls: list[dict[str, object]] = []

        def record_process_completed_utterance(**kwargs):
            orchestrator_calls.append(kwargs)
            return self._processed_record()

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                ["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false", "--lang-auto"]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("auto", orchestrator_calls[0]["language"])

    def test_main_reports_reused_openai_file_when_file_name_is_none(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation_without_file_name,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(0, exit_code)
        self.assertIn(
            "file_name=None (reused existing OpenAI file)",
            stdout.getvalue(),
        )

    def test_main_prints_chunk_level_logs_only_when_enabled(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.0, 0.0], dtype=numpy.float32),
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=lambda **_kwargs: self._processed_record(),
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(False),
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--max-pause-ms",
                    "200",
                    "--segment-duration",
                    "0.2",
                    "--stream-openai=false",
                    "--min-speech-ms",
                    "0",
                    "--log-detected-speech-chunks",
                    "--log-silence-inside-utterance",
                    "--log-discarded-silent-chunks",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertIn("Detected speech chunk", stdout.getvalue())
        self.assertIn("Silence inside utterance", stdout.getvalue())
        self.assertIn("Discarded silent chunk", stdout.getvalue())

    def test_main_debug_timings_enables_processing_timings_without_verbose_runner_debug(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        observed_kwargs: list[dict[str, object]] = []

        def record_process_completed_utterance(**kwargs):
            observed_kwargs.append(kwargs)
            kwargs["openai_request_started_callback"]()
            processed = self._processed_record()
            return ProcessedUtteranceRecord(
                transcript=processed.transcript,
                suggested_reply=processed.suggested_reply,
                conversation_id=processed.conversation_id,
                turn_id=processed.turn_id,
                audio_filename=processed.audio_filename,
                audio_recorded_at=processed.audio_recorded_at,
                timings=InMemoryAudioProcessingTimings(
                    transcription_ms=111,
                    openai_ttft_ms=120,
                    openai_ttfut_ms=160,
                    openai_full_ms=222,
                    post_openai_local_ms=9,
                    sqlite_ms=3,
                    processing_ms=336,
                    end_to_end_first_useful_ms=401,
                    end_to_end_full_reply_ms=463,
                ),
            )

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener) as listener_mock,
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--max-pause-ms",
                    "200",
                    "--segment-duration",
                    "0.2",
                    "--stream-openai=false",
                    "--debug-timings",
                    "--min-speech-ms",
                    "0",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertFalse(listener_mock.call_args.kwargs["debug"])
        self.assertTrue(observed_kwargs[0]["measure_timings"])
        self.assertIn(
            "[timing] transcribe=111ms openai_ttft=120ms openai_ttfut=160ms openai_full=222ms display_final_publish=0ms runner_post_stream_display=0ms post_openai_local=9ms sqlite=3ms processing=336ms end_to_end_first_useful=401ms end_to_end_full_reply=463ms",
            stdout.getvalue(),
        )
        self.assertNotIn("[runner] analyzed chunk", stdout.getvalue())

    def test_main_debug_timings_includes_display_fields_for_streamed_reply_in_stable_order(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )

        def record_process_completed_utterance(**kwargs):
            kwargs["openai_request_started_callback"]()
            kwargs["openai_reply_started_callback"]("stream transcript")
            kwargs["openai_text_delta_callback"]("Reply ")
            kwargs["openai_text_delta_callback"]("streamed.")
            return self._processed_record_with_timings(
                transcript="stream transcript",
                reply="Reply streamed.",
                transcription_ms=100,
                openai_ttft_ms=120,
                openai_ttfut_ms=160,
                openai_full_ms=300,
                post_openai_local_ms=9,
                sqlite_ms=3,
                processing_ms=412,
                end_to_end_first_useful_ms=260,
                end_to_end_full_reply_ms=400,
            )

        with (
            patch("run.capture_voice_activated_macos_stream.time.perf_counter", side_effect=[10.0, 10.15, 10.35, 10.5, 10.6]),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--max-pause-ms",
                    "200",
                    "--segment-duration",
                    "0.2",
                    "--debug-timings",
                    "--min-speech-ms",
                    "0",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertIn(
            "[timing] transcribe=100ms openai_ttft=120ms openai_ttfut=160ms openai_full=300ms display_first_delta=150ms display_last_delta=350ms display_final_publish=500ms runner_post_stream_display=300ms post_openai_local=9ms sqlite=3ms processing=412ms end_to_end_first_useful=260ms end_to_end_full_reply=400ms",
            stdout.getvalue(),
        )

    def test_main_debug_timings_distinguishes_final_only_output_from_streamed_deltas(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )

        with (
            patch("run.capture_voice_activated_macos_stream.time.perf_counter", side_effect=[20.0, 20.5, 20.58]),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=lambda **kwargs: (
                    kwargs["openai_request_started_callback"](),
                    self._processed_record_with_timings(
                        transcription_ms=100,
                        openai_ttft_ms=None,
                        openai_ttfut_ms=None,
                        openai_full_ms=300,
                        post_openai_local_ms=11,
                        sqlite_ms=4,
                        processing_ms=430,
                        end_to_end_first_useful_ms=None,
                        end_to_end_full_reply_ms=400,
                    ),
                )[-1],
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--max-pause-ms",
                    "200",
                    "--segment-duration",
                    "0.2",
                    "--stream-openai=false",
                    "--debug-timings",
                    "--min-speech-ms",
                    "0",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertIn(
            "[timing] transcribe=100ms openai_full=300ms display_final_publish=500ms runner_post_stream_display=280ms post_openai_local=11ms sqlite=4ms processing=430ms end_to_end_full_reply=400ms",
            stdout.getvalue(),
        )
        self.assertNotIn("display_first_delta=", stdout.getvalue())
        self.assertNotIn("display_last_delta=", stdout.getvalue())

    def test_main_streams_openai_reply_when_enabled(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        observed_callbacks: list[dict[str, object]] = []

        def record_process_completed_utterance(**kwargs):
            observed_callbacks.append(kwargs)
            kwargs["openai_reply_started_callback"]("stream transcript")
            kwargs["openai_text_delta_callback"]("Reply ")
            kwargs["openai_text_delta_callback"]("streamed.")
            return self._processed_record(transcript="stream transcript", reply="Reply streamed.")

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0"])

        self.assertEqual(0, exit_code)
        self.assertTrue(observed_callbacks[0]["openai_stream"])
        self.assertIsNotNone(observed_callbacks[0]["openai_reply_started_callback"])
        self.assertIsNotNone(observed_callbacks[0]["openai_text_delta_callback"])
        self.assertIn("Remote: stream transcript", stdout.getvalue())
        self.assertIn("Reply: Reply streamed.", stdout.getvalue())

    def test_main_starts_browser_ui_by_default_prints_urls_and_stops_runtime(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime(local_url="http://127.0.0.1:43181", lan_url="http://192.168.0.5:43181")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True) as runtime_class,
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--browser-port", "44000"])

        self.assertEqual(0, exit_code)
        runtime_class.assert_called_once()
        self.assertEqual(44000, runtime_class.call_args.kwargs["preferred_port"])
        self.assertIsNone(runtime_class.call_args.kwargs["database_path"])
        self.assertTrue(runtime_class.call_args.kwargs["task_feature_enabled"])
        self.assertTrue(callable(runtime_class.call_args.kwargs["on_task_screenshot"]))
        self.assertTrue(callable(runtime_class.call_args.kwargs["on_task_send"]))
        self.assertTrue(callable(runtime_class.call_args.kwargs["on_task_clear"]))
        self.assertEqual(1, runtime.start_calls)
        self.assertEqual(1, runtime.stop_calls)
        self.assertEqual(
            [("publish_session_started", "http://127.0.0.1:43181", "http://192.168.0.5:43181"), ("publish_session_stopped",)],
            runtime.publisher.calls,
        )
        self.assertIn("Browser UI local: http://127.0.0.1:43181", stdout.getvalue())
        self.assertIn("Browser UI lan: http://192.168.0.5:43181", stdout.getvalue())

    def test_main_continues_when_browser_ui_session_started_publish_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime(local_url="http://127.0.0.1:43181", lan_url="http://192.168.0.5:43181")
        runtime.publisher.raise_on_publish_session_started = RuntimeError("session started publish failed")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, runtime.start_calls)
        self.assertEqual(1, runtime.stop_calls)
        self.assertIn("listener.start", self.events)
        self.assertIn("Browser UI local: http://127.0.0.1:43181", stdout.getvalue())
        self.assertIn("Voice-activated stream capture started.", stdout.getvalue())
        self.assertIn("Failed to publish browser UI session started: session started publish failed", stdout.getvalue())

    def test_main_skips_browser_ui_runtime_when_disabled_via_cli(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", create=True) as runtime_class,
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--browser-ui=false"])

        self.assertEqual(0, exit_code)
        runtime_class.assert_not_called()
        self.assertNotIn("Browser UI local:", stdout.getvalue())

    def test_main_passes_task_feature_enabled_to_browser_ui_runtime_by_default(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime()

        with (
            patch(
                "run.capture_voice_activated_macos_stream.BrowserUiRuntime",
                return_value=runtime,
                create=True,
            ) as runtime_class,
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(0, exit_code)
        self.assertTrue(runtime_class.call_args.kwargs["task_feature_enabled"])

    def test_main_does_not_create_task_subsystem_when_task_feature_is_disabled(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime()

        with (
            patch(
                "run.capture_voice_activated_macos_stream.BrowserUiRuntime",
                return_value=runtime,
                create=True,
            ) as runtime_class,
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--no-task-feature"])

        self.assertEqual(0, exit_code)
        self.assertFalse(runtime_class.call_args.kwargs["task_feature_enabled"])
        self.assertEqual(
            ["resolve_target", "initialize_openai_conversation", "listener.start", "listener.read_chunk", "listener.stop"],
            self.events,
        )

    def test_main_does_not_create_task_subsystem_when_task_feature_is_enabled(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime()

        with (
            patch(
                "run.capture_voice_activated_macos_stream.BrowserUiRuntime",
                return_value=runtime,
                create=True,
            ) as browser_ui_runtime_class,
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(0, exit_code)
        self.assertTrue(browser_ui_runtime_class.call_args.kwargs["task_feature_enabled"])
        self.assertEqual(
            [
                "resolve_target",
                "initialize_openai_conversation",
                "listener.start",
                "listener.read_chunk",
                "listener.stop",
            ],
            self.events,
        )

    def test_main_browser_ui_can_be_disabled_while_task_feature_default_remains_enabled(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", create=True) as runtime_class,
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--browser-ui=false"])

        self.assertEqual(0, exit_code)
        runtime_class.assert_not_called()
        self.assertNotIn("Voice-activated stream capture failed", stdout.getvalue())

    def test_main_publishes_transcript_reply_final_and_processing_error_to_browser_ui(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
                numpy.array([0.5, -0.5], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        runtime = FakeBrowserUiRuntime()
        call_count = 0

        def record_process_completed_utterance(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first utterance failed")
            return self._processed_record(transcript="browser transcript", reply="browser reply")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                ["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false"]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [
                ("publish_session_started", "http://127.0.0.1:43181", None),
                ("publish_processing_error", "first utterance failed"),
                ("publish_transcript", "browser transcript"),
                ("publish_reply_final", "browser reply"),
                ("publish_session_stopped",),
            ],
            runtime.publisher.calls,
        )
        self.assertNotIn("Detected speech chunk", runtime.publisher.calls_as_text())
        self.assertNotIn("Silence inside utterance", runtime.publisher.calls_as_text())
        self.assertNotIn("Discarded silent chunk", runtime.publisher.calls_as_text())

    def test_main_publishes_reply_delta_and_final_reply_to_browser_ui_when_streaming(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        runtime = FakeBrowserUiRuntime()

        def record_process_completed_utterance(**kwargs):
            kwargs["openai_reply_started_callback"]("stream transcript")
            kwargs["openai_text_delta_callback"]("Reply ")
            kwargs["openai_text_delta_callback"]("streamed.")
            return self._processed_record(transcript="stream transcript", reply="Reply streamed.")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0"])

        self.assertEqual(0, exit_code)
        self.assertEqual(
            [
                ("publish_session_started", "http://127.0.0.1:43181", None),
                ("publish_transcript", "stream transcript"),
                ("publish_reply_delta", "Reply "),
                ("publish_reply_delta", "streamed."),
                ("publish_reply_final", "Reply streamed."),
                ("publish_session_stopped",),
            ],
            runtime.publisher.calls,
        )

    def test_main_continues_when_browser_ui_transcript_publish_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        runtime = FakeBrowserUiRuntime()
        runtime.publisher.raise_on_publish_transcript = RuntimeError("transcript publish failed")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=lambda **_kwargs: self._processed_record(transcript="transcript text", reply="final reply"),
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, runtime.stop_calls)
        self.assertIn("transcript text", stdout.getvalue())
        self.assertIn("final reply", stdout.getvalue())
        self.assertIn("Failed to publish browser UI transcript: transcript publish failed", stdout.getvalue())

    def test_main_continues_when_browser_ui_reply_delta_publish_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
                numpy.array([0.5, -0.5], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        runtime = FakeBrowserUiRuntime()
        runtime.publisher.raise_on_publish_reply_delta = RuntimeError("reply delta publish failed")
        call_count = 0

        def record_process_completed_utterance(**kwargs):
            nonlocal call_count
            call_count += 1
            kwargs["openai_reply_started_callback"](f"stream transcript {call_count}")
            kwargs["openai_text_delta_callback"]("Reply ")
            kwargs["openai_text_delta_callback"]("streamed.")
            return self._processed_record(
                transcript=f"stream transcript {call_count}",
                reply=f"Reply streamed {call_count}.",
            )

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0"])

        self.assertEqual(0, exit_code)
        self.assertEqual(2, call_count)
        self.assertIn("Remote: stream transcript 1", stdout.getvalue())
        self.assertIn("Remote: stream transcript 2", stdout.getvalue())
        self.assertIn("Failed to publish browser UI reply delta: reply delta publish failed", stdout.getvalue())

    def test_main_continues_when_browser_ui_reply_final_publish_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        runtime = FakeBrowserUiRuntime()
        runtime.publisher.raise_on_publish_reply_final = RuntimeError("final reply publish failed")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=lambda **_kwargs: self._processed_record(transcript="remote text", reply="reply text"),
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false"])

        self.assertEqual(0, exit_code)
        self.assertIn("remote text", stdout.getvalue())
        self.assertIn("reply text", stdout.getvalue())
        self.assertIn("Failed to publish browser UI final reply: final reply publish failed", stdout.getvalue())

    def test_main_stops_browser_ui_runtime_on_keyboard_interrupt(self) -> None:
        stdout = io.StringIO()
        listener = InterruptingContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime()

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, runtime.stop_calls)
        self.assertEqual(
            [("publish_session_started", "http://127.0.0.1:43181", None), ("publish_session_stopped",)],
            runtime.publisher.calls,
        )
        self.assertIn("Stopped by user.", stdout.getvalue())

    def test_main_returns_error_when_browser_ui_runtime_start_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime(start_error=RuntimeError("browser runtime failed"))

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(1, exit_code)
        self.assertEqual(1, runtime.start_calls)
        self.assertEqual(1, runtime.stop_calls)
        self.assertEqual([], runtime.publisher.calls)
        self.assertIn("Voice-activated stream capture failed: browser runtime failed", stdout.getvalue())

    def test_main_still_stops_browser_ui_runtime_when_session_stopped_publish_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)
        runtime = FakeBrowserUiRuntime()
        runtime.publisher.raise_on_publish_session_stopped = RuntimeError("session stopped publish failed")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(1, runtime.stop_calls)
        self.assertIn(("publish_session_stopped",), runtime.publisher.calls)
        self.assertIn("Failed to publish browser UI session stop event: session stopped publish failed", stdout.getvalue())

    def test_main_continues_when_browser_ui_processing_error_publish_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
                numpy.array([0.5, -0.5], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        runtime = FakeBrowserUiRuntime()
        runtime.publisher.raise_on_publish_processing_error = RuntimeError("processing error publish failed")
        call_count = 0

        def record_process_completed_utterance(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first utterance failed")
            return self._processed_record(transcript="second transcript", reply="second reply")

        with (
            patch("run.capture_voice_activated_macos_stream.BrowserUiRuntime", return_value=runtime, create=True),
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false"])

        self.assertEqual(0, exit_code)
        self.assertEqual(2, call_count)
        self.assertIn("In-memory utterance processing failed: first utterance failed", stdout.getvalue())
        self.assertIn("Failed to publish browser UI processing error: processing error publish failed", stdout.getvalue())
        self.assertIn("second transcript", stdout.getvalue())
        self.assertIn("second reply", stdout.getvalue())

    def test_main_continues_after_per_utterance_processing_failure(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
                numpy.array([0.5, -0.5], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        call_count = 0

        def record_process_completed_utterance(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first utterance failed")
            return self._processed_record(transcript="second transcript", reply="second reply")

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=record_process_completed_utterance,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0", "--stream-openai=false"])

        self.assertEqual(0, exit_code)
        self.assertEqual(2, call_count)
        self.assertIn("In-memory utterance processing failed: first utterance failed", stdout.getvalue())
        self.assertIn("second transcript", stdout.getvalue())
        self.assertIn("second reply", stdout.getvalue())

    def test_main_returns_error_on_startup_failure(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch(
                "run.capture_voice_activated_macos_stream.resolve_target",
                side_effect=RuntimeError("No shareable application matched name 'Missing App'."),
            ),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Missing App"])

        self.assertEqual(1, exit_code)
        self.assertEqual([], self.events)
        self.assertIn("Voice-activated stream capture failed: No shareable application matched name 'Missing App'.", stdout.getvalue())

    def test_main_discards_short_speech_even_when_pre_roll_makes_buffer_longer(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.1, 0.1], dtype=numpy.float32),
                numpy.array([0.2, 0.2], dtype=numpy.float32),
                numpy.array([1.0, 1.0], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        orchestrator_calls: list[dict[str, object]] = []

        with (
            patch("run.capture_voice_activated_macos_stream.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos_stream.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos_stream.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch(
                "run.capture_voice_activated_macos_stream.process_completed_utterance",
                side_effect=lambda **kwargs: orchestrator_calls.append(kwargs) or self._processed_record(),
            ),
            patch(
                "run.capture_voice_activated_macos_stream.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(False),
                    _vad_result(False),
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--app-name",
                    "Zoom",
                    "--sample-rate",
                    "10",
                    "--segment-duration",
                    "0.2",
                    "--pre-roll-ms",
                    "400",
                    "--post-roll-ms",
                    "300",
                    "--max-pause-ms",
                    "200",
                    "--min-speech-ms",
                    "300",
                    "--stream-openai=false",
                    "--debug",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual([], orchestrator_calls)
        self.assertIn("[runner] discarded short utterance speech_ms=200 min_speech_ms=300", stdout.getvalue())

    def _record_list_targets(self, **_kwargs) -> None:
        self.events.append("list_targets")

    def _record_initialized_conversation(self, _topic_hint: str, **_kwargs) -> InitializedConversationRecord:
        self.events.append("initialize_openai_conversation")
        return InitializedConversationRecord(
            conversation_id=1,
            openai_conversation_id="conv_123",
            topic_hint="",
            file_name="file.docx",
        )

    def _record_initialized_conversation_without_file_name(
        self, _topic_hint: str, **_kwargs
    ) -> InitializedConversationRecord:
        self.events.append("initialize_openai_conversation")
        return InitializedConversationRecord(
            conversation_id=1,
            openai_conversation_id="conv_123",
            topic_hint="",
            file_name=None,
        )

    def _record_resolved_target(self, **_kwargs) -> tuple[object, object]:
        self.events.append("resolve_target")
        return object(), object()

    def _record_existing_conversation(self, conversation_id: int, **_kwargs):
        self.events.append("get_conversation")
        self.assertEqual(7, conversation_id)
        return type(
            "FakeConversationRecord",
            (),
            {
                "conversation_id": 7,
                "topic_hint": "existing topic",
                "openai_conversation_id": "conv_existing",
                "openai_file_id": "file_existing",
                "is_active": True,
                "created_at": "2026-04-24T10:11:12+00:00",
            },
        )()

    def _record_missing_conversation(self, conversation_id: int, **_kwargs):
        self.events.append("get_conversation")
        self.assertEqual(999, conversation_id)
        return None

    def _record_existing_conversation_without_openai_id(self, conversation_id: int, **_kwargs):
        self.events.append("get_conversation")
        self.assertEqual(8, conversation_id)
        return type(
            "FakeConversationRecord",
            (),
            {
                "conversation_id": 8,
                "topic_hint": "existing topic",
                "openai_conversation_id": None,
                "openai_file_id": "file_existing",
                "is_active": True,
                "created_at": "2026-04-24T10:11:12+00:00",
            },
        )()

    def _processed_record(self, *, transcript: str = "remote transcript", reply: str = "suggested reply") -> ProcessedUtteranceRecord:
        return ProcessedUtteranceRecord(
            transcript=transcript,
            suggested_reply=reply,
            conversation_id=1,
            turn_id=1,
            audio_filename=None,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            timings=InMemoryAudioProcessingTimings(
                transcription_ms=10,
                openai_ttft_ms=None,
                openai_ttfut_ms=None,
                openai_full_ms=20,
                post_openai_local_ms=15,
                sqlite_ms=5,
                processing_ms=35,
                end_to_end_first_useful_ms=None,
                end_to_end_full_reply_ms=35,
            ),
        )

    def _processed_record_with_timings(
        self,
        *,
        transcript: str = "remote transcript",
        reply: str = "suggested reply",
        transcription_ms: int,
        openai_ttft_ms: int | None,
        openai_ttfut_ms: int | None,
        openai_full_ms: int,
        post_openai_local_ms: int,
        sqlite_ms: int,
        processing_ms: int,
        end_to_end_first_useful_ms: int | None,
        end_to_end_full_reply_ms: int | None,
    ) -> ProcessedUtteranceRecord:
        return ProcessedUtteranceRecord(
            transcript=transcript,
            suggested_reply=reply,
            conversation_id=1,
            turn_id=1,
            audio_filename=None,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            timings=InMemoryAudioProcessingTimings(
                transcription_ms=transcription_ms,
                openai_ttft_ms=openai_ttft_ms,
                openai_ttfut_ms=openai_ttfut_ms,
                openai_full_ms=openai_full_ms,
                post_openai_local_ms=post_openai_local_ms,
                sqlite_ms=sqlite_ms,
                processing_ms=processing_ms,
                end_to_end_first_useful_ms=end_to_end_first_useful_ms,
                end_to_end_full_reply_ms=end_to_end_full_reply_ms,
            ),
        )


class FakeContinuousAudioListener:
    def __init__(self, events: list[str], *, chunks: list[numpy.ndarray] | None = None) -> None:
        self._events = events
        self._chunks = list(chunks or [])

    def start(self) -> str:
        self._events.append("listener.start")
        return "Zoom"

    def read_chunk(self, _segment_duration: float):
        self._events.append("listener.read_chunk")
        if self._chunks:
            return self._chunks.pop(0)
        return None

    def debug_state(self) -> str:
        return "debug-state"

    def stop(self) -> None:
        self._events.append("listener.stop")


class InterruptingContinuousAudioListener(FakeContinuousAudioListener):
    def read_chunk(self, _segment_duration: float):
        self._events.append("listener.read_chunk")
        raise KeyboardInterrupt


class FakeBrowserUiRuntime:
    def __init__(
        self,
        *,
        local_url: str = "http://127.0.0.1:43181",
        lan_url: str | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.publisher = FakeUiPublisher()
        self.local_url = local_url
        self.lan_url = lan_url
        self.start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stop_calls += 1


class FakeUiPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.raise_on_publish_session_started: Exception | None = None
        self.raise_on_publish_transcript: Exception | None = None
        self.raise_on_publish_reply_delta: Exception | None = None
        self.raise_on_publish_reply_final: Exception | None = None
        self.raise_on_publish_processing_error: Exception | None = None
        self.raise_on_publish_session_stopped: Exception | None = None

    def publish_session_started(self, *, local_url: str, lan_url: str | None) -> None:
        self.calls.append(("publish_session_started", local_url, lan_url))
        if self.raise_on_publish_session_started is not None:
            raise self.raise_on_publish_session_started

    def publish_transcript(self, remote_text: str) -> None:
        self.calls.append(("publish_transcript", remote_text))
        if self.raise_on_publish_transcript is not None:
            raise self.raise_on_publish_transcript

    def publish_reply_delta(self, delta: str) -> None:
        self.calls.append(("publish_reply_delta", delta))
        if self.raise_on_publish_reply_delta is not None:
            raise self.raise_on_publish_reply_delta

    def publish_reply_final(self, reply_text: str) -> None:
        self.calls.append(("publish_reply_final", reply_text))
        if self.raise_on_publish_reply_final is not None:
            raise self.raise_on_publish_reply_final

    def publish_processing_error(self, message: str) -> None:
        self.calls.append(("publish_processing_error", message))
        if self.raise_on_publish_processing_error is not None:
            raise self.raise_on_publish_processing_error

    def publish_session_stopped(self) -> None:
        self.calls.append(("publish_session_stopped",))
        if self.raise_on_publish_session_stopped is not None:
            raise self.raise_on_publish_session_stopped

    def publish_task_snapshot(
        self,
        *,
        status: str,
        file_count: int,
        artifacts: list[dict[str, object]],
        latest_result: dict[str, object] | None,
        error: str | None,
    ) -> None:
        self.calls.append(("publish_task_snapshot", status, file_count, artifacts, latest_result, error))

    def calls_as_text(self) -> str:
        return repr(self.calls)

if __name__ == "__main__":
    unittest.main()
