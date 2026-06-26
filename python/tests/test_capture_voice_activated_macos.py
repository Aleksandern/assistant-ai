from __future__ import annotations

"""Tests for the voice-activated capture runner."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.openai_conversation_initializer import InitializedConversationRecord
from run.capture_voice_activated_macos import main


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


class CaptureVoiceActivatedMacosRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.output_dir = self.root_dir / "voice-activated"
        self.events: list[str] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_main_initializes_conversation_before_listener_starts(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

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
        self.assertIn("Conversation initialized:", stdout.getvalue())
        self.assertIn("Voice-activated capture started.", stdout.getvalue())

    def test_main_does_not_initialize_conversation_when_target_preflight_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch(
                "run.capture_voice_activated_macos.resolve_target",
                side_effect=RuntimeError("No shareable application matched name 'Missing App'."),
            ),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Missing App"])

        self.assertEqual(1, exit_code)
        self.assertEqual([], self.events)
        self.assertIn("Voice-activated capture failed: No shareable application matched name 'Missing App'.", stdout.getvalue())

    def test_main_does_not_start_listener_when_conversation_initialization_fails(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=RuntimeError("openai init failed"),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom"])

        self.assertEqual(1, exit_code)
        self.assertEqual(["resolve_target"], self.events)
        self.assertIn("Voice-activated capture failed: openai init failed", stdout.getvalue())

    def test_main_reports_reused_openai_file_when_file_name_is_none(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(self.events)

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
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

    def test_main_passes_initialized_conversation_id_to_published_audio(self) -> None:
        stdout = io.StringIO()
        listener = FakeContinuousAudioListener(
            self.events,
            chunks=[
                numpy.array([0.25, -0.25], dtype=numpy.float32),
                numpy.array([0.0, 0.0], dtype=numpy.float32),
            ],
        )
        published_calls: list[dict[str, object]] = []

        def record_publish_captured_audio(**kwargs):
            published_calls.append(kwargs)
            return self.output_dir / "inbox" / "utterance-conv-1-20260423-123456-123456.wav"

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch("run.capture_voice_activated_macos.publish_captured_audio", side_effect=record_publish_captured_audio),
            patch(
                "run.capture_voice_activated_macos.StreamingVoiceActivityDetector",
                return_value=FakeStreamingVoiceActivityDetector([
                    _vad_result(True, speech_started=True),
                    _vad_result(False, speech_ended=True),
                ]),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(["--app-name", "Zoom", "--max-pause-ms", "200", "--segment-duration", "0.2", "--min-speech-ms", "0"])

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(published_calls))
        self.assertEqual(1, published_calls[0]["conversation_id"])
        self.assertEqual("utterance", published_calls[0]["prefix"])

    def test_main_preserves_utterance_assembly_behavior_when_publishing(self) -> None:
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
        published_calls: list[dict[str, object]] = []

        def record_publish_captured_audio(**kwargs):
            published_calls.append(kwargs)
            return self.output_dir / "inbox" / "utterance-conv-1-20260423-123456-123456.wav"

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch("run.capture_voice_activated_macos.publish_captured_audio", side_effect=record_publish_captured_audio),
            patch(
                "run.capture_voice_activated_macos.StreamingVoiceActivityDetector",
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
                    "0",
                    "--max-pause-ms",
                    "200",
                    "--min-speech-ms",
                    "0",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(published_calls))
        numpy.testing.assert_array_equal(
            numpy.array([0.1, 0.1, 0.2, 0.2, 1.0, 1.0], dtype=numpy.float32),
            published_calls[0]["audio"],
        )
        self.assertIn("Saved utterance:", stdout.getvalue())
        self.assertNotIn("Detected speech chunk", stdout.getvalue())
        self.assertNotIn("Silence inside utterance", stdout.getvalue())
        self.assertNotIn("Discarded silent chunk", stdout.getvalue())

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
        published_calls: list[dict[str, object]] = []

        def record_publish_captured_audio(**kwargs):
            published_calls.append(kwargs)
            return self.output_dir / "inbox" / "utterance-conv-1-20260423-123456-123456.wav"

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch("run.capture_voice_activated_macos.publish_captured_audio", side_effect=record_publish_captured_audio),
            patch(
                "run.capture_voice_activated_macos.StreamingVoiceActivityDetector",
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
                    "--min-speech-ms",
                    "0",
                    "--log-detected-speech-chunks",
                    "--log-silence-inside-utterance",
                    "--log-discarded-silent-chunks",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(published_calls))
        self.assertIn("Detected speech chunk", stdout.getvalue())
        self.assertIn("Silence inside utterance", stdout.getvalue())
        self.assertIn("Discarded silent chunk", stdout.getvalue())

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
        published_calls: list[dict[str, object]] = []

        with (
            patch("run.capture_voice_activated_macos.ensure_output_dir", return_value=self.output_dir),
            patch("run.capture_voice_activated_macos.ContinuousAudioListener", return_value=listener),
            patch("run.capture_voice_activated_macos.resolve_target", side_effect=self._record_resolved_target),
            patch(
                "run.capture_voice_activated_macos.initialize_openai_conversation",
                side_effect=self._record_initialized_conversation,
            ),
            patch("run.capture_voice_activated_macos.publish_captured_audio", side_effect=lambda **kwargs: published_calls.append(kwargs)),
            patch(
                "run.capture_voice_activated_macos.StreamingVoiceActivityDetector",
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
                    "--debug",
                ]
            )

        self.assertEqual(0, exit_code)
        self.assertEqual([], published_calls)
        self.assertIn("[runner] discarded short utterance speech_ms=200 min_speech_ms=300", stdout.getvalue())

    def _record_initialized_conversation(self, _topic_hint: str) -> InitializedConversationRecord:
        self.events.append("initialize_openai_conversation")
        return InitializedConversationRecord(
            conversation_id=1,
            openai_conversation_id="conv_123",
            topic_hint="",
            file_name="file.docx",
        )

    def _record_initialized_conversation_without_file_name(
        self, _topic_hint: str
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


if __name__ == "__main__":
    unittest.main()
