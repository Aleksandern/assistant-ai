#!/usr/bin/env python3

"""Continuously capture macOS system audio and process finalized utterances in memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import sys
import time

import numpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.in_memory_audio_processing_orchestrator import process_completed_utterance
from modules.in_memory_utterance_worker import InMemoryUtteranceProcessorWorker, QueuedUtterance
from modules.macos_audio_listener import ContinuousAudioListener, list_targets, resolve_target
from modules.openai_conversation_initializer import InitializedConversationRecord, initialize_openai_conversation
from modules.browser_ui_runtime import BrowserUiRuntime
from modules.sqlite_conversation_store import get_conversation
from modules.task_flow_service import capture_task_screenshot, clear_current_task, solve_current_task
from modules.task_runtime_adapters import build_macos_task_screenshot_adapter
from modules.terminal_output import (
    build_terminal_block_separator_printer,
    build_terminal_reply_stream_writer,
    build_terminal_stream_start_printer,
    format_processed_audio_row,
    parse_bool_cli_value,
)
from modules.utterance_assembler import FinalizedUtterance, UtteranceAssembler
from modules.wav_voice_activity import StreamingVoiceActivityDetector


def _format_file_source_status(file_name: str | None) -> str:
    if file_name:
        return f"file_name={file_name}"
    return "file_name=None (reused existing OpenAI file)"


def build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Continuously monitor a macOS app/display and process finalized utterances directly in memory."
    )
    parser.add_argument("--display-index", type=int, default=0, help="Display index from shareable content.")
    parser.add_argument("--app-name", help="Monitor an application whose visible name matches the given text.")
    parser.add_argument("--bundle-id", help="Monitor an application by exact macOS bundle identifier.")
    parser.add_argument("--list-targets", action="store_true", help="List shareable displays and applications.")
    parser.add_argument("--segment-duration", type=float, default=0.2, help="Length of each analysis chunk in seconds. Smaller values reduce clipping at utterance boundaries.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target mono sample rate for VAD and in-memory transcription.")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Speech confidence threshold used by Silero VAD.")
    parser.add_argument("--min-speech-ms", type=int, default=250, help="Minimum speech duration passed to Silero VAD.")
    parser.add_argument("--speech-pad-ms", type=int, default=30, help="Padding added by Silero VAD around detected speech regions.")
    parser.add_argument("--pre-roll-ms", type=int, default=400, help="Audio kept before detected speech so the start is not cut off.")
    parser.add_argument("--max-pause-ms", type=int, default=800, help="Maximum tolerated pause inside one utterance before it is finalized.")
    parser.add_argument("--post-roll-ms", type=int, default=300, help="Silence tail kept at the end of a finalized utterance.")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Short pause while waiting during fully silent periods.")
    parser.add_argument("--include-self-audio", action="store_true", help="Include audio from the current process.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout for ScreenCaptureKit async operations.")
    parser.add_argument("--database-path", help="Optional SQLite database path. Defaults to the repository database directory.")
    parser.add_argument("--conv-id", type=int, help="Reuse an existing local conversation id from the conversations table.")
    parser.add_argument("--language", help="Optional Whisper language code such as en or ru.")
    parser.add_argument(
        "--lang-auto",
        action="store_true",
        help="Explicitly enable whisper.cpp language auto-detection by passing --language auto.",
    )
    parser.add_argument("--whisper-cli-path", help="Optional explicit path to the whisper.cpp CLI binary.")
    parser.add_argument("--whisper-model-path", help="Optional explicit path to the Whisper model file.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable whisper.cpp GPU inference.")
    parser.add_argument("--openai-api-key", help="Optional explicit OpenAI API key.")
    parser.add_argument("--openai-model", help="Optional explicit OpenAI model name.")
    parser.add_argument("--openai-instructions", help="Optional OpenAI system instructions for conversation init and reply generation.")
    parser.add_argument("--openai-dotenv-path", help="Optional explicit path to the .env file for OpenAI settings.")
    parser.add_argument("--debug", action="store_true", help="Print listener, VAD, and in-memory processing diagnostics.")
    parser.add_argument(
        "--debug-timings",
        action="store_true",
        help="Print only per-utterance processing timings without enabling listener/VAD debug logs.",
    )
    parser.add_argument("--debug-openai", action="store_true", help="Enable detailed OpenAI request timing diagnostics.")
    parser.add_argument(
        "--log-detected-speech-chunks",
        action="store_true",
        help="Print one line for each chunk classified as speech inside an utterance.",
    )
    parser.add_argument(
        "--log-silence-inside-utterance",
        action="store_true",
        help="Print one line for each silent chunk accumulated inside an open utterance.",
    )
    parser.add_argument(
        "--log-discarded-silent-chunks",
        action="store_true",
        help="Print one line for each fully silent chunk discarded before speech starts.",
    )
    parser.add_argument(
        "--stream-openai",
        type=parse_bool_cli_value,
        nargs="?",
        const=True,
        default=True,
        help="Stream reply text to the terminal while OpenAI is generating it. Defaults to true; set --stream-openai=false to disable.",
    )
    parser.add_argument(
        "--browser-ui",
        type=parse_bool_cli_value,
        nargs="?",
        const=True,
        default=True,
        help="Start the local browser UI runtime. Defaults to true; set --browser-ui=false to disable.",
    )
    parser.add_argument(
        "--browser-port",
        type=int,
        default=None,
        help="Preferred HTTP port for the browser UI runtime. Falls back automatically if the port is busy.",
    )
    parser.add_argument(
        "--no-task-feature",
        action="store_false",
        default=True,
        dest="task_feature",
        help="Disable the optional task screenshot + solve subsystem and hide task UI.",
    )
    return parser


def format_in_memory_processing_timings(timings, display_timings: RunnerDisplayTimings | None = None) -> str:
    parts = [f"transcribe={timings.transcription_ms}ms"]
    if timings.openai_ttft_ms is not None:
        parts.append(f"openai_ttft={timings.openai_ttft_ms}ms")
    if timings.openai_ttfut_ms is not None:
        parts.append(f"openai_ttfut={timings.openai_ttfut_ms}ms")
    parts.append(f"openai_full={timings.openai_full_ms}ms")
    if display_timings is not None and display_timings.display_first_delta_ms is not None:
        parts.append(f"display_first_delta={display_timings.display_first_delta_ms}ms")
    if display_timings is not None and display_timings.display_last_delta_ms is not None:
        parts.append(f"display_last_delta={display_timings.display_last_delta_ms}ms")
    if display_timings is not None and display_timings.display_final_publish_ms is not None:
        parts.append(f"display_final_publish={display_timings.display_final_publish_ms}ms")
    if display_timings is not None and display_timings.runner_post_stream_display_ms is not None:
        parts.append(f"runner_post_stream_display={display_timings.runner_post_stream_display_ms}ms")
    parts.extend(
        [
            f"post_openai_local={timings.post_openai_local_ms}ms",
            f"sqlite={timings.sqlite_ms}ms",
            f"processing={timings.processing_ms}ms",
        ]
    )
    if timings.end_to_end_first_useful_ms is not None:
        parts.append(f"end_to_end_first_useful={timings.end_to_end_first_useful_ms}ms")
    if timings.end_to_end_full_reply_ms is not None:
        parts.append(f"end_to_end_full_reply={timings.end_to_end_full_reply_ms}ms")
    return "[timing] " + " ".join(parts)


def _resolve_whisper_language_arg(args) -> str | None:
    if args.language is not None:
        normalized_language = args.language.strip()
        return normalized_language or None

    if args.lang_auto:
        return "auto"

    return None


@dataclass(frozen=True)
class RunnerDisplayTimings:
    display_first_delta_ms: int | None
    display_last_delta_ms: int | None
    display_final_publish_ms: int | None
    runner_post_stream_display_ms: int | None


class RunnerDisplayTimingCollector:
    """Capture runner-level display milestones after the OpenAI request actually starts.

    These timings measure when the runner finishes pushing text into its local display
    sinks (terminal and browser publisher). They do not confirm frontend repaint time.
    """

    def __init__(self, *, measure_enabled: bool) -> None:
        self._measure_enabled = measure_enabled
        self._openai_request_started_at_monotonic: float | None = None
        self._first_delta_displayed_at_monotonic: float | None = None
        self._last_delta_displayed_at_monotonic: float | None = None
        self._final_reply_published_at_monotonic: float | None = None
        self._terminal_output_completed_at_monotonic: float | None = None

    def mark_openai_request_started(self) -> None:
        if not self._measure_enabled:
            return
        if self._openai_request_started_at_monotonic is None:
            self._openai_request_started_at_monotonic = time.perf_counter()

    def mark_reply_delta_displayed(self) -> None:
        if not self._measure_enabled:
            return
        now = time.perf_counter()
        if self._first_delta_displayed_at_monotonic is None:
            self._first_delta_displayed_at_monotonic = now
        self._last_delta_displayed_at_monotonic = now

    def mark_reply_final_published(self) -> None:
        if not self._measure_enabled:
            return
        self._final_reply_published_at_monotonic = time.perf_counter()

    def mark_terminal_output_completed(self) -> None:
        if not self._measure_enabled:
            return
        self._terminal_output_completed_at_monotonic = time.perf_counter()

    def build_timings(self, processing_timings) -> RunnerDisplayTimings | None:
        openai_request_started_at_monotonic = self._openai_request_started_at_monotonic
        if openai_request_started_at_monotonic is None or processing_timings is None:
            return None

        openai_stream_completed_at_monotonic = (
            openai_request_started_at_monotonic + (processing_timings.openai_full_ms / 1000)
        )
        final_reply_displayed_at_monotonic = (
            self._final_reply_published_at_monotonic or self._terminal_output_completed_at_monotonic
        )
        final_display_completed_at_monotonic = _max_defined_monotonic(
            self._final_reply_published_at_monotonic,
            self._terminal_output_completed_at_monotonic,
        )
        return RunnerDisplayTimings(
            display_first_delta_ms=_elapsed_since_ms(
                openai_request_started_at_monotonic,
                self._first_delta_displayed_at_monotonic,
            ),
            display_last_delta_ms=_elapsed_since_ms(
                openai_request_started_at_monotonic,
                self._last_delta_displayed_at_monotonic,
            ),
            display_final_publish_ms=_elapsed_since_ms(
                openai_request_started_at_monotonic,
                final_reply_displayed_at_monotonic,
            ),
            runner_post_stream_display_ms=_elapsed_since_ms(
                openai_stream_completed_at_monotonic,
                final_display_completed_at_monotonic,
            ),
        )


def _elapsed_since_ms(started_at_monotonic: float, finished_at_monotonic: float | None) -> int | None:
    if finished_at_monotonic is None:
        return None
    return max(0, round((finished_at_monotonic - started_at_monotonic) * 1000))


def _max_defined_monotonic(*timestamps: float | None) -> float | None:
    defined_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if not defined_timestamps:
        return None
    return max(defined_timestamps)


def configure_logging(*, debug_openai: bool) -> None:
    if debug_openai:
        logging.basicConfig(level=logging.INFO)


def _resolve_initialized_conversation(args) -> tuple[InitializedConversationRecord, bool]:
    if args.conv_id is None:
        return (
            initialize_openai_conversation(
                "",
                api_key=args.openai_api_key,
                model=args.openai_model,
                dotenv_path=args.openai_dotenv_path,
                database_path=args.database_path,
                instructions=args.openai_instructions,
            ),
            False,
        )

    existing_conversation = get_conversation(args.conv_id, database_path=args.database_path)
    if existing_conversation is None:
        raise ValueError(f"Conversation with id={args.conv_id} was not found.")
    normalized_openai_conversation_id = (existing_conversation.openai_conversation_id or "").strip()
    if not normalized_openai_conversation_id:
        raise ValueError(
            f"Conversation with id={args.conv_id} does not have openai_conversation_id and cannot be reused."
        )

    return (
        InitializedConversationRecord(
            conversation_id=existing_conversation.conversation_id,
            openai_conversation_id=normalized_openai_conversation_id,
            topic_hint=existing_conversation.topic_hint,
            file_name=None,
        ),
        True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(debug_openai=args.debug_openai)
    measure_timings = args.debug or args.debug_timings

    listener = None
    worker = None
    browser_ui_runtime = None
    browser_ui_publisher = None
    browser_ui_started = False

    try:
        if args.list_targets:
            list_targets(timeout=args.timeout)
            return 0

        resolve_target(
            display_index=args.display_index,
            app_name=args.app_name,
            bundle_id=args.bundle_id,
            timeout=args.timeout,
        )
        initialized_conversation, reused_existing_conversation = _resolve_initialized_conversation(args)
        listener = ContinuousAudioListener(
            display_index=args.display_index,
            app_name=args.app_name,
            bundle_id=args.bundle_id,
            include_self_audio=args.include_self_audio,
            timeout=args.timeout,
            sample_rate=args.sample_rate,
            debug=args.debug,
        )
        print(
            "Conversation reused:" if reused_existing_conversation else "Conversation initialized:",
            f"conversation_id={initialized_conversation.conversation_id}",
            f"openai_conversation_id={initialized_conversation.openai_conversation_id}",
            _format_file_source_status(initialized_conversation.file_name),
        )
        if args.browser_ui:
            runtime_kwargs = {}
            if args.browser_port is not None:
                runtime_kwargs["preferred_port"] = args.browser_port
            runtime_kwargs["database_path"] = args.database_path
            runtime_kwargs["task_feature_enabled"] = args.task_feature
            if args.task_feature:
                screenshot_adapter = build_macos_task_screenshot_adapter()
                runtime_publisher_holder: dict[str, object] = {}
                runtime_kwargs["on_task_screenshot"] = lambda: capture_task_screenshot(
                    screenshot_adapter=screenshot_adapter,
                    publisher=runtime_publisher_holder["publisher"],
                    task_artifact_dir=None,
                    database_path=args.database_path,
                )
                runtime_kwargs["on_task_send"] = lambda: solve_current_task(
                    publisher=runtime_publisher_holder["publisher"],
                    task_artifact_dir=None,
                    database_path=args.database_path,
                    api_key=args.openai_api_key,
                    model=args.openai_model,
                    dotenv_path=args.openai_dotenv_path,
                )
                runtime_kwargs["on_task_clear"] = lambda: clear_current_task(
                    publisher=runtime_publisher_holder["publisher"],
                    task_artifact_dir=None,
                    database_path=args.database_path,
                )
                browser_ui_runtime = BrowserUiRuntime(**runtime_kwargs)
                runtime_publisher_holder["publisher"] = browser_ui_runtime.publisher
            else:
                browser_ui_runtime = BrowserUiRuntime(**runtime_kwargs)
            browser_ui_runtime.start()
            browser_ui_publisher = browser_ui_runtime.publisher
            _safe_browser_ui_publish(
                browser_ui_publisher.publish_session_started,
                local_url=browser_ui_runtime.local_url,
                lan_url=browser_ui_runtime.lan_url,
                failure_label="session started",
            )
            browser_ui_started = True
        block_separator_printer = build_terminal_block_separator_printer()

        def process_queued_utterance(queued_utterance: QueuedUtterance):
            raw_stream_start_printer = build_terminal_stream_start_printer() if args.stream_openai else None
            raw_stream_writer = build_terminal_reply_stream_writer() if args.stream_openai else None
            stream_start_printer = None
            transcript_published = False
            display_timing_collector = RunnerDisplayTimingCollector(measure_enabled=measure_timings)

            def publish_transcript(remote_text: str) -> None:
                nonlocal transcript_published
                if transcript_published:
                    return
                if browser_ui_publisher is not None:
                    _safe_browser_ui_publish(
                        browser_ui_publisher.publish_transcript,
                        remote_text,
                        failure_label="transcript",
                    )
                transcript_published = True

            def stream_writer(delta: str) -> None:
                if raw_stream_writer is not None:
                    raw_stream_writer(delta)
                if browser_ui_publisher is not None:
                    _safe_browser_ui_publish(
                        browser_ui_publisher.publish_reply_delta,
                        delta,
                        failure_label="reply delta",
                    )
                if delta:
                    display_timing_collector.mark_reply_delta_displayed()

            if raw_stream_start_printer is not None:
                def start_stream(remote_text: str) -> None:
                    block_separator_printer()
                    raw_stream_start_printer(remote_text)
                    publish_transcript(remote_text)

                setattr(start_stream, "has_started_output", raw_stream_start_printer.has_started_output)
                stream_start_printer = start_stream
            processed = process_completed_utterance(
                conversation_id=initialized_conversation.conversation_id,
                audio=queued_utterance.audio,
                sample_rate=queued_utterance.sample_rate,
                audio_recorded_at=queued_utterance.audio_recorded_at,
                database_path=args.database_path,
                whisper_cli_path=args.whisper_cli_path,
                whisper_model_path=args.whisper_model_path,
                language=_resolve_whisper_language_arg(args),
                use_gpu=args.use_gpu,
                openai_api_key=args.openai_api_key,
                openai_model=args.openai_model,
                openai_instructions=args.openai_instructions,
                openai_dotenv_path=args.openai_dotenv_path,
                openai_debug=args.debug_openai,
                openai_stream=args.stream_openai,
                openai_request_started_callback=display_timing_collector.mark_openai_request_started,
                openai_reply_started_callback=stream_start_printer,
                openai_text_delta_callback=stream_writer if args.stream_openai else None,
                measure_timings=measure_timings,
            )
            publish_transcript(processed.transcript)
            if browser_ui_publisher is not None:
                _safe_browser_ui_publish(
                    browser_ui_publisher.publish_reply_final,
                    processed.suggested_reply,
                    failure_label="final reply",
                )
                display_timing_collector.mark_reply_final_published()
            if args.stream_openai:
                if stream_start_printer is not None and not stream_start_printer.has_started_output():
                    stream_start_printer(processed.transcript)
                if raw_stream_writer is not None and not raw_stream_writer.has_written_output():
                    print(processed.suggested_reply, end="", flush=True)
                print()
            else:
                block_separator_printer()
                print(format_processed_audio_row(processed.transcript, processed.suggested_reply))
            display_timing_collector.mark_terminal_output_completed()
            if measure_timings and processed.timings is not None:
                display_timings = display_timing_collector.build_timings(processed.timings)
                print(format_in_memory_processing_timings(processed.timings, display_timings))
            print()
            return processed

        worker = InMemoryUtteranceProcessorWorker(
            process_utterance=process_queued_utterance,
            on_processing_error=lambda _queued, exc: _handle_processing_error(
                exc,
                browser_ui_publisher=browser_ui_publisher,
            ),
        )
        worker.start()

        target_description = listener.start()
        if browser_ui_runtime is not None:
            print(f"Browser UI local: {browser_ui_runtime.local_url}")
            if browser_ui_runtime.lan_url is not None:
                print(f"Browser UI lan: {browser_ui_runtime.lan_url}")
        print(f"Voice-activated stream capture started. Listening to {target_description}. Press Ctrl+C to stop.")

        assembler = UtteranceAssembler(
            sample_rate=args.sample_rate,
            segment_duration=args.segment_duration,
            pre_roll_ms=args.pre_roll_ms,
            max_pause_ms=0,
            post_roll_ms=args.post_roll_ms,
        )
        streaming_vad = StreamingVoiceActivityDetector(
            sample_rate=args.sample_rate,
            threshold=args.vad_threshold,
            min_silence_duration_ms=args.max_pause_ms,
            speech_pad_ms=args.speech_pad_ms,
        )

        def enqueue_finalized_utterance(finalized_utterance: FinalizedUtterance) -> None:
            speech_ms = finalized_utterance.speech_chunk_count * assembler.chunk_ms
            if speech_ms < args.min_speech_ms:
                if args.debug:
                    print(
                        "[runner] discarded short utterance",
                        f"speech_ms={speech_ms}",
                        f"min_speech_ms={args.min_speech_ms}",
                    )
                return
            audio_recorded_at = datetime.now(tz=UTC)
            worker.submit(
                QueuedUtterance(
                    audio=finalized_utterance.audio,
                    sample_rate=args.sample_rate,
                    audio_recorded_at=audio_recorded_at,
                    utterance_chunk_count=finalized_utterance.utterance_chunk_count,
                    trailing_chunk_count=finalized_utterance.trailing_chunk_count,
                    trailing_pause_ms=finalized_utterance.trailing_pause_ms,
                )
            )
            print(
                "Queued utterance for in-memory processing "
                f"(chunks={finalized_utterance.utterance_chunk_count + finalized_utterance.trailing_chunk_count}, "
                f"trailing_pause_ms={finalized_utterance.trailing_pause_ms}, "
                f"audio_recorded_at={audio_recorded_at.isoformat(timespec='seconds')})"
            )

        def finalize_utterance() -> None:
            finalized_utterance = assembler.finalize()
            if finalized_utterance is None:
                return
            enqueue_finalized_utterance(finalized_utterance)

        while True:
            audio_chunk = listener.read_chunk(args.segment_duration)
            if audio_chunk is None:
                if args.debug:
                    print("[runner] listener returned no chunk", listener.debug_state())
                finalize_utterance()
                break

            vad_result = streaming_vad.process_chunk(audio_chunk)
            if args.debug:
                peak = float(numpy.max(numpy.abs(audio_chunk))) if audio_chunk.size else 0.0
                rms = float(numpy.sqrt(numpy.mean(audio_chunk * audio_chunk))) if audio_chunk.size else 0.0
                print(
                    "[runner] analyzed chunk",
                    f"samples={audio_chunk.shape[0]}",
                    f"speech_active={vad_result.speech_active}",
                    f"speech_started={vad_result.speech_started}",
                    f"speech_ended={vad_result.speech_ended}",
                    f"contains_speech={vad_result.contains_speech}",
                    f"processed_samples={vad_result.processed_samples}",
                    f"peak={peak:.4f}",
                    f"rms={rms:.4f}",
                    f"vad_pending_samples={streaming_vad.pending_sample_count()}",
                )

            result = assembler.push_chunk(audio_chunk, has_recent_voice=vad_result.contains_speech)
            if result.event == "speech":
                if args.debug and result.prepended_pre_roll_samples:
                    print("[runner] prepended pre-roll samples", result.prepended_pre_roll_samples)
                if args.log_detected_speech_chunks:
                    print("Detected speech chunk")
            elif result.event == "silence":
                if args.log_silence_inside_utterance:
                    print(f"Silence inside utterance (accumulated_pause_ms={result.accumulated_pause_ms})")
                if result.finalized_utterance is not None:
                    enqueue_finalized_utterance(result.finalized_utterance)
            else:
                if args.log_discarded_silent_chunks:
                    print("Discarded silent chunk")
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        if "finalize_utterance" in locals():
            try:
                finalize_utterance()
            except Exception as exc:
                print(f"Failed to finalize pending utterance during shutdown: {exc}")
        print("\nStopped by user.")
        return 0
    except Exception as exc:
        print(f"Voice-activated stream capture failed: {exc}")
        return 1
    finally:
        if listener is not None:
            if args.debug:
                print("[runner] final listener state", listener.debug_state())
            listener.stop()
        if worker is not None:
            worker.stop(process_pending=True)
        _stop_browser_ui_runtime(
            browser_ui_runtime=browser_ui_runtime,
            browser_ui_publisher=browser_ui_publisher,
            browser_ui_started=browser_ui_started,
        )

    return 0


def _handle_processing_error(exc: Exception, *, browser_ui_publisher) -> None:
    print(f"In-memory utterance processing failed: {exc}\n")
    if browser_ui_publisher is not None:
        _safe_browser_ui_publish(
            browser_ui_publisher.publish_processing_error,
            str(exc),
            failure_label="processing error",
        )


def _safe_browser_ui_publish(publish_method, *args, failure_label: str, **kwargs) -> None:
    try:
        publish_method(*args, **kwargs)
    except Exception as exc:
        print(f"Failed to publish browser UI {failure_label}: {exc}")


def _stop_browser_ui_runtime(*, browser_ui_runtime, browser_ui_publisher, browser_ui_started: bool) -> None:
    if browser_ui_started and browser_ui_publisher is not None:
        try:
            browser_ui_publisher.publish_session_stopped()
        except Exception as exc:
            print(f"Failed to publish browser UI session stop event: {exc}")

    if browser_ui_runtime is not None:
        try:
            browser_ui_runtime.stop()
        except Exception as exc:
            print(f"Failed to stop browser UI runtime: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
