from __future__ import annotations

"""Process one completed in-memory utterance through STT, OpenAI, and SQLite persistence."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import time

import numpy

from modules.openai_conversation_reply_generator import generate_reply_in_openai_conversation
from modules.sqlite_conversation_store import add_conversation_turn
from modules.whisper_cpp_transcriber import transcribe_audio_buffer


@dataclass(frozen=True)
class InMemoryAudioProcessingTimings:
    transcription_ms: int
    openai_ttft_ms: int | None
    openai_ttfut_ms: int | None
    openai_full_ms: int
    post_openai_local_ms: int
    sqlite_ms: int
    processing_ms: int
    end_to_end_first_useful_ms: int | None
    end_to_end_full_reply_ms: int | None


@dataclass(frozen=True)
class ProcessedUtteranceRecord:
    transcript: str
    suggested_reply: str
    conversation_id: int
    turn_id: int
    audio_filename: str | None
    audio_recorded_at: str
    timings: InMemoryAudioProcessingTimings | None


class InMemoryAudioProcessingError(RuntimeError):
    def __init__(
        self,
        *,
        conversation_id: int,
        stage: str,
        audio_recorded_at: str,
        original_error: Exception,
    ) -> None:
        super().__init__(
            "In-memory utterance processing failed during "
            f"{stage}: {original_error}. conversation_id={conversation_id}, audio_recorded_at={audio_recorded_at}"
        )
        self.conversation_id = conversation_id
        self.stage = stage
        self.audio_recorded_at = audio_recorded_at
        self.original_error = original_error


def process_completed_utterance(
    *,
    conversation_id: int,
    audio: numpy.ndarray,
    sample_rate: int,
    audio_recorded_at: str | datetime,
    database_path: str | Path | None = None,
    transcribe_utterance_audio: Callable[[numpy.ndarray, int], str] | None = None,
    generate_reply_for_text: Callable[[int, str], str] | None = None,
    whisper_cli_path: str | Path | None = None,
    whisper_model_path: str | Path | None = None,
    language: str | None = None,
    use_gpu: bool = False,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_instructions: str | None = None,
    openai_dotenv_path: str | Path | None = None,
    openai_debug: bool = False,
    openai_fast_mode: bool = False,
    openai_fast_model: str | None = None,
    openai_service_tier: str | None = None,
    openai_use_conversation: bool = True,
    openai_disable_instructions: bool = False,
    openai_minimal_instructions: str | None = None,
    openai_stream: bool = False,
    openai_request_started_callback: Callable[[], None] | None = None,
    openai_reply_started_callback: Callable[[str], None] | None = None,
    openai_text_delta_callback: Callable[[str], None] | None = None,
    measure_timings: bool = False,
) -> ProcessedUtteranceRecord:
    normalized_audio_recorded_at = _normalize_audio_recorded_at(audio_recorded_at)
    processing_started_at_monotonic = time.perf_counter() if measure_timings else None

    transcribe_audio = transcribe_utterance_audio or (
        lambda utterance_audio, utterance_sample_rate: transcribe_audio_buffer(
            utterance_audio,
            utterance_sample_rate,
            whisper_cli_path=whisper_cli_path,
            model_path=whisper_model_path,
            language=language,
            use_gpu=use_gpu,
        )
    )

    transcription_started_at_monotonic = time.perf_counter() if measure_timings else None
    try:
        transcript = transcribe_audio(audio, sample_rate)
    except Exception as exc:
        raise InMemoryAudioProcessingError(
            conversation_id=conversation_id,
            stage="transcription",
            audio_recorded_at=normalized_audio_recorded_at,
            original_error=exc,
        ) from exc
    transcription_ms = _elapsed_ms(transcription_started_at_monotonic) if transcription_started_at_monotonic else None

    openai_started_at_monotonic = time.perf_counter() if measure_timings else None
    try:
        if generate_reply_for_text is None:
            if openai_request_started_callback is not None:
                openai_request_started_callback()
            if openai_stream and openai_reply_started_callback is not None:
                openai_reply_started_callback(transcript)
            openai_reply = generate_reply_in_openai_conversation(
                conversation_id,
                transcript,
                api_key=openai_api_key,
                model=openai_model,
                instructions=openai_instructions,
                dotenv_path=openai_dotenv_path,
                database_path=database_path,
                debug=openai_debug,
                fast_mode=openai_fast_mode,
                fast_model=openai_fast_model,
                service_tier=openai_service_tier,
                use_conversation=openai_use_conversation,
                disable_instructions=openai_disable_instructions,
                minimal_instructions=openai_minimal_instructions,
                stream=openai_stream,
                on_text_delta=openai_text_delta_callback,
            )
            suggested_reply = openai_reply.reply_text
            openai_reply_timings = openai_reply.timings
        else:
            suggested_reply = generate_reply_for_text(conversation_id, transcript)
            openai_reply_timings = None
    except Exception as exc:
        raise InMemoryAudioProcessingError(
            conversation_id=conversation_id,
            stage="openai",
            audio_recorded_at=normalized_audio_recorded_at,
            original_error=exc,
        ) from exc
    openai_full_ms = _elapsed_ms(openai_started_at_monotonic) if openai_started_at_monotonic else None
    post_openai_started_at_monotonic = time.perf_counter() if measure_timings else None

    sqlite_started_at_monotonic = time.perf_counter() if measure_timings else None
    try:
        created_turn = add_conversation_turn(
            conversation_id=conversation_id,
            remote_text=transcript,
            reply_text_suggest=suggested_reply,
            audio_filename=None,
            audio_recorded_at=normalized_audio_recorded_at,
            database_path=database_path,
        )
    except Exception as exc:
        raise InMemoryAudioProcessingError(
            conversation_id=conversation_id,
            stage="sqlite",
            audio_recorded_at=normalized_audio_recorded_at,
            original_error=exc,
        ) from exc
    sqlite_ms = _elapsed_ms(sqlite_started_at_monotonic) if sqlite_started_at_monotonic else None
    post_openai_local_ms = (
        _elapsed_ms(post_openai_started_at_monotonic) if post_openai_started_at_monotonic is not None else None
    )

    return ProcessedUtteranceRecord(
        transcript=transcript,
        suggested_reply=suggested_reply,
        conversation_id=created_turn.conversation_id,
        turn_id=created_turn.turn_id,
        audio_filename=created_turn.audio_filename,
        audio_recorded_at=created_turn.audio_recorded_at,
        timings=_build_timings(
            measure_timings=measure_timings,
            transcription_ms=transcription_ms,
            openai_reply_timings=openai_reply_timings,
            openai_full_ms=openai_full_ms,
            post_openai_local_ms=post_openai_local_ms,
            sqlite_ms=sqlite_ms,
            processing_started_at_monotonic=processing_started_at_monotonic,
        ),
    )


def _build_timings(
    *,
    measure_timings: bool,
    transcription_ms: int | None,
    openai_reply_timings: object | None,
    openai_full_ms: int | None,
    post_openai_local_ms: int | None,
    sqlite_ms: int | None,
    processing_started_at_monotonic: float | None,
) -> InMemoryAudioProcessingTimings | None:
    if not measure_timings:
        return None

    return InMemoryAudioProcessingTimings(
        transcription_ms=transcription_ms or 0,
        openai_ttft_ms=getattr(openai_reply_timings, "ttft_ms", None),
        openai_ttfut_ms=getattr(openai_reply_timings, "ttfut_ms", None),
        openai_full_ms=_resolve_openai_full_ms(openai_reply_timings=openai_reply_timings, fallback_openai_full_ms=openai_full_ms),
        post_openai_local_ms=post_openai_local_ms or 0,
        sqlite_ms=sqlite_ms or 0,
        processing_ms=_elapsed_ms(processing_started_at_monotonic) if processing_started_at_monotonic else 0,
        end_to_end_first_useful_ms=_sum_duration_ms(transcription_ms or 0, getattr(openai_reply_timings, "ttfut_ms", None)),
        end_to_end_full_reply_ms=_sum_duration_ms(transcription_ms or 0, _resolve_openai_full_ms(openai_reply_timings=openai_reply_timings, fallback_openai_full_ms=openai_full_ms)),
    )


def _elapsed_ms(started_at_monotonic: float) -> int:
    return int(round((time.perf_counter() - started_at_monotonic) * 1000))


def _sum_duration_ms(*parts: int | None) -> int | None:
    total = 0
    for part in parts:
        if part is None:
            return None
        total += part
    return total


def _resolve_openai_full_ms(*, openai_reply_timings: object | None, fallback_openai_full_ms: int | None) -> int:
    if openai_reply_timings is not None:
        full_ms = getattr(openai_reply_timings, "full_ms", None)
        if isinstance(full_ms, int) and full_ms >= 0:
            return full_ms
    return fallback_openai_full_ms or 0


def _normalize_audio_recorded_at(value: str | datetime) -> str:
    if isinstance(value, datetime):
        normalized_datetime = value
    else:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("audio_recorded_at must not be empty.")
        try:
            normalized_datetime = datetime.fromisoformat(normalized_value)
        except ValueError as exc:
            raise ValueError("audio_recorded_at must be an ISO-8601 string or datetime.") from exc

    if normalized_datetime.tzinfo is None:
        normalized_datetime = normalized_datetime.replace(tzinfo=UTC)
    else:
        normalized_datetime = normalized_datetime.astimezone(UTC)
    return normalized_datetime.isoformat(timespec="seconds")
