from __future__ import annotations

"""Orchestrate one queued audio file through transcription, reply generation, and persistence."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import time

from modules.captured_audio_queue import (
    claim_oldest_ready_audio,
    mark_audio_processing_failed,
    mark_audio_processing_succeeded,
    parse_conversation_id_from_captured_audio_filename,
)
from modules.openai_conversation_reply_generator import generate_reply_in_openai_conversation
from modules.sqlite_conversation_store import (
    ConversationTurnRecord,
    add_conversation_turn,
)
from modules.whisper_cpp_transcriber import transcribe_audio_file


@dataclass(frozen=True)
class ProcessedAudioRecord:
    processing_path: Path
    final_path: Path
    transcript: str
    suggested_reply: str
    conversation_id: int
    turn_id: int
    audio_filename: str
    timings: AudioProcessingTimings | None


@dataclass(frozen=True)
class AudioProcessingTimings:
    queue_lookup_ms: int
    queue_wait_ms: int | None
    transcription_ms: int
    openai_ttft_ms: int | None
    openai_ttfut_ms: int | None
    openai_full_ms: int
    post_openai_local_ms: int
    sqlite_ms: int
    finalize_ms: int
    processing_ms: int
    end_to_end_first_useful_ms: int | None
    end_to_end_full_reply_ms: int | None


class AudioProcessingFailedError(RuntimeError):
    def __init__(
        self,
        *,
        processing_path: Path,
        failed_path: Path | None,
        original_error: Exception,
        mark_failed_error: Exception | None = None,
    ) -> None:
        message = f"Audio processing failed for {processing_path.name}: {original_error}"
        if mark_failed_error is not None:
            message = f"{message}. Also failed to move file to failed/: {mark_failed_error}"
        super().__init__(message)
        self.processing_path = processing_path
        self.failed_path = failed_path
        self.original_error = original_error
        self.mark_failed_error = mark_failed_error


def process_one_ready_audio(
    queue_dir: str | Path,
    *,
    database_path: str | Path | None = None,
    transcribe_ready_audio: Callable[[Path], str] | None = None,
    generate_reply_for_text: Callable[[int, str], str] | None = None,
    whisper_cli_path: str | Path | None = None,
    whisper_model_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
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
    openai_reply_started_callback: Callable[[str], None] | None = None,
    openai_text_delta_callback: Callable[[str], None] | None = None,
    measure_timings: bool = False,
) -> ProcessedAudioRecord | None:
    claim_started_at_monotonic = time.perf_counter() if measure_timings else None
    processing_path = claim_oldest_ready_audio(queue_dir)
    queue_lookup_ms = _elapsed_ms(claim_started_at_monotonic) if claim_started_at_monotonic is not None else None
    if processing_path is None:
        return None

    transcribe_audio = transcribe_ready_audio or (
        lambda audio_path: transcribe_audio_file(
            audio_path,
            whisper_cli_path=whisper_cli_path,
            model_path=whisper_model_path,
            ffmpeg_path=ffmpeg_path,
            language=language,
            use_gpu=use_gpu,
        )
    )

    try:
        processing_started_at_monotonic = time.perf_counter() if measure_timings else None
        conversation_id = parse_conversation_id_from_captured_audio_filename(processing_path.name)
        recorded_at = _parse_recorded_at_from_filename(processing_path.name) if measure_timings else None
        claimed_at = datetime.now(tz=UTC) if measure_timings else None

        transcription_started_at_monotonic = time.perf_counter() if measure_timings else None
        transcript = transcribe_audio(processing_path)
        transcription_ms = _elapsed_ms(transcription_started_at_monotonic) if transcription_started_at_monotonic is not None else None

        if generate_reply_for_text is None:
            openai_started_at_monotonic = time.perf_counter() if measure_timings else None
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
            openai_started_at_monotonic = time.perf_counter() if measure_timings else None
            suggested_reply = generate_reply_for_text(conversation_id, transcript)
            openai_reply_timings = None
        openai_full_ms = _elapsed_ms(openai_started_at_monotonic) if openai_started_at_monotonic is not None else None
        post_openai_started_at_monotonic = time.perf_counter() if measure_timings else None

        sqlite_started_at_monotonic = time.perf_counter() if measure_timings else None
        created_turn = add_conversation_turn(
            conversation_id=conversation_id,
            remote_text=transcript,
            reply_text_suggest=suggested_reply,
            audio_filename=processing_path.name,
            database_path=database_path,
        )
        sqlite_ms = _elapsed_ms(sqlite_started_at_monotonic) if sqlite_started_at_monotonic is not None else None

        finalize_started_at_monotonic = time.perf_counter() if measure_timings else None
        final_path = mark_audio_processing_succeeded(processing_path)
        finalize_ms = _elapsed_ms(finalize_started_at_monotonic) if finalize_started_at_monotonic is not None else None
        post_openai_local_ms = (
            _elapsed_ms(post_openai_started_at_monotonic) if post_openai_started_at_monotonic is not None else None
        )
        finished_at = datetime.now(tz=UTC) if measure_timings else None
    except Exception as exc:
        failed_path = None
        mark_failed_error = None
        try:
            failed_path = mark_audio_processing_failed(processing_path)
        except Exception as mark_exc:
            mark_failed_error = mark_exc
        raise AudioProcessingFailedError(
            processing_path=processing_path,
            failed_path=failed_path,
            original_error=exc,
            mark_failed_error=mark_failed_error,
        ) from exc

    return _build_processed_audio_record(
        processing_path=processing_path,
        final_path=final_path,
        transcript=transcript,
        suggested_reply=suggested_reply,
        created_turn=created_turn,
        timings=_build_timings(
            measure_timings=measure_timings,
            queue_lookup_ms=queue_lookup_ms,
            recorded_at=recorded_at,
            claimed_at=claimed_at,
            transcription_ms=transcription_ms,
            openai_reply_timings=openai_reply_timings,
            openai_full_ms=openai_full_ms,
            post_openai_local_ms=post_openai_local_ms,
            sqlite_ms=sqlite_ms,
            finalize_ms=finalize_ms,
            processing_started_at_monotonic=processing_started_at_monotonic,
            finished_at=finished_at,
        ),
    )


def _build_processed_audio_record(
    *,
    processing_path: Path,
    final_path: Path,
    transcript: str,
    suggested_reply: str,
    created_turn: ConversationTurnRecord,
    timings: AudioProcessingTimings | None,
) -> ProcessedAudioRecord:
    return ProcessedAudioRecord(
        processing_path=processing_path,
        final_path=final_path,
        transcript=transcript,
        suggested_reply=suggested_reply,
        conversation_id=created_turn.conversation_id,
        turn_id=created_turn.turn_id,
        audio_filename=created_turn.audio_filename,
        timings=timings,
    )


def _elapsed_ms(started_at_monotonic: float) -> int:
    return int(round((time.perf_counter() - started_at_monotonic) * 1000))


def _build_timings(
    *,
    measure_timings: bool,
    queue_lookup_ms: int | None,
    recorded_at: datetime | None,
    claimed_at: datetime | None,
    transcription_ms: int | None,
    openai_reply_timings: object | None,
    openai_full_ms: int | None,
    post_openai_local_ms: int | None,
    sqlite_ms: int | None,
    finalize_ms: int | None,
    processing_started_at_monotonic: float | None,
    finished_at: datetime | None,
) -> AudioProcessingTimings | None:
    if not measure_timings:
        return None

    return AudioProcessingTimings(
        queue_lookup_ms=queue_lookup_ms or 0,
        queue_wait_ms=_duration_ms(recorded_at, claimed_at),
        transcription_ms=transcription_ms or 0,
        openai_ttft_ms=getattr(openai_reply_timings, "ttft_ms", None),
        openai_ttfut_ms=getattr(openai_reply_timings, "ttfut_ms", None),
        openai_full_ms=_resolve_openai_full_ms(openai_reply_timings=openai_reply_timings, fallback_openai_full_ms=openai_full_ms),
        post_openai_local_ms=post_openai_local_ms or 0,
        sqlite_ms=sqlite_ms or 0,
        finalize_ms=finalize_ms or 0,
        processing_ms=_elapsed_ms(processing_started_at_monotonic) if processing_started_at_monotonic is not None else 0,
        end_to_end_first_useful_ms=_sum_duration_ms(
            _duration_ms(recorded_at, claimed_at),
            transcription_ms or 0,
            getattr(openai_reply_timings, "ttfut_ms", None),
        ),
        end_to_end_full_reply_ms=_sum_duration_ms(
            _duration_ms(recorded_at, claimed_at),
            transcription_ms or 0,
            _resolve_openai_full_ms(openai_reply_timings=openai_reply_timings, fallback_openai_full_ms=openai_full_ms),
        ),
    )


def _duration_ms(started_at: datetime | None, finished_at: datetime | None) -> int | None:
    if started_at is None or finished_at is None:
        return None
    return int(round((finished_at - started_at).total_seconds() * 1000))


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


def _parse_recorded_at_from_filename(audio_filename: str) -> datetime | None:
    filename_stem = Path(audio_filename).stem
    parts = filename_stem.split("-")
    if len(parts) < 4:
        return None

    date_part = parts[-3]
    time_part = parts[-2]
    micros_part = parts[-1]

    try:
        parsed_datetime = datetime.strptime(
            f"{date_part}{time_part}{micros_part}",
            "%Y%m%d%H%M%S%f",
        )
    except ValueError:
        return None

    return parsed_datetime.replace(tzinfo=UTC)
