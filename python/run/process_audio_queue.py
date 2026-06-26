#!/usr/bin/env python3

"""Thin entry point for queued audio processing orchestration.

Usage examples:
- default run: streams reply text to the current terminal while OpenAI is generating it
- `--stream-openai=false`: disable live reply streaming and print the final two-column row only
- `--debug`: print compact per-stage timings after each processed audio file
- `--debug-openai`: enable detailed OpenAI request diagnostics from the reply generator module
- `--debug --debug-openai`: combine stage timings with detailed OpenAI diagnostics
"""

import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.audio_processing_orchestrator import AudioProcessingFailedError, process_one_ready_audio
from modules.terminal_output import (
    build_terminal_block_separator_printer,
    build_terminal_reply_stream_writer,
    build_terminal_stream_start_printer,
    format_terminal_block_separator,
    format_processed_audio_row,
    format_streaming_output_header,
    parse_bool_cli_value,
)


def format_processing_timings(timings) -> str:
    parts = [
        f"queue_lookup={timings.queue_lookup_ms}ms",
        f"transcribe={timings.transcription_ms}ms",
        f"openai_full={timings.openai_full_ms}ms",
        f"post_openai_local={timings.post_openai_local_ms}ms",
        f"sqlite={timings.sqlite_ms}ms",
        f"finalize={timings.finalize_ms}ms",
        f"processing={timings.processing_ms}ms",
    ]
    if timings.openai_ttft_ms is not None:
        parts.insert(2, f"openai_ttft={timings.openai_ttft_ms}ms")
    if timings.openai_ttfut_ms is not None:
        insert_index = 3 if timings.openai_ttft_ms is not None else 2
        parts.insert(insert_index, f"openai_ttfut={timings.openai_ttfut_ms}ms")
    if timings.queue_wait_ms is not None:
        parts.append(f"queue_wait={timings.queue_wait_ms}ms")
    if timings.end_to_end_first_useful_ms is not None:
        parts.append(f"end_to_end_first_useful={timings.end_to_end_first_useful_ms}ms")
    if timings.end_to_end_full_reply_ms is not None:
        parts.append(f"end_to_end_full_reply={timings.end_to_end_full_reply_ms}ms")
    return "[timing] " + " ".join(parts)


def build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Process queued utterance WAV files into transcript, reply, and SQLite turns.")
    parser.add_argument(
        "--queue-dir",
        default=str(PROJECT_ROOT / "artifacts" / "voice-activated"),
        help="Queue root directory that contains inbox/processing/done/failed.",
    )
    parser.add_argument(
        "--database-path",
        help="Optional SQLite database path. Defaults to the repository database directory.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Sleep interval while the queue is empty or after a failed attempt.")
    parser.add_argument("--once", action="store_true", help="Process at most one ready file and then exit.")
    parser.add_argument("--language", help="Optional Whisper language code such as en or ru.")
    parser.add_argument("--whisper-cli-path", help="Optional explicit path to the whisper.cpp CLI binary.")
    parser.add_argument("--whisper-model-path", help="Optional explicit path to the Whisper model file.")
    parser.add_argument("--ffmpeg-path", help="Optional explicit path to the ffmpeg binary.")
    parser.add_argument("--use-gpu", action="store_true", help="Enable whisper.cpp GPU inference.")
    parser.add_argument("--openai-api-key", help="Optional explicit OpenAI API key.")
    parser.add_argument("--openai-model", help="Optional explicit OpenAI model name.")
    parser.add_argument("--openai-instructions", help="Optional OpenAI system instructions for reply generation.")
    parser.add_argument("--openai-dotenv-path", help="Optional explicit path to the .env file for OpenAI settings.")
    parser.add_argument("--debug", action="store_true", help="Print per-stage timing diagnostics.")
    parser.add_argument("--debug-openai", action="store_true", help="Enable detailed OpenAI request timing diagnostics.")
    parser.add_argument(
        "--stream-openai",
        type=parse_bool_cli_value,
        nargs="?",
        const=True,
        default=True,
        help="Stream reply text to the terminal while OpenAI is generating it. Defaults to true; set --stream-openai=false to disable.",
    )
    return parser


def build_process_one_ready_audio_kwargs(
    args,
    *,
    openai_reply_started_callback=None,
    openai_text_delta_callback=None,
) -> dict[str, object]:
    return {
        "database_path": args.database_path,
        "whisper_cli_path": args.whisper_cli_path,
        "whisper_model_path": args.whisper_model_path,
        "ffmpeg_path": args.ffmpeg_path,
        "language": args.language,
        "use_gpu": args.use_gpu,
        "openai_api_key": args.openai_api_key,
        "openai_model": args.openai_model,
        "openai_instructions": args.openai_instructions,
        "openai_dotenv_path": args.openai_dotenv_path,
        "openai_debug": args.debug_openai,
        "openai_stream": args.stream_openai,
        "openai_reply_started_callback": openai_reply_started_callback,
        "openai_text_delta_callback": openai_text_delta_callback,
        "measure_timings": args.debug,
    }


def configure_logging(*, debug_openai: bool) -> None:
    if debug_openai:
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(debug_openai=args.debug_openai)

    try:
        block_separator_printer = build_terminal_block_separator_printer()
        while True:
            try:
                raw_stream_start_printer = build_terminal_stream_start_printer() if args.stream_openai else None
                stream_writer = build_terminal_reply_stream_writer() if args.stream_openai else None
                stream_start_printer = None
                if raw_stream_start_printer is not None:
                    def start_stream(remote_text: str) -> None:
                        block_separator_printer()
                        raw_stream_start_printer(remote_text)

                    setattr(start_stream, "has_started_output", raw_stream_start_printer.has_started_output)
                    stream_start_printer = start_stream
                result = process_one_ready_audio(
                    args.queue_dir,
                    **build_process_one_ready_audio_kwargs(
                        args,
                        openai_reply_started_callback=stream_start_printer,
                        openai_text_delta_callback=stream_writer,
                    ),
                )
            except AudioProcessingFailedError as exc:
                print(exc)
                if args.once:
                    raise SystemExit(1)
                time.sleep(args.poll_interval)
                continue

            if result is None:
                if args.once:
                    raise SystemExit(0)
                time.sleep(args.poll_interval)
                continue

            if args.stream_openai:
                if stream_start_printer is not None and not stream_start_printer.has_started_output():
                    stream_start_printer(result.transcript)
                if stream_writer is not None and not stream_writer.has_written_output():
                    print(result.suggested_reply, end="", flush=True)
                print()
            else:
                block_separator_printer()
                print(format_processed_audio_row(result.transcript, result.suggested_reply))
            if args.debug and result.timings is not None:
                print(format_processing_timings(result.timings))
            print()
            if args.once:
                raise SystemExit(0)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(0)
