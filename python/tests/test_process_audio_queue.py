from __future__ import annotations

"""Tests for terminal formatting in the queued audio processing runner."""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import call, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run.process_audio_queue import (
    build_terminal_block_separator_printer,
    build_argument_parser,
    build_process_one_ready_audio_kwargs,
    build_terminal_reply_stream_writer,
    build_terminal_stream_start_printer,
    configure_logging,
    format_terminal_block_separator,
    format_streaming_output_header,
    format_processed_audio_row,
    format_processing_timings,
    parse_bool_cli_value,
)


@dataclass(frozen=True)
class _FakeTimings:
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


class ProcessAudioQueueFormattingTests(unittest.TestCase):
    def test_format_processed_audio_row_renders_two_columns(self) -> None:
        formatted = format_processed_audio_row(
            "Tell me about your experience with Python and APIs.",
            "I have built backend services, APIs, and automation in Python.",
            terminal_width=80,
        )

        lines = formatted.splitlines()

        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(all(" | " in line for line in lines))
        self.assertIn("Tell me about your experience", formatted)
        self.assertIn("I have built backend services", formatted)

    def test_format_processed_audio_row_wraps_long_text_and_uses_placeholder_for_empty_side(self) -> None:
        formatted = format_processed_audio_row(
            "",
            "This is a very long suggested reply that should wrap across multiple visual lines in the terminal output.",
            terminal_width=60,
        )

        lines = formatted.splitlines()

        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].startswith("-"))
        self.assertTrue(all(" | " in line for line in lines))

    def test_format_processing_timings_renders_compact_debug_line(self) -> None:
        formatted = format_processing_timings(
            _FakeTimings(
                queue_lookup_ms=4,
                queue_wait_ms=1200,
                transcription_ms=850,
                openai_ttft_ms=140,
                openai_ttfut_ms=260,
                openai_full_ms=640,
                post_openai_local_ms=11,
                sqlite_ms=8,
                finalize_ms=1,
                processing_ms=1499,
                end_to_end_first_useful_ms=2260,
                end_to_end_full_reply_ms=2699,
            )
        )

        self.assertTrue(formatted.startswith("[timing] "))
        self.assertIn("queue_lookup=4ms", formatted)
        self.assertIn("transcribe=850ms", formatted)
        self.assertIn("openai_ttft=140ms", formatted)
        self.assertIn("openai_ttfut=260ms", formatted)
        self.assertIn("openai_full=640ms", formatted)
        self.assertIn("post_openai_local=11ms", formatted)
        self.assertIn("sqlite=8ms", formatted)
        self.assertIn("finalize=1ms", formatted)
        self.assertIn("processing=1499ms", formatted)
        self.assertIn("queue_wait=1200ms", formatted)
        self.assertIn("end_to_end_first_useful=2260ms", formatted)
        self.assertIn("end_to_end_full_reply=2699ms", formatted)

    def test_format_processing_timings_is_separate_from_main_row_output(self) -> None:
        row = format_processed_audio_row(
            "question",
            "answer",
            terminal_width=60,
        )
        timings = format_processing_timings(
            _FakeTimings(
                queue_lookup_ms=1,
                queue_wait_ms=None,
                transcription_ms=2,
                openai_ttft_ms=None,
                openai_ttfut_ms=None,
                openai_full_ms=3,
                post_openai_local_ms=6,
                sqlite_ms=4,
                finalize_ms=5,
                processing_ms=14,
                end_to_end_first_useful_ms=None,
                end_to_end_full_reply_ms=None,
            )
        )

        self.assertNotIn("[timing]", row)
        self.assertIn("[timing]", timings)

    def test_build_process_one_ready_audio_kwargs_defaults_to_streaming_openai(self) -> None:
        args = build_argument_parser().parse_args(["--openai-model", "gpt-5-nano"])

        kwargs = build_process_one_ready_audio_kwargs(args)

        self.assertFalse(kwargs["measure_timings"])
        self.assertFalse(kwargs["openai_debug"])
        self.assertTrue(kwargs["openai_stream"])
        self.assertEqual("gpt-5-nano", kwargs["openai_model"])

    def test_build_process_one_ready_audio_kwargs_maps_debug_openai_separately_from_stage_timings(self) -> None:
        args = build_argument_parser().parse_args(
            ["--debug", "--debug-openai", "--stream-openai=false", "--openai-model", "gpt-5-nano"]
        )

        kwargs = build_process_one_ready_audio_kwargs(args)

        self.assertTrue(kwargs["measure_timings"])
        self.assertTrue(kwargs["openai_debug"])
        self.assertFalse(kwargs["openai_stream"])
        self.assertEqual("gpt-5-nano", kwargs["openai_model"])

    def test_parse_bool_cli_value_accepts_true_and_false_variants(self) -> None:
        self.assertTrue(parse_bool_cli_value("true"))
        self.assertTrue(parse_bool_cli_value("ON"))
        self.assertFalse(parse_bool_cli_value("false"))
        self.assertFalse(parse_bool_cli_value("0"))

    def test_configure_logging_enables_info_logging_only_for_debug_openai(self) -> None:
        with patch("run.process_audio_queue.logging.basicConfig") as basic_config_mock:
            configure_logging(debug_openai=False)
            basic_config_mock.assert_not_called()

        with patch("run.process_audio_queue.logging.basicConfig") as basic_config_mock:
            configure_logging(debug_openai=True)
            basic_config_mock.assert_called_once_with(level=20)

    def test_format_streaming_output_header_renders_remote_and_reply_labels(self) -> None:
        formatted = format_streaming_output_header("Do you want to live on Mars?", use_color=False)

        self.assertEqual("Remote: Do you want to live on Mars?\n\nReply: ", formatted)

    def test_format_streaming_output_header_can_colorize_remote_line_and_reply_label(self) -> None:
        formatted = format_streaming_output_header("Do you want to live on Mars?", use_color=True)

        self.assertEqual(
            "\033[36m\033[1m\033[36mRemote:\033[22m Do you want to live on Mars?\033[0m\n\n\033[1m\033[32mReply:\033[22m ",
            formatted,
        )

    def test_build_terminal_stream_start_printer_prints_header_only_once(self) -> None:
        printer = build_terminal_stream_start_printer(use_color=False)

        with patch("builtins.print") as print_mock:
            printer("Question one?")
            printer("Question two?")

        print_mock.assert_called_once_with("Remote: Question one?\n\nReply: ", end="", flush=True)
        self.assertTrue(printer.has_started_output())

    def test_format_terminal_block_separator_can_colorize_divider(self) -> None:
        self.assertEqual("────────────────────", format_terminal_block_separator(use_color=False))
        self.assertEqual("\033[37m────────────────────\033[0m", format_terminal_block_separator(use_color=True))

    def test_build_terminal_block_separator_printer_prints_only_between_blocks(self) -> None:
        printer = build_terminal_block_separator_printer(use_color=False)

        with patch("builtins.print") as print_mock:
            printer()
            printer()
            printer()

        print_mock.assert_has_calls(
            [
                call("────────────────────"),
                call("────────────────────"),
            ]
        )
        self.assertTrue(printer.has_printed_block())

    def test_build_terminal_reply_stream_writer_prints_deltas_and_tracks_output(self) -> None:
        writer = build_terminal_reply_stream_writer(use_color=False)

        with patch("builtins.print") as print_mock:
            writer("Hello")
            writer("")
            writer(" world")

        print_mock.assert_has_calls(
            [
                call("Hello", end="", flush=True),
                call(" world", end="", flush=True),
            ]
        )
        self.assertTrue(writer.has_written_output())

    def test_build_terminal_reply_stream_writer_colorizes_reply_chunks(self) -> None:
        writer = build_terminal_reply_stream_writer(use_color=True)

        with patch("builtins.print") as print_mock:
            writer("Hello")
            writer(" world")

        print_mock.assert_has_calls(
            [
                call("\033[32mHello\033[0m", end="", flush=True),
                call("\033[32m world\033[0m", end="", flush=True),
            ]
        )


if __name__ == "__main__":
    unittest.main()
