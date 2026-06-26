from __future__ import annotations

"""Terminal formatting helpers shared by audio processing runners."""

import shutil
import sys
import textwrap

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_BOLD_OFF = "\033[22m"
_ANSI_WHITE = "\033[37m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_BLOCK_SEPARATOR = "────────────────────"


def format_processed_audio_row(
    remote_text: str,
    suggested_reply: str,
    *,
    terminal_width: int | None = None,
) -> str:
    width = max(60, terminal_width or shutil.get_terminal_size(fallback=(120, 24)).columns)
    separator = " | "
    column_width = max(20, (width - len(separator)) // 2)
    left_lines = _wrap_terminal_column(remote_text, column_width)
    right_lines = _wrap_terminal_column(suggested_reply, column_width)
    row_count = max(len(left_lines), len(right_lines))
    lines: list[str] = []

    for index in range(row_count):
        left_line = left_lines[index] if index < len(left_lines) else ""
        right_line = right_lines[index] if index < len(right_lines) else ""
        lines.append(f"{left_line:<{column_width}}{separator}{right_line:<{column_width}}".rstrip())

    return "\n".join(lines)


def format_streaming_output_header(remote_text: str, *, use_color: bool | None = None) -> str:
    normalized_remote_text = " ".join(remote_text.split()) or "-"
    remote_line = _format_terminal_text(
        f"{_format_terminal_label('Remote:', _ANSI_CYAN, use_color=use_color)} {normalized_remote_text}",
        _ANSI_CYAN,
        use_color=use_color,
    )
    reply_label = _format_terminal_label("Reply:", _ANSI_GREEN, use_color=use_color)
    return f"{remote_line}\n\n{reply_label} "


def build_terminal_reply_stream_writer(*, use_color: bool | None = None):
    has_written = False

    def write_delta(delta: str) -> None:
        nonlocal has_written
        if not delta:
            return
        print(_format_terminal_text(delta, _ANSI_GREEN, use_color=use_color), end="", flush=True)
        has_written = True

    setattr(write_delta, "has_written_output", lambda: has_written)
    return write_delta


def build_terminal_block_separator_printer(*, use_color: bool | None = None):
    has_printed_block = False

    def print_separator() -> None:
        nonlocal has_printed_block
        if has_printed_block:
            print(format_terminal_block_separator(use_color=use_color))
        has_printed_block = True

    setattr(print_separator, "has_printed_block", lambda: has_printed_block)
    return print_separator


def build_terminal_stream_start_printer(*, use_color: bool | None = None):
    has_started = False

    def start_stream(remote_text: str) -> None:
        nonlocal has_started
        if has_started:
            return
        print(format_streaming_output_header(remote_text, use_color=use_color), end="", flush=True)
        has_started = True

    setattr(start_stream, "has_started_output", lambda: has_started)
    return start_stream


def format_terminal_block_separator(*, use_color: bool | None = None) -> str:
    if not _should_use_terminal_color(use_color):
        return _BLOCK_SEPARATOR
    return f"{_ANSI_WHITE}{_BLOCK_SEPARATOR}{_ANSI_RESET}"


def parse_bool_cli_value(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _wrap_terminal_column(value: str, width: int) -> list[str]:
    normalized = " ".join(value.split())
    if not normalized:
        return ["-"]
    return textwrap.wrap(
        normalized,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def _format_terminal_label(label: str, color_code: str, *, use_color: bool | None) -> str:
    if not _should_use_terminal_color(use_color):
        return label
    return f"{_ANSI_BOLD}{color_code}{label}{_ANSI_BOLD_OFF}"


def _format_terminal_text(value: str, color_code: str, *, use_color: bool | None) -> str:
    if not _should_use_terminal_color(use_color):
        return value
    return f"{color_code}{value}{_ANSI_RESET}"


def _should_use_terminal_color(use_color: bool | None) -> bool:
    if use_color is not None:
        return use_color
    return sys.stdout.isatty()
