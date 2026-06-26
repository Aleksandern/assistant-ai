from __future__ import annotations

"""Contract tests for the whisper.cpp transcription module.

Usage:
- run `python/.venv/bin/python -m unittest python.tests.test_whisper_cpp_transcriber`
- covers the module contract without depending on a real whisper.cpp runtime
"""

import stat
import sys
import tempfile
import unittest
import wave
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.whisper_cpp_transcriber import transcribe_audio_buffer, transcribe_audio_file


class WhisperCppTranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.audio_path = self.root_dir / "sample.mp3"
        self.audio_path.write_bytes(b"fake-audio")
        self.model_path = self.root_dir / "ggml-base.bin"
        self.model_path.write_bytes(b"fake-model")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_transcribe_audio_file_returns_transcript_text(self) -> None:
        ffmpeg_path = self._write_script(
            "fake_ffmpeg.sh",
            """#!/bin/sh
set -eu
for arg in "$@"; do
  output="$arg"
done
printf 'RIFF' > "$output"
""",
        )
        whisper_path = self._write_script(
            "fake_whisper.sh",
            """#!/bin/sh
set -eu
output_prefix=""
language_seen=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-of" ]; then
    output_prefix="$2"
    shift 2
    continue
  fi
  if [ "$1" = "-l" ]; then
    language_seen="$2"
    shift 2
    continue
  fi
  shift
done
printf '%s' "$language_seen" > "${output_prefix}.lang"
printf 'hello transcript' > "${output_prefix}.txt"
""",
        )

        transcript = transcribe_audio_file(
            self.audio_path,
            whisper_cli_path=whisper_path,
            model_path=self.model_path,
            ffmpeg_path=ffmpeg_path,
            language="ru",
        )

        self.assertEqual("hello transcript", transcript)

    def test_transcribe_audio_buffer_returns_transcript_text_via_stdin(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        audio = numpy.array([0.0, 0.25, -0.25, 0.5], dtype=numpy.float32)
        captured: dict[str, object] = {}

        def fake_run(command, check, capture_output, input=None):  # type: ignore[no-untyped-def]
            captured["command"] = command
            captured["input"] = input
            output_prefix = Path(command[command.index("-of") + 1])
            output_prefix.with_suffix(".txt").write_text("buffer transcript", encoding="utf-8")
            return subprocess_completed_process(command)

        with mock.patch("modules.whisper_cpp_transcriber.subprocess.run", side_effect=fake_run):
            transcript = transcribe_audio_buffer(
                audio,
                sample_rate=8000,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
                language="ru",
            )

        self.assertEqual("buffer transcript", transcript)
        command = captured["command"]
        self.assertIn("-f", command)
        self.assertEqual("-", command[command.index("-f") + 1])
        self.assertEqual("ru", command[command.index("-l") + 1])

        stdin_payload = captured["input"]
        self.assertIsInstance(stdin_payload, bytes)
        self.assertGreater(len(stdin_payload), 44)

        with wave.open(BytesIO(stdin_payload), "rb") as wav_file:
            self.assertEqual(1, wav_file.getnchannels())
            self.assertEqual(16000, wav_file.getframerate())
            self.assertEqual(2, wav_file.getsampwidth())
            self.assertGreater(wav_file.getnframes(), 0)

    def test_transcribe_audio_buffer_omits_language_flag_when_language_is_missing(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        audio = numpy.array([0.0, 0.25, -0.25, 0.5], dtype=numpy.float32)
        captured: dict[str, object] = {}

        def fake_run(command, check, capture_output, input=None):  # type: ignore[no-untyped-def]
            captured["command"] = command
            output_prefix = Path(command[command.index("-of") + 1])
            output_prefix.with_suffix(".txt").write_text("buffer transcript", encoding="utf-8")
            return subprocess_completed_process(command)

        with mock.patch("modules.whisper_cpp_transcriber.subprocess.run", side_effect=fake_run):
            transcript = transcribe_audio_buffer(
                audio,
                sample_rate=16000,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
            )

        self.assertEqual("buffer transcript", transcript)
        command = captured["command"]
        self.assertNotIn("-l", command)

    def test_transcribe_audio_buffer_rejects_empty_audio(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        with self.assertRaises(ValueError) as error:
            transcribe_audio_buffer(
                numpy.array([], dtype=numpy.float32),
                sample_rate=16000,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
            )

        self.assertIn("must not be empty", str(error.exception))

    def test_transcribe_audio_buffer_rejects_wrong_dtype(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        with self.assertRaises(TypeError) as error:
            transcribe_audio_buffer(
                numpy.array([0.0, 1.0], dtype=numpy.float64),
                sample_rate=16000,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
            )

        self.assertIn("dtype float32", str(error.exception))

    def test_transcribe_audio_buffer_rejects_non_mono_shape(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        with self.assertRaises(ValueError) as error:
            transcribe_audio_buffer(
                numpy.zeros((2, 3), dtype=numpy.float32),
                sample_rate=16000,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
            )

        self.assertIn("one-dimensional", str(error.exception))

    def test_transcribe_audio_buffer_rejects_invalid_sample_rate(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        with self.assertRaises(ValueError) as error:
            transcribe_audio_buffer(
                numpy.array([0.0, 1.0], dtype=numpy.float32),
                sample_rate=0,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
            )

        self.assertIn("positive integer", str(error.exception))

    def test_transcribe_audio_file_rejects_missing_audio_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            transcribe_audio_file(
                self.root_dir / "missing.wav",
                whisper_cli_path=self.root_dir / "missing-whisper",
                model_path=self.root_dir / "missing-model",
                ffmpeg_path=self.root_dir / "missing-ffmpeg",
            )

    def test_transcribe_audio_file_reports_ffmpeg_failure(self) -> None:
        ffmpeg_path = self._write_script(
            "failing_ffmpeg.sh",
            """#!/bin/sh
echo "bad audio" >&2
exit 7
""",
        )
        whisper_path = self._write_script(
            "fake_whisper.sh",
            """#!/bin/sh
exit 0
""",
        )

        with self.assertRaises(RuntimeError) as error:
            transcribe_audio_file(
                self.audio_path,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
                ffmpeg_path=ffmpeg_path,
            )

        self.assertIn("audio conversion for whisper.cpp failed", str(error.exception))
        self.assertIn("bad audio", str(error.exception))

    def test_transcribe_audio_file_reports_whisper_failure(self) -> None:
        ffmpeg_path = self._write_script(
            "fake_ffmpeg.sh",
            """#!/bin/sh
set -eu
for arg in "$@"; do
  output="$arg"
done
printf 'RIFF' > "$output"
""",
        )
        whisper_path = self._write_script(
            "failing_whisper.sh",
            """#!/bin/sh
echo "model load failed" >&2
exit 9
""",
        )

        with self.assertRaises(RuntimeError) as error:
            transcribe_audio_file(
                self.audio_path,
                whisper_cli_path=whisper_path,
                model_path=self.model_path,
                ffmpeg_path=ffmpeg_path,
            )

        self.assertIn("whisper.cpp transcription failed", str(error.exception))
        self.assertIn("model load failed", str(error.exception))

    def test_transcribe_audio_buffer_reports_whisper_failure(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")

        with mock.patch(
            "modules.whisper_cpp_transcriber.subprocess.run",
            return_value=subprocess_completed_process(
                [str(whisper_path)],
                returncode=9,
                stderr=b"model load failed\n",
            ),
        ):
            with self.assertRaises(RuntimeError) as error:
                transcribe_audio_buffer(
                    numpy.array([0.0, 0.25], dtype=numpy.float32),
                    sample_rate=16000,
                    whisper_cli_path=whisper_path,
                    model_path=self.model_path,
                )

        self.assertIn("whisper.cpp transcription failed", str(error.exception))
        self.assertIn("model load failed", str(error.exception))

    def test_transcribe_audio_buffer_rejects_missing_whisper_binary(self) -> None:
        with self.assertRaises(FileNotFoundError) as error:
            transcribe_audio_buffer(
                numpy.array([0.0, 0.25], dtype=numpy.float32),
                sample_rate=16000,
                whisper_cli_path=self.root_dir / "missing-whisper",
                model_path=self.model_path,
            )

        self.assertIn("whisper.cpp binary was not found", str(error.exception))

    def test_transcribe_audio_buffer_rejects_missing_model(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")

        with self.assertRaises(FileNotFoundError) as error:
            transcribe_audio_buffer(
                numpy.array([0.0, 0.25], dtype=numpy.float32),
                sample_rate=16000,
                whisper_cli_path=whisper_path,
                model_path=self.root_dir / "missing-model",
            )

        self.assertIn("whisper.cpp model was not found", str(error.exception))

    def test_transcribe_audio_buffer_requires_transcript_output_file(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")

        with mock.patch(
            "modules.whisper_cpp_transcriber.subprocess.run",
            return_value=subprocess_completed_process([str(whisper_path)]),
        ):
            with self.assertRaises(RuntimeError) as error:
                transcribe_audio_buffer(
                    numpy.array([0.0, 0.25], dtype=numpy.float32),
                    sample_rate=16000,
                    whisper_cli_path=whisper_path,
                    model_path=self.model_path,
                )

        self.assertIn("did not produce transcript output", str(error.exception))

    def test_transcribe_audio_buffer_relinks_whisper_runtime_to_relative_rpaths(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        captured_commands: list[list[str]] = []

        def fake_run(command, check, capture_output, input=None):  # type: ignore[no-untyped-def]
            command_list = list(command)
            captured_commands.append(command_list)
            if command_list[:2] == ["otool", "-l"]:
                return subprocess_completed_process(
                    command_list,
                    stdout=(
                        "cmd LC_RPATH\n"
                        "path /tmp/old-project/python/libs/whisper.cpp/build/src (offset 12)\n"
                        "cmd LC_RPATH\n"
                        "path /tmp/old-project/python/libs/whisper.cpp/build/ggml/src (offset 12)\n"
                    ).encode("utf-8"),
                )
            if command_list[0] == "install_name_tool":
                return subprocess_completed_process(command_list)

            output_prefix = Path(command_list[command_list.index("-of") + 1])
            output_prefix.with_suffix(".txt").write_text("buffer transcript", encoding="utf-8")
            return subprocess_completed_process(command_list)

        with mock.patch("modules.whisper_cpp_transcriber.os.name", "posix"):
            with mock.patch("modules.whisper_cpp_transcriber._is_macho_binary", return_value=True):
                with mock.patch(
                    "modules.whisper_cpp_transcriber.shutil.which",
                    side_effect=lambda name: name,
                ):
                    with mock.patch("modules.whisper_cpp_transcriber.subprocess.run", side_effect=fake_run):
                        transcript = transcribe_audio_buffer(
                            numpy.array([0.0, 0.25], dtype=numpy.float32),
                            sample_rate=16000,
                            whisper_cli_path=whisper_path,
                            model_path=self.model_path,
                        )

        self.assertEqual("buffer transcript", transcript)
        install_name_tool_commands = [
            command for command in captured_commands if command and command[0] == "install_name_tool"
        ]
        self.assertEqual(1, len(install_name_tool_commands))
        update_command = install_name_tool_commands[0]
        self.assertIn("-delete_rpath", update_command)
        self.assertIn("/tmp/old-project/python/libs/whisper.cpp/build/src", update_command)
        self.assertIn("@executable_path/../src", update_command)
        self.assertIn("@executable_path/../ggml/src", update_command)

    def test_transcribe_audio_buffer_skips_runtime_relink_when_rpaths_are_already_relative(self) -> None:
        whisper_path = self._write_script("fake_whisper.sh", "#!/bin/sh\nexit 0\n")
        captured_commands: list[list[str]] = []

        def fake_run(command, check, capture_output, input=None):  # type: ignore[no-untyped-def]
            command_list = list(command)
            captured_commands.append(command_list)
            if command_list[:2] == ["otool", "-l"]:
                return subprocess_completed_process(
                    command_list,
                    stdout=(
                        "cmd LC_RPATH\n"
                        "path @executable_path/../src (offset 12)\n"
                        "cmd LC_RPATH\n"
                        "path @executable_path/../ggml/src (offset 12)\n"
                        "cmd LC_RPATH\n"
                        "path @executable_path/../ggml/src/ggml-blas (offset 12)\n"
                        "cmd LC_RPATH\n"
                        "path @executable_path/../ggml/src/ggml-metal (offset 12)\n"
                    ).encode("utf-8"),
                )

            output_prefix = Path(command_list[command_list.index("-of") + 1])
            output_prefix.with_suffix(".txt").write_text("buffer transcript", encoding="utf-8")
            return subprocess_completed_process(command_list)

        with mock.patch("modules.whisper_cpp_transcriber.os.name", "posix"):
            with mock.patch("modules.whisper_cpp_transcriber._is_macho_binary", return_value=True):
                with mock.patch(
                    "modules.whisper_cpp_transcriber.shutil.which",
                    side_effect=lambda name: name,
                ):
                    with mock.patch("modules.whisper_cpp_transcriber.subprocess.run", side_effect=fake_run):
                        transcript = transcribe_audio_buffer(
                            numpy.array([0.0, 0.25], dtype=numpy.float32),
                            sample_rate=16000,
                            whisper_cli_path=whisper_path,
                            model_path=self.model_path,
                        )

        self.assertEqual("buffer transcript", transcript)
        self.assertFalse(
            any(command and command[0] == "install_name_tool" for command in captured_commands)
        )

    def _write_script(self, name: str, contents: str) -> Path:
        path = self.root_dir / name
        path.write_text(contents, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path


def subprocess_completed_process(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
):
    return __import__("subprocess").CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
