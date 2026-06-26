from __future__ import annotations

import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_runtime_adapters import (
    _load_appkit,
    _load_quartz,
    _match_hotkey_event,
    _match_quartz_hotkey_event,
    _normalize_event_modifiers,
    _normalize_quartz_event_modifiers,
    _parse_window_bounds,
    _parse_quartz_hotkey_mapping,
    _parse_hotkey_binding,
    build_macos_task_screenshot_adapter,
)


class TaskRuntimeAdaptersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_screenshot_adapter_returns_none_when_capture_is_cancelled_without_file(self) -> None:
        adapter = build_macos_task_screenshot_adapter(temp_dir=self.root_dir)

        def fake_run(cmd, **_kwargs):
            executable = cmd[0]
            if executable == "/usr/bin/osascript":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="10,20,300,400\n", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="cancelled")

        with patch("modules.task_runtime_adapters._load_appkit", return_value=FakeAppKitWorkspaceModule()), patch(
            "modules.task_runtime_adapters.subprocess.run",
            side_effect=fake_run,
        ):
            result = adapter()

        self.assertIsNone(result)
        self.assertEqual([], list(self.root_dir.iterdir()))

    def test_screenshot_adapter_removes_empty_file_and_returns_none(self) -> None:
        adapter = build_macos_task_screenshot_adapter(temp_dir=self.root_dir)

        def fake_run(cmd, **_kwargs):
            executable = cmd[0]
            if executable == "/usr/bin/osascript":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="10,20,300,400\n", stderr="")
            screenshot_path = Path(cmd[-1])
            screenshot_path.write_bytes(b"")
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="cancelled")

        with patch("modules.task_runtime_adapters._load_appkit", return_value=FakeAppKitWorkspaceModule()), patch(
            "modules.task_runtime_adapters.subprocess.run",
            side_effect=fake_run,
        ):
            result = adapter()

        self.assertIsNone(result)
        self.assertEqual([], list(self.root_dir.iterdir()))

    def test_screenshot_adapter_returns_created_file_path_on_success(self) -> None:
        adapter = build_macos_task_screenshot_adapter(temp_dir=self.root_dir)
        commands: list[list[str]] = []

        def fake_run(cmd, **_kwargs):
            commands.append(list(cmd))
            if cmd[0] == "/usr/bin/osascript":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="10,20,300,400\n", stderr="")
            output_path = Path(cmd[-1])
            if cmd[0] == "/usr/sbin/screencapture":
                output_path.write_bytes(b"png-bytes")
            else:
                output_path.write_bytes(b"webp-bytes")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch("modules.task_runtime_adapters._load_appkit", return_value=FakeAppKitWorkspaceModule()), patch(
            "modules.task_runtime_adapters._load_quartz",
            return_value=FakeQuartzDisplayModule(),
        ), patch(
            "modules.task_runtime_adapters.subprocess.run",
            side_effect=fake_run,
        ):
            result = adapter()

        self.assertIsNotNone(result)
        resolved_result = Path(result).resolve()
        self.assertTrue(resolved_result.exists())
        self.assertEqual(b"webp-bytes", resolved_result.read_bytes())
        self.assertEqual(".webp", resolved_result.suffix)
        self.assertEqual(
            ["/usr/sbin/screencapture", "-x", "-R", "0,0,1440,900", str(resolved_result.with_suffix(".png"))],
            commands[0],
        )
        self.assertEqual(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(resolved_result.with_suffix(".png")),
                "-c:v",
                "libwebp",
                "-lossless",
                "1",
                "-compression_level",
                "6",
                "-quality",
                "100",
                str(resolved_result),
            ],
            commands[-1],
        )
        self.assertFalse(resolved_result.with_suffix(".png").exists())

    def test_screenshot_adapter_falls_back_to_main_display_when_frontmost_window_bounds_are_unavailable(self) -> None:
        adapter = build_macos_task_screenshot_adapter(temp_dir=self.root_dir)
        commands: list[list[str]] = []

        def fake_run(cmd, **_kwargs):
            commands.append(list(cmd))
            if cmd[0] == "/usr/bin/osascript":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            output_path = Path(cmd[-1])
            if cmd[0] == "/usr/sbin/screencapture":
                output_path.write_bytes(b"png-bytes")
            else:
                output_path.write_bytes(b"webp-bytes")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch("modules.task_runtime_adapters._load_appkit", return_value=FakeAppKitWorkspaceModule()), patch(
            "modules.task_runtime_adapters._load_quartz",
            return_value=FakeQuartzDisplayModule(),
        ), patch(
            "modules.task_runtime_adapters.subprocess.run",
            side_effect=fake_run,
        ):
            result = adapter()

        self.assertIsNotNone(result)
        self.assertEqual(
            ["/usr/sbin/screencapture", "-x", "-R", "0,0,1440,900", str(Path(result).resolve().with_suffix(".png"))],
            commands[0],
        )

    def test_screenshot_adapter_returns_png_when_lossless_webp_conversion_fails(self) -> None:
        adapter = build_macos_task_screenshot_adapter(temp_dir=self.root_dir)

        def fake_run(cmd, **_kwargs):
            output_path = Path(cmd[-1])
            if cmd[0] == "/usr/bin/osascript":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="10,20,300,400\n", stderr="")
            if cmd[0] == "/usr/sbin/screencapture":
                output_path.write_bytes(b"png-bytes")
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            output_path.unlink(missing_ok=True)
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="ffmpeg failed")

        with patch("modules.task_runtime_adapters._load_appkit", return_value=FakeAppKitWorkspaceModule()), patch(
            "modules.task_runtime_adapters._load_quartz",
            return_value=FakeQuartzDisplayModule(),
        ), patch(
            "modules.task_runtime_adapters.subprocess.run",
            side_effect=fake_run,
        ):
            result = adapter()

        self.assertIsNotNone(result)
        resolved_result = Path(result).resolve()
        self.assertEqual(".png", resolved_result.suffix)
        self.assertEqual(b"png-bytes", resolved_result.read_bytes())

    def test_screenshot_adapter_returns_none_when_frontmost_window_and_main_display_bounds_are_unavailable(self) -> None:
        adapter = build_macos_task_screenshot_adapter(temp_dir=self.root_dir)

        with patch("modules.task_runtime_adapters._load_appkit", return_value=FakeAppKitWorkspaceModule()), patch(
            "modules.task_runtime_adapters._load_quartz",
            return_value=FakeQuartzDisplayModule(bounds=FakeCGRect(0, 0, 0, 0)),
        ), patch(
            "modules.task_runtime_adapters.subprocess.run",
            return_value=subprocess.CompletedProcess(args=["/usr/bin/osascript"], returncode=0, stdout="", stderr=""),
        ) as subprocess_run:
            result = adapter()

        self.assertIsNone(result)
        self.assertEqual(0, subprocess_run.call_count)
        self.assertEqual([], list(self.root_dir.iterdir()))

    def test_parse_hotkey_binding_normalizes_modifiers_and_key(self) -> None:
        self.assertEqual(
            (frozenset({"command", "shift"}), "1"),
            _parse_hotkey_binding("cmd+shift+1"),
        )

    def test_parse_hotkey_binding_normalizes_command_arrow_down_alias(self) -> None:
        self.assertEqual(
            (frozenset({"command"}), "down"),
            _parse_hotkey_binding("cmd+ArrowDown"),
        )

    def test_parse_hotkey_binding_rejects_multiple_non_modifier_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one non-modifier key"):
            _parse_hotkey_binding("cmd+a+b")

    def test_normalize_event_modifiers_keeps_only_supported_modifier_flags(self) -> None:
        appkit = FakeAppKitModule()

        normalized = _normalize_event_modifiers(
            appkit=appkit,
            event_modifiers=(
                appkit.NSEventModifierFlagCommand
                | appkit.NSEventModifierFlagShift
                | FakeAppKitModule.UNRELATED_FLAG
            ),
        )

        self.assertEqual(frozenset({"command", "shift"}), normalized)

    def test_match_hotkey_event_returns_matching_action(self) -> None:
        appkit = FakeAppKitModule()
        parsed_mapping = {
            "capture": (frozenset({"command", "shift"}), "1"),
            "solve": (frozenset({"command", "shift"}), "2"),
        }
        event = FakeHotkeyEvent(
            characters="2",
            modifier_flags=appkit.NSEventModifierFlagCommand | appkit.NSEventModifierFlagShift,
        )

        action = _match_hotkey_event(
            event=event,
            parsed_mapping=parsed_mapping,
            appkit=appkit,
        )

        self.assertEqual("solve", action)

    def test_match_hotkey_event_maps_command_arrow_down_function_key_to_capture_action(self) -> None:
        appkit = FakeAppKitModule()
        parsed_mapping = {
            "capture": (frozenset({"command"}), "down"),
        }
        event = FakeHotkeyEvent(
            characters="\uf701",
            modifier_flags=appkit.NSEventModifierFlagCommand,
        )

        action = _match_hotkey_event(
            event=event,
            parsed_mapping=parsed_mapping,
            appkit=appkit,
        )

        self.assertEqual("capture", action)

    def test_parse_quartz_hotkey_mapping_translates_binding_to_modifier_flags_and_keycode(self) -> None:
        quartz = FakeQuartzModule()

        parsed_mapping = _parse_quartz_hotkey_mapping(
            {"capture": (frozenset({"command"}), "down")},
            quartz=quartz,
        )

        self.assertEqual(
            {"capture": (quartz.kCGEventFlagMaskCommand, 125)},
            parsed_mapping,
        )

    def test_match_quartz_hotkey_event_returns_matching_action(self) -> None:
        quartz = FakeQuartzModule()
        parsed_mapping = {
            "capture": (quartz.kCGEventFlagMaskCommand, 125),
        }
        event = FakeQuartzEvent(keycode=125, flags=quartz.kCGEventFlagMaskCommand)

        action = _match_quartz_hotkey_event(
            event=event,
            parsed_mapping=parsed_mapping,
            quartz=quartz,
        )

        self.assertEqual("capture", action)

    def test_normalize_quartz_event_modifiers_keeps_only_supported_flags(self) -> None:
        quartz = FakeQuartzModule()

        normalized = _normalize_quartz_event_modifiers(
            quartz=quartz,
            event_flags=quartz.kCGEventFlagMaskCommand | quartz.UNRELATED_FLAG,
        )

        self.assertEqual(quartz.kCGEventFlagMaskCommand, normalized)

    def test_parse_window_bounds_returns_none_for_invalid_shape(self) -> None:
        self.assertIsNone(_parse_window_bounds("10,20,30"))

    def test_parse_window_bounds_returns_expected_tuple(self) -> None:
        self.assertEqual((10, 20, 300, 400), _parse_window_bounds("10, 20, 300, 400"))

    def test_load_appkit_maps_import_error_to_runtime_error(self) -> None:
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "AppKit":
                raise ImportError("missing AppKit")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "PyObjC AppKit is required"):
                _load_appkit()

    def test_load_appkit_returns_imported_module(self) -> None:
        fake_module = types.SimpleNamespace(name="AppKit")

        with patch.dict(sys.modules, {"AppKit": fake_module}):
            self.assertIs(fake_module, _load_appkit())

    def test_load_quartz_maps_import_error_to_runtime_error(self) -> None:
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "Quartz":
                raise ImportError("missing Quartz")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "Install pyobjc-framework-Quartz"):
                _load_quartz()

    def test_load_quartz_returns_imported_module(self) -> None:
        fake_module = types.SimpleNamespace(name="Quartz")

        with patch.dict(sys.modules, {"Quartz": fake_module}):
            self.assertIs(fake_module, _load_quartz())


class FakeAppKitModule:
    NSEventModifierFlagCommand = 1 << 0
    NSEventModifierFlagShift = 1 << 1
    NSEventModifierFlagControl = 1 << 2
    NSEventModifierFlagOption = 1 << 3
    UNRELATED_FLAG = 1 << 8
    NSEvent = None

    def __init__(self) -> None:
        self.NSEvent = FakeNSEvent()


class FakeAppKitWorkspaceModule:
    NSWorkspace = None
    NSEvent = None

    def __init__(self) -> None:
        self.NSWorkspace = FakeNSWorkspace()
        self.NSEvent = FakeNSEvent()


class FakeNSWorkspace:
    def sharedWorkspace(self):
        return self

    def frontmostApplication(self):
        return FakeFrontmostApplication()


class FakeFrontmostApplication:
    def processIdentifier(self) -> int:
        return 4242


class FakeQuartzModule:
    kCGKeyboardEventKeycode = 9
    kCGEventFlagMaskCommand = 1 << 20
    kCGEventFlagMaskShift = 1 << 17
    kCGEventFlagMaskControl = 1 << 18
    kCGEventFlagMaskAlternate = 1 << 19
    UNRELATED_FLAG = 1 << 8

    @staticmethod
    def CGEventGetIntegerValueField(event, _field):
        return event.keycode

    @staticmethod
    def CGEventGetFlags(event):
        return event.flags


class FakeQuartzDisplayModule(FakeQuartzModule):
    def __init__(self, *, bounds=None) -> None:
        self._bounds = bounds or FakeCGRect(0, 0, 1440, 900)

    def CGGetActiveDisplayList(self, _max_displays, _displays, _display_count):
        return (0, (101,), 1)

    def CGMainDisplayID(self):
        return 101

    def CGDisplayBounds(self, _display_id):
        return self._bounds


class FakeQuartzEvent:
    def __init__(self, *, keycode: int, flags: int) -> None:
        self.keycode = keycode
        self.flags = flags


class FakeHotkeyEvent:
    def __init__(self, *, characters: str | None, modifier_flags: int) -> None:
        self._characters = characters
        self._modifier_flags = modifier_flags

    def charactersIgnoringModifiers(self):
        return self._characters

    def modifierFlags(self) -> int:
        return self._modifier_flags


class FakeNSEvent:
    @staticmethod
    def mouseLocation():
        return types.SimpleNamespace(x=100, y=100)


class FakeCGRect:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.origin = types.SimpleNamespace(x=x, y=y)
        self.size = types.SimpleNamespace(width=width, height=height)


if __name__ == "__main__":
    unittest.main()
