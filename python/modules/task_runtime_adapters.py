from __future__ import annotations

"""Concrete macOS runtime adapters for task screenshot capture and hotkeys."""

from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
import tempfile
import threading

from modules.task_hotkey_listener import TaskHotkeyBackendAdapter


_LOSSLESS_WEBP_COMPRESSION_LEVEL = "6"
_LOSSLESS_WEBP_QUALITY = "100"


def build_macos_task_screenshot_adapter(*, temp_dir: str | Path | None = None) -> Callable[[], str | Path | None]:
    def capture_screenshot() -> str | Path | None:
        screenshot_dir = Path(temp_dir).expanduser().resolve() if temp_dir is not None else None
        if screenshot_dir is not None:
            screenshot_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix="task-screenshot-",
            dir=screenshot_dir,
            delete=False,
        ) as temp_file:
            screenshot_path = Path(temp_file.name)

        capture_bounds = _resolve_frontmost_display_capture_bounds()
        if capture_bounds is None:
            screenshot_path.unlink(missing_ok=True)
            return None

        completed = subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-R", capture_bounds, str(screenshot_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 and not screenshot_path.exists():
            return None
        if not screenshot_path.exists() or screenshot_path.stat().st_size == 0:
            screenshot_path.unlink(missing_ok=True)
            return None
        return _convert_screenshot_to_lossless_webp(screenshot_path)

    return capture_screenshot


def _convert_screenshot_to_lossless_webp(screenshot_path: Path) -> Path:
    webp_path = screenshot_path.with_suffix(".webp")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(screenshot_path),
            "-c:v",
            "libwebp",
            "-lossless",
            "1",
            "-compression_level",
            _LOSSLESS_WEBP_COMPRESSION_LEVEL,
            "-quality",
            _LOSSLESS_WEBP_QUALITY,
            str(webp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not webp_path.exists() or webp_path.stat().st_size == 0:
        webp_path.unlink(missing_ok=True)
        return screenshot_path

    screenshot_path.unlink(missing_ok=True)
    return webp_path


class MacOSTaskHotkeyBackendAdapter(TaskHotkeyBackendAdapter):
    """Global macOS hotkey monitor backed by a Quartz event tap."""

    def __init__(self) -> None:
        self._event_tap = None
        self._run_loop = None
        self._thread: threading.Thread | None = None

    def start(
        self,
        *,
        hotkey_mapping: dict[str, str] | None,
        on_hotkey: Callable[[str], None],
    ) -> None:
        if self._thread is not None:
            return

        quartz = _load_quartz()
        parsed_mapping = _parse_hotkey_mapping(hotkey_mapping or _default_hotkey_mapping())
        parsed_quartz_mapping = _parse_quartz_hotkey_mapping(parsed_mapping, quartz=quartz)
        ready = threading.Event()
        start_error: list[Exception] = []

        def event_tap_callback(_proxy, event_type, event, _refcon):
            if event_type == quartz.kCGEventTapDisabledByTimeout:
                if self._event_tap is not None:
                    quartz.CGEventTapEnable(self._event_tap, True)
                return event
            if event_type != quartz.kCGEventKeyDown:
                return event

            action = _match_quartz_hotkey_event(
                event=event,
                parsed_mapping=parsed_quartz_mapping,
                quartz=quartz,
            )
            if action is None:
                return event

            on_hotkey(action)
            return None

        def run_event_tap() -> None:
            try:
                event_mask = quartz.CGEventMaskBit(quartz.kCGEventKeyDown)
                event_tap = quartz.CGEventTapCreate(
                    quartz.kCGSessionEventTap,
                    quartz.kCGHeadInsertEventTap,
                    quartz.kCGEventTapOptionDefault,
                    event_mask,
                    event_tap_callback,
                    None,
                )
                if event_tap is None:
                    raise RuntimeError(
                        "Quartz event tap could not be created. Check Input Monitoring/Accessibility permissions."
                    )

                run_loop_source = quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
                run_loop = quartz.CFRunLoopGetCurrent()
                quartz.CFRunLoopAddSource(run_loop, run_loop_source, quartz.kCFRunLoopCommonModes)
                quartz.CGEventTapEnable(event_tap, True)
                self._event_tap = event_tap
                self._run_loop = run_loop
                ready.set()
                quartz.CFRunLoopRun()
            except Exception as exc:
                start_error.append(exc)
                ready.set()
            finally:
                self._event_tap = None
                self._run_loop = None

        thread = threading.Thread(
            target=run_event_tap,
            name="task-hotkey-quartz-event-tap",
            daemon=True,
        )
        thread.start()
        ready.wait(timeout=2.0)
        if start_error:
            thread.join(timeout=1.0)
            raise start_error[0]
        if self._event_tap is None or self._run_loop is None:
            thread.join(timeout=1.0)
            raise RuntimeError("Quartz event tap failed to start.")
        self._thread = thread

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return

        quartz = _load_quartz()
        run_loop = self._run_loop
        if run_loop is not None:
            quartz.CFRunLoopStop(run_loop)
        thread.join(timeout=2.0)
        self._thread = None


def _default_hotkey_mapping() -> dict[str, str]:
    return {
        "capture": "cmd+down",
        "solve": "cmd+shift+2",
        "clear": "cmd+shift+3",
    }


def _parse_hotkey_mapping(hotkey_mapping: Mapping[str, str]) -> dict[str, tuple[frozenset[str], str]]:
    return {
        action: _parse_hotkey_binding(binding)
        for action, binding in hotkey_mapping.items()
    }


def _parse_hotkey_binding(binding: str) -> tuple[frozenset[str], str]:
    normalized_parts = [part.strip().lower() for part in binding.split("+") if part.strip()]
    if not normalized_parts:
        raise ValueError("Task hotkey binding must not be empty.")

    modifiers: set[str] = set()
    key: str | None = None
    for part in normalized_parts:
        if part in {"cmd", "command"}:
            modifiers.add("command")
        elif part == "shift":
            modifiers.add("shift")
        elif part in {"ctrl", "control"}:
            modifiers.add("control")
        elif part in {"alt", "option"}:
            modifiers.add("option")
        elif key is None:
            key = _normalize_hotkey_key(part)
        else:
            raise ValueError(f"Task hotkey binding must contain only one non-modifier key: {binding}")

    if key is None:
        raise ValueError(f"Task hotkey binding must include a key: {binding}")

    return frozenset(modifiers), key


def _match_hotkey_event(*, event, parsed_mapping: dict[str, tuple[frozenset[str], str]], appkit) -> str | None:
    characters = event.charactersIgnoringModifiers()
    if characters is None:
        return None

    normalized_key = _normalize_hotkey_key(str(characters).strip().lower())
    if not normalized_key:
        return None

    event_modifiers = _normalize_event_modifiers(
        appkit=appkit,
        event_modifiers=event.modifierFlags(),
    )
    for action, (required_modifiers, required_key) in parsed_mapping.items():
        if normalized_key == required_key and event_modifiers == required_modifiers:
            return action
    return None


def _normalize_event_modifiers(*, appkit, event_modifiers: int) -> frozenset[str]:
    relevant_flags = (
        appkit.NSEventModifierFlagCommand
        | appkit.NSEventModifierFlagShift
        | appkit.NSEventModifierFlagControl
        | appkit.NSEventModifierFlagOption
    )
    normalized_flags = event_modifiers & relevant_flags

    modifiers: set[str] = set()
    if normalized_flags & appkit.NSEventModifierFlagCommand:
        modifiers.add("command")
    if normalized_flags & appkit.NSEventModifierFlagShift:
        modifiers.add("shift")
    if normalized_flags & appkit.NSEventModifierFlagControl:
        modifiers.add("control")
    if normalized_flags & appkit.NSEventModifierFlagOption:
        modifiers.add("option")
    return frozenset(modifiers)


def _parse_quartz_hotkey_mapping(
    parsed_mapping: Mapping[str, tuple[frozenset[str], str]],
    *,
    quartz,
) -> dict[str, tuple[int, int]]:
    return {
        action: (
            _normalize_quartz_modifier_flags(modifiers, quartz=quartz),
            _resolve_quartz_keycode(key),
        )
        for action, (modifiers, key) in parsed_mapping.items()
    }


def _match_quartz_hotkey_event(*, event, parsed_mapping: dict[str, tuple[int, int]], quartz) -> str | None:
    event_keycode = int(quartz.CGEventGetIntegerValueField(event, quartz.kCGKeyboardEventKeycode))
    event_modifiers = _normalize_quartz_event_modifiers(
        quartz=quartz,
        event_flags=quartz.CGEventGetFlags(event),
    )
    for action, (required_modifiers, required_keycode) in parsed_mapping.items():
        if event_keycode == required_keycode and event_modifiers == required_modifiers:
            return action
    return None


def _normalize_quartz_modifier_flags(modifiers: frozenset[str], *, quartz) -> int:
    normalized_flags = 0
    if "command" in modifiers:
        normalized_flags |= int(quartz.kCGEventFlagMaskCommand)
    if "shift" in modifiers:
        normalized_flags |= int(quartz.kCGEventFlagMaskShift)
    if "control" in modifiers:
        normalized_flags |= int(quartz.kCGEventFlagMaskControl)
    if "option" in modifiers:
        normalized_flags |= int(quartz.kCGEventFlagMaskAlternate)
    return normalized_flags


def _normalize_quartz_event_modifiers(*, quartz, event_flags: int) -> int:
    relevant_flags = (
        int(quartz.kCGEventFlagMaskCommand)
        | int(quartz.kCGEventFlagMaskShift)
        | int(quartz.kCGEventFlagMaskControl)
        | int(quartz.kCGEventFlagMaskAlternate)
    )
    return int(event_flags) & relevant_flags


def _resolve_quartz_keycode(key: str) -> int:
    normalized_key = _normalize_hotkey_key(key)
    if normalized_key not in _QUARTZ_KEYCODES_BY_KEY:
        raise ValueError(f"Task hotkey key is not supported by the Quartz backend: {key}")
    return _QUARTZ_KEYCODES_BY_KEY[normalized_key]


def _load_appkit():
    try:
        import AppKit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyObjC AppKit is required for task hotkeys on macOS.") from exc
    return AppKit


def _load_quartz():
    try:
        import Quartz
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyObjC Quartz is required for task hotkeys on macOS. Install pyobjc-framework-Quartz."
        ) from exc
    return Quartz


def _normalize_hotkey_key(key: str) -> str:
    normalized_key = key.strip().lower()
    return _SPECIAL_HOTKEY_KEY_ALIASES.get(normalized_key, normalized_key)


def _resolve_frontmost_window_capture_bounds() -> str | None:
    frontmost_window_bounds = _read_frontmost_window_bounds()
    if frontmost_window_bounds is None:
        return None

    left, top, width, height = frontmost_window_bounds
    if width <= 0 or height <= 0:
        return None
    return f"{left},{top},{width},{height}"


def _resolve_frontmost_display_capture_bounds() -> str | None:
    pointer_display_bounds = _read_display_bounds_for_pointer()
    if pointer_display_bounds is not None:
        return _format_capture_bounds(pointer_display_bounds)

    main_display_bounds = _read_main_display_bounds()
    if main_display_bounds is None:
        return None
    return _format_capture_bounds(main_display_bounds)


def _read_frontmost_application_process_identifier() -> int | None:
    appkit = _load_appkit()
    workspace = getattr(appkit, "NSWorkspace", None)
    if workspace is None:
        raise RuntimeError("PyObjC AppKit NSWorkspace is required for task screenshots on macOS.")

    frontmost_application = workspace.sharedWorkspace().frontmostApplication()
    if frontmost_application is None:
        return None

    return int(frontmost_application.processIdentifier())


def _read_frontmost_window_bounds() -> tuple[int, int, int, int] | None:
    process_identifier = _read_frontmost_application_process_identifier()
    if process_identifier is None:
        return None

    applescript = """
tell application "System Events"
    tell (first process whose unix id is __PID__)
        if not (exists front window) then
            return ""
        end if
        set {leftPos, topPos} to position of front window
        set {windowWidth, windowHeight} to size of front window
        return (leftPos as string) & "," & (topPos as string) & "," & (windowWidth as string) & "," & (windowHeight as string)
    end tell
end tell
""".replace("__PID__", str(process_identifier))
    completed = subprocess.run(
        ["/usr/bin/osascript", "-e", applescript],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None

    return _parse_window_bounds(completed.stdout)


def _read_display_bounds_for_pointer() -> tuple[int, int, int, int] | None:
    active_display_bounds = _read_active_display_bounds()
    if not active_display_bounds:
        return None

    pointer_location = _read_pointer_location()
    if pointer_location is None:
        return None

    pointer_x, pointer_y = pointer_location

    for display_bounds in active_display_bounds:
        if _bounds_contains_point(display_bounds, x=pointer_x, y=pointer_y):
            return display_bounds

    return None


def _read_active_display_bounds() -> list[tuple[int, int, int, int]]:
    quartz = _load_quartz()
    error_code, display_ids, display_count = quartz.CGGetActiveDisplayList(32, None, None)
    if error_code != 0 or display_count <= 0 or not display_ids:
        return []

    bounds: list[tuple[int, int, int, int]] = []
    for display_id in display_ids[:display_count]:
        display_bounds = _read_display_bounds(display_id, quartz=quartz)
        if display_bounds is not None:
            bounds.append(display_bounds)
    return bounds


def _read_main_display_bounds() -> tuple[int, int, int, int] | None:
    quartz = _load_quartz()
    display_id = quartz.CGMainDisplayID()
    return _read_display_bounds(display_id, quartz=quartz)


def _read_display_bounds(display_id: int, *, quartz) -> tuple[int, int, int, int] | None:
    raw_bounds = quartz.CGDisplayBounds(display_id)
    try:
        left = int(raw_bounds.origin.x)
        top = int(raw_bounds.origin.y)
        width = int(raw_bounds.size.width)
        height = int(raw_bounds.size.height)
    except (AttributeError, TypeError, ValueError):
        return None

    return left, top, width, height


def _read_pointer_location() -> tuple[int, int] | None:
    try:
        appkit = _load_appkit()
    except RuntimeError:
        return None

    event_class = getattr(appkit, "NSEvent", None)
    if event_class is None or not hasattr(event_class, "mouseLocation"):
        return None

    try:
        pointer_location = event_class.mouseLocation()
        return int(pointer_location.x), int(pointer_location.y)
    except (AttributeError, TypeError, ValueError):
        return None


def _bounds_contains_point(bounds: tuple[int, int, int, int], *, x: int, y: int) -> bool:
    left, top, width, height = bounds
    return left <= x < (left + width) and top <= y < (top + height)


def _format_capture_bounds(bounds: tuple[int, int, int, int]) -> str | None:
    left, top, width, height = bounds
    if width <= 0 or height <= 0:
        return None
    return f"{left},{top},{width},{height}"


def _parse_window_bounds(raw_value: str) -> tuple[int, int, int, int] | None:
    normalized_value = raw_value.strip()
    if not normalized_value:
        return None

    parts = [part.strip() for part in normalized_value.split(",")]
    if len(parts) != 4:
        return None

    try:
        left, top, width, height = [int(part) for part in parts]
    except ValueError:
        return None

    return left, top, width, height


_SPECIAL_HOTKEY_KEY_ALIASES: dict[str, str] = {
    "arrowdown": "down",
    "downarrow": "down",
    "\uf701": "down",
}

_QUARTZ_KEYCODES_BY_KEY: dict[str, int] = {
    "0": 29,
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "5": 23,
    "6": 22,
    "7": 26,
    "8": 28,
    "9": 25,
    "a": 0,
    "b": 11,
    "c": 8,
    "d": 2,
    "e": 14,
    "f": 3,
    "g": 5,
    "h": 4,
    "i": 34,
    "j": 38,
    "k": 40,
    "l": 37,
    "m": 46,
    "n": 45,
    "o": 31,
    "p": 35,
    "q": 12,
    "r": 15,
    "s": 1,
    "t": 17,
    "u": 32,
    "v": 9,
    "w": 13,
    "x": 7,
    "y": 16,
    "z": 6,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
}
