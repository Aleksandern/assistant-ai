from __future__ import annotations

"""Lifecycle-managed boundary for task hotkey input dispatch."""

from collections.abc import Callable, Mapping
from queue import Queue
import threading
from typing import Literal, Protocol


TaskHotkeyAction = Literal["capture", "solve", "clear"]
_TASK_HOTKEY_ACTIONS: tuple[TaskHotkeyAction, ...] = ("capture", "solve", "clear")
_STOP_WORKER = object()


class TaskHotkeyListenerConfigurationError(ValueError):
    """Raised when the task hotkey listener is configured with invalid input."""


class TaskHotkeyListenerBackendError(RuntimeError):
    """Raised when the task hotkey backend cannot be started or stopped cleanly."""


class TaskHotkeyBackendAdapter(Protocol):
    def start(
        self,
        *,
        hotkey_mapping: dict[str, str] | None,
        on_hotkey: Callable[[str], None],
    ) -> None: ...

    def stop(self) -> None: ...


class TaskHotkeyListener:
    def __init__(
        self,
        *,
        on_capture: Callable[[], None],
        on_solve: Callable[[], None],
        on_clear: Callable[[], None],
        backend_adapter: TaskHotkeyBackendAdapter | None,
        hotkey_mapping: Mapping[str, str] | None = None,
    ) -> None:
        if backend_adapter is None:
            raise TaskHotkeyListenerConfigurationError("Task hotkey backend adapter is required.")

        self._backend_adapter = backend_adapter
        self._hotkey_mapping = _normalize_hotkey_mapping(hotkey_mapping)
        self._callbacks: dict[TaskHotkeyAction, Callable[[], None]] = {
            "capture": _validate_callback("on_capture", on_capture),
            "solve": _validate_callback("on_solve", on_solve),
            "clear": _validate_callback("on_clear", on_clear),
        }
        self._is_started = False
        self._is_accepting_actions = False
        self._queue: Queue[TaskHotkeyAction | object] = Queue()
        self._worker_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._is_started:
            return

        with self._state_lock:
            self._queue = Queue()
            self._is_accepting_actions = True

        try:
            self._backend_adapter.start(
                hotkey_mapping=self._hotkey_mapping,
                on_hotkey=self._dispatch_hotkey,
            )
        except Exception as exc:
            with self._state_lock:
                self._is_accepting_actions = False
                self._queue = Queue()
            raise TaskHotkeyListenerBackendError(f"Task hotkey listener failed to start: {exc}") from exc

        worker_thread = threading.Thread(
            target=self._run_worker,
            name="task-hotkey-listener-worker",
            daemon=True,
        )
        worker_thread.start()
        self._worker_thread = worker_thread
        self._is_started = True

    def stop(self) -> None:
        if not self._is_started:
            return

        with self._state_lock:
            self._is_accepting_actions = False

        try:
            self._backend_adapter.stop()
        except Exception as exc:
            with self._state_lock:
                self._is_accepting_actions = True
            raise TaskHotkeyListenerBackendError(f"Task hotkey listener failed to stop: {exc}") from exc

        worker_thread = self._worker_thread
        self._queue.put(_STOP_WORKER)
        if worker_thread is not None:
            worker_thread.join()

        self._worker_thread = None
        self._is_started = False

    def _dispatch_hotkey(self, action: str) -> None:
        if action not in self._callbacks:
            raise TaskHotkeyListenerConfigurationError(f"Unsupported task hotkey action: {action}")

        with self._state_lock:
            if not self._is_accepting_actions:
                return

        self._queue.put(action)

    def _run_worker(self) -> None:
        while True:
            queued_item = self._queue.get()
            try:
                if queued_item is _STOP_WORKER:
                    return
                self._callbacks[queued_item]()
            except Exception:
                continue


def _normalize_hotkey_mapping(hotkey_mapping: Mapping[str, str] | None) -> dict[str, str] | None:
    if hotkey_mapping is None:
        return None

    normalized_mapping: dict[str, str] = {}
    for action in _TASK_HOTKEY_ACTIONS:
        raw_hotkey = hotkey_mapping.get(action)
        if raw_hotkey is None or not raw_hotkey.strip():
            raise TaskHotkeyListenerConfigurationError(
                "Task hotkey mapping must define capture, solve, and clear with non-empty bindings."
            )
        normalized_mapping[action] = raw_hotkey.strip()

    unexpected_actions = sorted(set(hotkey_mapping) - set(_TASK_HOTKEY_ACTIONS))
    if unexpected_actions:
        raise TaskHotkeyListenerConfigurationError(
            f"Task hotkey mapping contains unsupported actions: {', '.join(unexpected_actions)}"
        )

    unique_hotkeys = set(normalized_mapping.values())
    if len(unique_hotkeys) != len(normalized_mapping):
        raise TaskHotkeyListenerConfigurationError("Task hotkey bindings must be unique.")

    return normalized_mapping


def _validate_callback(name: str, callback: Callable[[], None] | object) -> Callable[[], None]:
    if not callable(callback):
        raise TaskHotkeyListenerConfigurationError(f"Task hotkey listener {name} callback must be callable.")
    return callback
