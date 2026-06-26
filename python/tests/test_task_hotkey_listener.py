from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_hotkey_listener import (
    TaskHotkeyListener,
    TaskHotkeyListenerBackendError,
    TaskHotkeyListenerConfigurationError,
)


class TaskHotkeyListenerTests(unittest.TestCase):
    def test_start_calls_backend_once_and_repeated_start_is_safe(self) -> None:
        backend = FakeHotkeyBackend()
        listener = self._create_listener(backend=backend)

        listener.start()
        listener.start()

        self.assertEqual(1, backend.start_calls)
        self.assertEqual(0, backend.stop_calls)

    def test_stop_is_safe_before_start_and_on_repeated_calls(self) -> None:
        backend = FakeHotkeyBackend()
        listener = self._create_listener(backend=backend)

        listener.stop()
        listener.start()
        listener.stop()
        listener.stop()

        self.assertEqual(1, backend.start_calls)
        self.assertEqual(1, backend.stop_calls)

    def test_capture_event_dispatches_only_capture_callback(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: calls.append("capture"),
            on_solve=lambda: calls.append("solve"),
            on_clear=lambda: calls.append("clear"),
        )
        listener.start()

        backend.emit("capture")
        listener.stop()

        self.assertEqual(["capture"], calls)

    def test_solve_event_dispatches_only_solve_callback(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: calls.append("capture"),
            on_solve=lambda: calls.append("solve"),
            on_clear=lambda: calls.append("clear"),
        )
        listener.start()

        backend.emit("solve")
        listener.stop()

        self.assertEqual(["solve"], calls)

    def test_clear_event_dispatches_only_clear_callback(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: calls.append("capture"),
            on_solve=lambda: calls.append("solve"),
            on_clear=lambda: calls.append("clear"),
        )
        listener.start()

        backend.emit("clear")
        listener.stop()

        self.assertEqual(["clear"], calls)

    def test_listener_passes_mapping_to_backend_without_mutation(self) -> None:
        backend = FakeHotkeyBackend()
        hotkey_mapping = {
            "capture": "cmd+shift+1",
            "solve": "cmd+shift+2",
            "clear": "cmd+shift+3",
        }
        listener = self._create_listener(
            backend=backend,
            hotkey_mapping=hotkey_mapping,
        )

        listener.start()

        self.assertEqual(hotkey_mapping, backend.received_mapping)
        self.assertIsNot(hotkey_mapping, backend.received_mapping)

    def test_listener_rejects_missing_backend_adapter(self) -> None:
        with self.assertRaisesRegex(TaskHotkeyListenerConfigurationError, "backend adapter is required"):
            self._create_listener(backend=None)

    def test_listener_rejects_incomplete_hotkey_mapping(self) -> None:
        with self.assertRaisesRegex(TaskHotkeyListenerConfigurationError, "must define capture, solve, and clear"):
            self._create_listener(
                backend=FakeHotkeyBackend(),
                hotkey_mapping={
                    "capture": "cmd+shift+1",
                    "solve": "cmd+shift+2",
                },
            )

    def test_listener_rejects_duplicate_hotkey_bindings(self) -> None:
        with self.assertRaisesRegex(TaskHotkeyListenerConfigurationError, "must be unique"):
            self._create_listener(
                backend=FakeHotkeyBackend(),
                hotkey_mapping={
                    "capture": "cmd+shift+1",
                    "solve": "cmd+shift+1",
                    "clear": "cmd+shift+3",
                },
            )

    def test_listener_rejects_unsupported_hotkey_mapping_actions(self) -> None:
        with self.assertRaisesRegex(TaskHotkeyListenerConfigurationError, "contains unsupported actions: pause"):
            self._create_listener(
                backend=FakeHotkeyBackend(),
                hotkey_mapping={
                    "capture": "cmd+shift+1",
                    "solve": "cmd+shift+2",
                    "clear": "cmd+shift+3",
                    "pause": "cmd+shift+4",
                },
            )

    def test_listener_rejects_non_callable_callbacks(self) -> None:
        with self.assertRaisesRegex(TaskHotkeyListenerConfigurationError, "on_capture callback must be callable"):
            TaskHotkeyListener(
                on_capture="capture",
                on_solve=lambda: None,
                on_clear=lambda: None,
                backend_adapter=FakeHotkeyBackend(),
            )

    def test_listener_maps_backend_start_failure_to_boundary_error(self) -> None:
        backend = FakeHotkeyBackend(start_error=RuntimeError("backend unavailable"))
        listener = self._create_listener(backend=backend)

        with self.assertRaisesRegex(TaskHotkeyListenerBackendError, "failed to start: backend unavailable"):
            listener.start()

        self.assertEqual(1, backend.start_calls)
        self.assertEqual(0, backend.stop_calls)

    def test_listener_maps_backend_stop_failure_to_boundary_error(self) -> None:
        backend = FakeHotkeyBackend(stop_error=RuntimeError("backend stop failed"))
        listener = self._create_listener(backend=backend)
        listener.start()

        with self.assertRaisesRegex(TaskHotkeyListenerBackendError, "failed to stop: backend stop failed"):
            listener.stop()

        self.assertEqual(1, backend.start_calls)
        self.assertEqual(1, backend.stop_calls)

    def test_listener_remains_started_after_backend_stop_failure(self) -> None:
        backend = FakeHotkeyBackend(stop_error=RuntimeError("backend stop failed"))
        listener = self._create_listener(backend=backend)
        listener.start()

        with self.assertRaisesRegex(TaskHotkeyListenerBackendError, "failed to stop: backend stop failed"):
            listener.stop()
        with self.assertRaisesRegex(TaskHotkeyListenerBackendError, "failed to stop: backend stop failed"):
            listener.stop()

        self.assertEqual(1, backend.start_calls)
        self.assertEqual(2, backend.stop_calls)

    def test_hotkey_callback_returns_before_solve_callback_finishes(self) -> None:
        backend = FakeHotkeyBackend()
        solve_started = threading.Event()
        release_solve = threading.Event()
        emit_returned = threading.Event()
        listener = self._create_listener(
            backend=backend,
            on_solve=lambda: _blocking_callback(
                action_started=solve_started,
                release_action=release_solve,
            ),
        )
        listener.start()

        emitter = threading.Thread(target=lambda: _emit_and_signal(backend, "solve", emit_returned))
        emitter.start()

        self.assertTrue(solve_started.wait(timeout=1.0))
        self.assertTrue(emit_returned.wait(timeout=1.0))

        release_solve.set()
        emitter.join(timeout=1.0)
        listener.stop()

        self.assertFalse(emitter.is_alive())

    def test_listener_dispatches_actions_sequentially_in_arrival_order(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        solve_started = threading.Event()
        release_solve = threading.Event()
        clear_completed = threading.Event()
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: calls.append("capture"),
            on_solve=lambda: _blocking_callback(
                action_started=solve_started,
                release_action=release_solve,
                on_enter=lambda: calls.append("solve"),
            ),
            on_clear=lambda: _record_and_signal(calls, "clear", clear_completed),
        )
        listener.start()

        backend.emit("capture")
        backend.emit("solve")
        backend.emit("clear")

        self.assertTrue(solve_started.wait(timeout=1.0))
        self.assertEqual(["capture", "solve"], calls)
        self.assertFalse(clear_completed.is_set())

        release_solve.set()
        self.assertTrue(clear_completed.wait(timeout=1.0))
        listener.stop()

        self.assertEqual(["capture", "solve", "clear"], calls)

    def test_listener_continues_processing_actions_after_callback_error(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        solve_completed = threading.Event()
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: (_ for _ in ()).throw(RuntimeError("capture failed")),
            on_solve=lambda: _record_and_signal(calls, "solve", solve_completed),
        )
        listener.start()

        backend.emit("capture")
        backend.emit("solve")

        self.assertTrue(solve_completed.wait(timeout=1.0))
        listener.stop()

        self.assertEqual(["solve"], calls)

    def test_stop_drains_queued_actions_and_terminates_worker(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        capture_started = threading.Event()
        release_capture = threading.Event()
        solve_completed = threading.Event()
        stop_returned = threading.Event()
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: _blocking_callback(
                action_started=capture_started,
                release_action=release_capture,
                on_enter=lambda: calls.append("capture"),
            ),
            on_solve=lambda: _record_and_signal(calls, "solve", solve_completed),
        )
        listener.start()

        backend.emit("capture")
        backend.emit("solve")

        self.assertTrue(capture_started.wait(timeout=1.0))
        stopper = threading.Thread(target=lambda: _stop_and_signal(listener, stop_returned))
        stopper.start()

        self.assertFalse(stop_returned.wait(timeout=0.2))
        self.assertFalse(solve_completed.is_set())

        release_capture.set()
        self.assertTrue(stop_returned.wait(timeout=1.0))
        self.assertTrue(solve_completed.wait(timeout=1.0))
        stopper.join(timeout=1.0)

        self.assertFalse(stopper.is_alive())
        self.assertEqual(["capture", "solve"], calls)
        self.assertIsNone(listener._worker_thread)

    def test_listener_can_restart_after_stop_and_process_new_actions(self) -> None:
        backend = FakeHotkeyBackend()
        calls: list[str] = []
        listener = self._create_listener(
            backend=backend,
            on_capture=lambda: calls.append("capture"),
            on_solve=lambda: calls.append("solve"),
        )

        listener.start()
        backend.emit("capture")
        listener.stop()

        listener.start()
        backend.emit("solve")
        listener.stop()

        self.assertEqual(["capture", "solve"], calls)
        self.assertEqual(2, backend.start_calls)
        self.assertEqual(2, backend.stop_calls)

    def test_listener_rejects_unknown_backend_action(self) -> None:
        backend = FakeHotkeyBackend()
        listener = self._create_listener(backend=backend)
        listener.start()

        with self.assertRaisesRegex(TaskHotkeyListenerConfigurationError, "Unsupported task hotkey action: archive"):
            backend.emit("archive")

    def _create_listener(
        self,
        *,
        backend: FakeHotkeyBackend | None,
        on_capture=None,
        on_solve=None,
        on_clear=None,
        hotkey_mapping: dict[str, str] | None = None,
    ) -> TaskHotkeyListener:
        return TaskHotkeyListener(
            on_capture=(lambda: None) if on_capture is None else on_capture,
            on_solve=(lambda: None) if on_solve is None else on_solve,
            on_clear=(lambda: None) if on_clear is None else on_clear,
            backend_adapter=backend,
            hotkey_mapping=hotkey_mapping,
        )


class FakeHotkeyBackend:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.start_error = start_error
        self.stop_error = stop_error
        self.received_mapping: dict[str, str] | None = None
        self._on_hotkey = None

    def start(self, *, hotkey_mapping: dict[str, str] | None, on_hotkey) -> None:
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error
        self.received_mapping = None if hotkey_mapping is None else dict(hotkey_mapping)
        self._on_hotkey = on_hotkey

    def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        self._on_hotkey = None

    def emit(self, action: str) -> None:
        if self._on_hotkey is None:
            raise AssertionError("Hotkey backend is not started.")
        self._on_hotkey(action)


def _blocking_callback(
    *,
    action_started: threading.Event,
    release_action: threading.Event,
    on_enter=None,
) -> None:
    if on_enter is not None:
        on_enter()
    action_started.set()
    release_action.wait(timeout=1.0)


def _record_and_signal(calls: list[str], action: str, done_event: threading.Event) -> None:
    calls.append(action)
    done_event.set()


def _emit_and_signal(backend: FakeHotkeyBackend, action: str, done_event: threading.Event) -> None:
    backend.emit(action)
    done_event.set()


def _stop_and_signal(listener: TaskHotkeyListener, done_event: threading.Event) -> None:
    listener.stop()
    done_event.set()


if __name__ == "__main__":
    unittest.main()
