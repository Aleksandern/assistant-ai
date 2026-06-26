from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.browser_ui_runtime import BrowserUiRuntime, _select_port
from modules.sqlite_conversation_store import create_code_test
from modules.ui_publisher import UiPublisher


class BrowserUiRuntimeTests(unittest.TestCase):
    def test_start_skips_task_snapshot_hydration_when_task_feature_is_disabled(self) -> None:
        runtime = BrowserUiRuntime(
            socket_server_factory=FakeSocketServer,
            web_server_factory=FakeWebServer,
            port_selector=FixedPortSelector([43182, 43181]),
            task_feature_enabled=False,
        )

        with patch("modules.browser_ui_runtime.build_current_task_snapshot") as build_snapshot:
            runtime.start()

        build_snapshot.assert_not_called()
        self.assertEqual(
            [runtime.publisher.snapshot_provider()],
            runtime._initial_messages_provider(),
        )
        self.assertEqual("empty", runtime.publisher.task_snapshot_provider()["payload"]["status"])
        self.assertIsNone(runtime.publisher.task_snapshot_provider()["payload"]["latest_result"])

    def test_start_hydrates_task_snapshot_from_persisted_code_test(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = Path(temp_dir.name) / "database" / "test.sqlite3"
        create_code_test(
            "Persisted task solution",
            2,
            database_path=database_path,
        )

        runtime = BrowserUiRuntime(
            socket_server_factory=FakeSocketServer,
            web_server_factory=FakeWebServer,
            port_selector=FixedPortSelector([43182, 43181]),
            database_path=database_path,
            task_feature_enabled=True,
        )

        self.assertIsNone(runtime.publisher.task_snapshot_provider()["payload"]["latest_result"])

        runtime.start()

        self.assertEqual(
            {
                "name": "code_tests",
                "status": "passed",
                "summary": "Saved task solve from 2 screenshot(s).",
                "response_text": "Persisted task solution",
            },
            runtime.publisher.task_snapshot_provider()["payload"]["latest_result"],
        )

    def test_publisher_is_created_in_constructor_before_start(self) -> None:
        runtime = BrowserUiRuntime(
            socket_server_factory=FakeSocketServer,
            web_server_factory=FakeWebServer,
            port_selector=FixedPortSelector([43182, 43181]),
        )

        self.assertIsInstance(runtime.publisher, UiPublisher)
        self.assertIsNone(runtime.local_url)
        self.assertIsNone(runtime.lan_url)
        self.assertIsNone(runtime.websocket_url)

    def test_start_wires_publisher_socket_server_and_web_server(self) -> None:
        socket_factory = RecordingSocketServerFactory()
        web_factory = RecordingWebServerFactory()
        runtime = BrowserUiRuntime(
            host="0.0.0.0",
            preferred_port=43181,
            preferred_socket_port=43182,
            socket_server_factory=socket_factory,
            web_server_factory=web_factory,
            port_selector=FixedPortSelector([43182, 43181]),
            lan_ip_resolver=lambda: "192.168.1.5",
            task_feature_enabled=True,
        )

        runtime.start()

        self.assertEqual("ws://192.168.1.5:43182", runtime.websocket_url)
        self.assertEqual("http://127.0.0.1:43181", runtime.local_url)
        self.assertEqual("http://192.168.1.5:43181", runtime.lan_url)
        self.assertEqual(1, socket_factory.instance.start_calls)
        self.assertEqual(1, web_factory.instance.start_calls)
        self.assertEqual("ws://0.0.0.0:43182", web_factory.instance.websocket_url)
        self.assertTrue(web_factory.instance.task_feature_enabled)
        self.assertIs(socket_factory.instance.snapshot_provider.__self__, runtime.publisher)
        self.assertEqual(runtime.publisher.snapshot_provider(), socket_factory.instance.snapshot_provider())
        self.assertEqual(
            [
                runtime.publisher.snapshot_provider(),
                runtime.publisher.task_snapshot_provider(),
            ],
            socket_factory.instance.initial_messages_provider(),
        )
        self.assertIs(web_factory.instance.on_task_screenshot, None)
        self.assertIs(web_factory.instance.on_task_send, None)
        self.assertIs(web_factory.instance.on_task_clear, None)

    def test_start_is_noop_when_called_twice(self) -> None:
        socket_factory = RecordingSocketServerFactory()
        web_factory = RecordingWebServerFactory()
        runtime = BrowserUiRuntime(
            socket_server_factory=socket_factory,
            web_server_factory=web_factory,
            port_selector=FixedPortSelector([43182, 43181]),
            lan_ip_resolver=lambda: "192.168.1.5",
        )

        runtime.start()
        first_local_url = runtime.local_url
        first_websocket_url = runtime.websocket_url

        runtime.start()

        self.assertEqual(1, socket_factory.call_count)
        self.assertEqual(1, web_factory.call_count)
        self.assertEqual(1, socket_factory.instance.start_calls)
        self.assertEqual(1, web_factory.instance.start_calls)
        self.assertEqual(first_local_url, runtime.local_url)
        self.assertEqual(first_websocket_url, runtime.websocket_url)

    def test_stop_stops_web_server_before_socket_server_and_is_safe_twice(self) -> None:
        events: list[str] = []
        socket_factory = RecordingSocketServerFactory(events=events)
        web_factory = RecordingWebServerFactory(events=events)
        runtime = BrowserUiRuntime(
            socket_server_factory=socket_factory,
            web_server_factory=web_factory,
            port_selector=FixedPortSelector([43182, 43181]),
            lan_ip_resolver=lambda: "192.168.1.5",
        )
        runtime.start()

        runtime.stop()
        runtime.stop()

        self.assertEqual(["socket.start", "web.start", "web.stop", "socket.stop"], events)
        self.assertEqual(1, socket_factory.instance.stop_calls)
        self.assertEqual(1, web_factory.instance.stop_calls)
        self.assertIsNone(runtime.local_url)
        self.assertIsNone(runtime.lan_url)
        self.assertIsNone(runtime.websocket_url)

    def test_start_propagates_socket_server_error_without_starting_web_server(self) -> None:
        socket_factory = RecordingSocketServerFactory(start_error=RuntimeError("socket failed"))
        web_factory = RecordingWebServerFactory()
        runtime = BrowserUiRuntime(
            socket_server_factory=socket_factory,
            web_server_factory=web_factory,
            port_selector=FixedPortSelector([43182]),
        )

        with self.assertRaisesRegex(RuntimeError, "socket failed"):
            runtime.start()

        self.assertEqual(1, socket_factory.instance.start_calls)
        self.assertEqual(0, socket_factory.instance.stop_calls)
        self.assertEqual(0, web_factory.call_count)
        self.assertIsNone(runtime.local_url)
        self.assertIsNone(runtime.lan_url)
        self.assertIsNone(runtime.websocket_url)

    def test_web_server_start_failure_rolls_back_started_socket_server(self) -> None:
        events: list[str] = []
        socket_factory = RecordingSocketServerFactory(events=events)
        web_factory = RecordingWebServerFactory(events=events, start_error=RuntimeError("web failed"))
        runtime = BrowserUiRuntime(
            socket_server_factory=socket_factory,
            web_server_factory=web_factory,
            port_selector=FixedPortSelector([43182, 43181]),
            lan_ip_resolver=lambda: "192.168.1.5",
        )

        with self.assertRaisesRegex(RuntimeError, "web failed"):
            runtime.start()

        self.assertEqual(["socket.start", "web.start", "socket.stop"], events)
        self.assertEqual(1, socket_factory.instance.stop_calls)
        self.assertEqual(1, web_factory.instance.start_calls)
        self.assertEqual(0, web_factory.instance.stop_calls)
        self.assertIsNone(runtime.local_url)
        self.assertIsNone(runtime.lan_url)
        self.assertIsNone(runtime.websocket_url)

    def test_lan_url_is_none_when_lan_ip_cannot_be_resolved(self) -> None:
        web_factory = RecordingWebServerFactory()
        runtime = BrowserUiRuntime(
            socket_server_factory=RecordingSocketServerFactory(),
            web_server_factory=web_factory,
            port_selector=FixedPortSelector([43182, 43181]),
            lan_ip_resolver=lambda: None,
        )

        runtime.start()

        self.assertEqual("ws://127.0.0.1:43182", runtime.websocket_url)
        self.assertEqual("http://127.0.0.1:43181", runtime.local_url)
        self.assertIsNone(runtime.lan_url)
        self.assertEqual("ws://0.0.0.0:43182", web_factory.instance.websocket_url)

    def test_port_selector_receives_preferred_ports_and_excludes_used_port(self) -> None:
        selector = RecordingPortSelector([53182, 53181])
        runtime = BrowserUiRuntime(
            preferred_port=43181,
            preferred_socket_port=43182,
            socket_server_factory=RecordingSocketServerFactory(),
            web_server_factory=RecordingWebServerFactory(),
            port_selector=selector,
        )

        runtime.start()

        self.assertEqual(
            [
                (43182, frozenset()),
                (43181, frozenset({53182})),
            ],
            selector.calls,
        )
        self.assertEqual(53182, runtime.socket_port)
        self.assertEqual(53181, runtime.web_port)


class SelectPortTests(unittest.TestCase):
    def test_select_port_returns_preferred_port_when_available(self) -> None:
        selected = _select_port(43181, is_port_available=lambda port: port == 43181)

        self.assertEqual(43181, selected)

    def test_select_port_falls_back_to_dynamic_private_range(self) -> None:
        selected = _select_port(
            43181,
            is_port_available=lambda port: port == 49152,
        )

        self.assertEqual(49152, selected)

    def test_select_port_skips_excluded_ports_in_fallback_range(self) -> None:
        selected = _select_port(
            43181,
            excluded_ports={49152},
            is_port_available=lambda port: port in {49152, 49153},
        )

        self.assertEqual(49153, selected)


class RecordingSocketServerFactory:
    def __init__(self, *, events: list[str] | None = None, start_error: Exception | None = None) -> None:
        self.events = events
        self.start_error = start_error
        self.call_count = 0
        self.instance: FakeSocketServer | None = None

    def __call__(self, *, snapshot_provider, initial_messages_provider, host: str, port: int):
        self.call_count += 1
        self.instance = FakeSocketServer(
            snapshot_provider=snapshot_provider,
            initial_messages_provider=initial_messages_provider,
            host=host,
            port=port,
            events=self.events,
            start_error=self.start_error,
        )
        return self.instance


class RecordingWebServerFactory:
    def __init__(self, *, events: list[str] | None = None, start_error: Exception | None = None) -> None:
        self.events = events
        self.start_error = start_error
        self.call_count = 0
        self.instance: FakeWebServer | None = None

    def __call__(
        self,
        *,
        websocket_url: str,
        host: str,
        port: int,
        task_feature_enabled: bool = False,
        on_task_screenshot=None,
        on_task_send=None,
        on_task_clear=None,
    ):
        self.call_count += 1
        self.instance = FakeWebServer(
            websocket_url=websocket_url,
            host=host,
            port=port,
            task_feature_enabled=task_feature_enabled,
            on_task_screenshot=on_task_screenshot,
            on_task_send=on_task_send,
            on_task_clear=on_task_clear,
            events=self.events,
            start_error=self.start_error,
        )
        return self.instance


class FakeSocketServer:
    def __init__(
        self,
        *,
        snapshot_provider,
        initial_messages_provider,
        host: str,
        port: int,
        events: list[str] | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.initial_messages_provider = initial_messages_provider
        self.host = host
        self.requested_port = port
        self.events = events
        self.start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.bound_port = port

    def start(self) -> None:
        self.start_calls += 1
        if self.events is not None:
            self.events.append("socket.start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stop_calls += 1
        if self.events is not None:
            self.events.append("socket.stop")


class FakeWebServer:
    def __init__(
        self,
        *,
        websocket_url: str,
        host: str,
        port: int,
        task_feature_enabled: bool = False,
        on_task_screenshot=None,
        on_task_send=None,
        on_task_clear=None,
        events: list[str] | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.websocket_url = websocket_url
        self.host = host
        self.requested_port = port
        self.task_feature_enabled = task_feature_enabled
        self.on_task_screenshot = on_task_screenshot
        self.on_task_send = on_task_send
        self.on_task_clear = on_task_clear
        self.events = events
        self.start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.bound_port = port

    def start(self) -> None:
        self.start_calls += 1
        if self.events is not None:
            self.events.append("web.start")
        if self.start_error is not None:
            raise self.start_error

    def stop(self) -> None:
        self.stop_calls += 1
        if self.events is not None:
            self.events.append("web.stop")


class FixedPortSelector:
    def __init__(self, ports: list[int]) -> None:
        self._ports = list(ports)

    def __call__(self, preferred_port: int, *, excluded_ports: frozenset[int]) -> int:
        if not self._ports:
            raise AssertionError(f"Unexpected port selection request for {preferred_port}")
        return self._ports.pop(0)


class RecordingPortSelector:
    def __init__(self, ports: list[int]) -> None:
        self._ports = list(ports)
        self.calls: list[tuple[int, frozenset[int]]] = []

    def __call__(self, preferred_port: int, *, excluded_ports: frozenset[int]) -> int:
        self.calls.append((preferred_port, excluded_ports))
        if not self._ports:
            raise AssertionError(f"Unexpected port selection request for {preferred_port}")
        return self._ports.pop(0)


if __name__ == "__main__":
    unittest.main()
