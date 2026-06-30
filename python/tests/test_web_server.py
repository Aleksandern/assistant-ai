from __future__ import annotations

import http.client
import socket
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.web_server import WebServer


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._servers: list[WebServer] = []
        self.websocket_url = "ws://192.168.0.25:9100/browser-ui"

    def tearDown(self) -> None:
        for server in self._servers:
            server.stop()

    def test_server_starts_and_stops(self) -> None:
        server = self._start_server()

        self.assertGreater(server.bound_port, 0)

        server.stop()

    def test_port_zero_exposes_real_bound_port_after_start(self) -> None:
        server = self._start_server(port=0)

        self.assertIsInstance(server.bound_port, int)
        self.assertGreater(server.bound_port, 0)

    def test_local_url_uses_bound_port_after_start(self) -> None:
        server = self._start_server(host="127.0.0.1", port=0)

        self.assertEqual(f"http://127.0.0.1:{server.bound_port}", server.local_url)

    def test_root_route_returns_html_page(self) -> None:
        server = self._start_server()

        response = self._request(server, "GET", "/")

        self.assertEqual(200, response.status)
        self.assertIn("text/html", self._header(response, "Content-Type"))
        self.assertIn("<!DOCTYPE html>", response.text)
        self.assertIn('id="status"', response.text)
        self.assertIn('id="tab-button-convs"', response.text)
        self.assertIn('id="tab-button-task"', response.text)
        self.assertIn('id="tab-panel-convs"', response.text)
        self.assertIn('id="tab-panel-task"', response.text)
        self.assertIn('id="conversation-log"', response.text)
        self.assertIn('id="task-status-value"', response.text)
        self.assertIn('id="task-file-count-value"', response.text)
        self.assertIn('id="task-artifacts-list"', response.text)
        self.assertIn('id="task-prompt-input"', response.text)
        self.assertIn('id="task-action-screenshot"', response.text)
        self.assertIn('id="task-action-send"', response.text)
        self.assertIn('id="task-action-clear"', response.text)
        self.assertIn('id="task-result-text"', response.text)
        self.assertIn('id="task-error-text"', response.text)
        self.assertIn('class="text-block terminal-screen"', response.text)
        self.assertIn('class="text-block error-screen"', response.text)
        self.assertIn('src="/app.js"', response.text)

    def test_root_route_hides_task_ui_when_feature_is_disabled(self) -> None:
        server = self._start_server(task_feature_enabled=False)

        response = self._request(server, "GET", "/")

        self.assertEqual(200, response.status)
        self.assertNotIn('id="tab-button-task"', response.text)
        self.assertNotIn('id="tab-panel-task"', response.text)
        self.assertNotIn('id="task-artifacts-list"', response.text)
        self.assertIn('"task_feature_enabled": false', response.text)

    def test_root_route_includes_task_ui_when_feature_is_enabled(self) -> None:
        server = self._start_server(task_feature_enabled=True)

        response = self._request(server, "GET", "/")

        self.assertEqual(200, response.status)
        self.assertIn('id="tab-button-task"', response.text)
        self.assertIn('id="tab-panel-task"', response.text)
        self.assertIn('id="task-artifacts-list"', response.text)
        self.assertIn('id="task-prompt-input"', response.text)
        self.assertIn('id="task-action-screenshot"', response.text)
        self.assertIn('"task_feature_enabled": true', response.text)

    def test_task_action_screenshot_route_calls_handler_and_returns_json(self) -> None:
        calls: list[str] = []
        server = self._start_server(on_task_screenshot=lambda: calls.append("screenshot"))

        response = self._request(server, "POST", "/api/task/screenshot")

        self.assertEqual(200, response.status)
        self.assertEqual(["screenshot"], calls)
        self.assertIn("application/json", self._header(response, "Content-Type"))
        self.assertIn('"ok": true', response.text)
        self.assertIn('"action": "screenshot"', response.text)

    def test_task_action_send_route_calls_handler_and_returns_json(self) -> None:
        calls: list[str | None] = []
        server = self._start_server(on_task_send=lambda task_prompt=None: calls.append(task_prompt))

        response = self._request(
            server,
            "POST",
            "/api/task/send",
            body=b'{"task_prompt":"Answer in German."}',
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(200, response.status)
        self.assertEqual(["Answer in German."], calls)
        self.assertIn('"action": "send"', response.text)

    def test_task_action_send_route_passes_none_when_prompt_is_missing(self) -> None:
        calls: list[str | None] = []
        server = self._start_server(on_task_send=lambda task_prompt=None: calls.append(task_prompt))

        response = self._request(server, "POST", "/api/task/send")

        self.assertEqual(200, response.status)
        self.assertEqual([None], calls)
        self.assertIn('"action": "send"', response.text)

    def test_task_action_clear_route_calls_handler_and_returns_json(self) -> None:
        calls: list[str] = []
        server = self._start_server(on_task_clear=lambda: calls.append("clear"))

        response = self._request(server, "POST", "/api/task/clear")

        self.assertEqual(200, response.status)
        self.assertEqual(["clear"], calls)
        self.assertIn('"action": "clear"', response.text)

    def test_task_action_route_returns_404_when_task_feature_is_disabled(self) -> None:
        server = self._start_server(task_feature_enabled=False)

        response = self._request(server, "POST", "/api/task/screenshot")

        self.assertEqual(404, response.status)

    def test_task_action_route_returns_500_when_handler_is_missing(self) -> None:
        server = self._start_server()

        response = self._request(server, "POST", "/api/task/screenshot")

        self.assertEqual(500, response.status)
        self.assertIn("not configured", response.text)

    def test_task_action_route_returns_500_when_handler_raises(self) -> None:
        server = self._start_server(on_task_clear=lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        response = self._request(server, "POST", "/api/task/clear")

        self.assertEqual(500, response.status)
        self.assertIn("boom", response.text)

    def test_task_action_route_returns_500_json_when_handler_raises_non_runtime_error(self) -> None:
        server = self._start_server(on_task_send=lambda _task_prompt=None: (_ for _ in ()).throw(ValueError("bad input")))

        response = self._request(server, "POST", "/api/task/send")

        self.assertEqual(500, response.status)
        self.assertIn("application/json", self._header(response, "Content-Type"))
        self.assertEqual('{"error":"Internal Server Error"}', response.text)

    def test_root_route_embeds_passed_websocket_url(self) -> None:
        server = self._start_server()

        response = self._request(server, "GET", "/")

        self.assertEqual(200, response.status)
        self.assertIn(self.websocket_url, response.text)

    def test_root_route_rewrites_wildcard_websocket_host_for_local_request(self) -> None:
        server = WebServer(websocket_url="ws://0.0.0.0:9100")
        server.start()
        self._servers.append(server)

        response = self._request(
            server,
            "GET",
            "/",
            headers={"Host": f"127.0.0.1:{server.bound_port}"},
        )

        self.assertEqual(200, response.status)
        self.assertIn("ws://127.0.0.1:9100", response.text)

    def test_root_route_rewrites_wildcard_websocket_host_for_lan_request(self) -> None:
        server = WebServer(websocket_url="ws://0.0.0.0:9100")
        server.start()
        self._servers.append(server)

        response = self._request(
            server,
            "GET",
            "/",
            headers={"Host": f"192.168.1.5:{server.bound_port}"},
        )

        self.assertEqual(200, response.status)
        self.assertIn("ws://192.168.1.5:9100", response.text)

    def test_app_js_route_returns_javascript(self) -> None:
        server = self._start_server()

        response = self._request(server, "GET", "/app.js")

        self.assertEqual(200, response.status)
        self.assertIn("javascript", self._header(response, "Content-Type"))
        self.assertIn("window.BROWSER_UI_CONFIG", response.text)
        self.assertIn("new WebSocket", response.text)
        self.assertIn("reply_final", response.text)
        self.assertIn("processing_error", response.text)
        self.assertIn("session_stopped", response.text)
        self.assertIn("task_snapshot", response.text)

    def test_styles_css_route_returns_css(self) -> None:
        server = self._start_server()

        response = self._request(server, "GET", "/styles.css")

        self.assertEqual(200, response.status)
        self.assertIn("text/css", self._header(response, "Content-Type"))
        self.assertIn(":root", response.text)
        self.assertIn(".tab-panel.tab-panel-hidden", response.text)

    def test_unknown_route_returns_404(self) -> None:
        server = self._start_server()

        response = self._request(server, "GET", "/missing")

        self.assertEqual(404, response.status)

    def test_task_artifact_route_returns_image_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_root = Path(temp_dir) / "assets"
            artifact_root = Path(temp_dir) / "artifacts"
            asset_root.mkdir(parents=True, exist_ok=True)
            artifact_root.mkdir(parents=True, exist_ok=True)
            (asset_root / "index.html").write_text("<!DOCTYPE html>", encoding="utf-8")
            (asset_root / "app.js").write_text("console.log('ok');", encoding="utf-8")
            (asset_root / "styles.css").write_text(":root {}", encoding="utf-8")
            expected_bytes = b"\x89PNG\r\n\x1a\nfake"
            (artifact_root / "screen-1.png").write_bytes(expected_bytes)

            server = WebServer(
                websocket_url=self.websocket_url,
                asset_root=asset_root,
                task_artifact_dir=artifact_root,
            )
            server.start()
            self._servers.append(server)

            response = self._request(server, "GET", "/task-artifacts/screen-1.png")

        self.assertEqual(200, response.status)
        self.assertIn("image/png", self._header(response, "Content-Type"))
        self.assertEqual(expected_bytes, response.body)

    def test_web_server_constructor_does_not_create_task_artifact_dir_eagerly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_artifact_root = Path(temp_dir) / "missing-artifacts"

            self.assertFalse(missing_artifact_root.exists())

            WebServer(
                websocket_url=self.websocket_url,
                task_artifact_dir=missing_artifact_root,
            )

            self.assertFalse(missing_artifact_root.exists())

    def test_stop_before_start_is_noop(self) -> None:
        server = WebServer(websocket_url=self.websocket_url)

        server.stop()

    def test_stop_is_safe_when_called_twice(self) -> None:
        server = self._start_server()

        server.stop()
        server.stop()

    def test_start_is_noop_when_called_twice(self) -> None:
        server = self._start_server()

        first_bound_port = server.bound_port
        first_local_url = server.local_url
        server.start()

        self.assertEqual(first_bound_port, server.bound_port)
        self.assertEqual(first_local_url, server.local_url)

    def test_start_propagates_bind_errors(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
            reserved_socket.bind(("127.0.0.1", 0))
            reserved_socket.listen()

            _, occupied_port = reserved_socket.getsockname()
            server = WebServer(
                host="127.0.0.1",
                port=occupied_port,
                websocket_url=self.websocket_url,
            )

            with self.assertRaises(OSError):
                server.start()

    def test_server_only_depends_on_websocket_url_string(self) -> None:
        custom_websocket_url = "ws://example.invalid:9999/live"
        server = WebServer(websocket_url=custom_websocket_url)
        server.start()
        self._servers.append(server)

        response = self._request(server, "GET", "/")

        self.assertEqual(200, response.status)
        self.assertIn(custom_websocket_url, response.text)

    def test_missing_asset_returns_500_instead_of_dropping_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            asset_root = Path(temp_dir)
            (asset_root / "app.js").write_text("console.log('ok');", encoding="utf-8")
            (asset_root / "styles.css").write_text(":root {}", encoding="utf-8")

            server = WebServer(
                websocket_url=self.websocket_url,
                asset_root=asset_root,
            )
            server.start()
            self._servers.append(server)

            response = self._request(server, "GET", "/")

        self.assertEqual(500, response.status)
        self.assertIn("Internal Server Error", response.text)

    def _start_server(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        task_feature_enabled: bool = True,
        on_task_screenshot=None,
        on_task_send=None,
        on_task_clear=None,
    ) -> WebServer:
        server = WebServer(
            host=host,
            port=port,
            websocket_url=self.websocket_url,
            task_feature_enabled=task_feature_enabled,
            on_task_screenshot=on_task_screenshot,
            on_task_send=on_task_send,
            on_task_clear=on_task_clear,
        )
        server.start()
        self._servers.append(server)
        return server

    def _request(
        self,
        server: WebServer,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> "_HttpResponse":
        connection = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            body = response.read()
            headers = {key: value for key, value in response.getheaders()}
            return _HttpResponse(status=response.status, headers=headers, body=body)
        finally:
            connection.close()

    def _header(self, response: "_HttpResponse", name: str) -> str:
        return response.headers.get(name, "")


class _HttpResponse:
    def __init__(self, *, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.text = body.decode("utf-8", errors="replace")


if __name__ == "__main__":
    unittest.main()
