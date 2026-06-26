from __future__ import annotations

"""HTTP delivery layer for the browser UI assets."""

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit


class TaskActionHandler(Protocol):
    def __call__(self) -> object: ...


class WebServer:
    """Small synchronous wrapper around a threaded HTTP server."""

    def __init__(
        self,
        *,
        websocket_url: str,
        host: str = "127.0.0.1",
        port: int = 0,
        asset_root: str | Path | None = None,
        task_artifact_dir: str | Path | None = None,
        task_feature_enabled: bool = True,
        on_task_screenshot: TaskActionHandler | None = None,
        on_task_send: TaskActionHandler | None = None,
        on_task_clear: TaskActionHandler | None = None,
    ) -> None:
        self._host = host
        self._requested_port = port
        self._websocket_url = self._require_non_empty_text("websocket_url", websocket_url)
        self._asset_root = Path(asset_root) if asset_root is not None else self._default_asset_root()
        self._task_artifact_root = Path(task_artifact_dir).expanduser().resolve() if task_artifact_dir is not None else None
        self._task_feature_enabled = bool(task_feature_enabled)
        self._task_action_handlers = {
            "screenshot": self._require_optional_handler("on_task_screenshot", on_task_screenshot),
            "send": self._require_optional_handler("on_task_send", on_task_send),
            "clear": self._require_optional_handler("on_task_clear", on_task_clear),
        }

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bound_port = 0
        self._lifecycle_lock = threading.Lock()

    @property
    def bound_port(self) -> int:
        return self._bound_port

    @property
    def local_url(self) -> str:
        return f"http://{self._host}:{self._bound_port}"

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._server is not None:
                return

            handler = self._build_handler()
            server = ThreadingHTTPServer((self._host, self._requested_port), handler)
            thread = threading.Thread(
                target=server.serve_forever,
                name="browser-ui-web-server",
                daemon=True,
            )

            try:
                thread.start()
            except Exception:
                server.server_close()
                raise

            self._server = server
            self._thread = thread
            self._bound_port = int(server.server_address[1])

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._server is None or self._thread is None:
                return

            server = self._server
            thread = self._thread

            self._server = None
            self._thread = None
            self._bound_port = 0

            server.shutdown()
            server.server_close()
            thread.join()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        server_owner = self

        class BrowserUiHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                try:
                    if self.path == "/":
                        body = server_owner._render_index_html(request_host=self.headers.get("Host")).encode("utf-8")
                        self._respond(HTTPStatus.OK, body, "text/html; charset=utf-8")
                        return

                    if self.path == "/app.js":
                        body = server_owner._read_asset_text("app.js").encode("utf-8")
                        self._respond(HTTPStatus.OK, body, "text/javascript; charset=utf-8")
                        return

                    if self.path == "/styles.css":
                        body = server_owner._read_asset_text("styles.css").encode("utf-8")
                        self._respond(HTTPStatus.OK, body, "text/css; charset=utf-8")
                        return

                    if self.path.startswith("/task-artifacts/"):
                        body, content_type = server_owner._read_task_artifact(self.path)
                        self._respond(HTTPStatus.OK, body, content_type)
                        return

                    self._respond(HTTPStatus.NOT_FOUND, b"Not Found", "text/plain; charset=utf-8")
                except FileNotFoundError:
                    self._respond(HTTPStatus.NOT_FOUND, b"Not Found", "text/plain; charset=utf-8")
                except RuntimeError:
                    self._respond(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        b"Internal Server Error",
                        "text/plain; charset=utf-8",
                    )

            def do_POST(self) -> None:  # noqa: N802
                try:
                    status, body = server_owner._handle_post_request(self.path)
                    self._respond(status, body, "application/json; charset=utf-8")
                except FileNotFoundError:
                    self._respond(HTTPStatus.NOT_FOUND, b'{"error":"Not Found"}', "application/json; charset=utf-8")
                except RuntimeError as error:
                    error_body = json.dumps({"error": str(error)}).encode("utf-8")
                    self._respond(HTTPStatus.INTERNAL_SERVER_ERROR, error_body, "application/json; charset=utf-8")
                except Exception:
                    self._respond(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        b'{"error":"Internal Server Error"}',
                        "application/json; charset=utf-8",
                    )

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return

            def _respond(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return BrowserUiHandler

    def _render_index_html(self, *, request_host: str | None) -> str:
        config_json = json.dumps(
            {
                "websocket_url": self._resolve_request_websocket_url(request_host=request_host),
                "task_artifact_base_url": "/task-artifacts",
                "task_feature_enabled": self._task_feature_enabled,
            }
        )
        template = self._read_asset_text("index.html")
        rendered_task_nav = ""
        rendered_task_panel = ""
        if self._task_feature_enabled:
            rendered_task_nav = """
        <button id="tab-button-task" class="tab-button" type="button">task</button>"""
            rendered_task_panel = """
      <section id="tab-panel-task" class="tab-panel tab-panel-hidden">
        <section class="panel panel-task">
          <h2>Task Snapshot</h2>

          <div class="task-actions">
            <button id="task-action-screenshot" class="task-action-button" type="button">screenshot</button>
            <button id="task-action-send" class="task-action-button" type="button">send</button>
            <button id="task-action-clear" class="task-action-button" type="button">clear</button>
          </div>

          <dl class="task-summary">
            <div class="task-summary-row">
              <dt>Status</dt>
              <dd id="task-status-value">idle</dd>
            </div>
            <div class="task-summary-row">
              <dt>Files</dt>
              <dd id="task-file-count-value">0</dd>
            </div>
          </dl>

          <div class="task-section">
            <h3>Artifacts</h3>
            <div id="task-artifacts-list" class="task-list text-block" aria-live="polite"></div>
          </div>

          <div class="task-section">
            <h3>Preview</h3>
            <div id="task-preview-list" class="task-preview-grid" aria-live="polite"></div>
          </div>

          <div class="task-section">
            <h3>Latest Result</h3>
            <div id="task-result-text" class="task-result text-block" aria-live="polite">-</div>
          </div>

          <div class="task-section">
            <h3>Task Error</h3>
            <div id="task-error-text" class="task-error text-block" aria-live="polite">-</div>
          </div>
        </section>
      </section>"""
        return (
            template.replace("__BROWSER_UI_CONFIG__", config_json)
            .replace("__TASK_TAB_BUTTON__", rendered_task_nav)
            .replace("__TASK_TAB_PANEL__", rendered_task_panel)
        )

    def _handle_post_request(self, request_path: str) -> tuple[HTTPStatus, bytes]:
        if not self._task_feature_enabled:
            raise FileNotFoundError("Task feature is disabled.")

        action = self._resolve_task_action(request_path)
        if action is None:
            raise FileNotFoundError(f"Unknown route: {request_path}")

        handler = self._task_action_handlers.get(action)
        if handler is None:
            raise RuntimeError(f"Task action handler is not configured for: {action}")

        handler()
        response_body = json.dumps(
            {
                "ok": True,
                "action": action,
            }
        ).encode("utf-8")
        return HTTPStatus.OK, response_body

    def _resolve_request_websocket_url(self, *, request_host: str | None) -> str:
        parsed_url = urlsplit(self._websocket_url)
        websocket_host = parsed_url.hostname

        if websocket_host not in {"0.0.0.0", "::"}:
            return self._websocket_url

        request_hostname = self._extract_request_hostname(request_host)
        if request_hostname is None:
            return self._websocket_url

        return urlunsplit(
            SplitResult(
                scheme=parsed_url.scheme,
                netloc=self._format_netloc(request_hostname, parsed_url.port),
                path=parsed_url.path,
                query=parsed_url.query,
                fragment=parsed_url.fragment,
            )
        )

    def _read_asset_text(self, asset_name: str) -> str:
        asset_path = self._asset_root / asset_name
        try:
            return asset_path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError(f"Failed to read browser UI asset: {asset_path}") from error

    def _default_asset_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "web" / "browser_ui"

    def _read_task_artifact(self, request_path: str) -> tuple[bytes, str]:
        relative_name = unquote(request_path[len("/task-artifacts/") :]).strip()
        if not relative_name:
            raise FileNotFoundError("Missing task artifact path")

        task_artifact_root = self._task_artifact_root or self._default_task_artifact_root()
        resolved_path = (task_artifact_root / relative_name).resolve()
        try:
            resolved_path.relative_to(task_artifact_root)
        except ValueError as error:
            raise FileNotFoundError("Task artifact path escapes artifact root") from error

        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"Task artifact not found: {resolved_path}")

        try:
            body = resolved_path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"Failed to read task artifact: {resolved_path}") from error

        return body, self._task_artifact_content_type(resolved_path)

    def _task_artifact_content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "application/octet-stream"

    def _default_task_artifact_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "artifacts" / "process"

    def _resolve_task_action(self, request_path: str) -> str | None:
        if request_path == "/api/task/screenshot":
            return "screenshot"
        if request_path == "/api/task/send":
            return "send"
        if request_path == "/api/task/clear":
            return "clear"
        return None

    def _require_optional_handler(
        self,
        field_name: str,
        handler: TaskActionHandler | None,
    ) -> TaskActionHandler | None:
        if handler is None:
            return None
        if not callable(handler):
            raise ValueError(f"{field_name} must be callable")
        return handler

    def _extract_request_hostname(self, request_host: str | None) -> str | None:
        if request_host is None:
            return None

        normalized_host = request_host.strip()
        if not normalized_host:
            return None

        if normalized_host.startswith("["):
            closing_bracket_index = normalized_host.find("]")
            if closing_bracket_index == -1:
                return None
            return normalized_host[1:closing_bracket_index]

        if ":" in normalized_host:
            return normalized_host.rsplit(":", 1)[0]

        return normalized_host

    def _format_netloc(self, host: str, port: int | None) -> str:
        normalized_host = host
        if ":" in normalized_host and not normalized_host.startswith("["):
            normalized_host = f"[{normalized_host}]"

        if port is None:
            return normalized_host

        return f"{normalized_host}:{port}"

    def _require_non_empty_text(self, field_name: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError(f"{field_name} must not be empty")
        return normalized_value
