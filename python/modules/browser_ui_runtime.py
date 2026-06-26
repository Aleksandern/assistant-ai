from __future__ import annotations

"""Lifecycle orchestration for the browser UI subsystem."""

import socket
import threading
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Protocol

from modules.socket_server import SocketServer
from modules.task_flow_service import build_current_task_snapshot
from modules.ui_publisher import UiPublisher
from modules.web_server import WebServer


DEFAULT_PREFERRED_PORT = 43181
DEFAULT_SOCKET_PREFERRED_PORT = DEFAULT_PREFERRED_PORT + 1
_DYNAMIC_PORT_RANGE_START = 49152
_DYNAMIC_PORT_RANGE_END = 65535


class _SocketServerLike(Protocol):
    bound_port: int

    def start(self) -> None: ...

    def stop(self) -> None: ...


class _WebServerLike(Protocol):
    bound_port: int

    def start(self) -> None: ...

    def stop(self) -> None: ...


class BrowserUiRuntime:
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        preferred_port: int = DEFAULT_PREFERRED_PORT,
        preferred_socket_port: int = DEFAULT_SOCKET_PREFERRED_PORT,
        socket_server_factory: Callable[..., _SocketServerLike] = SocketServer,
        web_server_factory: Callable[..., _WebServerLike] = WebServer,
        port_selector: Callable[[int], int] | Callable[[int, Collection[int]], int] | None = None,
        lan_ip_resolver: Callable[[], str | None] | None = None,
        database_path: str | Path | None = None,
        task_feature_enabled: bool = True,
        on_task_screenshot: Callable[[], object] | None = None,
        on_task_send: Callable[[], object] | None = None,
        on_task_clear: Callable[[], object] | None = None,
    ) -> None:
        self._host = self._require_non_empty_text("host", host)
        self._preferred_port = self._require_port_number("preferred_port", preferred_port)
        self._preferred_socket_port = self._require_port_number("preferred_socket_port", preferred_socket_port)
        self._socket_server_factory = socket_server_factory
        self._web_server_factory = web_server_factory
        self._port_selector = port_selector or self._default_port_selector
        self._lan_ip_resolver = lan_ip_resolver or _resolve_lan_ip
        self._database_path = database_path
        self._task_feature_enabled = bool(task_feature_enabled)
        self._on_task_screenshot = on_task_screenshot
        self._on_task_send = on_task_send
        self._on_task_clear = on_task_clear

        self._publisher = UiPublisher(transport=_RuntimeTransportProxy(self))
        self._socket_server: _SocketServerLike | None = None
        self._web_server: _WebServerLike | None = None
        self._socket_port: int | None = None
        self._web_port: int | None = None
        self._local_url: str | None = None
        self._lan_url: str | None = None
        self._websocket_url: str | None = None
        self._lifecycle_lock = threading.Lock()

    @property
    def publisher(self) -> UiPublisher:
        return self._publisher

    @property
    def local_url(self) -> str | None:
        return self._local_url

    @property
    def lan_url(self) -> str | None:
        return self._lan_url

    @property
    def websocket_url(self) -> str | None:
        return self._websocket_url

    @property
    def web_port(self) -> int | None:
        return self._web_port

    @property
    def socket_port(self) -> int | None:
        return self._socket_port

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._socket_server is not None and self._web_server is not None:
                return

            if self._task_feature_enabled:
                initial_task_snapshot = build_current_task_snapshot(
                    database_path=self._database_path,
                    error=None,
                )
                self._publisher.hydrate_task_snapshot(
                    status=initial_task_snapshot.status,
                    file_count=initial_task_snapshot.file_count,
                    artifacts=initial_task_snapshot.artifacts,
                    latest_result=initial_task_snapshot.latest_result,
                    error=initial_task_snapshot.error,
                )

            selected_socket_port = self._select_port(self._preferred_socket_port, excluded_ports=frozenset())
            socket_server = self._socket_server_factory(
                snapshot_provider=self._publisher.snapshot_provider,
                initial_messages_provider=self._initial_messages_provider,
                host=self._host,
                port=selected_socket_port,
            )
            socket_server.start()

            lan_ip = self._browser_lan_ip()
            websocket_url = f"ws://{self._websocket_host(lan_ip=lan_ip)}:{socket_server.bound_port}"
            websocket_template_url = f"ws://{self._websocket_template_host()}:{socket_server.bound_port}"

            selected_web_port = self._select_port(
                self._preferred_port,
                excluded_ports=frozenset({socket_server.bound_port}),
            )
            web_server = self._web_server_factory(
                websocket_url=websocket_template_url,
                host=self._host,
                port=selected_web_port,
                task_feature_enabled=self._task_feature_enabled,
                on_task_screenshot=self._on_task_screenshot,
                on_task_send=self._on_task_send,
                on_task_clear=self._on_task_clear,
            )

            try:
                web_server.start()
            except Exception:
                socket_server.stop()
                raise

            self._socket_server = socket_server
            self._web_server = web_server
            self._socket_port = socket_server.bound_port
            self._web_port = web_server.bound_port
            local_browser_host = self._local_browser_host()
            self._local_url = f"http://{local_browser_host}:{web_server.bound_port}"
            self._lan_url = f"http://{lan_ip}:{web_server.bound_port}" if lan_ip is not None else None
            self._websocket_url = websocket_url

    def stop(self) -> None:
        with self._lifecycle_lock:
            web_server = self._web_server
            socket_server = self._socket_server

            if web_server is None and socket_server is None:
                return

            self._web_server = None
            self._socket_server = None
            self._web_port = None
            self._socket_port = None
            self._local_url = None
            self._lan_url = None
            self._websocket_url = None

            stop_error: Exception | None = None

            if web_server is not None:
                try:
                    web_server.stop()
                except Exception as error:  # pragma: no cover - defensive cleanup
                    stop_error = error

            if socket_server is not None:
                try:
                    socket_server.stop()
                except Exception as error:  # pragma: no cover - defensive cleanup
                    if stop_error is None:
                        stop_error = error

            if stop_error is not None:
                raise stop_error

    def _broadcast(self, message: dict[str, object]) -> None:
        socket_server = self._socket_server
        if socket_server is None:
            raise RuntimeError("BrowserUiRuntime must be started before publishing UI messages")
        broadcast = getattr(socket_server, "broadcast", None)
        if not callable(broadcast):
            raise TypeError("socket_server must provide a callable broadcast method")
        broadcast(message)

    def _select_port(self, preferred_port: int, *, excluded_ports: frozenset[int]) -> int:
        selector = self._port_selector
        return selector(preferred_port, excluded_ports=excluded_ports)

    def _default_port_selector(self, preferred_port: int, *, excluded_ports: frozenset[int]) -> int:
        return _select_port(
            preferred_port,
            excluded_ports=excluded_ports,
            is_port_available=lambda port: _is_port_available(port, host=self._host),
        )

    def _browser_lan_ip(self) -> str | None:
        if self._host in {"127.0.0.1", "localhost"}:
            return None
        lan_ip = self._lan_ip_resolver()
        if lan_ip is None:
            return None
        normalized_ip = lan_ip.strip()
        if not normalized_ip or normalized_ip.startswith("127.") or normalized_ip == "0.0.0.0":
            return None
        if self._host not in {"0.0.0.0", "::"} and self._host != normalized_ip:
            return None
        return normalized_ip

    def _local_browser_host(self) -> str:
        if self._host in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
            return "127.0.0.1"
        return self._host

    def _websocket_host(self, *, lan_ip: str | None) -> str:
        if lan_ip is not None:
            return lan_ip
        return self._local_browser_host()

    def _websocket_template_host(self) -> str:
        if self._host in {"0.0.0.0", "::"}:
            return self._host
        return self._websocket_host(lan_ip=self._browser_lan_ip())

    def _initial_messages_provider(self) -> list[dict[str, object]]:
        messages = [self._publisher.snapshot_provider()]
        if self._task_feature_enabled:
            messages.append(self._publisher.task_snapshot_provider())
        return messages

    def _require_non_empty_text(self, field_name: str, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")

        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError(f"{field_name} must not be empty")
        return normalized_value

    def _require_port_number(self, field_name: str, value: int) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
        if value <= 0 or value > 65535:
            raise ValueError(f"{field_name} must be between 1 and 65535")
        return value


class _RuntimeTransportProxy:
    def __init__(self, runtime: BrowserUiRuntime) -> None:
        self._runtime = runtime

    def broadcast(self, message: dict[str, object]) -> None:
        self._runtime._broadcast(message)


def _select_port(
    preferred_port: int,
    *,
    excluded_ports: Collection[int] = (),
    is_port_available: Callable[[int], bool] | None = None,
) -> int:
    if not isinstance(preferred_port, int):
        raise ValueError("preferred_port must be an integer")
    if preferred_port <= 0 or preferred_port > 65535:
        raise ValueError("preferred_port must be between 1 and 65535")

    excluded = set(excluded_ports)
    availability_checker = is_port_available or _is_port_available

    if preferred_port not in excluded and availability_checker(preferred_port):
        return preferred_port

    for candidate_port in range(_DYNAMIC_PORT_RANGE_START, _DYNAMIC_PORT_RANGE_END + 1):
        if candidate_port in excluded:
            continue
        if availability_checker(candidate_port):
            return candidate_port

    raise RuntimeError("No available port found in the dynamic/private range")


def _is_port_available(port: int, *, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe_socket.bind((host, port))
        except OSError:
            return False
    return True


def _resolve_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
            probe_socket.connect(("10.255.255.255", 1))
            return str(probe_socket.getsockname()[0])
    except OSError:
        return None
