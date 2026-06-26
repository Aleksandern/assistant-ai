from __future__ import annotations

"""WebSocket transport for broadcasting browser UI contract messages."""

import asyncio
import json
import threading
from collections.abc import Callable

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed


Message = dict[str, object]


class SocketServer:
    """Small synchronous wrapper around an asyncio WebSocket server."""

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[], Message],
        initial_messages_provider: Callable[[], list[Message]] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if not callable(snapshot_provider):
            raise TypeError("snapshot_provider must be callable")
        if initial_messages_provider is not None and not callable(initial_messages_provider):
            raise TypeError("initial_messages_provider must be callable")

        self._snapshot_provider = snapshot_provider
        self._initial_messages_provider = initial_messages_provider or self._default_initial_messages_provider
        self._host = host
        self._requested_port = port

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._clients: set[ServerConnection] = set()
        self._broadcast_tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = threading.Lock()
        self._bound_port = 0
        self._stopping = False

    @property
    def bound_port(self) -> int:
        return self._bound_port

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._server is not None:
                return

            self._validate_message(self._snapshot_provider(), source="snapshot_provider")

            self._loop = asyncio.new_event_loop()
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run_loop,
                name="browser-ui-socket-server",
                daemon=True,
            )
            self._thread.start()

            try:
                asyncio.run_coroutine_threadsafe(self._async_start(), self._loop).result()
            except Exception:
                self._shutdown_loop_thread()
                raise

    def broadcast(self, message: Message) -> None:
        self._validate_message(message, source="message")

        with self._lifecycle_lock:
            if self._server is None or self._loop is None or self._stopping:
                raise RuntimeError("SocketServer must be started before broadcast")
            loop = self._loop

            try:
                loop.call_soon_threadsafe(self._schedule_broadcast, message)
            except RuntimeError as error:
                raise RuntimeError("SocketServer must be started before broadcast") from error

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._loop is None or self._thread is None:
                return

            loop = self._loop
            thread = self._thread

            if self._server is not None:
                self._stopping = True
                asyncio.run_coroutine_threadsafe(self._async_stop(), loop).result()

            self._server = None
            self._clients.clear()
            self._broadcast_tasks.clear()
            self._bound_port = 0
            self._loop = None
            self._thread = None
            self._stopping = False

            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _async_start(self) -> None:
        self._server = await serve(self._handle_client, self._host, self._requested_port)
        sockets = getattr(self._server, "sockets", None) or []
        if not sockets:
            raise RuntimeError("SocketServer failed to expose bound sockets")
        self._bound_port = int(sockets[0].getsockname()[1])

    async def _handle_client(self, websocket: ServerConnection) -> None:
        self._clients.add(websocket)
        try:
            try:
                initial_messages = self._validate_messages(
                    self._initial_messages_provider(),
                    source="initial_messages_provider",
                )
            except Exception:
                await websocket.close(code=1011, reason="snapshot_provider returned invalid message")
                return

            for message in initial_messages:
                await websocket.send(json.dumps(message))
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    async def _async_broadcast(self, message: Message) -> None:
        if not self._clients:
            return

        serialized_message = json.dumps(message)
        disconnected_clients: list[ServerConnection] = []

        for websocket in tuple(self._clients):
            try:
                await websocket.send(serialized_message)
            except ConnectionClosed:
                disconnected_clients.append(websocket)

        for websocket in disconnected_clients:
            self._clients.discard(websocket)

    async def _async_stop(self) -> None:
        pending_broadcasts = tuple(self._broadcast_tasks)
        for task in pending_broadcasts:
            task.cancel()
        if pending_broadcasts:
            await asyncio.gather(*pending_broadcasts, return_exceptions=True)

        clients = tuple(self._clients)

        for websocket in clients:
            websocket.close()
        if clients:
            await asyncio.gather(*(websocket.wait_closed() for websocket in clients), return_exceptions=True)

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def _shutdown_loop_thread(self) -> None:
        if self._loop is None or self._thread is None:
            return

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        self._loop = None
        self._thread = None
        self._server = None
        self._broadcast_tasks.clear()
        self._bound_port = 0
        self._stopping = False

    def _schedule_broadcast(self, message: Message) -> None:
        if self._loop is None or self._server is None or self._stopping:
            return

        task = self._loop.create_task(self._run_broadcast(message))
        self._broadcast_tasks.add(task)
        task.add_done_callback(self._broadcast_tasks.discard)

    async def _run_broadcast(self, message: Message) -> None:
        try:
            await self._async_broadcast(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _validate_message(self, message: object, *, source: str) -> Message:
        if not isinstance(message, dict):
            raise ValueError(f"{source} must provide a dict message")
        if "type" not in message or "payload" not in message:
            raise ValueError(f"{source} message must contain 'type' and 'payload'")

        json.dumps(message)
        return message

    def _validate_messages(self, messages: object, *, source: str) -> list[Message]:
        if not isinstance(messages, list):
            raise ValueError(f"{source} must provide a list of messages")
        return [self._validate_message(message, source=source) for message in messages]

    def _default_initial_messages_provider(self) -> list[Message]:
        return [self._snapshot_provider()]
