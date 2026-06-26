from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import unittest
from contextlib import AsyncExitStack
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.socket_server import SocketServer

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError


class SocketServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._servers: list[SocketServer] = []
        self.snapshot_message = {
            "type": "snapshot",
            "payload": {
                "status": "listening",
                "remote_text": "Remote hello",
                "reply_text": "Suggested reply",
                "error": None,
            },
        }

    async def asyncTearDown(self) -> None:
        for server in self._servers:
            server.stop()

    async def test_server_starts_and_stops(self) -> None:
        server = self._start_server()

        self.assertGreater(server.bound_port, 0)

        server.stop()

    async def test_new_client_receives_snapshot_immediately(self) -> None:
        server = self._start_server()

        async with connect(self._server_url(server)) as websocket:
            self.assertEqual(self.snapshot_message, await self._recv_json(websocket))

    async def test_broadcast_delivers_message_to_single_client(self) -> None:
        server = self._start_server()
        message = {
            "type": "transcript",
            "payload": {
                "remote_text": "How are you?",
            },
        }

        async with connect(self._server_url(server)) as websocket:
            await self._recv_json(websocket)

            server.broadcast(message)

            self.assertEqual(message, await self._recv_json(websocket))

    async def test_broadcast_delivers_message_to_multiple_clients(self) -> None:
        server = self._start_server()
        message = {
            "type": "reply_final",
            "payload": {
                "reply_text": "Doing well, thanks.",
            },
        }

        async with AsyncExitStack() as stack:
            first = await stack.enter_async_context(connect(self._server_url(server)))
            second = await stack.enter_async_context(connect(self._server_url(server)))

            self.assertEqual(self.snapshot_message, await self._recv_json(first))
            self.assertEqual(self.snapshot_message, await self._recv_json(second))

            server.broadcast(message)

            self.assertEqual(message, await self._recv_json(first))
            self.assertEqual(message, await self._recv_json(second))

    async def test_broadcast_is_safe_when_no_clients_are_connected(self) -> None:
        server = self._start_server()

        server.broadcast(
            {
                "type": "processing_error",
                "payload": {
                    "message": "Nothing connected",
                },
            }
        )

    async def test_broadcast_returns_before_async_send_completes(self) -> None:
        server = self._start_server()
        started = threading.Event()
        completed = threading.Event()

        original_async_broadcast = server._async_broadcast

        async def blocked_async_broadcast(message):
            started.set()
            try:
                await original_async_broadcast(message)
                await asyncio.Event().wait()
            finally:
                completed.set()

        server._async_broadcast = blocked_async_broadcast
        message = {
            "type": "reply_delta",
            "payload": {
                "delta": "slow-send",
            },
        }

        try:
            async with connect(self._server_url(server)) as websocket:
                await self._recv_json(websocket)

                server.broadcast(message)

                self.assertTrue(started.wait(timeout=1.0))
                self.assertFalse(completed.is_set())
                self.assertEqual(message, await self._recv_json(websocket))
        finally:
            server.stop()
            self._servers.remove(server)

    async def test_stop_waits_for_pending_broadcast_task_cleanup(self) -> None:
        server = self._start_server()
        started = threading.Event()
        completed = threading.Event()

        original_async_broadcast = server._async_broadcast

        async def blocked_async_broadcast(message):
            started.set()
            try:
                await original_async_broadcast(message)
                await asyncio.Event().wait()
            finally:
                completed.set()

        server._async_broadcast = blocked_async_broadcast
        message = {
            "type": "reply_final",
            "payload": {
                "reply_text": "pending",
            },
        }

        async with connect(self._server_url(server)) as websocket:
            await self._recv_json(websocket)

            server.broadcast(message)

            self.assertTrue(started.wait(timeout=1.0))
            self.assertEqual(message, await self._recv_json(websocket))

        stop_thread = threading.Thread(target=server.stop)
        stop_thread.start()
        stop_thread.join(timeout=1.0)

        self.assertFalse(stop_thread.is_alive())
        self.assertTrue(completed.wait(timeout=1.0))

    async def test_port_zero_exposes_real_bound_port_after_start(self) -> None:
        server = self._start_server(port=0)

        self.assertIsInstance(server.bound_port, int)
        self.assertGreater(server.bound_port, 0)

    async def test_stop_is_safe_when_called_twice(self) -> None:
        server = self._start_server()

        server.stop()
        server.stop()

    async def test_start_is_noop_when_called_twice(self) -> None:
        server = self._start_server()

        first_bound_port = server.bound_port
        server.start()

        self.assertEqual(first_bound_port, server.bound_port)

    async def test_disconnected_client_does_not_break_broadcast(self) -> None:
        server = self._start_server()
        message = {
            "type": "reply_delta",
            "payload": {
                "delta": "partial",
            },
        }

        async with AsyncExitStack() as stack:
            first = await stack.enter_async_context(connect(self._server_url(server)))
            second = await stack.enter_async_context(connect(self._server_url(server)))

            await self._recv_json(first)
            await self._recv_json(second)

            await first.close()
            await first.wait_closed()

            server.broadcast(message)

            self.assertEqual(message, await self._recv_json(second))

    async def test_broadcast_before_start_raises_runtime_error(self) -> None:
        server = SocketServer(snapshot_provider=lambda: self.snapshot_message)

        with self.assertRaisesRegex(RuntimeError, "SocketServer must be started before broadcast"):
            server.broadcast(self.snapshot_message)

    async def test_broadcast_during_shutdown_raises_runtime_error_instead_of_loop_error(self) -> None:
        server = self._start_server()
        original_schedule_broadcast = server._schedule_broadcast
        release = threading.Event()

        def blocked_schedule_broadcast(message):
            release.wait(timeout=1.0)
            original_schedule_broadcast(message)

        server._schedule_broadcast = blocked_schedule_broadcast

        async with connect(self._server_url(server)) as websocket:
            await self._recv_json(websocket)
            server.broadcast(
                {
                    "type": "transcript",
                    "payload": {
                        "remote_text": "before-stop",
                    },
                }
            )

        stop_thread = threading.Thread(target=server.stop)
        stop_thread.start()
        self.assertTrue(stop_thread.is_alive())

        try:
            with self.assertRaisesRegex(RuntimeError, "SocketServer must be started before broadcast"):
                server.broadcast(self.snapshot_message)
        finally:
            release.set()
            stop_thread.join(timeout=1.0)

        self.assertFalse(stop_thread.is_alive())

    async def test_stop_before_start_is_noop(self) -> None:
        server = SocketServer(snapshot_provider=lambda: self.snapshot_message)

        server.stop()

    async def test_invalid_snapshot_provider_result_raises_value_error_on_start(self) -> None:
        server = SocketServer(snapshot_provider=lambda: None)

        with self.assertRaisesRegex(ValueError, "snapshot_provider must provide a dict message"):
            server.start()

    async def test_invalid_snapshot_provider_result_after_start_closes_new_connection_explicitly(self) -> None:
        snapshot_provider = FlakySnapshotProvider(
            [
                self.snapshot_message,
                None,
            ]
        )
        server = SocketServer(snapshot_provider=snapshot_provider)
        server.start()
        self._servers.append(server)

        async with connect(self._server_url(server)) as websocket:
            with self.assertRaises(ConnectionClosedError) as context:
                await websocket.recv()

        self.assertEqual(1011, context.exception.rcvd.code)
        self.assertEqual("snapshot_provider returned invalid message", context.exception.rcvd.reason)

    async def test_snapshot_provider_exception_after_start_closes_new_connection_explicitly(self) -> None:
        snapshot_provider = FlakySnapshotProvider(
            [
                self.snapshot_message,
                RuntimeError("boom"),
            ]
        )
        server = SocketServer(snapshot_provider=snapshot_provider)
        server.start()
        self._servers.append(server)

        async with connect(self._server_url(server)) as websocket:
            with self.assertRaises(ConnectionClosedError) as context:
                await websocket.recv()

        self.assertEqual(1011, context.exception.rcvd.code)
        self.assertEqual("snapshot_provider returned invalid message", context.exception.rcvd.reason)

    async def test_start_propagates_bind_errors(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
            reserved_socket.bind(("127.0.0.1", 0))
            reserved_socket.listen()

            _, occupied_port = reserved_socket.getsockname()
            server = SocketServer(
                host="127.0.0.1",
                port=occupied_port,
                snapshot_provider=lambda: self.snapshot_message,
            )

            with self.assertRaises(OSError):
                server.start()

    async def test_broadcast_preserves_payload_without_hidden_mutation(self) -> None:
        server = self._start_server()
        message = {
            "type": "transcript",
            "payload": {
                "remote_text": "Exact text",
                "parts": ["a", 1, True, None],
            },
        }

        async with connect(self._server_url(server)) as websocket:
            await self._recv_json(websocket)

            server.broadcast(message)

            self.assertEqual(message, await self._recv_json(websocket))

    async def test_snapshot_is_only_sent_on_connect_not_on_broadcast(self) -> None:
        server = self._start_server()
        broadcast_message = {
            "type": "session_stopped",
            "payload": {
                "status": "stopped",
            },
        }

        async with connect(self._server_url(server)) as websocket:
            self.assertEqual(self.snapshot_message, await self._recv_json(websocket))

            server.broadcast(broadcast_message)

            self.assertEqual(broadcast_message, await self._recv_json(websocket))

    async def test_initial_messages_provider_sends_conversation_and_task_snapshots_on_connect(self) -> None:
        server = SocketServer(
            snapshot_provider=lambda: self.snapshot_message,
            initial_messages_provider=lambda: [
                self.snapshot_message,
                {
                    "type": "task_snapshot",
                    "payload": {
                        "status": "ready",
                        "file_count": 1,
                        "artifacts": [],
                        "latest_result": None,
                        "error": None,
                    },
                },
            ],
        )
        server.start()
        self._servers.append(server)

        async with connect(self._server_url(server)) as websocket:
            self.assertEqual(self.snapshot_message, await self._recv_json(websocket))
            self.assertEqual(
                {
                    "type": "task_snapshot",
                    "payload": {
                        "status": "ready",
                        "file_count": 1,
                        "artifacts": [],
                        "latest_result": None,
                        "error": None,
                    },
                },
                await self._recv_json(websocket),
            )

    def _start_server(self, *, host: str = "127.0.0.1", port: int = 0) -> SocketServer:
        server = SocketServer(
            host=host,
            port=port,
            snapshot_provider=lambda: self.snapshot_message,
        )
        server.start()
        self._servers.append(server)
        return server

    def _server_url(self, server: SocketServer) -> str:
        return f"ws://127.0.0.1:{server.bound_port}"

    async def _recv_json(self, websocket) -> dict[str, object]:
        raw_message = await websocket.recv()
        self.assertIsInstance(raw_message, str)
        return json.loads(raw_message)


if __name__ == "__main__":
    unittest.main()


class FlakySnapshotProvider:
    def __init__(self, results: list[object]) -> None:
        self._results = list(results)

    def __call__(self) -> object:
        if not self._results:
            raise AssertionError("No more snapshot results configured")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
