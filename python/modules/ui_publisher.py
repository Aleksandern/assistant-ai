from __future__ import annotations

"""Stateful bridge between pipeline UI events and browser socket transport."""

from modules.browser_ui_contract import (
    build_processing_error_message,
    build_reply_delta_message,
    build_reply_final_message,
    build_session_started_message,
    build_session_stopped_message,
    build_snapshot_message,
    build_task_snapshot_message,
    build_task_state_payload,
    build_transcript_message,
)


class UiPublisher:
    def __init__(self, *, transport: object) -> None:
        broadcast = getattr(transport, "broadcast", None)
        if not callable(broadcast):
            raise TypeError("transport must provide a callable broadcast method")

        self._transport = transport
        self._status = "idle"
        self._remote_text = ""
        self._reply_text = ""
        self._error: str | None = None
        self._task_state = build_task_state_payload(
            status="empty",
            file_count=0,
            artifacts=[],
            latest_result=None,
            error=None,
        )

    def snapshot_provider(self) -> dict[str, object]:
        return build_snapshot_message(
            status=self._status,
            remote_text=self._remote_text,
            reply_text=self._reply_text,
            error=self._error,
        )

    def task_snapshot_provider(self) -> dict[str, object]:
        return build_task_snapshot_message(
            status=self._task_state["status"],
            file_count=self._task_state["file_count"],
            artifacts=self._task_state["artifacts"],
            latest_result=self._task_state["latest_result"],
            error=self._task_state["error"],
        )

    def publish_session_started(self, *, local_url: str, lan_url: str | None) -> None:
        message = build_session_started_message(local_url=local_url, lan_url=lan_url)
        self._status = "listening"
        self._error = None
        self._transport.broadcast(message)

    def publish_transcript(self, remote_text: str) -> None:
        message = build_transcript_message(remote_text)
        self._remote_text = remote_text
        self._reply_text = ""
        self._error = None
        self._status = "processing"
        self._transport.broadcast(message)

    def publish_reply_delta(self, delta: str) -> None:
        message = build_reply_delta_message(delta)
        self._reply_text = f"{self._reply_text}{delta}"
        self._error = None
        self._status = "processing"
        self._transport.broadcast(message)

    def publish_reply_final(self, reply_text: str) -> None:
        message = build_reply_final_message(reply_text)
        self._reply_text = reply_text
        self._error = None
        self._status = "listening"
        self._transport.broadcast(message)

    def publish_processing_error(self, message: str) -> None:
        transport_message = build_processing_error_message(message)
        self._error = message
        self._status = "error"
        self._transport.broadcast(transport_message)

    def publish_session_stopped(self) -> None:
        self._status = "stopped"
        self._error = None
        message = build_session_stopped_message()
        self._transport.broadcast(message)

    def publish_task_snapshot(
        self,
        *,
        status: str,
        file_count: int,
        artifacts: list[dict[str, object]],
        latest_result: dict[str, object] | None,
        error: str | None,
    ) -> None:
        message = self._store_task_snapshot(
            status=status,
            file_count=file_count,
            artifacts=artifacts,
            latest_result=latest_result,
            error=error,
        )
        self._transport.broadcast(message)

    def hydrate_task_snapshot(
        self,
        *,
        status: str,
        file_count: int,
        artifacts: list[dict[str, object]],
        latest_result: dict[str, object] | None,
        error: str | None,
    ) -> None:
        self._store_task_snapshot(
            status=status,
            file_count=file_count,
            artifacts=artifacts,
            latest_result=latest_result,
            error=error,
        )

    def _store_task_snapshot(
        self,
        *,
        status: str,
        file_count: int,
        artifacts: list[dict[str, object]],
        latest_result: dict[str, object] | None,
        error: str | None,
    ) -> dict[str, object]:
        message = build_task_snapshot_message(
            status=status,
            file_count=file_count,
            artifacts=artifacts,
            latest_result=latest_result,
            error=error,
        )
        self._task_state = dict(message["payload"])
        return message
