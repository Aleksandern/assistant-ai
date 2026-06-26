from __future__ import annotations

"""JSON-compatible message builders for the browser UI contract."""


def build_snapshot_message(
    *,
    status: str,
    remote_text: str,
    reply_text: str,
    error: str | None,
) -> dict[str, object]:
    return _build_message(
        "snapshot",
        {
            "status": _require_non_empty_text("status", status),
            "remote_text": _require_string("remote_text", remote_text),
            "reply_text": _require_string("reply_text", reply_text),
            "error": _require_optional_string("error", error),
        },
    )


def build_session_started_message(*, local_url: str, lan_url: str | None) -> dict[str, object]:
    return _build_message(
        "session_started",
        {
            "status": "listening",
            "local_url": _require_non_empty_text("local_url", local_url),
            "lan_url": _require_optional_string("lan_url", lan_url),
        },
    )


def build_transcript_message(remote_text: str) -> dict[str, object]:
    return _build_message(
        "transcript",
        {
            "remote_text": _require_non_empty_text("remote_text", remote_text),
        },
    )


def build_reply_delta_message(delta: str) -> dict[str, object]:
    return _build_message(
        "reply_delta",
        {
            "delta": _require_non_empty_string("delta", delta),
        },
    )


def build_reply_final_message(reply_text: str) -> dict[str, object]:
    return _build_message(
        "reply_final",
        {
            "reply_text": _require_non_empty_text("reply_text", reply_text),
        },
    )


def build_processing_error_message(message: str) -> dict[str, object]:
    return _build_message(
        "processing_error",
        {
            "message": _require_non_empty_text("message", message),
        },
    )


def build_session_stopped_message() -> dict[str, object]:
    return _build_message(
        "session_stopped",
        {
            "status": "stopped",
        },
    )


def build_task_snapshot_message(
    *,
    status: str,
    file_count: int,
    artifacts: list[dict[str, object]],
    latest_result: dict[str, object] | None,
    error: str | None,
) -> dict[str, object]:
    return _build_message(
        "task_snapshot",
        build_task_state_payload(
            status=status,
            file_count=file_count,
            artifacts=artifacts,
            latest_result=latest_result,
            error=error,
        ),
    )


def build_task_state_payload(
    *,
    status: str,
    file_count: int,
    artifacts: list[dict[str, object]],
    latest_result: dict[str, object] | None,
    error: str | None,
) -> dict[str, object]:
    return {
        "status": _require_non_empty_text("status", status),
        "file_count": _require_non_negative_int("file_count", file_count),
        "artifacts": _require_artifacts(artifacts),
        "latest_result": _require_optional_latest_result(latest_result),
        "error": _require_optional_string("error", error),
    }


def _build_message(message_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "type": message_type,
        "payload": payload,
    }


def _require_non_empty_text(field_name: str, value: str) -> str:
    normalized_value = _require_string(field_name, value).strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty")
    return normalized_value


def _require_non_empty_string(field_name: str, value: str) -> str:
    raw_value = _require_string(field_name, value)
    if raw_value == "":
        raise ValueError(f"{field_name} must not be empty")
    return raw_value


def _require_string(field_name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_optional_string(field_name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None")
    return value


def _require_non_negative_int(field_name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_artifacts(artifacts: list[dict[str, object]]) -> list[dict[str, object]]:
    if not isinstance(artifacts, list):
        raise ValueError("artifacts must be a list")

    normalized_artifacts: list[dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("artifact must be a dict")

        normalized_artifact = {
            "id": _require_non_empty_text("artifact.id", artifact.get("id")),
            "kind": _require_non_empty_text("artifact.kind", artifact.get("kind")),
            "label": _require_non_empty_text("artifact.label", artifact.get("label")),
            "path": _require_non_empty_text("artifact.path", artifact.get("path")),
        }
        content_type = _require_optional_string("artifact.content_type", artifact.get("content_type"))
        if content_type is not None:
            normalized_artifact["content_type"] = content_type
        normalized_artifacts.append(normalized_artifact)

    return normalized_artifacts


def _require_optional_latest_result(latest_result: dict[str, object] | None) -> dict[str, object] | None:
    if latest_result is None:
        return None
    if not isinstance(latest_result, dict):
        raise ValueError("latest_result must be a dict or None")

    normalized_latest_result = {
        "name": _require_non_empty_text("latest_result.name", latest_result.get("name")),
        "status": _require_non_empty_text("latest_result.status", latest_result.get("status")),
        "summary": _require_optional_string("latest_result.summary", latest_result.get("summary")),
    }
    response_text = _require_optional_string("latest_result.response_text", latest_result.get("response_text"))
    if response_text is not None:
        normalized_latest_result["response_text"] = response_text
    return normalized_latest_result
