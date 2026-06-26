from __future__ import annotations

"""Create and persist a new OpenAI-backed conversation."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.file_docx_loader import find_first_file_docx
from modules.openai_env_config import get_optional_str_env, get_required_str_env, load_dotenv_file
from modules.sqlite_conversation_store import create_conversation, get_latest_openai_file_id


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR = "OPENAI_CONVERSATION_INSTRUCTIONS"
OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR = "OPENAI_CONVERSATION_FILE_MESSAGE"
OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR = "OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE"
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_DOTENV_PATH = REPOSITORY_ROOT / ".env"


@dataclass(frozen=True)
class InitializedConversationRecord:
    conversation_id: int
    openai_conversation_id: str
    topic_hint: str
    # `None` is expected on reuse paths that attach an existing OpenAI file id
    # from `.env` or the latest stored database value instead of uploading a local DOCX.
    file_name: str | None


@dataclass(frozen=True)
class _FileSourceSelection:
    source_type: str
    local_docx_path: Path | None = None
    openai_file_id: str | None = None


class OpenAIConversationInitializationError(RuntimeError):
    def __init__(self, message: str, *, openai_conversation_id: str | None = None) -> None:
        super().__init__(message)
        self.openai_conversation_id = openai_conversation_id


def initialize_openai_conversation(
    topic_hint: str,
    *,
    file_dir: str | Path | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dotenv_path: str | Path | None = None,
    database_path: str | Path | None = None,
    instructions: str | None = None,
) -> InitializedConversationRecord:
    """Create an OpenAI conversation and persist the linked local conversation row.

    Contract notes:
    - `file_name` contains the local DOCX filename only when a local file
      was found and uploaded during this call.
    - `file_name` is `None` when the initializer reuses an existing
      `OPENAI_FILE_ID` from `.env` or the latest stored
      `openai_file_id` from the local database.
    - If no local `.docx`, `.env` file id, or stored database file id is
      available, initialization fails with `OpenAIConversationInitializationError`.
    """
    normalized_topic_hint = topic_hint.strip()

    _load_dotenv_file(dotenv_path=dotenv_path)
    resolved_model = _resolve_model_name(model=model)
    if not resolved_model:
        raise ValueError("Model name must not be empty.")

    resolved_instructions = _resolve_instructions(instructions=instructions)
    openai_client = client or _build_openai_client(api_key=api_key)
    file_source = _select_file_source(
        file_dir=file_dir,
        database_path=database_path,
    )
    file_name = (
        file_source.local_docx_path.name if file_source.local_docx_path is not None else None
    )

    uploaded_file_id = file_source.openai_file_id
    if file_source.local_docx_path is not None:
        local_docx_bytes = _read_local_docx_bytes(file_source.local_docx_path)
        try:
            uploaded_file = openai_client.files.create(
                file=(file_name, local_docx_bytes),
                purpose="user_data",
            )
        except Exception as exc:
            raise OpenAIConversationInitializationError(
                f"Failed to upload DOCX file to OpenAI: {exc}"
            ) from exc

        uploaded_file_id = _extract_openai_file_id(uploaded_file)

    try:
        openai_conversation = openai_client.conversations.create(
            items=_build_initial_items(
                instructions=resolved_instructions,
                uploaded_file_id=uploaded_file_id,
                topic_hint=normalized_topic_hint,
            ),
            metadata=_build_conversation_metadata(
                topic_hint=normalized_topic_hint,
                file_name=file_name,
                model=resolved_model,
            ),
        )
    except Exception as exc:
        raise OpenAIConversationInitializationError(
            f"Failed to create or initialize OpenAI conversation: {exc}"
        ) from exc

    openai_conversation_id = _extract_openai_conversation_id(openai_conversation)

    try:
        local_conversation = create_conversation(
            normalized_topic_hint,
            openai_conversation_id=openai_conversation_id,
            openai_file_id=uploaded_file_id,
            is_active=True,
            database_path=database_path,
        )
    except Exception as exc:
        raise OpenAIConversationInitializationError(
            "OpenAI conversation was created, but failed to create local conversation row: "
            f"{exc}. openai_conversation_id={openai_conversation_id}",
            openai_conversation_id=openai_conversation_id,
        ) from exc

    if file_source.local_docx_path is not None:
        _delete_local_docx(
            local_docx_path=file_source.local_docx_path,
            openai_conversation_id=openai_conversation_id,
        )

    return InitializedConversationRecord(
        conversation_id=local_conversation.conversation_id,
        openai_conversation_id=openai_conversation_id,
        topic_hint=local_conversation.topic_hint,
        file_name=file_name,
    )


def _build_initial_items(
    *,
    instructions: str,
    uploaded_file_id: str,
    topic_hint: str,
) -> list[dict[str, object]]:
    user_content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": _resolve_file_message(),
        },
        {
            "type": "input_file",
            "file_id": uploaded_file_id,
        },
    ]
    if topic_hint:
        user_content.insert(
            1,
            {
                "type": "input_text",
                "text": _resolve_topic_hint_message(topic_hint=topic_hint),
            },
        )

    return [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": instructions}],
        },
        {
            "type": "message",
            "role": "user",
            "content": user_content,
        },
    ]


def _build_conversation_metadata(
    *,
    topic_hint: str,
    file_name: str | None,
    model: str,
) -> dict[str, str]:
    metadata = {"model": model}
    if file_name:
        metadata["file_name"] = file_name
    if topic_hint:
        metadata["topic_hint"] = topic_hint
    return metadata


def _select_file_source(
    *,
    file_dir: str | Path | None,
    database_path: str | Path | None,
) -> _FileSourceSelection:
    try:
        local_docx_path = find_first_file_docx(file_dir)
    except FileNotFoundError:
        local_docx_path = None

    if local_docx_path is not None:
        return _FileSourceSelection(
            source_type="docx",
            local_docx_path=local_docx_path,
        )

    configured_file_id = get_optional_str_env("OPENAI_FILE_ID")
    if configured_file_id:
        return _FileSourceSelection(
            source_type="env",
            openai_file_id=configured_file_id,
        )

    stored_file_id = get_latest_openai_file_id(database_path=database_path)
    if stored_file_id:
        return _FileSourceSelection(
            source_type="db",
            openai_file_id=stored_file_id,
        )

    raise OpenAIConversationInitializationError(
        "File source was not found: no local .docx file, no OPENAI_FILE_ID in .env, "
        "and no saved openai_file_id in the database."
    )


def _extract_openai_conversation_id(openai_conversation: Any) -> str:
    conversation_id = getattr(openai_conversation, "id", None)
    normalized_conversation_id = str(conversation_id or "").strip()
    if not normalized_conversation_id:
        raise OpenAIConversationInitializationError(
            "Failed to create or initialize OpenAI conversation: OpenAI returned an empty conversation id."
        )
    return normalized_conversation_id


def _extract_openai_file_id(uploaded_file: Any) -> str:
    file_id = getattr(uploaded_file, "id", None)
    normalized_file_id = str(file_id or "").strip()
    if not normalized_file_id:
        raise OpenAIConversationInitializationError(
            "Failed to upload local DOCX to OpenAI: OpenAI returned an empty file id."
        )
    return normalized_file_id


def _read_local_docx_bytes(local_docx_path: Path) -> bytes:
    try:
        return local_docx_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Failed to read local DOCX file: {local_docx_path}") from exc


def _delete_local_docx(
    *,
    local_docx_path: Path,
    openai_conversation_id: str,
) -> None:
    try:
        local_docx_path.unlink()
    except OSError as exc:
        raise OpenAIConversationInitializationError(
            "OpenAI conversation and local conversation row were created, but failed to delete "
            f"local DOCX file: {local_docx_path}. "
            f"openai_conversation_id={openai_conversation_id}",
            openai_conversation_id=openai_conversation_id,
        ) from exc


def _load_dotenv_file(*, dotenv_path: str | Path | None) -> None:
    load_dotenv_file(dotenv_path=dotenv_path, default_dotenv_path=DEFAULT_DOTENV_PATH)


def _resolve_model_name(*, model: str | None) -> str:
    if model is not None:
        return model.strip()

    configured_model = get_optional_str_env("OPENAI_MODEL")
    if configured_model:
        return configured_model

    return DEFAULT_OPENAI_MODEL


def _resolve_instructions(*, instructions: str | None) -> str:
    if instructions is not None:
        normalized_instructions = instructions.strip()
        if normalized_instructions:
            return normalized_instructions

    return get_required_str_env(
        OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR,
        guidance="Set it in the repository-root `.env` file or pass instructions explicitly.",
    )


def _resolve_file_message() -> str:
    return get_required_str_env(
        OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR,
        guidance="Set it in the repository-root `.env` file.",
    )


def _resolve_topic_hint_message(*, topic_hint: str) -> str:
    configured_template = get_required_str_env(
        OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR,
        guidance="Set it in the repository-root `.env` file.",
    )
    if configured_template.count("{topic_hint}") != 1:
        raise ValueError(
            f"{OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR} must contain the "
            "`{topic_hint}` placeholder exactly once."
        )

    try:
        return configured_template.format(topic_hint=topic_hint)
    except KeyError as exc:
        raise ValueError(
            f"{OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR} must contain the "
            "`{topic_hint}` placeholder."
        ) from exc


def _build_openai_client(*, api_key: str | None) -> Any:
    resolved_api_key = _resolve_api_key(api_key=api_key)
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key was not provided. Set OPENAI_API_KEY in the repository-root `.env` file "
            "or pass api_key explicitly."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI Python package is not installed. Add `openai` to the environment before using this module."
        ) from exc

    return OpenAI(api_key=resolved_api_key)


def _resolve_api_key(*, api_key: str | None) -> str:
    if api_key is not None:
        return api_key.strip()

    return get_optional_str_env("OPENAI_API_KEY") or ""
