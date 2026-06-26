from __future__ import annotations

"""Continue an existing OpenAI conversation and return the model reply text."""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

from modules.openai_env_config import (
    get_optional_bool_env,
    get_optional_positive_int_env,
    get_optional_str_env,
    get_required_str_env,
    load_dotenv_file,
    validate_positive_int_env_value,
)
from modules.sqlite_conversation_store import get_conversation


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OPENAI_PROMPT_CACHE_KEY_PREFIX = "assistantai"
DEFAULT_OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS = 24
DEFAULT_OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS = 4
DEFAULT_OPENAI_REASONING_EFFORT = "minimal"
OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR = "OPENAI_CONVERSATION_REPLY_INSTRUCTIONS"
SUPPORTED_OPENAI_SERVICE_TIERS = {"priority"}
SUPPORTED_OPENAI_PROMPT_CACHE_RETENTION_BY_INPUT = {
    "in-memory": "in-memory",
    "in_memory": "in-memory",
    "24h": "24h",
}
LOGGER = logging.getLogger(__name__)
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_DOTENV_PATH = REPOSITORY_ROOT / ".env"
_DOTENV_LOAD_LOCK = threading.Lock()
_OPENAI_CLIENT_LOCK = threading.Lock()
_LOADED_DOTENV_PATHS: set[Path] = set()
_OPENAI_CLIENTS_BY_API_KEY: dict[str, Any] = {}


@dataclass(frozen=True)
class OpenAIConversationReplyRecord:
    conversation_id: int
    openai_conversation_id: str
    reply_text: str
    response_id: str | None = None
    timings: OpenAIConversationReplyTimings | None = None


@dataclass(frozen=True)
class OpenAIConversationReplyTimings:
    ttft_ms: int | None
    ttfut_ms: int | None
    full_ms: int
    cached_tokens: int | None = None


@dataclass(frozen=True)
class _PromptCacheConfig:
    key: str
    retention: str | None = None


@dataclass(frozen=True)
class _UsefulTextThresholdConfig:
    min_chars: int
    min_words: int


@dataclass(frozen=True)
class _StreamResponseResult:
    response: Any
    ttft_ms: int | None
    ttfut_ms: int | None


class OpenAIConversationReplyGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        conversation_id: int | None = None,
        openai_conversation_id: str | None = None,
        response_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.conversation_id = conversation_id
        self.openai_conversation_id = openai_conversation_id
        self.response_id = response_id


def generate_reply_in_openai_conversation(
    conversation_id: int,
    input_text: str,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dotenv_path: str | Path | None = None,
    database_path: str | Path | None = None,
    instructions: str | None = None,
    debug: bool = False,
    fast_mode: bool = False,
    fast_model: str | None = None,
    service_tier: str | None = None,
    use_conversation: bool = True,
    disable_instructions: bool = False,
    minimal_instructions: str | None = None,
    stream: bool = False,
    on_text_delta: Callable[[str], None] | None = None,
) -> OpenAIConversationReplyRecord:
    normalized_input_text = input_text.strip()
    if not normalized_input_text:
        raise ValueError("Input text must not be empty.")

    local_conversation = get_conversation(conversation_id, database_path=database_path)
    if local_conversation is None:
        raise OpenAIConversationReplyGenerationError(
            f"Local conversation was not found for conversation_id={conversation_id}.",
            conversation_id=conversation_id,
        )

    openai_conversation_id = (local_conversation.openai_conversation_id or "").strip()
    if use_conversation and not openai_conversation_id:
        raise OpenAIConversationReplyGenerationError(
            "Local conversation does not have an OpenAI conversation id. "
            f"conversation_id={conversation_id}",
            conversation_id=conversation_id,
        )

    _load_dotenv_file(dotenv_path=dotenv_path)

    request_preparation_started_at = time.perf_counter()
    normalized_model = _resolve_model_name(model=model, fast_mode=fast_mode, fast_model=fast_model)
    if not normalized_model:
        raise ValueError("Model name must not be empty.")

    resolved_instructions = _resolve_instructions(
        instructions=instructions,
        disable_instructions=disable_instructions,
        fast_mode=fast_mode,
        minimal_instructions=minimal_instructions,
    )
    resolved_service_tier = _resolve_service_tier(service_tier=service_tier)
    prompt_cache_config = _resolve_prompt_cache_config(conversation_id=conversation_id)
    resolved_max_output_tokens = _resolve_max_output_tokens()
    reasoning_config = _resolve_reasoning_config(model_name=normalized_model, stream=stream)
    useful_text_threshold_config = _resolve_first_useful_text_threshold_config()
    openai_client = client or _build_openai_client(api_key=api_key)

    request_payload: dict[str, object] = {
        "model": normalized_model,
        "input": normalized_input_text,
    }
    if use_conversation:
        request_payload["conversation"] = openai_conversation_id
    if resolved_instructions is not None:
        request_payload["instructions"] = resolved_instructions
    if resolved_service_tier is not None:
        request_payload["service_tier"] = resolved_service_tier
    if prompt_cache_config is not None:
        request_payload["prompt_cache_key"] = prompt_cache_config.key
        if prompt_cache_config.retention is not None:
            request_payload["prompt_cache_retention"] = prompt_cache_config.retention
    if resolved_max_output_tokens is not None:
        request_payload["max_output_tokens"] = resolved_max_output_tokens
    if reasoning_config is not None:
        request_payload["reasoning"] = reasoning_config
    request_preparation_ms = _elapsed_ms(request_preparation_started_at)

    api_call_started_at = time.perf_counter()
    ttft_ms: int | None = None
    ttfut_ms: int | None = None
    try:
        if stream:
            stream_result = _create_streamed_response(
                openai_client=openai_client,
                request_payload=request_payload,
                on_text_delta=on_text_delta,
                api_call_started_at=api_call_started_at,
                useful_text_threshold_config=useful_text_threshold_config,
                debug=debug,
            )
            response = stream_result.response
            ttft_ms = stream_result.ttft_ms
            ttfut_ms = stream_result.ttfut_ms
        else:
            response = openai_client.responses.create(**request_payload)
    except Exception as exc:
        if debug:
            LOGGER.info(
                "OpenAI conversation reply debug model=%s input_chars=%d instructions_present=%s instructions_chars=%d "
                "conversation_id=%s openai_conversation_id=%s prompt_cache_key=%s prompt_cache_retention=%s service_tier=%s "
                "use_conversation=%s fast_mode=%s max_output_tokens=%s request_prep_ms=%d responses_create_ms=%d "
                "response_id=%s cached_tokens=%s reply_chars=%d ttft_ms=%s ttfut_ms=%s error=%s",
                normalized_model,
                len(normalized_input_text),
                resolved_instructions is not None,
                len(resolved_instructions or ""),
                conversation_id,
                openai_conversation_id,
                prompt_cache_config.key if prompt_cache_config is not None else None,
                prompt_cache_config.retention if prompt_cache_config is not None else None,
                resolved_service_tier,
                use_conversation,
                fast_mode,
                resolved_max_output_tokens,
                request_preparation_ms,
                _elapsed_ms(api_call_started_at),
                None,
                None,
                0,
                ttft_ms,
                ttfut_ms,
                str(exc),
            )
        raise OpenAIConversationReplyGenerationError(
            "Failed to continue OpenAI conversation: "
            f"{exc}. conversation_id={conversation_id}, openai_conversation_id={openai_conversation_id}",
            conversation_id=conversation_id,
            openai_conversation_id=openai_conversation_id,
        ) from exc
    responses_create_ms = _elapsed_ms(api_call_started_at)

    reply_text = _extract_output_text(response)
    response_id = _extract_response_id(response)
    response_status = _extract_response_status(response)
    incomplete_reason = _extract_incomplete_reason(response)
    cached_tokens = _extract_cached_tokens(response)
    if debug:
        LOGGER.info(
            "OpenAI conversation reply debug model=%s input_chars=%d instructions_present=%s instructions_chars=%d "
            "conversation_id=%s openai_conversation_id=%s prompt_cache_key=%s prompt_cache_retention=%s service_tier=%s "
            "use_conversation=%s fast_mode=%s max_output_tokens=%s reasoning_effort=%s request_prep_ms=%d responses_create_ms=%d "
            "response_id=%s response_status=%s incomplete_reason=%s cached_tokens=%s reply_chars=%d ttft_ms=%s ttfut_ms=%s",
            normalized_model,
            len(normalized_input_text),
            resolved_instructions is not None,
            len(resolved_instructions or ""),
            conversation_id,
            openai_conversation_id,
            prompt_cache_config.key if prompt_cache_config is not None else None,
            prompt_cache_config.retention if prompt_cache_config is not None else None,
                resolved_service_tier,
                use_conversation,
                fast_mode,
                resolved_max_output_tokens,
                reasoning_config["effort"] if reasoning_config is not None else None,
                request_preparation_ms,
                responses_create_ms,
                response_id,
            response_status,
            incomplete_reason,
            cached_tokens,
            len(reply_text),
            ttft_ms,
            ttfut_ms,
        )
    if not reply_text:
        details_suffix = _build_empty_response_details_suffix(
            response_status=response_status,
            incomplete_reason=incomplete_reason,
        )
        raise OpenAIConversationReplyGenerationError(
            "Failed to continue OpenAI conversation: OpenAI returned an empty text response. "
            f"conversation_id={conversation_id}, openai_conversation_id={openai_conversation_id}{details_suffix}",
            conversation_id=conversation_id,
            openai_conversation_id=openai_conversation_id,
            response_id=response_id,
        )

    return OpenAIConversationReplyRecord(
        conversation_id=conversation_id,
        openai_conversation_id=openai_conversation_id,
        reply_text=reply_text,
        response_id=response_id,
        timings=OpenAIConversationReplyTimings(
            ttft_ms=ttft_ms,
            ttfut_ms=ttfut_ms,
            full_ms=responses_create_ms,
            cached_tokens=cached_tokens,
        ),
    )


def _load_dotenv_file(*, dotenv_path: str | Path | None) -> None:
    load_dotenv_file(
        dotenv_path=dotenv_path,
        default_dotenv_path=DEFAULT_DOTENV_PATH,
        loaded_paths=_LOADED_DOTENV_PATHS,
        lock=_DOTENV_LOAD_LOCK,
        dotenv_text_reader=_read_dotenv_text,
    )


def _resolve_model_name(*, model: str | None, fast_mode: bool, fast_model: str | None) -> str:
    if fast_mode and fast_model is not None:
        return fast_model.strip()

    if model is not None:
        return model.strip()

    configured_model = get_optional_str_env("OPENAI_MODEL")
    if configured_model:
        return configured_model

    return DEFAULT_OPENAI_MODEL


def _resolve_instructions(
    *,
    instructions: str | None,
    disable_instructions: bool,
    fast_mode: bool,
    minimal_instructions: str | None,
) -> str | None:
    if disable_instructions:
        return None

    if fast_mode and minimal_instructions is not None:
        normalized_minimal_instructions = minimal_instructions.strip()
        return normalized_minimal_instructions or None

    if instructions is not None:
        normalized_instructions = instructions.strip()
        if normalized_instructions:
            return normalized_instructions

    return get_required_str_env(
        OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR,
        guidance="Set it in the repository-root `.env` file or pass instructions explicitly.",
    )


def _resolve_service_tier(*, service_tier: str | None) -> str | None:
    if service_tier is None:
        service_tier = get_optional_str_env("OPENAI_SERVICE_TIER")

    if service_tier is None:
        return None

    normalized_service_tier = service_tier.strip()
    if not normalized_service_tier:
        return None

    if normalized_service_tier not in SUPPORTED_OPENAI_SERVICE_TIERS:
        supported_values = ", ".join(sorted(SUPPORTED_OPENAI_SERVICE_TIERS))
        raise ValueError(
            "Unsupported OPENAI_SERVICE_TIER value "
            f"{normalized_service_tier!r}. Supported values: {supported_values}."
        )

    return normalized_service_tier


def _resolve_prompt_cache_config(*, conversation_id: int) -> _PromptCacheConfig | None:
    prompt_cache_enabled = get_optional_bool_env("OPENAI_PROMPT_CACHE_ENABLED")
    if prompt_cache_enabled is not True:
        return None

    prompt_cache_key_prefix = _resolve_prompt_cache_key_prefix()
    prompt_cache_retention = _resolve_prompt_cache_retention()
    return _PromptCacheConfig(
        key=f"{prompt_cache_key_prefix}:conversation:{conversation_id}",
        retention=prompt_cache_retention,
    )

def _resolve_prompt_cache_key_prefix() -> str:
    raw_prefix = os.getenv("OPENAI_PROMPT_CACHE_KEY_PREFIX")
    if raw_prefix is None:
        return DEFAULT_OPENAI_PROMPT_CACHE_KEY_PREFIX
    normalized_prefix = raw_prefix.strip()
    if not normalized_prefix:
        raise ValueError(
            "OPENAI_PROMPT_CACHE_KEY_PREFIX must not be empty when prompt caching is enabled."
        )

    return normalized_prefix


def _resolve_prompt_cache_retention() -> str | None:
    normalized_retention = get_optional_str_env("OPENAI_PROMPT_CACHE_RETENTION")
    if not normalized_retention:
        return None

    resolved_retention = SUPPORTED_OPENAI_PROMPT_CACHE_RETENTION_BY_INPUT.get(normalized_retention)
    if resolved_retention is None:
        supported_values = ", ".join(sorted(SUPPORTED_OPENAI_PROMPT_CACHE_RETENTION_BY_INPUT))
        raise ValueError(
            "Unsupported OPENAI_PROMPT_CACHE_RETENTION value "
            f"{normalized_retention!r}. Supported values: {supported_values}."
        )

    return resolved_retention


def _resolve_max_output_tokens() -> int | None:
    return get_optional_positive_int_env("OPENAI_MAX_OUTPUT_TOKENS")


def _resolve_reasoning_config(*, model_name: str, stream: bool) -> dict[str, str] | None:
    if not stream:
        return None

    normalized_model_name = model_name.strip().lower()
    if not normalized_model_name.startswith("gpt-5"):
        return None

    return {"effort": DEFAULT_OPENAI_REASONING_EFFORT}


def _resolve_first_useful_text_threshold_config() -> _UsefulTextThresholdConfig:
    return _UsefulTextThresholdConfig(
        min_chars=_resolve_positive_int_env_with_default(
            "OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS",
            default_value=DEFAULT_OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS,
        ),
        min_words=_resolve_positive_int_env_with_default(
            "OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS",
            default_value=DEFAULT_OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS,
        ),
    )


def _resolve_positive_int_env_with_default(env_var_name: str, *, default_value: int) -> int:
    configured_value = get_optional_positive_int_env(env_var_name)
    if configured_value is None:
        return default_value
    return configured_value


def _validate_max_output_tokens(value: int | str) -> int:
    return validate_positive_int_env_value("OPENAI_MAX_OUTPUT_TOKENS", value)


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

    with _OPENAI_CLIENT_LOCK:
        cached_client = _OPENAI_CLIENTS_BY_API_KEY.get(resolved_api_key)
        if cached_client is not None:
            return cached_client

        openai_client = _instantiate_openai_client(OpenAI=OpenAI, api_key=resolved_api_key)
        _OPENAI_CLIENTS_BY_API_KEY[resolved_api_key] = openai_client
        return openai_client


def _resolve_api_key(*, api_key: str | None) -> str:
    if api_key is not None:
        return api_key.strip()

    return get_optional_str_env("OPENAI_API_KEY") or ""


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text).strip()

    extracted_output_text = _extract_output_text_from_output_items(response)
    if extracted_output_text:
        return extracted_output_text

    return ""


def _extract_response_id(response: Any) -> str | None:
    response_id = str(getattr(response, "id", "") or "").strip()
    return response_id or None


def _extract_response_status(response: Any) -> str | None:
    response_status = str(getattr(response, "status", "") or "").strip()
    return response_status or None


def _extract_incomplete_reason(response: Any) -> str | None:
    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details is None:
        return None

    incomplete_reason = str(getattr(incomplete_details, "reason", "") or "").strip()
    return incomplete_reason or None


def _extract_cached_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_tokens_details is None:
        return None

    cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
    if cached_tokens is None:
        return None

    try:
        return int(cached_tokens)
    except (TypeError, ValueError):
        return None


def _create_streamed_response(
    *,
    openai_client: Any,
    request_payload: dict[str, object],
    on_text_delta: Callable[[str], None] | None,
    api_call_started_at: float,
    useful_text_threshold_config: _UsefulTextThresholdConfig,
    debug: bool,
) -> _StreamResponseResult:
    stream = openai_client.responses.create(**request_payload, stream=True)
    accumulated_text_parts: list[str] = []
    response: Any | None = None
    ttft_ms: int | None = None
    ttfut_ms: int | None = None
    saw_output_text_delta = False

    for event in stream:
        event_type = str(getattr(event, "type", "") or "")
        if debug:
            _log_stream_event_debug(event)

        if event_type == "response.output_text.delta":
            saw_output_text_delta = True
            delta = _extract_stream_text_delta(event)
            if delta:
                if ttft_ms is None:
                    ttft_ms = _elapsed_ms(api_call_started_at)
                accumulated_text_parts.append(delta)
                if ttfut_ms is None and _is_useful_text(
                    "".join(accumulated_text_parts),
                    useful_text_threshold_config=useful_text_threshold_config,
                ):
                    ttfut_ms = _elapsed_ms(api_call_started_at)
                if on_text_delta is not None:
                    on_text_delta(delta)
            continue

        if event_type == "response.output_text.done":
            if saw_output_text_delta:
                continue

            text = _extract_stream_text_delta(event)
            if text:
                if ttft_ms is None:
                    ttft_ms = _elapsed_ms(api_call_started_at)
                accumulated_text_parts.append(text)
                if ttfut_ms is None and _is_useful_text(
                    "".join(accumulated_text_parts),
                    useful_text_threshold_config=useful_text_threshold_config,
                ):
                    ttfut_ms = _elapsed_ms(api_call_started_at)
                if on_text_delta is not None:
                    on_text_delta(text)
            continue

        if event_type == "response.completed":
            completed_response = getattr(event, "response", None)
            if debug:
                LOGGER.info(
                    "OpenAI stream debug stream_response_completed_has_response=%s",
                    completed_response is not None,
                )
            if completed_response is not None:
                response = completed_response
            continue

        if event_type == "response.incomplete":
            incomplete_response = getattr(event, "response", None)
            if debug:
                LOGGER.info(
                    "OpenAI stream debug stream_response_incomplete_has_response=%s",
                    incomplete_response is not None,
                )
            if incomplete_response is not None:
                response = incomplete_response
            continue

        if debug:
            LOGGER.info(
                "OpenAI stream debug unhandled_stream_event_type=%s",
                event_type or "<missing>",
            )

    if response is None:
        response = _build_streamed_response_fallback(
            stream=stream,
            accumulated_text="".join(accumulated_text_parts),
            debug=debug,
        )

    return _StreamResponseResult(response=response, ttft_ms=ttft_ms, ttfut_ms=ttfut_ms)


def _is_useful_text(
    accumulated_text: str,
    *,
    useful_text_threshold_config: _UsefulTextThresholdConfig,
) -> bool:
    normalized_text = _normalize_useful_text(accumulated_text)
    if not normalized_text:
        return False

    if len(normalized_text) >= useful_text_threshold_config.min_chars:
        return True

    return _count_normalized_words(normalized_text) >= useful_text_threshold_config.min_words


def _normalize_useful_text(text: str) -> str:
    return " ".join(text.split())


def _count_normalized_words(normalized_text: str) -> int:
    if not normalized_text:
        return 0
    return len(normalized_text.split(" "))


def _elapsed_ms(started_at_monotonic: float) -> int:
    return int(round((time.perf_counter() - started_at_monotonic) * 1000))


def _read_dotenv_text(dotenv_path: Path) -> str:
    return dotenv_path.read_text(encoding="utf-8")


def _instantiate_openai_client(*, OpenAI: Any, api_key: str) -> Any:
    return OpenAI(api_key=api_key)


def _extract_stream_text_delta(event: Any) -> str:
    delta = getattr(event, "delta", None)
    if delta is not None:
        return str(delta)

    text = getattr(event, "text", None)
    if text is not None:
        return str(text)

    return ""


def _extract_output_text_from_output_items(response: Any) -> str:
    output_items = getattr(response, "output", None)
    if output_items is None:
        return ""

    text_parts: list[str] = []
    for output_item in output_items:
        if str(getattr(output_item, "type", "") or "") != "message":
            continue

        content_items = getattr(output_item, "content", None)
        if content_items is None:
            continue

        for content_item in content_items:
            if str(getattr(content_item, "type", "") or "") != "output_text":
                continue

            text = getattr(content_item, "text", None)
            if text is None:
                continue

            normalized_text = str(text)
            if normalized_text:
                text_parts.append(normalized_text)

    return "".join(text_parts).strip()


def _build_empty_response_details_suffix(
    *,
    response_status: str | None,
    incomplete_reason: str | None,
) -> str:
    detail_parts: list[str] = []
    if response_status:
        detail_parts.append(f"response_status={response_status}")
    if incomplete_reason:
        detail_parts.append(f"incomplete_reason={incomplete_reason}")

    if not detail_parts:
        return ""

    return ", " + ", ".join(detail_parts)


def _log_stream_event_debug(event: Any) -> None:
    event_attrs = []
    for attr_name in ("delta", "text", "response", "item"):
        if hasattr(event, attr_name):
            event_attrs.append(attr_name)

    LOGGER.info(
        "OpenAI stream debug stream_event_type=%s stream_event_has_delta=%s stream_event_has_text=%s "
        "stream_event_has_response=%s stream_event_attrs=%s",
        str(getattr(event, "type", "") or "<missing>"),
        getattr(event, "delta", None) is not None,
        getattr(event, "text", None) is not None,
        getattr(event, "response", None) is not None,
        ",".join(event_attrs) or "-",
    )


def _build_streamed_response_fallback(*, stream: Any, accumulated_text: str, debug: bool) -> Any:
    get_final_response = getattr(stream, "get_final_response", None)
    if callable(get_final_response):
        final_response = get_final_response()
        if debug:
            LOGGER.info(
                "OpenAI stream debug stream_get_final_response_called=True stream_get_final_response_has_response=%s",
                final_response is not None,
            )
        if final_response is not None:
            return final_response
    elif debug:
        LOGGER.info("OpenAI stream debug stream_get_final_response_called=False")

    return _SyntheticStreamedResponse(output_text=accumulated_text)


class _SyntheticStreamedResponse:
    def __init__(self, *, output_text: str, response_id: str | None = None) -> None:
        self.output_text = output_text
        self.id = response_id


def _reset_runtime_caches_for_tests() -> None:
    with _DOTENV_LOAD_LOCK:
        _LOADED_DOTENV_PATHS.clear()
    with _OPENAI_CLIENT_LOCK:
        _OPENAI_CLIENTS_BY_API_KEY.clear()
