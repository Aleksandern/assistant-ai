from __future__ import annotations

"""Generate a plain-text reply from ChatGPT for one input string.

Usage:
- call `generate_chatgpt_reply("Hello")`
- the module loads the repository-root `.env` file when present
- then sends the input text to OpenAI Responses API
- returns the assistant reply as plain text
"""

import os
from pathlib import Path
from typing import Any

from modules.openai_env_config import get_optional_str_env, get_required_str_env, load_dotenv_file


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
OPENAI_REPLY_INSTRUCTIONS_ENV_VAR = "OPENAI_REPLY_INSTRUCTIONS"
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_DOTENV_PATH = REPOSITORY_ROOT / ".env"


def generate_chatgpt_reply(
    input_text: str,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    instructions: str | None = None,
    dotenv_path: str | Path | None = None,
) -> str:
    normalized_input_text = input_text.strip()
    if not normalized_input_text:
        raise ValueError("Input text must not be empty.")

    _load_dotenv_file(dotenv_path=dotenv_path)

    normalized_model = _resolve_model_name(model=model)
    if not normalized_model:
        raise ValueError("Model name must not be empty.")

    resolved_instructions = _resolve_instructions(instructions=instructions)
    openai_client = client or _build_openai_client(api_key=api_key)

    try:
        response = openai_client.responses.create(
            model=normalized_model,
            input=normalized_input_text,
            instructions=resolved_instructions,
        )
    except Exception as exc:
        raise RuntimeError(f"ChatGPT reply generation failed: {exc}") from exc

    reply_text = _extract_output_text(response)
    if not reply_text:
        raise RuntimeError("ChatGPT reply generation failed: OpenAI returned an empty text response.")

    return reply_text


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
        OPENAI_REPLY_INSTRUCTIONS_ENV_VAR,
        guidance="Set it in the repository-root `.env` file or pass instructions explicitly.",
    )


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


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is None:
        return ""
    return str(output_text).strip()
