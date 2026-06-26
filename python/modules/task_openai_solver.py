from __future__ import annotations

"""Solve a screenshot-based coding task through an isolated OpenAI boundary."""

import base64
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from modules.openai_env_config import get_optional_str_env, get_required_str_env, load_dotenv_file
from modules.task_artifact_store import TASK_ARTIFACT_SUFFIXES


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
OPENAI_TASK_SOLVER_PROMPT_ENV_VAR = "OPENAI_TASK_SOLVER_PROMPT"
PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_DOTENV_PATH = REPOSITORY_ROOT / ".env"


class _PathCarrier(Protocol):
    path: str | Path


class OpenAIResponsesClient(Protocol):
    responses: Any


ScreenshotInput: TypeAlias = str | Path | Mapping[str, object] | _PathCarrier


@dataclass(frozen=True)
class TaskOpenAISolveResult:
    response_text: str
    source_file_count: int


class TaskOpenAISolverError(RuntimeError):
    """Raised when the screenshot solve boundary cannot return a usable result."""


def solve_task_from_screenshots(
    screenshots: Sequence[ScreenshotInput],
    *,
    client: OpenAIResponsesClient | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dotenv_path: str | Path | None = None,
) -> TaskOpenAISolveResult:
    normalized_screenshot_paths = _normalize_screenshot_paths(screenshots)

    _load_dotenv_file(dotenv_path=dotenv_path)

    normalized_model = _resolve_model_name(model=model)
    if not normalized_model:
        raise ValueError("Model name must not be empty.")

    openai_client = client or _build_openai_client(api_key=api_key)
    request_input = _build_request_input(normalized_screenshot_paths)

    try:
        response = openai_client.responses.create(
            model=normalized_model,
            input=request_input,
        )
    except Exception as exc:
        raise TaskOpenAISolverError(f"Task solve request failed: {exc}") from exc

    response_text = _extract_output_text(response)
    if not response_text:
        raise TaskOpenAISolverError("Task solve request failed: OpenAI returned an empty text response.")

    return TaskOpenAISolveResult(
        response_text=response_text,
        source_file_count=len(normalized_screenshot_paths),
    )


def _normalize_screenshot_paths(screenshots: Sequence[ScreenshotInput]) -> list[Path]:
    if not screenshots:
        raise ValueError("Screenshots must not be empty.")

    normalized_paths: list[Path] = []
    for screenshot in screenshots:
        screenshot_path = _extract_screenshot_path(screenshot)
        resolved_path = Path(screenshot_path).expanduser().resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise ValueError(f"Screenshot file does not exist: {resolved_path}")
        if resolved_path.suffix.lower() not in TASK_ARTIFACT_SUFFIXES:
            raise ValueError(
                f"Screenshot file must use a supported screenshot extension: {resolved_path.suffix}"
            )
        normalized_paths.append(resolved_path)

    return normalized_paths


def _extract_screenshot_path(screenshot: ScreenshotInput) -> str | Path:
    if isinstance(screenshot, (str, Path)):
        if isinstance(screenshot, str) and not screenshot.strip():
            raise ValueError("Screenshot path must not be empty.")
        return screenshot

    if isinstance(screenshot, Mapping):
        path_value = screenshot.get("path")
        if path_value is None:
            raise ValueError("Screenshot input must provide a path.")
        return path_value

    path_value = getattr(screenshot, "path", None)
    if path_value is None:
        raise ValueError("Screenshot input must provide a path.")
    return path_value


def _build_request_input(screenshot_paths: Sequence[Path]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {
            "type": "input_text",
            "text": _resolve_task_prompt(),
        }
    ]
    for screenshot_path in screenshot_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": _build_image_data_url(screenshot_path),
            }
        )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def _build_image_data_url(screenshot_path: Path) -> str:
    encoded_bytes = base64.b64encode(screenshot_path.read_bytes()).decode("ascii")
    return f"data:{_resolve_content_type(screenshot_path)};base64,{encoded_bytes}"


def _resolve_content_type(screenshot_path: Path) -> str:
    suffix = screenshot_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Screenshot file must use a supported screenshot extension: {suffix}")


def _load_dotenv_file(*, dotenv_path: str | Path | None) -> None:
    load_dotenv_file(dotenv_path=dotenv_path, default_dotenv_path=DEFAULT_DOTENV_PATH)


def _resolve_model_name(*, model: str | None) -> str:
    if model is not None:
        return model.strip()

    configured_model = get_optional_str_env("OPENAI_MODEL")
    if configured_model:
        return configured_model

    return DEFAULT_OPENAI_MODEL


def _resolve_task_prompt() -> str:
    return get_required_str_env(
        OPENAI_TASK_SOLVER_PROMPT_ENV_VAR,
        guidance="Set it in the repository-root `.env` file.",
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
