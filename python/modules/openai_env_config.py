from __future__ import annotations

"""Shared helpers for OpenAI-related dotenv and env-string config."""

import os
from collections.abc import Callable
from pathlib import Path
import threading


def load_dotenv_file(
    *,
    dotenv_path: str | Path | None,
    default_dotenv_path: Path,
    loaded_paths: set[Path] | None = None,
    lock: threading.Lock | None = None,
    dotenv_text_reader: Callable[[Path], str] | None = None,
) -> None:
    resolved_dotenv_path = Path(dotenv_path or default_dotenv_path).expanduser().resolve()
    if not resolved_dotenv_path.is_file():
        return

    def _load() -> None:
        if loaded_paths is not None and resolved_dotenv_path in loaded_paths:
            return

        reader = dotenv_text_reader or _read_dotenv_text
        for raw_line in reader(resolved_dotenv_path).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            normalized_key = key.strip()
            if not normalized_key:
                continue

            os.environ.setdefault(normalized_key, strip_optional_quotes(value.strip()))

        if loaded_paths is not None:
            loaded_paths.add(resolved_dotenv_path)

    if lock is None:
        _load()
        return

    with lock:
        _load()


def get_optional_str_env(env_var_name: str) -> str | None:
    raw_value = os.getenv(env_var_name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    return normalized_value or None


def get_required_str_env(env_var_name: str, *, guidance: str) -> str:
    configured_value = get_optional_str_env(env_var_name)
    if configured_value is not None:
        return configured_value

    raise ValueError(f"{env_var_name} was not provided. {guidance}")


def get_optional_bool_env(env_var_name: str) -> bool | None:
    raw_value = os.getenv(env_var_name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip().lower()
    if not normalized_value:
        return None

    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Unsupported {env_var_name} value {raw_value!r}. Supported values: true, false."
    )


def get_optional_positive_int_env(env_var_name: str) -> int | None:
    raw_value = os.getenv(env_var_name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None

    return validate_positive_int_env_value(env_var_name, normalized_value)


def validate_positive_int_env_value(env_var_name: str, value: int | str) -> int:
    if isinstance(value, bool):
        normalized_value: int | None = None
    elif isinstance(value, int):
        normalized_value = value
    else:
        try:
            normalized_value = int(str(value).strip())
        except (TypeError, ValueError):
            normalized_value = None

    if normalized_value is None or normalized_value <= 0:
        raise ValueError(
            f"{env_var_name} must be a positive integer when set. Got {value!r}."
        )

    return normalized_value


def strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_dotenv_text(dotenv_path: Path) -> str:
    return dotenv_path.read_text(encoding="utf-8")
