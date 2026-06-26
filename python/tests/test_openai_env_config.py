from __future__ import annotations

"""Unit tests for shared OpenAI env config helpers."""

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.openai_env_config import get_optional_str_env, get_required_str_env, load_dotenv_file
from modules.openai_env_config import get_optional_bool_env, get_optional_positive_int_env
from modules.openai_env_config import validate_positive_int_env_value


class OpenAIEnvConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.dotenv_path = self.root_dir / ".env"
        self._original_test_key = os.environ.get("TEST_OPENAI_ENV_KEY")
        self._original_test_other = os.environ.get("TEST_OPENAI_ENV_OTHER")
        os.environ.pop("TEST_OPENAI_ENV_KEY", None)
        os.environ.pop("TEST_OPENAI_ENV_OTHER", None)

    def tearDown(self) -> None:
        if self._original_test_key is None:
            os.environ.pop("TEST_OPENAI_ENV_KEY", None)
        else:
            os.environ["TEST_OPENAI_ENV_KEY"] = self._original_test_key

        if self._original_test_other is None:
            os.environ.pop("TEST_OPENAI_ENV_OTHER", None)
        else:
            os.environ["TEST_OPENAI_ENV_OTHER"] = self._original_test_other

        self.temp_dir.cleanup()

    def test_load_dotenv_file_sets_missing_env_values_and_strips_optional_quotes(self) -> None:
        self.dotenv_path.write_text(
            "TEST_OPENAI_ENV_KEY='quoted value'\n"
            "TEST_OPENAI_ENV_OTHER=plain\n",
            encoding="utf-8",
        )

        load_dotenv_file(dotenv_path=self.dotenv_path, default_dotenv_path=self.dotenv_path)

        self.assertEqual("quoted value", os.environ["TEST_OPENAI_ENV_KEY"])
        self.assertEqual("plain", os.environ["TEST_OPENAI_ENV_OTHER"])

    def test_load_dotenv_file_does_not_override_existing_environment(self) -> None:
        os.environ["TEST_OPENAI_ENV_KEY"] = "already-set"
        self.dotenv_path.write_text("TEST_OPENAI_ENV_KEY=from-dotenv\n", encoding="utf-8")

        load_dotenv_file(dotenv_path=self.dotenv_path, default_dotenv_path=self.dotenv_path)

        self.assertEqual("already-set", os.environ["TEST_OPENAI_ENV_KEY"])

    def test_load_dotenv_file_uses_loaded_paths_cache(self) -> None:
        self.dotenv_path.write_text("TEST_OPENAI_ENV_KEY=first\n", encoding="utf-8")
        loaded_paths: set[Path] = set()

        load_dotenv_file(
            dotenv_path=self.dotenv_path,
            default_dotenv_path=self.dotenv_path,
            loaded_paths=loaded_paths,
            lock=threading.Lock(),
        )
        os.environ.pop("TEST_OPENAI_ENV_KEY", None)

        load_dotenv_file(
            dotenv_path=self.dotenv_path,
            default_dotenv_path=self.dotenv_path,
            loaded_paths=loaded_paths,
            lock=threading.Lock(),
        )

        self.assertNotIn("TEST_OPENAI_ENV_KEY", os.environ)

    def test_get_optional_str_env_returns_none_for_missing_or_blank_values(self) -> None:
        self.assertIsNone(get_optional_str_env("TEST_OPENAI_ENV_KEY"))
        os.environ["TEST_OPENAI_ENV_KEY"] = "   "
        self.assertIsNone(get_optional_str_env("TEST_OPENAI_ENV_KEY"))

    def test_get_required_str_env_raises_clear_error_for_missing_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "TEST_OPENAI_ENV_KEY was not provided"):
            get_required_str_env("TEST_OPENAI_ENV_KEY", guidance="Set it in `.env`.")

    def test_get_optional_bool_env_parses_supported_values(self) -> None:
        os.environ["TEST_OPENAI_ENV_KEY"] = "true"
        self.assertIs(get_optional_bool_env("TEST_OPENAI_ENV_KEY"), True)
        os.environ["TEST_OPENAI_ENV_KEY"] = "off"
        self.assertIs(get_optional_bool_env("TEST_OPENAI_ENV_KEY"), False)
        os.environ["TEST_OPENAI_ENV_KEY"] = "   "
        self.assertIsNone(get_optional_bool_env("TEST_OPENAI_ENV_KEY"))

    def test_get_optional_bool_env_rejects_invalid_values(self) -> None:
        os.environ["TEST_OPENAI_ENV_KEY"] = "maybe"
        with self.assertRaisesRegex(ValueError, "Supported values: true, false"):
            get_optional_bool_env("TEST_OPENAI_ENV_KEY")

    def test_get_optional_positive_int_env_returns_none_for_missing_or_blank_values(self) -> None:
        self.assertIsNone(get_optional_positive_int_env("TEST_OPENAI_ENV_KEY"))
        os.environ["TEST_OPENAI_ENV_KEY"] = "   "
        self.assertIsNone(get_optional_positive_int_env("TEST_OPENAI_ENV_KEY"))

    def test_get_optional_positive_int_env_parses_and_validates_value(self) -> None:
        os.environ["TEST_OPENAI_ENV_KEY"] = "42"
        self.assertEqual(42, get_optional_positive_int_env("TEST_OPENAI_ENV_KEY"))

    def test_validate_positive_int_env_value_rejects_invalid_values(self) -> None:
        for invalid_value in (0, -1, "abc", False):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_positive_int_env_value("TEST_OPENAI_ENV_KEY", invalid_value)


if __name__ == "__main__":
    unittest.main()
