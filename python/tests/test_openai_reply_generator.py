from __future__ import annotations

"""Contract tests for the OpenAI reply generation module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.openai_reply_generator import (
    OPENAI_REPLY_INSTRUCTIONS_ENV_VAR,
    generate_chatgpt_reply,
)


class OpenAIReplyGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.dotenv_path = self.root_dir / ".env"
        self._original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._original_openai_model = os.environ.get("OPENAI_MODEL")
        self._original_openai_reply_instructions = os.environ.get(OPENAI_REPLY_INSTRUCTIONS_ENV_VAR)
        self.default_reply_instructions = (
            "Reply naturally and keep it concise. "
            "If the input text is written in Russian, reply in Russian. Otherwise reply in English."
        )
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_MODEL", None)
        os.environ[OPENAI_REPLY_INSTRUCTIONS_ENV_VAR] = self.default_reply_instructions

    def tearDown(self) -> None:
        if self._original_openai_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._original_openai_api_key

        if self._original_openai_model is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = self._original_openai_model

        if self._original_openai_reply_instructions is None:
            os.environ.pop(OPENAI_REPLY_INSTRUCTIONS_ENV_VAR, None)
        else:
            os.environ[OPENAI_REPLY_INSTRUCTIONS_ENV_VAR] = self._original_openai_reply_instructions

        self.temp_dir.cleanup()

    def test_generate_chatgpt_reply_returns_plain_text_response(self) -> None:
        client = FakeOpenAIClient(response_text="Hello back")

        reply = generate_chatgpt_reply(
            "Hello there",
            client=client,
            model="gpt-5-mini",
            instructions="Answer briefly.",
        )

        self.assertEqual("Hello back", reply)
        self.assertEqual(
            {
                "model": "gpt-5-mini",
                "input": "Hello there",
                "instructions": "Answer briefly.",
            },
            client.recorded_request,
        )

    def test_generate_chatgpt_reply_uses_human_like_default_instructions(self) -> None:
        client = FakeOpenAIClient(response_text="Sure, I can help with that.")

        reply = generate_chatgpt_reply(
            "Can you help me with this?",
            client=client,
            model="gpt-5-mini",
        )

        self.assertEqual("Sure, I can help with that.", reply)
        self.assertEqual(self.default_reply_instructions, client.recorded_request["instructions"])
        self.assertIn("reply in Russian", self.default_reply_instructions)
        self.assertIn("reply in English", self.default_reply_instructions)

    def test_generate_chatgpt_reply_reads_instructions_from_dotenv_file(self) -> None:
        dotenv_instructions = "Answer in one short sentence."
        os.environ.pop(OPENAI_REPLY_INSTRUCTIONS_ENV_VAR, None)
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_MODEL=gpt-5-nano\n"
            f"{OPENAI_REPLY_INSTRUCTIONS_ENV_VAR}={dotenv_instructions}\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="Configured reply")

        reply = generate_chatgpt_reply(
            "Hello",
            client=client,
            dotenv_path=str(self.dotenv_path),
        )

        self.assertEqual("Configured reply", reply)
        self.assertEqual(dotenv_instructions, client.recorded_request["instructions"])

    def test_generate_chatgpt_reply_rejects_empty_input_text(self) -> None:
        with self.assertRaises(ValueError):
            generate_chatgpt_reply("   ", client=FakeOpenAIClient(response_text="unused"))

    def test_generate_chatgpt_reply_reads_model_from_dotenv_file(self) -> None:
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_MODEL=gpt-5-nano\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="Configured reply")

        reply = generate_chatgpt_reply(
            "Hello",
            client=client,
            dotenv_path=str(self.dotenv_path),
        )

        self.assertEqual("Configured reply", reply)
        self.assertEqual("gpt-5-nano", client.recorded_request["model"])

    def test_generate_chatgpt_reply_rejects_missing_api_key_when_building_default_client(self) -> None:
        with self.assertRaises(ValueError) as error:
            generate_chatgpt_reply("Hello", api_key="   ", dotenv_path=str(self.dotenv_path))

        self.assertIn("OpenAI API key was not provided", str(error.exception))

    def test_generate_chatgpt_reply_reports_client_failure(self) -> None:
        client = FakeOpenAIClient(raised_error=RuntimeError("upstream unavailable"))

        with self.assertRaises(RuntimeError) as error:
            generate_chatgpt_reply("Hello", client=client)

        self.assertIn("ChatGPT reply generation failed", str(error.exception))
        self.assertIn("upstream unavailable", str(error.exception))

    def test_generate_chatgpt_reply_rejects_empty_text_response(self) -> None:
        client = FakeOpenAIClient(response_text="   ")

        with self.assertRaises(RuntimeError) as error:
            generate_chatgpt_reply("Hello", client=client)

        self.assertIn("empty text response", str(error.exception))


class FakeOpenAIClient:
    def __init__(self, *, response_text: str | None = None, raised_error: Exception | None = None) -> None:
        self.recorded_request: dict[str, str | None] | None = None
        self._response_text = response_text
        self._raised_error = raised_error
        self.responses = FakeResponsesAPI(self)


class FakeResponsesAPI:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.owner = owner

    def create(self, *, model: str, input: str, instructions: str | None = None) -> object:
        self.owner.recorded_request = {
            "model": model,
            "input": input,
            "instructions": instructions,
        }
        if self.owner._raised_error is not None:
            raise self.owner._raised_error
        return FakeResponse(self.owner._response_text)


class FakeResponse:
    def __init__(self, output_text: str | None) -> None:
        self.output_text = output_text


if __name__ == "__main__":
    unittest.main()
