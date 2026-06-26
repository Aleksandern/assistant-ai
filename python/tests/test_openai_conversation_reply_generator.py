from __future__ import annotations

"""Contract tests for continuing an existing OpenAI conversation."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules.openai_conversation_reply_generator as openai_conversation_reply_generator_module
from modules.openai_conversation_reply_generator import (
    OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR,
    OpenAIConversationReplyGenerationError,
    generate_reply_in_openai_conversation,
)
from modules.sqlite_conversation_store import create_conversation


class OpenAIConversationReplyGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.database_path = self.root_dir / "database" / "test.sqlite3"
        self.dotenv_path = self.root_dir / ".env"
        self.missing_dotenv_path = self.root_dir / ".missing.env"
        self._original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._original_openai_model = os.environ.get("OPENAI_MODEL")
        self._original_openai_service_tier = os.environ.get("OPENAI_SERVICE_TIER")
        self._original_openai_prompt_cache_enabled = os.environ.get("OPENAI_PROMPT_CACHE_ENABLED")
        self._original_openai_prompt_cache_key_prefix = os.environ.get("OPENAI_PROMPT_CACHE_KEY_PREFIX")
        self._original_openai_prompt_cache_retention = os.environ.get("OPENAI_PROMPT_CACHE_RETENTION")
        self._original_openai_max_output_tokens = os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
        self._original_openai_first_useful_text_min_chars = os.environ.get("OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS")
        self._original_openai_first_useful_text_min_words = os.environ.get("OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS")
        self._original_openai_conversation_reply_instructions = os.environ.get(
            OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR
        )
        self.default_conversation_reply_instructions = (
            "If the latest conversation message is in Russian, answer in Russian. Otherwise answer in English."
        )
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_MODEL", None)
        os.environ.pop("OPENAI_SERVICE_TIER", None)
        os.environ.pop("OPENAI_PROMPT_CACHE_ENABLED", None)
        os.environ.pop("OPENAI_PROMPT_CACHE_KEY_PREFIX", None)
        os.environ.pop("OPENAI_PROMPT_CACHE_RETENTION", None)
        os.environ.pop("OPENAI_MAX_OUTPUT_TOKENS", None)
        os.environ.pop("OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS", None)
        os.environ.pop("OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS", None)
        os.environ[OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR] = self.default_conversation_reply_instructions
        openai_conversation_reply_generator_module._reset_runtime_caches_for_tests()

    def tearDown(self) -> None:
        if self._original_openai_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._original_openai_api_key

        if self._original_openai_model is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = self._original_openai_model

        if self._original_openai_service_tier is None:
            os.environ.pop("OPENAI_SERVICE_TIER", None)
        else:
            os.environ["OPENAI_SERVICE_TIER"] = self._original_openai_service_tier

        if self._original_openai_prompt_cache_enabled is None:
            os.environ.pop("OPENAI_PROMPT_CACHE_ENABLED", None)
        else:
            os.environ["OPENAI_PROMPT_CACHE_ENABLED"] = self._original_openai_prompt_cache_enabled

        if self._original_openai_prompt_cache_key_prefix is None:
            os.environ.pop("OPENAI_PROMPT_CACHE_KEY_PREFIX", None)
        else:
            os.environ["OPENAI_PROMPT_CACHE_KEY_PREFIX"] = self._original_openai_prompt_cache_key_prefix

        if self._original_openai_prompt_cache_retention is None:
            os.environ.pop("OPENAI_PROMPT_CACHE_RETENTION", None)
        else:
            os.environ["OPENAI_PROMPT_CACHE_RETENTION"] = self._original_openai_prompt_cache_retention

        if self._original_openai_max_output_tokens is None:
            os.environ.pop("OPENAI_MAX_OUTPUT_TOKENS", None)
        else:
            os.environ["OPENAI_MAX_OUTPUT_TOKENS"] = self._original_openai_max_output_tokens

        if self._original_openai_first_useful_text_min_chars is None:
            os.environ.pop("OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS", None)
        else:
            os.environ["OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS"] = self._original_openai_first_useful_text_min_chars

        if self._original_openai_first_useful_text_min_words is None:
            os.environ.pop("OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS", None)
        else:
            os.environ["OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS"] = self._original_openai_first_useful_text_min_words

        if self._original_openai_conversation_reply_instructions is None:
            os.environ.pop(OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR, None)
        else:
            os.environ[OPENAI_CONVERSATION_REPLY_INSTRUCTIONS_ENV_VAR] = (
                self._original_openai_conversation_reply_instructions
            )

        openai_conversation_reply_generator_module._reset_runtime_caches_for_tests()
        self.temp_dir.cleanup()

    def test_generate_reply_in_openai_conversation_sends_new_turn_to_existing_openai_conversation(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            is_active=True,
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="I have worked mostly on backend systems.", response_id="resp_123")

        reply = generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Tell me about your backend experience.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual(conversation.conversation_id, reply.conversation_id)
        self.assertEqual("conv_123", reply.openai_conversation_id)
        self.assertEqual("I have worked mostly on backend systems.", reply.reply_text)
        self.assertEqual("resp_123", reply.response_id)
        self.assertEqual(
            {
                "model": "gpt-5-mini",
                "input": "Tell me about your backend experience.",
                "stream": False,
                "conversation": "conv_123",
                "instructions": self.default_conversation_reply_instructions,
            },
            client.recorded_request,
        )

    def test_generate_reply_in_openai_conversation_reads_model_from_dotenv_file(self) -> None:
        conversation = create_conversation(
            "System design conversation",
            openai_conversation_id="conv_456",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_MODEL=gpt-5-nano\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="I would start with the main requirements.")

        reply = generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How would you design a URL shortener?",
            client=client,
            database_path=self.database_path,
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual("I would start with the main requirements.", reply.reply_text)
        self.assertEqual("gpt-5-nano", client.recorded_request["model"])

    def test_generate_reply_in_openai_conversation_sends_max_output_tokens_from_dotenv(self) -> None:
        conversation = create_conversation(
            "Latency conversation",
            openai_conversation_id="conv_max_tokens",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text("OPENAI_MAX_OUTPUT_TOKENS=120\n", encoding="utf-8")
        client = FakeOpenAIClient(response_text="Short answer.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Give me a concise answer.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual(120, client.recorded_request["max_output_tokens"])

    def test_generate_reply_in_openai_conversation_omits_max_output_tokens_when_config_is_empty(self) -> None:
        conversation = create_conversation(
            "Latency conversation",
            openai_conversation_id="conv_max_tokens",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text("OPENAI_MAX_OUTPUT_TOKENS=   \n", encoding="utf-8")
        client = FakeOpenAIClient(response_text="Normal answer.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Give me a normal answer.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
        )

        self.assertNotIn("max_output_tokens", client.recorded_request)

    def test_generate_reply_in_openai_conversation_accepts_optional_turn_instructions(self) -> None:
        conversation = create_conversation(
            "Behavioral conversation",
            openai_conversation_id="conv_789",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="I usually keep the answer concise.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How do you keep answers concise?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            instructions="Answer in one short paragraph.",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual("Answer in one short paragraph.", client.recorded_request["instructions"])

    def test_generate_reply_in_openai_conversation_uses_default_language_instructions_when_none_are_provided(self) -> None:
        conversation = create_conversation(
            "Language policy conversation",
            openai_conversation_id="conv_lang",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="Sure.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Привет, расскажи о себе.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual(
            self.default_conversation_reply_instructions,
            client.recorded_request["instructions"],
        )

    def test_generate_reply_in_openai_conversation_sends_minimal_reasoning_effort_for_streaming_gpt5_models(self) -> None:
        conversation = create_conversation(
            "Behavioral conversation",
            openai_conversation_id="conv_reasoning",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="I keep the answer concise.",
            response_id="resp_reasoning",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="I keep "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="the answer concise."),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="I keep the answer concise.",
                        response_id="resp_reasoning",
                    ),
                ),
            ],
        )

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How do you keep answers concise?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.missing_dotenv_path,
            stream=True,
        )

        self.assertEqual({"effort": "minimal"}, client.recorded_request["reasoning"])

    def test_generate_reply_in_openai_conversation_omits_reasoning_effort_for_non_streaming_gpt5_models(self) -> None:
        conversation = create_conversation(
            "Behavioral conversation",
            openai_conversation_id="conv_reasoning_non_streaming",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="I keep the answer concise.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How do you keep answers concise?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertNotIn("reasoning", client.recorded_request)

    def test_generate_reply_in_openai_conversation_omits_reasoning_effort_for_non_gpt5_models(self) -> None:
        conversation = create_conversation(
            "Behavioral conversation",
            openai_conversation_id="conv_reasoning_non_gpt5",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="I keep the answer concise.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How do you keep answers concise?",
            client=client,
            database_path=self.database_path,
            model="gpt-4o-mini",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertNotIn("reasoning", client.recorded_request)

    def test_generate_reply_in_openai_conversation_omits_service_tier_when_config_is_missing(self) -> None:
        conversation = create_conversation(
            "System design conversation",
            openai_conversation_id="conv_456",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="I would start with the main requirements.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How would you design a URL shortener?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertNotIn("service_tier", client.recorded_request)

    def test_generate_reply_in_openai_conversation_reads_priority_service_tier_from_dotenv_file(self) -> None:
        conversation = create_conversation(
            "System design conversation",
            openai_conversation_id="conv_456",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_SERVICE_TIER=priority\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="I would start with the main requirements.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "How would you design a URL shortener?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual("priority", client.recorded_request["service_tier"])

    def test_generate_reply_in_openai_conversation_sends_prompt_cache_key_when_enabled(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=true\n"
            "OPENAI_PROMPT_CACHE_KEY_PREFIX=assistantai\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="Cached reply.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Tell me about a recent project.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual(
            "assistantai:conversation:1",
            client.recorded_request["prompt_cache_key"],
        )
        self.assertNotIn("prompt_cache_retention", client.recorded_request)

    def test_generate_reply_in_openai_conversation_omits_prompt_cache_key_when_disabled(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="No cache reply.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Tell me about a recent project.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertNotIn("prompt_cache_key", client.recorded_request)
        self.assertNotIn("prompt_cache_retention", client.recorded_request)

    def test_generate_reply_in_openai_conversation_uses_stable_prompt_cache_key_for_same_conversation(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=true\n"
            "OPENAI_PROMPT_CACHE_KEY_PREFIX=assistantai\n",
            encoding="utf-8",
        )
        first_client = FakeOpenAIClient(response_text="First reply.")
        second_client = FakeOpenAIClient(response_text="Second reply.")

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "First question?",
            client=first_client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
        )
        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Second question?",
            client=second_client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual(first_client.recorded_request["prompt_cache_key"], second_client.recorded_request["prompt_cache_key"])

    def test_generate_reply_in_openai_conversation_streaming_request_sends_prompt_cache_key_and_retention(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=true\n"
            "OPENAI_PROMPT_CACHE_KEY_PREFIX=assistantai\n"
            "OPENAI_PROMPT_CACHE_RETENTION=24h\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(
            response_text="Streamed cached reply.",
            response_id="resp_stream_cache",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="Streamed "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="cached reply."),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="Streamed cached reply.",
                        response_id="resp_stream_cache",
                    ),
                ),
            ],
        )

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Tell me about your architecture decisions.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
            stream=True,
        )

        self.assertTrue(client.recorded_request["stream"])
        self.assertEqual("assistantai:conversation:1", client.recorded_request["prompt_cache_key"])
        self.assertEqual("24h", client.recorded_request["prompt_cache_retention"])

    def test_generate_reply_in_openai_conversation_streaming_request_sends_max_output_tokens(self) -> None:
        conversation = create_conversation(
            "Latency conversation",
            openai_conversation_id="conv_stream_tokens",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text("OPENAI_MAX_OUTPUT_TOKENS=120\n", encoding="utf-8")
        client = FakeOpenAIClient(
            response_text="Short streamed answer.",
            response_id="resp_stream_tokens",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="Short "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="streamed answer."),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="Short streamed answer.",
                        response_id="resp_stream_tokens",
                    ),
                ),
            ],
        )

        generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "Stream something concise.",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            dotenv_path=self.dotenv_path,
            stream=True,
        )

        self.assertEqual(120, client.recorded_request["max_output_tokens"])

    def test_generate_reply_in_openai_conversation_rejects_invalid_prompt_cache_configuration(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=maybe\n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.dotenv_path,
            )

        self.assertIn("OPENAI_PROMPT_CACHE_ENABLED", str(error.exception))
        self.assertIn("true", str(error.exception))
        self.assertIn("false", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_empty_prompt_cache_key_prefix(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=true\n"
            "OPENAI_PROMPT_CACHE_KEY_PREFIX=   \n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.dotenv_path,
            )

        self.assertIn("OPENAI_PROMPT_CACHE_KEY_PREFIX", str(error.exception))
        self.assertIn("must not be empty", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_invalid_prompt_cache_retention(self) -> None:
        conversation = create_conversation(
            "Prompt caching conversation",
            openai_conversation_id="conv_cache",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=true\n"
            "OPENAI_PROMPT_CACHE_RETENTION=7d\n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.dotenv_path,
            )

        self.assertIn("OPENAI_PROMPT_CACHE_RETENTION", str(error.exception))
        self.assertIn("24h", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_invalid_max_output_tokens_configuration(self) -> None:
        conversation = create_conversation(
            "Latency conversation",
            openai_conversation_id="conv_invalid_max_tokens",
            database_path=self.database_path,
        )

        for invalid_value in ("0", "-5", "abc"):
            with self.subTest(invalid_value=invalid_value):
                self.dotenv_path.write_text(
                    f"OPENAI_MAX_OUTPUT_TOKENS={invalid_value}\n",
                    encoding="utf-8",
                )
                openai_conversation_reply_generator_module._reset_runtime_caches_for_tests()

                with self.assertRaises(ValueError) as error:
                    generate_reply_in_openai_conversation(
                        conversation.conversation_id,
                        "Tell me about Python.",
                        client=FakeOpenAIClient(response_text="unused"),
                        database_path=self.database_path,
                        model="gpt-5-mini",
                        dotenv_path=self.dotenv_path,
                    )

                self.assertIn("OPENAI_MAX_OUTPUT_TOKENS", str(error.exception))
                self.assertIn("positive integer", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_invalid_service_tier_configuration(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_SERVICE_TIER=standard\n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.dotenv_path,
            )

        self.assertIn("OPENAI_SERVICE_TIER", str(error.exception))
        self.assertIn("priority", str(error.exception))
        self.assertIn("standard", str(error.exception))

    def test_generate_reply_in_openai_conversation_reuses_cached_openai_client_for_same_api_key(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        created_clients: list[FakeOpenAIClient] = []

        def build_client(*, api_key: str | None, OpenAI: object | None = None) -> FakeOpenAIClient:
            client = FakeOpenAIClient(response_text="cached response")
            created_clients.append(client)
            self.assertEqual("test-key", api_key)
            return client

        with patch(
            "modules.openai_conversation_reply_generator._instantiate_openai_client",
            side_effect=build_client,
        ) as build_client_mock:
            first_reply = generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "First question?",
                api_key="test-key",
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )
            second_reply = generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Second question?",
                api_key="test-key",
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertEqual("cached response", first_reply.reply_text)
        self.assertEqual("cached response", second_reply.reply_text)
        self.assertEqual(1, build_client_mock.call_count)
        self.assertEqual(1, len(created_clients))
        self.assertEqual("Second question?", created_clients[0].recorded_request["input"])

    def test_generate_reply_in_openai_conversation_loads_dotenv_only_once_per_path(self) -> None:
        conversation = create_conversation(
            "System design conversation",
            openai_conversation_id="conv_456",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_MODEL=gpt-5-nano\n",
            encoding="utf-8",
        )

        with patch(
            "modules.openai_conversation_reply_generator._read_dotenv_text",
            wraps=openai_conversation_reply_generator_module._read_dotenv_text,
        ) as read_text_mock:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "How would you scale reads?",
                client=FakeOpenAIClient(response_text="Add caching."),
                database_path=self.database_path,
                dotenv_path=self.dotenv_path,
            )
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "How would you scale writes?",
                client=FakeOpenAIClient(response_text="Partition carefully."),
                database_path=self.database_path,
                dotenv_path=self.dotenv_path,
            )

        self.assertEqual(1, read_text_mock.call_count)

    def test_generate_reply_in_openai_conversation_logs_debug_metrics_only_when_enabled(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_PROMPT_CACHE_ENABLED=true\n"
            "OPENAI_PROMPT_CACHE_KEY_PREFIX=assistantai\n"
            "OPENAI_PROMPT_CACHE_RETENTION=24h\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(
            response_text="A short answer.",
            response_id="resp_debug",
            usage=FakeUsage(cached_tokens=1024),
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.125, 20.0, 20.456],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    instructions="Keep it short.",
                    dotenv_path=self.dotenv_path,
                    debug=True,
                )

        self.assertEqual(1, len(captured_logs.output))
        debug_line = captured_logs.output[0]
        self.assertIn("model=gpt-5-mini", debug_line)
        self.assertIn("input_chars=21", debug_line)
        self.assertIn("instructions_present=True", debug_line)
        self.assertIn("instructions_chars=14", debug_line)
        self.assertIn("conversation_id=1", debug_line)
        self.assertIn("openai_conversation_id=conv_123", debug_line)
        self.assertIn("prompt_cache_key=assistantai:conversation:1", debug_line)
        self.assertIn("prompt_cache_retention=24h", debug_line)
        self.assertIn("service_tier=None", debug_line)
        self.assertIn("request_prep_ms=125", debug_line)
        self.assertIn("responses_create_ms=456", debug_line)
        self.assertIn("response_id=resp_debug", debug_line)
        self.assertIn("cached_tokens=1024", debug_line)
        self.assertIn("reply_chars=15", debug_line)
        self.assertIn("ttft_ms=None", debug_line)
        self.assertIn("ttfut_ms=None", debug_line)

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[30.0, 30.050, 31.0, 31.050],
        ):
            with self.assertNoLogs("modules.openai_conversation_reply_generator", level="INFO"):
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python again.",
                    client=FakeOpenAIClient(response_text="Still short."),
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.missing_dotenv_path,
                    debug=False,
                )

    def test_generate_reply_in_openai_conversation_logs_streaming_ttft_ms_in_debug_mode(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_debug",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text("OPENAI_MAX_OUTPUT_TOKENS=120\n", encoding="utf-8")
        client = FakeOpenAIClient(
            response_text="Streamed short answer.",
            response_id="resp_stream_debug",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="Streamed "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="short answer."),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="Streamed short answer.",
                        response_id="resp_stream_debug",
                        usage=FakeUsage(cached_tokens=64),
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.020, 20.0, 20.020, 20.150],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertGreaterEqual(len(captured_logs.output), 1)
        debug_line = captured_logs.output[-1]
        self.assertIn("responses_create_ms=150", debug_line)
        self.assertIn("ttft_ms=20", debug_line)
        self.assertIn("max_output_tokens=120", debug_line)
        self.assertIn("response_id=resp_stream_debug", debug_line)
        self.assertIn("cached_tokens=64", debug_line)

    def test_generate_reply_in_openai_conversation_returns_streaming_latency_metadata(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_metrics",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="Streamed short answer.",
            response_id="resp_stream_metrics",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="Streamed "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="short answer."),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="Streamed short answer.",
                        response_id="resp_stream_metrics",
                        usage=FakeUsage(cached_tokens=64),
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.020, 20.0, 20.020, 20.150],
        ):
            reply = generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
                stream=True,
            )

        self.assertIsNotNone(reply.timings)
        assert reply.timings is not None
        self.assertEqual(20, reply.timings.ttft_ms)
        self.assertIsNone(reply.timings.ttfut_ms)
        self.assertEqual(150, reply.timings.full_ms)
        self.assertEqual(64, reply.timings.cached_tokens)

    def test_generate_reply_in_openai_conversation_returns_non_streaming_latency_metadata_without_ttft(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_non_stream_metrics",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="A short answer.",
            response_id="resp_non_stream_metrics",
            usage=FakeUsage(cached_tokens=1024),
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.125, 20.0, 20.456],
        ):
            reply = generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIsNotNone(reply.timings)
        assert reply.timings is not None
        self.assertIsNone(reply.timings.ttft_ms)
        self.assertIsNone(reply.timings.ttfut_ms)
        self.assertEqual(456, reply.timings.full_ms)
        self.assertEqual(1024, reply.timings.cached_tokens)

    def test_generate_reply_in_openai_conversation_logs_streaming_ttft_none_when_no_text_delta_arrives(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_debug",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="Completed without deltas.",
            response_id="resp_stream_no_delta",
            stream_events=[
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="Completed without deltas.",
                        response_id="resp_stream_no_delta",
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.025, 20.0, 20.110],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.missing_dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertGreaterEqual(len(captured_logs.output), 1)
        self.assertIn("ttft_ms=None", captured_logs.output[-1])
        self.assertIn("ttfut_ms=None", captured_logs.output[-1])

    def test_generate_reply_in_openai_conversation_logs_streaming_ttfut_ms_for_char_threshold(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_ttfut_chars",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS=10\n"
            "OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS=99\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(
            response_text="Short useful answer.",
            response_id="resp_stream_ttfut_chars",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="Hey "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="there friend"),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="Hey there friend",
                        response_id="resp_stream_ttfut_chars",
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.010, 20.0, 20.020, 20.080, 20.150],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertGreaterEqual(len(captured_logs.output), 1)
        debug_line = captured_logs.output[-1]
        self.assertIn("ttft_ms=20", debug_line)
        self.assertIn("ttfut_ms=80", debug_line)

    def test_generate_reply_in_openai_conversation_logs_streaming_ttfut_ms_for_word_threshold(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_ttfut_words",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS=999\n"
            "OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS=4\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(
            response_text="one two three four",
            response_id="resp_stream_ttfut_words",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="one "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="two "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="three "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="four"),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="one two three four",
                        response_id="resp_stream_ttfut_words",
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.010, 20.0, 20.015, 20.040, 20.070, 20.110, 20.160],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertGreaterEqual(len(captured_logs.output), 1)
        debug_line = captured_logs.output[-1]
        self.assertIn("ttft_ms=15", debug_line)
        self.assertIn("ttfut_ms=40", debug_line)

    def test_generate_reply_in_openai_conversation_logs_streaming_ttfut_none_when_useful_threshold_is_not_reached(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_ttfut_none",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS=100\n"
            "OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS=10\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(
            response_text="too short",
            response_id="resp_stream_ttfut_none",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="too "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="short"),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="too short",
                        response_id="resp_stream_ttfut_none",
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.010, 20.0, 20.020, 20.060, 20.100],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertGreaterEqual(len(captured_logs.output), 1)
        debug_line = captured_logs.output[-1]
        self.assertIn("ttft_ms=20", debug_line)
        self.assertIn("ttfut_ms=None", debug_line)

    def test_generate_reply_in_openai_conversation_uses_default_useful_text_thresholds_when_env_is_missing(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_ttfut_defaults",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="one two three four",
            response_id="resp_stream_ttfut_defaults",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="one "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="two "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="three "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="four"),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="one two three four",
                        response_id="resp_stream_ttfut_defaults",
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.010, 20.0, 20.015, 20.040, 20.070, 20.110, 20.160],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.missing_dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertIn("ttfut_ms=40", captured_logs.output[-1])

    def test_generate_reply_in_openai_conversation_uses_default_useful_text_thresholds_when_env_values_are_empty(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_ttfut_empty_defaults",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS=   \n"
            "OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS= \n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(
            response_text="one two three four",
            response_id="resp_stream_ttfut_empty_defaults",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="one "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="two "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="three "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="four"),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="one two three four",
                        response_id="resp_stream_ttfut_empty_defaults",
                    ),
                ),
            ],
        )

        with patch(
            "modules.openai_conversation_reply_generator.time.perf_counter",
            side_effect=[10.0, 10.010, 20.0, 20.015, 20.040, 20.070, 20.110, 20.160],
        ):
            with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.dotenv_path,
                    debug=True,
                    stream=True,
                )

        self.assertIn("ttfut_ms=40", captured_logs.output[-1])

    def test_generate_reply_in_openai_conversation_rejects_invalid_useful_text_threshold_configuration(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_ttfut_invalid",
            database_path=self.database_path,
        )

        for env_var_name in (
            "OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS",
            "OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS",
        ):
            for invalid_value in ("0", "-5", "abc"):
                with self.subTest(env_var_name=env_var_name, invalid_value=invalid_value):
                    self.dotenv_path.write_text(
                        f"{env_var_name}={invalid_value}\n",
                        encoding="utf-8",
                    )
                    os.environ.pop("OPENAI_FIRST_USEFUL_TEXT_MIN_CHARS", None)
                    os.environ.pop("OPENAI_FIRST_USEFUL_TEXT_MIN_WORDS", None)
                    openai_conversation_reply_generator_module._reset_runtime_caches_for_tests()

                    with self.assertRaises(ValueError) as error:
                        generate_reply_in_openai_conversation(
                            conversation.conversation_id,
                            "Tell me about Python.",
                            client=FakeOpenAIClient(response_text="unused"),
                            database_path=self.database_path,
                            model="gpt-5-mini",
                            dotenv_path=self.dotenv_path,
                        )

                    self.assertIn(env_var_name, str(error.exception))
                    self.assertIn("positive integer", str(error.exception))

    def test_generate_reply_in_openai_conversation_fast_mode_can_skip_conversation_and_instructions(self) -> None:
        conversation = create_conversation("Python conversation", database_path=self.database_path)
        client = FakeOpenAIClient(response_text="Fast answer.")

        reply = generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "What is your strongest skill?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            fast_mode=True,
            fast_model="gpt-5-nano",
            use_conversation=False,
            disable_instructions=True,
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual("Fast answer.", reply.reply_text)
        self.assertEqual("", reply.openai_conversation_id)
        self.assertEqual(
            {
                "model": "gpt-5-nano",
                "input": "What is your strongest skill?",
                "stream": False,
            },
            client.recorded_request,
        )

    def test_generate_reply_in_openai_conversation_streams_text_for_display_and_returns_final_record(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        streamed_chunks: list[str] = []
        client = FakeOpenAIClient(
            response_text="I enjoy solving backend problems.",
            response_id="resp_stream",
            stream_events=[
                FakeStreamEvent(event_type="response.output_text.delta", delta="I enjoy "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="solving "),
                FakeStreamEvent(event_type="response.output_text.delta", delta="backend problems."),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponse(
                        output_text="I enjoy solving backend problems.",
                        response_id="resp_stream",
                    ),
                ),
            ],
        )

        reply = generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "What do you like building?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            stream=True,
            on_text_delta=streamed_chunks.append,
            service_tier="priority",
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual("I enjoy solving backend problems.", "".join(streamed_chunks))
        self.assertEqual("I enjoy solving backend problems.", reply.reply_text)
        self.assertEqual("resp_stream", reply.response_id)
        self.assertTrue(client.recorded_request["stream"])
        self.assertEqual("priority", client.recorded_request["service_tier"])

    def test_generate_reply_in_openai_conversation_streams_output_text_done_events_without_deltas(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_done_only",
            database_path=self.database_path,
        )
        streamed_chunks: list[str] = []
        client = FakeOpenAIClient(
            response_text="unused",
            response_id="resp_done_only",
            stream_events=[
                FakeStreamEvent(event_type="response.created"),
                FakeStreamEvent(
                    event_type="response.output_item.added",
                    item=FakeOutputMessage(
                        content=[FakeOutputTextContent(text="I enjoy solving backend problems.")],
                    ),
                ),
                FakeStreamEvent(
                    event_type="response.output_text.done",
                    text="I enjoy solving backend problems.",
                ),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponseWithoutOutputText(
                        response_id="resp_done_only",
                        output=[
                            FakeOutputMessage(
                                content=[FakeOutputTextContent(text="I enjoy solving backend problems.")],
                            )
                        ],
                    ),
                ),
            ],
        )

        reply = generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "What do you like building?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            stream=True,
            on_text_delta=streamed_chunks.append,
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual(["I enjoy solving backend problems."], streamed_chunks)
        self.assertEqual("I enjoy solving backend problems.", reply.reply_text)
        self.assertEqual("resp_done_only", reply.response_id)

    def test_generate_reply_in_openai_conversation_streaming_uses_final_response_output_items_without_response_completed(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_final_response",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="unused",
            response_id="resp_final_response",
            stream_events=[
                FakeStreamEvent(event_type="response.created"),
                FakeStreamEvent(event_type="response.in_progress"),
                FakeStreamEvent(
                    event_type="response.output_item.done",
                    item=FakeOutputMessage(
                        content=[FakeOutputTextContent(text="Recovered from final response output.")],
                    ),
                ),
            ],
            stream_final_response=FakeResponseWithoutOutputText(
                response_id="resp_final_response",
                output=[
                    FakeOutputMessage(
                        content=[FakeOutputTextContent(text="Recovered from final response output.")],
                    )
                ],
            ),
        )

        reply = generate_reply_in_openai_conversation(
            conversation.conversation_id,
            "What happened?",
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
            stream=True,
            dotenv_path=self.missing_dotenv_path,
        )

        self.assertEqual("Recovered from final response output.", reply.reply_text)
        self.assertEqual("resp_final_response", reply.response_id)

    def test_generate_reply_in_openai_conversation_logs_unknown_stream_event_types_in_debug_mode(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_stream_event_debug",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="unused",
            response_id="resp_stream_event_debug",
            stream_events=[
                FakeStreamEvent(event_type="response.created"),
                FakeStreamEvent(
                    event_type="response.output_text.done",
                    text="Completed from done event.",
                ),
                FakeStreamEvent(
                    event_type="response.completed",
                    response=FakeResponseWithoutOutputText(
                        response_id="resp_stream_event_debug",
                        output=[
                            FakeOutputMessage(
                                content=[FakeOutputTextContent(text="Completed from done event.")],
                            )
                        ],
                    ),
                ),
            ],
        )

        with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
                debug=True,
                stream=True,
            )

        joined_logs = "\n".join(captured_logs.output)
        self.assertIn("stream_event_type=response.created", joined_logs)
        self.assertIn("unhandled_stream_event_type=response.created", joined_logs)
        self.assertIn("stream_event_type=response.output_text.done", joined_logs)
        self.assertIn("stream_response_completed_has_response=True", joined_logs)

    def test_generate_reply_in_openai_conversation_rejects_empty_input_text(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )

        with self.assertRaises(ValueError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "   ",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIn("Input text must not be empty", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_missing_local_conversation(self) -> None:
        with self.assertRaises(OpenAIConversationReplyGenerationError) as error:
            generate_reply_in_openai_conversation(
                999,
                "Hello",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIn("Local conversation was not found", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_missing_openai_conversation_id(self) -> None:
        conversation = create_conversation("Python conversation", database_path=self.database_path)

        with self.assertRaises(OpenAIConversationReplyGenerationError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Hello",
                client=FakeOpenAIClient(response_text="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIn("does not have an OpenAI conversation id", str(error.exception))

    def test_generate_reply_in_openai_conversation_reports_openai_failure(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(raised_error=RuntimeError("upstream unavailable"))

        with self.assertRaises(OpenAIConversationReplyGenerationError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIn("Failed to continue OpenAI conversation", str(error.exception))
        self.assertIn("upstream unavailable", str(error.exception))

    def test_generate_reply_in_openai_conversation_rejects_empty_openai_text_response(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(response_text="   ")

        with self.assertRaises(OpenAIConversationReplyGenerationError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIn("empty text response", str(error.exception))

    def test_generate_reply_in_openai_conversation_reports_incomplete_reason_for_empty_streaming_response(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_incomplete_stream",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="unused",
            stream_events=[
                FakeStreamEvent(
                    event_type="response.incomplete",
                    response=FakeResponseWithoutOutputText(
                        response_id="resp_incomplete_stream",
                        output=[FakeReasoningItem()],
                        status="incomplete",
                        incomplete_reason="max_output_tokens",
                    ),
                ),
            ],
        )

        with self.assertRaises(OpenAIConversationReplyGenerationError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
                debug=True,
                stream=True,
            )

        error_text = str(error.exception)
        self.assertIn("empty text response", error_text)
        self.assertIn("response_status=incomplete", error_text)
        self.assertIn("incomplete_reason=max_output_tokens", error_text)

    def test_generate_reply_in_openai_conversation_logs_incomplete_reason_for_empty_streaming_response(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_incomplete_stream_debug",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="unused",
            stream_events=[
                FakeStreamEvent(
                    event_type="response.incomplete",
                    response=FakeResponseWithoutOutputText(
                        response_id="resp_incomplete_stream_debug",
                        output=[FakeReasoningItem()],
                        status="incomplete",
                        incomplete_reason="max_output_tokens",
                    ),
                ),
            ],
        )

        with self.assertLogs("modules.openai_conversation_reply_generator", level="INFO") as captured_logs:
            with self.assertRaises(OpenAIConversationReplyGenerationError):
                generate_reply_in_openai_conversation(
                    conversation.conversation_id,
                    "Tell me about Python.",
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                    dotenv_path=self.missing_dotenv_path,
                    debug=True,
                    stream=True,
                )

        joined_logs = "\n".join(captured_logs.output)
        self.assertIn("stream_event_type=response.incomplete", joined_logs)
        self.assertIn("reply_chars=0", joined_logs)
        self.assertIn("response_status=incomplete", joined_logs)
        self.assertIn("incomplete_reason=max_output_tokens", joined_logs)

    def test_generate_reply_in_openai_conversation_uses_response_incomplete_event_when_stream_has_no_final_response_accessor(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_incomplete_stream_no_accessor",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(
            response_text="unused",
            stream_events=[
                FakeStreamEvent(
                    event_type="response.incomplete",
                    response=FakeResponseWithoutOutputText(
                        response_id="resp_incomplete_stream_no_accessor",
                        output=[FakeReasoningItem()],
                        status="incomplete",
                        incomplete_reason="max_output_tokens",
                    ),
                ),
            ],
            stream_supports_final_response_accessor=False,
        )

        with self.assertRaises(OpenAIConversationReplyGenerationError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
                debug=True,
                stream=True,
            )

        error_text = str(error.exception)
        self.assertIn("response_status=incomplete", error_text)
        self.assertIn("incomplete_reason=max_output_tokens", error_text)

    def test_generate_reply_in_openai_conversation_rejects_missing_api_key_when_building_default_client(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            database_path=self.database_path,
        )

        with self.assertRaises(ValueError) as error:
            generate_reply_in_openai_conversation(
                conversation.conversation_id,
                "Tell me about Python.",
                api_key="   ",
                database_path=self.database_path,
                model="gpt-5-mini",
                dotenv_path=self.missing_dotenv_path,
            )

        self.assertIn("OpenAI API key was not provided", str(error.exception))


class FakeOpenAIClient:
    def __init__(
        self,
        *,
        response_text: str | None = None,
        response_id: str | None = None,
        raised_error: Exception | None = None,
        stream_events: list[object] | None = None,
        stream_final_response: object | None = None,
        stream_supports_final_response_accessor: bool = True,
        usage: object | None = None,
    ) -> None:
        self.recorded_request: dict[str, str | None] | None = None
        self._response_text = response_text
        self._response_id = response_id or "resp_default"
        self._raised_error = raised_error
        self._stream_events = stream_events or []
        self._stream_final_response = stream_final_response
        self._stream_supports_final_response_accessor = stream_supports_final_response_accessor
        self._usage = usage
        self.responses = FakeResponsesAPI(self)


class FakeResponsesAPI:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.owner = owner

    def create(
        self,
        *,
        model: str,
        input: str,
        conversation: str | None = None,
        instructions: str | None = None,
        service_tier: str | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        max_output_tokens: int | None = None,
        reasoning: object | None = None,
        stream: bool = False,
    ) -> object:
        self.owner.recorded_request = {
            "model": model,
            "input": input,
            "stream": stream,
        }
        if conversation is not None:
            self.owner.recorded_request["conversation"] = conversation
        if instructions is not None:
            self.owner.recorded_request["instructions"] = instructions
        if service_tier is not None:
            self.owner.recorded_request["service_tier"] = service_tier
        if prompt_cache_key is not None:
            self.owner.recorded_request["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            self.owner.recorded_request["prompt_cache_retention"] = prompt_cache_retention
        if max_output_tokens is not None:
            self.owner.recorded_request["max_output_tokens"] = max_output_tokens
        if reasoning is not None:
            self.owner.recorded_request["reasoning"] = reasoning
        if self.owner._raised_error is not None:
            raise self.owner._raised_error
        if stream:
            if self.owner._stream_supports_final_response_accessor:
                return FakeResponseStream(
                    self.owner._stream_events,
                    final_response=self.owner._stream_final_response,
                )
            return FakeResponseStreamWithoutFinalResponseAccessor(self.owner._stream_events)
        return FakeResponse(
            output_text=self.owner._response_text,
            response_id=self.owner._response_id,
            usage=self.owner._usage,
        )


class FakeResponse:
    def __init__(self, *, output_text: str | None, response_id: str | None, usage: object | None = None) -> None:
        self.output_text = output_text
        self.id = response_id
        self.usage = usage


class FakeStreamEvent:
    def __init__(
        self,
        *,
        event_type: str,
        delta: str | None = None,
        text: str | None = None,
        response: object | None = None,
        item: object | None = None,
    ) -> None:
        self.type = event_type
        self.delta = delta
        self.text = text
        self.response = response
        self.item = item


class FakeResponseStream:
    def __init__(self, events: list[object], *, final_response: object | None = None) -> None:
        self._events = list(events)
        self._final_response = final_response

    def __iter__(self):
        return iter(self._events)

    def get_final_response(self) -> object | None:
        if self._final_response is not None:
            return self._final_response
        for event in reversed(self._events):
            response = getattr(event, "response", None)
            if response is not None:
                return response
        return None


class FakeResponseStreamWithoutFinalResponseAccessor:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)

    def __iter__(self):
        return iter(self._events)


class FakeResponseWithoutOutputText:
    def __init__(
        self,
        *,
        response_id: str | None,
        output: list[object],
        usage: object | None = None,
        status: str | None = None,
        incomplete_reason: str | None = None,
    ) -> None:
        self.id = response_id
        self.output = output
        self.usage = usage
        self.status = status
        self.incomplete_details = (
            FakeIncompleteDetails(reason=incomplete_reason) if incomplete_reason is not None else None
        )


class FakeOutputMessage:
    def __init__(self, *, content: list[object], status: str = "completed") -> None:
        self.type = "message"
        self.role = "assistant"
        self.content = content
        self.status = status


class FakeOutputTextContent:
    def __init__(self, *, text: str) -> None:
        self.type = "output_text"
        self.text = text


class FakeReasoningItem:
    def __init__(self) -> None:
        self.type = "reasoning"
        self.status = None


class FakeIncompleteDetails:
    def __init__(self, *, reason: str | None) -> None:
        self.reason = reason


class FakePromptTokensDetails:
    def __init__(self, *, cached_tokens: int | None = None) -> None:
        self.cached_tokens = cached_tokens


class FakeUsage:
    def __init__(self, *, cached_tokens: int | None = None) -> None:
        self.prompt_tokens_details = FakePromptTokensDetails(cached_tokens=cached_tokens)


if __name__ == "__main__":
    unittest.main()
