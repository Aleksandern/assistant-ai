from __future__ import annotations

"""Contract tests for OpenAI conversation initialization."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.openai_conversation_initializer import (
    OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR,
    OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR,
    OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR,
    OpenAIConversationInitializationError,
    initialize_openai_conversation,
)
from modules.sqlite_conversation_store import create_conversation, get_active_conversation


class OpenAIConversationInitializerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.file_dir = self.root_dir / "data" / "file"
        self.file_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root_dir / "database" / "test.sqlite3"
        self.dotenv_path = self.root_dir / ".env"
        self._original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._original_openai_model = os.environ.get("OPENAI_MODEL")
        self._original_env_openai_file_id = os.environ.get("OPENAI_FILE_ID")
        self._original_conversation_instructions = os.environ.get(OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR)
        self._original_file_message = os.environ.get(OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR)
        self._original_topic_hint_template = os.environ.get(OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR)
        self.default_conversation_instructions = (
            "Answer as me in a live interview. Keep it natural, short, spoken, and honest."
        )
        self.default_file_message = "Use the attached DOCX file as the source of truth about the candidate."
        self.default_topic_hint_template = "Conversation topic hint: {topic_hint}"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_MODEL", None)
        os.environ.pop("OPENAI_FILE_ID", None)
        os.environ[OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR] = self.default_conversation_instructions
        os.environ[OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR] = self.default_file_message
        os.environ[OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR] = self.default_topic_hint_template

    def tearDown(self) -> None:
        if self._original_openai_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._original_openai_api_key

        if self._original_openai_model is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = self._original_openai_model

        if self._original_env_openai_file_id is None:
            os.environ.pop("OPENAI_FILE_ID", None)
        else:
            os.environ["OPENAI_FILE_ID"] = self._original_env_openai_file_id

        if self._original_conversation_instructions is None:
            os.environ.pop(OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR, None)
        else:
            os.environ[OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR] = self._original_conversation_instructions

        if self._original_file_message is None:
            os.environ.pop(OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR, None)
        else:
            os.environ[OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR] = self._original_file_message

        if self._original_topic_hint_template is None:
            os.environ.pop(OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR, None)
        else:
            os.environ[OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR] = self._original_topic_hint_template

        self.temp_dir.cleanup()

    def test_initialize_openai_conversation_creates_openai_conversation_and_local_row(self) -> None:
        file_path = self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(conversation_id="conv_123")

        initialized = initialize_openai_conversation(
            "Python conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
        )

        self.assertEqual(1, initialized.conversation_id)
        self.assertEqual("conv_123", initialized.openai_conversation_id)
        self.assertEqual("Python conversation", initialized.topic_hint)
        self.assertEqual("file.docx", initialized.file_name)
        self.assertEqual(1, client.create_call_count)
        self.assertEqual(1, client.file_upload_call_count)

        active_conversation = get_active_conversation(database_path=self.database_path)
        self.assertIsNotNone(active_conversation)
        self.assertEqual("conv_123", active_conversation.openai_conversation_id)
        self.assertEqual("file_123", active_conversation.openai_file_id)
        self.assertIs(active_conversation.is_active, True)
        self.assertFalse(file_path.exists())

    def test_initialize_openai_conversation_uses_first_docx_and_references_uploaded_file_in_initial_items(self) -> None:
        self._create_file_docx("b-file.docx", b"second")
        self._create_file_docx("a-file.docx", b"first")
        client = FakeOpenAIClient(conversation_id="conv_abc", uploaded_file_id="file_abc")

        initialize_openai_conversation(
            "Backend conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
        )

        recorded_request = client.recorded_create_request
        self.assertIsNotNone(recorded_request)

        items = recorded_request["items"]
        self.assertEqual("developer", items[0]["role"])
        self.assertEqual("user", items[1]["role"])

        user_content = items[1]["content"]
        input_file_item = next(item for item in user_content if item["type"] == "input_file")
        self.assertEqual("file_abc", input_file_item["file_id"])
        self.assertNotIn("filename", input_file_item)
        self.assertEqual(
            {
                "file": ("a-file.docx", b"first"),
                "purpose": "user_data",
            },
            client.recorded_file_upload_request,
        )

    def test_initialize_openai_conversation_uses_default_instructions(self) -> None:
        self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(conversation_id="conv_123")

        initialize_openai_conversation(
            "System design",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
        )

        developer_message = client.recorded_create_request["items"][0]
        self.assertEqual(
            self.default_conversation_instructions,
            developer_message["content"][0]["text"],
        )

    def test_initialize_openai_conversation_uses_configured_file_message_and_topic_hint_template(self) -> None:
        self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(conversation_id="conv_topic")

        initialize_openai_conversation(
            "Distributed systems",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
        )

        user_content = client.recorded_create_request["items"][1]["content"]
        self.assertEqual(self.default_file_message, user_content[0]["text"])
        self.assertEqual("Conversation topic hint: Distributed systems", user_content[1]["text"])

    def test_initialize_openai_conversation_reads_instructions_from_dotenv_file(self) -> None:
        self._create_file_docx("file.docx", b"file-binary")
        os.environ.pop(OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR, None)
        dotenv_instructions = "Speak as the candidate in one concise answer."
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_MODEL=gpt-5-nano\n"
            f"{OPENAI_CONVERSATION_INSTRUCTIONS_ENV_VAR}={dotenv_instructions}\n"
            f"{OPENAI_CONVERSATION_FILE_MESSAGE_ENV_VAR}=Use the attached candidate profile.\n"
            f"{OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR}=Topic hint: {{topic_hint}}\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(conversation_id="conv_123")

        initialize_openai_conversation(
            "Behavioral conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual(
            dotenv_instructions,
            client.recorded_create_request["items"][0]["content"][0]["text"],
        )

    def test_initialize_openai_conversation_rejects_topic_hint_template_without_placeholder(self) -> None:
        self._create_file_docx("file.docx", b"file-binary")
        os.environ[OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE_ENV_VAR] = "Topic hint only"

        with self.assertRaisesRegex(OpenAIConversationInitializationError, "placeholder exactly once"):
            initialize_openai_conversation(
                "Behavioral conversation",
                file_dir=self.file_dir,
                client=FakeOpenAIClient(conversation_id="conv_invalid_template"),
                database_path=self.database_path,
                model="gpt-5-mini",
            )

    def test_initialize_openai_conversation_reads_model_from_dotenv_file(self) -> None:
        self._create_file_docx("file.docx", b"file-binary")
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_MODEL=gpt-5-nano\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(conversation_id="conv_123")

        initialize_openai_conversation(
            "Behavioral conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual("gpt-5-nano", client.recorded_create_request["metadata"]["model"])

    def test_initialize_openai_conversation_raises_clear_error_when_file_is_missing(self) -> None:
        with self.assertRaises(OpenAIConversationInitializationError) as error:
            initialize_openai_conversation(
                "Missing file conversation",
                file_dir=self.file_dir,
                client=FakeOpenAIClient(conversation_id="unused"),
                database_path=self.database_path,
                model="gpt-5-mini",
            )

        self.assertIn("no local .docx file", str(error.exception))
        self.assertIn("OPENAI_FILE_ID", str(error.exception))
        self.assertIn("saved openai_file_id in the database", str(error.exception))

    def test_initialize_openai_conversation_uses_env_openai_file_id_when_docx_is_missing(self) -> None:
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_FILE_ID= file_env_123 \n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(conversation_id="conv_env")

        initialized = initialize_openai_conversation(
            "Env file conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            dotenv_path=self.dotenv_path,
            model="gpt-5-mini",
        )

        self.assertIsNone(initialized.file_name)
        self.assertEqual(0, client.file_upload_call_count)
        user_content = client.recorded_create_request["items"][1]["content"]
        input_file_item = next(item for item in user_content if item["type"] == "input_file")
        self.assertEqual("file_env_123", input_file_item["file_id"])
        self.assertEqual({"model": "gpt-5-mini", "topic_hint": "Env file conversation"}, client.recorded_create_request["metadata"])

        active_conversation = get_active_conversation(database_path=self.database_path)
        self.assertIsNotNone(active_conversation)
        self.assertEqual("file_env_123", active_conversation.openai_file_id)

    def test_initialize_openai_conversation_uses_db_openai_file_id_when_docx_and_env_are_missing(self) -> None:
        create_conversation(
            "Earlier conversation",
            openai_file_id="file_db_123",
            database_path=self.database_path,
        )
        client = FakeOpenAIClient(conversation_id="conv_db")

        initialized = initialize_openai_conversation(
            "DB file conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            model="gpt-5-mini",
        )

        self.assertIsNone(initialized.file_name)
        self.assertEqual(0, client.file_upload_call_count)
        user_content = client.recorded_create_request["items"][1]["content"]
        input_file_item = next(item for item in user_content if item["type"] == "input_file")
        self.assertEqual("file_db_123", input_file_item["file_id"])

        active_conversation = get_active_conversation(database_path=self.database_path)
        self.assertIsNotNone(active_conversation)
        self.assertEqual("file_db_123", active_conversation.openai_file_id)

    def test_initialize_openai_conversation_prefers_env_openai_file_id_over_db(self) -> None:
        create_conversation(
            "Earlier conversation",
            openai_file_id="file_db_123",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_FILE_ID= file_env_456 \n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(conversation_id="conv_env_priority")

        initialize_openai_conversation(
            "Env priority conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            dotenv_path=self.dotenv_path,
            model="gpt-5-mini",
        )

        self.assertEqual(0, client.file_upload_call_count)
        user_content = client.recorded_create_request["items"][1]["content"]
        input_file_item = next(item for item in user_content if item["type"] == "input_file")
        self.assertEqual("file_env_456", input_file_item["file_id"])

    def test_initialize_openai_conversation_prefers_local_docx_over_env_and_db(self) -> None:
        file_path = self._create_file_docx("file.docx", b"file-binary")
        create_conversation(
            "Earlier conversation",
            openai_file_id="file_db_123",
            database_path=self.database_path,
        )
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_FILE_ID=file_env_789\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(conversation_id="conv_docx_priority", uploaded_file_id="file_uploaded_789")

        initialized = initialize_openai_conversation(
            "Docx priority conversation",
            file_dir=self.file_dir,
            client=client,
            database_path=self.database_path,
            dotenv_path=self.dotenv_path,
            model="gpt-5-mini",
        )

        self.assertEqual("file.docx", initialized.file_name)
        self.assertEqual(1, client.file_upload_call_count)
        self.assertEqual(
            {
                "file": ("file.docx", b"file-binary"),
                "purpose": "user_data",
            },
            client.recorded_file_upload_request,
        )
        user_content = client.recorded_create_request["items"][1]["content"]
        input_file_item = next(item for item in user_content if item["type"] == "input_file")
        self.assertEqual("file_uploaded_789", input_file_item["file_id"])
        self.assertFalse(file_path.exists())

    def test_initialize_openai_conversation_reports_openai_failure(self) -> None:
        file_path = self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(raised_error=RuntimeError("upstream unavailable"))

        with self.assertRaises(OpenAIConversationInitializationError) as error:
            initialize_openai_conversation(
                "OpenAI failure conversation",
                file_dir=self.file_dir,
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
            )

        self.assertIn("Failed to create or initialize OpenAI conversation", str(error.exception))
        self.assertIn("upstream unavailable", str(error.exception))
        self.assertIsNone(error.exception.openai_conversation_id)
        self.assertTrue(file_path.exists())

    def test_initialize_openai_conversation_reports_file_upload_failure(self) -> None:
        file_path = self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(file_upload_error=RuntimeError("upload rejected"))

        with self.assertRaises(OpenAIConversationInitializationError) as error:
            initialize_openai_conversation(
                "Upload failure conversation",
                file_dir=self.file_dir,
                client=client,
                database_path=self.database_path,
                model="gpt-5-mini",
            )

        self.assertIn("Failed to upload DOCX file to OpenAI", str(error.exception))
        self.assertIn("upload rejected", str(error.exception))
        self.assertIsNone(error.exception.openai_conversation_id)
        self.assertTrue(file_path.exists())

    def test_initialize_openai_conversation_reports_local_database_failure_with_partial_result(self) -> None:
        file_path = self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(conversation_id="conv_partial")

        with patch(
            "modules.openai_conversation_initializer.create_conversation",
            side_effect=RuntimeError("database locked"),
        ):
            with self.assertRaises(OpenAIConversationInitializationError) as error:
                initialize_openai_conversation(
                    "DB failure conversation",
                    file_dir=self.file_dir,
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                )

        self.assertEqual("conv_partial", error.exception.openai_conversation_id)
        self.assertIn("database locked", str(error.exception))
        self.assertIn("openai_conversation_id=conv_partial", str(error.exception))
        self.assertTrue(file_path.exists())

    def test_initialize_openai_conversation_does_not_upload_or_delete_file_for_env_reuse(self) -> None:
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_FILE_ID=file_env_keep\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(conversation_id="conv_env_keep")

        with patch("modules.openai_conversation_initializer._delete_local_docx") as delete_mock:
            initialize_openai_conversation(
                "Env reuse conversation",
                file_dir=self.file_dir,
                client=client,
                database_path=self.database_path,
                dotenv_path=self.dotenv_path,
                model="gpt-5-mini",
            )

        self.assertEqual(0, client.file_upload_call_count)
        delete_mock.assert_not_called()

    def test_initialize_openai_conversation_reports_file_delete_failure_after_full_success(self) -> None:
        file_path = self._create_file_docx("file.docx", b"file-binary")
        client = FakeOpenAIClient(conversation_id="conv_delete_failure", uploaded_file_id="file_delete_failure")

        with patch(
            "modules.openai_conversation_initializer.Path.unlink",
            side_effect=OSError("permission denied"),
        ):
            with self.assertRaises(OpenAIConversationInitializationError) as error:
                initialize_openai_conversation(
                    "Delete failure conversation",
                    file_dir=self.file_dir,
                    client=client,
                    database_path=self.database_path,
                    model="gpt-5-mini",
                )

        self.assertEqual("conv_delete_failure", error.exception.openai_conversation_id)
        self.assertIn("failed to delete local DOCX file", str(error.exception))
        self.assertIn("openai_conversation_id=conv_delete_failure", str(error.exception))
        self.assertTrue(file_path.exists())

        active_conversation = get_active_conversation(database_path=self.database_path)
        self.assertIsNotNone(active_conversation)
        self.assertEqual("conv_delete_failure", active_conversation.openai_conversation_id)
        self.assertEqual("file_delete_failure", active_conversation.openai_file_id)

    def test_initialize_openai_conversation_rejects_missing_api_key_when_building_default_client(self) -> None:
        self._create_file_docx("file.docx", b"file-binary")

        with self.assertRaises(ValueError) as error:
            initialize_openai_conversation(
                "API key conversation",
                file_dir=self.file_dir,
                api_key="   ",
                database_path=self.database_path,
                model="gpt-5-mini",
            )

        self.assertIn("OpenAI API key was not provided", str(error.exception))

    def _create_file_docx(self, filename: str, payload: bytes) -> Path:
        path = self.file_dir / filename
        path.write_bytes(payload)
        return path


class FakeOpenAIClient:
    def __init__(
        self,
        *,
        conversation_id: str | None = None,
        uploaded_file_id: str | None = None,
        raised_error: Exception | None = None,
        file_upload_error: Exception | None = None,
    ) -> None:
        self.create_call_count = 0
        self.file_upload_call_count = 0
        self.recorded_create_request: dict[str, object] | None = None
        self.recorded_file_upload_request: dict[str, object] | None = None
        self._conversation_id = conversation_id
        self._uploaded_file_id = uploaded_file_id or "file_123"
        self._raised_error = raised_error
        self._file_upload_error = file_upload_error
        self.files = FakeFilesAPI(self)
        self.conversations = FakeConversationsAPI(self)


class FakeFilesAPI:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.owner = owner

    def create(self, *, file: tuple[str, bytes], purpose: str) -> object:
        self.owner.file_upload_call_count += 1
        self.owner.recorded_file_upload_request = {
            "file": file,
            "purpose": purpose,
        }
        if self.owner._file_upload_error is not None:
            raise self.owner._file_upload_error
        return FakeUploadedFile(self.owner._uploaded_file_id)


class FakeConversationsAPI:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.owner = owner

    def create(self, *, items: list[dict[str, object]], metadata: dict[str, str]) -> object:
        self.owner.create_call_count += 1
        self.owner.recorded_create_request = {
            "items": items,
            "metadata": metadata,
        }
        if self.owner._raised_error is not None:
            raise self.owner._raised_error
        return FakeConversation(self.owner._conversation_id)


class FakeConversation:
    def __init__(self, conversation_id: str | None) -> None:
        self.id = conversation_id


class FakeUploadedFile:
    def __init__(self, file_id: str | None) -> None:
        self.id = file_id


if __name__ == "__main__":
    unittest.main()
