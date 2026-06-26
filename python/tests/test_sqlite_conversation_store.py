from __future__ import annotations

"""Contract tests for SQLite-backed conversation storage."""

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.sqlite_conversation_store import (
    add_conversation_turn,
    add_turn_to_active_conversation,
    create_code_test,
    create_conversation,
    get_active_conversation,
    get_code_test,
    get_conversation,
    get_latest_code_test,
    get_latest_openai_file_id,
    initialize_database,
    list_conversation_turns,
)


class SQLiteConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.database_path = self.root_dir / "database" / "test.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initialize_database_creates_database_file_and_expected_tables(self) -> None:
        resolved_path = initialize_database(self.database_path)

        self.assertEqual(self.database_path.resolve(), resolved_path)
        self.assertTrue(resolved_path.exists())

        with closing(sqlite3.connect(resolved_path)) as connection:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            conversation_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            turn_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(conversation_turns)").fetchall()
            }
            code_test_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(code_tests)").fetchall()
            }

        self.assertEqual({"code_tests", "conversations", "conversation_turns"}, table_names)
        self.assertIn("openai_conversation_id", conversation_columns)
        self.assertIn("openai_file_id", conversation_columns)
        self.assertIn("is_active", conversation_columns)
        self.assertIn("audio_recorded_at", turn_columns)
        self.assertIn("remote_text_translate", turn_columns)
        self.assertIn("reply_text_suggest", turn_columns)
        self.assertEqual(
            {"id", "response_text", "source_file_count", "created_at"},
            code_test_columns,
        )

    def test_initialize_database_enables_wal_journal_mode(self) -> None:
        resolved_path = initialize_database(self.database_path)

        with closing(sqlite3.connect(resolved_path)) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

        self.assertEqual("wal", str(journal_mode).lower())

    def test_initialize_database_adds_code_tests_table_without_changing_existing_tables(self) -> None:
        resolved_path = initialize_database(self.database_path)

        with closing(sqlite3.connect(resolved_path)) as connection:
            conversation_columns = [
                row[1] for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            ]
            turn_columns = [
                row[1] for row in connection.execute("PRAGMA table_info(conversation_turns)").fetchall()
            ]

        self.assertEqual(
            [
                "id",
                "topic_hint",
                "openai_conversation_id",
                "openai_file_id",
                "is_active",
                "created_at",
            ],
            conversation_columns,
        )
        self.assertEqual(
            [
                "id",
                "conversation_id",
                "remote_text",
                "remote_text_translate",
                "reply_text",
                "reply_text_suggest",
                "audio_filename",
                "audio_recorded_at",
                "created_at",
            ],
            turn_columns,
        )

    def test_get_active_conversation_returns_none_when_no_conversation_is_active(self) -> None:
        create_conversation("Python conversation", database_path=self.database_path)
        create_conversation("System design discussion", database_path=self.database_path)

        active_conversation = get_active_conversation(database_path=self.database_path)

        self.assertIsNone(active_conversation)

    def test_get_active_conversation_prefers_latest_explicitly_active_conversation(self) -> None:
        create_conversation("Older default conversation", database_path=self.database_path)
        active_conversation = create_conversation(
            "Conversation session",
            is_active=True,
            database_path=self.database_path,
        )
        create_conversation(
            "Latest inactive conversation",
            is_active=False,
            database_path=self.database_path,
        )

        selected_conversation = get_active_conversation(database_path=self.database_path)

        self.assertEqual(active_conversation, selected_conversation)

    def test_create_conversation_allows_empty_topic_hint(self) -> None:
        conversation = create_conversation("   ", database_path=self.database_path)

        self.assertEqual("", conversation.topic_hint)
        self.assertIsNone(conversation.openai_conversation_id)
        self.assertIsNone(conversation.openai_file_id)
        self.assertIsNone(conversation.is_active)

    def test_create_conversation_stores_openai_conversation_and_file_ids(self) -> None:
        conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            openai_file_id="file_123",
            is_active=True,
            database_path=self.database_path,
        )

        self.assertEqual("conv_123", conversation.openai_conversation_id)
        self.assertEqual("file_123", conversation.openai_file_id)

        active_conversation = get_active_conversation(database_path=self.database_path)

        self.assertIsNotNone(active_conversation)
        self.assertEqual("conv_123", active_conversation.openai_conversation_id)
        self.assertEqual("file_123", active_conversation.openai_file_id)

    def test_get_conversation_returns_row_by_local_conversation_id(self) -> None:
        older_conversation = create_conversation("Older conversation", database_path=self.database_path)
        expected_conversation = create_conversation(
            "Python conversation",
            openai_conversation_id="conv_123",
            openai_file_id="file_123",
            is_active=True,
            database_path=self.database_path,
        )

        selected_conversation = get_conversation(
            expected_conversation.conversation_id,
            database_path=self.database_path,
        )

        self.assertEqual(expected_conversation, selected_conversation)
        self.assertNotEqual(older_conversation, selected_conversation)

    def test_get_conversation_returns_none_when_local_conversation_id_is_missing(self) -> None:
        create_conversation("Python conversation", database_path=self.database_path)

        selected_conversation = get_conversation(999, database_path=self.database_path)

        self.assertIsNone(selected_conversation)

    def test_get_latest_openai_file_id_returns_none_when_no_conversations_exist(self) -> None:
        latest_file_id = get_latest_openai_file_id(database_path=self.database_path)

        self.assertIsNone(latest_file_id)

    def test_get_latest_openai_file_id_returns_none_when_all_values_are_empty(self) -> None:
        create_conversation(
            "First conversation",
            openai_file_id=None,
            database_path=self.database_path,
        )
        create_conversation(
            "Second conversation",
            openai_file_id="   ",
            database_path=self.database_path,
        )

        latest_file_id = get_latest_openai_file_id(database_path=self.database_path)

        self.assertIsNone(latest_file_id)

    def test_get_latest_openai_file_id_returns_latest_non_empty_value(self) -> None:
        create_conversation(
            "First conversation",
            openai_file_id="file_old",
            database_path=self.database_path,
        )
        create_conversation(
            "Second conversation",
            openai_file_id=None,
            database_path=self.database_path,
        )
        create_conversation(
            "Third conversation",
            openai_file_id="file_latest",
            database_path=self.database_path,
        )

        latest_file_id = get_latest_openai_file_id(database_path=self.database_path)

        self.assertEqual("file_latest", latest_file_id)

    def test_create_code_test_stores_required_fields(self) -> None:
        created_code_test = create_code_test(
            response_text="Generated solution summary",
            source_file_count=3,
            database_path=self.database_path,
        )

        self.assertGreater(created_code_test.code_test_id, 0)
        self.assertEqual("Generated solution summary", created_code_test.response_text)
        self.assertEqual(3, created_code_test.source_file_count)
        self.assertTrue(created_code_test.created_at)

    def test_create_code_test_trims_response_text(self) -> None:
        created_code_test = create_code_test(
            response_text="  Generated solution summary  ",
            source_file_count=1,
            database_path=self.database_path,
        )

        self.assertEqual("Generated solution summary", created_code_test.response_text)

    def test_create_code_test_rejects_empty_response_text(self) -> None:
        with self.assertRaises(ValueError) as error:
            create_code_test(
                response_text="   ",
                source_file_count=1,
                database_path=self.database_path,
            )

        self.assertIn("response text", str(error.exception).lower())

    def test_create_code_test_rejects_negative_source_file_count(self) -> None:
        with self.assertRaises(ValueError) as error:
            create_code_test(
                response_text="Generated solution summary",
                source_file_count=-1,
                database_path=self.database_path,
            )

        self.assertIn("source_file_count", str(error.exception))

    def test_initialize_database_creates_code_tests_table_with_non_negative_source_file_count_constraint(self) -> None:
        resolved_path = initialize_database(self.database_path)

        with closing(sqlite3.connect(resolved_path)) as connection:
            create_table_sql = connection.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type = 'table' AND name = 'code_tests'
                """
            ).fetchone()[0]

        self.assertIn("CHECK (source_file_count >= 0)", create_table_sql)

    def test_get_latest_code_test_returns_none_when_no_rows_exist(self) -> None:
        latest_code_test = get_latest_code_test(database_path=self.database_path)

        self.assertIsNone(latest_code_test)

    def test_get_latest_code_test_returns_most_recent_row(self) -> None:
        older_code_test = create_code_test(
            response_text="Older summary",
            source_file_count=1,
            database_path=self.database_path,
        )
        latest_created_code_test = create_code_test(
            response_text="Latest summary",
            source_file_count=2,
            database_path=self.database_path,
        )

        latest_code_test = get_latest_code_test(database_path=self.database_path)

        self.assertEqual(latest_created_code_test, latest_code_test)
        self.assertNotEqual(older_code_test, latest_code_test)

    def test_get_code_test_returns_row_by_id(self) -> None:
        older_code_test = create_code_test(
            response_text="Older summary",
            source_file_count=1,
            database_path=self.database_path,
        )
        expected_code_test = create_code_test(
            response_text="Expected summary",
            source_file_count=2,
            database_path=self.database_path,
        )

        selected_code_test = get_code_test(
            expected_code_test.code_test_id,
            database_path=self.database_path,
        )

        self.assertEqual(expected_code_test, selected_code_test)
        self.assertNotEqual(older_code_test, selected_code_test)

    def test_get_code_test_returns_none_when_id_is_missing(self) -> None:
        create_code_test(
            response_text="Existing summary",
            source_file_count=1,
            database_path=self.database_path,
        )

        selected_code_test = get_code_test(999, database_path=self.database_path)

        self.assertIsNone(selected_code_test)

    def test_create_conversation_stores_nullable_is_active_flag(self) -> None:
        active_conversation = create_conversation(
            "Python conversation",
            is_active=True,
            database_path=self.database_path,
        )
        inactive_conversation = create_conversation(
            "System design conversation",
            is_active=False,
            database_path=self.database_path,
        )
        default_conversation = create_conversation(
            "Behavioral conversation",
            database_path=self.database_path,
        )

        self.assertIs(active_conversation.is_active, True)
        self.assertIs(inactive_conversation.is_active, False)
        self.assertIsNone(default_conversation.is_active)

        latest_conversation = get_active_conversation(database_path=self.database_path)

        self.assertEqual(active_conversation, latest_conversation)

    def test_create_conversation_with_is_active_true_deactivates_previous_active_conversation(self) -> None:
        first_active = create_conversation(
            "First active conversation",
            is_active=True,
            database_path=self.database_path,
        )
        second_active = create_conversation(
            "Second active conversation",
            is_active=True,
            database_path=self.database_path,
        )

        selected_conversation = get_active_conversation(database_path=self.database_path)

        self.assertEqual(second_active, selected_conversation)

        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT id, is_active
                FROM conversations
                ORDER BY id ASC
                """
            ).fetchall()

        self.assertEqual(
            [
                (first_active.conversation_id, 0),
                (second_active.conversation_id, 1),
            ],
            [(int(row[0]), row[1]) for row in rows],
        )

    def test_add_turn_to_active_conversation_links_turn_to_latest_conversation(self) -> None:
        older_conversation = create_conversation("English practice", database_path=self.database_path)
        active_conversation = create_conversation("Frontend conversation", is_active=True, database_path=self.database_path)

        created_turn = add_turn_to_active_conversation(
            remote_text="Tell me about your React experience.",
            remote_text_translate="Расскажите о вашем опыте с React.",
            reply_text="I have built production apps with React and TypeScript.",
            reply_text_suggest="I have strong hands-on React experience in production systems.",
            audio_filename="utterance-20260422-102530-000001.wav",
            database_path=self.database_path,
        )

        self.assertEqual(active_conversation.conversation_id, created_turn.conversation_id)
        self.assertNotEqual(older_conversation.conversation_id, created_turn.conversation_id)
        self.assertEqual("2026-04-22T10:25:30+00:00", created_turn.audio_recorded_at)
        self.assertEqual("Расскажите о вашем опыте с React.", created_turn.remote_text_translate)
        self.assertEqual("I have strong hands-on React experience in production systems.", created_turn.reply_text_suggest)

    def test_list_conversation_turns_returns_turns_in_insert_order_for_one_conversation(self) -> None:
        conversation = create_conversation("Data structures", database_path=self.database_path)
        first_turn = add_conversation_turn(
            conversation_id=conversation.conversation_id,
            remote_text="What is a hash table?",
            remote_text_translate="Что такое хеш-таблица?",
            reply_text="It is a key-value data structure optimized for fast lookup.",
            reply_text_suggest="A hash table stores key-value pairs and is optimized for fast lookup.",
            audio_filename="utterance-20260422-080000-000101.wav",
            database_path=self.database_path,
        )
        second_turn = add_conversation_turn(
            conversation_id=conversation.conversation_id,
            remote_text="What about collisions?",
            reply_text_suggest="Collisions happen when multiple keys end up in the same bucket.",
            audio_filename="utterance-20260422-080100-000102.wav",
            database_path=self.database_path,
        )

        turns = list_conversation_turns(conversation.conversation_id, database_path=self.database_path)

        self.assertEqual([first_turn, second_turn], turns)

    def test_add_turn_to_active_conversation_rejects_missing_active_conversation(self) -> None:
        with self.assertRaises(ValueError) as error:
            add_turn_to_active_conversation(
                remote_text="Hello",
                reply_text_suggest="Hi",
                audio_filename="utterance.wav",
                database_path=self.database_path,
            )

        self.assertIn("no active conversation", str(error.exception).lower())

    def test_add_conversation_turn_rejects_missing_required_text_fields(self) -> None:
        conversation = create_conversation("Algorithms", database_path=self.database_path)

        with self.assertRaises(ValueError):
            add_conversation_turn(
                conversation_id=conversation.conversation_id,
                remote_text="   ",
                reply_text_suggest="Valid suggested reply",
                audio_filename="utterance-20260422-090000-000001.wav",
                database_path=self.database_path,
            )

        with self.assertRaises(ValueError):
            add_conversation_turn(
                conversation_id=conversation.conversation_id,
                remote_text="Valid question",
                reply_text_suggest="   ",
                audio_filename="utterance-20260422-090000-000001.wav",
                database_path=self.database_path,
            )

        with self.assertRaises(ValueError):
            add_conversation_turn(
                conversation_id=conversation.conversation_id,
                remote_text="Valid question",
                reply_text_suggest="Valid suggested reply",
                audio_filename="   ",
                database_path=self.database_path,
            )

    def test_add_conversation_turn_rejects_audio_filename_without_embedded_timestamp(self) -> None:
        conversation = create_conversation("Algorithms", database_path=self.database_path)

        with self.assertRaises(ValueError) as error:
            add_conversation_turn(
                conversation_id=conversation.conversation_id,
                remote_text="Valid question",
                reply_text_suggest="Valid suggested reply",
                audio_filename="utterance.wav",
                database_path=self.database_path,
            )

        self.assertIn("timestamp", str(error.exception).lower())

    def test_add_conversation_turn_allows_explicit_audio_recorded_at_without_audio_filename(self) -> None:
        conversation = create_conversation("Algorithms", database_path=self.database_path)

        created_turn = add_conversation_turn(
            conversation_id=conversation.conversation_id,
            remote_text="Explain binary search.",
            reply_text_suggest="Binary search works on sorted data by halving the search space.",
            audio_filename=None,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            database_path=self.database_path,
        )

        self.assertIsNone(created_turn.audio_filename)
        self.assertEqual("2026-04-24T10:11:12+00:00", created_turn.audio_recorded_at)

    def test_add_conversation_turn_rejects_missing_audio_filename_and_audio_recorded_at(self) -> None:
        conversation = create_conversation("Algorithms", database_path=self.database_path)

        with self.assertRaises(ValueError) as error:
            add_conversation_turn(
                conversation_id=conversation.conversation_id,
                remote_text="Explain binary search.",
                reply_text_suggest="Binary search works on sorted data by halving the search space.",
                audio_filename=None,
                audio_recorded_at=None,
                database_path=self.database_path,
            )

        self.assertIn("audio_filename", str(error.exception))

    def test_add_turn_to_active_conversation_allows_explicit_audio_recorded_at_without_audio_filename(self) -> None:
        create_conversation("Algorithms", is_active=True, database_path=self.database_path)

        created_turn = add_turn_to_active_conversation(
            remote_text="Explain binary search.",
            reply_text_suggest="Binary search works on sorted data by halving the search space.",
            audio_filename=None,
            audio_recorded_at="2026-04-24T10:11:12+00:00",
            database_path=self.database_path,
        )

        self.assertIsNone(created_turn.audio_filename)
        self.assertEqual("2026-04-24T10:11:12+00:00", created_turn.audio_recorded_at)

    def test_add_conversation_turn_allows_optional_reply_and_translation_to_be_missing(self) -> None:
        conversation = create_conversation("Algorithms", database_path=self.database_path)

        created_turn = add_conversation_turn(
            conversation_id=conversation.conversation_id,
            remote_text="Explain binary search.",
            reply_text_suggest="Binary search works on sorted data by halving the search space.",
            audio_filename="utterance-20260422-090000-000001.wav",
            database_path=self.database_path,
        )

        self.assertIsNone(created_turn.remote_text_translate)
        self.assertIsNone(created_turn.reply_text)
        self.assertEqual(
            "Binary search works on sorted data by halving the search space.",
            created_turn.reply_text_suggest,
        )

    def test_initialize_database_migrates_legacy_schema_where_reply_text_was_not_nullable(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_hint TEXT NOT NULL,
                        openai_conversation_id TEXT,
                        openai_file_id TEXT,
                        is_active INTEGER,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER NOT NULL,
                        remote_text TEXT NOT NULL,
                        reply_text TEXT NOT NULL,
                        audio_filename TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    );
                    """
                )

        initialize_database(self.database_path)
        conversation = create_conversation("Legacy conversation", database_path=self.database_path)

        created_turn = add_conversation_turn(
            conversation_id=conversation.conversation_id,
            remote_text="Tell me about yourself.",
            reply_text_suggest="I am a software engineer with backend and frontend experience.",
            audio_filename="utterance-20260422-090000-000001.wav",
            reply_text=None,
            database_path=self.database_path,
        )

        self.assertIsNone(created_turn.reply_text)

    def test_initialize_database_migrates_legacy_schema_where_audio_filename_was_not_nullable(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_hint TEXT NOT NULL,
                        openai_conversation_id TEXT,
                        openai_file_id TEXT,
                        is_active INTEGER,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id INTEGER NOT NULL,
                        remote_text TEXT NOT NULL,
                        remote_text_translate TEXT,
                        reply_text TEXT,
                        reply_text_suggest TEXT NOT NULL,
                        audio_filename TEXT NOT NULL,
                        audio_recorded_at TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    );
                    """
                )

        initialize_database(self.database_path)
        with closing(sqlite3.connect(self.database_path)) as connection:
            audio_filename_column = next(
                row for row in connection.execute("PRAGMA table_info(conversation_turns)").fetchall() if row[1] == "audio_filename"
            )

        self.assertEqual(0, audio_filename_column[3])


if __name__ == "__main__":
    unittest.main()
