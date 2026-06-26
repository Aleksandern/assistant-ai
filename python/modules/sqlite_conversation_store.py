from __future__ import annotations

"""SQLite storage for conversations and generated reply history."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
DEFAULT_DATABASE_DIR = REPOSITORY_ROOT / "database"
DEFAULT_DATABASE_PATH = DEFAULT_DATABASE_DIR / "assistantai.sqlite3"


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: int
    topic_hint: str
    openai_conversation_id: str | None
    openai_file_id: str | None
    is_active: bool | None
    created_at: str


@dataclass(frozen=True)
class ConversationTurnRecord:
    turn_id: int
    conversation_id: int
    remote_text: str
    remote_text_translate: str | None
    reply_text: str | None
    reply_text_suggest: str
    audio_filename: str | None
    audio_recorded_at: str
    created_at: str


@dataclass(frozen=True)
class CodeTestRecord:
    code_test_id: int
    response_text: str
    source_file_count: int
    created_at: str


def get_database_path(database_path: str | Path | None = None) -> Path:
    resolved_database_path = Path(database_path or DEFAULT_DATABASE_PATH).expanduser()
    resolved_database_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_database_path.resolve()


def connect_database(database_path: str | Path | None = None) -> sqlite3.Connection:
    resolved_database_path = get_database_path(database_path)
    connection = sqlite3.connect(resolved_database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def open_database(database_path: str | Path | None = None):
    connection = connect_database(database_path)
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(database_path: str | Path | None = None) -> Path:
    resolved_database_path = get_database_path(database_path)
    with open_database(resolved_database_path) as connection:
        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_hint TEXT NOT NULL,
                    openai_conversation_id TEXT,
                    openai_file_id TEXT,
                    is_active INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                remote_text TEXT NOT NULL,
                remote_text_translate TEXT,
                reply_text TEXT,
                reply_text_suggest TEXT NOT NULL,
                audio_filename TEXT,
                audio_recorded_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS code_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_text TEXT NOT NULL,
                    source_file_count INTEGER NOT NULL CHECK (source_file_count >= 0),
                    created_at TEXT NOT NULL
                );
                """
            )
            _ensure_conversation_turns_schema(connection)
    return resolved_database_path


def create_conversation(
    topic_hint: str,
    *,
    openai_conversation_id: str | None = None,
    openai_file_id: str | None = None,
    is_active: bool | None = None,
    database_path: str | Path | None = None,
) -> ConversationRecord:
    normalized_topic_hint = topic_hint.strip()
    normalized_openai_conversation_id = _normalize_optional_text(openai_conversation_id)
    normalized_openai_file_id = _normalize_optional_text(openai_file_id)
    normalized_is_active = _normalize_optional_bool(is_active)

    initialize_database(database_path)
    created_at = _make_timestamp()

    with open_database(database_path) as connection:
        with connection:
            if normalized_is_active == 1:
                connection.execute(
                    """
                    UPDATE conversations
                    SET is_active = 0
                    WHERE is_active = 1
                    """
                )
            cursor = connection.execute(
                """
                INSERT INTO conversations (
                    topic_hint,
                    openai_conversation_id,
                    openai_file_id,
                    is_active,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    normalized_topic_hint,
                    normalized_openai_conversation_id,
                    normalized_openai_file_id,
                    normalized_is_active,
                    created_at,
                ),
            )
            conversation_id = int(cursor.lastrowid)

    return ConversationRecord(
        conversation_id=conversation_id,
        topic_hint=normalized_topic_hint,
        openai_conversation_id=normalized_openai_conversation_id,
        openai_file_id=normalized_openai_file_id,
        is_active=is_active,
        created_at=created_at,
    )


def get_active_conversation(*, database_path: str | Path | None = None) -> ConversationRecord | None:
    initialize_database(database_path)

    with open_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, topic_hint, openai_conversation_id, openai_file_id, is_active, created_at
            FROM conversations
            WHERE is_active = 1
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return _build_conversation_record(row)


def get_conversation(
    conversation_id: int,
    *,
    database_path: str | Path | None = None,
) -> ConversationRecord | None:
    initialize_database(database_path)

    with open_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, topic_hint, openai_conversation_id, openai_file_id, is_active, created_at
            FROM conversations
            WHERE id = ?
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()

    if row is None:
        return None

    return _build_conversation_record(row)


def get_latest_openai_file_id(
    *,
    database_path: str | Path | None = None,
) -> str | None:
    initialize_database(database_path)

    with open_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT openai_file_id
            FROM conversations
            WHERE openai_file_id IS NOT NULL AND TRIM(openai_file_id) != ''
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return _coerce_nullable_text(row["openai_file_id"])


def create_code_test(
    response_text: str,
    source_file_count: int,
    *,
    database_path: str | Path | None = None,
) -> CodeTestRecord:
    normalized_response_text = response_text.strip()
    if not normalized_response_text:
        raise ValueError("Code test response text must not be empty.")
    if source_file_count < 0:
        raise ValueError("source_file_count must be greater than or equal to 0.")

    initialize_database(database_path)
    created_at = _make_timestamp()

    with open_database(database_path) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO code_tests (
                    response_text,
                    source_file_count,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    normalized_response_text,
                    source_file_count,
                    created_at,
                ),
            )
            code_test_id = int(cursor.lastrowid)

    return CodeTestRecord(
        code_test_id=code_test_id,
        response_text=normalized_response_text,
        source_file_count=source_file_count,
        created_at=created_at,
    )


def get_latest_code_test(*, database_path: str | Path | None = None) -> CodeTestRecord | None:
    initialize_database(database_path)

    with open_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, response_text, source_file_count, created_at
            FROM code_tests
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return _build_code_test_record(row)


def get_code_test(
    code_test_id: int,
    *,
    database_path: str | Path | None = None,
) -> CodeTestRecord | None:
    initialize_database(database_path)

    with open_database(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, response_text, source_file_count, created_at
            FROM code_tests
            WHERE id = ?
            LIMIT 1
            """,
            (code_test_id,),
        ).fetchone()

    if row is None:
        return None

    return _build_code_test_record(row)


def add_turn_to_active_conversation(
    remote_text: str,
    reply_text_suggest: str,
    audio_filename: str | None = None,
    *,
    audio_recorded_at: str | datetime | None = None,
    remote_text_translate: str | None = None,
    reply_text: str | None = None,
    database_path: str | Path | None = None,
) -> ConversationTurnRecord:
    active_conversation = get_active_conversation(database_path=database_path)
    if active_conversation is None:
        raise ValueError("Cannot add a conversation turn because there is no active conversation.")

    return add_conversation_turn(
        conversation_id=active_conversation.conversation_id,
        remote_text=remote_text,
        remote_text_translate=remote_text_translate,
        reply_text=reply_text,
        reply_text_suggest=reply_text_suggest,
        audio_filename=audio_filename,
        audio_recorded_at=audio_recorded_at,
        database_path=database_path,
    )


def add_conversation_turn(
    conversation_id: int,
    remote_text: str,
    reply_text_suggest: str,
    audio_filename: str | None = None,
    *,
    audio_recorded_at: str | datetime | None = None,
    remote_text_translate: str | None = None,
    reply_text: str | None = None,
    database_path: str | Path | None = None,
) -> ConversationTurnRecord:
    normalized_remote_text = remote_text.strip()
    normalized_remote_text_translate = _normalize_optional_text(remote_text_translate)
    normalized_reply_text = _normalize_optional_text(reply_text)
    normalized_reply_text_suggest = reply_text_suggest.strip()
    normalized_audio_filename = _normalize_optional_text(audio_filename)
    normalized_audio_recorded_at = _normalize_optional_datetime_text(audio_recorded_at)

    if not normalized_remote_text:
        raise ValueError("Remote speaker text must not be empty.")
    if not normalized_reply_text_suggest:
        raise ValueError("Suggested reply text must not be empty.")
    if normalized_audio_filename is None and normalized_audio_recorded_at is None:
        raise ValueError("Either audio_filename or audio_recorded_at must be provided.")

    initialize_database(database_path)
    created_at = _make_timestamp()
    resolved_audio_recorded_at = normalized_audio_recorded_at
    if resolved_audio_recorded_at is None and normalized_audio_filename is not None:
        resolved_audio_recorded_at = _parse_audio_recorded_at(normalized_audio_filename)
    if resolved_audio_recorded_at is None:
        raise ValueError("Audio recorded timestamp must not be empty.")

    with open_database(database_path) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO conversation_turns (
                    conversation_id,
                    remote_text,
                    remote_text_translate,
                    reply_text,
                    reply_text_suggest,
                    audio_filename,
                    audio_recorded_at,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    normalized_remote_text,
                    normalized_remote_text_translate,
                    normalized_reply_text,
                    normalized_reply_text_suggest,
                    normalized_audio_filename,
                    resolved_audio_recorded_at,
                    created_at,
                ),
            )
            turn_id = int(cursor.lastrowid)

    return ConversationTurnRecord(
        turn_id=turn_id,
        conversation_id=conversation_id,
        remote_text=normalized_remote_text,
        remote_text_translate=normalized_remote_text_translate,
        reply_text=normalized_reply_text,
        reply_text_suggest=normalized_reply_text_suggest,
        audio_filename=normalized_audio_filename,
        audio_recorded_at=resolved_audio_recorded_at,
        created_at=created_at,
    )


def list_conversation_turns(
    conversation_id: int,
    *,
    database_path: str | Path | None = None,
) -> list[ConversationTurnRecord]:
    initialize_database(database_path)

    with open_database(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                conversation_id,
                remote_text,
                remote_text_translate,
                reply_text,
                reply_text_suggest,
                audio_filename,
                audio_recorded_at,
                created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    return [_build_conversation_turn_record(row) for row in rows]


def _build_conversation_record(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=int(row["id"]),
        topic_hint=str(row["topic_hint"]),
        openai_conversation_id=_coerce_nullable_text(row["openai_conversation_id"]),
        openai_file_id=_coerce_nullable_text(row["openai_file_id"]),
        is_active=_coerce_nullable_bool(row["is_active"]),
        created_at=str(row["created_at"]),
    )


def _build_conversation_turn_record(row: sqlite3.Row) -> ConversationTurnRecord:
    return ConversationTurnRecord(
        turn_id=int(row["id"]),
        conversation_id=int(row["conversation_id"]),
        remote_text=str(row["remote_text"]),
        remote_text_translate=_coerce_nullable_text(row["remote_text_translate"]),
        reply_text=_coerce_nullable_text(row["reply_text"]),
        reply_text_suggest=str(row["reply_text_suggest"]),
        audio_filename=_coerce_nullable_text(row["audio_filename"]),
        audio_recorded_at=str(row["audio_recorded_at"]),
        created_at=str(row["created_at"]),
    )


def _build_code_test_record(row: sqlite3.Row) -> CodeTestRecord:
    return CodeTestRecord(
        code_test_id=int(row["id"]),
        response_text=str(row["response_text"]),
        source_file_count=int(row["source_file_count"]),
        created_at=str(row["created_at"]),
    )


def _make_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _ensure_conversation_turns_schema(connection: sqlite3.Connection) -> None:
    table_info_rows = connection.execute("PRAGMA table_info(conversation_turns)").fetchall()
    column_names = {str(row["name"]) for row in table_info_rows}

    if _conversation_turns_requires_rebuild(table_info_rows):
        _rebuild_conversation_turns_table(connection, column_names)
        return

    if "audio_recorded_at" not in column_names:
        connection.execute(
            """
            ALTER TABLE conversation_turns
            ADD COLUMN audio_recorded_at TEXT NOT NULL DEFAULT ''
            """
        )
    if "remote_text_translate" not in column_names:
        connection.execute(
            """
            ALTER TABLE conversation_turns
            ADD COLUMN remote_text_translate TEXT
            """
        )
    if "reply_text_suggest" not in column_names:
        connection.execute(
            """
            ALTER TABLE conversation_turns
            ADD COLUMN reply_text_suggest TEXT NOT NULL DEFAULT ''
            """
        )


def _conversation_turns_requires_rebuild(table_info_rows: list[sqlite3.Row]) -> bool:
    for row in table_info_rows:
        if str(row["name"]) == "reply_text" and bool(row["notnull"]):
            return True
        if str(row["name"]) == "audio_filename" and bool(row["notnull"]):
            return True
    return False


def _rebuild_conversation_turns_table(
    connection: sqlite3.Connection,
    existing_column_names: set[str],
) -> None:
    existing_rows = connection.execute(
        "SELECT * FROM conversation_turns ORDER BY id ASC"
    ).fetchall()

    connection.execute("ALTER TABLE conversation_turns RENAME TO conversation_turns_legacy")
    connection.execute(
        """
        CREATE TABLE conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            remote_text TEXT NOT NULL,
            remote_text_translate TEXT,
            reply_text TEXT,
            reply_text_suggest TEXT NOT NULL,
            audio_filename TEXT,
            audio_recorded_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )

    for row in existing_rows:
        remote_text_translate = (
            _coerce_nullable_text(row["remote_text_translate"])
            if "remote_text_translate" in existing_column_names
            else None
        )
        reply_text = _coerce_nullable_text(row["reply_text"]) if "reply_text" in existing_column_names else None
        reply_text_suggest = (
            _coerce_nullable_text(row["reply_text_suggest"])
            if "reply_text_suggest" in existing_column_names
            else None
        )
        audio_filename = _coerce_nullable_text(row["audio_filename"])
        audio_recorded_at = (
            _coerce_nullable_text(row["audio_recorded_at"])
            if "audio_recorded_at" in existing_column_names
            else None
        )

        connection.execute(
            """
            INSERT INTO conversation_turns (
                id,
                conversation_id,
                remote_text,
                remote_text_translate,
                reply_text,
                reply_text_suggest,
                audio_filename,
                audio_recorded_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["id"]),
                int(row["conversation_id"]),
                str(row["remote_text"]),
                remote_text_translate,
                reply_text,
                reply_text_suggest or reply_text or "",
                audio_filename,
                audio_recorded_at or _parse_audio_recorded_at_if_possible(audio_filename),
                str(row["created_at"]),
            ),
        )

    connection.execute("DROP TABLE conversation_turns_legacy")


def _parse_audio_recorded_at(audio_filename: str) -> str:
    filename_stem = Path(audio_filename).stem
    parts = filename_stem.split("-")
    if len(parts) < 4:
        raise ValueError(
            "Audio filename must contain a timestamp like `prefix-YYYYMMDD-HHMMSS-micros.wav`."
        )

    date_part = parts[-3]
    time_part = parts[-2]
    try:
        parsed_datetime = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(
            "Audio filename must contain a timestamp like `prefix-YYYYMMDD-HHMMSS-micros.wav`."
        ) from exc

    return parsed_datetime.replace(tzinfo=UTC).isoformat(timespec="seconds")


def _parse_audio_recorded_at_if_possible(audio_filename: str | None) -> str:
    if audio_filename is None:
        return ""
    try:
        return _parse_audio_recorded_at(audio_filename)
    except ValueError:
        return ""


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized_value = value.strip()
    return normalized_value or None


def _normalize_optional_datetime_text(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized_datetime = value
    else:
        normalized_value = value.strip()
        if not normalized_value:
            return None
        try:
            normalized_datetime = datetime.fromisoformat(normalized_value)
        except ValueError as exc:
            raise ValueError("audio_recorded_at must be an ISO-8601 string or datetime.") from exc

    if normalized_datetime.tzinfo is None:
        normalized_datetime = normalized_datetime.replace(tzinfo=UTC)
    else:
        normalized_datetime = normalized_datetime.astimezone(UTC)
    return normalized_datetime.isoformat(timespec="seconds")


def _coerce_nullable_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_optional_bool(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_nullable_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(int(value))
