from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.browser_ui_contract import (
    build_processing_error_message,
    build_reply_delta_message,
    build_reply_final_message,
    build_session_started_message,
    build_session_stopped_message,
    build_snapshot_message,
    build_task_snapshot_message,
    build_task_state_payload,
    build_transcript_message,
)


class BrowserUiContractTests(unittest.TestCase):
    def test_build_snapshot_message_returns_expected_shape(self) -> None:
        message = build_snapshot_message(
            status="listening",
            remote_text="Hello there",
            reply_text="Hi back",
            error=None,
        )

        self.assertEqual(
            {
                "type": "snapshot",
                "payload": {
                    "status": "listening",
                    "remote_text": "Hello there",
                    "reply_text": "Hi back",
                    "error": None,
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="snapshot")

    def test_build_session_started_message_returns_expected_shape(self) -> None:
        message = build_session_started_message(
            local_url="http://127.0.0.1:8000",
            lan_url="http://192.168.1.10:8000",
        )

        self.assertEqual(
            {
                "type": "session_started",
                "payload": {
                    "status": "listening",
                    "local_url": "http://127.0.0.1:8000",
                    "lan_url": "http://192.168.1.10:8000",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="session_started")

    def test_build_transcript_message_returns_expected_shape(self) -> None:
        message = build_transcript_message("Remote says hello")

        self.assertEqual(
            {
                "type": "transcript",
                "payload": {
                    "remote_text": "Remote says hello",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="transcript")

    def test_build_reply_delta_message_returns_expected_shape(self) -> None:
        message = build_reply_delta_message("partial")

        self.assertEqual(
            {
                "type": "reply_delta",
                "payload": {
                    "delta": "partial",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="reply_delta")

    def test_build_reply_delta_message_preserves_leading_space(self) -> None:
        message = build_reply_delta_message(" can")

        self.assertEqual(" can", message["payload"]["delta"])
        self._assert_contract_shape(message, expected_type="reply_delta")

    def test_build_reply_delta_message_preserves_trailing_space(self) -> None:
        message = build_reply_delta_message("wrong ")

        self.assertEqual("wrong ", message["payload"]["delta"])
        self._assert_contract_shape(message, expected_type="reply_delta")

    def test_build_reply_delta_message_preserves_surrounding_spaces(self) -> None:
        message = build_reply_delta_message(" - ")

        self.assertEqual(" - ", message["payload"]["delta"])
        self._assert_contract_shape(message, expected_type="reply_delta")

    def test_build_reply_final_message_returns_expected_shape(self) -> None:
        message = build_reply_final_message("Full reply")

        self.assertEqual(
            {
                "type": "reply_final",
                "payload": {
                    "reply_text": "Full reply",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="reply_final")

    def test_build_processing_error_message_returns_expected_shape(self) -> None:
        message = build_processing_error_message("Something failed")

        self.assertEqual(
            {
                "type": "processing_error",
                "payload": {
                    "message": "Something failed",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="processing_error")

    def test_build_session_stopped_message_returns_expected_shape(self) -> None:
        message = build_session_stopped_message()

        self.assertEqual(
            {
                "type": "session_stopped",
                "payload": {
                    "status": "stopped",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="session_stopped")

    def test_build_task_snapshot_message_returns_expected_shape(self) -> None:
        message = build_task_snapshot_message(
            status="running",
            file_count=3,
            artifacts=[
                {
                    "id": "artifact-1",
                    "kind": "image",
                    "label": "Screenshot 1",
                    "path": "/tmp/screenshot-1.png",
                    "content_type": "image/png",
                },
                {
                    "id": "artifact-2",
                    "kind": "report",
                    "label": "Test report",
                    "path": "/tmp/report.txt",
                },
            ],
            latest_result={
                "name": "code_tests",
                "status": "failed",
                "summary": "1 failed, 4 passed",
            },
            error="Lint failed",
        )

        self.assertEqual(
            {
                "type": "task_snapshot",
                "payload": {
                    "status": "running",
                    "file_count": 3,
                    "artifacts": [
                        {
                            "id": "artifact-1",
                            "kind": "image",
                            "label": "Screenshot 1",
                            "path": "/tmp/screenshot-1.png",
                            "content_type": "image/png",
                        },
                        {
                            "id": "artifact-2",
                            "kind": "report",
                            "label": "Test report",
                            "path": "/tmp/report.txt",
                        },
                    ],
                    "latest_result": {
                        "name": "code_tests",
                        "status": "failed",
                        "summary": "1 failed, 4 passed",
                    },
                    "error": "Lint failed",
                },
            },
            message,
        )
        self._assert_contract_shape(message, expected_type="task_snapshot")

    def test_build_task_state_payload_returns_json_compatible_shape(self) -> None:
        payload = build_task_state_payload(
            status="idle",
            file_count=0,
            artifacts=[],
            latest_result=None,
            error=None,
        )

        self.assertEqual(
            {
                "status": "idle",
                "file_count": 0,
                "artifacts": [],
                "latest_result": None,
                "error": None,
            },
            payload,
        )
        json.dumps(payload)

    def test_build_transcript_message_rejects_empty_remote_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "remote_text must not be empty"):
            build_transcript_message("   ")

    def test_build_reply_delta_message_rejects_empty_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "delta must not be empty"):
            build_reply_delta_message("")

    def test_build_reply_final_message_rejects_empty_reply_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "reply_text must not be empty"):
            build_reply_final_message("   ")

    def test_build_processing_error_message_rejects_empty_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "message must not be empty"):
            build_processing_error_message("   ")

    def test_build_session_started_message_rejects_empty_local_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_url must not be empty"):
            build_session_started_message(local_url="   ", lan_url=None)

    def test_build_snapshot_message_rejects_empty_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "status must not be empty"):
            build_snapshot_message(
                status="   ",
                remote_text="Hello",
                reply_text="World",
                error=None,
            )

    def test_build_transcript_message_rejects_non_string_remote_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "remote_text must be a string"):
            build_transcript_message(123)  # type: ignore[arg-type]

    def test_build_reply_delta_message_rejects_non_string_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "delta must be a string"):
            build_reply_delta_message(123)  # type: ignore[arg-type]

    def test_build_reply_final_message_rejects_non_string_reply_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "reply_text must be a string"):
            build_reply_final_message(123)  # type: ignore[arg-type]

    def test_build_processing_error_message_rejects_non_string_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "message must be a string"):
            build_processing_error_message(123)  # type: ignore[arg-type]

    def test_build_session_started_message_rejects_non_string_local_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_url must be a string"):
            build_session_started_message(local_url=123, lan_url=None)  # type: ignore[arg-type]

    def test_build_session_started_message_rejects_non_string_lan_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "lan_url must be a string or None"):
            build_session_started_message(
                local_url="http://127.0.0.1:8000",
                lan_url=123,  # type: ignore[arg-type]
            )

    def test_build_snapshot_message_rejects_non_string_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "status must be a string"):
            build_snapshot_message(
                status=123,  # type: ignore[arg-type]
                remote_text="Hello",
                reply_text="World",
                error=None,
            )

    def test_build_snapshot_message_rejects_non_string_remote_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "remote_text must be a string"):
            build_snapshot_message(
                status="listening",
                remote_text=123,  # type: ignore[arg-type]
                reply_text="World",
                error=None,
            )

    def test_build_snapshot_message_rejects_non_string_reply_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "reply_text must be a string"):
            build_snapshot_message(
                status="listening",
                remote_text="Hello",
                reply_text=123,  # type: ignore[arg-type]
                error=None,
            )

    def test_build_snapshot_message_rejects_non_string_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "error must be a string or None"):
            build_snapshot_message(
                status="listening",
                remote_text="Hello",
                reply_text="World",
                error=123,  # type: ignore[arg-type]
            )

    def test_build_task_snapshot_message_rejects_empty_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "status must not be empty"):
            build_task_snapshot_message(
                status="   ",
                file_count=0,
                artifacts=[],
                latest_result=None,
                error=None,
            )

    def test_build_task_snapshot_message_rejects_negative_file_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "file_count must be a non-negative integer"):
            build_task_snapshot_message(
                status="idle",
                file_count=-1,
                artifacts=[],
                latest_result=None,
                error=None,
            )

    def test_build_task_snapshot_message_rejects_non_list_artifacts(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifacts must be a list"):
            build_task_snapshot_message(
                status="idle",
                file_count=0,
                artifacts="artifact-1",  # type: ignore[arg-type]
                latest_result=None,
                error=None,
            )

    def test_build_task_snapshot_message_rejects_artifact_with_missing_required_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact.id must not be empty"):
            build_task_snapshot_message(
                status="idle",
                file_count=1,
                artifacts=[
                    {
                        "id": "   ",
                        "kind": "image",
                        "label": "Screenshot 1",
                        "path": "/tmp/screenshot-1.png",
                    }
                ],
                latest_result=None,
                error=None,
            )

    def test_build_task_snapshot_message_rejects_invalid_latest_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "latest_result.name must not be empty"):
            build_task_snapshot_message(
                status="idle",
                file_count=0,
                artifacts=[],
                latest_result={
                    "name": "   ",
                    "status": "passed",
                    "summary": None,
                },
                error=None,
            )

    def test_build_task_snapshot_message_accepts_latest_result_response_text(self) -> None:
        message = build_task_snapshot_message(
            status="complete",
            file_count=0,
            artifacts=[],
            latest_result={
                "name": "code_tests",
                "status": "passed",
                "summary": "Saved task solve from 1 screenshot(s).",
                "response_text": "function solveTask() { return true; }",
            },
            error=None,
        )

        self.assertEqual(
            "function solveTask() { return true; }",
            message["payload"]["latest_result"]["response_text"],
        )
        self._assert_contract_shape(message, expected_type="task_snapshot")

    def test_build_task_snapshot_message_rejects_non_string_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "error must be a string or None"):
            build_task_snapshot_message(
                status="idle",
                file_count=0,
                artifacts=[],
                latest_result=None,
                error=123,  # type: ignore[arg-type]
            )

    def _assert_contract_shape(self, message: dict[str, object], *, expected_type: str) -> None:
        self.assertEqual({"type", "payload"}, set(message.keys()))
        self.assertEqual(expected_type, message["type"])
        self.assertIsInstance(message["payload"], dict)
        json.dumps(message)


if __name__ == "__main__":
    unittest.main()
