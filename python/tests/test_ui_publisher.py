from __future__ import annotations

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
    build_transcript_message,
)
from modules.ui_publisher import UiPublisher


class UiPublisherTests(unittest.TestCase):
    def test_initial_snapshot_is_idle_and_empty(self) -> None:
        publisher = UiPublisher(transport=FakeTransport())

        self.assertEqual(
            build_snapshot_message(
                status="idle",
                remote_text="",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_session_started_updates_state_and_broadcasts_contract_message(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.publish_session_started(
            local_url="http://127.0.0.1:8000",
            lan_url="http://192.168.0.5:8000",
        )

        self.assertEqual(
            [
                build_session_started_message(
                    local_url="http://127.0.0.1:8000",
                    lan_url="http://192.168.0.5:8000",
                )
            ],
            transport.messages,
        )
        self.assertEqual(
            build_snapshot_message(
                status="listening",
                remote_text="",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_transcript_updates_remote_text_clears_reply_and_sets_processing(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)
        publisher.publish_reply_delta("Old partial reply")

        publisher.publish_transcript("Remote says hello")

        self.assertEqual(build_transcript_message("Remote says hello"), transport.messages[-1])
        self.assertEqual(
            build_snapshot_message(
                status="processing",
                remote_text="Remote says hello",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_reply_delta_appends_to_existing_reply_text(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.publish_reply_delta("Hello")
        publisher.publish_reply_delta(", world")

        self.assertEqual(
            [
                build_reply_delta_message("Hello"),
                build_reply_delta_message(", world"),
            ],
            transport.messages,
        )
        self.assertEqual(
            build_snapshot_message(
                status="processing",
                remote_text="",
                reply_text="Hello, world",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_reply_delta_preserves_spaces_between_streaming_chunks(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.publish_reply_delta("Lots")
        publisher.publish_reply_delta(" can")
        publisher.publish_reply_delta(" go")
        publisher.publish_reply_delta(" wrong")

        self.assertEqual(
            [
                build_reply_delta_message("Lots"),
                build_reply_delta_message(" can"),
                build_reply_delta_message(" go"),
                build_reply_delta_message(" wrong"),
            ],
            transport.messages,
        )
        self.assertEqual(
            build_snapshot_message(
                status="processing",
                remote_text="",
                reply_text="Lots can go wrong",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_reply_final_sets_final_reply_and_returns_to_listening(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)
        publisher.publish_transcript("Question")
        publisher.publish_reply_delta("Part")

        publisher.publish_reply_final("Final answer")

        self.assertEqual(build_reply_final_message("Final answer"), transport.messages[-1])
        self.assertEqual(
            build_snapshot_message(
                status="listening",
                remote_text="Question",
                reply_text="Final answer",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_processing_error_preserves_existing_text_and_sets_error_state(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)
        publisher.publish_transcript("Question")
        publisher.publish_reply_delta("Partial answer")

        publisher.publish_processing_error("Model timeout")

        self.assertEqual(build_processing_error_message("Model timeout"), transport.messages[-1])
        self.assertEqual(
            build_snapshot_message(
                status="error",
                remote_text="Question",
                reply_text="Partial answer",
                error="Model timeout",
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_session_stopped_updates_state_and_broadcasts_contract_message(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)
        publisher.publish_transcript("Question")
        publisher.publish_reply_final("Answer")

        publisher.publish_session_stopped()

        self.assertEqual(build_session_stopped_message(), transport.messages[-1])
        self.assertEqual(
            build_snapshot_message(
                status="stopped",
                remote_text="Question",
                reply_text="Answer",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_publish_task_snapshot_updates_task_state_and_broadcasts_task_message(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.publish_task_snapshot(
            status="running",
            file_count=2,
            artifacts=[
                {
                    "id": "artifact-1",
                    "kind": "image",
                    "label": "Screenshot 1",
                    "path": "/tmp/screenshot-1.png",
                    "content_type": "image/png",
                }
            ],
            latest_result={
                "name": "code_tests",
                "status": "passed",
                "summary": "2 passed",
                "response_text": "Full persisted result",
            },
            error=None,
        )

        self.assertEqual(
            build_task_snapshot_message(
                status="running",
                file_count=2,
                artifacts=[
                    {
                        "id": "artifact-1",
                        "kind": "image",
                        "label": "Screenshot 1",
                        "path": "/tmp/screenshot-1.png",
                        "content_type": "image/png",
                    }
                ],
                latest_result={
                    "name": "code_tests",
                    "status": "passed",
                    "summary": "2 passed",
                    "response_text": "Full persisted result",
                },
                error=None,
            ),
            transport.messages[-1],
        )
        self.assertEqual(
            build_task_snapshot_message(
                status="running",
                file_count=2,
                artifacts=[
                    {
                        "id": "artifact-1",
                        "kind": "image",
                        "label": "Screenshot 1",
                        "path": "/tmp/screenshot-1.png",
                        "content_type": "image/png",
                    }
                ],
                latest_result={
                    "name": "code_tests",
                    "status": "passed",
                    "summary": "2 passed",
                    "response_text": "Full persisted result",
                },
                error=None,
            ),
            publisher.task_snapshot_provider(),
        )

    def test_publish_task_snapshot_does_not_change_conversation_state(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)
        publisher.publish_transcript("Question")
        publisher.publish_reply_delta("Partial answer")

        publisher.publish_task_snapshot(
            status="ready",
            file_count=1,
            artifacts=[
                {
                    "id": "artifact-1",
                    "kind": "report",
                    "label": "Summary",
                    "path": "/tmp/summary.txt",
                }
            ],
            latest_result=None,
            error="Task failed",
        )

        snapshot = publisher.snapshot_provider()
        self.assertEqual("processing", snapshot["payload"]["status"])
        self.assertEqual("Question", snapshot["payload"]["remote_text"])
        self.assertEqual("Partial answer", snapshot["payload"]["reply_text"])
        self.assertIsNone(snapshot["payload"]["error"])
        self.assertEqual(
            build_task_snapshot_message(
                status="ready",
                file_count=1,
                artifacts=[
                    {
                        "id": "artifact-1",
                        "kind": "report",
                        "label": "Summary",
                        "path": "/tmp/summary.txt",
                    }
                ],
                latest_result=None,
                error="Task failed",
            ),
            publisher.task_snapshot_provider(),
        )

    def test_task_snapshot_provider_returns_task_state_in_consistent_shape(self) -> None:
        publisher = UiPublisher(transport=FakeTransport())

        publisher.publish_session_started(local_url="http://127.0.0.1:8000", lan_url=None)
        publisher.publish_task_snapshot(
            status="complete",
            file_count=3,
            artifacts=[
                {
                    "id": "artifact-1",
                    "kind": "image",
                    "label": "Screenshot 1",
                    "path": "/tmp/screenshot-1.png",
                }
            ],
            latest_result={
                "name": "code_tests",
                "status": "failed",
                "summary": "1 failed",
                "response_text": "AssertionError on test_login",
            },
            error=None,
        )

        snapshot = publisher.task_snapshot_provider()
        self.assertEqual("task_snapshot", snapshot["type"])
        self.assertEqual(
            {
                "status": "complete",
                "file_count": 3,
                "artifacts": [
                    {
                        "id": "artifact-1",
                        "kind": "image",
                        "label": "Screenshot 1",
                        "path": "/tmp/screenshot-1.png",
                    }
                ],
                "latest_result": {
                    "name": "code_tests",
                    "status": "failed",
                    "summary": "1 failed",
                    "response_text": "AssertionError on test_login",
                },
                "error": None,
            },
            snapshot["payload"],
        )

    def test_initial_task_snapshot_is_empty_and_not_idle(self) -> None:
        publisher = UiPublisher(transport=FakeTransport())

        self.assertEqual(
            build_task_snapshot_message(
                status="empty",
                file_count=0,
                artifacts=[],
                latest_result=None,
                error=None,
            ),
            publisher.task_snapshot_provider(),
        )

    def test_hydrate_task_snapshot_updates_state_without_broadcasting(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.hydrate_task_snapshot(
            status="empty",
            file_count=0,
            artifacts=[],
            latest_result={
                "name": "code_tests",
                "status": "passed",
                "summary": "Saved task solve from 1 screenshot(s).",
                "response_text": "Persisted task solution",
            },
            error=None,
        )

        self.assertEqual([], transport.messages)
        self.assertEqual(
            build_task_snapshot_message(
                status="empty",
                file_count=0,
                artifacts=[],
                latest_result={
                    "name": "code_tests",
                    "status": "passed",
                    "summary": "Saved task solve from 1 screenshot(s).",
                    "response_text": "Persisted task solution",
                },
                error=None,
            ),
            publisher.task_snapshot_provider(),
        )

    def test_transport_receives_task_message_without_affecting_other_publish_paths(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.publish_session_started(local_url="http://127.0.0.1:8000", lan_url=None)
        publisher.publish_task_snapshot(
            status="running",
            file_count=1,
            artifacts=[],
            latest_result=None,
            error=None,
        )
        publisher.publish_reply_final("Final")

        self.assertEqual("session_started", transport.messages[0]["type"])
        self.assertEqual("task_snapshot", transport.messages[1]["type"])
        self.assertEqual("reply_final", transport.messages[2]["type"])

    def test_snapshot_provider_reflects_latest_state_after_each_step(self) -> None:
        publisher = UiPublisher(transport=FakeTransport())

        publisher.publish_session_started(local_url="http://127.0.0.1:8000", lan_url=None)
        self.assertEqual("listening", publisher.snapshot_provider()["payload"]["status"])

        publisher.publish_transcript("Remote text")
        self.assertEqual("Remote text", publisher.snapshot_provider()["payload"]["remote_text"])

        publisher.publish_reply_delta("Partial")
        self.assertEqual("Partial", publisher.snapshot_provider()["payload"]["reply_text"])

        publisher.publish_processing_error("Oops")
        self.assertEqual("Oops", publisher.snapshot_provider()["payload"]["error"])

    def test_transport_receives_exactly_one_message_per_publish_call(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        publisher.publish_session_started(local_url="http://127.0.0.1:8000", lan_url=None)
        publisher.publish_transcript("Remote text")
        publisher.publish_reply_delta("Part")
        publisher.publish_reply_final("Final")
        publisher.publish_processing_error("Oops")
        publisher.publish_session_stopped()

        self.assertEqual(6, len(transport.messages))

    def test_transport_exception_is_propagated_and_state_remains_updated(self) -> None:
        transport = FailingTransport(RuntimeError("transport down"))
        publisher = UiPublisher(transport=transport)

        with self.assertRaisesRegex(RuntimeError, "transport down"):
            publisher.publish_transcript("Remote text")

        self.assertEqual(
            build_snapshot_message(
                status="processing",
                remote_text="Remote text",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_invalid_transcript_input_raises_contract_error_and_does_not_broadcast(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        with self.assertRaisesRegex(ValueError, "remote_text must not be empty"):
            publisher.publish_transcript("   ")

        self.assertEqual([], transport.messages)
        self.assertEqual(
            build_snapshot_message(
                status="idle",
                remote_text="",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_invalid_reply_delta_input_raises_contract_error_and_does_not_broadcast(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        with self.assertRaisesRegex(ValueError, "delta must not be empty"):
            publisher.publish_reply_delta("")

        self.assertEqual([], transport.messages)
        self.assertEqual(
            build_snapshot_message(
                status="idle",
                remote_text="",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_invalid_session_started_input_raises_contract_error_and_does_not_broadcast(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        with self.assertRaisesRegex(ValueError, "local_url must not be empty"):
            publisher.publish_session_started(local_url="   ", lan_url=None)

        self.assertEqual([], transport.messages)
        self.assertEqual(
            build_snapshot_message(
                status="idle",
                remote_text="",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )

    def test_invalid_task_input_raises_contract_error_and_does_not_broadcast(self) -> None:
        transport = FakeTransport()
        publisher = UiPublisher(transport=transport)

        with self.assertRaisesRegex(ValueError, "file_count must be a non-negative integer"):
            publisher.publish_task_snapshot(
                status="idle",
                file_count=-1,
                artifacts=[],
                latest_result=None,
                error=None,
            )

        self.assertEqual([], transport.messages)
        self.assertEqual(
            build_snapshot_message(
                status="idle",
                remote_text="",
                reply_text="",
                error=None,
            ),
            publisher.snapshot_provider(),
        )
        self.assertEqual(
            build_task_snapshot_message(
                status="empty",
                file_count=0,
                artifacts=[],
                latest_result=None,
                error=None,
            ),
            publisher.task_snapshot_provider(),
        )

    def test_task_transport_exception_is_propagated_and_task_state_remains_updated(self) -> None:
        transport = FailingTransport(RuntimeError("transport down"))
        publisher = UiPublisher(transport=transport)

        with self.assertRaisesRegex(RuntimeError, "transport down"):
            publisher.publish_task_snapshot(
                status="running",
                file_count=1,
                artifacts=[],
                latest_result=None,
                error=None,
            )

        self.assertEqual(
            build_task_snapshot_message(
                status="running",
                file_count=1,
                artifacts=[],
                latest_result=None,
                error=None,
            ),
            publisher.task_snapshot_provider(),
        )


class FakeTransport:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def broadcast(self, message: dict[str, object]) -> None:
        self.messages.append(message)


class FailingTransport:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def broadcast(self, message: dict[str, object]) -> None:
        raise self._error


if __name__ == "__main__":
    unittest.main()
