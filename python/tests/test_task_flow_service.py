from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.sqlite_conversation_store import create_code_test, get_latest_code_test
from modules.task_flow_service import (
    capture_task_screenshot,
    clear_current_task,
    publish_current_task_snapshot,
    solve_current_task,
)
from modules.task_openai_solver import TaskOpenAISolveResult, TaskOpenAISolverError
from modules.task_screenshot_service import create_task_screenshot as create_task_screenshot_boundary
from modules.ui_publisher import UiPublisher


class TaskFlowServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.task_dir = self.root_dir / "python" / "artifacts" / "process"
        self.capture_dir = self.root_dir / "capture"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root_dir / "database" / "test.sqlite3"
        self.transport = FakeTransport()
        self.publisher = UiPublisher(transport=self.transport)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_capture_task_screenshot_calls_screenshot_service_and_publishes_ready_snapshot(self) -> None:
        captured_path = self.capture_dir / "screen-1.png"
        captured_path.write_bytes(b"fake-png")

        self.publisher.publish_transcript("Question")
        self.publisher.publish_reply_delta("Draft")
        conversation_snapshot_before = self.publisher.snapshot_provider()

        with patch(
            "modules.task_flow_service.create_task_screenshot",
            wraps=create_task_screenshot_boundary,
        ) as create_screenshot:
            snapshot = capture_task_screenshot(
                screenshot_adapter=lambda: captured_path,
                publisher=self.publisher,
                task_artifact_dir=self.task_dir,
                database_path=self.database_path,
            )

        create_screenshot.assert_called_once()
        self.assertEqual("ready", snapshot.status)
        self.assertEqual(1, snapshot.file_count)
        self.assertEqual("task_snapshot", self.transport.messages[-1]["type"])
        self.assertEqual("ready", self.transport.messages[-1]["payload"]["status"])
        self.assertIsNone(self.transport.messages[-1]["payload"]["error"])
        self.assertIsNone(self.transport.messages[-1]["payload"]["latest_result"])
        self.assertEqual(conversation_snapshot_before, self.publisher.snapshot_provider())

    def test_capture_task_screenshot_publishes_error_state_when_capture_fails(self) -> None:
        self.publisher.publish_transcript("Question")
        self.publisher.publish_reply_delta("Draft")
        conversation_snapshot_before = self.publisher.snapshot_provider()

        with patch(
            "modules.task_flow_service.create_task_screenshot",
            side_effect=RuntimeError("capture backend exploded"),
        ) as create_screenshot:
            snapshot = capture_task_screenshot(
                screenshot_adapter=lambda: None,
                publisher=self.publisher,
                task_artifact_dir=self.task_dir,
                database_path=self.database_path,
            )

        create_screenshot.assert_called_once()
        self.assertEqual("error", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertIsNone(snapshot.latest_result)
        self.assertEqual("Task screenshot capture failed: capture backend exploded", snapshot.error)
        self.assertEqual("task_snapshot", self.transport.messages[-1]["type"])
        self.assertEqual("error", self.transport.messages[-1]["payload"]["status"])
        self.assertEqual(conversation_snapshot_before, self.publisher.snapshot_provider())

    def test_solve_current_task_does_not_call_openai_when_task_dir_is_empty_and_publishes_error_state(self) -> None:
        self.publisher.publish_transcript("Question")
        self.publisher.publish_reply_delta("Draft")
        conversation_snapshot_before = self.publisher.snapshot_provider()

        with patch("modules.task_flow_service.solve_task_from_screenshots") as solve_task:
            snapshot = solve_current_task(
                publisher=self.publisher,
                task_artifact_dir=self.task_dir,
                database_path=self.database_path,
            )

        solve_task.assert_not_called()
        self.assertEqual("error", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertEqual("No task screenshots are available to solve.", snapshot.error)
        self.assertIsNone(get_latest_code_test(database_path=self.database_path))
        self.assertEqual(self.transport.messages[-1], self.publisher.task_snapshot_provider())
        self.assertEqual(conversation_snapshot_before, self.publisher.snapshot_provider())

    def test_solve_current_task_success_path_preserves_required_order(self) -> None:
        first_path = self._create_task_artifact("screen-1.png", b"first")
        second_path = self._create_task_artifact("screen-2.jpg", b"second")
        order: list[str] = []

        def fake_solver(screenshots: list[Path], **_: object) -> TaskOpenAISolveResult:
            order.append("solve")
            self.assertEqual([first_path, second_path], screenshots)
            return TaskOpenAISolveResult(
                response_text="TypeScript solution",
                source_file_count=2,
            )

        def fake_create_code_test(response_text: str, source_file_count: int, **_: object) -> object:
            order.append("save")
            self.assertEqual("TypeScript solution", response_text)
            self.assertEqual(2, source_file_count)
            return create_code_test(
                response_text,
                source_file_count,
                database_path=self.database_path,
            )

        def fake_clear_task_artifacts(task_artifact_dir: Path) -> Path:
            order.append("clear")
            for artifact_path in sorted(task_artifact_dir.iterdir()):
                artifact_path.unlink()
            return task_artifact_dir

        with patch(
            "modules.task_flow_service.solve_task_from_screenshots",
            side_effect=fake_solver,
        ), patch(
            "modules.task_flow_service.create_code_test",
            side_effect=fake_create_code_test,
        ), patch(
            "modules.task_flow_service.clear_task_artifacts",
            side_effect=fake_clear_task_artifacts,
        ):
            snapshot = solve_current_task(
                publisher=self.publisher,
                task_artifact_dir=self.task_dir,
                database_path=self.database_path,
            )

        self.assertEqual(["solve", "save", "clear"], order)
        latest_code_test = get_latest_code_test(database_path=self.database_path)
        self.assertIsNotNone(latest_code_test)
        self.assertEqual("TypeScript solution", latest_code_test.response_text)
        self.assertEqual("complete", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertEqual(
            {
                "name": "code_tests",
                "status": "passed",
                "summary": "Saved task solve from 2 screenshot(s).",
                "response_text": "TypeScript solution",
            },
            snapshot.latest_result,
        )
        self.assertIsNone(snapshot.error)
        self.assertFalse(any(self.task_dir.iterdir()))
        self.assertEqual(self.transport.messages[-1], self.publisher.task_snapshot_provider())

    def test_solve_current_task_success_cleanup_removes_all_task_owned_contents(self) -> None:
        self._create_task_artifact("screen-1.png", b"first")
        notes_path = self._create_task_file("notes.txt", b"remove")
        metadata_file = self._create_task_file("metadata/state.json", b"{}")

        snapshot = solve_current_task(
            publisher=self.publisher,
            task_artifact_dir=self.task_dir,
            database_path=self.database_path,
            client=FakeOpenAIClient(response_text="Persisted task solution"),
        )

        self.assertEqual("complete", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertFalse(notes_path.exists())
        self.assertFalse(metadata_file.exists())
        self.assertEqual([], list(self.task_dir.iterdir()))

    def test_solve_current_task_success_snapshot_matches_persisted_code_test_result(self) -> None:
        self._create_task_artifact("screen-1.png", b"first")

        snapshot = solve_current_task(
            publisher=self.publisher,
            task_artifact_dir=self.task_dir,
            database_path=self.database_path,
            client=FakeOpenAIClient(response_text="Persisted task solution"),
        )

        latest_code_test = get_latest_code_test(database_path=self.database_path)

        self.assertIsNotNone(latest_code_test)
        assert latest_code_test is not None
        self.assertEqual("complete", snapshot.status)
        self.assertEqual("code_tests", snapshot.latest_result["name"])
        self.assertEqual("Persisted task solution", latest_code_test.response_text)
        self.assertEqual(
            latest_code_test.response_text,
            snapshot.latest_result["response_text"],
        )
        self.assertEqual(
            self.transport.messages[-1]["payload"]["latest_result"]["response_text"],
            latest_code_test.response_text,
        )

    def test_solve_current_task_openai_error_does_not_write_code_test_or_cleanup(self) -> None:
        self._create_task_artifact("screen-1.png", b"first")

        with patch(
            "modules.task_flow_service.solve_task_from_screenshots",
            side_effect=TaskOpenAISolverError("upstream unavailable"),
        ), patch("modules.task_flow_service.create_code_test") as create_code_test_mock, patch(
            "modules.task_flow_service.clear_task_artifacts"
        ) as clear_task_artifacts_mock:
            snapshot = solve_current_task(
                publisher=self.publisher,
                task_artifact_dir=self.task_dir,
                database_path=self.database_path,
            )

        create_code_test_mock.assert_not_called()
        clear_task_artifacts_mock.assert_not_called()
        self.assertEqual("error", snapshot.status)
        self.assertEqual("Task solve failed: upstream unavailable", snapshot.error)
        self.assertIsNone(get_latest_code_test(database_path=self.database_path))
        self.assertEqual(1, len(list(self.task_dir.iterdir())))

    def test_solve_current_task_save_error_does_not_cleanup_and_publishes_error_state(self) -> None:
        self._create_task_artifact("screen-1.png", b"first")

        with patch(
            "modules.task_flow_service.solve_task_from_screenshots",
            return_value=TaskOpenAISolveResult(
                response_text="TypeScript solution",
                source_file_count=1,
            ),
        ), patch(
            "modules.task_flow_service.create_code_test",
            side_effect=RuntimeError("database locked"),
        ), patch("modules.task_flow_service.clear_task_artifacts") as clear_task_artifacts_mock:
            snapshot = solve_current_task(
                publisher=self.publisher,
                task_artifact_dir=self.task_dir,
                database_path=self.database_path,
            )

        clear_task_artifacts_mock.assert_not_called()
        self.assertEqual("error", snapshot.status)
        self.assertEqual("Task result persistence failed: database locked", snapshot.error)
        self.assertIsNone(get_latest_code_test(database_path=self.database_path))
        self.assertEqual(1, len(list(self.task_dir.iterdir())))

    def test_clear_current_task_removes_artifacts_without_deleting_code_test_history(self) -> None:
        self._create_task_artifact("screen-1.png", b"first")
        expected_code_test = create_code_test(
            "Persisted solution",
            1,
            database_path=self.database_path,
        )

        self.publisher.publish_transcript("Question")
        self.publisher.publish_reply_delta("Draft")
        conversation_snapshot_before = self.publisher.snapshot_provider()

        snapshot = clear_current_task(
            publisher=self.publisher,
            task_artifact_dir=self.task_dir,
            database_path=self.database_path,
        )

        self.assertEqual("empty", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertEqual(
            {
                "name": "code_tests",
                "status": "passed",
                "summary": "Saved task solve from 1 screenshot(s).",
                "response_text": "Persisted solution",
            },
            snapshot.latest_result,
        )
        self.assertEqual(expected_code_test, get_latest_code_test(database_path=self.database_path))
        self.assertFalse(any(self.task_dir.iterdir()))
        self.assertEqual("empty", self.transport.messages[-1]["payload"]["status"])
        self.assertEqual(
            "Persisted solution",
            self.transport.messages[-1]["payload"]["latest_result"]["response_text"],
        )
        self.assertEqual(conversation_snapshot_before, self.publisher.snapshot_provider())

    def test_clear_current_task_removes_all_task_owned_contents(self) -> None:
        self._create_task_artifact("screen-1.png", b"first")
        notes_path = self._create_task_file("notes.txt", b"remove")
        metadata_file = self._create_task_file("metadata/state.json", b"{}")

        snapshot = clear_current_task(
            publisher=self.publisher,
            task_artifact_dir=self.task_dir,
            database_path=self.database_path,
        )

        self.assertEqual("empty", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertFalse(notes_path.exists())
        self.assertFalse(metadata_file.exists())
        self.assertEqual([], list(self.task_dir.iterdir()))

    def test_clear_current_task_is_safe_noop_when_task_dir_is_already_empty(self) -> None:
        snapshot = clear_current_task(
            publisher=self.publisher,
            task_artifact_dir=self.task_dir,
            database_path=self.database_path,
        )

        self.assertEqual("empty", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertEqual(self.transport.messages[-1], self.publisher.task_snapshot_provider())

    def test_publish_current_task_snapshot_restores_latest_result_from_persisted_code_test(self) -> None:
        create_code_test(
            "Persisted solution",
            2,
            database_path=self.database_path,
        )

        snapshot = publish_current_task_snapshot(
            publisher=self.publisher,
            task_artifact_dir=self.task_dir,
            database_path=self.database_path,
            latest_result=None,
            error=None,
        )

        self.assertEqual("empty", snapshot.status)
        self.assertEqual(0, snapshot.file_count)
        self.assertEqual([], snapshot.artifacts)
        self.assertEqual(
            {
                "name": "code_tests",
                "status": "passed",
                "summary": "Saved task solve from 2 screenshot(s).",
                "response_text": "Persisted solution",
            },
            snapshot.latest_result,
        )
        self.assertEqual(
            snapshot.latest_result,
            self.transport.messages[-1]["payload"]["latest_result"],
        )

    def _create_task_artifact(self, filename: str, content: bytes) -> Path:
        self.task_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.task_dir / filename
        artifact_path.write_bytes(content)
        return artifact_path.resolve()

    def _create_task_file(self, relative_path: str, content: bytes) -> Path:
        file_path = self.task_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return file_path.resolve()


class FakeTransport:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def broadcast(self, message: dict[str, object]) -> None:
        self.messages.append(message)


class FakeOpenAIClient:
    def __init__(self, *, response_text: str) -> None:
        self._response_text = response_text
        self.responses = self._ResponsesApi(owner=self)

    class _ResponsesApi:
        def __init__(self, *, owner: "FakeOpenAIClient") -> None:
            self._owner = owner

        def create(self, **_: object):
            return type(
                "FakeResponse",
                (),
                {
                    "output_text": self._owner._response_text,
                },
            )()


if __name__ == "__main__":
    unittest.main()
