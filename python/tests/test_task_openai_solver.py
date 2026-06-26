from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_openai_solver import (
    OPENAI_TASK_SOLVER_PROMPT_ENV_VAR,
    TaskOpenAISolveResult,
    TaskOpenAISolverError,
    solve_task_from_screenshots,
)


class TaskOpenAISolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.dotenv_path = self.root_dir / ".env"
        self._original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._original_openai_model = os.environ.get("OPENAI_MODEL")
        self._original_task_solver_prompt = os.environ.get(OPENAI_TASK_SOLVER_PROMPT_ENV_VAR)
        self.default_task_solver_prompt = (
            "You are an experienced programmer. Recover the task from screenshots, explain the idea briefly, "
            "and write a TypeScript solution with Russian code comments."
        )
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_MODEL", None)
        os.environ[OPENAI_TASK_SOLVER_PROMPT_ENV_VAR] = self.default_task_solver_prompt

    def tearDown(self) -> None:
        if self._original_openai_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._original_openai_api_key

        if self._original_openai_model is None:
            os.environ.pop("OPENAI_MODEL", None)
        else:
            os.environ["OPENAI_MODEL"] = self._original_openai_model

        if self._original_task_solver_prompt is None:
            os.environ.pop(OPENAI_TASK_SOLVER_PROMPT_ENV_VAR, None)
        else:
            os.environ[OPENAI_TASK_SOLVER_PROMPT_ENV_VAR] = self._original_task_solver_prompt

        self.temp_dir.cleanup()

    def test_solve_task_from_screenshots_returns_predictable_result_for_path_inputs(self) -> None:
        first_bytes = b"fake-png-binary"
        second_bytes = b"fake-jpg-binary"
        first_path = self._create_screenshot("screen-1.png", first_bytes)
        second_path = self._create_screenshot("screen-2.jpg", second_bytes)
        client = FakeOpenAIClient(response_text="TypeScript solution here")

        result = solve_task_from_screenshots(
            [first_path, str(second_path)],
            client=client,
            model="gpt-5-mini",
        )

        self.assertEqual(
            TaskOpenAISolveResult(
                response_text="TypeScript solution here",
                source_file_count=2,
            ),
            result,
        )
        self.assertEqual("gpt-5-mini", client.recorded_request["model"])
        self.assertEqual(1, len(client.recorded_request["input"]))
        self.assertEqual("user", client.recorded_request["input"][0]["role"])
        content = client.recorded_request["input"][0]["content"]
        self.assertEqual(self.default_task_solver_prompt, content[0]["text"])
        self.assertEqual("input_text", content[0]["type"])
        self.assertEqual("input_image", content[1]["type"])
        self.assertEqual(
            f"data:image/png;base64,{base64.b64encode(first_bytes).decode('ascii')}",
            content[1]["image_url"],
        )
        self.assertEqual("input_image", content[2]["type"])
        self.assertEqual(
            f"data:image/jpeg;base64,{base64.b64encode(second_bytes).decode('ascii')}",
            content[2]["image_url"],
        )

    def test_solve_task_from_screenshots_accepts_artifact_like_inputs(self) -> None:
        screenshot_path = self._create_screenshot("artifact-shot.webp")
        client = FakeOpenAIClient(response_text="Recovered task statement")

        result = solve_task_from_screenshots(
            [{"path": str(screenshot_path)}],
            client=client,
            model="gpt-5-mini",
        )

        self.assertEqual("Recovered task statement", result.response_text)
        self.assertEqual(1, result.source_file_count)

    def test_solve_task_from_screenshots_accepts_object_with_path_attribute(self) -> None:
        screenshot_path = self._create_screenshot("artifact-object.png")
        client = FakeOpenAIClient(response_text="Recovered from object")

        result = solve_task_from_screenshots(
            [ArtifactLikeInput(path=str(screenshot_path))],
            client=client,
            model="gpt-5-mini",
        )

        self.assertEqual("Recovered from object", result.response_text)
        self.assertEqual(1, result.source_file_count)

    def test_solve_task_from_screenshots_uses_configured_prompt(self) -> None:
        screenshot_path = self._create_screenshot("fixed-prompt.png")
        client = FakeOpenAIClient(response_text="Answer")

        solve_task_from_screenshots(
            [screenshot_path],
            client=client,
            model="gpt-5-mini",
        )

        content = client.recorded_request["input"][0]["content"]
        self.assertEqual(self.default_task_solver_prompt, content[0]["text"])
        self.assertIn("TypeScript solution", content[0]["text"])

    def test_solve_task_from_screenshots_reads_prompt_from_dotenv_file(self) -> None:
        screenshot_path = self._create_screenshot("dotenv-prompt.png")
        dotenv_prompt = "Recover the problem from the screenshots and answer in English."
        os.environ.pop(OPENAI_TASK_SOLVER_PROMPT_ENV_VAR, None)
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\n"
            "OPENAI_MODEL=gpt-5-nano\n"
            f"{OPENAI_TASK_SOLVER_PROMPT_ENV_VAR}={dotenv_prompt}\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="Configured prompt reply")

        result = solve_task_from_screenshots(
            [screenshot_path],
            client=client,
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual("Configured prompt reply", result.response_text)
        self.assertEqual(dotenv_prompt, client.recorded_request["input"][0]["content"][0]["text"])

    def test_solve_task_from_screenshots_reads_model_from_dotenv_file(self) -> None:
        screenshot_path = self._create_screenshot("dotenv.png")
        self.dotenv_path.write_text(
            "OPENAI_API_KEY=test-key\nOPENAI_MODEL=gpt-5-nano\n",
            encoding="utf-8",
        )
        client = FakeOpenAIClient(response_text="Configured model reply")

        result = solve_task_from_screenshots(
            [screenshot_path],
            client=client,
            dotenv_path=self.dotenv_path,
        )

        self.assertEqual("Configured model reply", result.response_text)
        self.assertEqual("gpt-5-nano", client.recorded_request["model"])

    def test_solve_task_from_screenshots_rejects_empty_screenshot_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "Screenshots must not be empty"):
            solve_task_from_screenshots([], client=FakeOpenAIClient(response_text="unused"))

    def test_solve_task_from_screenshots_rejects_missing_screenshot_path(self) -> None:
        missing_path = self.root_dir / "missing.png"

        with self.assertRaisesRegex(ValueError, "Screenshot file does not exist"):
            solve_task_from_screenshots([missing_path], client=FakeOpenAIClient(response_text="unused"))

    def test_solve_task_from_screenshots_rejects_directory_path(self) -> None:
        directory_path = self.root_dir / "screenshots"
        directory_path.mkdir()

        with self.assertRaisesRegex(ValueError, "Screenshot file does not exist"):
            solve_task_from_screenshots([directory_path], client=FakeOpenAIClient(response_text="unused"))

    def test_solve_task_from_screenshots_rejects_unsupported_file_extension(self) -> None:
        notes_path = self.root_dir / "notes.txt"
        notes_path.write_text("not a screenshot", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "supported screenshot extension"):
            solve_task_from_screenshots([notes_path], client=FakeOpenAIClient(response_text="unused"))

    def test_solve_task_from_screenshots_rejects_input_without_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "Screenshot input must provide a path"):
            solve_task_from_screenshots([{}], client=FakeOpenAIClient(response_text="unused"))

    def test_solve_task_from_screenshots_maps_client_failure_to_service_error(self) -> None:
        screenshot_path = self._create_screenshot("client-failure.png")
        client = FakeOpenAIClient(raised_error=RuntimeError("upstream unavailable"))

        with self.assertRaisesRegex(TaskOpenAISolverError, "Task solve request failed: upstream unavailable"):
            solve_task_from_screenshots([screenshot_path], client=client, model="gpt-5-mini")

    def test_solve_task_from_screenshots_rejects_empty_text_response(self) -> None:
        screenshot_path = self._create_screenshot("empty-response.png")
        client = FakeOpenAIClient(response_text="   ")

        with self.assertRaisesRegex(TaskOpenAISolverError, "empty text response"):
            solve_task_from_screenshots([screenshot_path], client=client, model="gpt-5-mini")

    def test_solve_task_from_screenshots_requires_no_real_api_key_when_client_is_injected(self) -> None:
        screenshot_path = self._create_screenshot("no-api-key.png")
        client = FakeOpenAIClient(response_text="Mocked")

        result = solve_task_from_screenshots([screenshot_path], client=client, model="gpt-5-mini")

        self.assertEqual("Mocked", result.response_text)

    def test_solve_task_from_screenshots_rejects_missing_api_key_when_client_is_not_injected(self) -> None:
        screenshot_path = self._create_screenshot("missing-api-key.png")

        with self.assertRaisesRegex(ValueError, "OpenAI API key was not provided"):
            solve_task_from_screenshots([screenshot_path], model="gpt-5-mini", dotenv_path=self.dotenv_path)

    def _create_screenshot(self, filename: str, content: bytes = b"fake-image-bytes") -> Path:
        screenshot_path = self.root_dir / filename
        screenshot_path.write_bytes(content)
        return screenshot_path


@dataclass(frozen=True)
class ArtifactLikeInput:
    path: str


class FakeOpenAIClient:
    def __init__(self, *, response_text: str | None = None, raised_error: Exception | None = None) -> None:
        self.recorded_request: dict[str, object] | None = None
        self._response_text = response_text
        self._raised_error = raised_error
        self.responses = FakeResponsesAPI(self)


class FakeResponsesAPI:
    def __init__(self, owner: FakeOpenAIClient) -> None:
        self.owner = owner

    def create(self, *, model: str, input: list[dict[str, object]]) -> object:
        self.owner.recorded_request = {
            "model": model,
            "input": input,
        }
        if self.owner._raised_error is not None:
            raise self.owner._raised_error
        return FakeResponse(self.owner._response_text)


class FakeResponse:
    def __init__(self, output_text: str | None) -> None:
        self.output_text = output_text


if __name__ == "__main__":
    unittest.main()
