from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
APP_JS_PATH = PYTHON_ROOT / "web" / "browser_ui" / "app.js"


class BrowserUiFrontendLogicTests(unittest.TestCase):
    def test_task_ui_is_hidden_and_task_messages_are_ignored_when_feature_is_disabled(self) -> None:
        self._run_node_assertions(
            """
            assert.equal(document.getElementById("tab-button-task"), null);
            assert.equal(document.getElementById("tab-panel-task"), null);
            assert.equal(document.getElementById("task-status-value"), null);
            assert.equal(document.getElementById("task-artifacts-list"), null);

            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "error",
                  file_count: 3,
                  artifacts: [
                    {
                      id: "screen-1.png",
                      kind: "image",
                      label: "screen-1.png",
                      path: "/tmp/screen-1.png",
                      content_type: "image/png"
                    }
                  ],
                  latest_result: {
                    name: "code_tests",
                    status: "failed",
                    summary: "should stay hidden",
                    response_text: "stale task result"
                  },
                  error: "Task solve failed"
                }
              })
            });

            assert.equal(elements["conversation-log"].innerHTML.includes("screen-1.png"), false);
            assert.equal(elements["conversation-log"].innerHTML.includes("stale task result"), false);
            assert.equal(elements["error-text"].textContent, "");
            """
            ,
            task_feature_enabled=False,
        )

    def test_tab_switching_only_changes_visibility_and_preserves_rendered_content(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("open");
            socket.emit("message", {
              data: JSON.stringify({
                type: "snapshot",
                payload: {
                  status: "listening",
                  remote_text: "How would you solve this?",
                  reply_text: "I would start by clarifying requirements.",
                  error: null
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "ready",
                  file_count: 2,
                  artifacts: [
                    {
                      id: "screen-1.png",
                      kind: "image",
                      label: "screen-1.png",
                      path: "/tmp/screen-1.png",
                      content_type: "image/png"
                    }
                  ],
                  latest_result: {
                    name: "code_tests",
                    status: "passed",
                    summary: "Saved task solve from 2 screenshot(s)."
                  },
                  error: null
                }
              })
            });

            const convPanel = elements["tab-panel-convs"];
            const taskPanel = elements["tab-panel-task"];
            const convLog = elements["conversation-log"];
            const taskList = elements["task-artifacts-list"];
            const convHtmlBefore = convLog.innerHTML;
            const taskHtmlBefore = taskList.innerHTML;

            assert.equal(convPanel.classList.contains("tab-panel-hidden"), false);
            assert.equal(taskPanel.classList.contains("tab-panel-hidden"), true);

            elements["tab-button-task"].click();

            assert.equal(convPanel.classList.contains("tab-panel-hidden"), true);
            assert.equal(taskPanel.classList.contains("tab-panel-hidden"), false);
            assert.equal(convLog.innerHTML, convHtmlBefore);
            assert.equal(taskList.innerHTML, taskHtmlBefore);

            elements["tab-button-convs"].click();

            assert.equal(convPanel.classList.contains("tab-panel-hidden"), false);
            assert.equal(taskPanel.classList.contains("tab-panel-hidden"), true);
            assert.equal(convLog.innerHTML, convHtmlBefore);
            assert.equal(taskList.innerHTML, taskHtmlBefore);
            """
        )

    def test_task_snapshot_updates_task_panel_without_touching_conversation_content(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("open");
            socket.emit("message", {
              data: JSON.stringify({
                type: "snapshot",
                payload: {
                  status: "listening",
                  remote_text: "Existing question",
                  reply_text: "Existing answer",
                  error: null
                }
              })
            });

            const conversationHtmlBefore = elements["conversation-log"].innerHTML;

            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "error",
                  file_count: 3,
                  artifacts: [
                    {
                      id: "screen-1.png",
                      kind: "image",
                      label: "screen-1.png",
                      path: "/tmp/screen-1.png",
                      content_type: "image/png"
                    },
                    {
                      id: "notes.txt",
                      kind: "report",
                      label: "notes.txt",
                      path: "/tmp/notes.txt"
                    }
                  ],
                  latest_result: {
                    name: "code_tests",
                    status: "failed",
                    summary: "1 failed, 4 passed",
                    response_text: "Traceback: expected 5 assertions"
                  },
                  error: "Task solve failed"
                }
              })
            });

            assert.equal(elements["task-status-value"].textContent, "error");
            assert.equal(elements["task-file-count-value"].textContent, "3");
            assert.match(elements["task-artifacts-list"].innerHTML, /screen-1\\.png/);
            assert.match(elements["task-artifacts-list"].innerHTML, /notes\\.txt/);
            assert.match(elements["task-preview-list"].innerHTML, /<img/);
            assert.match(elements["task-result-text"].innerHTML, /code_tests/);
            assert.match(elements["task-result-text"].innerHTML, /1 failed, 4 passed/);
            assert.match(elements["task-result-text"].innerHTML, /Traceback: expected 5 assertions/);
            assert.equal(elements["task-error-text"].textContent, "Task solve failed");
            assert.equal(elements["conversation-log"].innerHTML, conversationHtmlBefore);
            """
            ,
            task_feature_enabled=True,
        )

    def test_conversation_updates_without_auto_scroll_when_task_tab_is_active(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("open");
            socket.emit("message", {
              data: JSON.stringify({
                type: "snapshot",
                payload: {
                  status: "listening",
                  remote_text: "Initial question",
                  reply_text: "Initial answer",
                  error: null
                }
              })
            });

            const scrollCallsBeforeTaskTab = windowObject.scrollCalls.length;
            elements["tab-button-task"].click();

            socket.emit("message", {
              data: JSON.stringify({
                type: "transcript",
                payload: {
                  remote_text: "Follow-up question"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_delta",
                payload: {
                  delta: "Follow-up"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_final",
                payload: {
                  reply_text: "Follow-up answer complete"
                }
              })
            });

            assert.equal(windowObject.scrollCalls.length, scrollCallsBeforeTaskTab);
            assert.match(elements["conversation-log"].innerHTML, /Initial question/);
            assert.match(elements["conversation-log"].innerHTML, /Initial answer/);
            assert.match(elements["conversation-log"].innerHTML, /Follow-up question/);
            assert.match(elements["conversation-log"].innerHTML, /Follow-up answer complete/);
            assert.equal(elements["tab-panel-convs"].classList.contains("tab-panel-hidden"), true);
            assert.equal(elements["tab-panel-task"].classList.contains("tab-panel-hidden"), false);
            """
            ,
            task_feature_enabled=True,
        )

    def test_conversation_auto_scroll_resumes_after_switching_back_to_convs_tab(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("open");
            socket.emit("message", {
              data: JSON.stringify({
                type: "snapshot",
                payload: {
                  status: "listening",
                  remote_text: "Initial question",
                  reply_text: "Initial answer",
                  error: null
                }
              })
            });

            elements["tab-button-task"].click();
            const scrollCallsBeforeHiddenUpdate = windowObject.scrollCalls.length;

            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_delta",
                payload: {
                  delta: " hidden update"
                }
              })
            });

            assert.equal(windowObject.scrollCalls.length, scrollCallsBeforeHiddenUpdate);

            elements["tab-button-convs"].click();
            const scrollCallsBeforeVisibleUpdate = windowObject.scrollCalls.length;

            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_final",
                payload: {
                  reply_text: "Visible answer after return"
                }
              })
            });

            assert.equal(windowObject.scrollCalls.length, scrollCallsBeforeVisibleUpdate + 1);
            assert.match(elements["conversation-log"].innerHTML, /Visible answer after return/);
            assert.equal(elements["tab-panel-convs"].classList.contains("tab-panel-hidden"), false);
            assert.equal(elements["tab-panel-task"].classList.contains("tab-panel-hidden"), true);
            """
            ,
            task_feature_enabled=True,
        )

    def test_task_panel_defaults_to_empty_state_before_first_snapshot(self) -> None:
        self._run_node_assertions(
            """
            assert.equal(elements["task-status-value"].textContent, "empty");
            assert.equal(elements["task-file-count-value"].textContent, "0");
            assert.equal(elements["task-result-text"].innerHTML, "-");
            assert.equal(elements["task-error-text"].textContent, "-");
            """,
            task_feature_enabled=True,
        )

    def test_task_result_renders_fenced_code_blocks_as_html_code(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "ready",
                  file_count: 1,
                  artifacts: [],
                  latest_result: {
                    name: "code_tests",
                    status: "passed",
                    summary: "Saved task solve from 1 screenshot.",
                    response_text: "Solution:\\n\\n```typescript\\nconst answer = 42;\\nconsole.log(answer);\\n```"
                  },
                  error: null
                }
              })
            });

            assert.match(elements["task-result-text"].innerHTML, /<pre class="task-code-block"><code class="language-typescript">/);
            assert.match(elements["task-result-text"].innerHTML, /const answer = 42;/);
            assert.match(elements["task-result-text"].innerHTML, /console\\.log\\(answer\\);/);
            assert.doesNotMatch(elements["task-result-text"].innerHTML, /```/);
            """
            ,
            task_feature_enabled=True,
        )

    def test_task_result_escapes_html_in_markdown_content(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "ready",
                  file_count: 1,
                  artifacts: [],
                  latest_result: {
                    name: "code_tests",
                    status: "passed",
                    summary: "Saved task solve from 1 screenshot.",
                    response_text: "<b>bold?</b>\\n\\n```html\\n<script>alert(1)</script>\\n```"
                  },
                  error: null
                }
              })
            });

            assert.match(elements["task-result-text"].innerHTML, /&lt;b&gt;bold\\?&lt;\\/b&gt;/);
            assert.match(elements["task-result-text"].innerHTML, /&lt;script&gt;alert\\(1\\)&lt;\\/script&gt;/);
            assert.doesNotMatch(elements["task-result-text"].innerHTML, /<script>/);
            """
            ,
            task_feature_enabled=True,
        )

    def test_task_action_buttons_call_matching_api_routes(self) -> None:
        self._run_node_assertions(
            """
            elements["tab-button-task"].click();

            elements["task-action-screenshot"].click();
            await new Promise((resolve) => setTimeout(resolve, 0));

            elements["task-action-send"].click();
            await new Promise((resolve) => setTimeout(resolve, 0));

            elements["task-action-clear"].click();
            await new Promise((resolve) => setTimeout(resolve, 0));

            assert.deepEqual(fetchCalls.map((call) => [call.url, call.options.method]), [
              ["/api/task/screenshot", "POST"],
              ["/api/task/send", "POST"],
              ["/api/task/clear", "POST"]
            ]);
            assert.equal(elements["task-action-screenshot"].disabled, false);
            assert.equal(elements["task-action-send"].disabled, false);
            assert.equal(elements["task-action-clear"].disabled, false);
            """
        )

    def test_send_action_clears_latest_result_before_request_completes(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "ready",
                  file_count: 1,
                  artifacts: [],
                  latest_result: {
                    name: "code_tests",
                    status: "passed",
                    summary: "Saved task solve from 1 screenshot.",
                    response_text: "Previous result"
                  },
                  error: null
                }
              })
            });

            assert.match(elements["task-result-text"].innerHTML, /Previous result/);

            elements["tab-button-task"].click();
            elements["task-action-send"].click();

            assert.equal(fetchCalls.length, 1);
            assert.equal(fetchCalls[0].url, "/api/task/send");
            assert.equal(fetchCalls[0].options.method, "POST");
            assert.equal(elements["task-result-text"].innerHTML, "-");

            await new Promise((resolve) => setTimeout(resolve, 0));
            """
            ,
            task_feature_enabled=True,
        )

    def test_send_action_includes_task_prompt_json_body_when_textarea_is_filled(self) -> None:
        self._run_node_assertions(
            """
            elements["task-prompt-input"].value = "Solve in Kotlin.";
            elements["tab-button-task"].click();
            elements["task-action-send"].click();

            assert.equal(fetchCalls.length, 1);
            assert.equal(fetchCalls[0].url, "/api/task/send");
            assert.equal(fetchCalls[0].options.method, "POST");
            assert.equal(fetchCalls[0].options.headers["Content-Type"], "application/json");
            assert.equal(fetchCalls[0].options.body, JSON.stringify({ task_prompt: "Solve in Kotlin." }));

            await new Promise((resolve) => setTimeout(resolve, 0));
            """,
            task_feature_enabled=True,
        )

    def test_send_action_omits_request_body_when_task_prompt_textarea_is_blank(self) -> None:
        self._run_node_assertions(
            """
            elements["task-prompt-input"].value = "   ";
            elements["tab-button-task"].click();
            elements["task-action-send"].click();

            assert.equal(fetchCalls.length, 1);
            assert.equal(fetchCalls[0].url, "/api/task/send");
            assert.equal(fetchCalls[0].options.method, "POST");
            assert.equal(fetchCalls[0].options.body, null);

            await new Promise((resolve) => setTimeout(resolve, 0));
            """,
            task_feature_enabled=True,
        )

    def test_task_preview_uses_artifact_path_filename_when_id_is_opaque(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("message", {
              data: JSON.stringify({
                type: "task_snapshot",
                payload: {
                  status: "ready",
                  file_count: 1,
                  artifacts: [
                    {
                      id: "artifact-123",
                      kind: "image",
                      label: "Captured screen",
                      path: "/tmp/process/screen-1.png",
                      content_type: "image/png"
                    }
                  ],
                  latest_result: null,
                  error: null
                }
              })
            });

            assert.match(elements["task-preview-list"].innerHTML, /\\/task-artifacts\\/screen-1\\.png/);
            assert.doesNotMatch(elements["task-preview-list"].innerHTML, /artifact-123/);
            """,
            task_feature_enabled=True,
        )

    def test_existing_conversation_message_flow_continues_to_work(self) -> None:
        self._run_node_assertions(
            """
            socket.emit("open");

            socket.emit("message", {
              data: JSON.stringify({
                type: "snapshot",
                payload: {
                  status: "listening",
                  remote_text: "First question",
                  reply_text: "First answer",
                  error: null
                }
              })
            });

            socket.emit("message", {
              data: JSON.stringify({
                type: "transcript",
                payload: {
                  remote_text: "Second question"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_delta",
                payload: {
                  delta: "Second"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_delta",
                payload: {
                  delta: " answer"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "reply_final",
                payload: {
                  reply_text: "Second answer complete"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "processing_error",
                payload: {
                  message: "Temporary issue"
                }
              })
            });
            socket.emit("message", {
              data: JSON.stringify({
                type: "session_stopped",
                payload: {
                  status: "stopped"
                }
              })
            });

            assert.equal(elements["status"].textContent, "stopped");
            assert.equal(elements["error-text"].textContent, "Temporary issue");
            assert.match(elements["conversation-log"].innerHTML, /First question/);
            assert.match(elements["conversation-log"].innerHTML, /First answer/);
            assert.match(elements["conversation-log"].innerHTML, /Second question/);
            assert.match(elements["conversation-log"].innerHTML, /Second answer complete/);
            """
        )

    def _run_node_assertions(self, assertions: str, *, task_feature_enabled: bool = True) -> None:
        app_js_source = APP_JS_PATH.read_text(encoding="utf-8")
        node_script = self._build_node_script(
            app_js_source=app_js_source,
            assertions=assertions,
            task_feature_enabled=task_feature_enabled,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "browser-ui-frontend-test.mjs"
            script_path.write_text(node_script, encoding="utf-8")
            completed = subprocess.run(
                ["node", str(script_path)],
                cwd=PYTHON_ROOT.parent,
                capture_output=True,
                text=True,
                check=False,
            )

        if completed.returncode != 0:
            self.fail(
                "Node frontend assertions failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

    def _build_node_script(self, *, app_js_source: str, assertions: str, task_feature_enabled: bool) -> str:
        return textwrap.dedent(
            f"""
            import assert from "node:assert/strict";
            import vm from "node:vm";

            const appJsSource = {json.dumps(app_js_source)};

            function createClassList(initialClasses = []) {{
              const classes = new Set(initialClasses);
              return {{
                add(...names) {{
                  for (const name of names) classes.add(name);
                }},
                remove(...names) {{
                  for (const name of names) classes.delete(name);
                }},
                toggle(name, force) {{
                  if (force === undefined) {{
                    if (classes.has(name)) {{
                      classes.delete(name);
                      return false;
                    }}
                    classes.add(name);
                    return true;
                  }}
                  if (force) {{
                    classes.add(name);
                    return true;
                  }}
                  classes.delete(name);
                  return false;
                }},
                contains(name) {{
                  return classes.has(name);
                }},
                toString() {{
                  return Array.from(classes).join(" ");
                }}
              }};
            }}

            function createElement(id, options = {{}}) {{
              const listeners = new Map();
              return {{
                id,
                innerHTML: options.innerHTML || "",
                textContent: options.textContent || "",
                value: options.value || "",
                disabled: Boolean(options.disabled),
                classList: createClassList(options.classes || []),
                attributes: new Map(),
                addEventListener(type, handler) {{
                  if (!listeners.has(type)) {{
                    listeners.set(type, []);
                  }}
                  listeners.get(type).push(handler);
                }},
                setAttribute(name, value) {{
                  this.attributes.set(name, String(value));
                }},
                getAttribute(name) {{
                  return this.attributes.has(name) ? this.attributes.get(name) : null;
                }},
                dispatchEvent(event) {{
                  const handlers = listeners.get(event.type) || [];
                  for (const handler of handlers) {{
                    handler(event);
                  }}
                }},
                click() {{
                  this.dispatchEvent({{ type: "click", preventDefault() {{}} }});
                }}
              }};
            }}

            const elements = {{
              "status": createElement("status"),
              "conversation-log": createElement("conversation-log"),
              "error-panel": createElement("error-panel", {{ classes: ["panel-hidden"] }}),
              "error-text": createElement("error-text"),
              "tab-button-convs": createElement("tab-button-convs"),
              "tab-panel-convs": createElement("tab-panel-convs")
            }};

            if ({str(task_feature_enabled).lower()}) {{
              elements["tab-button-task"] = createElement("tab-button-task");
              elements["tab-panel-task"] = createElement("tab-panel-task", {{ classes: ["tab-panel-hidden"] }});
              elements["task-status-value"] = createElement("task-status-value");
              elements["task-file-count-value"] = createElement("task-file-count-value");
              elements["task-artifacts-list"] = createElement("task-artifacts-list");
              elements["task-preview-list"] = createElement("task-preview-list");
              elements["task-result-text"] = createElement("task-result-text");
              elements["task-error-text"] = createElement("task-error-text");
              elements["task-prompt-input"] = createElement("task-prompt-input");
              elements["task-action-screenshot"] = createElement("task-action-screenshot");
              elements["task-action-send"] = createElement("task-action-send");
              elements["task-action-clear"] = createElement("task-action-clear");
            }}

            const document = {{
              documentElement: {{ scrollHeight: 1000 }},
              getElementById(id) {{
                return elements[id] || null;
              }}
            }};

            class FakeWebSocket {{
              static instances = [];

              constructor(url) {{
                this.url = url;
                this.listeners = new Map();
                FakeWebSocket.instances.push(this);
              }}

              addEventListener(type, handler) {{
                if (!this.listeners.has(type)) {{
                  this.listeners.set(type, []);
                }}
                this.listeners.get(type).push(handler);
              }}

              emit(type, payload = {{}}) {{
                const handlers = this.listeners.get(type) || [];
                for (const handler of handlers) {{
                  handler(payload);
                }}
              }}
            }}

            const fetchCalls = [];

            async function fetch(url, options = {{}}) {{
              fetchCalls.push({{ url, options }});
              return {{
                ok: true,
                status: 200
              }};
            }}

            const windowObject = {{
              BROWSER_UI_CONFIG: {{
                websocket_url: "ws://127.0.0.1:43182/browser-ui",
                task_feature_enabled: {str(task_feature_enabled).lower()}
              }},
              scrollCalls: [],
              scrollTo(args) {{
                this.scrollCalls.push(args);
              }}
            }};
            windowObject.fetch = fetch;

            const context = {{
              window: windowObject,
              document,
              WebSocket: FakeWebSocket,
              console,
              fetch,
              setTimeout,
              clearTimeout
            }};
            windowObject.window = windowObject;
            windowObject.document = document;
            context.globalThis = context;

            vm.runInNewContext(appJsSource, context);

            const socket = FakeWebSocket.instances[0];
            assert.ok(socket, "expected app.js to create a WebSocket");

            await (async () => {{
            {textwrap.indent(textwrap.dedent(assertions), "  ")}
            }})();
            """
        )


if __name__ == "__main__":
    unittest.main()
