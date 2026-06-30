(function () {
  const config = window.BROWSER_UI_CONFIG || {};
  const websocketUrl = config.websocket_url;
  const taskFeatureEnabled = Boolean(config.task_feature_enabled);
  const taskArtifactBaseUrl = config.task_artifact_base_url || "/task-artifacts";

  const statusEl = document.getElementById("status");
  const conversationLogEl = document.getElementById("conversation-log");
  const errorPanelEl = document.getElementById("error-panel");
  const errorTextEl = document.getElementById("error-text");
  const convsTabButtonEl = document.getElementById("tab-button-convs");
  const taskTabButtonEl = document.getElementById("tab-button-task");
  const convsTabPanelEl = document.getElementById("tab-panel-convs");
  const taskTabPanelEl = document.getElementById("tab-panel-task");
  const taskStatusValueEl = document.getElementById("task-status-value");
  const taskFileCountValueEl = document.getElementById("task-file-count-value");
  const taskArtifactsListEl = document.getElementById("task-artifacts-list");
  const taskPreviewListEl = document.getElementById("task-preview-list");
  const taskResultTextEl = document.getElementById("task-result-text");
  const taskErrorTextEl = document.getElementById("task-error-text");
  const taskPromptInputEl = document.getElementById("task-prompt-input");
  const taskActionScreenshotEl = document.getElementById("task-action-screenshot");
  const taskActionSendEl = document.getElementById("task-action-send");
  const taskActionClearEl = document.getElementById("task-action-clear");
  const terminalSeparator = "\u2500".repeat(20);

  let conversationEntries = [];
  let currentEntry = null;
  let activeTab = "convs";
  let taskActionPending = false;
  let taskState = {
    status: "empty",
    fileCount: 0,
    artifacts: [],
    latestResult: null,
    error: null,
  };

  function setStatus(text) {
    statusEl.textContent = text || "";
  }

  function setErrorText(text) {
    const normalized = String(text || "").trim();
    errorTextEl.textContent = normalized;
    if (normalized) {
      errorPanelEl.classList.remove("panel-hidden");
      return;
    }
    errorPanelEl.classList.add("panel-hidden");
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function normalizeTerminalText(text) {
    const normalized = String(text || "").replace(/\r\n/g, "\n").trim();
    return normalized || "-";
  }

  function normalizeTaskText(text) {
    const normalized = String(text || "").replace(/\r\n/g, "\n").trim();
    return normalized || "-";
  }

  function normalizeTaskMarkdown(text) {
    return String(text || "").replace(/\r\n/g, "\n").trim();
  }

  function hasVisibleText(text) {
    return Boolean(String(text || "").trim());
  }

  function isTaskCodeFenceLine(line) {
    return /^\s*```[\w-]*\s*$/.test(String(line || ""));
  }

  function isTaskCodeFenceStart(line) {
    return isTaskCodeFenceLine(line);
  }

  function extractTaskCodeFenceLanguage(line) {
    return String(line || "").trim().slice(3).trim();
  }

  function buildTaskCodeBlock(codeLines, language) {
    const languageClass = language ? ' class="language-' + escapeHtml(language) + '"' : "";
    return (
      '<pre class="task-code-block"><code' +
      languageClass +
      ">" +
      escapeHtml(codeLines.join("\n")) +
      "</code></pre>"
    );
  }

  function buildTaskListBlock(items) {
    return (
      "<ul>" +
      items
        .map(function (item) {
          return "<li>" + escapeHtml(item) + "</li>";
        })
        .join("") +
      "</ul>"
    );
  }

  function buildTaskParagraphBlock(lines) {
    return (
      "<p>" +
      lines
        .map(function (line) {
          return escapeHtml(line);
        })
        .join("<br>") +
      "</p>"
    );
  }

  function renderTaskMarkdown(text) {
    const normalized = normalizeTaskMarkdown(text);
    if (!normalized) {
      return "";
    }

    const lines = normalized.split("\n");
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const currentLine = lines[index];

      if (!String(currentLine || "").trim()) {
        index += 1;
        continue;
      }

      if (isTaskCodeFenceStart(currentLine)) {
        const language = extractTaskCodeFenceLanguage(currentLine);
        const codeLines = [];
        index += 1;

        while (index < lines.length && !isTaskCodeFenceLine(lines[index])) {
          codeLines.push(lines[index]);
          index += 1;
        }

        if (index < lines.length && isTaskCodeFenceLine(lines[index])) {
          index += 1;
        }

        blocks.push(buildTaskCodeBlock(codeLines, language));
        continue;
      }

      if (/^\s*[-*]\s+/.test(currentLine)) {
        const items = [];
        while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
          items.push(String(lines[index]).replace(/^\s*[-*]\s+/, ""));
          index += 1;
        }
        blocks.push(buildTaskListBlock(items));
        continue;
      }

      const paragraphLines = [];
      while (
        index < lines.length &&
        String(lines[index] || "").trim() &&
        !isTaskCodeFenceStart(lines[index]) &&
        !/^\s*[-*]\s+/.test(lines[index])
      ) {
        paragraphLines.push(lines[index]);
        index += 1;
      }
      blocks.push(buildTaskParagraphBlock(paragraphLines));
    }

    return blocks.join("");
  }

  function cloneEntry(entry) {
    return {
      remoteText: String(entry && entry.remoteText ? entry.remoteText : ""),
      replyText: String(entry && entry.replyText ? entry.replyText : ""),
    };
  }

  function finalizeCurrentEntry() {
    if (!currentEntry) {
      return;
    }

    if (hasVisibleText(currentEntry.remoteText) || hasVisibleText(currentEntry.replyText)) {
      conversationEntries = conversationEntries.concat([cloneEntry(currentEntry)]);
    }

    currentEntry = null;
  }

  function ensureCurrentEntry() {
    if (!currentEntry) {
      currentEntry = {
        remoteText: "",
        replyText: "",
      };
    }
    return currentEntry;
  }

  function buildTerminalLine(label, labelClass, text, textClass) {
    return (
      '<div class="terminal-entry">' +
      '<span class="terminal-label ' +
      labelClass +
      '">' +
      escapeHtml(label) +
      "</span> " +
      '<span class="terminal-value ' +
      textClass +
      '">' +
      escapeHtml(normalizeTerminalText(text)) +
      "</span>" +
      "</div>"
    );
  }

  function buildConversationBlock(entry, index) {
    const separator =
      index > 0
        ? '<div class="terminal-separator">' + escapeHtml(terminalSeparator) + "</div>"
        : "";

    return (
      separator +
      '<div class="terminal-block">' +
      buildTerminalLine("Remote:", "terminal-label-remote", entry.remoteText, "terminal-text-remote") +
      '<div class="terminal-spacer"></div>' +
      buildTerminalLine("Reply:", "terminal-label-reply", entry.replyText, "terminal-text-reply") +
      "</div>"
    );
  }

  function renderConversationLog() {
    const entries = conversationEntries.slice();
    if (currentEntry && (hasVisibleText(currentEntry.remoteText) || hasVisibleText(currentEntry.replyText))) {
      entries.push(cloneEntry(currentEntry));
    }

    if (!entries.length) {
      conversationLogEl.innerHTML =
        '<div class="terminal-log"><div class="terminal-empty">Waiting for output...</div></div>';
      return;
    }

    conversationLogEl.innerHTML =
      '<div class="terminal-log">' +
      entries
        .map(function (entry, index) {
          return buildConversationBlock(entry, index);
        })
        .join("") +
      "</div>";
    if (activeTab === "convs") {
      scrollPageToBottom();
    }
  }

  function setActiveTab(tabName) {
    if (!taskFeatureEnabled) {
      activeTab = "convs";
      convsTabPanelEl.classList.remove("tab-panel-hidden");
      convsTabButtonEl.classList.add("tab-button-active");
      if (taskTabPanelEl) {
        taskTabPanelEl.classList.add("tab-panel-hidden");
      }
      if (taskTabButtonEl) {
        taskTabButtonEl.classList.remove("tab-button-active");
      }
      return;
    }

    activeTab = tabName === "task" ? "task" : "convs";

    const showingConvs = activeTab === "convs";
    convsTabPanelEl.classList.toggle("tab-panel-hidden", !showingConvs);
    if (taskTabPanelEl) {
      taskTabPanelEl.classList.toggle("tab-panel-hidden", showingConvs);
    }
    convsTabButtonEl.classList.toggle("tab-button-active", showingConvs);
    if (taskTabButtonEl) {
      taskTabButtonEl.classList.toggle("tab-button-active", !showingConvs);
    }
  }

  function scrollPageToBottom() {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "auto",
    });
  }

  function hydrateCurrentEntry(remoteText, replyText) {
    currentEntry = {
      remoteText: String(remoteText || ""),
      replyText: String(replyText || ""),
    };
    renderConversationLog();
  }

  function startTranscript(remoteText) {
    finalizeCurrentEntry();
    currentEntry = {
      remoteText: String(remoteText || ""),
      replyText: "",
    };
    renderConversationLog();
  }

  function appendReplyDelta(delta) {
    const entry = ensureCurrentEntry();
    entry.replyText = entry.replyText + String(delta || "");
    renderConversationLog();
  }

  function finalizeReply(replyText) {
    const entry = ensureCurrentEntry();
    entry.replyText = String(replyText || "");
    renderConversationLog();
  }

  function isImageArtifact(artifact) {
    const contentType = String((artifact && artifact.content_type) || "").toLowerCase();
    return contentType.indexOf("image/") === 0;
  }

  function getArtifactRelativeName(artifact) {
    const artifactPath = String((artifact && artifact.path) || "");
    const segments = artifactPath.split(/[\\\\/]/);
    const basename = segments[segments.length - 1] || "";
    if (basename) {
      return basename;
    }
    if (artifact && typeof artifact.id === "string" && artifact.id.trim()) {
      return artifact.id.trim();
    }
    return "";
  }

  function buildTaskArtifactList(artifacts) {
    if (!artifacts.length) {
      return '<div class="task-empty">No artifacts yet.</div>';
    }

    return (
      '<div class="task-artifact-list">' +
      artifacts
        .map(function (artifact) {
          const meta = [artifact.kind || "artifact", artifact.content_type || ""]
            .filter(function (value) {
              return Boolean(String(value || "").trim());
            })
            .join(" \u2022 ");
          return (
            '<div class="task-artifact-item">' +
            '<span class="task-artifact-label">' +
            escapeHtml(artifact.label || artifact.id || artifact.path || "artifact") +
            "</span>" +
            '<span class="task-artifact-meta">' +
            escapeHtml(meta || "-") +
            "</span>" +
            "</div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function buildTaskPreviewList(artifacts) {
    const previewArtifacts = artifacts.filter(isImageArtifact);
    if (!previewArtifacts.length) {
      return '<div class="task-empty">No preview available.</div>';
    }

    return previewArtifacts
      .map(function (artifact) {
        const relativeName = getArtifactRelativeName(artifact);
        const previewUrl = taskArtifactBaseUrl + "/" + encodeURIComponent(relativeName);
        return (
          '<figure class="task-preview-card">' +
          '<img src="' +
          escapeHtml(previewUrl) +
          '" alt="' +
          escapeHtml(artifact.label || relativeName || "Task artifact preview") +
          '">' +
          '<figcaption class="task-preview-label">' +
          escapeHtml(artifact.label || relativeName || "artifact") +
          "</figcaption>" +
          "</figure>"
        );
      })
      .join("");
  }

  function renderLatestResult(latestResult) {
    if (!latestResult || typeof latestResult !== "object") {
      return escapeHtml("-");
    }

    const metadata = [latestResult.name || "-", latestResult.status || "-", latestResult.summary || ""].filter(
      function (value, index) {
        return index < 2 || hasVisibleText(value);
      }
    );
    const responseMarkup = renderTaskMarkdown(latestResult.response_text || "");

    return (
      '<div class="task-result-meta">' +
      metadata
        .map(function (line) {
          return '<div class="task-result-line">' + escapeHtml(line) + "</div>";
        })
        .join("") +
      "</div>" +
      (responseMarkup ? '<div class="task-result-body">' + responseMarkup + "</div>" : "")
    );
  }

  function renderTaskPanel() {
    if (!taskFeatureEnabled) {
      return;
    }
    if (
      !taskStatusValueEl ||
      !taskFileCountValueEl ||
      !taskArtifactsListEl ||
      !taskPreviewListEl ||
      !taskResultTextEl ||
      !taskErrorTextEl
    ) {
      return;
    }
    taskStatusValueEl.textContent = normalizeTaskText(taskState.status);
    taskFileCountValueEl.textContent = String(taskState.fileCount || 0);
    taskArtifactsListEl.innerHTML = buildTaskArtifactList(taskState.artifacts);
    taskPreviewListEl.innerHTML = buildTaskPreviewList(taskState.artifacts);
    taskResultTextEl.innerHTML = renderLatestResult(taskState.latestResult);
    taskErrorTextEl.textContent = normalizeTaskText(taskState.error);
    syncTaskActionButtons();
  }

  function syncTaskActionButtons() {
    const buttons = [taskActionScreenshotEl, taskActionSendEl, taskActionClearEl];
    buttons.forEach(function (button) {
      if (!button) {
        return;
      }
      button.disabled = taskActionPending;
      button.setAttribute("aria-disabled", taskActionPending ? "true" : "false");
    });
  }

  function clearTaskLatestResult() {
    if (!taskFeatureEnabled) {
      return;
    }
    taskState = {
      status: taskState.status,
      fileCount: taskState.fileCount,
      artifacts: taskState.artifacts.slice(),
      latestResult: null,
      error: taskState.error,
    };
    renderTaskPanel();
  }

  async function runTaskAction(actionName) {
    if (!taskFeatureEnabled || taskActionPending) {
      return;
    }
    let requestBody = null;
    if (actionName === "send") {
      clearTaskLatestResult();
      const taskPrompt = taskPromptInputEl ? String(taskPromptInputEl.value || "").trim() : "";
      if (taskPrompt) {
        requestBody = JSON.stringify({
          task_prompt: taskPrompt,
        });
      }
    }
    taskActionPending = true;
    syncTaskActionButtons();
    setErrorText("");

    try {
      const response = await fetch("/api/task/" + encodeURIComponent(actionName), {
        method: "POST",
        headers: requestBody ? { "Content-Type": "application/json" } : undefined,
        body: requestBody,
      });
      if (!response.ok) {
        throw new Error("Task action request failed with status " + response.status + ".");
      }
    } catch (error) {
      setErrorText(error && error.message ? error.message : "Task action request failed.");
    } finally {
      taskActionPending = false;
      syncTaskActionButtons();
    }
  }

  function applyTaskSnapshot(payload) {
    if (!taskFeatureEnabled) {
      return;
    }
    taskState = {
      status: String((payload && payload.status) || "empty"),
      fileCount: Number((payload && payload.file_count) || 0),
      artifacts: Array.isArray(payload && payload.artifacts) ? payload.artifacts.slice() : [],
      latestResult: payload && payload.latest_result ? payload.latest_result : null,
      error: payload && typeof payload.error === "string" ? payload.error : null,
    };
    renderTaskPanel();
  }

  function applyMessage(message) {
    if (!message || typeof message !== "object") {
      return;
    }

    const payload = message.payload || {};

    switch (message.type) {
      case "snapshot":
        setStatus(payload.status);
        conversationEntries = [];
        hydrateCurrentEntry(payload.remote_text, payload.reply_text);
        setErrorText(payload.error);
        break;
      case "transcript":
        startTranscript(payload.remote_text);
        break;
      case "reply_delta":
        appendReplyDelta(payload.delta);
        break;
      case "reply_final":
        finalizeReply(payload.reply_text);
        break;
      case "processing_error":
        setErrorText(payload.message);
        break;
      case "session_stopped":
        setStatus(payload.status || "stopped");
        break;
      case "task_snapshot":
        applyTaskSnapshot(payload);
        break;
      default:
        break;
    }
  }

  convsTabButtonEl.addEventListener("click", function () {
    setActiveTab("convs");
  });

  if (taskTabButtonEl) {
    taskTabButtonEl.addEventListener("click", function () {
      setActiveTab("task");
    });
  }

  if (taskActionScreenshotEl) {
    taskActionScreenshotEl.addEventListener("click", function () {
      void runTaskAction("screenshot");
    });
  }

  if (taskActionSendEl) {
    taskActionSendEl.addEventListener("click", function () {
      void runTaskAction("send");
    });
  }

  if (taskActionClearEl) {
    taskActionClearEl.addEventListener("click", function () {
      void runTaskAction("clear");
    });
  }

  if (!taskFeatureEnabled) {
    if (taskTabButtonEl) {
      taskTabButtonEl.classList.add("tab-button-hidden");
    }
    if (taskTabPanelEl) {
      taskTabPanelEl.classList.add("tab-panel-hidden");
    }
  }

  setActiveTab(activeTab);
  renderTaskPanel();

  if (!websocketUrl) {
    setStatus("Missing websocket URL");
    setErrorText("Browser UI configuration is incomplete.");
    renderConversationLog();
    return;
  }

  renderConversationLog();

  const socket = new WebSocket(websocketUrl);

  socket.addEventListener("open", function () {
    setStatus("connected");
  });

  socket.addEventListener("message", function (event) {
    try {
      applyMessage(JSON.parse(event.data));
    } catch (error) {
      setErrorText("Failed to parse socket message.");
    }
  });

  socket.addEventListener("close", function () {
    setStatus("disconnected");
  });

  socket.addEventListener("error", function () {
    setErrorText("WebSocket connection error.");
  });
})();
