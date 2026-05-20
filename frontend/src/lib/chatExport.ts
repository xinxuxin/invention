import type { ChatMessage, ChatTraceEvent } from "../hooks/useChat";

type ChatExportOptions = {
  includeTrace: boolean;
  sessionName?: string | null;
  activeDatasetName?: string | null;
  branchName?: string | null;
};

export function downloadChatTranscript(messages: ChatMessage[], options: ChatExportOptions) {
  const markdown = buildChatTranscriptMarkdown(messages, options);
  const filename = `data-analysis-chat-${new Date().toISOString().replace(/[:.]/g, "-")}.md`;
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function buildChatTranscriptMarkdown(messages: ChatMessage[], options: ChatExportOptions) {
  const lines = [
    "# Data Analysis Agent Chat Export",
    "",
    `- Exported: ${new Date().toLocaleString()}`,
    `- Session: ${options.sessionName || "Untitled session"}`,
    `- Active dataset: ${options.activeDatasetName || "None"}`,
    `- Branch: ${options.branchName || "main"}`,
    `- Trace included: ${options.includeTrace ? "yes" : "no"}`,
    "",
  ];

  if (messages.length === 0) {
    lines.push("_No chat messages have been sent in this browser session yet._", "");
    return lines.join("\n");
  }

  messages.forEach((message, index) => {
    const title = message.role === "user" ? "User" : "Assistant";
    lines.push(`## ${index + 1}. ${title}`, "");

    if (message.role === "user") {
      lines.push(message.content || "_Empty message_", "");
      return;
    }

    if (message.finalAnswer) {
      lines.push("### Final Answer", "", message.finalAnswer, "");
    } else if (message.status === "waiting_confirmation") {
      lines.push("_Assistant is waiting for confirmation._", "");
    } else if (message.status === "waiting_clarification") {
      lines.push("_Assistant is waiting for clarification._", "");
    } else if (message.status === "streaming") {
      lines.push("_Assistant response was still streaming when this transcript was exported._", "");
    } else if (message.status === "error") {
      lines.push("_Assistant response ended with an error._", "");
    } else {
      lines.push("_No final answer recorded._", "");
    }

    if (message.stateChanged !== undefined && !message.finalAnswer?.toLowerCase().includes("state changed")) {
      lines.push(`State changed: ${message.stateChanged ? "yes" : "no"}`, "");
    }

    if (message.artifacts.length > 0) {
      lines.push("### Artifacts", "");
      message.artifacts.forEach((artifact) => {
        lines.push(`- ${artifact.name} (${artifact.kind}) - ${artifact.id}`);
      });
      lines.push("");
    }

    if (options.includeTrace && message.trace.length > 0) {
      lines.push(`<details open>`, `<summary>Trace events (${message.trace.length})</summary>`, "");
      message.trace.forEach((trace, traceIndex) => {
        lines.push(`#### Trace ${traceIndex + 1}: ${traceTitle(trace)}`, "");
        lines.push(...formatTrace(trace), "");
      });
      lines.push("</details>", "");
    }
  });

  return lines.join("\n");
}

function formatTrace(trace: ChatTraceEvent) {
  if (trace.type === "trace") {
    return [trace.message || "Trace event"];
  }

  if (trace.type === "code_started") {
    return ["```python", trace.code || "", "```"];
  }

  if (trace.type === "code_result_summary") {
    const lines = [`Status: ${trace.ok ? "completed" : "failed"}`];
    if (trace.resultSummary) {
      lines.push("", "summary:", fenced(JSON.stringify(trace.resultSummary, null, 2), "json"));
    }
    if (trace.stdout) {
      lines.push("", "stdout:", fenced(trace.stdout, "text"));
    }
    if (trace.stderr) {
      lines.push("", "stderr:", fenced(trace.stderr, "text"));
    }
    if (trace.traceback) {
      lines.push("", "traceback:", fenced(trace.traceback, "text"));
    }
    if (trace.resultPreview !== undefined) {
      lines.push("", "result preview:", fenced(JSON.stringify(trace.resultPreview, null, 2), "json"));
    }
    if (trace.updatedDatasets?.length) {
      lines.push("", "updated datasets:");
      trace.updatedDatasets.forEach((dataset) => {
        lines.push(`- ${dataset.key}: ${dataset.mutation_summary} (${dataset.version_id})`);
      });
    }
    return lines;
  }

  if (trace.type === "error") {
    return [trace.message || "Unknown error"];
  }

  if (trace.type === "confirmation_required") {
    return [trace.message || "Confirmation required", trace.code ? fenced(trace.code, "python") : ""].filter(Boolean);
  }

  return [trace.message || trace.type];
}

function traceTitle(trace: ChatTraceEvent) {
  if (trace.type === "trace") {
    return trace.message || "Progress";
  }
  if (trace.type === "code_started") {
    return "Python code";
  }
  if (trace.type === "code_result_summary") {
    return trace.ok ? "Code completed" : "Code failed";
  }
  if (trace.type === "confirmation_required") {
    return "Confirmation required";
  }
  if (trace.type === "error") {
    return "Error";
  }
  return trace.type;
}

function fenced(value: string, language: string) {
  const fence = value.includes("```") ? "````" : "```";
  return `${fence}${language}\n${value}\n${fence}`;
}
