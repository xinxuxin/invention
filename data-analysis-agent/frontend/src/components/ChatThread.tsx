import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Code2,
  Loader2,
  MessageSquare,
  Radio,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  UserRound,
} from "lucide-react";

import { ArtifactCard } from "./ArtifactCard";
import { Button } from "./Button";
import { cn } from "../lib/utils";
import type { ChatMessage, ChatTraceEvent, PendingConfirmation } from "../hooks/useChat";

type ChatThreadProps = {
  sessionId?: string | null;
  messages: ChatMessage[];
  pendingConfirmation: PendingConfirmation | null;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ChatThread({
  sessionId,
  messages,
  pendingConfirmation,
  onConfirm,
  onCancel,
}: ChatThreadProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const [nearBottom, setNearBottom] = useState(true);
  const [showNewUpdates, setShowNewUpdates] = useState(false);
  const activityKey = useMemo(() => messageActivityKey(messages), [messages]);
  const hasStreamingMessage = messages.some((message) => message.status === "streaming");

  useEffect(() => {
    const parent = getScrollParent(containerRef.current);
    if (!parent) {
      return;
    }

    const updatePosition = () => {
      const nextNearBottom = isNearScrollBottom(parent);
      setNearBottom(nextNearBottom);
      if (nextNearBottom) {
        setShowNewUpdates(false);
      }
    };

    updatePosition();
    parent.addEventListener("scroll", updatePosition, { passive: true });
    return () => parent.removeEventListener("scroll", updatePosition);
  }, []);

  useEffect(() => {
    const latestMessage = messages[messages.length - 1];
    if (!hasStreamingMessage && !latestMessage?.finalAnswer) {
      return;
    }

    const parent = getScrollParent(containerRef.current);
    const shouldScroll = parent ? isNearScrollBottom(parent) || nearBottom : nearBottom;
    if (shouldScroll) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      setShowNewUpdates(false);
    } else if (hasStreamingMessage) {
      setShowNewUpdates(true);
    }
  }, [activityKey, hasStreamingMessage, messages, nearBottom]);

  if (messages.length === 0) {
    return <ChatEmptyState />;
  }

  return (
    <div ref={containerRef} className="relative space-y-4">
      {messages.map((message, index) => (
        <motion.div
          key={message.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: Math.min(index * 0.03, 0.2), duration: 0.25 }}
        >
          {message.role === "user" ? (
            <UserMessage message={message} />
          ) : (
            <AssistantMessage
              sessionId={sessionId}
              message={message}
              pendingConfirmation={pendingConfirmation}
              onConfirm={onConfirm}
              onCancel={onCancel}
            />
          )}
        </motion.div>
      ))}
      <div ref={endRef} />
      {showNewUpdates ? (
        <button
          type="button"
          className="sticky bottom-3 left-full z-10 ml-auto flex w-fit items-center gap-2 rounded-full border border-teal-200 bg-white/95 px-3 py-2 text-xs font-bold text-teal-800 shadow-lg shadow-teal-900/10 backdrop-blur"
          onClick={() => {
            endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
            setShowNewUpdates(false);
          }}
        >
          <Radio className="h-3.5 w-3.5" />
          New updates
        </button>
      ) : null}
    </div>
  );
}

function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="ml-auto max-w-[82%] rounded-lg border border-teal-200 bg-teal-700 p-4 text-white shadow-lg shadow-teal-900/10">
      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-teal-100">
        <UserRound className="h-3.5 w-3.5" />
        You
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-6">{message.content}</p>
    </div>
  );
}

function AssistantMessage({
  sessionId,
  message,
  pendingConfirmation,
  onConfirm,
  onCancel,
}: {
  sessionId?: string | null;
  message: ChatMessage;
  pendingConfirmation: PendingConfirmation | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const isStreaming = message.status === "streaming";
  const progress = agentProgress(message);

  return (
    <div className="max-w-[92%] space-y-3">
      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-muted-foreground">
        <span className="flex h-7 w-7 items-center justify-center rounded-md bg-slate-950 text-white">
          <Bot className="h-3.5 w-3.5" />
        </span>
        Agent
        {message.status === "streaming" ? (
          <span className="ml-1 flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-amber-700">
            <Loader2 className="h-3 w-3 animate-spin" />
            Working
          </span>
        ) : null}
      </div>

      {isStreaming ? <ThinkingProgress progress={progress} /> : null}

      {message.trace.length > 0 ? <TracePanel trace={message.trace} isStreaming={isStreaming} /> : null}

      {pendingConfirmation?.assistantMessageId === message.id ? (
        <ConfirmationCard pending={pendingConfirmation} onConfirm={onConfirm} onCancel={onCancel} />
      ) : null}

      {message.finalAnswer ? (
        <div className="rounded-lg border-2 border-teal-600 bg-white p-5 shadow-lg shadow-teal-900/10">
          <div className="flex items-center gap-2 text-sm font-bold text-teal-800">
            <CheckCircle2 className="h-4 w-4" />
            Final answer
            {message.stateChanged ? (
              <span className="rounded-full bg-teal-700 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
                State changed
              </span>
            ) : null}
          </div>
          {message.highlights?.length ? <HighlightChips highlights={message.highlights} /> : null}
          <AnswerMarkdown markdown={message.finalAnswer} />
          {message.warnings?.length ? (
            <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs font-semibold text-amber-900">
              {message.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {message.artifacts.length > 0 && sessionId ? (
        <div className="space-y-3">
          {message.artifacts.map((artifact) => (
            <ArtifactCard key={artifact.id} sessionId={sessionId} artifact={artifact} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ThinkingProgress({ progress }: { progress: AgentProgress }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-lg border border-dashed border-slate-300 bg-white/70 p-4"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            <Sparkles className="h-4 w-4 text-teal-700" />
            Working through agent steps
          </div>
          <p className="mt-1 truncate text-xs font-medium text-muted-foreground">
            {progress.label} • {progress.steps} step{progress.steps === 1 ? "" : "s"}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-teal-50 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-teal-800">
          Live
        </span>
      </div>
      <div className="relative mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
        <motion.div
          className="h-full rounded-full bg-teal-600"
          initial={false}
          animate={{ width: `${progress.value}%` }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        />
        {progress.waiting ? (
          <motion.div
            className="absolute inset-y-0 w-1/3 rounded-full bg-white/35"
            initial={{ x: "-120%" }}
            animate={{ x: "320%" }}
            transition={{ duration: 1.35, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : null}
      </div>
    </motion.div>
  );
}

function TracePanel({ trace, isStreaming }: { trace: ChatTraceEvent[]; isStreaming: boolean }) {
  const [open, setOpen] = useState(true);
  const [hasNewTraceWhileCollapsed, setHasNewTraceWhileCollapsed] = useState(false);
  const lastTraceIdRef = useRef<string | null>(null);
  const latest = trace[trace.length - 1];

  useEffect(() => {
    if (!latest) {
      return;
    }

    const previousId = lastTraceIdRef.current;
    lastTraceIdRef.current = latest.id;
    if (previousId === null || previousId === latest.id) {
      return;
    }

    if (isStreaming && !open) {
      setOpen(true);
      setHasNewTraceWhileCollapsed(true);
      window.setTimeout(() => setHasNewTraceWhileCollapsed(false), 1800);
    }
  }, [isStreaming, latest?.id, open]);

  const handleToggle = () => {
    setOpen((value) => {
      const next = !value;
      if (next) {
        setHasNewTraceWhileCollapsed(false);
      }
      return next;
    });
  };

  return (
    <motion.section layout className="overflow-hidden rounded-lg border border-border bg-white/72 shadow-sm">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
        onClick={handleToggle}
      >
        <span className="flex items-center gap-2 text-sm font-bold">
          <TerminalSquare className="h-4 w-4 text-indigo-700" />
          Trace
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-600">
            {trace.length}
          </span>
          {hasNewTraceWhileCollapsed ? (
            <span className="rounded-full bg-teal-100 px-2 py-0.5 text-[11px] font-bold text-teal-800">
              New trace update
            </span>
          ) : null}
        </span>
        <span className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
          <span className="truncate">{traceSummary(latest)}</span>
          <ChevronDown className={cn("h-4 w-4 transition", open ? "rotate-180" : "")} />
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            key="trace-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden border-t border-border/70"
          >
            <div className="space-y-2 p-3">
              {trace.map((item) => (
                <motion.div
                  key={item.id}
                  layout
                  initial={{ opacity: 0, y: 7, scale: 0.99 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.22, ease: "easeOut" }}
                >
                  <TraceRow item={item} />
                </motion.div>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}

function TraceRow({ item }: { item: ChatTraceEvent }) {
  if (item.type === "code_started") {
    return (
      <div className="rounded-md border border-slate-200 bg-slate-950 p-3">
        <div className="mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300">
          <Code2 className="h-3.5 w-3.5" />
          Python
        </div>
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-100">
          {item.code}
        </pre>
      </div>
    );
  }

  if (item.type === "code_result_summary") {
    return <CodeResultRow item={item} />;
  }

  if (item.type === "error") {
    return (
      <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-xs font-semibold text-rose-800">
        {item.message}
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2 rounded-md border border-border bg-white p-3 text-sm">
      <span className="mt-1 h-2 w-2 rounded-full bg-teal-500" />
      <span className="leading-5 text-slate-700">{item.message ?? traceSummary(item)}</span>
    </div>
  );
}

function CodeResultRow({ item }: { item: ChatTraceEvent }) {
  const [open, setOpen] = useState(true);
  const detail = item.ok ? item.stdout : item.stderr || item.traceback || item.stdout;

  useEffect(() => {
    if (item.ok) {
      const timer = window.setTimeout(() => setOpen(false), 3600);
      return () => window.clearTimeout(timer);
    }

    const timer = window.setTimeout(() => setOpen(false), 5200);
    return () => window.clearTimeout(timer);
  }, [item.id, item.ok]);

  return (
    <motion.div
      layout
      className={cn(
        "overflow-hidden rounded-md border text-xs",
        item.ok ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50",
      )}
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left"
        onClick={() => setOpen((value) => !value)}
      >
        <span className="flex min-w-0 items-center gap-2 font-bold">
          {item.ok ? (
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-700" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-rose-700" />
          )}
          <span>Code {item.ok ? "completed" : "failed"}</span>
          {item.updatedDatasets?.length ? (
            <span className="rounded-full bg-teal-700 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
              saved
            </span>
          ) : null}
        </span>
        <span className="flex min-w-0 items-center gap-2 text-muted-foreground">
          <span className="truncate">{compactResultSummary(item)}</span>
          <ChevronDown className={cn("h-4 w-4 shrink-0 transition", open ? "rotate-180" : "")} />
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            key="code-result-detail"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="border-t border-white/80 px-3 py-2.5">
              {detail ? (
                <pre
                  className={cn(
                    "max-h-32 overflow-auto whitespace-pre-wrap rounded-md bg-white/70 p-2 leading-5",
                    item.ok ? "text-emerald-900" : "text-rose-800",
                  )}
                >
                  {detail}
                </pre>
              ) : (
                <p className="font-medium text-muted-foreground">No console output.</p>
              )}
              {item.updatedDatasets?.length ? (
                <p className="mt-2 font-semibold text-teal-800">
                  Saved {item.updatedDatasets.length} dataset version
                  {item.updatedDatasets.length === 1 ? "" : "s"}.
                </p>
              ) : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.div>
  );
}

function ConfirmationCard({
  pending,
  onConfirm,
  onCancel,
}: {
  pending: PendingConfirmation;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="rounded-lg border-2 border-amber-400 bg-amber-50 p-5 shadow-lg shadow-amber-900/10">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-amber-500 text-white">
          <AlertTriangle className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold text-amber-950">
            {pending.title || "Confirmation required"}
          </p>
          <span className="mt-2 inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-emerald-800">
            <ShieldCheck className="h-3 w-3" />
            Safe mode
          </span>
          <p className="mt-2 text-sm leading-6 text-amber-950/80">
            {pending.operationSummary || pending.message}
          </p>
          {pending.expectedEffect ? (
            <p className="mt-2 text-sm leading-6 text-amber-950/80">{pending.expectedEffect}</p>
          ) : null}
          <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
            <RiskMetric label="Risk level" value={formatRisk(pending.riskLevel)} />
            <RiskMetric label="Dataset" value={pending.datasetName ?? "Current dataset"} />
            <RiskMetric
              label="Affected data"
              value={
                pending.affectedDatasetIds?.length
                  ? `${pending.affectedDatasetIds.length} dataset${pending.affectedDatasetIds.length === 1 ? "" : "s"}`
                  : "Current dataset"
              }
            />
          </div>
          <div className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
            <RiskMetric label="State impact" value={pending.stateImpact ?? "Creates a new version if applied"} />
            <RiskMetric
              label="Reversibility"
              value={pending.reversible === false ? "Not automatically reversible" : "Rollback available"}
            />
          </div>
          {pending.currentRowCount !== undefined && pending.currentRowCount !== null ? (
            <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
              <RiskMetric label="Current rows" value={pending.currentRowCount.toLocaleString()} />
              <RiskMetric label="New rows" value={(pending.newRowCount ?? pending.currentRowCount).toLocaleString()} />
              <RiskMetric label="Affected rows" value={(pending.affectedCount ?? 0).toLocaleString()} />
            </div>
          ) : null}
          {pending.rollbackNote ? (
            <p className="mt-3 rounded-md border border-amber-200 bg-white/60 p-3 text-xs leading-5 text-amber-950/75">
              {pending.rollbackNote}
            </p>
          ) : null}
          {pending.code ? (
            <details className="mt-3 rounded-md border border-amber-200 bg-white/70 p-3">
              <summary className="cursor-pointer text-xs font-bold text-amber-950">Show code</summary>
              <pre className="mt-3 max-h-36 overflow-auto whitespace-pre-wrap rounded-md bg-slate-950 p-3 text-xs text-slate-100">
                {pending.code}
              </pre>
            </details>
          ) : null}
          <div className="mt-4 flex gap-2">
            <Button onClick={onConfirm}>{pending.confirmLabel || "Apply change"}</Button>
            <Button variant="secondary" onClick={onCancel}>
              {pending.cancelLabel || "Cancel"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatRisk(risk?: string | null) {
  if (!risk) {
    return "Medium";
  }
  return risk.charAt(0).toUpperCase() + risk.slice(1);
}

function RiskMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-amber-200 bg-white/70 p-2">
      <p className="text-[10px] font-bold uppercase tracking-wide text-amber-900/60">{label}</p>
      <p className="mt-1 truncate font-bold text-amber-950">{value}</p>
    </div>
  );
}

function ChatEmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white/64 p-8 text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-slate-950 text-white">
        <MessageSquare className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm font-bold">Ask the agent about your data</p>
      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted-foreground">
        Try “What’s in this file?”, “Summarize the nested structure”, or “Create a CSV export”.
        Trace events will stream here as the agent works.
      </p>
    </div>
  );
}

function traceSummary(item?: ChatTraceEvent) {
  if (!item) {
    return "Waiting";
  }
  if (item.type === "trace") {
    return item.message ?? "Trace";
  }
  if (item.type === "code_started") {
    return "Running Python";
  }
  if (item.type === "code_result_summary") {
    return item.ok ? "Python completed" : "Python failed";
  }
  if (item.type === "confirmation_required") {
    return "Confirmation required";
  }
  if (item.type === "verifier_result") {
    return item.message ?? "Verifier checked result";
  }
  if (item.type === "error") {
    return "Error";
  }
  return item.type;
}

function HighlightChips({ highlights }: { highlights: Array<Record<string, unknown>> }) {
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {highlights.slice(0, 6).map((item, index) => (
        <span
          key={`${String(item.label)}-${index}`}
          className="rounded-full border border-teal-100 bg-teal-50 px-2.5 py-1 text-[11px] font-bold text-teal-900"
        >
          {String(item.label ?? "Metric")}: {String(item.value ?? "")}
        </span>
      ))}
    </div>
  );
}

function AnswerMarkdown({ markdown }: { markdown: string }) {
  const lines = markdown.split("\n");
  return (
    <div className="mt-3 space-y-2 text-sm leading-6 text-slate-700">
      {lines.map((line, index) => {
        if (!line.trim()) {
          return <div key={index} className="h-1" />;
        }
        if (line.startsWith("### ")) {
          return (
            <h4 key={index} className="pt-2 text-sm font-bold text-slate-900">
              {inlineMarkdown(line.slice(4))}
            </h4>
          );
        }
        if (line.startsWith("## ")) {
          return (
            <h3 key={index} className="text-base font-bold text-slate-950">
              {inlineMarkdown(line.slice(3))}
            </h3>
          );
        }
        if (line.startsWith("- ")) {
          return (
            <div key={index} className="flex gap-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-teal-600" />
              <span>{inlineMarkdown(line.slice(2))}</span>
            </div>
          );
        }
        if (line.startsWith("```")) {
          return null;
        }
        return <p key={index}>{inlineMarkdown(line)}</p>;
      })}
    </div>
  );
}

function inlineMarkdown(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={index} className="rounded bg-slate-100 px-1 py-0.5 text-[0.9em] font-semibold text-slate-800">
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

type AgentProgress = {
  value: number;
  label: string;
  steps: number;
  waiting: boolean;
};

function agentProgress(message: ChatMessage): AgentProgress {
  let value = 5;
  let label = "Preparing...";
  const steps = message.trace.length + message.artifacts.length + (message.finalAnswer ? 1 : 0);

  message.trace.forEach((item) => {
    if (item.type === "trace") {
      value = Math.max(value, 12) + 4;
      label = item.message || "Inspecting data...";
      return;
    }

    if (item.type === "code_started") {
      value = Math.max(value, 25) + 6;
      label = "Running Python...";
      return;
    }

    if (item.type === "code_result_summary") {
      value += 10;
      label = item.ok ? "Reviewing Python result..." : "Recovering from Python error...";
      return;
    }

    if (item.type === "confirmation_required") {
      value = Math.max(value, 70);
      label = "Waiting for confirmation...";
      return;
    }

    if (item.type === "verifier_result") {
      value = Math.max(value, 78);
      label = item.message || "Verifying result...";
      return;
    }

    if (item.type === "error") {
      value = 100;
      label = "Handling an error...";
    }
  });

  if (message.artifacts.length > 0) {
    value += message.artifacts.length * 8;
    label = "Creating artifact...";
  }

  if (message.finalAnswer) {
    return {
      value: 100,
      label: "Finalizing answer...",
      steps,
      waiting: false,
    };
  }

  if (message.status === "done" || message.status === "error" || message.status === "waiting_confirmation") {
    value = 100;
  }

  const cappedValue = message.status === "streaming" ? Math.min(value, 92) : Math.min(value, 100);
  return {
    value: Math.max(5, cappedValue),
    label,
    steps,
    waiting: message.status === "streaming",
  };
}

function messageActivityKey(messages: ChatMessage[]) {
  return messages
    .map((message) =>
      [
        message.id,
        message.status,
        message.trace.length,
        message.trace[message.trace.length - 1]?.id ?? "none",
        message.artifacts.length,
        message.finalAnswer ? "final" : "no-final",
      ].join(":"),
    )
    .join("|");
}

function getScrollParent(element: HTMLElement | null): HTMLElement | null {
  let current = element?.parentElement ?? null;
  while (current) {
    const style = window.getComputedStyle(current);
    if (/(auto|scroll)/.test(style.overflowY) && current.scrollHeight > current.clientHeight) {
      return current;
    }
    current = current.parentElement;
  }
  return document.scrollingElement instanceof HTMLElement ? document.scrollingElement : null;
}

function isNearScrollBottom(element: HTMLElement) {
  return element.scrollHeight - element.scrollTop - element.clientHeight < 160;
}

function compactResultSummary(item: ChatTraceEvent) {
  if (item.type !== "code_result_summary") {
    return traceSummary(item);
  }

  if (item.updatedDatasets?.length) {
    return `${item.updatedDatasets.length} version${item.updatedDatasets.length === 1 ? "" : "s"} saved`;
  }

  const text = item.ok ? item.stdout : item.stderr || item.traceback;
  if (!text) {
    return item.ok ? "No output" : "See error";
  }
  return text.replace(/\s+/g, " ").slice(0, 96);
}
