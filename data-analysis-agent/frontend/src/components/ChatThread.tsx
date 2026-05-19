import { useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Code2,
  Loader2,
  MessageSquare,
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
  if (messages.length === 0) {
    return <ChatEmptyState />;
  }

  return (
    <div className="space-y-4">
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

      {message.trace.length > 0 ? <TracePanel trace={message.trace} /> : null}

      {pendingConfirmation?.assistantMessageId === message.id ? (
        <ConfirmationCard pending={pendingConfirmation} onConfirm={onConfirm} onCancel={onCancel} />
      ) : null}

      {message.artifacts.length > 0 && sessionId ? (
        <div className="space-y-3">
          {message.artifacts.map((artifact) => (
            <ArtifactCard key={artifact.id} sessionId={sessionId} artifact={artifact} />
          ))}
        </div>
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
          <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-700">{message.finalAnswer}</p>
        </div>
      ) : message.status === "streaming" ? (
        <div className="rounded-lg border border-dashed border-slate-300 bg-white/70 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
            <Sparkles className="h-4 w-4 text-teal-700" />
            The agent is working through the data...
          </div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-200">
            <div className="h-full w-1/2 animate-pulse rounded-full bg-teal-600" />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function TracePanel({ trace }: { trace: ChatTraceEvent[] }) {
  const [open, setOpen] = useState(true);
  const latest = trace[trace.length - 1];

  return (
    <section className="overflow-hidden rounded-lg border border-border bg-white/72">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
        onClick={() => setOpen((value) => !value)}
      >
        <span className="flex items-center gap-2 text-sm font-bold">
          <TerminalSquare className="h-4 w-4 text-indigo-700" />
          Trace
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-600">
            {trace.length}
          </span>
        </span>
        <span className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
          <span className="truncate">{traceSummary(latest)}</span>
          <ChevronDown className={cn("h-4 w-4 transition", open ? "rotate-180" : "")} />
        </span>
      </button>
      {open ? (
        <div className="space-y-2 border-t border-border/70 p-3">
          {trace.map((item) => (
            <TraceRow key={item.id} item={item} />
          ))}
        </div>
      ) : null}
    </section>
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
    return (
      <div
        className={cn(
          "rounded-md border p-3 text-xs",
          item.ok ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50",
        )}
      >
        <div className="flex items-center gap-2 font-bold">
          {item.ok ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-700" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 text-rose-700" />
          )}
          Code {item.ok ? "completed" : "failed"}
        </div>
        {item.stdout ? <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap">{item.stdout}</pre> : null}
        {item.stderr || item.traceback ? (
          <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap text-rose-800">
            {item.stderr || item.traceback}
          </pre>
        ) : null}
        {item.updatedDatasets?.length ? (
          <p className="mt-2 font-semibold text-teal-800">
            Saved {item.updatedDatasets.length} dataset version
            {item.updatedDatasets.length === 1 ? "" : "s"}.
          </p>
        ) : null}
      </div>
    );
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
          <p className="text-sm font-bold text-amber-950">Confirmation required</p>
          <span className="mt-2 inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-emerald-800">
            <ShieldCheck className="h-3 w-3" />
            Safe mode
          </span>
          <p className="mt-2 text-sm leading-6 text-amber-950/80">{pending.message}</p>
          <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
            <RiskMetric label="Risk level" value={formatRisk(pending.riskLevel)} />
            <RiskMetric label="Operation" value={pending.operationSummary ?? pending.mutationSummary ?? "Mutation"} />
            <RiskMetric
              label="Affected data"
              value={
                pending.affectedDatasetIds?.length
                  ? `${pending.affectedDatasetIds.length} dataset${pending.affectedDatasetIds.length === 1 ? "" : "s"}`
                  : "Current dataset"
              }
            />
          </div>
          {pending.code ? (
            <pre className="mt-3 max-h-32 overflow-auto whitespace-pre-wrap rounded-md bg-slate-950 p-3 text-xs text-slate-100">
              {pending.code}
            </pre>
          ) : null}
          <div className="mt-4 flex gap-2">
            <Button onClick={onConfirm}>Apply change</Button>
            <Button variant="secondary" onClick={onCancel}>
              Cancel
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
  if (item.type === "error") {
    return "Error";
  }
  return item.type;
}
