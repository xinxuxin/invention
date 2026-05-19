import { useEffect, useState } from "react";
import { CheckCircle2, Download, FileText, ListChecks } from "lucide-react";

import { Button } from "./Button";
import type { ChatMessage } from "../hooks/useChat";
import { downloadChatTranscript } from "../lib/chatExport";

type ChatExportCardProps = {
  messages: ChatMessage[];
  sessionName?: string | null;
  activeDatasetName?: string | null;
  branchName?: string | null;
};

export function ChatExportCard({
  messages,
  sessionName,
  activeDatasetName,
  branchName,
}: ChatExportCardProps) {
  const [includeTrace, setIncludeTrace] = useState(false);
  const [exported, setExported] = useState(false);
  const exportableMessages = messages.filter(
    (message) => message.role === "user" || message.finalAnswer || message.trace.length > 0,
  );
  const traceCount = messages.reduce((total, message) => total + message.trace.length, 0);
  const canExport = exportableMessages.length > 0;

  useEffect(() => {
    if (!exported) {
      return;
    }
    const timer = window.setTimeout(() => setExported(false), 2200);
    return () => window.clearTimeout(timer);
  }, [exported]);

  return (
    <div className="rounded-lg border border-border bg-white/72 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-bold">
            <FileText className="h-4 w-4 text-teal-700" />
            Export chat
          </div>
          <p className="mt-2 text-sm leading-5 text-muted-foreground">
            Download the current browser chat transcript as Markdown.
          </p>
        </div>
        {exported ? (
          <span className="flex shrink-0 items-center gap-1 rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-bold uppercase tracking-wide text-emerald-800">
            <CheckCircle2 className="h-3 w-3" />
            Saved
          </span>
        ) : null}
      </div>

      <label className="mt-4 flex cursor-pointer items-start gap-3 rounded-md border border-border bg-slate-50/80 p-3">
        <input
          type="checkbox"
          className="mt-1 h-4 w-4 rounded border-slate-300 text-teal-700 focus:ring-teal-600"
          checked={includeTrace}
          onChange={(event) => setIncludeTrace(event.target.checked)}
        />
        <span className="min-w-0">
          <span className="flex items-center gap-1.5 text-sm font-bold">
            <ListChecks className="h-3.5 w-3.5 text-indigo-700" />
            Include trace events
          </span>
          <span className="mt-1 block text-xs leading-5 text-muted-foreground">
            Adds streamed progress, generated Python, stdout, stderr, tracebacks, and artifact summaries.
          </span>
        </span>
      </label>

      <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
        <MiniExportMetric label="Messages" value={String(exportableMessages.length)} />
        <MiniExportMetric label="Trace" value={String(traceCount)} />
      </div>

      <Button
        type="button"
        variant="secondary"
        className="mt-4 w-full"
        disabled={!canExport}
        onClick={() => {
          downloadChatTranscript(messages, {
            includeTrace,
            sessionName,
            activeDatasetName,
            branchName,
          });
          setExported(true);
        }}
      >
        <Download className="h-4 w-4" />
        {canExport ? "Download Markdown" : "No chat yet"}
      </Button>
    </div>
  );
}

function MiniExportMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-white/78 p-2">
      <p className="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm font-bold">{value}</p>
    </div>
  );
}
