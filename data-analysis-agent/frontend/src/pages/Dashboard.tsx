import { motion } from "framer-motion";
import {
  Activity,
  Bot,
  CheckCircle2,
  Clock3,
  Database,
  FileSearch,
  GitBranch,
  Layers3,
  Loader2,
  MessageSquareText,
  PanelRight,
  Plus,
  Send,
  Sparkles,
  Upload,
} from "lucide-react";

import { Button } from "../components/Button";
import { CollapsibleCard } from "../components/CollapsibleCard";
import { DatasetCard, formatPrimaryMetric } from "../components/DatasetCard";
import { Panel } from "../components/Panel";
import { ProfileInspector } from "../components/ProfileInspector";
import { UploadDropzone } from "../components/UploadDropzone";
import { useHealth } from "../hooks/useHealth";
import { useWorkspace } from "../hooks/useWorkspace";

const traceItems = [
  {
    label: "Session bootstrapped",
    detail: "A workspace session is created automatically when the app opens.",
    tone: "teal",
  },
  {
    label: "Dataset profiler ready",
    detail: "Uploads are inspected generically without schema assumptions.",
    tone: "indigo",
  },
  {
    label: "Agent standby",
    detail: "Streaming trace events will appear here in the next phase.",
    tone: "rose",
  },
];

export function Dashboard() {
  const health = useHealth();
  const workspace = useWorkspace();
  const isOnline = health.status === "online";
  const branch = workspace.session?.branches[0] ?? null;

  return (
    <main className="min-h-screen p-4 text-foreground sm:p-6 lg:p-8">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-[1540px] flex-col gap-5">
        <header className="flex flex-col gap-4 rounded-lg border border-white/80 bg-white/64 px-5 py-4 shadow-glow backdrop-blur-xl md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-teal-700 text-white shadow-lg shadow-teal-900/15">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">AI data workspace</p>
              <h1 className="text-2xl font-bold tracking-normal">Data Analysis Agent</h1>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <StatusPill
              label={`API ${health.status === "loading" ? "checking" : isOnline ? "online" : "offline"}`}
              status={isOnline ? "ok" : health.status === "loading" ? "loading" : "warn"}
            />
            <StatusPill
              label={
                workspace.sessionStatus === "ready"
                  ? "Session ready"
                  : workspace.sessionStatus === "loading"
                    ? "Creating session"
                    : "Session error"
              }
              status={
                workspace.sessionStatus === "ready"
                  ? "ok"
                  : workspace.sessionStatus === "loading"
                    ? "loading"
                    : "warn"
              }
            />
            <Button variant="secondary" disabled={!workspace.session}>
              <Upload className="h-4 w-4" />
              Upload ready
            </Button>
            <Button disabled>
              <Plus className="h-4 w-4" />
              New branch
            </Button>
          </div>
        </header>

        <div className="grid flex-1 gap-5 xl:grid-cols-[330px_minmax(0,1fr)_390px]">
          <Panel className="flex max-h-[calc(100vh-8.5rem)] flex-col overflow-hidden">
            <div className="border-b border-border/80 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Workspace
              </p>
              <h2 className="mt-2 text-lg font-bold">Session and datasets</h2>
              {workspace.sessionError ? (
                <p className="mt-2 text-xs font-semibold text-rose-700">{workspace.sessionError}</p>
              ) : null}
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              <UploadDropzone
                disabled={!workspace.session || workspace.upload.status === "uploading"}
                upload={workspace.upload}
                onUpload={workspace.uploadFiles}
              />

              <CollapsibleCard
                title="Uploaded datasets"
                eyebrow={`${workspace.datasets.length} total`}
                icon={<Layers3 className="h-4 w-4" />}
              >
                {workspace.datasets.length === 0 ? (
                  <EmptyDatasets />
                ) : (
                  <div className="space-y-2">
                    {workspace.datasets.map((dataset) => (
                      <DatasetCard
                        key={dataset.id}
                        dataset={dataset}
                        active={workspace.activeDataset?.id === dataset.id}
                        onSelect={() => workspace.setActiveDatasetId(dataset.id)}
                      />
                    ))}
                  </div>
                )}
              </CollapsibleCard>

              <CollapsibleCard
                title={workspace.session?.name ?? "Creating workspace"}
                eyebrow="Session"
                icon={
                  workspace.sessionStatus === "loading" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Database className="h-4 w-4" />
                  )
                }
              >
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <MiniMetric label="Datasets" value={String(workspace.datasets.length)} />
                  <MiniMetric label="Branches" value={String(workspace.session?.branches.length ?? 0)} />
                </div>
                <p className="mt-3 break-all rounded-md bg-slate-50 p-2 text-[11px] font-medium text-muted-foreground">
                  {workspace.session?.id ?? "Waiting for backend session..."}
                </p>
              </CollapsibleCard>

              <CollapsibleCard
                title={branch?.name ?? "main"}
                eyebrow="Active branch"
                icon={<GitBranch className="h-4 w-4" />}
              >
                <div className="space-y-3">
                  <div className="rounded-md border border-teal-200 bg-teal-50 p-3">
                    <p className="text-xs font-bold text-teal-950">main</p>
                    <p className="mt-1 text-xs text-teal-900/70">Initial version nodes land here.</p>
                  </div>
                  <div className="space-y-2">
                    {workspace.datasets.length === 0 ? (
                      <TimelineEmpty />
                    ) : (
                      workspace.datasets.slice(0, 4).map((dataset) => (
                        <div key={dataset.id} className="flex gap-2 text-xs">
                          <span className="mt-1.5 h-2 w-2 rounded-full bg-teal-600" />
                          <span className="min-w-0">
                            <span className="block truncate font-semibold">{dataset.original_filename}</span>
                            <span className="text-muted-foreground">{dataset.current_version.label}</span>
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </CollapsibleCard>
            </div>
          </Panel>

          <Panel className="flex min-h-[720px] flex-col overflow-hidden">
            <div className="border-b border-border/80 p-5">
              <div className="flex flex-col gap-4 2xl:flex-row 2xl:items-center 2xl:justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">
                    {workspace.activeDataset
                      ? `Active dataset: ${workspace.activeDataset.original_filename}`
                      : "Waiting for a dataset"}
                  </p>
                  <h2 className="mt-1 text-2xl font-bold">Chat with your data runtime</h2>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-semibold text-muted-foreground">
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    {workspace.datasets.length} datasets
                  </div>
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    {branch?.name ?? "main"} branch
                  </div>
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    SSE queued
                  </div>
                </div>
              </div>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto p-5">
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45 }}
                className="max-w-3xl rounded-lg border border-teal-100 bg-teal-50/80 p-4"
              >
                <div className="flex items-center gap-2 text-sm font-bold text-teal-900">
                  <Sparkles className="h-4 w-4" />
                  Ready for arbitrary Python objects
                </div>
                <p className="mt-2 text-sm leading-6 text-teal-950/75">
                  Upload trusted pickle files to create initial dataset versions. Profiles can be
                  tabular, nested, array-like, or custom object shaped.
                </p>
              </motion.div>

              {workspace.activeDataset ? (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-lg border border-slate-200 bg-white/76 p-4"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 text-sm font-bold">
                        <FileSearch className="h-4 w-4 text-teal-700" />
                        Profile loaded
                      </div>
                      <p className="mt-2 truncate text-sm text-muted-foreground">
                        {workspace.activeDataset.object_type}
                        {formatPrimaryMetric(
                          workspace.activeDataset.profile.shape,
                          workspace.activeDataset.profile.length,
                        )
                          ? ` · ${formatPrimaryMetric(
                              workspace.activeDataset.profile.shape,
                              workspace.activeDataset.profile.length,
                            )}`
                          : ""}
                      </p>
                    </div>
                    <span className="rounded-full bg-slate-950 px-3 py-1 text-xs font-bold text-white">
                      Dataset
                    </span>
                  </div>
                </motion.div>
              ) : null}

              <div className="space-y-3">
                {traceItems.map((item, index) => (
                  <TraceEvent key={item.label} item={item} index={index} />
                ))}
              </div>

              <div className="rounded-lg border-2 border-teal-600 bg-white p-5 shadow-lg shadow-teal-900/10">
                <div className="flex items-center gap-2 text-sm font-bold text-teal-800">
                  <CheckCircle2 className="h-4 w-4" />
                  Final answer
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">
                  Agent conclusions, charts, and export links will render here separately from trace
                  events. For now, the uploaded object profile is available in the inspector.
                </p>
              </div>
            </div>

            <div className="border-t border-border/80 bg-white/60 p-4">
              <div className="flex items-end gap-3 rounded-lg border border-border bg-white p-3 shadow-sm">
                <textarea
                  className="min-h-16 flex-1 resize-none border-0 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                  placeholder={
                    workspace.activeDataset
                      ? "Ask the agent about this dataset in the next phase..."
                      : "Upload a dataset to start the analysis workspace..."
                  }
                  disabled
                />
                <Button className="h-11 w-11 px-0" aria-label="Send message" disabled>
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </Panel>

          <Panel className="flex max-h-[calc(100vh-8.5rem)] flex-col overflow-hidden">
            <div className="border-b border-border/80 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Inspector
              </p>
              <h2 className="mt-2 text-lg font-bold">Dataset profile</h2>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              <div className="rounded-lg border border-border bg-white/72 p-4">
                <div className="flex items-center gap-2 text-sm font-bold">
                  <Activity className="h-4 w-4 text-teal-700" />
                  API status
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  {health.status === "offline"
                    ? health.error
                    : health.status === "loading"
                      ? "Checking FastAPI health endpoint"
                      : `${health.data.service} responded ok`}
                </p>
              </div>

              <ProfileInspector dataset={workspace.activeDataset} />

              <CollapsibleCard
                title="Artifacts"
                eyebrow="Coming next"
                icon={<PanelRight className="h-4 w-4" />}
                defaultOpen={false}
              >
                <div className="rounded-lg border border-dashed border-slate-300 bg-white/70 p-4 text-sm text-muted-foreground">
                  Charts, exported CSVs, generated figures, and branch snapshots will appear here.
                </div>
              </CollapsibleCard>
            </div>
          </Panel>
        </div>
      </div>
    </main>
  );
}

function StatusPill({ label, status }: { label: string; status: "ok" | "loading" | "warn" }) {
  return (
    <div className="flex h-10 items-center gap-2 rounded-md border border-border bg-white/78 px-3 text-sm font-medium">
      <span
        className={`h-2.5 w-2.5 rounded-full ${
          status === "ok" ? "bg-emerald-500" : status === "loading" ? "bg-amber-500" : "bg-rose-500"
        }`}
      />
      {label}
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-white/78 p-2">
      <p className="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-sm font-bold">{value}</p>
    </div>
  );
}

function TraceEvent({
  item,
  index,
}: {
  item: { label: string; detail: string; tone: string };
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.08 * index, duration: 0.35 }}
      className="flex items-start gap-3 rounded-lg border border-border bg-white/68 p-4"
    >
      <span
        className={`mt-1 h-2.5 w-2.5 rounded-full ${
          item.tone === "teal"
            ? "bg-teal-500"
            : item.tone === "indigo"
              ? "bg-indigo-500"
              : "bg-rose-500"
        }`}
      />
      <div>
        <p className="text-sm font-semibold">{item.label}</p>
        <p className="mt-1 text-sm text-muted-foreground">{item.detail}</p>
      </div>
    </motion.div>
  );
}

function EmptyDatasets() {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white/64 p-5 text-center">
      <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-lg bg-slate-950 text-white">
        <MessageSquareText className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm font-bold">No datasets yet</p>
      <p className="mt-2 text-xs leading-5 text-muted-foreground">
        Upload one or more pickle files to populate the workspace.
      </p>
    </div>
  );
}

function TimelineEmpty() {
  return (
    <div className="flex gap-2 text-xs text-muted-foreground">
      <Clock3 className="mt-0.5 h-3.5 w-3.5" />
      <span>Initial version timeline appears after upload.</span>
    </div>
  );
}
