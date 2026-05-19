import { useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Database,
  FileDown,
  GitBranch,
  GitFork,
  Layers3,
  Loader2,
  MessageSquareText,
  PanelRight,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

import { Button } from "../components/Button";
import { ArtifactCard } from "../components/ArtifactCard";
import { BranchTimeline } from "../components/BranchTimeline";
import { ChatExportCard } from "../components/ChatExportCard";
import { ChatInput } from "../components/ChatInput";
import { ChatThread } from "../components/ChatThread";
import { CollapsibleCard } from "../components/CollapsibleCard";
import { DatasetCard, formatPrimaryMetric } from "../components/DatasetCard";
import { Panel } from "../components/Panel";
import { ProfileInspector } from "../components/ProfileInspector";
import { UploadDropzone } from "../components/UploadDropzone";
import { useChat } from "../hooks/useChat";
import { useHealth } from "../hooks/useHealth";
import { useWorkspace } from "../hooks/useWorkspace";

type ToastMessage = {
  id: string;
  tone: "success" | "error";
  title: string;
  message: string;
};

type WorkspaceView = "chat" | "explore";

export function Dashboard() {
  const [view, setView] = useState<WorkspaceView>("chat");
  const [dismissedToastIds, setDismissedToastIds] = useState<Set<string>>(() => new Set());
  const health = useHealth();
  const workspace = useWorkspace();
  const chat = useChat({
    sessionId: workspace.session?.id,
    activeDatasetId: workspace.activeDataset?.id,
    branchName: branchName(workspace.activeBranch?.name),
    onStateChanged: workspace.refreshWorkspace,
  });
  const isOnline = health.status === "online";
  const branch = workspace.activeBranch;
  const artifacts = [...workspace.exportArtifacts, ...chat.artifacts];
  const toasts = [
    workspace.sessionError
      ? { id: `session:${workspace.sessionError}`, tone: "error" as const, title: "Session error", message: workspace.sessionError }
      : null,
    workspace.upload.status === "error" && workspace.upload.message
      ? { id: `upload-error:${workspace.upload.message}`, tone: "error" as const, title: "Upload failed", message: workspace.upload.message }
      : null,
    workspace.upload.status === "success" && workspace.upload.message
      ? { id: `upload-success:${workspace.upload.message}`, tone: "success" as const, title: "Upload complete", message: workspace.upload.message }
      : null,
    workspace.exportStatus === "error" && workspace.exportMessage
      ? { id: `export-error:${workspace.exportMessage}`, tone: "error" as const, title: "Export failed", message: workspace.exportMessage }
      : null,
    workspace.exportStatus === "success" && workspace.exportMessage
      ? { id: `export-success:${workspace.exportMessage}`, tone: "success" as const, title: "CSV ready", message: workspace.exportMessage }
      : null,
  ].filter((toast): toast is ToastMessage => toast !== null);
  const visibleToasts = toasts.filter((toast) => !dismissedToastIds.has(toast.id));
  const dismissToast = (id: string) => {
    setDismissedToastIds((current) => {
      const next = new Set(current);
      next.add(id);
      return next;
    });
  };

  return (
    <main className="min-h-screen p-4 text-foreground sm:p-6 lg:p-8">
      <ToastStack toasts={visibleToasts} onDismiss={dismissToast} />
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
            <StatusPill
              label="Safe mode on"
              status="ok"
              icon={<ShieldCheck className="h-3.5 w-3.5 text-emerald-700" />}
            />
            <ViewSwitcher value={view} onChange={setView} />
            <Button variant="secondary" disabled={!workspace.session}>
              <Upload className="h-4 w-4" />
              Upload ready
            </Button>
            <Button
              variant="secondary"
              disabled={!workspace.activeDataset || workspace.exportStatus === "exporting"}
              onClick={() => void workspace.exportCurrentDataset()}
            >
              <FileDown className="h-4 w-4" />
              {workspace.exportStatus === "exporting" ? "Exporting" : "Export CSV"}
            </Button>
            <Button
              disabled={!workspace.session}
              onClick={() => {
                const name = window.prompt("Name this branch", `branch-${(workspace.session?.branches.length ?? 0) + 1}`);
                if (name) {
                  void workspace.createBranch(name, workspace.selectedVersion?.id ?? null);
                }
              }}
            >
              <GitFork className="h-4 w-4" />
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
                <BranchTimeline
                  branches={workspace.session?.branches ?? []}
                  activeBranchId={workspace.session?.active_branch_id ?? null}
                  versions={workspace.history}
                  selectedVersionId={workspace.selectedVersionId}
                  busyAction={workspace.historyAction}
                  onSelectVersion={workspace.setSelectedVersionId}
                  onCheckoutBranch={(branchId) => void workspace.checkoutBranch(branchId)}
                  onRollbackVersion={(versionId) => void workspace.rollbackVersion(versionId)}
                  onForkVersion={(versionId, name) => void workspace.forkVersion(versionId, name)}
                />
              </CollapsibleCard>
            </div>
          </Panel>

          <Panel className="flex min-h-[720px] flex-col overflow-hidden">
            <div className="border-b border-border/80 p-5">
              <div className="flex flex-col gap-4 2xl:flex-row 2xl:items-center 2xl:justify-between">
                <div>
                  <p className="max-w-2xl break-words text-sm font-medium text-muted-foreground">
                    {workspace.activeDataset
                      ? `Active dataset: ${workspace.activeDataset.original_filename} (${workspace.activeDataset.dataset_key})`
                      : "Waiting for a dataset"}
                  </p>
                  <h2 className="mt-1 text-2xl font-bold">
                    {view === "chat" ? "Chat with your data runtime" : "Explore dataset profiles"}
                  </h2>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-semibold text-muted-foreground">
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    {workspace.datasets.length} datasets
                  </div>
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    {branch?.name ?? "main"} branch
                  </div>
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    {chat.isStreaming ? "Streaming" : "SSE ready"}
                  </div>
                </div>
              </div>
            </div>

            {view === "chat" ? (
              <>
                <div className="flex-1 space-y-4 overflow-y-auto p-5">
                  <motion.div
                    key="chat-intro"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
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

                  <ChatThread
                    sessionId={workspace.session?.id}
                    messages={chat.messages}
                    pendingConfirmation={chat.pendingConfirmation}
                    onConfirm={chat.confirmPending}
                    onCancel={chat.cancelPending}
                  />
                </div>

                <div className="border-t border-border/80 bg-white/60 p-4">
                  <ChatInput
                    disabled={!workspace.session || !workspace.activeDataset}
                    streaming={chat.isStreaming}
                    onSend={(message) => void chat.sendMessage(message)}
                    onStop={chat.stop}
                  />
                </div>
              </>
            ) : (
              <div className="flex-1 space-y-4 overflow-y-auto p-5">
                <motion.div
                  key="explore-intro"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                  className="rounded-lg border border-indigo-100 bg-indigo-50/70 p-4"
                >
                  <div className="flex items-center gap-2 text-sm font-bold text-indigo-950">
                    <PanelRight className="h-4 w-4" />
                    Dataset explorer
                  </div>
                  <p className="mt-2 text-sm leading-6 text-indigo-950/70">
                    Profiles, previews, generated charts, tables, CSV artifacts, and selected history
                    nodes live here so the chat surface can stay focused.
                  </p>
                </motion.div>

                <ProfileInspector dataset={workspace.activeDataset} />
              </div>
            )}
          </Panel>

          <Panel className="flex max-h-[calc(100vh-8.5rem)] flex-col overflow-hidden">
            <div className="border-b border-border/80 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                {view === "chat" ? "Context" : "Artifacts"}
              </p>
              <h2 className="mt-2 text-lg font-bold">
                {view === "chat" ? "Run context" : "Exports and charts"}
              </h2>
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

              {view === "chat" ? (
                <>
                  <div className="rounded-lg border border-border bg-white/72 p-4">
                    <div className="flex items-center gap-2 text-sm font-bold">
                      <Database className="h-4 w-4 text-teal-700" />
                      Active dataset
                    </div>
                    {workspace.activeDataset ? (
                      <div className="mt-3 space-y-3">
                        <p className="break-words text-sm font-bold">
                          {workspace.activeDataset.original_filename}
                        </p>
                        <p className="break-all rounded-md bg-slate-50 p-2 font-mono text-[11px] leading-5 text-slate-600">
                          {workspace.activeDataset.dataset_key}
                        </p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <MiniMetric label="Type" value={workspace.activeDataset.object_type} />
                          <MiniMetric
                            label="Size"
                            value={
                              formatPrimaryMetric(
                                workspace.activeDataset.profile.shape,
                                workspace.activeDataset.profile.length,
                              ) ?? "unknown"
                            }
                          />
                        </div>
                      </div>
                    ) : (
                      <p className="mt-2 text-sm leading-5 text-muted-foreground">
                        Upload a trusted pickle file to start the runtime.
                      </p>
                    )}
                  </div>

                  <div className="rounded-lg border border-border bg-white/72 p-4">
                    <div className="flex items-center gap-2 text-sm font-bold">
                      <GitBranch className="h-4 w-4 text-teal-700" />
                      Branch focus
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <MiniMetric label="Current" value={branch?.name ?? "main"} />
                      <MiniMetric label="History" value={`${workspace.history.length} versions`} />
                    </div>
                    <p className="mt-3 text-xs leading-5 text-muted-foreground">
                      Timeline controls stay in the left panel. Rollbacks ask for confirmation before
                      changing state.
                    </p>
                  </div>

                  <ChatExportCard
                    messages={chat.messages}
                    sessionName={workspace.session?.name}
                    activeDatasetName={workspace.activeDataset?.original_filename}
                    branchName={branch?.name}
                  />

                  <div className="rounded-lg border border-indigo-100 bg-indigo-50/70 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2 text-sm font-bold text-indigo-950">
                          <PanelRight className="h-4 w-4" />
                          Explore view
                        </div>
                        <p className="mt-2 text-sm leading-5 text-indigo-950/70">
                          Dataset profiles, charts, CSV exports, and artifact previews are separated
                          from the chat for a cleaner demo surface.
                        </p>
                      </div>
                      <Button
                        type="button"
                        variant="secondary"
                        className="h-8 shrink-0 px-3 text-xs"
                        onClick={() => setView("explore")}
                      >
                        Open
                      </Button>
                    </div>
                    <p className="mt-3 text-xs font-semibold text-indigo-950/70">
                      {artifacts.length} generated artifact{artifacts.length === 1 ? "" : "s"}
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div className="rounded-lg border border-border bg-white/72 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2 text-sm font-bold">
                          <FileDown className="h-4 w-4 text-teal-700" />
                          CSV export
                        </div>
                        <p className="mt-2 text-sm leading-5 text-muted-foreground">
                          Export the active dataset exactly as it exists on the current branch.
                        </p>
                      </div>
                      <Button
                        type="button"
                        variant="secondary"
                        className="h-8 px-3 text-xs"
                        disabled={!workspace.activeDataset || workspace.exportStatus === "exporting"}
                        onClick={() => void workspace.exportCurrentDataset()}
                      >
                        {workspace.exportStatus === "exporting" ? "Working" : "Export"}
                      </Button>
                    </div>
                    {workspace.exportMessage ? (
                      <p
                        className={`mt-3 rounded-md p-2 text-xs font-semibold ${
                          workspace.exportStatus === "error"
                            ? "bg-rose-50 text-rose-700"
                            : "bg-teal-50 text-teal-900"
                        }`}
                      >
                        {workspace.exportMessage}
                      </p>
                    ) : null}
                  </div>

                  {workspace.selectedVersion ? (
                    <CollapsibleCard
                      title={workspace.selectedVersion.mutation_summary ?? workspace.selectedVersion.label}
                      eyebrow="Selected version"
                      icon={<GitBranch className="h-4 w-4" />}
                      defaultOpen={false}
                    >
                      <div className="space-y-2 text-xs">
                        <MiniMetric label="Branch" value={workspace.selectedVersion.branch_name ?? "unknown"} />
                        <MiniMetric
                          label="Dataset"
                          value={workspace.selectedVersion.dataset_filename ?? "unknown"}
                        />
                        <p className="break-all rounded-md bg-slate-50 p-2 font-medium text-muted-foreground">
                          {workspace.selectedVersion.id}
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                          <Button
                            type="button"
                            variant="secondary"
                            className="h-8 px-2 text-xs"
                            onClick={() => void workspace.rollbackVersion(workspace.selectedVersion!.id)}
                          >
                            Rollback
                          </Button>
                          <Button
                            type="button"
                            variant="secondary"
                            className="h-8 px-2 text-xs"
                            onClick={() => {
                              const name = window.prompt(
                                "Name this branch",
                                `branch-${(workspace.session?.branches.length ?? 0) + 1}`,
                              );
                              if (name && workspace.selectedVersion) {
                                void workspace.forkVersion(workspace.selectedVersion.id, name);
                              }
                            }}
                          >
                            Fork
                          </Button>
                        </div>
                      </div>
                    </CollapsibleCard>
                  ) : null}

                  <CollapsibleCard
                    title="Artifacts"
                    eyebrow={`${artifacts.length} generated`}
                    icon={<PanelRight className="h-4 w-4" />}
                    defaultOpen={artifacts.length > 0}
                  >
                    {artifacts.length > 0 && workspace.session ? (
                      <div className="space-y-3">
                        {artifacts.map((artifact) =>
                          workspace.session ? (
                            <ArtifactCard
                              key={artifact.id}
                              sessionId={workspace.session.id}
                              artifact={artifact}
                            />
                          ) : null,
                        )}
                      </div>
                    ) : (
                      <div className="rounded-lg border border-dashed border-slate-300 bg-white/70 p-4 text-sm text-muted-foreground">
                        Tables, charts, and CSV exports created by the agent will appear here.
                      </div>
                    )}
                  </CollapsibleCard>
                </>
              )}
            </div>
          </Panel>
        </div>
      </div>
    </main>
  );
}

function StatusPill({
  label,
  status,
  icon,
}: {
  label: string;
  status: "ok" | "loading" | "warn";
  icon?: ReactNode;
}) {
  return (
    <div className="flex h-10 items-center gap-2 rounded-md border border-border bg-white/78 px-3 text-sm font-medium">
      <span
        className={`h-2.5 w-2.5 rounded-full ${
          status === "ok" ? "bg-emerald-500" : status === "loading" ? "bg-amber-500" : "bg-rose-500"
        }`}
      />
      {icon}
      {label}
    </div>
  );
}

function ViewSwitcher({
  value,
  onChange,
}: {
  value: WorkspaceView;
  onChange: (value: WorkspaceView) => void;
}) {
  const options: Array<{ value: WorkspaceView; label: string; icon: ReactNode }> = [
    { value: "chat", label: "Chat", icon: <MessageSquareText className="h-3.5 w-3.5" /> },
    { value: "explore", label: "Explore", icon: <PanelRight className="h-3.5 w-3.5" /> },
  ];

  return (
    <div className="flex rounded-md border border-border bg-white/70 p-1">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`relative flex h-8 items-center gap-1.5 rounded px-3 text-xs font-bold transition ${
            value === option.value ? "text-teal-950" : "text-muted-foreground hover:text-slate-900"
          }`}
        >
          {value === option.value ? (
            <motion.span
              layoutId="workspace-view-indicator"
              className="absolute inset-0 rounded bg-teal-100 shadow-sm"
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            />
          ) : null}
          <span className="relative z-10">{option.icon}</span>
          <span className="relative z-10">{option.label}</span>
        </button>
      ))}
    </div>
  );
}

function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}) {
  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="fixed right-4 top-4 z-50 w-[min(360px,calc(100vw-2rem))] space-y-2">
      {toasts.map((toast) => (
        <motion.div
          key={toast.id}
          initial={{ opacity: 0, x: 18, scale: 0.98 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          className={`rounded-lg border bg-white/95 p-4 shadow-xl backdrop-blur ${
            toast.tone === "success" ? "border-emerald-200" : "border-rose-200"
          }`}
        >
          <div className="flex items-start gap-3">
            <span
              className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md ${
                toast.tone === "success" ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700"
              }`}
            >
              {toast.tone === "success" ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertCircle className="h-4 w-4" />
              )}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-bold text-slate-950">{toast.title}</p>
              <p className="mt-1 max-h-16 overflow-hidden text-sm leading-5 text-slate-600">{toast.message}</p>
            </div>
            <button
              type="button"
              aria-label="Dismiss notification"
              className="rounded-md p-1 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
              onClick={() => onDismiss(toast.id)}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </motion.div>
      ))}
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

function branchName(value?: string) {
  return value ?? "main";
}
