import { motion } from "framer-motion";
import { Check, GitBranch, GitFork, History, RotateCcw } from "lucide-react";

import { cn } from "../lib/utils";
import type { Branch, HistoryVersion } from "../types/api";
import { Button } from "./Button";

type BranchTimelineProps = {
  branches: Branch[];
  activeBranchId: string | null;
  versions: HistoryVersion[];
  selectedVersionId: string | null;
  busyAction: string | null;
  onSelectVersion: (versionId: string) => void;
  onCheckoutBranch: (branchId: string) => void;
  onRollbackVersion: (versionId: string) => void;
  onForkVersion: (versionId: string, name: string) => void;
};

export function BranchTimeline({
  branches,
  activeBranchId,
  versions,
  selectedVersionId,
  busyAction,
  onSelectVersion,
  onCheckoutBranch,
  onRollbackVersion,
  onForkVersion,
}: BranchTimelineProps) {
  const selected = versions.find((version) => version.id === selectedVersionId) ?? versions[versions.length - 1] ?? null;

  if (branches.length === 0 || versions.length === 0) {
    return <TimelineEmpty />;
  }

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        {branches.map((branch) => {
          const branchVersions = versions
            .filter((version) => version.branch_id === branch.id)
            .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
          const isActive = branch.id === activeBranchId;

          return (
            <section
              key={branch.id}
              className={cn(
                "rounded-lg border bg-white/70 p-3",
                isActive ? "border-teal-300 shadow-sm shadow-teal-900/5" : "border-border",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className={cn(
                      "flex h-7 w-7 shrink-0 items-center justify-center rounded-md",
                      isActive ? "bg-teal-700 text-white" : "bg-slate-100 text-slate-700",
                    )}
                  >
                    <GitBranch className="h-3.5 w-3.5" />
                  </span>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-bold">{branch.name}</p>
                    <p className="text-[11px] font-medium text-muted-foreground">
                      {branchVersions.length} version{branchVersions.length === 1 ? "" : "s"}
                    </p>
                  </div>
                </div>
                {isActive ? (
                  <span className="inline-flex h-7 items-center gap-1 rounded-md border border-teal-200 bg-teal-50 px-2 text-[11px] font-bold text-teal-900">
                    <Check className="h-3 w-3" />
                    Active
                  </span>
                ) : (
                  <Button
                    type="button"
                    variant="ghost"
                    className="h-7 px-2 text-xs"
                    disabled={busyAction === `checkout:${branch.id}`}
                    onClick={() => onCheckoutBranch(branch.id)}
                  >
                    Checkout
                  </Button>
                )}
              </div>

              <div className="mt-3 space-y-2">
                {branchVersions.map((version, index) => (
                  <motion.button
                    key={version.id}
                    type="button"
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.18, delay: index * 0.025 }}
                    onClick={() => onSelectVersion(version.id)}
                    className={cn(
                      "group grid w-full grid-cols-[18px_minmax(0,1fr)] gap-2 rounded-md border p-2 text-left transition",
                      selected?.id === version.id
                        ? "border-slate-950 bg-slate-950 text-white"
                        : "border-border bg-white/75 hover:border-teal-200 hover:bg-teal-50/70",
                    )}
                  >
                    <span className="relative mt-0.5 flex justify-center">
                      <span
                        className={cn(
                          "h-2.5 w-2.5 rounded-full",
                          version.is_current ? "bg-emerald-400" : "bg-slate-300",
                        )}
                      />
                      {index < branchVersions.length - 1 ? (
                        <span className="absolute top-3 h-9 w-px bg-border" />
                      ) : null}
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-xs font-bold">
                        {version.mutation_summary ?? version.label}
                      </span>
                      <span
                        className={cn(
                          "mt-1 block truncate text-[11px]",
                          selected?.id === version.id ? "text-white/70" : "text-muted-foreground",
                        )}
                      >
                        {version.dataset_filename ?? "Dataset"} - {formatDate(version.created_at)}
                      </span>
                    </span>
                  </motion.button>
                ))}
              </div>
            </section>
          );
        })}
      </div>

      {selected ? (
        <div className="rounded-lg border border-slate-200 bg-slate-50/85 p-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-muted-foreground">
                Selected version
              </p>
              <p className="mt-1 truncate text-sm font-bold">
                {selected.mutation_summary ?? selected.label}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {selected.branch_name ?? "branch"} - {selected.dataset_filename ?? "dataset"}
              </p>
            </div>
            {selected.is_current ? (
              <span className="rounded-md bg-emerald-50 px-2 py-1 text-[11px] font-bold text-emerald-800">
                Current
              </span>
            ) : null}
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Button
              type="button"
              variant="secondary"
              className="h-8 px-2 text-xs"
              disabled={busyAction === `rollback:${selected.id}`}
              onClick={() => onRollbackVersion(selected.id)}
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Rollback
            </Button>
            <Button
              type="button"
              variant="secondary"
              className="h-8 px-2 text-xs"
              disabled={busyAction === `fork:${selected.id}`}
              onClick={() => {
                const name = window.prompt("Name this branch", `branch-${versions.length + 1}`);
                if (name) {
                  onForkVersion(selected.id, name);
                }
              }}
            >
              <GitFork className="h-3.5 w-3.5" />
              Fork
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function TimelineEmpty() {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white/64 p-4">
      <div className="flex items-center gap-2 text-sm font-bold">
        <History className="h-4 w-4 text-teal-700" />
        No version history yet
      </div>
      <p className="mt-2 text-xs leading-5 text-muted-foreground">
        Upload a pickle file to create the first version node on main.
      </p>
    </div>
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
