import { Box, Columns3, Database, ListTree } from "lucide-react";

import { cn } from "../lib/utils";
import type { Dataset } from "../types/api";

type DatasetCardProps = {
  dataset: Dataset;
  active: boolean;
  onSelect: () => void;
};

export function DatasetCard({ dataset, active, onSelect }: DatasetCardProps) {
  const profile = dataset.profile;
  const metric = formatPrimaryMetric(profile.shape, profile.length);
  const columns = Array.isArray(profile.columns) ? profile.columns : [];
  const keys = Array.isArray(profile.keys) ? profile.keys : [];
  const previewItems = columns.length > 0 ? columns : keys;

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "w-full rounded-lg border p-3 text-left transition",
        active
          ? "border-teal-500 bg-teal-50 shadow-sm"
          : "border-border bg-white/70 hover:border-teal-300 hover:bg-white",
      )}
    >
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-white",
            active ? "bg-teal-700" : "bg-slate-950",
          )}
        >
          {columns.length > 0 ? (
            <Columns3 className="h-4 w-4" />
          ) : keys.length > 0 ? (
            <ListTree className="h-4 w-4" />
          ) : (
            <Box className="h-4 w-4" />
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center justify-between gap-2">
            <span className="truncate text-sm font-bold">{dataset.original_filename}</span>
            {active ? (
              <span className="rounded-full bg-teal-700 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white">
                Active
              </span>
            ) : null}
          </span>
          <span className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span>{dataset.object_type}</span>
            <span className="font-mono text-[11px] text-slate-500">{dataset.dataset_key}</span>
            {metric ? <span className="font-semibold text-slate-700">{metric}</span> : null}
          </span>
          {previewItems.length > 0 ? (
            <span className="mt-2 flex flex-wrap gap-1.5">
              {previewItems.slice(0, 4).map((item) => (
                <span
                  key={String(item)}
                  className="max-w-[8rem] truncate rounded-md border border-border bg-white px-2 py-1 text-[11px] font-semibold text-slate-700"
                >
                  {String(item)}
                </span>
              ))}
            </span>
          ) : (
            <span className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
              <Database className="h-3.5 w-3.5" />
              Generic object profile
            </span>
          )}
        </span>
      </div>
    </button>
  );
}

export function formatPrimaryMetric(shape: unknown, length?: number) {
  if (Array.isArray(shape) && shape.length > 0) {
    return `shape ${shape.map(String).join(" x ")}`;
  }

  if (typeof length === "number") {
    return `${length.toLocaleString()} items`;
  }

  return null;
}
