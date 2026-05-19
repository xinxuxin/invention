import { useMemo } from "react";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
} from "@tanstack/react-table";
import { AlertTriangle, Braces, Columns3, FileText, Layers3, Table2 } from "lucide-react";

import { CollapsibleCard } from "./CollapsibleCard";
import { formatPrimaryMetric } from "./DatasetCard";
import type { Dataset } from "../types/api";

type ProfileInspectorProps = {
  dataset: Dataset | null;
};

export function ProfileInspector({ dataset }: ProfileInspectorProps) {
  if (!dataset) {
    return <InspectorEmptyState />;
  }

  const profile = dataset.profile;
  const metric = formatPrimaryMetric(profile.shape, profile.length);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-teal-200 bg-teal-50/75 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-bold text-teal-950">{dataset.original_filename}</p>
            <p className="mt-1 text-xs text-teal-900/70">{profile.module ?? "unknown module"}</p>
          </div>
          <span className="rounded-full bg-teal-700 px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide text-white">
            {dataset.object_type}
          </span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <Metric label="Type" value={profile.object_type} />
          <Metric label="Size" value={formatBytes(profile.approximate_size)} />
          <Metric label="Shape" value={metric ?? "n/a"} />
          <Metric label="Version" value={dataset.current_version.label} />
        </div>
      </div>

      {profile.warnings?.length ? (
        <CollapsibleCard
          title="Profile warnings"
          icon={<AlertTriangle className="h-4 w-4 text-amber-300" />}
        >
          <div className="space-y-2">
            {profile.warnings.map((warning) => (
              <p key={warning} className="rounded-md bg-amber-50 p-2 text-xs text-amber-900">
                {warning}
              </p>
            ))}
          </div>
        </CollapsibleCard>
      ) : null}

      <CollapsibleCard title="Schema" icon={<Columns3 className="h-4 w-4" />}>
        <SchemaBlock dataset={dataset} />
      </CollapsibleCard>

      <CollapsibleCard title="Sample Preview" icon={<Table2 className="h-4 w-4" />}>
        <SamplePreview dataset={dataset} />
      </CollapsibleCard>

      <CollapsibleCard
        title="Nested Structure"
        icon={<Layers3 className="h-4 w-4" />}
        defaultOpen={!profile.sample_rows?.length}
      >
        <NestedTree value={profile.nested_summary ?? profile} />
      </CollapsibleCard>

      <CollapsibleCard title="Repr" icon={<FileText className="h-4 w-4" />} defaultOpen={false}>
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">
          {profile.repr_preview ?? "No repr preview available."}
        </pre>
      </CollapsibleCard>
    </div>
  );
}

function SchemaBlock({ dataset }: { dataset: Dataset }) {
  const profile = dataset.profile;
  const columns = Array.isArray(profile.columns) ? profile.columns : [];
  const dtypes = profile.dtypes ?? {};
  const keys = Array.isArray(profile.keys) ? profile.keys : [];
  const publicAttributes = profile.public_attributes ?? {};

  if (columns.length > 0) {
    return (
      <div className="space-y-2">
        {columns.slice(0, 20).map((column) => (
          <div
            key={String(column)}
            className="flex items-center justify-between gap-3 rounded-md border border-border bg-white px-3 py-2"
          >
            <span className="truncate text-xs font-semibold">{String(column)}</span>
            <span className="shrink-0 rounded-md bg-slate-100 px-2 py-1 text-[11px] font-semibold text-slate-600">
              {dtypes[String(column)] ?? "unknown"}
            </span>
          </div>
        ))}
      </div>
    );
  }

  if (keys.length > 0) {
    return <TokenList label="Keys" values={keys} />;
  }

  if (Object.keys(publicAttributes).length > 0) {
    return <TokenList label="Public attributes" values={Object.keys(publicAttributes)} />;
  }

  return <p className="text-sm text-muted-foreground">No tabular schema was detected.</p>;
}

function SamplePreview({ dataset }: { dataset: Dataset }) {
  const profile = dataset.profile;

  if (Array.isArray(profile.sample_rows) && profile.sample_rows.length > 0) {
    return <SampleTable rows={profile.sample_rows} />;
  }

  const values = profile.sample_items ?? (Array.isArray(profile.sample) ? profile.sample : [profile.sample]);
  const filtered = values.filter((value) => value !== undefined);

  if (filtered.length === 0) {
    return <p className="text-sm text-muted-foreground">No sample preview is available.</p>;
  }

  return (
    <div className="space-y-2">
      {filtered.slice(0, 6).map((value, index) => (
        <pre
          key={index}
          className="max-h-36 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-white p-3 text-xs leading-5 text-slate-700"
        >
          {formatJson(value)}
        </pre>
      ))}
    </div>
  );
}

function SampleTable({ rows }: { rows: Record<string, unknown>[] }) {
  const columns = useMemo<ColumnDef<Record<string, unknown>>[]>(() => {
    const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
    return keys.map((key) => ({
      accessorKey: key,
      header: key,
      cell: ({ getValue }) => <span>{formatCell(getValue())}</span>,
    }));
  }, [rows]);

  const table = useReactTable({
    data: rows,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="overflow-hidden rounded-md border border-border bg-white">
      <div className="max-h-72 overflow-auto">
        <table className="w-full min-w-[420px] border-collapse text-left text-xs">
          <thead className="sticky top-0 bg-slate-100 text-slate-700">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id} className="border-b border-border px-3 py-2 font-bold">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="odd:bg-white even:bg-slate-50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="border-b border-border/70 px-3 py-2 text-slate-700">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function NestedTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || typeof value !== "object") {
    return <span className="text-xs text-slate-700">{String(value)}</span>;
  }

  if (Array.isArray(value)) {
    return (
      <div className="space-y-1">
        {value.slice(0, 8).map((item, index) => (
          <TreeRow key={index} label={`[${index}]`} depth={depth}>
            <NestedTree value={item} depth={depth + 1} />
          </TreeRow>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {Object.entries(value as Record<string, unknown>)
        .slice(0, 18)
        .map(([key, item]) => (
          <TreeRow key={key} label={key} depth={depth}>
            <NestedTree value={item} depth={depth + 1} />
          </TreeRow>
        ))}
    </div>
  );
}

function TreeRow({
  label,
  depth,
  children,
}: {
  label: string;
  depth: number;
  children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-md border border-border/70 bg-white px-2 py-1.5"
      style={{ marginLeft: depth ? `${Math.min(depth, 4) * 10}px` : undefined }}
    >
      <div className="flex items-start gap-2">
        <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-bold text-slate-600">
          {label}
        </span>
        <div className="min-w-0 flex-1 overflow-hidden">{children}</div>
      </div>
    </div>
  );
}

function TokenList({ label, values }: { label: string; values: unknown[] }) {
  return (
    <div>
      <p className="mb-2 text-xs font-bold text-muted-foreground">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {values.slice(0, 30).map((value) => (
          <span
            key={String(value)}
            className="max-w-full truncate rounded-md border border-border bg-white px-2 py-1 text-xs font-semibold text-slate-700"
          >
            {String(value)}
          </span>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-teal-200 bg-white/80 p-2">
      <p className="text-[10px] font-bold uppercase tracking-wide text-teal-900/60">{label}</p>
      <p className="mt-1 truncate text-xs font-bold text-teal-950">{value}</p>
    </div>
  );
}

function InspectorEmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white/64 p-6 text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-slate-950 text-white">
        <Braces className="h-5 w-5" />
      </div>
      <p className="mt-4 text-sm font-bold">No active dataset</p>
      <p className="mt-2 text-sm leading-6 text-muted-foreground">
        Upload a pickle file to inspect its generic object profile.
      </p>
    </div>
  );
}

function formatBytes(value?: number | null) {
  if (!value) {
    return "n/a";
  }

  if (value < 1024) {
    return `${value} B`;
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }

  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "object") {
    return formatJson(value);
  }

  return String(value);
}

function formatJson(value: unknown) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
