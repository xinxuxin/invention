import { useEffect, useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
} from "@tanstack/react-table";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BarChart3,
  CalendarClock,
  ChevronDown,
  Download,
  FileJson2,
  Table2,
} from "lucide-react";

import { artifactDownloadUrl, getArtifactContent } from "../lib/api";
import type { ExecutionArtifact } from "../types/api";

type ArtifactCardProps = {
  sessionId: string;
  artifact: ExecutionArtifact;
};

export function ArtifactCard({ sessionId, artifact }: ArtifactCardProps) {
  const [content, setContent] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const title = artifactTitle(artifact);
  const description = artifactDescription(artifact);

  useEffect(() => {
    let mounted = true;
    setContent(null);
    setError(null);
    getArtifactContent(sessionId, artifact)
      .then((value) => {
        if (mounted) {
          setContent(value);
        }
      })
      .catch((err: unknown) => {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Unable to load artifact");
        }
      });

    return () => {
      mounted = false;
    };
  }, [artifact, sessionId]);

  return (
    <section className="overflow-hidden rounded-lg border border-border bg-white/84 shadow-sm shadow-slate-900/5">
      <div className="flex items-start justify-between gap-3 border-b border-border/70 bg-white/70 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-bold">
            <ArtifactIcon kind={artifact.kind} />
            <span className="truncate">{title}</span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] font-bold uppercase tracking-wide text-muted-foreground">
            <span>{artifact.kind}</span>
            {artifactRowCount(artifact) ? <span>{artifactRowCount(artifact)} rows</span> : null}
            {artifact.created_at ? (
              <span className="inline-flex items-center gap-1 normal-case tracking-normal">
                <CalendarClock className="h-3 w-3" />
                {formatDate(artifact.created_at)}
              </span>
            ) : null}
          </div>
          {description ? (
            <p className="mt-2 line-clamp-2 text-xs leading-5 text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {artifact.kind === "csv" ? (
          <a
            className="shrink-0 rounded-md border border-border bg-white px-2.5 py-1.5 text-xs font-bold text-slate-700 transition hover:bg-slate-50"
            href={artifactDownloadUrl(sessionId, artifact.id)}
          >
            Download
          </a>
        ) : null}
      </div>

      <div className="p-4">
        {error ? (
          <p className="rounded-md bg-rose-50 p-3 text-xs font-semibold text-rose-700">{error}</p>
        ) : artifact.kind === "csv" ? (
          <CsvPreview content={typeof content === "string" ? content : ""} />
        ) : artifact.kind === "table" ? (
          <TableArtifact artifact={artifact} content={content} />
        ) : artifact.kind === "chart" ? (
          <ChartPreview spec={content} fallback={artifact.chart_spec ?? artifact.metadata.chart_spec} />
        ) : (
          <JsonPreview content={content} />
        )}
      </div>
    </section>
  );
}

function ArtifactIcon({ kind }: { kind: string }) {
  if (kind === "csv") {
    return <Download className="h-4 w-4 text-teal-700" />;
  }
  if (kind === "table") {
    return <Table2 className="h-4 w-4 text-indigo-700" />;
  }
  if (kind === "chart") {
    return <BarChart3 className="h-4 w-4 text-cyan-700" />;
  }
  return <FileJson2 className="h-4 w-4 text-rose-700" />;
}

function TableArtifact({ artifact, content }: { artifact: ExecutionArtifact; content: unknown }) {
  const table = normalizeTableContent(content, artifact);
  if (!table) {
    return <p className="text-sm text-muted-foreground">Table artifact is loading...</p>;
  }

  const visibleRows = table.previewRows.length > 0 ? table.previewRows : table.rows.slice(0, 50);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2 text-xs font-bold text-slate-600">
          <span className="rounded-full bg-slate-100 px-2 py-1">
            {table.rowCount.toLocaleString()} total rows
          </span>
          <span className="rounded-full bg-slate-100 px-2 py-1">
            showing {visibleRows.length.toLocaleString()}
          </span>
          {table.truncated ? (
            <span className="rounded-full bg-amber-50 px-2 py-1 text-amber-800">preview truncated</span>
          ) : null}
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-white px-2.5 py-1.5 text-xs font-bold text-slate-700 transition hover:bg-slate-50"
          onClick={() => downloadRowsAsCsv(`${table.title || artifact.name}.csv`, table.rows)}
        >
          <Download className="h-3.5 w-3.5" />
          Download CSV
        </button>
      </div>
      <DataTable rows={visibleRows} columns={table.columns} />
    </div>
  );
}

function DataTable({ rows, columns }: { rows: Record<string, unknown>[]; columns?: TableColumn[] }) {
  const resolvedColumns = useMemo(() => {
    if (columns?.length) {
      return columns;
    }
    return Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
      .slice(0, 30)
      .map((key) => ({ key, label: key, type: inferCellType(rows.find((row) => row[key] !== undefined)?.[key]) }));
  }, [columns, rows]);

  const columnDefs = useMemo<ColumnDef<Record<string, unknown>, unknown>[]>(
    () =>
      resolvedColumns.map((column) => ({
        id: column.key,
        accessorFn: (row) => row[column.key],
        header: () => (
          <div className="flex min-w-28 items-center gap-2">
            <span className="truncate">{column.label || column.key}</span>
            <span className="rounded-full bg-slate-200 px-1.5 py-0.5 text-[10px] font-bold uppercase text-slate-600">
              {column.type || "value"}
            </span>
          </div>
        ),
        cell: (info) => <CellValue value={info.getValue()} />,
      })),
    [resolvedColumns],
  );

  const table = useReactTable({
    data: rows,
    columns: columnDefs,
    getCoreRowModel: getCoreRowModel(),
  });

  if (rows.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-muted-foreground">
        This table has no preview rows.
      </div>
    );
  }

  return (
    <div className="max-h-80 overflow-auto rounded-md border border-border bg-white">
      <table className="w-full min-w-[640px] text-left text-xs">
        <thead className="sticky top-0 z-10 bg-slate-100 shadow-sm">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id} className="border-b border-border px-3 py-2 font-bold text-slate-700">
                  {header.isPlaceholder
                    ? null
                    : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="odd:bg-white even:bg-slate-50/80">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="max-w-[260px] border-b border-border/70 px-3 py-2 align-top">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CellValue({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === "") {
    return <span className="text-slate-400">empty</span>;
  }

  if (Array.isArray(value)) {
    const label = `${value.length} item${value.length === 1 ? "" : "s"}`;
    return (
      <span
        className="inline-flex max-w-full items-center rounded-full bg-indigo-50 px-2 py-0.5 text-[11px] font-bold text-indigo-800"
        title={JSON.stringify(value)}
      >
        {label}
      </span>
    );
  }

  if (isRecord(value)) {
    return (
      <span
        className="inline-flex max-w-full rounded-md bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-700"
        title={JSON.stringify(value)}
      >
        object
      </span>
    );
  }

  const text = String(value);
  return (
    <span className="block max-w-[240px] truncate text-slate-700" title={text}>
      {text}
    </span>
  );
}

function CsvPreview({ content }: { content: string }) {
  if (!content) {
    return <p className="text-sm text-muted-foreground">CSV artifact is loading...</p>;
  }

  return (
    <pre className="max-h-44 overflow-auto whitespace-pre rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">
      {content.split("\n").slice(0, 8).join("\n")}
    </pre>
  );
}

function JsonPreview({ content }: { content: unknown }) {
  return (
    <details className="rounded-md border border-border bg-slate-50 p-3">
      <summary className="flex cursor-pointer list-none items-center justify-between text-xs font-bold text-slate-700">
        JSON artifact
        <ChevronDown className="h-4 w-4" />
      </summary>
      <pre className="mt-3 max-h-52 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-100">
        {JSON.stringify(content, null, 2)}
      </pre>
    </details>
  );
}

function ChartPreview({ spec, fallback }: { spec: unknown; fallback?: unknown }) {
  const chart = parseChartSpec(spec) ?? parseChartSpec(fallback);

  if (!chart) {
    return <JsonPreview content={spec} />;
  }

  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-bold">{chart.title}</p>
          {chart.description ? (
            <p className="mt-1 text-xs leading-5 text-muted-foreground">{chart.description}</p>
          ) : null}
        </div>
        <button
          type="button"
          onClick={() => downloadRowsAsCsv(`${chart.title || "chart-data"}.csv`, chart.data)}
          className="shrink-0 rounded-md border border-border bg-white px-2.5 py-1.5 text-xs font-bold text-slate-700 transition hover:bg-slate-50"
        >
          Export data
        </button>
      </div>
      <div className="h-72 rounded-md border border-border bg-white p-3">
        <ResponsiveContainer width="100%" height="100%">
          <RenderedChart chart={chart} />
        </ResponsiveContainer>
      </div>
      <DataTable rows={chart.data.slice(0, 8)} />
    </div>
  );
}

type ChartSpec = {
  title: string;
  chart_type: "bar" | "line" | "pie" | "scatter" | "area";
  data: Record<string, unknown>[];
  x: string;
  y: string;
  series?: string | null;
  color?: string | null;
  description?: string | null;
};

function RenderedChart({ chart }: { chart: ChartSpec }) {
  const stroke = "#0f766e";
  const fill = "#14b8a6";
  const legend = chart.series ? <Legend /> : null;

  if (chart.chart_type === "line") {
    return (
      <LineChart data={chart.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={chart.x} />
        <YAxis />
        <Tooltip />
        {legend}
        <Line type="monotone" dataKey={chart.y} stroke={stroke} strokeWidth={2} dot={false} />
      </LineChart>
    );
  }

  if (chart.chart_type === "area") {
    return (
      <AreaChart data={chart.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={chart.x} />
        <YAxis />
        <Tooltip />
        {legend}
        <Area type="monotone" dataKey={chart.y} stroke={stroke} fill={fill} fillOpacity={0.28} />
      </AreaChart>
    );
  }

  if (chart.chart_type === "pie") {
    return (
      <PieChart>
        <Tooltip />
        <Legend />
        <Pie data={chart.data} dataKey={chart.y} nameKey={chart.x} outerRadius="82%" label>
          {chart.data.map((_, index) => (
            <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />
          ))}
        </Pie>
      </PieChart>
    );
  }

  if (chart.chart_type === "scatter") {
    return (
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={chart.x} name={chart.x} />
        <YAxis dataKey={chart.y} name={chart.y} />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
        {legend}
        <Scatter data={chart.data} fill={fill} />
      </ScatterChart>
    );
  }

  return (
    <BarChart data={chart.data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey={chart.x} />
      <YAxis />
      <Tooltip />
      {legend}
      <Bar dataKey={chart.y} fill={fill} radius={[4, 4, 0, 0]} />
    </BarChart>
  );
}

type TableColumn = {
  key: string;
  label?: string;
  type?: string;
};

type TableContent = {
  title: string;
  description?: string | null;
  columns: TableColumn[];
  rows: Record<string, unknown>[];
  previewRows: Record<string, unknown>[];
  rowCount: number;
  truncated: boolean;
};

const CHART_COLORS = ["#0f766e", "#2563eb", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#65a30d"];

function normalizeTableContent(content: unknown, artifact: ExecutionArtifact): TableContent | null {
  const metadataColumns = normalizeColumns(artifact.columns?.length ? artifact.columns : artifact.metadata.columns);
  if (Array.isArray(content)) {
    const rows = content.filter(isRecord);
    return {
      title: artifactTitle(artifact),
      description: artifactDescription(artifact),
      columns: metadataColumns.length ? metadataColumns : deriveColumns(rows),
      rows,
      previewRows: rows.slice(0, 50),
      rowCount: rows.length,
      truncated: false,
    };
  }

  if (!isRecord(content)) {
    return null;
  }

  const rows = Array.isArray(content.rows) ? content.rows.filter(isRecord) : [];
  const previewRows = Array.isArray(content.preview_rows) ? content.preview_rows.filter(isRecord) : rows.slice(0, 50);
  const columns = normalizeColumns(content.columns).length
    ? normalizeColumns(content.columns)
    : metadataColumns.length
      ? metadataColumns
      : deriveColumns(rows.length ? rows : previewRows);
  const rowCount = typeof content.row_count === "number" ? content.row_count : rows.length || previewRows.length;

  return {
    title: typeof content.title === "string" ? content.title : artifactTitle(artifact),
    description: typeof content.description === "string" ? content.description : artifactDescription(artifact),
    columns,
    rows,
    previewRows,
    rowCount,
    truncated: Boolean(content.truncated),
  };
}

function parseChartSpec(spec: unknown): ChartSpec | null {
  const raw = isRecord(spec) && isRecord(spec.chart_spec) ? spec.chart_spec : spec;
  if (!isRecord(raw) || !Array.isArray(raw.data)) {
    return null;
  }

  const data = raw.data.filter(isRecord);
  if (!data.length) {
    return null;
  }

  const chartType =
    typeof raw.chart_type === "string"
      ? raw.chart_type
      : typeof raw.mark === "string"
        ? raw.mark
        : "bar";
  if (!["bar", "line", "pie", "scatter", "area"].includes(chartType)) {
    return null;
  }

  const keys = Object.keys(data[0]);
  const x = typeof raw.x === "string" ? raw.x : keys[0];
  const y = typeof raw.y === "string" ? raw.y : keys.find((key) => key !== x) ?? keys[1];
  if (!x || !y) {
    return null;
  }

  return {
    title: typeof raw.title === "string" ? raw.title : "Chart",
    chart_type: chartType as ChartSpec["chart_type"],
    data,
    x,
    y,
    series: typeof raw.series === "string" ? raw.series : null,
    color: typeof raw.color === "string" ? raw.color : null,
    description: typeof raw.description === "string" ? raw.description : null,
  };
}

function normalizeColumns(value: unknown): TableColumn[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const columns: TableColumn[] = [];
  value.forEach((item) => {
    if (typeof item === "string") {
      columns.push({ key: item, label: item, type: "value" });
      return;
    }
    if (isRecord(item) && typeof item.key === "string") {
      columns.push({
        key: item.key,
        label: typeof item.label === "string" ? item.label : item.key,
        type: typeof item.type === "string" ? item.type : "value",
      });
    }
  });
  return columns;
}

function deriveColumns(rows: Record<string, unknown>[]): TableColumn[] {
  return Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
    .slice(0, 30)
    .map((key) => ({ key, label: key, type: inferCellType(rows.find((row) => row[key] !== undefined)?.[key]) }));
}

function inferCellType(value: unknown) {
  if (Array.isArray(value)) {
    return "list";
  }
  if (isRecord(value)) {
    return "object";
  }
  if (typeof value === "number") {
    return "number";
  }
  if (typeof value === "boolean") {
    return "boolean";
  }
  return "value";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function downloadRowsAsCsv(filename: string, rows: Record<string, unknown>[]) {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const csv = [
    columns.join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename.replace(/[^a-z0-9_.-]+/gi, "-");
  anchor.click();
  URL.revokeObjectURL(url);
}

function csvCell(value: unknown) {
  const text = value === null || value === undefined ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function artifactTitle(artifact: ExecutionArtifact) {
  return artifact.title || stringMeta(artifact.metadata.title) || artifact.name;
}

function artifactDescription(artifact: ExecutionArtifact) {
  return artifact.description || stringMeta(artifact.metadata.description);
}

function artifactRowCount(artifact: ExecutionArtifact) {
  const rows = artifact.metadata.row_count ?? artifact.metadata.rows;
  return typeof rows === "number" || typeof rows === "string" ? String(rows) : null;
}

function stringMeta(value: unknown) {
  return typeof value === "string" ? value : null;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
