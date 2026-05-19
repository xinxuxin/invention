import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { BarChart3, CalendarClock, Download, FileJson2, Table2 } from "lucide-react";

import { artifactDownloadUrl, getArtifactContent } from "../lib/api";
import type { ExecutionArtifact } from "../types/api";

type ArtifactCardProps = {
  sessionId: string;
  artifact: ExecutionArtifact;
};

export function ArtifactCard({ sessionId, artifact }: ArtifactCardProps) {
  const [content, setContent] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
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
    <section className="rounded-lg border border-border bg-white/78 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-bold">
            {artifact.kind === "csv" ? (
              <Download className="h-4 w-4 text-teal-700" />
            ) : artifact.kind === "table" ? (
              <Table2 className="h-4 w-4 text-indigo-700" />
            ) : artifact.kind === "chart" ? (
              <BarChart3 className="h-4 w-4 text-cyan-700" />
            ) : (
              <FileJson2 className="h-4 w-4 text-rose-700" />
            )}
            <span className="truncate">{artifact.name}</span>
          </div>
          <p className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">
            {artifact.kind}
            {rowCount(artifact.metadata) ? ` - ${rowCount(artifact.metadata)} rows` : ""}
          </p>
          {artifact.created_at ? (
            <p className="mt-1 flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
              <CalendarClock className="h-3 w-3" />
              {formatDate(artifact.created_at)}
            </p>
          ) : null}
        </div>
        {artifact.kind === "csv" ? (
          <a
            className="rounded-md border border-border bg-white px-2.5 py-1.5 text-xs font-bold text-slate-700 transition hover:bg-slate-50"
            href={artifactDownloadUrl(sessionId, artifact.id)}
          >
            Download
          </a>
        ) : null}
      </div>

      <div className="mt-4">
        {error ? (
          <p className="rounded-md bg-rose-50 p-3 text-xs font-semibold text-rose-700">{error}</p>
        ) : artifact.kind === "csv" ? (
          <CsvPreview content={typeof content === "string" ? content : ""} />
        ) : artifact.kind === "table" ? (
          <MiniTable rows={Array.isArray(content) ? content : []} />
        ) : artifact.kind === "chart" ? (
          <ChartPreview spec={content} />
        ) : (
          <pre className="max-h-52 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-100">
            {JSON.stringify(content, null, 2)}
          </pre>
        )}
      </div>
    </section>
  );
}

function MiniTable({ rows }: { rows: unknown[] }) {
  const normalized = rows.filter(isRecord).slice(0, 8);
  const columns = useMemo(
    () => Array.from(new Set(normalized.flatMap((row) => Object.keys(row)))).slice(0, 8),
    [normalized],
  );

  if (normalized.length === 0) {
    return <p className="text-sm text-muted-foreground">Table artifact is loading...</p>;
  }

  return (
    <div className="max-h-64 overflow-auto rounded-md border border-border bg-white">
      <table className="w-full min-w-[320px] text-left text-xs">
        <thead className="sticky top-0 bg-slate-100">
          <tr>
            {columns.map((column) => (
              <th key={column} className="border-b border-border px-3 py-2 font-bold">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {normalized.map((row, index) => (
            <tr key={index} className="odd:bg-white even:bg-slate-50">
              {columns.map((column) => (
                <td key={column} className="border-b border-border/70 px-3 py-2">
                  {formatCell(row[column])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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

function ChartPreview({ spec }: { spec: unknown }) {
  const chart = parseChartSpec(spec);

  if (!chart) {
    return (
      <pre className="max-h-52 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-100">
        {JSON.stringify(spec, null, 2)}
      </pre>
    );
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
      <div className="h-64 rounded-md border border-border bg-white p-3">
        <ResponsiveContainer width="100%" height="100%">
          <RenderedChart chart={chart} />
        </ResponsiveContainer>
      </div>
      <MiniTable rows={chart.data.slice(0, 6)} />
    </div>
  );
}

type ChartSpec = {
  title: string;
  chart_type: "bar" | "line" | "pie" | "scatter" | "area";
  data: Record<string, unknown>[];
  x: string;
  y: string;
  color?: string | null;
  description?: string | null;
};

function RenderedChart({ chart }: { chart: ChartSpec }) {
  const stroke = "#0f766e";
  const fill = "#14b8a6";

  if (chart.chart_type === "line") {
    return (
      <LineChart data={chart.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={chart.x} />
        <YAxis />
        <Tooltip />
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
        <Area type="monotone" dataKey={chart.y} stroke={stroke} fill={fill} fillOpacity={0.28} />
      </AreaChart>
    );
  }

  if (chart.chart_type === "pie") {
    return (
      <PieChart>
        <Tooltip />
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
      <Bar dataKey={chart.y} fill={fill} radius={[4, 4, 0, 0]} />
    </BarChart>
  );
}

const CHART_COLORS = ["#0f766e", "#2563eb", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#65a30d"];

function parseChartSpec(spec: unknown): ChartSpec | null {
  if (!isRecord(spec) || !Array.isArray(spec.data)) {
    return null;
  }

  const data = spec.data.filter(isRecord);
  if (!data.length) {
    return null;
  }

  const chartType =
    typeof spec.chart_type === "string"
      ? spec.chart_type
      : typeof spec.mark === "string"
        ? spec.mark
        : "bar";
  if (!["bar", "line", "pie", "scatter", "area"].includes(chartType)) {
    return null;
  }

  const keys = Object.keys(data[0]);
  const x = typeof spec.x === "string" ? spec.x : keys[0];
  const y = typeof spec.y === "string" ? spec.y : keys.find((key) => key !== x) ?? keys[1];
  if (!x || !y) {
    return null;
  }

  return {
    title: typeof spec.title === "string" ? spec.title : "Chart",
    chart_type: chartType as ChartSpec["chart_type"],
    data,
    x,
    y,
    color: typeof spec.color === "string" ? spec.color : null,
    description: typeof spec.description === "string" ? spec.description : null,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
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

function rowCount(metadata: Record<string, unknown>) {
  const rows = metadata.rows;
  return typeof rows === "number" || typeof rows === "string" ? String(rows) : null;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
