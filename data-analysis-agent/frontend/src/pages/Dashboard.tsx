import { motion } from "framer-motion";
import {
  Activity,
  Bot,
  Braces,
  ChartNoAxesColumnIncreasing,
  CheckCircle2,
  Database,
  GitBranch,
  Play,
  Plus,
  Send,
  Sparkles,
  Upload,
} from "lucide-react";

import { Button } from "../components/Button";
import { Panel } from "../components/Panel";
import { useHealth } from "../hooks/useHealth";

const traceItems = [
  { label: "Session initialized", detail: "Waiting for datasets", tone: "teal" },
  { label: "Runtime ready", detail: "Python workspace prepared", tone: "indigo" },
  { label: "Agent standby", detail: "Trace stream will appear here", tone: "rose" },
];

const datasets = [
  { name: "Upload .pkl", meta: "No schema assumptions", icon: Upload },
  { name: "Branch state", meta: "Fork mutations per session", icon: GitBranch },
  { name: "Export CSV", meta: "Current or intermediate result", icon: Database },
];

export function Dashboard() {
  const health = useHealth();
  const isOnline = health.status === "online";

  return (
    <main className="min-h-screen p-4 text-foreground sm:p-6 lg:p-8">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-[1500px] flex-col gap-5">
        <header className="flex flex-col gap-4 rounded-lg border border-white/80 bg-white/64 px-5 py-4 shadow-glow backdrop-blur-xl md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-teal-700 text-white shadow-lg shadow-teal-900/15">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Take-home project</p>
              <h1 className="text-2xl font-bold tracking-normal">Data Analysis Agent</h1>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex h-10 items-center gap-2 rounded-md border border-border bg-white/78 px-3 text-sm font-medium">
              <span
                className={`h-2.5 w-2.5 rounded-full ${isOnline ? "bg-emerald-500" : "bg-amber-500"}`}
              />
              API {health.status === "loading" ? "checking" : isOnline ? "online" : "offline"}
            </div>
            <Button variant="secondary">
              <Upload className="h-4 w-4" />
              Upload dataset
            </Button>
            <Button>
              <Plus className="h-4 w-4" />
              New session
            </Button>
          </div>
        </header>

        <div className="grid flex-1 gap-5 lg:grid-cols-[260px_minmax(0,1fr)_320px]">
          <Panel className="flex flex-col overflow-hidden">
            <div className="border-b border-border/80 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Workspace
              </p>
              <h2 className="mt-2 text-lg font-bold">Analysis sessions</h2>
            </div>
            <nav className="flex-1 space-y-2 p-3">
              {datasets.map((item) => (
                <button
                  key={item.name}
                  className="flex w-full items-center gap-3 rounded-md px-3 py-3 text-left transition hover:bg-white"
                >
                  <span className="flex h-9 w-9 items-center justify-center rounded-md bg-slate-900 text-white">
                    <item.icon className="h-4 w-4" />
                  </span>
                  <span>
                    <span className="block text-sm font-semibold">{item.name}</span>
                    <span className="block text-xs text-muted-foreground">{item.meta}</span>
                  </span>
                </button>
              ))}
            </nav>
            <div className="border-t border-border/80 p-4">
              <div className="rounded-lg bg-slate-950 p-4 text-white">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <Bot className="h-4 w-4 text-teal-300" />
                  Coding agent
                </div>
                <p className="mt-2 text-xs leading-5 text-slate-300">
                  Minimal tool loop prepared for execute_python, final_answer, and confirmations.
                </p>
              </div>
            </div>
          </Panel>

          <Panel className="flex min-h-[680px] flex-col overflow-hidden">
            <div className="border-b border-border/80 p-5">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Session alpha</p>
                  <h2 className="mt-1 text-2xl font-bold">Chat with your data runtime</h2>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs font-semibold text-muted-foreground">
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    0 datasets
                  </div>
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    1 branch
                  </div>
                  <div className="rounded-md border border-border bg-white/70 px-3 py-2">
                    SSE ready
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
                  Ready for arbitrary datasets
                </div>
                <p className="mt-2 text-sm leading-6 text-teal-950/75">
                  Upload one or more pickle files, then ask the agent to inspect, transform,
                  visualize, branch, and export results. The live reasoning trace will stream
                  below while final answers stay visually distinct.
                </p>
              </motion.div>

              <div className="space-y-3">
                {traceItems.map((item, index) => (
                  <motion.div
                    key={item.label}
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
                ))}
              </div>

              <div className="rounded-lg border-2 border-teal-600 bg-white p-5 shadow-lg shadow-teal-900/10">
                <div className="flex items-center gap-2 text-sm font-bold text-teal-800">
                  <CheckCircle2 className="h-4 w-4" />
                  Final answer placeholder
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-700">
                  Completed agent responses will render here with charts, export links, and a clear
                  separation from intermediate trace events.
                </p>
              </div>
            </div>

            <div className="border-t border-border/80 bg-white/60 p-4">
              <div className="flex items-end gap-3 rounded-lg border border-border bg-white p-3 shadow-sm">
                <textarea
                  className="min-h-16 flex-1 resize-none border-0 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
                  placeholder="Ask the agent to inspect, clean, visualize, branch, or export your data..."
                />
                <Button className="h-11 w-11 px-0" aria-label="Send message">
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </Panel>

          <Panel className="flex flex-col overflow-hidden">
            <div className="border-b border-border/80 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Inspector
              </p>
              <h2 className="mt-2 text-lg font-bold">Runtime state</h2>
            </div>

            <div className="space-y-4 p-4">
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

              <div className="rounded-lg border border-border bg-white/72 p-4">
                <div className="flex items-center gap-2 text-sm font-bold">
                  <Braces className="h-4 w-4 text-indigo-700" />
                  Tool contract
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {["execute_python", "final_answer", "request_confirmation"].map((tool) => (
                    <span
                      key={tool}
                      className="rounded-md border border-border bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-700"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-border bg-white/72 p-4">
                <div className="flex items-center gap-2 text-sm font-bold">
                  <ChartNoAxesColumnIncreasing className="h-4 w-4 text-rose-700" />
                  Visualization lane
                </div>
                <div className="mt-4 flex h-28 items-end gap-2">
                  {[38, 74, 52, 88, 63, 94, 57].map((height, index) => (
                    <div
                      key={index}
                      className="flex-1 rounded-t-md bg-gradient-to-t from-teal-700 to-rose-300"
                      style={{ height: `${height}%` }}
                    />
                  ))}
                </div>
              </div>

              <Button variant="secondary" className="w-full">
                <Play className="h-4 w-4" />
                Preview SSE trace
              </Button>
            </div>
          </Panel>
        </div>
      </div>
    </main>
  );
}
