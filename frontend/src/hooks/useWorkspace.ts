import { useCallback, useEffect, useMemo, useState } from "react";

import {
  activateDataset as activateDatasetRequest,
  checkoutBranch,
  createBranch as createBranchRequest,
  createSession,
  exportDataset,
  forkVersion as forkVersionRequest,
  getHistory,
  getSession,
  listSessions,
  listDatasets,
  rollbackVersion as rollbackVersionRequest,
  uploadDatasets,
} from "../lib/api";
import type { AnalysisSession, Branch, Dataset, ExecutionArtifact, HistoryVersion } from "../types/api";

type UploadStatus = "idle" | "uploading" | "success" | "error";

export type UploadState = {
  status: UploadStatus;
  progress: number;
  filenames: string[];
  message: string | null;
};

const initialUploadState: UploadState = {
  status: "idle",
  progress: 0,
  filenames: [],
  message: null,
};

const SESSION_STORAGE_KEY = "data_analysis_agent:last_session_id";
const LEGACY_SESSION_STORAGE_KEY = "data-analysis-agent-session-id";
let sessionBootstrap: Promise<{
  session: AnalysisSession;
  datasets: Dataset[];
  history: HistoryVersion[];
  restored: boolean;
}> | null = null;

export function useWorkspace() {
  const [session, setSession] = useState<AnalysisSession | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [history, setHistory] = useState<HistoryVersion[]>([]);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [historyAction, setHistoryAction] = useState<string | null>(null);
  const [exportArtifacts, setExportArtifacts] = useState<ExecutionArtifact[]>([]);
  const [exportStatus, setExportStatus] = useState<"idle" | "exporting" | "success" | "error">("idle");
  const [exportMessage, setExportMessage] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<"loading" | "ready" | "error">("loading");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [restoreMessage, setRestoreMessage] = useState<string | null>(null);
  const [recentSessions, setRecentSessions] = useState<AnalysisSession[]>([]);
  const [upload, setUpload] = useState<UploadState>(initialUploadState);

  useEffect(() => {
    let isMounted = true;

    bootstrapSession()
      .then(({ session: readySession, datasets: readyDatasets, history: readyHistory, restored }) => {
        if (!isMounted) {
          return;
        }

        setSession(readySession);
        setDatasets(readyDatasets);
        setHistory(readyHistory);
        setActiveDatasetId(readySession.active_dataset_id ?? readyDatasets[0]?.id ?? null);
        setSelectedVersionId(lastItem(readyHistory)?.id ?? null);
        setRestoreMessage(restored ? "Restored previous session" : null);
        setSessionStatus("ready");
        void refreshRecentSessions(setRecentSessions);
      })
      .catch((error: unknown) => {
        if (!isMounted) {
          return;
        }

        setSessionStatus("error");
        setSessionError(error instanceof Error ? error.message : "Unable to create session");
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const activeDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === activeDatasetId) ?? datasets[0] ?? null,
    [activeDatasetId, datasets],
  );

  const activeBranch = useMemo(
    () => activeBranchFromSession(session),
    [session],
  );

  const selectedVersion = useMemo(
    () => history.find((version) => version.id === selectedVersionId) ?? lastItem(history) ?? null,
    [history, selectedVersionId],
  );

  const refreshWorkspace = useCallback(async () => {
    if (!session) {
      return;
    }

    const refreshed = await loadWorkspace(session.id);
    setSession(refreshed.session);
    setDatasets(refreshed.datasets);
    setHistory(refreshed.history);
    setActiveDatasetId((current) =>
      refreshed.session.active_dataset_id ??
        (current && refreshed.datasets.some((dataset) => dataset.id === current)
          ? current
          : refreshed.datasets[0]?.id ?? null),
    );
    setSelectedVersionId((current) =>
      current && refreshed.history.some((version) => version.id === current)
        ? current
        : lastItem(refreshed.history)?.id ?? null,
    );
  }, [session]);

  const restoreSession = useCallback(async (sessionId: string) => {
    setSessionStatus("loading");
    setSessionError(null);
    try {
      const restored = await loadWorkspace(sessionId);
      sessionBootstrap = Promise.resolve({ ...restored, restored: true });
      applyActiveSession(restored.session.id);
      setSession(restored.session);
      setDatasets(restored.datasets);
      setHistory(restored.history);
      setActiveDatasetId(restored.session.active_dataset_id ?? restored.datasets[0]?.id ?? null);
      setSelectedVersionId(lastItem(restored.history)?.id ?? null);
      setExportArtifacts([]);
      setRestoreMessage("Restored previous session");
      setSessionStatus("ready");
      await refreshRecentSessions(setRecentSessions);
    } catch (error: unknown) {
      setSessionStatus("error");
      setSessionError(error instanceof Error ? error.message : "Unable to restore session");
    }
  }, []);

  const startNewSession = useCallback(async () => {
    setSessionStatus("loading");
    setSessionError(null);
    try {
      const created = await createSession("New conversation");
      applyActiveSession(created.id);
      const loaded = await loadWorkspace(created.id);
      sessionBootstrap = Promise.resolve({ ...loaded, restored: false });
      setSession(loaded.session);
      setDatasets(loaded.datasets);
      setHistory(loaded.history);
      setActiveDatasetId(null);
      setSelectedVersionId(null);
      setExportArtifacts([]);
      setRestoreMessage("Started a new conversation");
      setUpload(initialUploadState);
      setSessionStatus("ready");
      await refreshRecentSessions(setRecentSessions);
    } catch (error: unknown) {
      setSessionStatus("error");
      setSessionError(error instanceof Error ? error.message : "Unable to create session");
    }
  }, []);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      if (!session || files.length === 0) {
        return;
      }

      const pickleFiles = files.filter((file) => file.name.toLowerCase().endsWith(".pkl"));
      if (pickleFiles.length !== files.length) {
        setUpload({
          status: "error",
          progress: 0,
          filenames: files.map((file) => file.name),
          message: "Only .pkl files are supported.",
        });
        return;
      }

      setUpload({
        status: "uploading",
        progress: 2,
        filenames: pickleFiles.map((file) => file.name),
        message: "Uploading and profiling datasets...",
      });

      try {
        const response = await uploadDatasets(session.id, pickleFiles, (progress) => {
          setUpload((current) => ({ ...current, progress }));
        });

        setDatasets((current) => [...response.datasets, ...current]);
        const nextActiveDatasetId = response.datasets[0]?.id ?? activeDatasetId;
        setActiveDatasetId(nextActiveDatasetId);
        if (nextActiveDatasetId) {
          await activateDatasetRequest(session.id, nextActiveDatasetId);
        }
        void refreshWorkspace();
        setUpload({
          status: "success",
          progress: 100,
          filenames: pickleFiles.map((file) => file.name),
          message: `${response.datasets.length} dataset${response.datasets.length === 1 ? "" : "s"} uploaded.`,
        });
      } catch (error: unknown) {
        setUpload({
          status: "error",
          progress: 0,
          filenames: pickleFiles.map((file) => file.name),
          message: error instanceof Error ? error.message : "Upload failed",
        });
      }
    },
    [activeDatasetId, refreshWorkspace, session],
  );

  const refreshDatasets = useCallback(async () => {
    await refreshWorkspace();
  }, [refreshWorkspace]);

  const activateDataset = useCallback(
    async (datasetId: string) => {
      if (!session) {
        setActiveDatasetId(datasetId);
        return;
      }

      setActiveDatasetId(datasetId);
      try {
        const updatedSession = await activateDatasetRequest(session.id, datasetId);
        setSession(updatedSession);
      } catch (error: unknown) {
        setSessionError(error instanceof Error ? error.message : "Unable to activate dataset");
        await refreshWorkspace();
      }
    },
    [refreshWorkspace, session],
  );

  const createBranch = useCallback(
    async (name: string, fromVersionId?: string | null) => {
      if (!session || !name.trim()) {
        return;
      }

      setHistoryAction("create");
      try {
        await createBranchRequest(session.id, { name: name.trim(), from_version_id: fromVersionId ?? null });
        await refreshWorkspace();
      } finally {
        setHistoryAction(null);
      }
    },
    [refreshWorkspace, session],
  );

  const checkout = useCallback(
    async (branchId: string) => {
      if (!session) {
        return;
      }

      setHistoryAction(`checkout:${branchId}`);
      try {
        await checkoutBranch(session.id, branchId);
        await refreshWorkspace();
      } finally {
        setHistoryAction(null);
      }
    },
    [refreshWorkspace, session],
  );

  const rollback = useCallback(
    async (versionId: string) => {
      if (!session) {
        return;
      }

      const shouldRollback = window.confirm(
        "Rollback restores an earlier snapshot as the current dataset state. Apply this change?",
      );
      if (!shouldRollback) {
        return;
      }

      setHistoryAction(`rollback:${versionId}`);
      try {
        const response = await rollbackVersionRequest(session.id, versionId);
        await refreshWorkspace();
        setSelectedVersionId(response.version.id);
      } finally {
        setHistoryAction(null);
      }
    },
    [refreshWorkspace, session],
  );

  const forkVersion = useCallback(
    async (versionId: string, name: string) => {
      if (!session || !name.trim()) {
        return;
      }

      setHistoryAction(`fork:${versionId}`);
      try {
        await forkVersionRequest(session.id, versionId, { name: name.trim() });
        await refreshWorkspace();
        setSelectedVersionId(versionId);
      } finally {
        setHistoryAction(null);
      }
    },
    [refreshWorkspace, session],
  );

  const exportCurrentDataset = useCallback(async () => {
    if (!session || !activeDataset) {
      return;
    }

    setExportStatus("exporting");
    setExportMessage("Creating CSV artifact...");
    try {
      const response = await exportDataset(session.id, {
        dataset_id: activeDataset.id,
        name: `${activeDataset.original_filename}-current-branch`,
      });
      if (response.artifact) {
        const artifact = response.artifact;
        setExportArtifacts((current) => [artifact, ...current]);
      }
      setExportStatus(response.ok ? "success" : "error");
      setExportMessage(response.message);
    } catch (error: unknown) {
      setExportStatus("error");
      setExportMessage(error instanceof Error ? error.message : "Export failed");
    }
  }, [activeDataset, session]);

  return {
    session,
    sessionStatus,
    sessionError,
    restoreMessage,
    recentSessions,
    datasets,
    activeDataset,
    activeDatasetId,
    activeBranch,
    history,
    selectedVersion,
    selectedVersionId,
    setSelectedVersionId,
    historyAction,
    exportArtifacts,
    exportStatus,
    exportMessage,
    setActiveDatasetId: activateDataset,
    upload,
    uploadFiles,
    refreshDatasets,
    refreshWorkspace,
    restoreSession,
    startNewSession,
    createBranch,
    checkoutBranch: checkout,
    rollbackVersion: rollback,
    forkVersion,
    exportCurrentDataset,
  };
}

async function bootstrapSession() {
  if (sessionBootstrap) {
    return sessionBootstrap;
  }

  sessionBootstrap = (async () => {
    const urlSessionId = new URLSearchParams(window.location.search).get("session");
    const existingId =
      urlSessionId ||
      window.localStorage.getItem(SESSION_STORAGE_KEY) ||
      window.sessionStorage.getItem(LEGACY_SESSION_STORAGE_KEY);
    if (existingId) {
      try {
        const loaded = await loadWorkspace(existingId);
        applyActiveSession(loaded.session.id);
        return { ...loaded, restored: true };
      } catch {
        window.localStorage.removeItem(SESSION_STORAGE_KEY);
        window.sessionStorage.removeItem(LEGACY_SESSION_STORAGE_KEY);
      }
    }

    const existingSessions = await listSessions();
    const latestSession = existingSessions.sessions[0];
    if (latestSession) {
      applyActiveSession(latestSession.id);
      return { ...(await loadWorkspace(latestSession.id)), restored: true };
    }

    const createdSession = await createSession("Demo workspace");
    applyActiveSession(createdSession.id);
    return { ...(await loadWorkspace(createdSession.id)), restored: false };
  })();

  return sessionBootstrap;
}

function applyActiveSession(sessionId: string) {
  window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  window.sessionStorage.removeItem(LEGACY_SESSION_STORAGE_KEY);
  const url = new URL(window.location.href);
  url.searchParams.set("session", sessionId);
  window.history.replaceState({}, "", url);
}

async function refreshRecentSessions(setRecentSessions: (sessions: AnalysisSession[]) => void) {
  try {
    const response = await listSessions();
    setRecentSessions(response.sessions.slice(0, 10));
  } catch {
    // Recent sessions are a convenience; workspace restore errors are surfaced elsewhere.
  }
}

async function loadWorkspace(sessionId: string) {
  const [session, datasetResponse, historyResponse] = await Promise.all([
    getSession(sessionId),
    listDatasets(sessionId),
    getHistory(sessionId),
  ]);

  return {
    session,
    datasets: datasetResponse.datasets,
    history: historyResponse.versions,
  };
}

function activeBranchFromSession(session: AnalysisSession | null): Branch | null {
  if (!session) {
    return null;
  }

  return (
    session.branches.find((branch) => branch.id === session.active_branch_id) ??
    session.branches[0] ??
    null
  );
}

function lastItem<T>(items: T[]): T | undefined {
  return items[items.length - 1];
}
