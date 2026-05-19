import { useCallback, useEffect, useMemo, useState } from "react";

import {
  checkoutBranch,
  createBranch as createBranchRequest,
  createSession,
  forkVersion as forkVersionRequest,
  getHistory,
  getSession,
  listDatasets,
  rollbackVersion as rollbackVersionRequest,
  uploadDatasets,
} from "../lib/api";
import type { AnalysisSession, Branch, Dataset, HistoryVersion } from "../types/api";

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

const SESSION_STORAGE_KEY = "data-analysis-agent-session-id";
let sessionBootstrap: Promise<{
  session: AnalysisSession;
  datasets: Dataset[];
  history: HistoryVersion[];
}> | null = null;

export function useWorkspace() {
  const [session, setSession] = useState<AnalysisSession | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [history, setHistory] = useState<HistoryVersion[]>([]);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [historyAction, setHistoryAction] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<"loading" | "ready" | "error">("loading");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadState>(initialUploadState);

  useEffect(() => {
    let isMounted = true;

    bootstrapSession()
      .then(({ session: readySession, datasets: readyDatasets, history: readyHistory }) => {
        if (!isMounted) {
          return;
        }

        setSession(readySession);
        setDatasets(readyDatasets);
        setHistory(readyHistory);
        setActiveDatasetId(readyDatasets[0]?.id ?? null);
        setSelectedVersionId(lastItem(readyHistory)?.id ?? null);
        setSessionStatus("ready");
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
      current && refreshed.datasets.some((dataset) => dataset.id === current)
        ? current
        : refreshed.datasets[0]?.id ?? null,
    );
    setSelectedVersionId((current) =>
      current && refreshed.history.some((version) => version.id === current)
        ? current
        : lastItem(refreshed.history)?.id ?? null,
    );
  }, [session]);

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
        setActiveDatasetId(response.datasets[0]?.id ?? activeDatasetId);
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

  return {
    session,
    sessionStatus,
    sessionError,
    datasets,
    activeDataset,
    activeDatasetId,
    activeBranch,
    history,
    selectedVersion,
    selectedVersionId,
    setSelectedVersionId,
    historyAction,
    setActiveDatasetId,
    upload,
    uploadFiles,
    refreshDatasets,
    refreshWorkspace,
    createBranch,
    checkoutBranch: checkout,
    rollbackVersion: rollback,
    forkVersion,
  };
}

async function bootstrapSession() {
  if (sessionBootstrap) {
    return sessionBootstrap;
  }

  sessionBootstrap = (async () => {
    const existingId = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existingId) {
      try {
        return await loadWorkspace(existingId);
      } catch {
        window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
      }
    }

    const createdSession = await createSession("Demo workspace");
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, createdSession.id);
    return await loadWorkspace(createdSession.id);
  })();

  return sessionBootstrap;
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
