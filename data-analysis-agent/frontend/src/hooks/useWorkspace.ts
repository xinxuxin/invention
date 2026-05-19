import { useCallback, useEffect, useMemo, useState } from "react";

import { createSession, getSession, listDatasets, uploadDatasets } from "../lib/api";
import type { AnalysisSession, Dataset } from "../types/api";

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
let sessionBootstrap: Promise<{ session: AnalysisSession; datasets: Dataset[] }> | null = null;

export function useWorkspace() {
  const [session, setSession] = useState<AnalysisSession | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<"loading" | "ready" | "error">("loading");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadState>(initialUploadState);

  useEffect(() => {
    let isMounted = true;

    bootstrapSession()
      .then(({ session: readySession, datasets: readyDatasets }) => {
        if (!isMounted) {
          return;
        }

        setSession(readySession);
        setDatasets(readyDatasets);
        setActiveDatasetId(readyDatasets[0]?.id ?? null);
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
    [activeDatasetId, session],
  );

  const refreshDatasets = useCallback(async () => {
    if (!session) {
      return;
    }

    const response = await listDatasets(session.id);
    setDatasets(response.datasets);
    setActiveDatasetId((current) => current ?? response.datasets[0]?.id ?? null);
  }, [session]);

  return {
    session,
    sessionStatus,
    sessionError,
    datasets,
    activeDataset,
    activeDatasetId,
    setActiveDatasetId,
    upload,
    uploadFiles,
    refreshDatasets,
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
        const existingSession = await getSession(existingId);
        const response = await listDatasets(existingSession.id);
        return { session: existingSession, datasets: response.datasets };
      } catch {
        window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
      }
    }

    const createdSession = await createSession("Demo workspace");
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, createdSession.id);
    const response = await listDatasets(createdSession.id);
    return { session: createdSession, datasets: response.datasets };
  })();

  return sessionBootstrap;
}
