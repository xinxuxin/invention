import type {
  AnalysisSession,
  BranchActionResponse,
  BranchListResponse,
  ChatStreamEvent,
  ChatStreamRequest,
  ConfirmationActionResponse,
  DatasetListResponse,
  DatasetUploadResponse,
  ExecutionArtifact,
  ExportResponse,
  HealthResponse,
  HistoryResponse,
  VersionActionResponse,
} from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }

  return response.json() as Promise<HealthResponse>;
}

export async function createSession(name?: string): Promise<AnalysisSession> {
  return request<AnalysisSession>("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function getSession(sessionId: string): Promise<AnalysisSession> {
  return request<AnalysisSession>(`/api/sessions/${sessionId}`);
}

export async function listDatasets(sessionId: string): Promise<DatasetListResponse> {
  return request<DatasetListResponse>(`/api/sessions/${sessionId}/datasets`);
}

export async function activateDataset(sessionId: string, datasetId: string): Promise<AnalysisSession> {
  return request<AnalysisSession>(`/api/sessions/${sessionId}/datasets/${datasetId}/activate`, {
    method: "POST",
  });
}

export async function listBranches(sessionId: string): Promise<BranchListResponse> {
  return request<BranchListResponse>(`/api/sessions/${sessionId}/branches`);
}

export async function createBranch(
  sessionId: string,
  payload: { name: string; from_version_id?: string | null },
): Promise<BranchActionResponse> {
  return request<BranchActionResponse>(`/api/sessions/${sessionId}/branches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function checkoutBranch(
  sessionId: string,
  branchId: string,
): Promise<BranchActionResponse> {
  return request<BranchActionResponse>(`/api/sessions/${sessionId}/branches/${branchId}/checkout`, {
    method: "POST",
  });
}

export async function rollbackVersion(
  sessionId: string,
  versionId: string,
): Promise<VersionActionResponse> {
  return request<VersionActionResponse>(`/api/sessions/${sessionId}/versions/${versionId}/rollback`, {
    method: "POST",
  });
}

export async function forkVersion(
  sessionId: string,
  versionId: string,
  payload: { name: string },
): Promise<BranchActionResponse> {
  return request<BranchActionResponse>(`/api/sessions/${sessionId}/versions/${versionId}/fork`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getHistory(sessionId: string): Promise<HistoryResponse> {
  return request<HistoryResponse>(`/api/sessions/${sessionId}/history`);
}

export async function exportDataset(
  sessionId: string,
  payload: { dataset_id?: string | null; version_id?: string | null; name?: string | null },
): Promise<ExportResponse> {
  return request<ExportResponse>(`/api/sessions/${sessionId}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function uploadDatasets(
  sessionId: string,
  files: File[],
  onProgress?: (progress: number) => void,
): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/api/sessions/${sessionId}/datasets`);

    request.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        onProgress?.(100);
        resolve(JSON.parse(request.responseText) as DatasetUploadResponse);
        return;
      }

      reject(new Error(readErrorMessage(request.responseText, request.status)));
    };

    request.onerror = () => reject(new Error("Upload failed before the server responded"));
    request.send(formData);
  });
}

export async function streamChat(
  sessionId: string,
  payload: ChatStreamRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(readErrorMessage(await response.text(), response.status));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    blocks.forEach((block) => parseSseBlock(block, onEvent));
  }

  if (buffer.trim()) {
    parseSseBlock(buffer, onEvent);
  }
}

export async function approveConfirmation(
  sessionId: string,
  confirmationId: string,
): Promise<ConfirmationActionResponse> {
  return request<ConfirmationActionResponse>(
    `/api/sessions/${sessionId}/confirmations/${confirmationId}/approve`,
    { method: "POST" },
  );
}

export async function rejectConfirmation(
  sessionId: string,
  confirmationId: string,
): Promise<ConfirmationActionResponse> {
  return request<ConfirmationActionResponse>(
    `/api/sessions/${sessionId}/confirmations/${confirmationId}/reject`,
    { method: "POST" },
  );
}

export async function getArtifactContent(
  sessionId: string,
  artifact: ExecutionArtifact,
): Promise<unknown> {
  const response = await fetch(
    `${API_BASE_URL}/api/sessions/${sessionId}/artifacts/${artifact.id}/content`,
  );

  if (!response.ok) {
    throw new Error(readErrorMessage(await response.text(), response.status));
  }

  const text = await response.text();
  if (artifact.kind === "csv") {
    return text;
  }

  return JSON.parse(text);
}

export function artifactDownloadUrl(sessionId: string, artifactId: string): string {
  return `${API_BASE_URL}/api/sessions/${sessionId}/artifacts/${artifactId}/download`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);

  if (!response.ok) {
    throw new Error(readErrorMessage(await response.text(), response.status));
  }

  return response.json() as Promise<T>;
}

function readErrorMessage(payload: string, status: number): string {
  try {
    const parsed = JSON.parse(payload) as { detail?: unknown };
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // Fall through to generic message.
  }

  return `Request failed with status ${status}`;
}

function parseSseBlock(block: string, onEvent: (event: ChatStreamEvent) => void) {
  const dataLines = block
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6));

  if (dataLines.length === 0) {
    return;
  }

  const parsed = JSON.parse(dataLines.join("\n")) as ChatStreamEvent;
  onEvent(parsed);
}
