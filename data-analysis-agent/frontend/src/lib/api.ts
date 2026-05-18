import type {
  AnalysisSession,
  DatasetListResponse,
  DatasetUploadResponse,
  HealthResponse,
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
