export type HealthResponse = {
  status: string;
  service: string;
};

export type Branch = {
  id: string;
  name: string;
  created_at: string;
};

export type AnalysisSession = {
  id: string;
  name: string | null;
  created_at: string;
  updated_at: string;
  branches: Branch[];
};

export type VersionNode = {
  id: string;
  branch_id: string;
  parent_id: string | null;
  label: string;
  snapshot_path: string;
  created_at: string;
};

export type ObjectProfile = {
  object_type: string;
  module?: string | null;
  repr_preview?: string;
  approximate_size?: number | null;
  shape?: unknown;
  columns?: unknown[];
  dtypes?: Record<string, string>;
  index_preview?: unknown[];
  sample_rows?: Record<string, unknown>[];
  keys?: unknown[];
  length?: number;
  sample_items?: unknown[];
  sample?: unknown;
  dtype?: string;
  public_attributes?: Record<string, unknown>;
  nested_summary?: Record<string, unknown>;
  warnings?: string[];
  [key: string]: unknown;
};

export type Dataset = {
  id: string;
  session_id: string;
  original_filename: string;
  object_type: string;
  module?: string | null;
  profile: ObjectProfile;
  current_version: VersionNode;
  created_at: string;
  updated_at: string;
};

export type DatasetListResponse = {
  datasets: Dataset[];
};

export type DatasetUploadResponse = {
  datasets: Dataset[];
};

export type ChatHistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ExecutionArtifact = {
  id: string;
  name: string;
  kind: "table" | "chart" | "csv" | string;
  path: string;
  metadata: Record<string, unknown>;
};

export type UpdatedDataset = {
  dataset_id: string;
  key: string;
  version_id: string;
  profile: ObjectProfile;
  mutation_summary: string;
};

export type ChatStreamEvent =
  | { type: "message_started" }
  | { type: "trace"; message: string }
  | { type: "code_started"; code: string }
  | {
      type: "code_result_summary";
      ok: boolean;
      stdout?: string;
      stderr?: string;
      traceback?: string | null;
      result_preview?: unknown;
      updated_datasets?: UpdatedDataset[];
    }
  | {
      type: "confirmation_required";
      message: string;
      code?: string | null;
      mutation_summary?: string | null;
    }
  | { type: "artifact_created"; artifact: ExecutionArtifact }
  | { type: "final_answer"; answer: string; state_changed?: boolean }
  | { type: "message_done" }
  | { type: "error"; message: string };

export type ChatStreamRequest = {
  message: string;
  active_dataset_id?: string | null;
  branch_name?: string;
  conversation_history?: ChatHistoryMessage[];
  confirmed?: boolean;
};
