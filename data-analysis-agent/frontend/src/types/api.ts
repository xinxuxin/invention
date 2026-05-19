export type HealthResponse = {
  status: string;
  service: string;
};

export type Branch = {
  id: string;
  name: string;
  current_version_id: string | null;
  root_version_id: string | null;
  created_at: string;
};

export type AnalysisSession = {
  id: string;
  name: string | null;
  active_branch_id: string | null;
  active_dataset_id: string | null;
  created_at: string;
  updated_at: string;
  branches: Branch[];
};

export type VersionNode = {
  id: string;
  branch_id: string;
  parent_version_id: string | null;
  label: string;
  snapshot_path: string;
  mutation_summary: string | null;
  created_by_message_id: string | null;
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
  dataset_key: string;
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

export type BranchListResponse = {
  branches: Branch[];
  active_branch_id: string | null;
};

export type BranchActionResponse = {
  branch: Branch;
  datasets: Dataset[];
};

export type VersionActionResponse = {
  branch: Branch;
  dataset: Dataset;
  version: VersionNode;
};

export type HistoryVersion = {
  id: string;
  dataset_id: string;
  dataset_filename: string | null;
  branch_id: string;
  branch_name: string | null;
  parent_version_id: string | null;
  mutation_summary: string | null;
  created_by_message_id: string | null;
  label: string;
  profile: ObjectProfile;
  created_at: string;
  is_current: boolean;
};

export type HistoryResponse = {
  active_branch_id: string | null;
  versions: HistoryVersion[];
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
  created_at?: string | null;
};

export type ExportResponse = {
  artifact: ExecutionArtifact | null;
  message: string;
  ok: boolean;
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
      confirmation_id?: string | null;
      message: string;
      code?: string | null;
      mutation_summary?: string | null;
      operation_summary?: string | null;
      risk_level?: "low" | "medium" | "high" | string;
      affected_dataset_ids?: string[];
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

export type ConfirmationRead = {
  id: string;
  session_id: string;
  proposed_code: string;
  operation_summary: string;
  affected_dataset_ids: string[];
  risk_level: "low" | "medium" | "high" | string;
  status: "pending" | "approved" | "rejected" | "failed" | string;
  active_dataset_id?: string | null;
  branch_name: string;
  created_at: string;
  resolved_at?: string | null;
};

export type ConfirmationActionResponse = {
  confirmation: ConfirmationRead;
  events: ChatStreamEvent[];
  result?: Record<string, unknown> | null;
};
