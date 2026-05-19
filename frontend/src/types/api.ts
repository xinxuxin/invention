export type HealthResponse = {
  status: string;
  service: string;
};

export type HealthConfigResponse = {
  agent_mode: "real" | "fake" | string;
  agent_model_mode: string;
  fake_agent_mode: boolean;
  llm_verifier_enabled: boolean;
  verifier_mode: string;
  llm_verifier_model: string;
  llm_verifier_timeout_seconds: number;
  verifier_skip_llm_after_step: number;
  has_openai_api_key: boolean;
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
  active_version_id?: string | null;
  dataset_count?: number;
  message_count?: number;
  active_dataset_name?: string | null;
  active_branch_name?: string | null;
  created_at: string;
  updated_at: string;
  branches: Branch[];
};

export type SessionListResponse = {
  sessions: AnalysisSession[];
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
  type?: "table" | "chart" | "csv" | "json" | string | null;
  title?: string | null;
  description?: string | null;
  columns?: Array<Record<string, unknown>>;
  rows?: Array<Record<string, unknown>>;
  chart_spec?: Record<string, unknown> | null;
  payload?: Record<string, unknown> | null;
  download_url?: string | null;
  source_message_id?: string | null;
  status?: string | null;
  path: string;
  metadata: Record<string, unknown>;
  created_at?: string | null;
};

export type PersistedChatTraceEvent = {
  id: string;
  type: ChatStreamEvent["type"] | string;
  message?: string | null;
  code?: string | null;
  ok?: boolean | null;
  stdout?: string | null;
  stderr?: string | null;
  traceback?: string | null;
  result_summary?: Record<string, unknown> | null;
  result_preview?: unknown;
  updated_datasets?: UpdatedDataset[];
  severity?: string | null;
  source?: string | null;
};

export type PersistedChatMessage = {
  id: string;
  session_id: string;
  role: "user" | "assistant" | string;
  content: string;
  status: "streaming" | "done" | "error" | "waiting_confirmation" | string;
  final_answer?: string | null;
  highlights?: Array<Record<string, unknown>>;
  key_findings?: string[];
  warnings?: string[];
  state_changed?: boolean | null;
  artifact_ids?: string[];
  trace_events?: PersistedChatTraceEvent[];
  artifacts?: ExecutionArtifact[];
  created_at: string;
  updated_at: string;
};

export type ChatMessageListResponse = {
  messages: PersistedChatMessage[];
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
      result_summary?: Record<string, unknown>;
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
      title?: string | null;
      dataset_name?: string | null;
      expected_effect?: string | null;
      affected_count?: number | null;
      current_row_count?: number | null;
      new_row_count?: number | null;
      state_impact?: string | null;
      reversible?: boolean | null;
      rollback_note?: string | null;
      proposed_code?: string | null;
      confirm_label?: string | null;
      cancel_label?: string | null;
      risk_level?: "low" | "medium" | "high" | string;
      affected_dataset_ids?: string[];
    }
  | { type: "artifact_created"; artifact: ExecutionArtifact }
  | {
      type: "final_answer";
      answer: string;
      state_changed?: boolean;
      highlights?: Array<Record<string, unknown>>;
      key_findings?: string[];
      warnings?: string[];
      artifact_ids?: string[];
    }
  | {
      type: "verifier_result";
      message?: string;
      passed?: boolean;
      severity?: string;
      source?: string;
      reasons?: string[];
    }
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
