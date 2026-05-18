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
