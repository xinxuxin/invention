from datetime import datetime
from typing import Any

from pydantic import BaseModel


class VersionNodeRead(BaseModel):
    id: str
    branch_id: str
    parent_version_id: str | None
    label: str
    snapshot_path: str
    mutation_summary: str | None
    created_by_message_id: str | None
    created_at: datetime


class DatasetRead(BaseModel):
    id: str
    session_id: str
    dataset_key: str
    original_filename: str
    object_type: str
    module: str | None
    profile: dict[str, Any]
    current_version: VersionNodeRead
    created_at: datetime
    updated_at: datetime


class DatasetListResponse(BaseModel):
    datasets: list[DatasetRead]


class DatasetUploadResponse(BaseModel):
    datasets: list[DatasetRead]
