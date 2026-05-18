from datetime import datetime
from typing import Any

from pydantic import BaseModel


class VersionNodeRead(BaseModel):
    id: str
    branch_id: str
    parent_id: str | None
    label: str
    snapshot_path: str
    created_at: datetime


class DatasetRead(BaseModel):
    id: str
    session_id: str
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
