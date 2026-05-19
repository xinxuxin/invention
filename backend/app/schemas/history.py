from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.dataset import DatasetRead, VersionNodeRead
from app.schemas.session import BranchRead


class BranchCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    from_version_id: str | None = None


class BranchListResponse(BaseModel):
    branches: list[BranchRead]
    active_branch_id: str | None


class BranchActionResponse(BaseModel):
    branch: BranchRead
    datasets: list[DatasetRead]


class VersionActionResponse(BaseModel):
    branch: BranchRead
    dataset: DatasetRead
    version: VersionNodeRead


class ForkVersionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class HistoryVersion(BaseModel):
    id: str
    dataset_id: str
    dataset_filename: str | None
    branch_id: str
    branch_name: str | None
    parent_version_id: str | None
    mutation_summary: str | None
    created_by_message_id: str | None
    label: str
    profile: dict[str, Any]
    created_at: datetime
    is_current: bool


class HistoryResponse(BaseModel):
    active_branch_id: str | None
    versions: list[HistoryVersion]
