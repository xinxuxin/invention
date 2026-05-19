from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class BranchRead(BaseModel):
    id: str
    name: str
    current_version_id: str | None
    root_version_id: str | None
    created_at: datetime


class SessionRead(BaseModel):
    id: str
    name: str | None
    active_branch_id: str | None
    active_dataset_id: str | None
    created_at: datetime
    updated_at: datetime
    branches: list[BranchRead]
