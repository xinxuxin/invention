from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConfirmationRead(BaseModel):
    id: str
    session_id: str
    proposed_code: str
    operation_summary: str
    affected_dataset_ids: list[str] = Field(default_factory=list)
    risk_level: str
    status: str
    active_dataset_id: str | None = None
    branch_name: str
    created_at: datetime
    resolved_at: datetime | None = None


class ConfirmationActionResponse(BaseModel):
    confirmation: ConfirmationRead
    events: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
