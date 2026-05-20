from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.artifact import ArtifactRead


class ChatTraceEventRead(BaseModel):
    id: str
    type: str
    message: str | None = None
    code: str | None = None
    ok: bool | None = None
    stdout: str | None = None
    stderr: str | None = None
    traceback: str | None = None
    result_summary: dict[str, Any] | None = None
    result_preview: Any | None = None
    updated_datasets: list[dict[str, Any]] = Field(default_factory=list)
    severity: str | None = None
    source: str | None = None


class ChatMessageRead(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    status: str
    final_answer: str | None = None
    highlights: list[dict[str, Any]] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    state_changed: bool | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    trace_events: list[ChatTraceEventRead] = Field(default_factory=list)
    artifacts: list[ArtifactRead] = Field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class ChatMessageListResponse(BaseModel):
    messages: list[ChatMessageRead]
