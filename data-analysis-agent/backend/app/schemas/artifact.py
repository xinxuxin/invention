from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ArtifactRead(BaseModel):
    id: str
    name: str
    kind: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExportRequest(BaseModel):
    dataset_id: str | None = None
    version_id: str | None = None
    name: str | None = Field(default=None, max_length=120)


class ExportResponse(BaseModel):
    artifact: ArtifactRead | None
    message: str
    ok: bool = True
