from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ArtifactRead(BaseModel):
    id: str
    name: str
    kind: str
    type: str | None = None
    title: str | None = None
    description: str | None = None
    columns: list[dict[str, Any]] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    chart_spec: dict[str, Any] | None = None
    download_url: str | None = None
    source_message_id: str | None = None
    status: str | None = None
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


ChartType = Literal["bar", "line", "pie", "scatter", "area"]


class ChartArtifactSpec(BaseModel):
    id: str | None = None
    title: str
    chart_type: ChartType
    data: list[dict[str, Any]]
    x: str
    y: str
    series: str | None = None
    color: str | None = None
    description: str | None = None

    @field_validator("data")
    @classmethod
    def require_data(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("Chart data must contain at least one row")
        return value

    @field_validator("x", "y")
    @classmethod
    def require_field_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chart field names cannot be empty")
        return value
