from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisSession(SQLModel, table=True):
    __tablename__ = "analysis_sessions"

    id: str = Field(default_factory=new_id, primary_key=True)
    name: str | None = None
    active_branch_id: str | None = Field(default=None, foreign_key="branches.id")
    active_dataset_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class Branch(SQLModel, table=True):
    __tablename__ = "branches"

    id: str = Field(default_factory=new_id, primary_key=True)
    session_id: str = Field(foreign_key="analysis_sessions.id", index=True)
    name: str = Field(index=True)
    current_version_id: str | None = Field(default=None, foreign_key="version_nodes.id")
    root_version_id: str | None = Field(default=None, foreign_key="version_nodes.id")
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    id: str = Field(default_factory=new_id, primary_key=True)
    session_id: str = Field(foreign_key="analysis_sessions.id", index=True)
    dataset_key: str = Field(default="", index=True)
    original_filename: str
    object_type: str
    module: str | None = None
    original_path: str
    current_snapshot_path: str
    current_version_id: str | None = Field(default=None, index=True)
    profile: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


class VersionNode(SQLModel, table=True):
    __tablename__ = "version_nodes"

    id: str = Field(default_factory=new_id, primary_key=True)
    dataset_id: str = Field(foreign_key="datasets.id", index=True)
    branch_id: str = Field(foreign_key="branches.id", index=True)
    parent_version_id: str | None = Field(default=None, foreign_key="version_nodes.id")
    label: str = "mutation"
    snapshot_path: str
    mutation_summary: str | None = None
    created_by_message_id: str | None = None
    profile: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class Artifact(SQLModel, table=True):
    __tablename__ = "artifacts"

    id: str = Field(default_factory=new_id, primary_key=True)
    session_id: str = Field(foreign_key="analysis_sessions.id", index=True)
    dataset_id: str | None = Field(default=None, foreign_key="datasets.id", index=True)
    version_id: str | None = Field(default=None, foreign_key="version_nodes.id", index=True)
    name: str
    kind: str = Field(index=True)
    path: str
    artifact_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
