from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.models.entities import Artifact, new_id
from app.schemas.artifact import ArtifactRead
from app.storage.files import artifacts_root

MAX_ARTIFACT_BYTES = 2_000_000


def persist_artifact(
    db: Session,
    *,
    session_id: str,
    name: str,
    kind: str,
    content: str | bytes,
    metadata: Mapping[str, Any] | None = None,
    dataset_id: str | None = None,
    version_id: str | None = None,
) -> Artifact:
    artifact_id = new_id()
    extension = artifact_extension(kind)
    path = artifacts_root(session_id) / f"{artifact_id}-{safe_artifact_name(name)}.{extension}"
    if isinstance(content, bytes):
        path.write_bytes(content[:MAX_ARTIFACT_BYTES])
    else:
        path.write_text(content[:MAX_ARTIFACT_BYTES], encoding="utf-8")

    artifact = Artifact(
        id=artifact_id,
        session_id=session_id,
        dataset_id=dataset_id,
        version_id=version_id,
        name=name,
        kind=kind,
        path=str(path),
        artifact_metadata=dict(metadata or {}),
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def artifact_read(artifact: Artifact) -> ArtifactRead:
    return ArtifactRead(
        id=artifact.id,
        name=artifact.name,
        kind=artifact.kind,
        path=artifact.path,
        metadata=artifact.artifact_metadata,
        created_at=artifact.created_at,
    )


def artifact_extension(kind: str) -> str:
    if kind == "csv":
        return "csv"
    if kind in {"table", "chart", "json"}:
        return "json"
    return "txt"


def safe_artifact_name(name: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in name)
    return sanitized.strip("-")[:80] or "artifact"


def stable_download_filename(artifact: Artifact) -> str:
    extension = artifact_extension(artifact.kind)
    return f"{Path(artifact.path).stem}.{extension}"
