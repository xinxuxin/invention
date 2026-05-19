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
    metadata = artifact.artifact_metadata if isinstance(artifact.artifact_metadata, dict) else {}
    columns = metadata.get("columns")
    rows = metadata.get("rows")
    chart_spec = metadata.get("chart_spec")
    description = metadata.get("description")
    source_message_id = metadata.get("source_message_id")
    return ArtifactRead(
        id=artifact.id,
        name=artifact.name,
        kind=artifact.kind,
        type=str(metadata.get("type") or artifact.kind),
        title=str(metadata.get("title") or artifact.name),
        description=str(description) if description is not None else None,
        columns=_artifact_column_defs(columns),
        rows=rows if isinstance(rows, list) else [],
        chart_spec=chart_spec if isinstance(chart_spec, dict) else None,
        download_url=f"/api/sessions/{artifact.session_id}/artifacts/{artifact.id}/download",
        source_message_id=str(source_message_id) if source_message_id is not None else None,
        path=artifact.path,
        metadata=metadata,
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


def _artifact_column_defs(columns: Any) -> list[dict[str, Any]]:
    if not isinstance(columns, list):
        return []
    normalized: list[dict[str, Any]] = []
    for column in columns:
        if isinstance(column, Mapping):
            key = str(column.get("key") or column.get("name") or column.get("label") or "")
            if key:
                normalized.append(
                    {
                        "key": key,
                        "label": str(column.get("label") or key),
                        "type": str(column.get("type") or "value"),
                    }
                )
        elif column is not None:
            key = str(column)
            normalized.append({"key": key, "label": key, "type": "value"})
    return normalized
