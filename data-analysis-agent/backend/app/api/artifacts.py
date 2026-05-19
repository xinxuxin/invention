from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, Response
from sqlmodel import Session

from app.db.session import get_session
from app.models.entities import Artifact
from app.services.artifacts import stable_download_filename

router = APIRouter(prefix="/api/sessions", tags=["artifacts"])


@router.get("/{session_id}/artifacts/{artifact_id}/content")
def get_artifact_content(
    session_id: str,
    artifact_id: str,
    db: Session = Depends(get_session),
) -> Response:
    artifact = _get_artifact(session_id, artifact_id, db)
    path = Path(artifact.path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")

    media_type = "text/csv" if artifact.kind == "csv" else "application/json"
    return Response(content=path.read_text(encoding="utf-8"), media_type=media_type)


@router.get("/{session_id}/artifacts/{artifact_id}/download")
def download_artifact(
    session_id: str,
    artifact_id: str,
    db: Session = Depends(get_session),
) -> FileResponse:
    artifact = _get_artifact(session_id, artifact_id, db)
    path = Path(artifact.path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found")

    media_type = "text/csv" if artifact.kind == "csv" else "application/json"
    return FileResponse(path=path, filename=stable_download_filename(artifact), media_type=media_type)


def _get_artifact(session_id: str, artifact_id: str, db: Session) -> Artifact:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None or artifact.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    return artifact
