from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db.session import get_session
from app.schemas.artifact import ExportRequest, ExportResponse
from app.services.artifacts import artifact_read
from app.services.export import export_dataset_csv

router = APIRouter(prefix="/api/sessions", tags=["export"])


@router.post("/{session_id}/export", response_model=ExportResponse)
def export_session_dataset(
    session_id: str,
    payload: ExportRequest,
    db: Session = Depends(get_session),
) -> ExportResponse:
    try:
        result = export_dataset_csv(
            db,
            session_id=session_id,
            dataset_id=payload.dataset_id,
            version_id=payload.version_id,
            name=payload.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ExportResponse(
        artifact=artifact_read(result.artifact) if result.artifact else None,
        message=result.message,
        ok=result.ok,
    )
