from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.entities import AnalysisSession, Branch, Dataset, VersionNode, new_id, utc_now
from app.schemas.dataset import DatasetListResponse, DatasetRead, DatasetUploadResponse
from app.schemas.session import SessionCreate, SessionListResponse, SessionRead
from app.services.introspection import introspect_object
from app.services.versioning import (
    active_branch,
    branch_read,
    current_version,
    dataset_key,
    dataset_read,
    sync_branch_pointer,
    unique_dataset_key,
)
from app.storage.files import load_pickle, save_snapshot, save_upload

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_session)) -> SessionRead:
    analysis_session = AnalysisSession(name=payload.name)
    branch = Branch(session_id=analysis_session.id, name="main")
    analysis_session.active_branch_id = branch.id

    db.add(analysis_session)
    db.add(branch)
    db.commit()
    db.refresh(analysis_session)
    db.refresh(branch)

    return _session_read(analysis_session, [branch])


@router.get("", response_model=SessionListResponse)
def list_analysis_sessions(db: Session = Depends(get_session)) -> SessionListResponse:
    sessions = list(db.exec(select(AnalysisSession).order_by(AnalysisSession.updated_at.desc())).all())
    branches = list(db.exec(select(Branch)).all())
    branches_by_session: dict[str, list[Branch]] = {}
    for branch in branches:
        branches_by_session.setdefault(branch.session_id, []).append(branch)

    return SessionListResponse(
        sessions=[
            _session_read(analysis_session, branches_by_session.get(analysis_session.id, []))
            for analysis_session in sessions
        ]
    )


@router.get("/{session_id}", response_model=SessionRead)
def get_analysis_session(session_id: str, db: Session = Depends(get_session)) -> SessionRead:
    analysis_session = db.get(AnalysisSession, session_id)
    if analysis_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if analysis_session.active_branch_id is None:
        active_branch(analysis_session, db)
    _ensure_active_dataset(analysis_session, db)
    branches = db.exec(select(Branch).where(Branch.session_id == session_id)).all()
    return _session_read(analysis_session, list(branches))


@router.post("/{session_id}/datasets", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_datasets(
    session_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_session),
) -> DatasetUploadResponse:
    analysis_session = db.get(AnalysisSession, session_id)
    if analysis_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required")

    main_branch = _get_main_branch(session_id, db)
    existing_datasets = list(db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
    used_keys = {dataset_key(dataset) for dataset in existing_datasets}
    uploaded: list[DatasetRead] = []

    for upload in files:
        _validate_pickle_upload(upload)
        dataset_id = new_id()
        version_id = new_id()

        original_path = await save_upload(session_id, dataset_id, upload)
        try:
            value = load_pickle(original_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to load pickle file {Path(original_path).name}: {exc}",
            ) from exc

        profile = introspect_object(value)
        snapshot_path = save_snapshot(session_id, dataset_id, version_id, value)

        dataset = Dataset(
            id=dataset_id,
            session_id=session_id,
            dataset_key=unique_dataset_key(upload.filename or "uploaded.pkl", used_keys),
            original_filename=upload.filename or "uploaded.pkl",
            object_type=profile.get("object_type", type(value).__qualname__),
            module=profile.get("module"),
            original_path=str(original_path),
            current_snapshot_path=str(snapshot_path),
            current_version_id=version_id,
            profile=profile,
        )
        version = VersionNode(
            id=version_id,
            dataset_id=dataset_id,
            branch_id=main_branch.id,
            parent_version_id=None,
            label="initial",
            snapshot_path=str(snapshot_path),
            mutation_summary="Initial upload",
            profile=profile,
        )
        sync_branch_pointer(main_branch, version)

        db.add(dataset)
        db.add(version)
        db.add(main_branch)
        if analysis_session.active_dataset_id is None:
            analysis_session.active_dataset_id = dataset_id
        analysis_session.updated_at = utc_now()
        db.add(analysis_session)
        db.commit()
        db.refresh(dataset)
        db.refresh(version)
        uploaded.append(dataset_read(dataset, version))

    return DatasetUploadResponse(datasets=uploaded)


@router.get("/{session_id}/datasets", response_model=DatasetListResponse)
def list_datasets(session_id: str, db: Session = Depends(get_session)) -> DatasetListResponse:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    datasets = db.exec(select(Dataset).where(Dataset.session_id == session_id)).all()
    return DatasetListResponse(
        datasets=[dataset_read(dataset, _current_version(dataset, db)) for dataset in datasets]
    )


@router.get("/{session_id}/datasets/{dataset_id}", response_model=DatasetRead)
def get_dataset(session_id: str, dataset_id: str, db: Session = Depends(get_session)) -> DatasetRead:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return dataset_read(dataset, _current_version(dataset, db))


@router.post("/{session_id}/datasets/{dataset_id}/activate", response_model=SessionRead)
def activate_dataset(session_id: str, dataset_id: str, db: Session = Depends(get_session)) -> SessionRead:
    analysis_session = db.get(AnalysisSession, session_id)
    if analysis_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    analysis_session.active_dataset_id = dataset_id
    analysis_session.updated_at = utc_now()
    db.add(analysis_session)
    db.commit()
    db.refresh(analysis_session)

    branches = db.exec(select(Branch).where(Branch.session_id == session_id)).all()
    return _session_read(analysis_session, list(branches))


def _validate_pickle_upload(upload: UploadFile) -> None:
    if not upload.filename or not upload.filename.lower().endswith(".pkl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pkl files are supported",
        )


def _get_main_branch(session_id: str, db: Session) -> Branch:
    branch = db.exec(
        select(Branch).where(Branch.session_id == session_id).where(Branch.name == "main")
    ).first()

    if branch is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Main branch missing")

    return branch


def _current_version(dataset: Dataset, db: Session) -> VersionNode:
    try:
        return current_version(dataset, db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def _ensure_active_dataset(analysis_session: AnalysisSession, db: Session) -> None:
    if analysis_session.active_dataset_id:
        dataset = db.get(Dataset, analysis_session.active_dataset_id)
        if dataset is not None and dataset.session_id == analysis_session.id:
            return

    dataset = db.exec(select(Dataset).where(Dataset.session_id == analysis_session.id)).first()
    if dataset is None:
        return

    analysis_session.active_dataset_id = dataset.id
    db.add(analysis_session)
    db.commit()
    db.refresh(analysis_session)


def _session_read(analysis_session: AnalysisSession, branches: list[Branch]) -> SessionRead:
    return SessionRead(
        id=analysis_session.id,
        name=analysis_session.name,
        active_branch_id=analysis_session.active_branch_id,
        active_dataset_id=analysis_session.active_dataset_id,
        created_at=analysis_session.created_at,
        updated_at=analysis_session.updated_at,
        branches=[branch_read(branch) for branch in branches],
    )
