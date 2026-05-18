from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.entities import AnalysisSession, Branch, Dataset, VersionNode, new_id, utc_now
from app.schemas.dataset import DatasetListResponse, DatasetRead, DatasetUploadResponse, VersionNodeRead
from app.schemas.session import BranchRead, SessionCreate, SessionRead
from app.services.introspection import introspect_object
from app.storage.files import load_pickle, save_snapshot, save_upload

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate, db: Session = Depends(get_session)) -> SessionRead:
    analysis_session = AnalysisSession(name=payload.name)
    branch = Branch(session_id=analysis_session.id, name="main")

    db.add(analysis_session)
    db.add(branch)
    db.commit()
    db.refresh(analysis_session)
    db.refresh(branch)

    return _session_read(analysis_session, [branch])


@router.get("/{session_id}", response_model=SessionRead)
def get_analysis_session(session_id: str, db: Session = Depends(get_session)) -> SessionRead:
    analysis_session = db.get(AnalysisSession, session_id)
    if analysis_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

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
            parent_id=None,
            label="initial",
            snapshot_path=str(snapshot_path),
            profile=profile,
        )

        db.add(dataset)
        db.add(version)
        analysis_session.updated_at = utc_now()
        db.add(analysis_session)
        db.commit()
        db.refresh(dataset)
        db.refresh(version)
        uploaded.append(_dataset_read(dataset, version))

    return DatasetUploadResponse(datasets=uploaded)


@router.get("/{session_id}/datasets", response_model=DatasetListResponse)
def list_datasets(session_id: str, db: Session = Depends(get_session)) -> DatasetListResponse:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    datasets = db.exec(select(Dataset).where(Dataset.session_id == session_id)).all()
    return DatasetListResponse(
        datasets=[_dataset_read(dataset, _current_version(dataset, db)) for dataset in datasets]
    )


@router.get("/{session_id}/datasets/{dataset_id}", response_model=DatasetRead)
def get_dataset(session_id: str, dataset_id: str, db: Session = Depends(get_session)) -> DatasetRead:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return _dataset_read(dataset, _current_version(dataset, db))


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
    if dataset.current_version_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dataset has no current version",
        )

    version = db.get(VersionNode, dataset.current_version_id)
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Current version node missing",
        )

    return version


def _session_read(analysis_session: AnalysisSession, branches: list[Branch]) -> SessionRead:
    return SessionRead(
        id=analysis_session.id,
        name=analysis_session.name,
        created_at=analysis_session.created_at,
        updated_at=analysis_session.updated_at,
        branches=[BranchRead(id=branch.id, name=branch.name, created_at=branch.created_at) for branch in branches],
    )


def _dataset_read(dataset: Dataset, version: VersionNode) -> DatasetRead:
    return DatasetRead(
        id=dataset.id,
        session_id=dataset.session_id,
        original_filename=dataset.original_filename,
        object_type=dataset.object_type,
        module=dataset.module,
        profile=dataset.profile,
        current_version=VersionNodeRead(
            id=version.id,
            branch_id=version.branch_id,
            parent_id=version.parent_id,
            label=version.label,
            snapshot_path=version.snapshot_path,
            created_at=version.created_at,
        ),
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )
