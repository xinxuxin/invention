from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.entities import AnalysisSession, Artifact, Branch, ChatMessage, Dataset, VersionNode, new_id, utc_now
from app.schemas.dataset import DatasetListResponse, DatasetRead, DatasetUploadResponse
from app.schemas.artifact import ArtifactRead
from app.schemas.message import ChatMessageListResponse, ChatMessageRead, ChatTraceEventRead
from app.schemas.session import SessionCreate, SessionListResponse, SessionRead
from app.services.artifacts import artifact_read
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

    return _session_read(analysis_session, [branch], db)


@router.get("", response_model=SessionListResponse)
def list_analysis_sessions(db: Session = Depends(get_session)) -> SessionListResponse:
    sessions = list(db.exec(select(AnalysisSession).order_by(AnalysisSession.updated_at.desc())).all())
    branches = list(db.exec(select(Branch)).all())
    branches_by_session: dict[str, list[Branch]] = {}
    for branch in branches:
        branches_by_session.setdefault(branch.session_id, []).append(branch)

    return SessionListResponse(
        sessions=[
            _session_read(analysis_session, branches_by_session.get(analysis_session.id, []), db)
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
    return _session_read(analysis_session, list(branches), db)


@router.get("/{session_id}/messages", response_model=ChatMessageListResponse)
def list_chat_messages(session_id: str, db: Session = Depends(get_session)) -> ChatMessageListResponse:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = list(
        db.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        ).all()
    )
    artifact_ids = {
        artifact_id
        for message in messages
        for artifact_id in _json_list(message.artifact_ids)
        if isinstance(artifact_id, str)
    }
    artifacts = []
    if artifact_ids:
        artifacts = list(
            db.exec(
                select(Artifact)
                .where(Artifact.session_id == session_id)
                .where(Artifact.id.in_(artifact_ids))
            ).all()
        )
    artifacts_by_id = {artifact.id: artifact_read(artifact) for artifact in artifacts}

    return ChatMessageListResponse(
        messages=[_chat_message_read(message, artifacts_by_id) for message in messages]
    )


@router.get("/{session_id}/artifacts", response_model=list[ArtifactRead])
def list_session_artifacts(
    session_id: str,
    include_all: bool = Query(default=False),
    db: Session = Depends(get_session),
) -> list[ArtifactRead]:
    if db.get(AnalysisSession, session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    artifacts = list(
        db.exec(
            select(Artifact)
            .where(Artifact.session_id == session_id)
            .order_by(Artifact.created_at.desc())
        ).all()
    )
    if not include_all:
        artifacts = [
            artifact
            for artifact in artifacts
            if str((artifact.artifact_metadata or {}).get("status") or "verified")
            not in {"discarded", "superseded", "pending_verification"}
        ]
    return [artifact_read(artifact) for artifact in artifacts]


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
    return _session_read(analysis_session, list(branches), db)


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


def _session_read(analysis_session: AnalysisSession, branches: list[Branch], db: Session | None = None) -> SessionRead:
    active_branch = next((branch for branch in branches if branch.id == analysis_session.active_branch_id), None)
    active_dataset_name = None
    dataset_count = 0
    message_count = 0
    if db is not None:
        datasets = list(db.exec(select(Dataset).where(Dataset.session_id == analysis_session.id)).all())
        dataset_count = len(datasets)
        active_dataset = next(
            (dataset for dataset in datasets if dataset.id == analysis_session.active_dataset_id),
            None,
        )
        active_dataset_name = active_dataset.original_filename if active_dataset else None
        message_count = len(list(db.exec(select(ChatMessage.id).where(ChatMessage.session_id == analysis_session.id)).all()))

    return SessionRead(
        id=analysis_session.id,
        name=analysis_session.name,
        active_branch_id=analysis_session.active_branch_id,
        active_dataset_id=analysis_session.active_dataset_id,
        active_version_id=active_branch.current_version_id if active_branch else None,
        dataset_count=dataset_count,
        message_count=message_count,
        active_dataset_name=active_dataset_name,
        active_branch_name=active_branch.name if active_branch else None,
        created_at=analysis_session.created_at,
        updated_at=analysis_session.updated_at,
        branches=[branch_read(branch) for branch in branches],
    )


def _chat_message_read(
    message: ChatMessage,
    artifacts_by_id: dict[str, ArtifactRead],
) -> ChatMessageRead:
    trace_events = [
        ChatTraceEventRead.model_validate(event)
        for event in _json_list(message.trace_events)
        if isinstance(event, dict) and event.get("type")
    ]
    artifact_ids = [artifact_id for artifact_id in _json_list(message.artifact_ids) if isinstance(artifact_id, str)]
    return ChatMessageRead(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        status=message.status,
        final_answer=message.final_answer,
        highlights=[item for item in _json_list(message.highlights) if isinstance(item, dict)],
        key_findings=[str(item) for item in _json_list(message.key_findings)],
        warnings=[str(item) for item in _json_list(message.warnings)],
        state_changed=message.state_changed,
        artifact_ids=artifact_ids,
        trace_events=trace_events,
        artifacts=[artifacts_by_id[artifact_id] for artifact_id in artifact_ids if artifact_id in artifacts_by_id],
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def _json_list(value: object) -> list:
    return value if isinstance(value, list) else []
