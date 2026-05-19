from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.entities import AnalysisSession, Branch, Dataset, VersionNode, new_id, utc_now
from app.schemas.history import (
    BranchActionResponse,
    BranchCreate,
    BranchListResponse,
    ForkVersionRequest,
    HistoryResponse,
    HistoryVersion,
    VersionActionResponse,
)
from app.services.versioning import (
    active_branch,
    apply_version_to_dataset,
    branch_read,
    checkout_branch,
    current_version,
    dataset_read,
    sync_branch_pointer,
    version_read,
)

router = APIRouter(prefix="/api/sessions", tags=["history"])


@router.get("/{session_id}/branches", response_model=BranchListResponse)
def list_branches(session_id: str, db: Session = Depends(get_session)) -> BranchListResponse:
    session = _session_or_404(session_id, db)
    if session.active_branch_id is None:
        active_branch(session, db)
    branches = list(db.exec(select(Branch).where(Branch.session_id == session_id)).all())
    return BranchListResponse(
        branches=[branch_read(branch) for branch in branches],
        active_branch_id=session.active_branch_id,
    )


@router.post("/{session_id}/branches", response_model=BranchActionResponse, status_code=status.HTTP_201_CREATED)
def create_branch(
    session_id: str,
    payload: BranchCreate,
    db: Session = Depends(get_session),
) -> BranchActionResponse:
    session = _session_or_404(session_id, db)
    if _branch_by_name(session_id, payload.name, db) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Branch name already exists")

    from_version = _version_or_current(session, payload.from_version_id, db)
    branch = Branch(session_id=session_id, name=payload.name)
    db.add(branch)
    db.commit()
    db.refresh(branch)

    if from_version is not None:
        fork_version = _copy_version_to_branch(
            source=from_version,
            branch=branch,
            parent_version_id=None,
            mutation_summary=f"Branch created from version {from_version.id}",
            db=db,
        )
        branch.root_version_id = fork_version.id
        branch.current_version_id = fork_version.id
        db.add(branch)
        db.commit()
        db.refresh(branch)

    return BranchActionResponse(branch=branch_read(branch), datasets=_current_datasets(session_id, db))


@router.post("/{session_id}/branches/{branch_id}/checkout", response_model=BranchActionResponse)
def checkout_branch_endpoint(
    session_id: str,
    branch_id: str,
    db: Session = Depends(get_session),
) -> BranchActionResponse:
    session = _session_or_404(session_id, db)
    branch = _branch_or_404(session_id, branch_id, db)
    datasets = list(db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
    checkout_branch(session, branch, datasets, db)
    return BranchActionResponse(branch=branch_read(branch), datasets=_current_datasets(session_id, db))


@router.post("/{session_id}/versions/{version_id}/rollback", response_model=VersionActionResponse)
def rollback_version(
    session_id: str,
    version_id: str,
    db: Session = Depends(get_session),
) -> VersionActionResponse:
    session = _session_or_404(session_id, db)
    target = _version_or_404(session_id, version_id, db)
    branch = _branch_or_404(session_id, target.branch_id, db)
    dataset = _dataset_or_404(session_id, target.dataset_id, db)
    parent_id = branch.current_version_id or dataset.current_version_id

    rollback = _copy_version_to_branch(
        source=target,
        branch=branch,
        parent_version_id=parent_id,
        mutation_summary=f"Rollback to: {target.mutation_summary or target.label}",
        db=db,
    )
    apply_version_to_dataset(dataset, rollback)
    sync_branch_pointer(branch, rollback)
    session.active_branch_id = branch.id
    session.updated_at = utc_now()

    db.add(dataset)
    db.add(branch)
    db.add(session)
    db.commit()
    db.refresh(dataset)
    db.refresh(branch)
    db.refresh(rollback)

    return VersionActionResponse(
        branch=branch_read(branch),
        dataset=dataset_read(dataset, rollback),
        version=version_read(rollback),
    )


@router.post("/{session_id}/versions/{version_id}/fork", response_model=BranchActionResponse, status_code=status.HTTP_201_CREATED)
def fork_version(
    session_id: str,
    version_id: str,
    payload: ForkVersionRequest,
    db: Session = Depends(get_session),
) -> BranchActionResponse:
    session = _session_or_404(session_id, db)
    target = _version_or_404(session_id, version_id, db)
    if _branch_by_name(session_id, payload.name, db) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Branch name already exists")

    branch = Branch(session_id=session_id, name=payload.name)
    db.add(branch)
    db.commit()
    db.refresh(branch)

    forked = _copy_version_to_branch(
        source=target,
        branch=branch,
        parent_version_id=None,
        mutation_summary=f"Forked from: {target.mutation_summary or target.label}",
        db=db,
    )
    branch.root_version_id = forked.id
    branch.current_version_id = forked.id
    session.active_branch_id = branch.id
    session.updated_at = utc_now()

    dataset = _dataset_or_404(session_id, forked.dataset_id, db)
    apply_version_to_dataset(dataset, forked)
    db.add(branch)
    db.add(session)
    db.add(dataset)
    db.commit()

    return BranchActionResponse(branch=branch_read(branch), datasets=_current_datasets(session_id, db))


@router.get("/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str, db: Session = Depends(get_session)) -> HistoryResponse:
    session = _session_or_404(session_id, db)
    if session.active_branch_id is None:
        active_branch(session, db)
    versions = list(
        db.exec(
            select(VersionNode)
            .join(Dataset, Dataset.id == VersionNode.dataset_id)
            .where(Dataset.session_id == session_id)
            .order_by(VersionNode.created_at)
        ).all()
    )
    datasets = {
        dataset.id: dataset for dataset in db.exec(select(Dataset).where(Dataset.session_id == session_id)).all()
    }
    branches = {branch.id: branch for branch in db.exec(select(Branch).where(Branch.session_id == session_id)).all()}
    current_ids = {branch.current_version_id for branch in branches.values()} | {
        dataset.current_version_id for dataset in datasets.values()
    }

    return HistoryResponse(
        active_branch_id=session.active_branch_id,
        versions=[
            HistoryVersion(
                id=version.id,
                dataset_id=version.dataset_id,
                dataset_filename=datasets[version.dataset_id].original_filename if version.dataset_id in datasets else None,
                branch_id=version.branch_id,
                branch_name=branches.get(version.branch_id).name if branches.get(version.branch_id) else None,
                parent_version_id=version.parent_version_id,
                mutation_summary=version.mutation_summary,
                created_by_message_id=version.created_by_message_id,
                label=version.label,
                profile=version.profile,
                created_at=version.created_at,
                is_current=version.id in current_ids,
            )
            for version in versions
        ],
    )


def _session_or_404(session_id: str, db: Session) -> AnalysisSession:
    session = db.get(AnalysisSession, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


def _branch_or_404(session_id: str, branch_id: str, db: Session) -> Branch:
    branch = db.get(Branch, branch_id)
    if branch is None or branch.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Branch not found")
    return branch


def _dataset_or_404(session_id: str, dataset_id: str, db: Session) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or dataset.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


def _version_or_404(session_id: str, version_id: str, db: Session) -> VersionNode:
    version = db.get(VersionNode, version_id)
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    dataset = _dataset_or_404(session_id, version.dataset_id, db)
    if dataset.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    return version


def _branch_by_name(session_id: str, name: str, db: Session) -> Branch | None:
    return db.exec(select(Branch).where(Branch.session_id == session_id).where(Branch.name == name)).first()


def _version_or_current(
    session: AnalysisSession,
    version_id: str | None,
    db: Session,
) -> VersionNode | None:
    if version_id:
        return _version_or_404(session.id, version_id, db)
    branch = db.get(Branch, session.active_branch_id) if session.active_branch_id else None
    if branch and branch.current_version_id:
        return db.get(VersionNode, branch.current_version_id)
    return None


def _copy_version_to_branch(
    source: VersionNode,
    branch: Branch,
    parent_version_id: str | None,
    mutation_summary: str,
    db: Session,
) -> VersionNode:
    version = VersionNode(
        id=new_id(),
        dataset_id=source.dataset_id,
        branch_id=branch.id,
        parent_version_id=parent_version_id,
        label="branch",
        snapshot_path=source.snapshot_path,
        mutation_summary=mutation_summary,
        profile=source.profile,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def _current_datasets(session_id: str, db: Session) -> list:
    datasets = list(db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
    return [dataset_read(dataset, current_version(dataset, db)) for dataset in datasets]
