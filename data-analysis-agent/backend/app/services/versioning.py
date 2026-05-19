from collections.abc import Sequence
from pathlib import Path

from sqlmodel import Session, select

from app.models.entities import AnalysisSession, Branch, Dataset, VersionNode, utc_now
from app.schemas.dataset import DatasetRead, VersionNodeRead
from app.schemas.session import BranchRead
from app.storage.files import load_pickle


def branch_read(branch: Branch) -> BranchRead:
    return BranchRead(
        id=branch.id,
        name=branch.name,
        current_version_id=branch.current_version_id,
        root_version_id=branch.root_version_id,
        created_at=branch.created_at,
    )


def version_read(version: VersionNode) -> VersionNodeRead:
    return VersionNodeRead(
        id=version.id,
        branch_id=version.branch_id,
        parent_version_id=version.parent_version_id,
        label=version.label,
        snapshot_path=version.snapshot_path,
        mutation_summary=version.mutation_summary,
        created_by_message_id=version.created_by_message_id,
        created_at=version.created_at,
    )


def dataset_read(dataset: Dataset, version: VersionNode) -> DatasetRead:
    return DatasetRead(
        id=dataset.id,
        session_id=dataset.session_id,
        original_filename=dataset.original_filename,
        object_type=dataset.object_type,
        module=dataset.module,
        profile=dataset.profile,
        current_version=version_read(version),
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


def current_version(dataset: Dataset, db: Session) -> VersionNode:
    if dataset.current_version_id is None:
        raise ValueError("Dataset has no current version")

    version = db.get(VersionNode, dataset.current_version_id)
    if version is None:
        raise ValueError("Current version node missing")

    return version


def active_branch(session: AnalysisSession, db: Session, fallback_name: str = "main") -> Branch:
    if session.active_branch_id:
        branch = db.get(Branch, session.active_branch_id)
        if branch is not None:
            return branch

    branch = db.exec(
        select(Branch).where(Branch.session_id == session.id).where(Branch.name == fallback_name)
    ).first()
    if branch is None:
        raise ValueError(f"Branch not found: {fallback_name}")

    session.active_branch_id = branch.id
    db.add(session)
    db.commit()
    db.refresh(session)
    return branch


def latest_versions_for_branch(branch_id: str, db: Session) -> dict[str, VersionNode]:
    versions = list(
        db.exec(
            select(VersionNode)
            .where(VersionNode.branch_id == branch_id)
            .order_by(VersionNode.created_at)
        ).all()
    )
    latest: dict[str, VersionNode] = {}
    for version in versions:
        latest[version.dataset_id] = version
    return latest


def apply_version_to_dataset(dataset: Dataset, version: VersionNode) -> None:
    dataset.current_version_id = version.id
    dataset.current_snapshot_path = version.snapshot_path
    dataset.profile = version.profile
    dataset.object_type = version.profile.get("object_type", dataset.object_type)
    dataset.module = version.profile.get("module")
    dataset.updated_at = utc_now()


def checkout_branch(session: AnalysisSession, branch: Branch, datasets: Sequence[Dataset], db: Session) -> None:
    latest = latest_versions_for_branch(branch.id, db)
    for dataset in datasets:
        version = latest.get(dataset.id)
        if version is not None:
            apply_version_to_dataset(dataset, version)
            db.add(dataset)

    session.active_branch_id = branch.id
    session.updated_at = utc_now()
    db.add(session)
    db.commit()


def sync_branch_pointer(branch: Branch, version: VersionNode) -> None:
    branch.current_version_id = version.id
    if branch.root_version_id is None:
        branch.root_version_id = version.id


def version_value(version: VersionNode) -> object:
    return load_pickle(Path(version.snapshot_path))
