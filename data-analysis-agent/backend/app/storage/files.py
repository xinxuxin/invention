from pathlib import Path
from uuid import uuid4

import cloudpickle
from fastapi import UploadFile

from app.core.config import get_settings


def storage_root() -> Path:
    root = Path(get_settings().storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_root(session_id: str) -> Path:
    root = storage_root() / "sessions" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def originals_root(session_id: str, dataset_id: str) -> Path:
    root = session_root(session_id) / "datasets" / dataset_id / "originals"
    root.mkdir(parents=True, exist_ok=True)
    return root


def snapshots_root(session_id: str, dataset_id: str) -> Path:
    root = session_root(session_id) / "datasets" / dataset_id / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifacts_root(session_id: str) -> Path:
    root = session_root(session_id) / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    return candidate or "uploaded.pkl"


async def save_upload(session_id: str, dataset_id: str, upload: UploadFile) -> Path:
    filename = safe_filename(upload.filename or "uploaded.pkl")
    destination = originals_root(session_id, dataset_id) / f"{uuid4()}-{filename}"

    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            target.write(chunk)

    await upload.close()
    return destination


def load_pickle(path: Path) -> object:
    with path.open("rb") as source:
        return cloudpickle.load(source)


def save_snapshot(session_id: str, dataset_id: str, version_id: str, value: object) -> Path:
    destination = snapshots_root(session_id, dataset_id) / f"{version_id}.pkl"
    with destination.open("wb") as target:
        cloudpickle.dump(value, target)

    return destination
