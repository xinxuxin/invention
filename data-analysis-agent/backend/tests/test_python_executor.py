import time
from pathlib import Path

import pandas as pd
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.config import get_settings
from app.models.entities import AnalysisSession, Artifact, Branch, Dataset, VersionNode, new_id
from app.runtime.python_executor import PythonExecutor
from app.services.introspection import introspect_object
from app.storage.files import load_pickle, save_snapshot


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    storage_path = tmp_path / ".data"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_DIR", str(storage_path))
    get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    get_settings.cache_clear()


def test_executor_can_inspect_dataframe(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "print(data.shape)\npreview(data.head(2))",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert "(3, 2)" in result.stdout
    assert result.traceback is None
    assert isinstance(result.result_preview, dict)
    assert result.result_preview["object_type"] == "DataFrame"
    assert result.result_preview["shape"] == [2, 2]


def test_read_only_execution_does_not_create_version(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "_result = data['value'].sum()\nprint(_result)",
        active_dataset_id=dataset_id,
        mutates_state=False,
    )

    versions = db_session.exec(select(VersionNode).where(VersionNode.dataset_id == dataset_id)).all()
    assert result.ok is True
    assert result.stdout.strip() == "6"
    assert result.updated_datasets == []
    assert len(versions) == 1


def test_mutating_execution_creates_new_version(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "data['doubled'] = data['value'] * 2\npreview(data)",
        active_dataset_id=dataset_id,
        mutates_state=True,
        mutation_summary="Add doubled column",
    )

    dataset = db_session.get(Dataset, dataset_id)
    versions = db_session.exec(select(VersionNode).where(VersionNode.dataset_id == dataset_id)).all()
    persisted = load_pickle(Path(dataset.current_snapshot_path))

    assert result.ok is True
    assert len(result.updated_datasets) == 1
    assert result.updated_datasets[0].mutation_summary == "Add doubled column"
    assert len(versions) == 2
    assert dataset.current_version_id == result.updated_datasets[0].version_id
    assert "doubled" in persisted.columns
    assert dataset.profile["columns"] == ["value", "group", "doubled"]
    assert dataset.profile["_mutation"]["summary"] == "Add doubled column"


def test_dataset_assignment_mutation_creates_new_version(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session, filename="frame.pkl")
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "datasets['frame.pkl'] = data.assign(offset=data['value'] + 10)",
        active_dataset_id=dataset_id,
        mutates_state=True,
    )

    dataset = db_session.get(Dataset, dataset_id)
    persisted = load_pickle(Path(dataset.current_snapshot_path))

    assert result.ok is True
    assert len(result.updated_datasets) == 1
    assert "offset" in persisted.columns


def test_exceptions_return_traceback(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(session_id, "print('before')\n1 / 0", active_dataset_id=dataset_id)

    assert result.ok is False
    assert result.stdout.strip() == "before"
    assert result.traceback is not None
    assert "ZeroDivisionError" in result.traceback


def test_result_preview_is_json_safe(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "preview({'scalar': data['value'].iloc[0], 'stamp': pd.Timestamp('2026-05-18')})",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert isinstance(result.result_preview, dict)
    assert result.result_preview["object_type"] == "dict"
    assert result.result_preview["sample_items"][0]["key"] == "scalar"


def test_artifact_helpers_store_files_and_metadata(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "save_csv('values export', data)",
                "save_table('values table', data.head(2))",
                "save_chart('values chart', {'mark': 'bar', 'encoding': {'x': 'group'}})",
                "preview(data)",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    artifacts = db_session.exec(select(Artifact).where(Artifact.session_id == session_id)).all()

    assert result.ok is True
    assert {artifact.kind for artifact in result.artifacts} == {"csv", "table", "chart"}
    assert len(artifacts) == 3
    for artifact in artifacts:
        assert Path(artifact.path).exists()
        assert artifact.artifact_metadata


def test_timeout_returns_failure(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session, timeout_seconds=1)
    started = time.monotonic()

    result = executor.execute(session_id, "while True:\n    pass", active_dataset_id=dataset_id)

    assert result.ok is False
    assert "timed out" in result.stderr
    assert time.monotonic() - started < 3


def test_network_import_is_blocked(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(session_id, "import socket", active_dataset_id=dataset_id)

    assert result.ok is False
    assert result.traceback is not None
    assert "blocked" in result.traceback


def _create_dataset(
    db_session: Session,
    *,
    filename: str = "dataset.pkl",
    value: pd.DataFrame | None = None,
) -> tuple[str, str]:
    frame = value if value is not None else pd.DataFrame({"value": [1, 2, 3], "group": ["a", "b", "a"]})
    analysis_session = AnalysisSession(name="Runtime test")
    branch = Branch(session_id=analysis_session.id, name="main")
    dataset_id = new_id()
    version_id = new_id()

    snapshot_path = save_snapshot(analysis_session.id, dataset_id, version_id, frame)
    profile = introspect_object(frame)
    dataset = Dataset(
        id=dataset_id,
        session_id=analysis_session.id,
        original_filename=filename,
        object_type=profile["object_type"],
        module=profile["module"],
        original_path=str(snapshot_path),
        current_snapshot_path=str(snapshot_path),
        current_version_id=version_id,
        profile=profile,
    )
    version = VersionNode(
        id=version_id,
        dataset_id=dataset_id,
        branch_id=branch.id,
        parent_id=None,
        label="initial",
        snapshot_path=str(snapshot_path),
        profile=profile,
    )

    db_session.add(analysis_session)
    db_session.add(branch)
    db_session.add(dataset)
    db_session.add(version)
    db_session.commit()

    return analysis_session.id, dataset_id
