import json
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


def test_executor_allows_generic_object_introspection_builtins(db_session: Session) -> None:
    class DemoObject:
        def __init__(self) -> None:
            self.alpha = 7

    session_id, dataset_id = _create_dataset(db_session, filename="custom.pkl", value=DemoObject())
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "print('alpha' in dir(data))",
                "print(vars(data)['alpha'])",
                "print(callable(data))",
                "preview({'public': [name for name in dir(data) if not name.startswith('_')]})",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.stdout.splitlines() == ["True", "7", "False"]
    assert result.result_preview is not None


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
        "datasets['frame'] = data.assign(offset=data['value'] + 10)",
        active_dataset_id=dataset_id,
        mutates_state=True,
    )

    dataset = db_session.get(Dataset, dataset_id)
    persisted = load_pickle(Path(dataset.current_snapshot_path))

    assert result.ok is True
    assert len(result.updated_datasets) == 1
    assert "offset" in persisted.columns


def test_multi_dataset_mutation_updates_only_target_dataset(db_session: Session) -> None:
    session_id, first_id, second_id = _create_two_datasets(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "print(sorted(datasets.keys()))\ndata = data.assign(double=data['value'] * 2)\npreview(datasets)",
        active_dataset_id=first_id,
        mutates_state=True,
        mutation_summary="Mutate first dataset only",
    )

    first = db_session.get(Dataset, first_id)
    second = db_session.get(Dataset, second_id)
    first_versions = db_session.exec(select(VersionNode).where(VersionNode.dataset_id == first_id)).all()
    second_versions = db_session.exec(select(VersionNode).where(VersionNode.dataset_id == second_id)).all()
    first_persisted = load_pickle(Path(first.current_snapshot_path))
    second_persisted = load_pickle(Path(second.current_snapshot_path))
    mutation_version = next(version for version in first_versions if version.id == result.updated_datasets[0].version_id)

    assert result.ok is True
    assert result.updated_datasets[0].dataset_id == first_id
    assert "first_frame" in result.stdout
    assert "second_frame" in result.stdout
    assert "double" in first_persisted.columns
    assert "double" not in second_persisted.columns
    assert len(first_versions) == 2
    assert len(second_versions) == 1
    assert mutation_version.parent_version_id != second.current_version_id


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
                "chart_rows = data.groupby('group', as_index=False)['value'].sum().to_dict('records')",
                "save_chart('values chart', {'title': 'Values by group', 'chart_type': 'bar', 'data': chart_rows, 'x': 'group', 'y': 'value', 'description': 'Grouped totals'})",
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


def test_chart_artifact_validates_and_reduces_large_data(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "rows = [{'category': f'c{i}', 'value': i} for i in range(120)]",
                "save_chart('large chart', {'title': 'Large chart', 'chart_type': 'bar', 'data': rows, 'x': 'category', 'y': 'value'})",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    artifact = next(item for item in result.artifacts if item.kind == "chart")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert result.ok is True
    assert artifact.metadata["chart_type"] == "bar"
    assert artifact.metadata["rows"] == 50
    assert artifact.metadata["sampled"] is True
    assert payload["chart_type"] == "bar"
    assert len(payload["data"]) == 50
    assert payload["_sampling"]["method"] == "aggregate_top_categories"


def test_invalid_chart_artifact_returns_traceback(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "save_chart('bad chart', {'chart_type': 'radar', 'data': [{'x': 'a', 'y': 1}], 'x': 'x', 'y': 'y'})",
        active_dataset_id=dataset_id,
    )

    assert result.ok is False
    assert result.traceback is not None
    assert "Unsupported chart_type" in result.traceback


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
    value: object | None = None,
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
        dataset_key=Path(filename).stem.replace("-", "_"),
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
        parent_version_id=None,
        label="initial",
        mutation_summary="Initial upload",
        snapshot_path=str(snapshot_path),
        profile=profile,
    )
    branch.root_version_id = version_id
    branch.current_version_id = version_id
    analysis_session.active_branch_id = branch.id
    analysis_session.active_dataset_id = dataset_id

    db_session.add(analysis_session)
    db_session.add(branch)
    db_session.add(dataset)
    db_session.add(version)
    db_session.commit()

    return analysis_session.id, dataset_id


def _create_two_datasets(db_session: Session) -> tuple[str, str, str]:
    analysis_session = AnalysisSession(name="Multi runtime test")
    branch = Branch(session_id=analysis_session.id, name="main")
    db_session.add(analysis_session)
    db_session.add(branch)

    first_id, first_version = _add_dataset_row(
        db_session,
        analysis_session.id,
        branch.id,
        "first-frame.pkl",
        "first_frame",
        pd.DataFrame({"value": [1, 2], "group": ["a", "b"]}),
    )
    second_id, second_version = _add_dataset_row(
        db_session,
        analysis_session.id,
        branch.id,
        "second-frame.pkl",
        "second_frame",
        pd.DataFrame({"value": [10, 20], "kind": ["x", "y"]}),
    )
    branch.root_version_id = first_version
    branch.current_version_id = second_version
    analysis_session.active_branch_id = branch.id
    analysis_session.active_dataset_id = first_id
    db_session.add(branch)
    db_session.add(analysis_session)
    db_session.commit()

    return analysis_session.id, first_id, second_id


def _add_dataset_row(
    db_session: Session,
    session_id: str,
    branch_id: str,
    filename: str,
    dataset_key: str,
    value: pd.DataFrame,
) -> tuple[str, str]:
    dataset_id = new_id()
    version_id = new_id()
    snapshot_path = save_snapshot(session_id, dataset_id, version_id, value)
    profile = introspect_object(value)
    dataset = Dataset(
        id=dataset_id,
        session_id=session_id,
        dataset_key=dataset_key,
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
        branch_id=branch_id,
        parent_version_id=None,
        label="initial",
        mutation_summary="Initial upload",
        snapshot_path=str(snapshot_path),
        profile=profile,
    )
    db_session.add(dataset)
    db_session.add(version)
    return dataset_id, version_id
