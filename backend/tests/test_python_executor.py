import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.config import get_settings
from app.models.entities import AnalysisSession, Artifact, Branch, Dataset, VersionNode, new_id
from app.runtime.python_executor import (
    PythonExecutor,
    fast_get_field,
    find_record_collections,
    flatten_records_at_path,
    inspect_object,
    object_to_record,
    objects_to_records,
    safe_attrs,
    summarize_structure,
    to_dataframe,
)
from scripts.create_agent_test_datasets import (
    build_custom_sensor_fleet,
    build_mixed_dataframe_numpy_bundle,
    build_mixed_top_level_collection,
    build_nested_customer_events,
)
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


class DummyMissingPatent:
    def __init__(self, **fields: object) -> None:
        self.attrs = {"__dict__": fields}

    def __repr__(self) -> str:
        return (
            "<MissingPickleClass filing.patent_portfolio.api_classes."
            f"GenericPatentMetadata attrs={self.attrs!r}>"
        )


class DummyPydanticWrapper:
    def __init__(self) -> None:
        self.__dict__ = {
            "__dict__": {
                "country": "CA",
                "title": "CLEANING ROBOT",
                "owners": ["Softbank"],
                "filing_date": datetime(2019, 3, 6),
            },
            "__pydantic_extra__": None,
            "__pydantic_fields_set__": {"country", "title"},
            "__pydantic_private__": None,
        }


@dataclass
class DummyDataclassRecord:
    country: str
    title: str


class DummyCustomRecord:
    def __init__(self, country: str) -> None:
        self.country = country


def test_fast_get_field_reads_common_record_shapes() -> None:
    assert fast_get_field({"country": "JP"}, "country") == "JP"
    assert fast_get_field(DummyDataclassRecord(country="CN", title="Robot"), "country") == "CN"
    assert fast_get_field(DummyCustomRecord("US"), "country") == "US"
    assert fast_get_field(DummyPydanticWrapper(), "country") == "CA"
    assert fast_get_field(DummyMissingPatent(country="JP", title="Robot"), "country") == "JP"
    assert fast_get_field(DummyCustomRecord("US"), "missing") is None


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
    assert result.result_preview["type"] == "dataframe"
    assert result.result_preview["shape"] == [2, 2]
    assert result.result_preview["rows"] == [{"value": 1, "group": "a"}, {"value": 2, "group": "b"}]


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


def test_final_expression_capture_returns_json_result(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "x = 1\n{'answer': x + 1}",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview == {"answer": 2}


def test_expression_capture_can_return_columns_and_rows(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "columns = ['a', 'b']",
                "rows = [{'a': 1, 'b': 2}]",
                "{'columns': columns, 'rows': rows}",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview == {"columns": ["a", "b"], "rows": [{"a": 1, "b": 2}]}


def test_dataframe_final_expression_capture_uses_dataframe_preview(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "import pandas as pd\npd.DataFrame([{'a': 1}, {'a': 2}])",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert isinstance(result.result_preview, dict)
    assert result.result_preview["type"] == "dataframe"
    assert result.result_preview["shape"] == [2, 1]
    assert result.result_preview["columns"] == ["a"]
    assert result.result_preview["rows"] == [{"a": 1}, {"a": 2}]


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


def test_missing_pickle_class_like_helpers_extract_inner_attrs(db_session: Session) -> None:
    class DummyMissing:
        def __init__(self) -> None:
            self.attrs = {"__dict__": {"country": "CA", "title": "CLEANING ROBOT"}}

    session_id, dataset_id = _create_dataset(db_session, filename="patent.pkl", value=[DummyMissing()])
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "record = safe_attrs(data[0])",
                "frame = to_dataframe(data)",
                "RESULT = {'record': record, 'columns': list(frame.columns), 'rows': frame.to_dict('records')}",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview["record"] == {"country": "CA", "title": "CLEANING ROBOT"}
    assert result.result_preview["columns"] == ["country", "title"]
    assert result.result_preview["rows"] == [{"country": "CA", "title": "CLEANING ROBOT"}]


def test_generic_structure_helpers_detect_nested_customer_events() -> None:
    dataset = build_nested_customer_events()

    inspected = inspect_object(dataset)
    summary = summarize_structure(dataset)
    collections = find_record_collections(dataset)
    customer_rows = flatten_records_at_path(dataset, "customers")
    event_rows = flatten_records_at_path(dataset, "customers.events")

    assert set(inspected["keys"]) >= {"metadata", "customers", "lookup_tables"}
    assert any(item["path"] == "customers" for item in collections)
    assert any(item["path"] == "customers.events" for item in collections)
    assert summary["top_level_keys"] == ["metadata", "customers", "lookup_tables"]
    assert {"customer_id", "country", "segment", "joined_at", "churn_risk"} <= set(customer_rows[0])
    assert {"customer_id", "event_id", "timestamp", "channel", "event_type", "order_total"} <= set(event_rows[0])


def test_generic_structure_helpers_detect_bundle_tables_and_arrays() -> None:
    dataset = build_mixed_dataframe_numpy_bundle()

    summary = summarize_structure(dataset)
    table_paths = {item["path"] for item in summary["tables_detected"]}
    array_paths = {item["path"] for item in summary["arrays_detected"]}

    assert {"users", "orders", "daily_metrics"} <= table_paths
    assert {"user_embedding_matrix", "cohort_tensor"} <= array_paths


def test_generic_structure_helpers_detect_custom_sensor_fleet_readings() -> None:
    dataset = build_custom_sensor_fleet()

    inspected = inspect_object(dataset)
    collections = find_record_collections(dataset)
    reading_rows = flatten_records_at_path(dataset, "sensors.readings")

    assert inspected["type"] == "SensorFleet"
    assert any(item["path"] == "sensors.readings" for item in collections)
    assert {"sensor_id", "timestamp", "temperature_c", "vibration_g", "battery_pct"} <= set(reading_rows[0])


def test_generic_structure_helpers_detect_mixed_top_level_collection() -> None:
    dataset = build_mixed_top_level_collection()

    inspected = inspect_object(dataset)
    item_types = {item["type"] for item in inspected["item_types"]}

    assert {"dict", "DataFrame", "ndarray", "list", "tuple"} <= item_types


def test_softbank_patent_helper_conversion_matches_missing_pickle_shape() -> None:
    dataset = _softbank_patent_dataset()
    first = dataset[0]
    frame = to_dataframe(dataset)
    expected_columns = [
        "country",
        "doc_number",
        "kind",
        "title",
        "owners",
        "inventors",
        "assignees",
        "filing_date",
        "publication_date",
        "status",
        "family_size",
        "forward_citation_count",
    ]

    assert safe_attrs(first)["country"] == "CA"
    assert object_to_record(first)["title"] == "CLEANING ROBOT"
    assert frame.columns.tolist() == expected_columns


def test_pydantic_wrapper_object_to_record_flattens_domain_fields() -> None:
    record = object_to_record(DummyPydanticWrapper())

    assert record == {
        "country": "CA",
        "title": "CLEANING ROBOT",
        "owners": ["Softbank"],
        "filing_date": "2019-03-06T00:00:00",
    }
    assert "__dict__" not in record
    assert "__pydantic_extra__" not in record
    assert "__pydantic_fields_set__" not in record
    assert "__pydantic_private__" not in record


def test_pydantic_wrapper_to_dataframe_uses_domain_columns() -> None:
    frame = to_dataframe([DummyPydanticWrapper()])
    records = objects_to_records([DummyPydanticWrapper()])

    assert frame.columns.tolist() == ["country", "title", "owners", "filing_date"]
    assert "__dict__" not in frame.columns
    assert records[0]["owners"] == ["Softbank"]
    assert not isinstance(records[0]["owners"], str)


def test_to_dataframe_and_objects_to_records_are_full_by_default() -> None:
    records = [{"country": ["A", "B", "C"][index % 3], "value": index} for index in range(100)]

    frame = to_dataframe(records)
    limited = to_dataframe(records, limit=20)
    converted = objects_to_records(records)

    assert len(frame) == 100
    assert len(limited) == 20
    assert len(converted) == 100
    assert frame["country"].value_counts().sum() == 100


def test_preview_dataframe_limits_without_changing_full_dataframe(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(
        db_session,
        filename="hundred.pkl",
        value=pd.DataFrame({"country": ["A"] * 100, "value": list(range(100))}),
    )
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "df = to_dataframe(data)\npreview_df = preview_dataframe(data, limit=20)\nRESULT = {'full_rows': len(df), 'preview_rows': len(preview_df), 'preview': preview(df), 'preview_df': preview(preview_df)}",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview["full_rows"] == 100
    assert result.result_preview["preview_rows"] == 20
    assert result.result_preview["preview"]["shape"] == [100, 2]
    assert result.result_preview["preview"]["source_total_row_count"] == 100
    assert result.result_preview["preview"]["preview_row_count"] == 20
    assert result.result_preview["preview"]["is_preview"] is False
    assert len(result.result_preview["preview"]["rows"]) == 20
    assert result.result_preview["preview_df"]["source_total_row_count"] == 100
    assert result.result_preview["preview_df"]["preview_row_count"] == 20
    assert result.result_preview["preview_df"]["analyzed_row_count"] == 20
    assert result.result_preview["preview_df"]["is_preview"] is True


def test_execute_python_calls_are_isolated(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    first = executor.execute(session_id, "x = 123", active_dataset_id=dataset_id)
    second = executor.execute(session_id, "x", active_dataset_id=dataset_id)

    assert first.ok is True
    assert first.result_preview is None
    assert second.ok is False
    assert second.traceback is not None
    assert "NameError" in second.traceback


def test_dataset_profiles_namespace_is_available(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session, filename="profile-frame.pkl")
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "{'keys': list(dataset_profiles.keys()), 'active_type': active_dataset_profile['object_type']}",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview["keys"] == ["profile_frame"]
    assert result.result_preview["active_type"] == "DataFrame"


def test_artifact_history_namespace_is_available(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session, filename="artifact-frame.pkl")
    executor = PythonExecutor(db_session)
    first = executor.execute(
        session_id,
        "save_table('Preview', data.head(1))",
        active_dataset_id=dataset_id,
    )

    second = executor.execute(
        session_id,
        "{'artifact_names': [artifact['name'] for artifact in artifact_history], 'alias_names': [artifact['name'] for artifact in artifacts]}",
        active_dataset_id=dataset_id,
    )

    assert first.ok is True
    assert second.ok is True
    assert second.result_preview["artifact_names"] == ["Preview"]
    assert second.result_preview["alias_names"] == ["Preview"]


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
    assert result.result_preview["scalar"] == 1
    assert result.result_preview["stamp"] == "2026-05-18T00:00:00"


def test_softbank_like_tabular_preview_succeeds_without_null_preview(db_session: Session) -> None:
    class DummyMissing:
        def __init__(self, country: str, title: str, owners: list[str], filing_date: pd.Timestamp) -> None:
            self.attrs = {
                "__dict__": {
                    "country": country,
                    "title": title,
                    "owners": owners,
                    "filing_date": filing_date,
                }
            }

    value = [
        DummyMissing("CA", "CLEANING ROBOT", ["Softbank"], pd.Timestamp("2019-03-06")),
        DummyMissing("CN", "ROBOT CONTROL", ["Softbank", "Other"], pd.Timestamp("2020-05-22")),
    ]
    session_id, dataset_id = _create_dataset(db_session, filename="softbank-like.pkl", value=value)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "frame = to_dataframe(data)",
                "RESULT = {'columns': list(frame.columns), 'rows': frame.head(5).to_dict('records')}",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview is not None
    assert result.result_preview["columns"] == ["country", "title", "owners", "filing_date"]
    assert result.result_preview["rows"][0]["country"] == "CA"
    assert result.result_preview["rows"][0]["title"] == "CLEANING ROBOT"


def test_softbank_like_schema_summary_uses_structured_helpers(db_session: Session) -> None:
    class DummyMissing:
        def __init__(self) -> None:
            self.attrs = {
                "__dict__": {
                    "country": "CA",
                    "title": "CLEANING ROBOT",
                    "owners": ["Softbank"],
                    "filing_date": pd.Timestamp("2019-03-06"),
                    "family_size": 15,
                }
            }

    session_id, dataset_id = _create_dataset(db_session, filename="softbank-schema.pkl", value=[DummyMissing()])
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "records = objects_to_records(data, limit=5)",
                "scalar_fields = []",
                "date_fields = []",
                "list_like_fields = []",
                "for key, value in records[0].items():",
                "    if isinstance(value, list):",
                "        list_like_fields.append(key)",
                "    elif 'date' in key or hasattr(value, 'isoformat'):",
                "        date_fields.append(key)",
                "    else:",
                "        scalar_fields.append(key)",
                "{'scalar_fields': scalar_fields, 'date_fields': date_fields, 'list_like_fields': list_like_fields}",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview["scalar_fields"] == ["country", "title", "family_size"]
    assert result.result_preview["date_fields"] == ["filing_date"]
    assert result.result_preview["list_like_fields"] == ["owners"]


def test_softbank_patent_final_expression_tabular_preview(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(
        db_session,
        filename="softbank-patents.pkl",
        value=_softbank_patent_dataset(),
    )
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "df = to_dataframe(data)",
                "RESULT = {",
                "    'columns': list(df.columns),",
                "    'rows': df.head(5).to_dict(orient='records'),",
                "}",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview is not None
    assert result.result_preview["columns"][:4] == ["country", "doc_number", "kind", "title"]
    assert "owners" in result.result_preview["columns"]
    assert "status" in result.result_preview["columns"]
    assert "__dict__" not in result.result_preview["columns"]
    assert result.result_preview["rows"][0]["country"] == "CA"
    assert result.result_preview["rows"][0]["title"] == "CLEANING ROBOT"
    assert result.result_preview["rows"][0]["owners"] == ["Softbank", "SOFTBANK ROBOTICS GROUP CORP"]
    assert result.result_preview["rows"][0]["filing_date"] == "2019-03-06T00:00:00"


def test_softbank_patent_schema_classification_code(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(
        db_session,
        filename="softbank-patents.pkl",
        value=_softbank_patent_dataset(),
    )
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "df = to_dataframe(data)",
                "sample = df.iloc[0].to_dict()",
                "schema = {}",
                "for k, v in sample.items():",
                "    if isinstance(v, list):",
                "        schema[k] = 'list-like'",
                "    elif hasattr(v, 'isoformat'):",
                "        schema[k] = 'date'",
                "    elif isinstance(v, (str, int, float, bool)):",
                "        schema[k] = 'scalar'",
                "    else:",
                "        schema[k] = type(v).__name__",
                "RESULT = schema",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    assert result.result_preview["owners"] == "list-like"
    assert result.result_preview["assignees"] == "list-like"
    assert result.result_preview["filing_date"] == "date"
    assert result.result_preview["country"] == "scalar"
    assert result.result_preview["family_size"] == "scalar"


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

    table_artifact = next(artifact for artifact in result.artifacts if artifact.kind == "table")
    table_payload = json.loads(Path(table_artifact.path).read_text(encoding="utf-8"))
    assert table_payload["type"] == "table"
    assert table_payload["title"] == "values table"
    assert table_payload["columns"] == [
        {"key": "value", "label": "value", "type": "number"},
        {"key": "group", "label": "group", "type": "text"},
    ]
    assert table_payload["rows"] == [{"value": 1, "group": "a"}, {"value": 2, "group": "b"}]
    assert table_payload["preview_row_count"] == 2


def test_save_table_accepts_objects_and_preserves_structured_cells(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(
        db_session,
        filename="patents.pkl",
        value=[
            DummyMissingPatent(
                country="CA",
                title="CLEANING ROBOT",
                owners=["Softbank"],
                filing_date=datetime(2019, 3, 6),
            )
        ],
    )
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "save_table('patent preview', data, description='First patent rows')",
        active_dataset_id=dataset_id,
    )

    artifact = next(item for item in result.artifacts if item.kind == "table")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert result.ok is True
    assert artifact.title == "patent preview"
    assert artifact.description == "First patent rows"
    assert [column["key"] for column in payload["columns"]] == [
        "country",
        "title",
        "owners",
        "filing_date",
    ]
    assert payload["rows"][0]["owners"] == ["Softbank"]
    assert payload["rows"][0]["filing_date"] == "2019-03-06T00:00:00"


def test_save_table_keeps_full_columns_while_exposing_display_metadata(db_session: Session) -> None:
    columns = [
        "country",
        "doc_number",
        "kind",
        "title",
        "owners",
        "inventors",
        "assignees",
        "filing_date",
        "application_date",
        "publication_date",
        "priority_date",
        "expiration_date",
        "is_application",
        "is_grant",
        "is_other",
        "num_claims",
        "num_independent_claims",
        "backward_citation_count",
        "forward_citation_count",
        "npl_citation_count",
        "family_size",
        "family_countries",
        "cpc_sections",
        "cpc_classes",
        "cpc_subclasses",
        "cpc_main_groups",
        "cpc_subgroups",
        "all_cpc_codes",
        "ipc_codes",
        "status",
        "has_continuation",
        "has_division",
        "pta_days",
        "num_legal_events",
    ]
    frame = pd.DataFrame([{column: column for column in columns}])
    session_id, dataset_id = _create_dataset(db_session, value=frame)
    executor = PythonExecutor(db_session)

    result = executor.execute(session_id, "save_table('wide preview', data.head(1))", active_dataset_id=dataset_id)

    artifact = next(item for item in result.artifacts if item.kind == "table")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert result.ok is True
    assert len(payload["columns"]) == 34
    assert len(payload["display_columns"]) == 30
    assert payload["total_column_count"] == 34
    assert payload["hidden_columns"] == ["has_continuation", "has_division", "pta_days", "num_legal_events"]
    assert "has_continuation" in payload["rows"][0]
    assert "has_continuation" not in payload["preview_rows"][0]
    assert artifact.metadata["total_column_count"] == 34


def test_table_rows_preserve_identifier_strings_and_structured_lists(db_session: Session) -> None:
    frame = pd.DataFrame(
        [
            {
                "country": "CA",
                "doc_number": 186426,
                "owners": ["Softbank", "Huawei"],
                "filing_date": datetime(2019, 3, 6),
            }
        ]
    )
    session_id, dataset_id = _create_dataset(db_session, value=frame)
    executor = PythonExecutor(db_session)

    result = executor.execute(session_id, "save_table('identifier preview', data)", active_dataset_id=dataset_id)

    artifact = next(item for item in result.artifacts if item.kind == "table")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert result.ok is True
    assert payload["columns"][1]["type"] == "identifier"
    assert payload["rows"][0]["doc_number"] == "186426"
    assert payload["rows"][0]["owners"] == ["Softbank", "Huawei"]
    assert payload["rows"][0]["filing_date"] == "2019-03-06T00:00:00"


def test_result_preview_columns_rows_creates_inline_table_artifact(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "rows = [{'country': 'CA', 'count': 2}, {'country': 'CN', 'count': 1}]",
                "RESULT = {'columns': ['country', 'count'], 'rows': rows}",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    artifact = next(item for item in result.artifacts if item.kind == "table")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert result.ok is True
    assert result.result_preview["rows"][0]["country"] == "CA"
    assert artifact.metadata["csv_download_available"] is True
    assert [column["key"] for column in payload["columns"]] == ["country", "count"]
    assert payload["rows"] == [{"country": "CA", "count": 2}, {"country": "CN", "count": 1}]


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
    assert artifact.metadata["row_count"] == 50
    assert artifact.metadata["sampled"] is True
    assert artifact.chart_spec is not None
    assert artifact.chart_spec["data"]
    assert payload["chart_spec"]["chart_type"] == "bar"
    assert len(payload["chart_spec"]["data"]) == 50
    assert payload["chart_spec"]["_sampling"]["method"] == "aggregate_top_categories"


def test_save_chart_accepts_single_spec_argument(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "rows = [{'year': 2020, 'count': 2}, {'year': 2021, 'count': 4}]",
                "save_chart({'title': 'Filings by year', 'chart_type': 'line', 'data': rows, 'x': 'year', 'y': 'count'})",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    artifact = next(item for item in result.artifacts if item.kind == "chart")
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))

    assert result.ok is True
    assert artifact.title == "Filings by year"
    assert artifact.chart_spec is not None
    assert artifact.chart_spec["data"][0] == {"year": 2020, "count": 2}
    assert payload["chart_spec"]["x"] == "year"


def test_save_chart_accepts_keyword_data_merge(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "rows = [{'country': 'A', 'count': 3}]",
                "save_chart(name='Country chart', chart_spec={'title': 'Country chart', 'chart_type': 'bar', 'x': 'country', 'y': 'count'}, data=rows)",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    artifact = next(item for item in result.artifacts if item.kind == "chart")

    assert result.ok is True
    assert artifact.chart_spec is not None
    assert artifact.chart_spec["data"] == [{"country": "A", "count": 3}]


def test_save_chart_accepts_keyword_chart_spec_and_chart_type_synonyms(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "\n".join(
            [
                "rows = [{'country': 'CA', 'record_count': 4}]",
                "save_chart(chart_spec={'title': 'Country pie', 'chart_type': 'pie chart', 'data': rows, 'x': 'country', 'y': 'record_count'})",
            ]
        ),
        active_dataset_id=dataset_id,
    )

    artifact = next(item for item in result.artifacts if item.kind == "chart")

    assert result.ok is True
    assert artifact.title == "Country pie"
    assert artifact.chart_spec is not None
    assert artifact.chart_spec["chart_type"] == "pie"


def test_save_chart_invalid_data_error_is_specific(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "save_chart('bad', {'title': 'Bad', 'chart_type': 'bar', 'data': 'not rows', 'x': 'x', 'y': 'y'})",
        active_dataset_id=dataset_id,
    )

    assert result.ok is False
    assert result.traceback is not None
    assert "chart_spec.data must be a non-empty list of dict rows; got str" in result.traceback


def test_save_csv_accepts_data_only_argument(db_session: Session) -> None:
    session_id, dataset_id = _create_dataset(db_session)
    executor = PythonExecutor(db_session)

    result = executor.execute(
        session_id,
        "save_csv(data.head(1))",
        active_dataset_id=dataset_id,
    )

    assert result.ok is True
    artifact = next(item for item in result.artifacts if item.kind == "csv")
    assert artifact.title == "CSV export"
    assert artifact.metadata["row_count"] == 1


def test_save_csv_preserves_identifier_strings_and_json_list_cells(db_session: Session) -> None:
    frame = pd.DataFrame([{"doc_number": 186426, "owners": ["Softbank", "Huawei"], "value": 3}])
    session_id, dataset_id = _create_dataset(db_session, value=frame)
    executor = PythonExecutor(db_session)

    result = executor.execute(session_id, "save_csv('patent csv', data)", active_dataset_id=dataset_id)

    artifact = next(item for item in result.artifacts if item.kind == "csv")
    rows = list(csv.DictReader(Path(artifact.path).read_text(encoding="utf-8").splitlines()))

    assert result.ok is True
    assert rows[0]["doc_number"] == "186426"
    assert json.loads(rows[0]["owners"]) == ["Softbank", "Huawei"]
    assert rows[0]["value"] == "3"


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


def _softbank_patent_dataset() -> list[DummyMissingPatent]:
    return [
        DummyMissingPatent(
            country="CA",
            doc_number="186426",
            kind="S",
            title="CLEANING ROBOT",
            owners=["Softbank", "SOFTBANK ROBOTICS GROUP CORP"],
            inventors=[],
            assignees=["SOFTBANK ROBOTICS GROUP"],
            filing_date=datetime(2019, 3, 6),
            publication_date=datetime(2020, 5, 22),
            status="Filed",
            family_size=3,
            forward_citation_count=0,
        ),
        DummyMissingPatent(
            country="CN",
            doc_number="109842558",
            kind="A",
            title="Message forwarding method",
            owners=["Softbank", "Huawei"],
            inventors=["SHEN ZHIMIN"],
            assignees=["HUAWEI TECH"],
            filing_date=datetime(2017, 11, 28),
            publication_date=datetime(2019, 6, 7),
            status="Granted",
            family_size=5,
            forward_citation_count=2,
        ),
    ]


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
