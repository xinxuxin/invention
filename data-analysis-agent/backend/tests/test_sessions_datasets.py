from collections.abc import Generator

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings
from app.db.session import get_session
from app.main import app


class DemoObject:
    def __init__(self) -> None:
        self.name = "demo"
        self.values = [1, 2, {"nested": True}]

    def greet(self) -> str:
        return "hello"


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "test.db"
    storage_path = tmp_path / ".data"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_DIR", str(storage_path))
    get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    def get_test_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_create_session_creates_main_branch(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"name": "Exploration"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "Exploration"
    assert [branch["name"] for branch in payload["branches"]] == ["main"]

    fetched = client.get(f"/api/sessions/{payload['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == payload["id"]


def test_upload_dataframe_pickle_returns_dataframe_profile(client: TestClient) -> None:
    session_id = _create_session(client)
    frame = pd.DataFrame(
        {
            "city": ["New York", "Lisbon", "Tokyo"],
            "score": [9.5, 8.25, 9.0],
            "active": [True, False, True],
        }
    )

    dataset = _upload_pickle(client, session_id, "frame.pkl", frame)
    profile = dataset["profile"]

    assert profile["object_type"] == "DataFrame"
    assert profile["shape"] == [3, 3]
    assert profile["columns"] == ["city", "score", "active"]
    assert profile["dtypes"]["city"] in {"object", "str", "string"}
    assert profile["dtypes"]["score"] == "float64"
    assert profile["dtypes"]["active"] == "bool"
    assert profile["sample_rows"][0] == {"city": "New York", "score": 9.5, "active": True}
    assert dataset["current_version"]["label"] == "initial"


def test_upload_list_of_dicts_returns_generic_collection_profile(client: TestClient) -> None:
    session_id = _create_session(client)
    value = [
        {"name": "alpha", "metrics": {"count": 3, "ok": True}},
        {"name": "beta", "metrics": {"count": 5, "ok": False}},
    ]

    dataset = _upload_pickle(client, session_id, "records.pkl", value)
    profile = dataset["profile"]

    assert profile["object_type"] == "list"
    assert profile["length"] == 2
    assert profile["keys"] == ["name", "metrics"]
    assert profile["sample_items"][0]["keys"] == ["name", "metrics"]
    assert profile["nested_summary"]["kind"] == "sequence"
    assert profile["nested_summary"]["items"][0]["kind"] == "mapping"


def test_upload_numpy_array_returns_array_profile(client: TestClient) -> None:
    session_id = _create_session(client)
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    dataset = _upload_pickle(client, session_id, "array.pkl", value)
    profile = dataset["profile"]

    assert profile["object_type"] == "ndarray"
    assert profile["shape"] == [2, 3]
    assert profile["dtype"] == "int64"
    assert profile["sample"] == [1, 2, 3, 4, 5]


def test_upload_custom_object_does_not_crash(client: TestClient) -> None:
    session_id = _create_session(client)

    dataset = _upload_pickle(client, session_id, "object.pkl", DemoObject())
    profile = dataset["profile"]

    assert profile["object_type"] == "DemoObject"
    assert profile["module"] == "test_sessions_datasets"
    assert profile["public_attributes"]["name"] == "demo"
    assert profile["public_attributes"]["values"]["type"] == "list"
    assert "repr_preview" in profile


def test_upload_multiple_pickle_files(client: TestClient) -> None:
    session_id = _create_session(client)
    files = [
        ("files", ("one.pkl", cloudpickle.dumps({"a": 1}), "application/octet-stream")),
        ("files", ("two.pkl", cloudpickle.dumps([{"b": 2}]), "application/octet-stream")),
    ]

    response = client.post(f"/api/sessions/{session_id}/datasets", files=files)

    assert response.status_code == 201
    assert len(response.json()["datasets"]) == 2

    listed = client.get(f"/api/sessions/{session_id}/datasets")
    assert listed.status_code == 200
    assert len(listed.json()["datasets"]) == 2


def test_dataset_keys_are_safe_unique_and_active_dataset_can_change(client: TestClient) -> None:
    session_id = _create_session(client)
    files = [
        ("files", ("Revenue 2026.pkl", cloudpickle.dumps({"a": 1}), "application/octet-stream")),
        ("files", ("Revenue 2026.pkl", cloudpickle.dumps({"b": 2}), "application/octet-stream")),
    ]

    upload = client.post(f"/api/sessions/{session_id}/datasets", files=files)
    assert upload.status_code == 201

    datasets = upload.json()["datasets"]
    assert [dataset["dataset_key"] for dataset in datasets] == ["revenue_2026", "revenue_2026_2"]

    activate = client.post(f"/api/sessions/{session_id}/datasets/{datasets[1]['id']}/activate")
    assert activate.status_code == 200
    assert activate.json()["active_dataset_id"] == datasets[1]["id"]

    fetched = client.get(f"/api/sessions/{session_id}")
    assert fetched.json()["active_dataset_id"] == datasets[1]["id"]


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={})
    assert response.status_code == 201
    return response.json()["id"]


def _upload_pickle(client: TestClient, session_id: str, filename: str, value: object) -> dict:
    response = client.post(
        f"/api/sessions/{session_id}/datasets",
        files={"files": (filename, cloudpickle.dumps(value), "application/octet-stream")},
    )

    assert response.status_code == 201, response.text
    return response.json()["datasets"][0]
