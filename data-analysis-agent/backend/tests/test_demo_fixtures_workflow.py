import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings
from app.db.session import get_session
from app.main import app


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "generated"
SOFTBANK_FIXTURE = Path("/Users/macbook/Desktop/softbank_group_patent_portfolio_metadata.pkl")


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "demo-fixtures.db"
    storage_path = tmp_path / ".data"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_DIR", str(storage_path))
    monkeypatch.setenv("AGENT_MODEL_MODE", "fake")
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


@pytest.mark.parametrize(
    ("filename", "expectation"),
    [
        (
            "dataframe_transactions.pkl",
            {
                "object_type": "DataFrame",
                "shape": [6, 7],
                "columns": {"transaction_id", "amount", "category", "purchased_at"},
            },
        ),
        (
            "list_of_dicts_nested.pkl",
            {"object_type": "list", "length": 4, "keys": {"id", "profile", "metrics"}},
        ),
        ("numpy_array.pkl", {"object_type": "ndarray", "shape": [3, 4], "dtype": "float64"}),
        ("custom_objects.pkl", {"object_type": "list", "length": 3, "sample_type": "DemoPatentRecord"}),
        (
            "mixed_collection.pkl",
            {"object_type": "dict", "length": 5, "keys": {"frame", "records", "matrix", "nested"}},
        ),
    ],
)
def test_generated_fixture_upload_introspection(
    client: TestClient,
    filename: str,
    expectation: dict,
) -> None:
    session_id = _create_session(client)

    dataset = _upload_fixture(client, session_id, filename)
    profile = dataset["profile"]

    assert profile["object_type"] == expectation["object_type"]
    if "shape" in expectation:
        assert profile["shape"] == expectation["shape"]
    if "dtype" in expectation:
        assert profile["dtype"] == expectation["dtype"]
    if "length" in expectation:
        assert profile["length"] == expectation["length"]
    if "columns" in expectation:
        assert set(profile["columns"]) >= expectation["columns"]
    if "keys" in expectation:
        assert set(profile["keys"]) >= expectation["keys"]
    if "sample_type" in expectation:
        assert profile["sample_items"][0]["type"] == expectation["sample_type"]
    assert dataset["current_version"]["label"] == "initial"


def test_fake_agent_chat_what_is_this_file(client: TestClient) -> None:
    session_id = _create_session(client)
    dataset = _upload_fixture(client, session_id, "dataframe_transactions.pkl")

    events = _chat(client, session_id, dataset["id"], "What is this file?")

    assert events[0]["type"] == "message_started"
    assert any(event["type"] == "trace" for event in events)
    assert any(event["type"] == "code_result_summary" and event["ok"] for event in events)
    assert events[-2]["type"] == "final_answer"
    assert "inspected" in events[-2]["answer"].lower()


def test_fake_agent_mutation_persistence_rollback_fork_and_csv_export(client: TestClient) -> None:
    session_id = _create_session(client)
    dataset = _upload_fixture(client, session_id, "dataframe_transactions.pkl")

    mutation_events = _chat(
        client,
        session_id,
        dataset["id"],
        "Please add test column mutation",
    )
    result = next(event for event in mutation_events if event["type"] == "code_result_summary")
    assert len(result["updated_datasets"]) == 1
    assert _dataset(client, session_id, dataset["id"])["profile"]["shape"] == [6, 8]

    export = client.post(
        f"/api/sessions/{session_id}/export",
        json={"dataset_id": dataset["id"], "name": "mutated transactions"},
    )
    assert export.status_code == 200
    artifact = export.json()["artifact"]
    csv_body = client.get(f"/api/sessions/{session_id}/artifacts/{artifact['id']}/download").text
    assert "__agent_test_row_number" in csv_body

    versions = client.get(f"/api/sessions/{session_id}/history").json()["versions"]
    initial_version_id = next(version["id"] for version in versions if version["label"] == "initial")
    mutated_version_id = next(
        version["id"]
        for version in versions
        if version["mutation_summary"] == "Add agent test row number column"
    )

    rollback = client.post(f"/api/sessions/{session_id}/versions/{initial_version_id}/rollback")
    assert rollback.status_code == 200
    assert _dataset(client, session_id, dataset["id"])["profile"]["shape"] == [6, 7]

    fork = client.post(
        f"/api/sessions/{session_id}/versions/{mutated_version_id}/fork",
        json={"name": "mutated-demo"},
    )
    assert fork.status_code == 201
    assert fork.json()["branch"]["name"] == "mutated-demo"
    assert _dataset(client, session_id, dataset["id"])["profile"]["shape"] == [6, 8]


def test_fake_agent_chart_artifact_creation(client: TestClient) -> None:
    session_id = _create_session(client)
    dataset = _upload_fixture(client, session_id, "dataframe_transactions.pkl")

    events = _chat(client, session_id, dataset["id"], "Visualize this dataset")
    artifact = next(event["artifact"] for event in events if event["type"] == "artifact_created")
    content = client.get(f"/api/sessions/{session_id}/artifacts/{artifact['id']}/content").json()

    assert artifact["kind"] == "chart"
    assert artifact["metadata"]["chart_type"] == "bar"
    assert content["title"] == "Fake agent chart"
    assert content["data"]


def test_fake_agent_multi_dataset_comparison(client: TestClient) -> None:
    session_id = _create_session(client)
    users, orders = _upload_many(
        client,
        session_id,
        ["multi_dataset_users.pkl", "multi_dataset_orders.pkl"],
    )

    events = _chat(client, session_id, users["id"], "Compare these two datasets")
    result = next(event for event in events if event["type"] == "code_result_summary")

    assert result["ok"] is True
    assert "multi_dataset_users" in result["stdout"]
    assert "multi_dataset_orders" in result["stdout"]
    assert client.get(f"/api/sessions/{session_id}/datasets/{orders['id']}").status_code == 200


def test_fake_agent_executor_error_recovery_path(client: TestClient) -> None:
    session_id = _create_session(client)
    dataset = _upload_fixture(client, session_id, "dataframe_transactions.pkl")

    events = _chat(client, session_id, dataset["id"], "Trigger executor error recovery path")
    results = [event for event in events if event["type"] == "code_result_summary"]

    assert [event["ok"] for event in results] == [False, True]
    assert any("retrying" in event.get("message", "") for event in events if event["type"] == "trace")
    assert events[-2]["type"] == "final_answer"


def test_fake_agent_asks_clarification_before_ambiguous_destructive_write(client: TestClient) -> None:
    session_id = _create_session(client)
    dataset = _upload_fixture(client, session_id, "dataframe_transactions.pkl")

    events = _chat(
        client,
        session_id,
        dataset["id"],
        "Drop rows with missing values in the most important identifier field.",
    )

    assert not any(event["type"] == "code_result_summary" for event in events)
    assert events[-2]["type"] == "final_answer"
    assert "which identifier field" in events[-2]["answer"].lower()
    assert events[-2]["state_changed"] is False


@pytest.mark.skipif(not SOFTBANK_FIXTURE.exists(), reason="Local SoftBank demo pickle is not present")
def test_local_softbank_pickle_uploads_and_profiles(client: TestClient) -> None:
    session_id = _create_session(client)

    dataset = _upload_path(client, session_id, SOFTBANK_FIXTURE)
    profile = dataset["profile"]

    assert profile["object_type"]
    assert "nested_summary" in profile
    assert dataset["original_filename"] == SOFTBANK_FIXTURE.name


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"name": "Generated fixture demo"})
    assert response.status_code == 201
    return response.json()["id"]


def _upload_fixture(client: TestClient, session_id: str, filename: str) -> dict:
    return _upload_path(client, session_id, FIXTURE_DIR / filename)


def _upload_path(client: TestClient, session_id: str, path: Path) -> dict:
    with path.open("rb") as source:
        response = client.post(
            f"/api/sessions/{session_id}/datasets",
            files={"files": (path.name, source, "application/octet-stream")},
        )
    assert response.status_code == 201, response.text
    return response.json()["datasets"][0]


def _upload_many(client: TestClient, session_id: str, filenames: list[str]) -> list[dict]:
    handles = [(FIXTURE_DIR / filename).open("rb") for filename in filenames]
    try:
        files = [
            ("files", (filename, handle, "application/octet-stream"))
            for filename, handle in zip(filenames, handles, strict=True)
        ]
        response = client.post(f"/api/sessions/{session_id}/datasets", files=files)
    finally:
        for handle in handles:
            handle.close()

    assert response.status_code == 201, response.text
    return response.json()["datasets"]


def _dataset(client: TestClient, session_id: str, dataset_id: str) -> dict:
    response = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}")
    assert response.status_code == 200
    return response.json()


def _chat(client: TestClient, session_id: str, dataset_id: str, message: str) -> list[dict]:
    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": message, "active_dataset_id": dataset_id, "confirmed": True},
    )
    assert response.status_code == 200, response.text
    return _parse_sse(response.text)


def _parse_sse(payload: str) -> list[dict]:
    events: list[dict] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events
