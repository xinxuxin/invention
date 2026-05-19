import json
from collections.abc import Generator
from pathlib import Path

import cloudpickle
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from app.agent.agent import AgentModelResponse, AgentToolCall
from app.api.chat import get_model_client
from app.core.config import get_settings
from app.db.session import get_session
from app.main import app
from app.models.entities import new_id


class ScriptedModelClient:
    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def create_response(self, **_: object) -> AgentModelResponse:
        if self.calls >= len(self.responses):
            raise AssertionError("No scripted model response left")
        response = self.responses[self.calls]
        self.calls += 1
        return response


@pytest.fixture()
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "export.db"
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


def test_export_current_dataset_reflects_mutated_branch_state(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    _run_confirmed_mutation(
        client,
        session_id,
        dataset_id,
        "data = data[data['revenue'] > 1000].copy()\npreview(data)",
        "Keep high revenue rows",
    )

    export_response = client.post(
        f"/api/sessions/{session_id}/export",
        json={"dataset_id": dataset_id, "name": "high revenue export"},
    )

    assert export_response.status_code == 200
    payload = export_response.json()
    assert payload["ok"] is True
    assert payload["artifact"]["kind"] == "csv"
    assert payload["artifact"]["metadata"]["rows"] == 2

    download = client.get(f"/api/sessions/{session_id}/artifacts/{payload['artifact']['id']}/download")
    assert download.status_code == 200
    assert "1200" in download.text
    assert "2500" in download.text
    assert "400" not in download.text


def test_export_list_of_dicts_uses_generic_normalization(client: TestClient) -> None:
    session_id = _create_session(client)
    records = [
        {"name": "alpha", "metrics": {"revenue": 1200, "active": True}},
        {"name": "beta", "metrics": {"revenue": 400, "active": False}},
    ]
    dataset_id = _upload_pickle(client, session_id, "records.pkl", records)

    response = client.post(f"/api/sessions/{session_id}/export", json={"dataset_id": dataset_id})

    assert response.status_code == 200
    artifact = response.json()["artifact"]
    csv_content = client.get(f"/api/sessions/{session_id}/artifacts/{artifact['id']}/content").text
    assert "metrics.revenue" in csv_content
    assert "alpha" in csv_content
    assert artifact["metadata"]["rows"] == 2


def test_agent_save_csv_exports_intermediate_without_mutating_state(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "subset = data[data['revenue'] > 1000].head(1)\nsave_csv('top high revenue', subset)\npreview(subset)",
                    "mutates_state": False,
                },
            ),
            _tool_response(
                "final_answer",
                {"answer": "Created a CSV for the filtered intermediate result.", "state_changed": False},
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Export rows where revenue > 1000 but do not change the dataset", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    artifact = next(event["artifact"] for event in events if event["type"] == "artifact_created")

    assert response.status_code == 200
    assert artifact["kind"] == "csv"
    assert artifact["metadata"]["rows"] == 1
    assert client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}").json()["profile"]["shape"] == [3, 3]

    download = client.get(f"/api/sessions/{session_id}/artifacts/{artifact['id']}/download")
    assert download.status_code == 200
    assert "growth" in download.text
    assert "trial" not in download.text
    assert Path(artifact["path"]).exists()


def _run_confirmed_mutation(
    client: TestClient,
    session_id: str,
    dataset_id: str,
    code: str,
    mutation_summary: str,
) -> None:
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": code,
                    "mutates_state": True,
                    "mutation_summary": mutation_summary,
                },
            ),
            _tool_response("final_answer", {"answer": "Saved mutation.", "state_changed": True}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake
    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": mutation_summary, "active_dataset_id": dataset_id, "confirmed": True},
    )
    assert response.status_code == 200


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"name": "Export test"})
    assert response.status_code == 201
    return response.json()["id"]


def _create_dataset(client: TestClient) -> tuple[str, str]:
    session_id = _create_session(client)
    frame = pd.DataFrame(
        {
            "segment": ["trial", "growth", "enterprise"],
            "revenue": [400, 1200, 2500],
            "active": [True, True, False],
        }
    )
    dataset_id = _upload_pickle(client, session_id, "revenue.pkl", frame)
    return session_id, dataset_id


def _upload_pickle(client: TestClient, session_id: str, filename: str, value: object) -> str:
    response = client.post(
        f"/api/sessions/{session_id}/datasets",
        files={"files": (filename, cloudpickle.dumps(value), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return response.json()["datasets"][0]["id"]


def _tool_response(name: str, arguments: dict) -> AgentModelResponse:
    call_id = f"call-{name}-{new_id()}"
    return AgentModelResponse(
        tool_calls=[AgentToolCall(id=call_id, name=name, arguments=arguments)],
        raw_output_items=[
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ],
    )


def _parse_sse(payload: str) -> list[dict]:
    events: list[dict] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events
