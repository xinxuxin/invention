import json
from collections.abc import Generator

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
    db_path = tmp_path / "history.db"
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


def test_mutation_rollback_fork_and_checkout_use_branch_state(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    main_branch_id = client.get(f"/api/sessions/{session_id}").json()["active_branch_id"]

    mutation_events = _run_confirmed_mutation(
        client,
        session_id,
        dataset_id,
        "data = data[data['value'] > 1].copy()\npreview(data)",
        "Keep values above one",
    )
    assert any(event["type"] == "code_result_summary" and event["updated_datasets"] for event in mutation_events)
    assert _active_shape(client, session_id) == [2, 2]

    history = client.get(f"/api/sessions/{session_id}/history").json()["versions"]
    initial_version_id = next(version["id"] for version in history if version["label"] == "initial")
    filtered_version_id = next(
        version["id"] for version in history if version["mutation_summary"] == "Keep values above one"
    )

    rollback_response = client.post(f"/api/sessions/{session_id}/versions/{initial_version_id}/rollback")
    assert rollback_response.status_code == 200
    assert rollback_response.json()["version"]["parent_version_id"] == filtered_version_id
    assert _active_shape(client, session_id) == [3, 2]

    fork_response = client.post(
        f"/api/sessions/{session_id}/versions/{filtered_version_id}/fork",
        json={"name": "filtered"},
    )
    assert fork_response.status_code == 201
    filtered_branch_id = fork_response.json()["branch"]["id"]
    assert _active_shape(client, session_id) == [2, 2]

    _run_confirmed_mutation(
        client,
        session_id,
        dataset_id,
        "data = data[data['value'] == 2].copy()\npreview(data)",
        "Keep value two on filtered branch",
    )
    assert _active_shape(client, session_id) == [1, 2]

    checkout_main = client.post(f"/api/sessions/{session_id}/branches/{main_branch_id}/checkout")
    assert checkout_main.status_code == 200
    assert _active_shape(client, session_id) == [3, 2]

    checkout_filtered = client.post(f"/api/sessions/{session_id}/branches/{filtered_branch_id}/checkout")
    assert checkout_filtered.status_code == 200
    assert _active_shape(client, session_id) == [1, 2]


def _run_confirmed_mutation(
    client: TestClient,
    session_id: str,
    dataset_id: str,
    code: str,
    mutation_summary: str,
) -> list[dict]:
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
            _tool_response(
                "final_answer",
                {"answer": f"Saved mutation: {mutation_summary}.", "state_changed": True},
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake
    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={
            "message": mutation_summary,
            "active_dataset_id": dataset_id,
            "confirmed": True,
        },
    )
    assert response.status_code == 200
    return _parse_sse(response.text)


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


def _create_dataset(client: TestClient) -> tuple[str, str]:
    session_response = client.post("/api/sessions", json={"name": "History test"})
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]
    frame = pd.DataFrame({"value": [1, 2, 3], "group": ["a", "b", "a"]})
    upload_response = client.post(
        f"/api/sessions/{session_id}/datasets",
        files={
            "files": (
                "history-frame.pkl",
                cloudpickle.dumps(frame),
                "application/octet-stream",
            )
        },
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["datasets"][0]["id"]
    return session_id, dataset_id


def _active_shape(client: TestClient, session_id: str) -> list[int]:
    response = client.get(f"/api/sessions/{session_id}/datasets")
    assert response.status_code == 200
    return response.json()["datasets"][0]["profile"]["shape"]


def _parse_sse(payload: str) -> list[dict]:
    events: list[dict] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events
