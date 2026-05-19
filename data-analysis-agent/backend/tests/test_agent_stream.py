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
    db_path = tmp_path / "agent.db"
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


def test_streams_trace_python_result_and_final_answer(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response("execute_python", {"code": "print(data.shape)\npreview(data.head(2))"}),
            _tool_response(
                "final_answer",
                {
                    "answer": "The file contains a DataFrame with value and group fields.",
                    "state_changed": False,
                },
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "What's in this file?", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert events[0]["type"] == "message_started"
    assert any(event["type"] == "trace" for event in events)
    assert any(event["type"] == "code_started" for event in events)
    assert any(event["type"] == "code_result_summary" and event["ok"] for event in events)
    assert events[-2]["type"] == "final_answer"
    assert events[-2]["answer"].startswith("The file contains")
    assert events[-1]["type"] == "message_done"


def test_agent_retries_after_python_error(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response("execute_python", {"code": "preview(data['missing_column'])"}),
            _tool_response("execute_python", {"code": "preview(data)"}),
            _tool_response("final_answer", {"answer": "I retried with a generic preview."}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Summarize this data", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    result_events = [event for event in events if event["type"] == "code_result_summary"]
    assert [event["ok"] for event in result_events] == [False, True]
    assert any("retrying" in event.get("message", "") for event in events if event["type"] == "trace")
    assert events[-2]["type"] == "final_answer"


def test_destructive_mutation_requires_confirmation(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data.drop(columns=['value'], inplace=True)",
                    "mutates_state": True,
                    "mutation_summary": "Drop value column",
                },
            )
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Drop the value column", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert any(event["type"] == "confirmation_required" for event in events)
    assert not any(event["type"] == "code_result_summary" for event in events)
    assert events[-1]["type"] == "message_done"


def test_confirmed_mutation_runs_and_reports_state_change(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data['double'] = data['value'] * 2\npreview(data)",
                    "mutates_state": True,
                    "mutation_summary": "Add double column",
                },
            ),
            _tool_response(
                "final_answer",
                {"answer": "Added a double column and saved a new version.", "state_changed": True},
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Add a double column", "active_dataset_id": dataset_id, "confirmed": True},
    )
    events = _parse_sse(response.text)

    result_event = next(event for event in events if event["type"] == "code_result_summary")
    assert len(result_event["updated_datasets"]) == 1
    assert events[-2]["type"] == "final_answer"
    assert events[-2]["state_changed"] is True


def test_artifact_created_event_streams_separately(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {"code": "save_csv('exported values', data)\npreview(data)"},
            ),
            _tool_response("final_answer", {"answer": "Created a CSV artifact."}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Export this as CSV", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    artifacts = [event for event in events if event["type"] == "artifact_created"]
    assert len(artifacts) == 1
    assert artifacts[0]["artifact"]["kind"] == "csv"
    assert Path(artifacts[0]["artifact"]["path"]).exists()

    artifact_id = artifacts[0]["artifact"]["id"]
    content_response = client.get(f"/api/sessions/{session_id}/artifacts/{artifact_id}/content")
    download_response = client.get(f"/api/sessions/{session_id}/artifacts/{artifact_id}/download")

    assert content_response.status_code == 200
    assert "value,group" in content_response.text
    assert download_response.status_code == 200


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
    session_response = client.post("/api/sessions", json={"name": "Agent test"})
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]
    frame = pd.DataFrame({"value": [1, 2, 3], "group": ["a", "b", "a"]})
    upload_response = client.post(
        f"/api/sessions/{session_id}/datasets",
        files={
            "files": (
                "agent-frame.pkl",
                cloudpickle.dumps(frame),
                "application/octet-stream",
            )
        },
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["datasets"][0]["id"]
    return session_id, dataset_id


def _parse_sse(payload: str) -> list[dict]:
    events: list[dict] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events
