import json
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import cloudpickle
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from app.agent.agent import MAX_AGENT_STEPS, AgentModelResponse, AgentToolCall
from app.api.chat import get_model_client
from app.core.config import get_settings
from app.db.session import get_session
from app.main import app
from app.models.entities import new_id


class DummyMissingPatent:
    def __init__(self, **fields: object) -> None:
        self.attrs = {"__dict__": fields}

    def __repr__(self) -> str:
        return (
            "<MissingPickleClass filing.patent_portfolio.api_classes."
            f"GenericPatentMetadata attrs={self.attrs!r}>"
        )


class ScriptedModelClient:
    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = responses
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    def create_response(self, **_: object) -> AgentModelResponse:
        self.requests.append(dict(_))
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
    assert "table shown in the chat below" in events[-2]["answer"]
    assert fake.calls == 1
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


def test_agent_returns_final_answer_after_retry_budget(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response("execute_python", {"code": "preview(data['missing_one'])"}),
            _tool_response("execute_python", {"code": "preview(data['missing_two'])"}),
            _tool_response("execute_python", {"code": "preview(data['missing_three'])"}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Trigger the retry limit", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert [event["ok"] for event in events if event["type"] == "code_result_summary"] == [False, False, False]
    assert events[-2]["type"] == "final_answer"
    assert "could not complete" in events[-2]["answer"].lower()
    assert not any(event["type"] == "error" and "Maximum Python retry" in event["message"] for event in events)


def test_agent_summarizes_latest_execution_after_step_limit(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [_tool_response("execute_python", {"code": "print('useful step')\npreview({'rows': len(data)})"})]
        + [
            _tool_response("execute_python", {"code": "pass"})
            for _ in range(MAX_AGENT_STEPS - 1)
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Run repeated execution without final answer", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert len([event for event in events if event["type"] == "code_result_summary"]) == 1
    assert events[-2]["type"] == "final_answer"
    assert "step budget" not in events[-2]["answer"].lower()
    assert not any(event["type"] == "error" and "step limit" in event["message"] for event in events)


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
    confirmation = next(event for event in events if event["type"] == "confirmation_required")

    assert confirmation["confirmation_id"]
    assert confirmation["title"] == "Confirm dataset mutation"
    assert confirmation["operation_summary"] == "Drop value column"
    assert confirmation["dataset_name"] == "agent_frame"
    assert confirmation["expected_effect"]
    assert confirmation["state_impact"]
    assert confirmation["reversible"] is True
    assert confirmation["proposed_code"] == "data.drop(columns=['value'], inplace=True)"
    assert confirmation["risk_level"] == "high"
    assert confirmation["affected_dataset_ids"] == [dataset_id]
    assert not any(event["type"] == "code_result_summary" for event in events)
    assert events[-1]["type"] == "message_done"


def test_approving_pending_confirmation_executes_and_versions_dataset(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data.drop(columns=['value'], inplace=True)\npreview(data)",
                    "mutates_state": True,
                    "mutation_summary": "Drop value column",
                },
            ),
            _tool_response(
                "final_answer",
                {"answer": "Dropped the value column and saved a new version.", "state_changed": True},
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Drop the value column", "active_dataset_id": dataset_id},
    )
    confirmation = next(event for event in _parse_sse(response.text) if event["type"] == "confirmation_required")

    approve_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
    )
    payload = approve_response.json()
    events = payload["events"]
    result_event = next(event for event in events if event["type"] == "code_result_summary")

    assert approve_response.status_code == 200
    assert payload["confirmation"]["status"] == "approved"
    assert result_event["ok"] is True
    assert len(result_event["updated_datasets"]) == 1
    assert events[-2]["type"] == "final_answer"
    assert events[-2]["state_changed"] is True

    history_response = client.get(f"/api/sessions/{session_id}/history")
    versions = [item for item in history_response.json()["versions"] if item["dataset_id"] == dataset_id]
    assert len(versions) == 2
    assert versions[-1]["created_by_message_id"] == confirmation["confirmation_id"]


def test_rejecting_pending_confirmation_does_not_mutate_dataset(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data.drop(columns=['value'], inplace=True)\npreview(data)",
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
    confirmation = next(event for event in _parse_sse(response.text) if event["type"] == "confirmation_required")

    reject_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/reject",
    )
    payload = reject_response.json()

    assert reject_response.status_code == 200
    assert payload["confirmation"]["status"] == "rejected"
    assert payload["events"][0]["type"] == "final_answer"
    assert payload["events"][0]["state_changed"] is False

    history_response = client.get(f"/api/sessions/{session_id}/history")
    versions = [item for item in history_response.json()["versions"] if item["dataset_id"] == dataset_id]
    assert len(versions) == 1


def test_chat_rollback_requires_confirmation_and_approval_creates_version(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data.drop(columns=['value'], inplace=True)\npreview(data)",
                    "mutates_state": True,
                    "mutation_summary": "Drop value column",
                },
            ),
            _tool_response("final_answer", {"answer": "Added a double column.", "state_changed": True}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake
    mutation_response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Add a double column", "active_dataset_id": dataset_id, "confirmed": True},
    )
    assert mutation_response.status_code == 200

    rollback_response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "rollback one step", "active_dataset_id": dataset_id},
    )
    confirmation = next(
        event for event in _parse_sse(rollback_response.text) if event["type"] == "confirmation_required"
    )

    assert confirmation["risk_level"] == "high"
    assert confirmation["affected_dataset_ids"] == [dataset_id]

    approve_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["confirmation"]["status"] == "approved"

    history_response = client.get(f"/api/sessions/{session_id}/history")
    versions = [item for item in history_response.json()["versions"] if item["dataset_id"] == dataset_id]
    assert len(versions) == 3
    assert versions[-1]["created_by_message_id"] == confirmation["confirmation_id"]
    assert versions[-1]["mutation_summary"].startswith("Rollback to:")


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
    assert '"value","group"' in content_response.text
    assert download_response.status_code == 200


def test_chart_artifact_streams_with_valid_spec(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "chart_rows = data.groupby('group', as_index=False)['value'].sum().to_dict('records')",
                            "save_chart('Group totals', {'title': 'Group totals', 'chart_type': 'bar', 'data': chart_rows, 'x': 'group', 'y': 'value', 'description': 'Total value by group'})",
                            "preview(chart_rows)",
                        ]
                    )
                },
            ),
            _tool_response("final_answer", {"answer": "Created a bar chart."}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Visualize this", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    chart = next(event["artifact"] for event in events if event["type"] == "artifact_created")
    content = client.get(f"/api/sessions/{session_id}/artifacts/{chart['id']}/content").json()

    assert response.status_code == 200
    assert chart["kind"] == "chart"
    assert chart["metadata"]["chart_type"] == "bar"
    assert chart["chart_spec"]["data"]
    assert content["title"] == "Group totals"
    assert content["chart_spec"]["x"] == "group"
    assert content["chart_spec"]["y"] == "value"


def test_chat_messages_and_artifacts_are_persisted_for_restore(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "rows = data.head(2)",
                            "save_table('Restored preview', rows)",
                            "RESULT = {'rows': len(rows)}",
                        ]
                    )
                },
            ),
            _tool_response("final_answer", {"answer": "I created the preview table.", "state_changed": False}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    stream_response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Show a preview table", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(stream_response.text)
    artifact_id = next(event["artifact"]["id"] for event in events if event["type"] == "artifact_created")

    messages_response = client.get(f"/api/sessions/{session_id}/messages")
    messages = messages_response.json()["messages"]
    sessions_response = client.get("/api/sessions")
    artifacts_response = client.get(f"/api/sessions/{session_id}/artifacts")

    assert messages_response.status_code == 200
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Show a preview table"
    assert "table" in messages[1]["final_answer"].lower()
    assert messages[1]["trace_events"]
    assert messages[1]["artifact_ids"] == [artifact_id]
    assert messages[1]["artifacts"][0]["id"] == artifact_id
    assert sessions_response.json()["sessions"][0]["message_count"] == 2
    assert artifacts_response.json()[0]["id"] == artifact_id


def test_confirmation_response_updates_persisted_assistant_message(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data.drop(columns=['value'], inplace=True)\npreview(data)",
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
    confirmation = next(event for event in _parse_sse(response.text) if event["type"] == "confirmation_required")

    approve_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
    )
    messages = client.get(f"/api/sessions/{session_id}/messages").json()["messages"]

    assert approve_response.status_code == 200
    assert messages[-1]["status"] == "done"
    assert messages[-1]["state_changed"] is True
    assert "Saved" in messages[-1]["final_answer"] or "Applied" in messages[-1]["final_answer"]


def test_agent_context_and_executor_include_multiple_datasets(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "Multi agent test"})
    session_id = session_response.json()["id"]
    customers_id = _upload_frame(
        client,
        session_id,
        "Customers 2026.pkl",
        pd.DataFrame({"customer_id": [1, 2], "segment": ["trial", "paid"]}),
    )
    _upload_frame(
        client,
        session_id,
        "Orders 2026.pkl",
        pd.DataFrame({"customer_id": [1, 2], "revenue": [100, 250]}),
    )
    fake = ScriptedModelClient(
        [
            _tool_response("execute_python", {"code": "print(sorted(datasets.keys()))\npreview(datasets)"}),
            _tool_response("final_answer", {"answer": "Both datasets are available."}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Compare these two datasets", "active_dataset_id": customers_id},
    )
    events = _parse_sse(response.text)
    stdout = next(event["stdout"] for event in events if event["type"] == "code_result_summary")
    first_input = json.dumps(fake.requests[0]["input_items"], default=str)

    assert response.status_code == 200
    assert "customers_2026" in stdout
    assert "orders_2026" in stdout
    assert "dataset_keys" in first_input
    assert "active_dataset_key" in first_input
    assert "customers_2026" in first_input


def test_agent_softbank_like_prompt_gets_non_null_preview_without_retry(client: TestClient) -> None:
    class DummyMissing:
        def __init__(self, country: str, title: str) -> None:
            self.attrs = {
                "__dict__": {
                    "country": country,
                    "title": title,
                    "owners": ["Softbank"],
                    "filing_date": pd.Timestamp("2019-03-06"),
                }
            }

    session_response = client.post("/api/sessions", json={"name": "SoftBank-like agent test"})
    session_id = session_response.json()["id"]
    dataset_id = _upload_pickle(
        client,
        session_id,
        "softbank-like.pkl",
        [DummyMissing("CA", "CLEANING ROBOT"), DummyMissing("CN", "ROBOT CONTROL")],
    )
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "frame = to_dataframe(data)",
                            "{'columns': list(frame.columns), 'rows': frame.head(5).to_dict('records')}",
                        ]
                    )
                },
            ),
            _tool_response("final_answer", {"answer": "Converted the custom objects into a table preview."}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={
            "message": "Convert this dataset into a tabular preview if possible. Show columns and 5 rows.",
            "active_dataset_id": dataset_id,
        },
    )
    events = _parse_sse(response.text)
    result_events = [event for event in events if event["type"] == "code_result_summary"]

    assert response.status_code == 200
    assert len(result_events) == 1
    assert result_events[0]["ok"] is True
    assert result_events[0]["result_preview"]["columns"] == ["country", "title", "owners", "filing_date"]
    assert result_events[0]["result_preview"]["rows"][0]["title"] == "CLEANING ROBOT"
    assert events[-2]["type"] == "final_answer"


def test_agent_softbank_patent_exported_chat_failure_regression(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "SoftBank patent regression"})
    session_id = session_response.json()["id"]
    dataset_id = _upload_pickle(client, session_id, "softbank-patents.pkl", _softbank_patent_dataset())
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "df = to_dataframe(data)",
                            "RESULT = {",
                            "    'columns': list(df.columns),",
                            "    'rows': df.head(5).to_dict(orient='records'),",
                            "}",
                        ]
                    )
                },
            ),
            _tool_response("final_answer", {"answer": "Inferred columns and rows from patent metadata."}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={
            "message": "Convert this dataset into a tabular preview if possible. Show inferred columns and first 5 rows.",
            "active_dataset_id": dataset_id,
        },
    )
    events = _parse_sse(response.text)
    result_events = [event for event in events if event["type"] == "code_result_summary"]

    assert response.status_code == 200
    assert len(result_events) == 1
    assert result_events[0]["ok"] is True
    assert result_events[0]["result_preview"] is not None
    assert result_events[0]["result_preview"]["columns"][:4] == ["country", "doc_number", "kind", "title"]
    assert result_events[0]["result_preview"]["rows"][0]["title"] == "CLEANING ROBOT"
    table_artifact = next(event["artifact"] for event in events if event["type"] == "artifact_created")
    table_payload = client.get(f"/api/sessions/{session_id}/artifacts/{table_artifact['id']}/content").json()
    assert table_artifact["kind"] == "table"
    assert [column["key"] for column in table_payload["columns"][:4]] == [
        "country",
        "doc_number",
        "kind",
        "title",
    ]
    assert table_payload["rows"][0]["title"] == "CLEANING ROBOT"
    assert not any(event["type"] == "error" and "step budget" in event.get("message", "") for event in events)
    assert events[-2]["type"] == "final_answer"


def test_table_prompt_streams_inline_artifact_from_result_preview(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "rows = data.head(5).to_dict(orient='records')",
                            "RESULT = {'columns': list(data.columns), 'rows': rows}",
                        ]
                    )
                },
            )
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Show the inferred columns and the first 5 rows", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    artifact_events = [event for event in events if event["type"] == "artifact_created"]

    assert response.status_code == 200
    assert len(artifact_events) == 1
    assert artifact_events[0]["artifact"]["kind"] == "table"
    assert events[-2]["type"] == "final_answer"
    assert "shown in the chat" in events[-2]["answer"]


def test_verifier_retries_when_table_artifact_missing(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response("execute_python", {"code": "RESULT = {'summary': 'I saw rows but made no table'}"}),
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "df = to_dataframe(data)",
                            "save_table('Verified preview', df.head(5))",
                            "RESULT = {'columns': list(df.columns), 'rows': df.head(5).to_dict('records')}",
                        ]
                    )
                },
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Show the first 5 rows as a table", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert len([event for event in events if event["type"] == "code_result_summary"]) == 2
    assert any(
        event["type"] == "verifier_result" and event["severity"] == "retry"
        for event in events
    )
    assert any(event["type"] == "artifact_created" and event["artifact"]["kind"] == "table" for event in events)
    assert events[-2]["type"] == "final_answer"


def test_agent_stops_after_useful_inspection_result(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": (
                        "{'object_type': type(data).__name__, "
                        "'object_length': len(data), "
                        "'sample_examples': data.head(2).to_dict('records')}"
                    )
                },
            ),
            _tool_response("execute_python", {"code": "raise RuntimeError('should not run')"}),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "What is in this file?", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert fake.calls == 1
    assert any(
        event["type"] == "trace" and "Found enough information" in event.get("message", "")
        for event in events
    )
    assert events[-2]["type"] == "final_answer"
    assert "{'object_type':" not in events[-2]["answer"]
    assert "Representative fields" in events[-2]["answer"]
    assert not any(event["type"] == "error" for event in events)


def test_ambiguous_destructive_prompt_asks_clarification_without_python(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient([_tool_response("execute_python", {"code": "raise RuntimeError('should not run')"})])
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Remove bad records.", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    final = events[-2]

    assert response.status_code == 200
    assert fake.calls == 0
    assert any(
        event["type"] == "trace" and "Clarification needed" in event.get("message", "")
        for event in events
    )
    assert not any(event["type"] == "code_started" for event in events)
    assert final["type"] == "final_answer"
    assert "Please choose the rule" in final["answer"]
    assert final["state_changed"] is False


def test_clean_dataset_prompt_asks_clarification_without_python(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient([_tool_response("execute_python", {"code": "raise RuntimeError('should not run')"})])
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Clean this dataset.", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert fake.calls == 0
    assert not any(event["type"] == "code_result_summary" for event in events)
    assert "missing title" in events[-2]["answer"].lower()


def test_specific_destructive_prompt_can_proceed_to_confirmation(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "Agent test"})
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]
    dataset_id = _upload_frame(
        client,
        session_id,
        "titles.pkl",
        pd.DataFrame({"title": ["A", "", None], "group": ["a", "b", "c"]}),
    )
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "data = data[data['group'].notna()].copy()",
                    "mutates_state": True,
                    "mutation_summary": "Remove records with missing group values",
                },
            )
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Remove records with missing titles", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert fake.calls == 0
    assert any(event["type"] == "confirmation_required" for event in events)


def test_delete_last_entries_shortcut_confirms_and_approval_mutates(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient([_tool_response("execute_python", {"code": "raise RuntimeError('should not run')"})])
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "delete last 2 entries", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    confirmation = next(event for event in events if event["type"] == "confirmation_required")

    assert fake.calls == 0
    assert confirmation["operation_summary"] == "Delete the last 2 records from the current working dataset"
    assert confirmation["current_row_count"] == 3
    assert confirmation["new_row_count"] == 1
    assert confirmation["affected_count"] == 2

    approve_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
    )

    assert approve_response.status_code == 200
    result_event = next(event for event in approve_response.json()["events"] if event["type"] == "code_result_summary")
    assert result_event["ok"] is True
    dataset_response = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}")
    assert dataset_response.json()["profile"]["shape"] == [1, 2]


def test_delete_everything_requires_phrase_and_rolls_back(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient([_tool_response("execute_python", {"code": "raise RuntimeError('should not run')"})])
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Delete everything.", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    confirmation = next(event for event in events if event["type"] == "confirmation_required")

    assert fake.calls == 0
    assert "bad record" not in json.dumps(events).lower()
    assert confirmation["risk_level"] == "high"
    assert confirmation["operation_summary"] == "Delete all records from the current working dataset"
    assert confirmation["affected_count"] == 3
    assert confirmation["new_row_count"] == 0
    assert confirmation["required_confirmation_phrase"] == "Yes, delete all records"

    missing_phrase = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
    )
    assert missing_phrase.status_code == 400
    assert client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}").json()["profile"]["shape"] == [3, 2]

    approve_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
        json={"confirmation_phrase": "Yes, delete all records"},
    )
    assert approve_response.status_code == 200
    assert client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}").json()["profile"]["shape"] == [0, 2]

    rollback = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Roll back to the original uploaded dataset.", "active_dataset_id": dataset_id},
    )
    rollback_confirmation = next(event for event in _parse_sse(rollback.text) if event["type"] == "confirmation_required")
    rollback_approve = client.post(
        f"/api/sessions/{session_id}/confirmations/{rollback_confirmation['confirmation_id']}/approve",
    )
    assert rollback_approve.status_code == 200
    assert client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}").json()["profile"]["shape"] == [3, 2]


def test_reject_delete_everything_leaves_dataset_unchanged(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Delete everything.", "active_dataset_id": dataset_id},
    )
    confirmation = next(event for event in _parse_sse(response.text) if event["type"] == "confirmation_required")
    reject = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/reject",
    )

    assert reject.status_code == 200
    assert reject.json()["events"][-2]["state_changed"] is False
    assert client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}").json()["profile"]["shape"] == [3, 2]


def test_custom_object_low_battery_mutation_is_clear_unsupported_noop(client: TestClient) -> None:
    class SensorFleet:
        def __init__(self) -> None:
            self.sensors = [{"readings": [{"battery_pct": 10}, {"battery_pct": 90}]}]

    session_response = client.post("/api/sessions", json={"name": "Custom object no-op"})
    session_id = session_response.json()["id"]
    dataset_id = _upload_pickle(client, session_id, "fleet.pkl", SensorFleet())

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Remove readings with battery_pct below 80, but ask for confirmation first.", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert not any(event["type"] == "confirmation_required" for event in events)
    assert events[-2]["type"] == "final_answer"
    assert events[-2]["state_changed"] is False
    assert "custom object" in events[-2]["answer"].lower()


def test_delete_empty_title_scans_full_dataset_and_confirms(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "Delete title test"})
    session_id = session_response.json()["id"]
    dataset_id = _upload_frame(
        client,
        session_id,
        "titles.pkl",
        pd.DataFrame({"title": ["A", "", None, "D"], "country": ["US", "CA", "CN", "JP"]}),
    )
    fake = ScriptedModelClient([_tool_response("execute_python", {"code": "raise RuntimeError('should not run')"})])
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "delete empty title", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    confirmation = next(event for event in events if event["type"] == "confirmation_required")

    assert fake.calls == 0
    assert confirmation["affected_count"] == 2
    assert confirmation["current_row_count"] == 4
    assert confirmation["new_row_count"] == 2

    approve_response = client.post(
        f"/api/sessions/{session_id}/confirmations/{confirmation['confirmation_id']}/approve",
    )
    dataset_response = client.get(f"/api/sessions/{session_id}/datasets/{dataset_id}")

    assert approve_response.status_code == 200
    assert dataset_response.json()["profile"]["shape"] == [2, 2]


def test_delete_empty_title_zero_matches_reports_full_scan_no_mutation(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "Delete title no-op"})
    session_id = session_response.json()["id"]
    dataset_id = _upload_frame(
        client,
        session_id,
        "titles-ok.pkl",
        pd.DataFrame({"title": ["A", "B"], "country": ["US", "CA"]}),
    )

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "delete empty title", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)

    assert not any(event["type"] == "confirmation_required" for event in events)
    assert "scanned all 2 records" in events[-2]["answer"].lower()
    assert events[-2]["state_changed"] is False


def test_mutation_history_shortcut_does_not_call_python(client: TestClient) -> None:
    session_id, dataset_id = _create_dataset(client)
    fake = ScriptedModelClient([_tool_response("execute_python", {"code": "history"})])
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "Show me the mutation history so far.", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    answer = events[-2]["answer"]

    assert response.status_code == 200
    assert fake.calls == 0
    assert "Current branch" in answer
    assert "NameError" not in answer
    assert events[-2]["state_changed"] is False


def test_filing_date_retry_discards_sample_artifact_and_dedupes(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "Filing date transaction test"})
    session_id = session_response.json()["id"]
    frame = pd.DataFrame(
        {
            "filing_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2021-03-01", "2021-04-01"]),
            "country": ["US", "CA", "US", "CN"],
            "title": ["A", "B", "C", "D"],
        }
    )
    dataset_id = _upload_frame(client, session_id, "filings.pkl", frame)
    fake = ScriptedModelClient(
        [
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "sample = to_dataframe(data, limit=2)",
                            "dates = pd.to_datetime(sample['filing_date'], errors='coerce')",
                            "grouped = dates.dropna().dt.year.value_counts().sort_index().reset_index()",
                            "grouped.columns = ['filing_year', 'filing_count']",
                            "grouped.attrs['source_total_row_count'] = len(data)",
                            "grouped.attrs['source_row_count'] = len(data)",
                            "grouped.attrs['analyzed_row_count'] = len(sample)",
                            "save_table('Filings by year (sample)', grouped)",
                            "RESULT = {'filing_date_min': dates.min().date().isoformat(), 'filing_date_max': dates.max().date().isoformat(), 'source_total_row_count': len(data), 'analyzed_row_count': len(sample)}",
                        ]
                    )
                },
            ),
            _tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "df = to_dataframe(data, limit=None)",
                            "dates = pd.to_datetime(df['filing_date'], errors='coerce')",
                            "grouped = dates.dropna().dt.year.value_counts().sort_index().reset_index()",
                            "grouped.columns = ['filing_year', 'filing_count']",
                            "grouped.attrs['source_total_row_count'] = len(df)",
                            "grouped.attrs['source_row_count'] = len(df)",
                            "grouped.attrs['analyzed_row_count'] = len(df)",
                            "save_table('Filings by year (filing_date)', grouped)",
                            "RESULT = {'filing_date_min': dates.min().date().isoformat(), 'filing_date_max': dates.max().date().isoformat(), 'source_total_row_count': len(df), 'analyzed_row_count': len(df)}",
                        ]
                    )
                },
            ),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={
            "message": "What is the filing date range of this portfolio? Also summarize the number of filings by year in a table.",
            "active_dataset_id": dataset_id,
        },
    )
    events = _parse_sse(response.text)
    artifacts = [event["artifact"] for event in events if event["type"] == "artifact_created"]

    assert response.status_code == 200
    assert len([event for event in events if event["type"] == "code_result_summary"]) == 2
    assert len(artifacts) == 1
    assert artifacts[0]["title"] == "Filings by year (filing_date)"
    assert "sample" not in artifacts[0]["title"].lower()
    assert [column["key"] for column in artifacts[0]["columns"]] == ["filing_year", "filing_count"]
    assert events[-2]["type"] == "final_answer"
    assert "step budget" not in events[-2]["answer"].lower()


def test_country_pie_chart_shortcut_creates_valid_chart_without_model_call(client: TestClient) -> None:
    session_response = client.post("/api/sessions", json={"name": "Agent test"})
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]
    dataset_id = _upload_pickle(client, session_id, "patents.pkl", _softbank_patent_dataset())
    app.dependency_overrides.pop(get_model_client, None)

    response = client.post(
        f"/api/sessions/{session_id}/chat/stream",
        json={"message": "create a pie chart for country", "active_dataset_id": dataset_id},
    )
    events = _parse_sse(response.text)
    artifacts = [event["artifact"] for event in events if event["type"] == "artifact_created"]
    chart = next(artifact for artifact in artifacts if artifact["kind"] == "chart")

    assert response.status_code == 200
    assert chart["chart_spec"]["chart_type"] == "pie"
    assert chart["chart_spec"]["x"] == "country"
    assert chart["chart_spec"]["data"]
    assert not any("NameError" in str(event) for event in events)


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
    dataset_id = _upload_frame(client, session_id, "agent-frame.pkl", frame)
    return session_id, dataset_id


def _upload_frame(client: TestClient, session_id: str, filename: str, frame: pd.DataFrame) -> str:
    return _upload_pickle(client, session_id, filename, frame)


def _upload_pickle(client: TestClient, session_id: str, filename: str, value: object) -> str:
    upload_response = client.post(
        f"/api/sessions/{session_id}/datasets",
        files={
            "files": (
                filename,
                cloudpickle.dumps(value),
                "application/octet-stream",
            )
        },
    )
    assert upload_response.status_code == 201
    return upload_response.json()["datasets"][0]["id"]


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


def _parse_sse(payload: str) -> list[dict]:
    events: list[dict] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events
