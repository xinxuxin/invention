from __future__ import annotations

from pathlib import Path

from scripts.create_agent_test_datasets import write_agent_test_datasets
from scripts.evaluate_agent_demo import (
    GENERATED_DATASET_FILENAMES,
    _assistant_text_from_events,
    _build_report,
    _markdown_report,
    _resolve_confirmation,
    _run_check,
    _state_operation_has_empty_response,
    _summarize_artifacts,
)


def test_generated_agent_datasets_are_created(tmp_path: Path) -> None:
    written = write_agent_test_datasets(tmp_path)

    assert {path.name for path in written} == set(GENERATED_DATASET_FILENAMES.values())
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_table_artifact_column_checks_use_full_content() -> None:
    artifact = {"id": "table-1", "kind": "table", "title": "Preview"}
    content = {
        "columns": [
            {"key": "country"},
            {"key": "doc_number"},
            {"key": "title"},
            {"key": "owners"},
            {"key": "assignees"},
            {"key": "filing_date"},
            {"key": "status"},
        ],
        "rows": [
            {
                "country": "CA",
                "doc_number": "186426",
                "title": "CLEANING ROBOT",
                "owners": ["Softbank"],
                "assignees": ["SOFTBANK ROBOTICS GROUP"],
                "filing_date": "2019-03-06T00:00:00",
                "status": "Filed",
            }
        ],
        "row_count": 5,
        "total_column_count": 34,
    }

    ok, reason = _run_check(
        {
            "table_required_columns": [
                "country",
                "doc_number",
                "title",
                "owners",
                "assignees",
                "filing_date",
                "status",
            ]
        },
        {"answer": "I created a table.", "state_changed": False},
        [artifact],
        {"table-1": content},
        [],
    )

    assert ok, reason
    summary = _summarize_artifacts([artifact], {"table-1": content})[0]
    assert summary["row_count"] == 5
    assert summary["column_count"] == 34


def test_eval_check_rejects_raw_python_repr_answer() -> None:
    ok, reason = _run_check(
        "no_raw_json",
        {"answer": "Object type: {'type': 'type', 'attrs': {'bad': True}}"},
        [],
        {},
        [],
    )

    assert not ok
    assert "raw JSON" in reason


def test_all_records_check_rejects_generic_preview_table() -> None:
    artifact = {"id": "table-1", "kind": "table", "title": "Result preview table"}
    content = {
        "columns": [{"key": "customer_id"}, {"key": "event_id"}],
        "rows": [{"customer_id": "c1", "event_id": "e1"} for _ in range(5)],
        "row_count": 5,
        "preview_row_count": 5,
    }

    ok, reason = _run_check(
        "all_records_full_table",
        {"answer": "Here is the preview."},
        [artifact],
        {"table-1": content},
        [{"type": "code_started", "code": "RESULT = df.head(5)"}],
    )

    assert not ok
    assert "Result preview table" in reason


def test_alert_chart_check_rejects_generic_dataset_chart() -> None:
    artifact = {
        "id": "chart-1",
        "kind": "chart",
        "title": "Dataset chart",
        "chart_spec": {
            "chart_type": "bar",
            "data": [{"row": 0, "record_count": 1}],
            "x": "row",
            "y": "record_count",
        },
    }

    ok, reason = _run_check("alert_count_chart_fields", {"answer": ""}, [artifact], {}, [])

    assert not ok
    assert "generic Dataset chart" in reason


def test_dataset_comparison_chart_requires_dataset_rows() -> None:
    artifact = {
        "id": "chart-1",
        "kind": "chart",
        "title": "Dataset Record Count Comparison",
        "chart_spec": {
            "chart_type": "bar",
            "data": [{"dataset_name": "one", "record_count": 10}],
            "x": "dataset_name",
            "y": "record_count",
        },
    }

    ok, reason = _run_check("dataset_comparison_chart_fields", {"answer": ""}, [artifact], {}, [])

    assert not ok
    assert "expected at least 3" in reason


def test_zero_affected_mutation_must_not_confirm() -> None:
    events = [
        {
            "type": "final_answer",
            "answer": "Full scan found 0 matching records; no mutation was applied.",
            "state_changed": False,
        }
    ]

    ok, reason = _run_check(
        "zero_affected_no_confirmation_or_confirmation_required",
        events[-1],
        [],
        {},
        events,
    )

    assert ok, reason


def test_state_operation_empty_response_is_detected() -> None:
    assert _state_operation_has_empty_response([{"type": "final_answer", "answer": "", "state_changed": True}])
    assert _state_operation_has_empty_response([{"type": "confirmation_required", "operation_summary": ""}])
    assert not _state_operation_has_empty_response(
        [{"type": "confirmation_required", "operation_summary": "Delete all records", "message": "Confirm"}]
    )


def test_confirmation_events_render_as_assistant_text() -> None:
    text = _assistant_text_from_events(
        [
            {
                "type": "confirmation_required",
                "operation_summary": "Delete the last 500 records",
                "expected_effect": "Row count will change from 24,410 to 23,910.",
                "state_impact": "Creates a new version.",
                "current_row_count": 24410,
                "new_row_count": 23910,
                "affected_count": 500,
            }
        ]
    )

    assert "Delete the last 500 records" in text
    assert "23,910" in text
    assert "current rows: 24410" in text


def test_resolve_confirmation_sends_required_phrase(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_json_request(base_url: str, method: str, path: str, payload: dict) -> dict:
        captured.update({"base_url": base_url, "method": method, "path": path, "payload": payload})
        return {"events": [{"type": "final_answer", "answer": "done", "state_changed": True}]}

    monkeypatch.setattr("scripts.evaluate_agent_demo._json_request", fake_json_request)

    events = _resolve_confirmation(
        "http://api",
        "session-1",
        {"confirmation_id": "confirm-1", "required_confirmation_phrase": "Yes, delete all records"},
        action="approve",
    )

    assert captured["payload"] == {"confirmation_phrase": "Yes, delete all records"}
    assert events[0]["from_confirmation_approval"] is True


def test_markdown_report_contains_reviewable_question_sections() -> None:
    report = _build_report(
        suite="unit",
        suite_names=["unit"],
        dataset_paths=["demo.pkl"],
        session_id="session-1",
        base_url="http://localhost:8000",
        frontend_url="http://localhost:5173",
        health_config={"agent_mode": "fake", "verifier_mode": "deterministic"},
        results=[],
    )

    markdown = _markdown_report(report)

    assert "# Data Analysis Agent Evaluation Report" in markdown
    assert "Pass/fail/warning" in markdown
    assert "`session-1`" in markdown
