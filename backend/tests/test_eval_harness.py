from __future__ import annotations

from pathlib import Path

from scripts.create_agent_test_datasets import write_agent_test_datasets
from scripts.evaluate_agent_demo import (
    GENERATED_DATASET_FILENAMES,
    _build_report,
    _markdown_report,
    _run_check,
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
