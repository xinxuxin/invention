from __future__ import annotations

from app.agent.response_composer import ResponseComposer
from app.agent.types import VerificationResult
from app.runtime.python_executor import ExecutionArtifact, ExecutionResult


def test_inspection_answer_summarizes_preview_without_raw_dict() -> None:
    result = ExecutionResult(
        ok=True,
        stdout="",
        stderr="",
        traceback=None,
        result_preview={
            "object_type": "list",
            "object_length": 24410,
            "sample_examples": [
                {
                    "country": "CA",
                    "doc_number": "186426",
                    "title": "CLEANING ROBOT",
                    "owners": ["Softbank"],
                    "filing_date": "2019-03-06T00:00:00",
                }
            ],
        },
    )

    answer = ResponseComposer().compose(
        user_message="What is in this file?",
        execution_result=result,
        artifacts=[],
        verification=_pass(),
        state_changed=False,
    )

    assert "{'object_type':" not in answer.markdown
    assert "... truncated ..." not in answer.markdown
    assert "Object type: list" in answer.markdown
    assert "Records/items observed: 24,410" in answer.markdown
    assert answer.markdown.count("State changed") == 1


def test_read_only_operation_is_not_labeled_mutation() -> None:
    result = ExecutionResult(
        ok=True,
        stdout="",
        stderr="",
        traceback=None,
        result_preview={"rows": 3},
    )

    answer = ResponseComposer().compose(
        user_message="Count rows",
        execution_result=result,
        artifacts=[],
        verification=_pass(),
        state_changed=False,
        mutation_summary="Count and tabulate rows",
    )

    assert "Mutation:" not in answer.markdown
    assert "Analysis performed: Count and tabulate rows" in answer.markdown


def test_table_answer_references_artifact_instead_of_pasting_rows() -> None:
    table = ExecutionArtifact(
        id="table-1",
        name="Preview",
        kind="table",
        type="table",
        title="Preview",
        columns=[{"key": "country", "label": "country", "type": "value"}],
        rows=[{"country": "CA"}],
        path="/tmp/table.json",
        metadata={"row_count": 1, "columns": [{"key": "country"}]},
    )

    answer = ResponseComposer().compose(
        user_message="Show first rows as a table",
        execution_result=None,
        artifacts=[table],
        verification=_pass(),
        state_changed=False,
    )

    assert "table shown in the chat below" in answer.markdown
    assert "{'country':" not in answer.markdown
    assert "State changed" in answer.markdown


def _pass() -> VerificationResult:
    return VerificationResult(
        passed=True,
        severity="pass",
        should_finalize=True,
        reasons=["ok"],
    )
