from __future__ import annotations

import time

from app.agent.llm_verifier import LLMVerifier
from app.agent.types import LLMVerificationResult
from app.agent.verifier import ResultVerifier, merge_verification_results
from app.core.config import Settings
from app.runtime.python_executor import ExecutionArtifact, ExecutionResult


def test_table_request_without_table_artifact_retries() -> None:
    result = ResultVerifier().verify(
        user_message="Show the inferred columns and first 5 rows",
        execution_result=_execution(result_preview={"summary": "ok"}),
        artifacts_created_this_turn=[],
    )

    assert result.severity == "retry"
    assert "save_table" in result.retry_instruction


def test_chart_request_without_chart_artifact_retries() -> None:
    result = ResultVerifier().verify(
        user_message="Create a bar chart by country",
        execution_result=_execution(result_preview={"summary": "ok"}),
        artifacts_created_this_turn=[],
    )

    assert result.severity == "retry"
    assert "save_chart" in result.retry_instruction


def test_csv_request_without_csv_artifact_retries() -> None:
    result = ResultVerifier().verify(
        user_message="Export this as CSV",
        execution_result=_execution(result_preview={"summary": "ok"}),
        artifacts_created_this_turn=[],
    )

    assert result.severity == "retry"
    assert "save_csv" in result.retry_instruction


def test_wrapper_columns_retry() -> None:
    table = _artifact(
        "table",
        columns=[{"key": "__dict__", "label": "__dict__", "type": "object"}],
    )
    result = ResultVerifier().verify(
        user_message="Show a table",
        execution_result=_execution(result_preview={"rows": 1}),
        artifacts_created_this_turn=[table],
    )

    assert result.severity == "retry"
    assert "wrapper columns" in result.retry_instruction


def test_schema_wrapper_only_result_retries() -> None:
    result = ResultVerifier().verify(
        user_message="Which fields are scalar fields, dates, and list-like fields?",
        execution_result=_execution(
            result_preview={
                "scalar_fields": [],
                "date_fields": [],
                "list_fields": ["__pydantic_fields_set__"],
                "unknown_fields": ["__dict__"],
            }
        ),
    )

    assert result.severity == "retry"
    assert "wrapper" in result.retry_instruction.lower() or "domain fields" in result.retry_instruction.lower()


def test_sample_only_top_country_result_retries() -> None:
    table = _artifact(
        "table",
        columns=[{"key": "country"}, {"key": "patent_count"}],
        metadata={"source_row_count": 24410, "analyzed_row_count": 20, "row_count": 2},
    )
    result = ResultVerifier().verify(
        user_message="Show the top countries by number of patent records",
        execution_result=_execution(result_preview={"source_row_count": 24410, "analyzed_row_count": 20}),
        artifacts_created_this_turn=[table],
        latest_code="df = to_dataframe(data, limit=20)",
    )

    assert result.severity == "retry"
    assert "full dataset" in result.retry_instruction


def test_mutation_without_confirmation_retries() -> None:
    result = ResultVerifier().verify(
        user_message="delete last 500 entries",
        execution_result=_execution(result_preview={"length": 24410}),
        state_changed=False,
    )

    assert result.severity == "retry"
    assert "request_confirmation" in result.retry_instruction


def test_invalid_chart_artifact_retries() -> None:
    chart = ExecutionArtifact(
        id="chart-1",
        name="Broken chart",
        kind="chart",
        type="chart",
        title="Broken chart",
        chart_spec={"chart_type": "bar", "data": [{"country": "CA"}], "x": "country", "y": "count"},
        path="/tmp/chart.json",
        metadata={"chart_spec": {"chart_type": "bar", "data": [{"country": "CA"}], "x": "country", "y": "count"}},
    )
    result = ResultVerifier().verify(
        user_message="Create a bar chart",
        execution_result=_execution(result_preview={"ok": True}),
        artifacts_created_this_turn=[chart],
    )

    assert result.severity == "retry"
    assert "Chart artifact is invalid" in result.reasons[0]


def test_preview_request_rejects_top_countries_table() -> None:
    table = ExecutionArtifact(
        id="table-1",
        name="Top countries",
        kind="table",
        type="table",
        title="Top countries",
        columns=[{"key": "country"}, {"key": "record_count"}],
        rows=[{"country": "CN", "record_count": 19}, {"country": "CA", "record_count": 1}],
        path="/tmp/table.json",
        metadata={"columns": [{"key": "country"}, {"key": "record_count"}]},
    )
    result = ResultVerifier().verify(
        user_message="Convert this dataset into a tabular preview. Show inferred columns and first 5 rows.",
        execution_result=_execution(result_preview={"columns": ["country", "record_count"]}),
        artifacts_created_this_turn=[table],
    )

    assert result.severity == "retry"
    assert "country-count" in " ".join(result.reasons)


def test_filing_date_request_rejects_top_countries_table() -> None:
    table = ExecutionArtifact(
        id="table-1",
        name="Top countries",
        kind="table",
        type="table",
        title="Top countries",
        columns=[{"key": "country"}, {"key": "record_count"}],
        rows=[{"country": "CN", "record_count": 19}],
        path="/tmp/table.json",
        metadata={"columns": [{"key": "country"}, {"key": "record_count"}]},
    )
    result = ResultVerifier().verify(
        user_message="What is the filing date range? Also summarize filings by year in a table.",
        execution_result=_execution(result_preview={"summary": "ok"}),
        artifacts_created_this_turn=[table],
    )

    assert result.severity == "retry"
    assert "filings by year" in " ".join(result.reasons).lower()


def test_filing_date_request_accepts_year_count_table() -> None:
    table = ExecutionArtifact(
        id="table-1",
        name="Patent filings by year",
        kind="table",
        type="table",
        title="Patent filings by year",
        columns=[{"key": "filing_year"}, {"key": "count"}],
        rows=[{"filing_year": 2020, "count": 4}],
        path="/tmp/table.json",
        metadata={
            "columns": [{"key": "filing_year"}, {"key": "count"}],
            "row_count": 1,
            "source_total_row_count": 10,
            "analyzed_row_count": 10,
        },
    )
    result = ResultVerifier().verify(
        user_message="What is the filing date range? Also summarize filings by year in a table.",
        execution_result=_execution(
            result_preview={
                "filing_date_min": "2020-01-01",
                "filing_date_max": "2020-12-31",
                "source_total_row_count": 10,
                "analyzed_row_count": 10,
            }
        ),
        artifacts_created_this_turn=[table],
        all_artifacts_for_message=[
            _artifact(
                "table",
                columns=[{"key": "country"}, {"key": "record_count"}],
                metadata={"row_count": 1},
            )
        ],
    )

    assert result.severity == "pass"


def test_filing_date_sample_table_retries() -> None:
    table = ExecutionArtifact(
        id="table-1",
        name="Filings by year sample",
        kind="table",
        type="table",
        title="Filings by year sample",
        columns=[{"key": "filing_year"}, {"key": "filing_count"}],
        rows=[{"filing_year": 2020, "filing_count": 4}],
        path="/tmp/table.json",
        metadata={
            "columns": [{"key": "filing_year"}, {"key": "filing_count"}],
            "row_count": 1,
            "source_total_row_count": 24410,
            "analyzed_row_count": 1000,
        },
    )
    result = ResultVerifier().verify(
        user_message="What is the filing date range? Also summarize filings by year in a table.",
        execution_result=_execution(
            result_preview={
                "filing_date_min": "2020-01-01",
                "filing_date_max": "2020-12-31",
                "source_total_row_count": 24410,
                "analyzed_row_count": 1000,
            }
        ),
        artifacts_created_this_turn=[table],
    )

    assert result.severity == "retry"
    assert "sampled" in " ".join(result.reasons).lower() or "full dataset" in result.retry_instruction


def test_line_chart_request_rejects_bar_chart() -> None:
    chart = ExecutionArtifact(
        id="chart-1",
        name="Patent filings by year",
        kind="chart",
        type="chart",
        title="Patent filings by year",
        chart_spec={
            "chart_type": "bar",
            "data": [{"filing_year": 2020, "filing_count": 5}],
            "x": "filing_year",
            "y": "filing_count",
        },
        path="/tmp/chart.json",
        metadata={
            "chart_spec": {
                "chart_type": "bar",
                "data": [{"filing_year": 2020, "filing_count": 5}],
                "x": "filing_year",
                "y": "filing_count",
            }
        },
    )
    result = ResultVerifier().verify(
        user_message="Create a line chart showing filings by year based on filing_date.",
        execution_result=_execution(result_preview={"ok": True}),
        artifacts_created_this_turn=[chart],
    )

    assert result.severity == "retry"
    assert "line chart" in " ".join(result.reasons).lower()


def test_country_count_table_matches_country_request() -> None:
    table = ExecutionArtifact(
        id="table-1",
        name="Top countries",
        kind="table",
        type="table",
        title="Top countries",
        columns=[{"key": "country"}, {"key": "record_count"}],
        rows=[{"country": "CN", "record_count": 19}, {"country": "CA", "record_count": 1}],
        path="/tmp/table.json",
        metadata={
            "columns": [{"key": "country"}, {"key": "record_count"}],
            "source_row_count": 20,
            "analyzed_row_count": 20,
        },
    )
    result = ResultVerifier().verify(
        user_message="Show top countries by number of patent records",
        execution_result=_execution(result_preview={"source_row_count": 20, "analyzed_row_count": 20}),
        artifacts_created_this_turn=[table],
    )

    assert result.severity == "pass"


def test_final_answer_raw_attrs_retries() -> None:
    result = ResultVerifier().verify(
        user_message="What is in this file?",
        final_answer_draft="Object type: {'type': 'type', 'attrs': {'__dict__': {}}}",
    )

    assert result.severity == "retry"
    assert "wrapper" in result.retry_instruction.lower()


def test_result_nameerror_retry_mentions_isolated_execution() -> None:
    result = ResultVerifier().verify(
        user_message="create a pie chart for country",
        execution_result=ExecutionResult(
            ok=False,
            stdout="",
            stderr="",
            traceback="NameError: name 'RESULT' is not defined",
            result_preview=None,
        ),
        retries_remaining=True,
    )

    assert result.severity == "retry"
    assert "RESULT is not preserved" in result.retry_instruction


def test_table_nameerror_retry_mentions_atomic_chart_block() -> None:
    result = ResultVerifier().verify(
        user_message="create a pie chart for country",
        execution_result=ExecutionResult(
            ok=False,
            stdout="",
            stderr="",
            traceback="NameError: name 'table' is not defined",
            result_preview=None,
        ),
        retries_remaining=True,
    )

    assert result.severity == "retry"
    assert "Recompute chart data" in result.retry_instruction


def test_missing_chart_data_error_retries_with_data_instruction() -> None:
    result = ResultVerifier().verify(
        user_message="create a pie chart for country",
        execution_result=ExecutionResult(
            ok=False,
            stdout="",
            stderr="",
            traceback="ValueError: chart_spec.data must be a non-empty list of dict rows; got str",
            result_preview=None,
        ),
        retries_remaining=True,
    )

    assert result.severity == "retry"
    assert "chart_spec['data']" in result.retry_instruction


def test_useful_inspection_preview_finalizes() -> None:
    result = ResultVerifier().verify(
        user_message="What is in this file?",
        execution_result=_execution(result_preview={"object_type": "list", "length": 2}),
    )

    assert result.passed is True
    assert result.should_finalize is True


def test_execution_failure_retries_with_traceback() -> None:
    result = ResultVerifier().verify(
        user_message="Summarize this",
        execution_result=ExecutionResult(
            ok=False,
            stdout="",
            stderr="",
            traceback="NameError: name 'x' is not defined",
            result_preview=None,
        ),
        retries_remaining=True,
    )

    assert result.severity == "retry"
    assert "NameError" in result.retry_instruction


def test_non_mutating_prompt_changed_state_hard_fails() -> None:
    result = ResultVerifier().verify(
        user_message="What is in this file?",
        execution_result=_execution(result_preview={"ok": True}),
        state_changed=True,
        retries_remaining=False,
    )

    assert result.hard_fail is True
    assert result.severity == "fail"


def test_confirmed_mutation_without_state_change_retries() -> None:
    result = ResultVerifier().verify(
        user_message="Add a new field and persist it",
        execution_result=_execution(result_preview={"ok": True}),
        state_changed=False,
        confirmation_status="approved",
    )

    assert result.severity == "retry"
    assert "mutates_state" in result.retry_instruction


def test_step_budget_final_answer_hard_fails() -> None:
    result = ResultVerifier().verify(
        user_message="What is this?",
        final_answer_draft="The agent reached its internal step budget.",
        retries_remaining=False,
    )

    assert result.hard_fail is True
    assert result.severity == "fail"


def test_ambiguous_clean_mutation_hard_fails() -> None:
    result = ResultVerifier().verify(
        user_message="Clean this dataset",
        execution_result=_execution(result_preview={"ok": True}),
        state_changed=True,
        retries_remaining=False,
    )

    assert result.hard_fail is True


def test_llm_verifier_disabled_uses_deterministic() -> None:
    settings = Settings(llm_verifier_enabled=False, verifier_mode="deterministic")
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm, trace = LLMVerifier(settings=settings).verify_if_allowed(
        user_message="What is this?",
        context={},
        execution_result=_execution(result_preview={"ok": True}),
        artifacts=[],
        state_changed=False,
        latest_code=None,
        deterministic_result=deterministic,
        current_step=1,
        turn_started_at=time.monotonic(),
    )

    assert llm is None
    assert trace is None


def test_llm_verifier_timeout_falls_back() -> None:
    settings = Settings(openai_api_key="test", llm_verifier_enabled=True, verifier_mode="hybrid")
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm, trace = LLMVerifier(settings=settings, client=_RaisingClient()).verify_if_allowed(
        user_message="What is this?",
        context={},
        execution_result=_execution(result_preview={"ok": True}),
        artifacts=[],
        state_changed=False,
        latest_code=None,
        deterministic_result=deterministic,
        current_step=1,
        turn_started_at=time.monotonic(),
    )

    assert llm is None
    assert trace == "Verifier completed using deterministic checks."


def test_llm_verifier_invalid_json_falls_back() -> None:
    settings = Settings(openai_api_key="test", llm_verifier_enabled=True, verifier_mode="hybrid")
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm, trace = LLMVerifier(settings=settings, client=_InvalidJsonClient()).verify_if_allowed(
        user_message="What is this?",
        context={},
        execution_result=_execution(result_preview={"ok": True}),
        artifacts=[],
        state_changed=False,
        latest_code=None,
        deterministic_result=deterministic,
        current_step=1,
        turn_started_at=time.monotonic(),
    )

    assert llm is None
    assert trace == "Verifier completed using deterministic checks."
    assert "JSONDecodeError" not in trace


def test_llm_verifier_debug_trace_can_include_exception_type() -> None:
    settings = Settings(
        openai_api_key="test",
        llm_verifier_enabled=True,
        verifier_mode="hybrid",
        show_verifier_debug_trace=True,
    )
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm, trace = LLMVerifier(settings=settings, client=_InvalidJsonClient()).verify_if_allowed(
        user_message="What is this?",
        context={},
        execution_result=_execution(result_preview={"ok": True}),
        artifacts=[],
        state_changed=False,
        latest_code=None,
        deterministic_result=deterministic,
        current_step=1,
        turn_started_at=time.monotonic(),
    )

    assert llm is None
    assert "JSONDecodeError" in trace


def test_llm_verifier_missing_fields_get_defaults() -> None:
    settings = Settings(openai_api_key="test", llm_verifier_enabled=True, verifier_mode="hybrid")
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm, trace = LLMVerifier(settings=settings, client=_PartialJsonClient()).verify_if_allowed(
        user_message="What is this?",
        context={},
        execution_result=_execution(result_preview={"ok": True}),
        artifacts=[],
        state_changed=False,
        latest_code=None,
        deterministic_result=deterministic,
        current_step=1,
        turn_started_at=time.monotonic(),
    )

    assert trace is None
    assert llm is not None
    assert llm.passed is True
    assert llm.confidence == 0.0
    assert llm.hallucination_risk == "medium"


def test_llm_verifier_schema_question_not_skipped_by_time_budget() -> None:
    settings = Settings(
        openai_api_key="test",
        llm_verifier_enabled=True,
        verifier_mode="hybrid",
        verifier_time_budget_per_turn_seconds=1,
    )
    deterministic = ResultVerifier().verify(
        user_message="Summarize the schema and list scalar/date/list-like fields",
        execution_result=_execution(result_preview={"scalar_fields": ["country"], "date_fields": ["filing_date"]}),
    )
    client = _PartialJsonClient()
    llm, trace = LLMVerifier(settings=settings, client=client).verify_if_allowed(
        user_message="Summarize the schema and list scalar/date/list-like fields",
        context={},
        execution_result=_execution(result_preview={"scalar_fields": ["country"], "date_fields": ["filing_date"]}),
        artifacts=[],
        state_changed=False,
        latest_code=None,
        deterministic_result=deterministic,
        current_step=1,
        turn_started_at=time.monotonic() - 5,
    )

    assert trace is None
    assert llm is not None
    assert client.calls == 1


def test_deterministic_hard_fail_wins_over_llm_pass() -> None:
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
        state_changed=True,
        retries_remaining=False,
    )
    merged = merge_verification_results(deterministic, _llm_pass())

    assert merged.hard_fail is True
    assert merged.source == "deterministic"


def test_missing_table_retry_wins_over_llm_pass() -> None:
    deterministic = ResultVerifier().verify(
        user_message="Show a table",
        execution_result=_execution(result_preview={"ok": True}),
    )
    merged = merge_verification_results(deterministic, _llm_pass())

    assert merged.severity == "retry"
    assert merged.source == "deterministic"


def test_high_confidence_llm_retry_can_override_deterministic_pass() -> None:
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm = LLMVerificationResult(
        passed=False,
        confidence=0.91,
        retry_instruction="Need one more schema check.",
        reasons=["Schema evidence is incomplete."],
    )
    merged = merge_verification_results(deterministic, llm)

    assert merged.severity == "retry"
    assert merged.source == "hybrid"


def test_low_confidence_llm_retry_falls_back_to_deterministic_pass() -> None:
    deterministic = ResultVerifier().verify(
        user_message="What is this?",
        execution_result=_execution(result_preview={"ok": True}),
    )
    llm = LLMVerificationResult(
        passed=False,
        confidence=0.4,
        retry_instruction="Maybe retry.",
        reasons=["Uncertain."],
    )
    merged = merge_verification_results(deterministic, llm)

    assert merged.passed is True
    assert merged.source == "deterministic"


def _execution(result_preview: object | None) -> ExecutionResult:
    return ExecutionResult(
        ok=True,
        stdout="",
        stderr="",
        traceback=None,
        result_preview=result_preview,
    )


def _artifact(
    kind: str,
    columns: list[dict[str, object]] | None = None,
    metadata: dict[str, object] | None = None,
) -> ExecutionArtifact:
    artifact_metadata = {"columns": columns or [], **(metadata or {})}
    return ExecutionArtifact(
        id=f"{kind}-1",
        name=f"{kind} artifact",
        kind=kind,
        type=kind,
        title=f"{kind} artifact",
        columns=columns or [],
        rows=[],
        path="/tmp/artifact",
        metadata=artifact_metadata,
    )


def _llm_pass() -> LLMVerificationResult:
    return LLMVerificationResult(
        passed=True,
        confidence=0.95,
        should_finalize=True,
        reasons=["Looks complete."],
    )


class _RaisingClient:
    @property
    def responses(self) -> "_RaisingClient":
        return self

    def create(self, **_: object) -> object:
        raise TimeoutError("timeout")


class _InvalidJsonClient:
    @property
    def responses(self) -> "_InvalidJsonClient":
        return self

    def create(self, **_: object) -> object:
        class Response:
            output_text = "not json"

        return Response()


class _PartialJsonClient:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def responses(self) -> "_PartialJsonClient":
        return self

    def create(self, **_: object) -> object:
        self.calls += 1

        class Response:
            output_text = "```json\n{\"passed\": true}\n```"

        return Response()
