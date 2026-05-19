from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.types import LLMVerificationResult, VerificationResult
from app.runtime.python_executor import ExecutionArtifact, ExecutionResult

TABLE_MARKERS = (
    "table",
    "tabular",
    "rows",
    "first 5 rows",
    "preview",
    "inferred columns",
    "columns",
    "top",
    "top 5",
    "top 10",
    "breakdown",
    "group by",
    "summarize by",
    "distribution by",
    "count per",
    "show records",
)
CHART_MARKERS = (
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualization",
    "bar chart",
    "line chart",
    "pie chart",
    "scatter",
    "distribution",
)
CSV_MARKERS = ("export", "csv", "download")
MUTATION_MARKERS = (
    "drop",
    "remove",
    "delete",
    "trim",
    "keep only",
    "exclude",
    "filter the working dataset",
    "persist",
    "mutate",
    "clean working dataset",
    "normalize and persist",
    "add a",
    "add column",
    "add a column",
    "add a new field",
    "derive a field",
    "rollback",
    "fork",
    "branch",
)
AMBIGUOUS_DANGEROUS_MARKERS = ("clean this dataset", "remove bad records", "fix the data")
INSPECTION_MARKERS = (
    "what is in",
    "what's in",
    "what is this",
    "summarize",
    "summary",
    "schema",
    "sample",
    "inspect",
    "describe",
    "how many",
    "count",
)
WRAPPER_COLUMNS = {
    "__dict__",
    "__pydantic_extra__",
    "__pydantic_fields_set__",
    "__pydantic_private__",
}


class ResultVerifier:
    """Fast deterministic verifier for the general coding-agent workflow."""

    def verify(
        self,
        *,
        user_message: str,
        execution_result: ExecutionResult | None = None,
        artifacts_created_this_turn: Sequence[ExecutionArtifact] | None = None,
        all_artifacts_for_message: Sequence[ExecutionArtifact] | None = None,
        current_step: int = 0,
        max_steps: int = 6,
        retries_remaining: bool = True,
        state_changed: bool = False,
        confirmation_status: str | None = None,
        latest_code: str | None = None,
        final_answer_draft: str | None = None,
    ) -> VerificationResult:
        message = user_message.lower()
        artifacts = list(artifacts_created_this_turn or all_artifacts_for_message or [])
        latest_artifacts = list(artifacts_created_this_turn or [])
        reasons: list[str] = []

        if _contains_step_budget_failure(final_answer_draft):
            return _fail(
                "Final answer exposed an internal step-budget failure.",
                retry_instruction="Use the latest useful execution result or artifact to compose a concise answer.",
                hard_fail=True,
                retry=retries_remaining,
            )

        if _contains_wrapper_leak(execution_result, artifacts, final_answer_draft) and not _allows_internal_metadata(message):
            return _retry(
                "Internal wrapper fields leaked into the result.",
                (
                    "The result exposes wrapper columns or fields such as __dict__ or __pydantic metadata. "
                    "Use safe_attrs/object_to_record/to_dataframe to flatten domain fields before answering."
                ),
            )

        invalid_helper_reason = _invalid_helper_usage_reason(latest_code, execution_result)
        if invalid_helper_reason:
            if "artifact" in invalid_helper_reason.lower():
                instruction = "Do not use `artifacts`; use `artifact_history` if injected, or recompute from data."
            elif "result" in invalid_helper_reason.lower():
                instruction = (
                    "RESULT is not preserved between executions. Recompute the data and create the "
                    "requested artifact in the same code block, or read prior outputs from artifact_history."
                )
            elif "chart data" in invalid_helper_reason.lower():
                instruction = (
                    "Put a non-empty list of dict rows inside chart_spec['data'], or pass data=rows "
                    "to save_chart in the same execute_python block."
                )
            elif "chart/table variables" in invalid_helper_reason.lower():
                instruction = (
                    "Variables from prior execute_python calls are not available. Recompute chart data "
                    "and call save_chart in the same code block."
                )
            else:
                instruction = (
                    "Do not import helpers. Runtime helpers are injected directly. Do not rely on "
                    "variables from previous execute_python calls."
                )
            return _retry(
                invalid_helper_reason,
                instruction,
                hard_fail=False,
            )

        if execution_result and not execution_result.ok:
            error_text = execution_result.stderr or execution_result.traceback or "Python execution failed."
            if retries_remaining:
                return _retry(
                    "Python execution failed.",
                    f"Fix this traceback and retry with a generic approach:\n{_compact(error_text, 900)}",
                )
            return _fail("Python execution failed and no retries remain.", hard_fail=False, retry=False)

        if asks_for_mutation(message) and confirmation_status != "approved" and execution_result and execution_result.ok and not state_changed:
            if not _full_scan_found_zero_matches(execution_result):
                return _retry(
                    "User requested a dataset mutation, but no confirmation or state change occurred.",
                    (
                        "This is a risky write request. Scan the full dataset if needed, then call "
                        "request_confirmation with a clear operation summary and proposed mutation code."
                    ),
                )

        if asks_for_table(message) and not asks_for_csv(message) and not _has_artifact(artifacts, "table"):
            return _retry(
                "User requested a table/preview, but no table artifact was created.",
                "User requested a table/preview. Create a table artifact with save_table(...). Do not only summarize in prose.",
            )

        if asks_for_chart(message) and not _has_artifact(artifacts, "chart"):
            return _retry(
                "User requested a chart, but no chart artifact was created.",
                "User requested a chart. Create a chart artifact with save_chart(...).",
            )

        invalid_chart = _invalid_chart_artifact(artifacts)
        if invalid_chart:
            return _retry(
                f"Chart artifact is invalid: {invalid_chart}.",
                "Create a valid chart artifact with chart_spec.data as a non-empty list of objects and matching x/y keys.",
            )

        intent_issue = _intent_artifact_mismatch(message, execution_result, artifacts)
        if intent_issue:
            return _retry(intent_issue, _intent_retry_instruction(message))

        if asks_for_csv(message) and not _has_artifact(artifacts, "csv"):
            return _retry(
                "User requested CSV/export, but no CSV artifact exists.",
                "User requested CSV/export. Create a CSV artifact with save_csv(...).",
            )

        wrapper_columns = _wrapper_columns(artifacts)
        if wrapper_columns:
            return _retry(
                f"Table exposes wrapper columns: {', '.join(sorted(wrapper_columns))}.",
                (
                    "Table exposes wrapper columns. Use object_to_record/to_dataframe to flatten to "
                    "domain fields such as country, title, owners, assignees, status."
                ),
            )

        if _schema_request_quality_failed(message, execution_result, final_answer_draft, artifacts):
            return _retry(
                "Schema result does not describe real domain fields.",
                (
                    "For schema requests, use to_dataframe(data, limit=None) or objects_to_records(data, limit=None), "
                    "then group real fields such as country, title, owners, assignees, filing_date, and status into "
                    "scalar/date/list-like categories."
                ),
            )

        if not state_changed and _inspection_size_missing(message, execution_result, final_answer_draft):
            return _retry(
                "Inspection request asked for size, but no object length or row count was returned.",
                "Return object_length/length using len(data) when available, along with representative fields.",
            )

        sample_issue = _sample_only_analysis_issue(message, execution_result, artifacts, latest_code)
        if sample_issue:
            return _retry(
                sample_issue,
                "Use to_dataframe(data, limit=None) or objects_to_records(data, limit=None) for full dataset analysis.",
            )

        if _has_stringified_structured_fields(artifacts, execution_result):
            return _retry(
                "Structured list/date fields were stringified.",
                "Preserve list fields as lists and convert datetime/date fields to ISO strings.",
            )

        if not asks_for_mutation(message) and state_changed:
            return _fail(
                "State changed on a non-mutating request.",
                retry_instruction="Redo the request without mutating datasets or setting mutates_state=true.",
                hard_fail=True,
                retry=retries_remaining,
            )

        if _ambiguous_dangerous(message) and state_changed:
            return _fail(
                "Ambiguous dangerous prompt mutated state without clarification.",
                retry_instruction="Ask a concise clarification question before changing data.",
                hard_fail=True,
                retry=retries_remaining,
            )

        if (
            confirmation_status == "approved"
            and asks_for_mutation(message)
            and execution_result
            and execution_result.ok
            and not state_changed
        ):
            return _retry(
                "Mutation was requested and confirmed, but no state change was saved.",
                "The user requested a persisted mutation. Update the intended dataset and set mutates_state=true.",
            )

        if latest_artifacts:
            reasons.append("Required artifact output is available.")
            return VerificationResult(
                passed=True,
                severity="pass",
                reasons=reasons,
                should_finalize=True,
                source="deterministic",
            )

        if execution_result and execution_result.ok and _has_useful_preview(execution_result):
            if _is_inspection_like(message) or not (asks_for_table(message) or asks_for_chart(message) or asks_for_csv(message)):
                reasons.append("Useful result available; final answer should be prepared.")
                return VerificationResult(
                    passed=True,
                    severity="pass",
                    reasons=reasons,
                    should_finalize=True,
                    source="deterministic",
                )

        if current_step >= max_steps - 1 and (latest_artifacts or (execution_result and _has_useful_preview(execution_result))):
            return VerificationResult(
                passed=True,
                severity="finalize_with_warning",
                reasons=["Near step budget; finalize with best verified result."],
                should_finalize=True,
                source="deterministic",
            )

        return VerificationResult(
            passed=True,
            severity="pass",
            reasons=["No deterministic blockers found."],
            should_finalize=False,
            source="deterministic",
        )


def merge_verification_results(
    deterministic: VerificationResult,
    llm: LLMVerificationResult | None,
    *,
    min_confidence: float = 0.70,
) -> VerificationResult:
    if deterministic.hard_fail:
        return deterministic

    if deterministic.severity == "retry" and _is_hard_requirement_retry(deterministic):
        return deterministic

    if llm is None or llm.confidence < min_confidence:
        return deterministic

    if deterministic.passed and llm.passed:
        return VerificationResult(
            passed=True,
            severity=deterministic.severity,
            reasons=[*deterministic.reasons, *llm.reasons],
            should_finalize=deterministic.should_finalize or llm.should_finalize,
            confidence=llm.confidence,
            source="hybrid",
            metadata={"llm": llm.model_dump(mode="json")},
        )

    if deterministic.passed and not llm.passed:
        return VerificationResult(
            passed=False,
            severity="retry",
            reasons=[*deterministic.reasons, *llm.reasons, *llm.missing_requirements],
            retry_instruction=llm.retry_instruction or "The verifier found the response incomplete; retry with stronger evidence.",
            should_finalize=False,
            confidence=llm.confidence,
            source="hybrid",
            metadata={"llm": llm.model_dump(mode="json")},
        )

    if deterministic.severity == "finalize_with_warning" and llm.passed:
        return deterministic.model_copy(
            update={
                "source": "hybrid",
                "confidence": llm.confidence,
                "metadata": {"llm": llm.model_dump(mode="json")},
            }
        )

    return deterministic


def asks_for_table(message: str) -> bool:
    return any(marker in message for marker in TABLE_MARKERS)


def asks_for_chart(message: str) -> bool:
    return any(marker in message for marker in CHART_MARKERS)


def asks_for_csv(message: str) -> bool:
    return any(marker in message for marker in CSV_MARKERS)


def asks_for_mutation(message: str) -> bool:
    return any(marker in message for marker in MUTATION_MARKERS)


def verifier_trace_message(result: VerificationResult) -> str:
    if result.severity == "retry":
        return f"Verifier requested retry: {result.reasons[0] if result.reasons else 'more evidence needed'}"
    if result.severity == "fail":
        return f"Verifier failed: {result.reasons[0] if result.reasons else 'hard rule failed'}"
    if result.severity == "finalize_with_warning":
        return "Verifier passed with a warning; composing the best answer now."
    return "Deterministic verifier passed."


def _retry(reason: str, instruction: str, *, hard_fail: bool = False) -> VerificationResult:
    return VerificationResult(
        passed=False,
        severity="retry",
        reasons=[reason],
        retry_instruction=instruction,
        should_finalize=False,
        source="deterministic",
        hard_fail=hard_fail,
    )


def _fail(reason: str, *, retry_instruction: str | None = None, hard_fail: bool, retry: bool) -> VerificationResult:
    return VerificationResult(
        passed=False,
        severity="retry" if retry else "fail",
        reasons=[reason],
        retry_instruction=retry_instruction,
        should_finalize=False,
        source="deterministic",
        hard_fail=hard_fail,
    )


def _has_artifact(artifacts: Sequence[ExecutionArtifact], kind: str) -> bool:
    return any(artifact.kind == kind or artifact.type == kind for artifact in artifacts)


def _wrapper_columns(artifacts: Sequence[ExecutionArtifact]) -> set[str]:
    found: set[str] = set()
    for artifact in artifacts:
        if artifact.kind != "table":
            continue
        for column in artifact.columns or artifact.metadata.get("columns", []):
            key = None
            if isinstance(column, Mapping):
                key = column.get("key") or column.get("label")
            else:
                key = column
            if str(key) in WRAPPER_COLUMNS:
                found.add(str(key))
    return found


def _has_stringified_structured_fields(
    artifacts: Sequence[ExecutionArtifact],
    result: ExecutionResult | None,
) -> bool:
    values: list[Any] = []
    for artifact in artifacts:
        if artifact.kind == "table":
            values.extend(_iter_values(artifact.rows or artifact.metadata.get("rows", [])))
    if result and result.result_preview is not None:
        values.extend(_iter_values(result.result_preview))

    for value in values:
        if not isinstance(value, str):
            continue
        if re.search(r"^\[[\"']?.+[, ].*\]$", value) or "datetime.datetime(" in value:
            return True
    return False


def _contains_wrapper_leak(
    result: ExecutionResult | None,
    artifacts: Sequence[ExecutionArtifact],
    final_answer: str | None,
) -> bool:
    if final_answer and any(marker in final_answer for marker in WRAPPER_COLUMNS | {"'attrs':", '"attrs"'}):
        return True
    if result and result.result_preview is not None and _contains_forbidden_key_or_value(result.result_preview):
        return True
    for artifact in artifacts:
        if _contains_forbidden_key_or_value(artifact.columns) or _contains_forbidden_key_or_value(artifact.rows):
            return True
        if _contains_forbidden_key_or_value(artifact.metadata):
            return True
    return False


def _contains_forbidden_key_or_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in WRAPPER_COLUMNS or str(key) == "attrs":
                return True
            if _contains_forbidden_key_or_value(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_key_or_value(item) for item in value[:50])
    if isinstance(value, str):
        return any(marker in value for marker in WRAPPER_COLUMNS) or "'attrs':" in value or '"attrs"' in value
    return False


def _allows_internal_metadata(message: str) -> bool:
    return any(marker in message for marker in ("internal", "wrapper", "pydantic", "__dict__", "raw object metadata"))


def _schema_request_quality_failed(
    message: str,
    result: ExecutionResult | None,
    final_answer: str | None,
    artifacts: Sequence[ExecutionArtifact] | None = None,
) -> bool:
    if not _asks_for_schema_groups(message):
        return False
    if artifacts and _schema_artifact_has_domain_groups(artifacts):
        return False
    if result and result.result_preview is not None:
        fields = _schema_domain_fields(result.result_preview)
        if fields and _has_domain_schema_fields(fields):
            return False
        return True
    if final_answer:
        lowered = final_answer.lower()
        groups = sum(1 for marker in ("scalar", "date", "list") if marker in lowered)
        if groups < 2:
            return True
        if any(marker in lowered for marker in ("__pydantic", "__dict__", "attrs")):
            return True
    return False


def _schema_artifact_has_domain_groups(artifacts: Sequence[ExecutionArtifact]) -> bool:
    for artifact in artifacts:
        if artifact.kind != "table":
            continue
        rows = artifact.rows or artifact.metadata.get("rows") or []
        columns = _table_column_keys(artifact)
        if not isinstance(rows, list) or not rows:
            continue
        if "field" not in columns or not ({"category", "type", "field_type"} & columns):
            continue
        fields: set[str] = set()
        groups: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            field = row.get("field")
            category = row.get("category") or row.get("type") or row.get("field_type")
            if field is not None:
                fields.add(str(field))
            if category is not None:
                groups.add(str(category).lower())
        if _has_domain_schema_fields(fields) and sum(marker in " ".join(groups) for marker in ("scalar", "date", "list")) >= 2:
            return True
    return False


def _asks_for_schema_groups(message: str) -> bool:
    return "schema" in message or "scalar" in message or "date fields" in message or "list-like" in message


def _inspection_size_missing(
    message: str,
    result: ExecutionResult | None,
    final_answer: str | None,
) -> bool:
    if not any(marker in message for marker in ("what is in this file", "what's in this file", "size", "structure")):
        return False
    if final_answer and re.search(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b", final_answer):
        return False
    preview = result.result_preview if result else None
    if isinstance(preview, Mapping):
        for key in (
            "length",
            "object_length",
            "records",
            "row_count",
            "source_total_row_count",
            "source_row_count",
        ):
            value = preview.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return False
        shape = preview.get("shape")
        if isinstance(shape, Sequence) and shape:
            return False
    return result is not None and result.ok


def _schema_domain_fields(value: Any) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).endswith("_fields") and isinstance(item, list):
                fields.update(str(field) for field in item)
            elif isinstance(item, Mapping | list):  # type: ignore[operator]
                fields.update(_schema_domain_fields(item))
    elif isinstance(value, list):
        for item in value[:20]:
            fields.update(_schema_domain_fields(item))
    return {field for field in fields if field not in WRAPPER_COLUMNS}


def _has_domain_schema_fields(fields: set[str]) -> bool:
    domain_markers = {"country", "title", "owners", "assignees", "filing_date", "status", "doc_number"}
    return len(fields & domain_markers) >= 2


def _sample_only_analysis_issue(
    message: str,
    result: ExecutionResult | None,
    artifacts: Sequence[ExecutionArtifact],
    latest_code: str | None,
) -> str | None:
    if not _asks_for_full_dataset_analysis(message):
        return None
    code = latest_code or ""
    if re.search(r"(to_dataframe|objects_to_records)\(\s*data\s*,\s*limit\s*=\s*(?:20|50|100|500)\b", code):
        return "Full-dataset analysis used a row limit."
    for artifact in artifacts:
        analyzed = _numeric_meta(artifact.metadata, "analyzed_row_count")
        source = _numeric_meta(artifact.metadata, "source_total_row_count") or _numeric_meta(artifact.metadata, "source_row_count")
        if source is not None and analyzed is not None and analyzed < source:
            return "Artifact was created from a sampled subset instead of the full dataset."
    if result and isinstance(result.result_preview, Mapping):
        analyzed = _numeric_meta(result.result_preview, "analyzed_row_count")
        source = (
            _numeric_meta(result.result_preview, "source_total_row_count")
            or _numeric_meta(result.result_preview, "source_row_count")
            or _numeric_meta(result.result_preview, "total_row_count")
        )
        if source is not None and analyzed is not None and analyzed < source:
            return "Result was computed from a sampled subset instead of the full dataset."
    return None


def _asks_for_full_dataset_analysis(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "top countries",
            "count",
            "distribution",
            "breakdown",
            "date range",
            "filings by year",
            "number of patent records",
        )
    ) and "sample" not in message


def _numeric_meta(value: Mapping[str, Any], key: str) -> int | None:
    item = value.get(key)
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return int(item)
    return None


def _invalid_chart_artifact(artifacts: Sequence[ExecutionArtifact]) -> str | None:
    for artifact in artifacts:
        if artifact.kind != "chart":
            continue
        spec = artifact.chart_spec or artifact.metadata.get("chart_spec")
        if not isinstance(spec, Mapping):
            return "missing chart_spec"
        chart_type = spec.get("chart_type")
        if chart_type not in {"bar", "line", "pie", "scatter", "area"}:
            return f"unsupported chart_type {chart_type!r}"
        data = spec.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
            return "chart_spec.data is empty or not rows"
        x = spec.get("x")
        y = spec.get("y")
        if not isinstance(x, str) or x not in data[0]:
            return "x field missing from chart data"
        if not isinstance(y, str) or y not in data[0]:
            return "y field missing from chart data"
        if chart_type in {"bar", "line", "area", "scatter", "pie"}:
            sample_y = next((row.get(y) for row in data if isinstance(row, Mapping) and row.get(y) is not None), None)
            if sample_y is not None and not isinstance(sample_y, (int, float)) and chart_type != "pie":
                return "y field should be numeric"
    return None


def _intent_artifact_mismatch(
    message: str,
    result: ExecutionResult | None,
    artifacts: Sequence[ExecutionArtifact],
) -> str | None:
    if _asks_for_tabular_preview(message):
        table = _first_artifact(artifacts, "table")
        if table and _is_country_count_table(table):
            return "Tabular preview request was answered with an aggregate country-count table."

    if _asks_for_filing_date_range(message):
        table = _best_table_artifact(artifacts, _table_has_year_count_fields)
        has_range = _result_has_date_range(result) or _artifact_metadata_has_date_range(artifacts)
        if not table and _first_artifact(artifacts, "table"):
            return "Filing-date request was answered with a table that is not filings by year."
        if not has_range and "date range" in message:
            return "Filing-date range request did not return min/max filing dates."

    if _asks_for_chart_domain(message):
        chart = _first_artifact(artifacts, "chart")
        if chart:
            chart_issue = _chart_intent_issue(message, chart)
            if chart_issue:
                return chart_issue

    return None


def _intent_retry_instruction(message: str) -> str:
    if _asks_for_tabular_preview(message):
        return (
            "The user asked for a tabular row preview. Use df = to_dataframe(data, limit=None), "
            "then save_table('Tabular preview', df.head(5), description='First 5 rows with inferred columns')."
        )
    if _asks_for_filing_date_range(message):
        return (
            "Use the full dataset to parse filing_date, compute filing_date_min and filing_date_max, "
            "then save_table with filing_year and filing_count/count columns."
        )
    if "line chart" in message:
        return (
            "Create a line chart artifact matching the requested domain. For filings by year, use "
            "chart_type='line', x='filing_year', y='filing_count', and non-empty yearly data."
        )
    if "country" in message:
        return "Create a country-count artifact with country and record_count/count fields."
    return "Create artifacts whose columns, chart fields, title, and result preview match the user request."


def _asks_for_tabular_preview(message: str) -> bool:
    return any(marker in message for marker in ("tabular preview", "first 5 rows", "inferred columns"))


def _asks_for_filing_date_range(message: str) -> bool:
    return "filing date range" in message or "filings by year" in message


def _asks_for_chart_domain(message: str) -> bool:
    return asks_for_chart(message) and any(marker in message for marker in ("line chart", "bar chart", "filing", "year", "country"))


def _first_artifact(artifacts: Sequence[ExecutionArtifact], kind: str) -> ExecutionArtifact | None:
    return next((artifact for artifact in artifacts if artifact.kind == kind or artifact.type == kind), None)


def _best_table_artifact(
    artifacts: Sequence[ExecutionArtifact],
    predicate: Any,
) -> ExecutionArtifact | None:
    return next(
        (
            artifact
            for artifact in artifacts
            if (artifact.kind == "table" or artifact.type == "table") and predicate(artifact)
        ),
        None,
    )


def _artifact_title(artifact: ExecutionArtifact) -> str:
    return str(artifact.title or artifact.name or artifact.metadata.get("title") or "").lower()


def _table_column_keys(artifact: ExecutionArtifact) -> set[str]:
    keys: set[str] = set()
    for column in artifact.columns or artifact.metadata.get("columns", []):
        if isinstance(column, Mapping):
            key = column.get("key") or column.get("label")
        else:
            key = column
        if key is not None:
            keys.add(str(key).lower())
    if not keys:
        for row in artifact.rows[:5]:
            if isinstance(row, Mapping):
                keys.update(str(key).lower() for key in row.keys())
    return keys


def _is_count_field(key: str) -> bool:
    lowered = key.lower()
    return lowered in {"count", "record_count", "patent_count", "filing_count"} or lowered.endswith("_count")


def _is_country_count_table(artifact: ExecutionArtifact) -> bool:
    keys = _table_column_keys(artifact)
    return "country" in keys and any(_is_count_field(key) for key in keys) and len(keys) <= 3


def _table_has_preview_shape(artifact: ExecutionArtifact) -> bool:
    keys = _table_column_keys(artifact)
    if len(keys) >= 4:
        return True
    representative = {"country", "doc_number", "title", "owners", "assignees", "filing_date", "status"}
    return len(keys & representative) >= 3


def _table_has_year_count_fields(artifact: ExecutionArtifact) -> bool:
    keys = _table_column_keys(artifact)
    has_year = any(key in keys for key in ("filing_year", "year"))
    has_count = any(_is_count_field(key) for key in keys)
    row_count = _numeric_meta(artifact.metadata, "row_count")
    stored_rows = artifact.rows or artifact.metadata.get("rows") or []
    has_rows = (row_count is None or row_count > 0) and (bool(stored_rows) or row_count is not None)
    return has_year and has_count and has_rows


def _result_has_date_range(result: ExecutionResult | None) -> bool:
    if result is None or not isinstance(result.result_preview, Mapping):
        return False
    keys = {str(key).lower() for key in result.result_preview.keys()}
    return (
        {"filing_date_min", "filing_date_max"} <= keys
        or {"min_filing_date", "max_filing_date"} <= keys
        or {"date_min", "date_max"} <= keys
    )


def _artifact_metadata_has_date_range(artifacts: Sequence[ExecutionArtifact]) -> bool:
    for artifact in artifacts:
        metadata = artifact.metadata if isinstance(artifact.metadata, Mapping) else {}
        keys = {str(key).lower() for key in metadata.keys()}
        if {"filing_date_min", "filing_date_max"} <= keys or {"min_filing_date", "max_filing_date"} <= keys:
            return True
    return False


def _chart_intent_issue(message: str, artifact: ExecutionArtifact) -> str | None:
    title = _artifact_title(artifact)
    if title == "fake agent chart":
        return "Chart artifact title is still the demo placeholder 'Fake agent chart'."

    spec = artifact.chart_spec or artifact.metadata.get("chart_spec")
    if not isinstance(spec, Mapping):
        return "Chart artifact is missing chart_spec."

    chart_type = str(spec.get("chart_type") or "").lower()
    x = str(spec.get("x") or "").lower()
    y = str(spec.get("y") or "").lower()
    data = spec.get("data")
    first_row = data[0] if isinstance(data, list) and data and isinstance(data[0], Mapping) else {}
    row_keys = {str(key).lower() for key in first_row.keys()} if isinstance(first_row, Mapping) else set()

    if "line chart" in message and chart_type != "line":
        return "User requested a line chart, but the chart artifact is not a line chart."
    if "line chart or bar chart" in message and any(marker in message for marker in ("filing", "year", "date")) and chart_type != "line":
        return "Time-series chart request should prefer a line chart."
    if "bar chart" in message and "line chart" not in message and chart_type != "bar":
        return "User requested a bar chart, but the chart artifact is not a bar chart."
    if "filing" in message or "year" in message:
        if x not in {"filing_year", "year"} and not ({"filing_year", "year"} & row_keys):
            return "Filings-by-year chart does not use a filing_year/year x field."
        if y not in {"filing_count", "count", "record_count"} and not any(_is_count_field(key) for key in row_keys):
            return "Filings-by-year chart does not use a filing_count/count y field."
        if "country" in row_keys and "filing_year" not in row_keys and "year" not in row_keys:
            return "Filings-by-year chart appears to contain country counts instead of yearly filings."
    if "country" in message:
        if x != "country" and "country" not in row_keys:
            return "Country chart does not use country as a data field."
    return None


def _full_scan_found_zero_matches(result: ExecutionResult) -> bool:
    preview = result.result_preview
    if not isinstance(preview, Mapping):
        return False
    affected = None
    for key in ("affected_count", "empty_title_count", "deleted_count"):
        if key in preview:
            affected = preview.get(key)
            break
    full_scan = preview.get("full_scan") or preview.get("scanned_full_dataset")
    return affected == 0 and bool(full_scan)


def _iter_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        values: list[Any] = []
        for item in value.values():
            values.extend(_iter_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value[:20]:
            values.extend(_iter_values(item))
        return values
    return [value]


def _has_useful_preview(result: ExecutionResult) -> bool:
    preview = result.result_preview
    if preview is None:
        return bool(result.stdout.strip() or result.artifacts)
    if preview in ({}, [], ""):
        return False
    return True


def _is_inspection_like(message: str) -> bool:
    return any(marker in message for marker in INSPECTION_MARKERS) or asks_for_table(message)


def _contains_step_budget_failure(answer: str | None) -> bool:
    if not answer:
        return False
    lowered = answer.lower()
    return any(marker in lowered for marker in ("reached its internal step budget", "could not complete", "latest python error"))


def _invalid_helper_usage_reason(code: str | None, result: ExecutionResult | None) -> str | None:
    text = "\n".join(part for part in (code, result.traceback if result else None, result.stderr if result else None) if part)
    markers = (
        "No module named 'runtime'",
        'No module named "runtime"',
        "No module named 'helpers'",
        'No module named "helpers"',
        "globals is not defined",
        "NameError: name 'columns' is not defined",
        "NameError: name 'RESULT' is not defined",
        "NameError: name 'table' is not defined",
        "NameError: name 'rows' is not defined",
        "NameError: name 'artifacts' is not defined",
        "chart_spec.data must be a non-empty list of dict rows",
        "Chart spec must include data as a list of objects",
        "positional argument follows keyword argument",
        "save_chart got unsupported keyword argument",
        "save_table got unsupported keyword argument",
        "save_chart() got an unexpected keyword argument",
        "save_table() got an unexpected keyword argument",
    )
    if "NameError: name 'artifacts' is not defined" in text:
        return "Undefined artifact variable used."
    if "NameError: name 'RESULT' is not defined" in text:
        return "Isolated execution tried to reuse RESULT from a previous Python call."
    if "NameError: name 'table' is not defined" in text or "NameError: name 'rows' is not defined" in text:
        return "Isolated execution tried to reuse chart/table variables from a previous Python call."
    if "chart_spec.data must be a non-empty list of dict rows" in text or "Chart spec must include data as a list of objects" in text:
        return "Chart data was not supplied as list-of-dict rows."
    if "positional argument follows keyword argument" in text:
        return "Invalid helper call syntax."
    if any(
        marker in text
        for marker in (
            "save_chart got unsupported keyword argument",
            "save_table got unsupported keyword argument",
            "save_chart() got an unexpected keyword argument",
            "save_table() got an unexpected keyword argument",
        )
    ):
        return "Invalid artifact helper call signature."
    return "Invalid helper import or isolated-execution variable reuse detected." if any(marker in text for marker in markers) else None


def _is_hard_requirement_retry(result: VerificationResult) -> bool:
    text = " ".join([*result.reasons, result.retry_instruction or ""]).lower()
    return any(marker in text for marker in ("table artifact", "chart artifact", "csv artifact", "wrapper columns", "state changed"))


def _ambiguous_dangerous(message: str) -> bool:
    return any(marker in message for marker in AMBIGUOUS_DANGEROUS_MARKERS)


def _compact(value: str, limit: int) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit].rstrip()}\n... truncated ..."
