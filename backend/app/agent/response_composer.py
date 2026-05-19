from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from typing import Any

from app.agent.types import ComposedAnswer, VerificationResult
from app.runtime.python_executor import ExecutionArtifact, ExecutionResult


class ResponseComposer:
    def compose(
        self,
        *,
        user_message: str,
        execution_result: ExecutionResult | None,
        artifacts: list[ExecutionArtifact],
        verification: VerificationResult,
        state_changed: bool,
        mutation_summary: str | None = None,
    ) -> ComposedAnswer:
        warnings = _warnings_from_verification(verification)
        key_findings = _key_findings(execution_result, artifacts)
        highlights = _highlights(execution_result, artifacts, state_changed)
        artifact_ids = [artifact.id for artifact in artifacts]
        markdown = _markdown_answer(
            user_message=user_message,
            execution_result=execution_result,
            artifacts=artifacts,
            key_findings=key_findings,
            warnings=warnings,
            state_changed=state_changed,
            mutation_summary=mutation_summary,
        )
        return ComposedAnswer(
            markdown=markdown,
            highlights=highlights,
            key_findings=key_findings,
            warnings=warnings,
            state_changed=state_changed,
            artifact_ids=artifact_ids,
        )

    def compose_failure(
        self,
        *,
        execution_result: ExecutionResult | None,
        verification: VerificationResult,
        state_changed: bool,
    ) -> ComposedAnswer:
        latest_error = ""
        if execution_result is not None:
            latest_error = execution_result.stderr or execution_result.traceback or ""
        reason = verification.reasons[0] if verification.reasons else "The request could not be verified."
        markdown = (
            "## Unable to complete\n"
            f"I could not complete the request. {reason}\n\n"
            f"{_error_block(latest_error)}"
            f"\n\n**State changed:** {'Yes' if state_changed else 'No'}"
        )
        return ComposedAnswer(
            markdown=markdown,
            warnings=[reason],
            state_changed=state_changed,
        )


def _markdown_answer(
    *,
    user_message: str,
    execution_result: ExecutionResult | None,
    artifacts: list[ExecutionArtifact],
    key_findings: list[str],
    warnings: list[str],
    state_changed: bool,
    mutation_summary: str | None,
) -> str:
    title = _answer_title(user_message, artifacts)
    lines = [f"## {title}"]

    artifact_sentence = _artifact_sentence(artifacts)
    if artifact_sentence:
        lines.append(artifact_sentence)
    elif execution_result and execution_result.result_preview is not None:
        lines.append(_summary_from_preview(execution_result.result_preview))
    elif execution_result and execution_result.stdout.strip():
        lines.append(_summary_from_stdout(execution_result.stdout))
    else:
        lines.append("I completed the analysis with the available session context.")

    if key_findings:
        lines.extend(["", "### Key findings"])
        lines.extend(f"- {finding}" for finding in key_findings[:5])

    if mutation_summary:
        label = "Mutation" if state_changed else "Analysis performed"
        lines.extend(["", f"{label}: {mutation_summary}"])

    if warnings:
        lines.extend(["", "### Notes"])
        lines.extend(f"- {warning}" for warning in warnings[:3])

    lines.extend(["", f"**State changed:** {'Yes' if state_changed else 'No'}"])
    return "\n".join(lines)


def _answer_title(user_message: str, artifacts: list[ExecutionArtifact]) -> str:
    lowered = user_message.lower()
    if any(artifact.kind == "chart" for artifact in artifacts):
        return "Visualization"
    if any(artifact.kind == "table" for artifact in artifacts):
        return "Tabular result"
    if any(artifact.kind == "csv" for artifact in artifacts):
        return "CSV export"
    if "schema" in lowered:
        return "Schema summary"
    return "Summary"


def _artifact_sentence(artifacts: list[ExecutionArtifact]) -> str:
    kinds = {artifact.kind for artifact in artifacts}
    parts: list[str] = []
    if "table" in kinds:
        parts.append("I created the table shown in the chat below")
    if "chart" in kinds:
        parts.append("I created the chart shown in the chat below")
    if "csv" in kinds:
        parts.append("I created a CSV download")
    if not parts:
        return ""
    sentence = "; ".join(parts) + "."
    return sentence


def _key_findings(
    execution_result: ExecutionResult | None,
    artifacts: list[ExecutionArtifact],
) -> list[str]:
    findings: list[str] = []
    for artifact in artifacts:
        if artifact.kind == "table":
            row_count = artifact.metadata.get("row_count") or artifact.metadata.get("rows")
            column_count = len(artifact.columns)
            if row_count is not None:
                findings.append(f"Table `{artifact.title or artifact.name}` contains {row_count} row{'s' if str(row_count) != '1' else ''}.")
            if column_count:
                sample = ", ".join(str(column.get("key")) for column in artifact.columns[:8])
                findings.append(f"Inferred columns include {sample}.")
            row_names = _table_row_names(artifact)
            if row_names:
                findings.append(f"Rows include {', '.join(row_names[:8])}.")
        elif artifact.kind == "chart":
            chart_type = artifact.metadata.get("chart_type")
            if chart_type is None and artifact.chart_spec:
                chart_type = artifact.chart_spec.get("chart_type")
            findings.append(f"Chart `{artifact.title or artifact.name}` is a {chart_type or 'chart'} artifact.")
        elif artifact.kind == "csv":
            row_count = artifact.metadata.get("row_count") or artifact.metadata.get("rows")
            findings.append(f"CSV `{artifact.title or artifact.name}` is ready to download{f' with {row_count} rows' if row_count is not None else ''}.")

    preview = execution_result.result_preview if execution_result else None
    if preview is not None:
        findings.extend(_date_range_findings(preview))
    if preview is not None and len(findings) < 3:
        findings.extend(_findings_from_preview(preview))
    return _dedupe(findings)[:5]


def _table_row_names(artifact: ExecutionArtifact) -> list[str]:
    rows = artifact.rows or artifact.metadata.get("rows") or []
    if not isinstance(rows, list):
        return []
    candidates: list[str] = []
    for row in rows[:8]:
        if not isinstance(row, Mapping):
            continue
        for key in ("name", "path", "dataset_name", "element_index", "country", "category", "sensor_id"):
            value = row.get(key)
            if value is not None:
                candidates.append(str(value))
                break
    return _dedupe(candidates)


def _findings_from_preview(preview: Any) -> list[str]:
    if isinstance(preview, Mapping):
        date_findings = _date_range_findings(preview)
        if date_findings:
            return date_findings
        if preview.get("type") == "dataframe":
            shape = preview.get("shape")
            columns = preview.get("columns")
            findings = []
            if isinstance(shape, list) and len(shape) >= 2:
                findings.append(f"The tabular preview reports {shape[0]} rows and {shape[1]} columns.")
            if isinstance(columns, list) and columns:
                findings.append(f"Columns include {', '.join(str(column) for column in columns[:8])}.")
            return findings
        structural = _structural_findings(preview)
        if structural:
            return structural
        schema_findings = _schema_findings(preview)
        if schema_findings:
            return schema_findings
        simple = []
        for key, value in list(preview.items())[:5]:
            if str(key).startswith("__") or str(key) == "attrs":
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                simple.append(f"{key}: {_format_scalar(value)}")
            elif isinstance(value, list):
                if str(key).lower() == "shape" and all(isinstance(item, (int, float)) for item in value):
                    simple.append(f"{key}: {' x '.join(str(int(item)) for item in value)}")
                else:
                    simple.append(f"{key}: {len(value)} item{'s' if len(value) != 1 else ''}")
        return simple
    if isinstance(preview, list):
        return [f"Preview contains {len(preview)} item{'s' if len(preview) != 1 else ''}."]
    return []


def _date_range_findings(preview: Any) -> list[str]:
    if not isinstance(preview, Mapping):
        return []
    pairs = (
        ("filing_date_min", "filing_date_max", "Filing date range"),
        ("min_filing_date", "max_filing_date", "Filing date range"),
        ("date_min", "date_max", "Date range"),
    )
    for start_key, end_key, label in pairs:
        start = preview.get(start_key)
        end = preview.get(end_key)
        if start is not None and end is not None:
            return [f"{label}: {_format_scalar(start)} to {_format_scalar(end)}."]
    return []


def _summary_from_preview(preview: Any) -> str:
    if isinstance(preview, Mapping):
        semantic = _semantic_summary_from_preview(preview)
        if semantic:
            return semantic
        if preview.get("type") == "dataframe":
            shape = preview.get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                return f"I inspected the data and found a DataFrame preview with {shape[0]} rows and {shape[1]} columns."
        if _structural_findings(preview):
            return "I inspected the data and summarized the object type, size, and representative fields."
        if _schema_findings(preview):
            return "I inspected the data and grouped the inferred fields by their observed value types."
        return "I inspected the data and extracted a structured preview."
    if isinstance(preview, list):
        return f"I inspected the data and extracted {len(preview)} preview record{'s' if len(preview) != 1 else ''}."
    return _compact(str(preview), 500)


def _highlights(
    execution_result: ExecutionResult | None,
    artifacts: list[ExecutionArtifact],
    state_changed: bool,
) -> list[dict[str, Any]]:
    del state_changed
    highlights: list[dict[str, Any]] = []
    for artifact in artifacts[:4]:
        if artifact.kind == "table":
            highlights.append({"label": "Table", "value": artifact.metadata.get("row_count") or artifact.metadata.get("rows") or "created"})
        elif artifact.kind == "chart":
            highlights.append({"label": "Chart", "value": artifact.metadata.get("chart_type") or "created"})
        elif artifact.kind == "csv":
            highlights.append({"label": "CSV", "value": artifact.metadata.get("row_count") or artifact.metadata.get("rows") or "download"})
    preview = execution_result.result_preview if execution_result else None
    if isinstance(preview, Mapping) and preview.get("type") == "dataframe":
        shape = preview.get("shape")
        if isinstance(shape, list) and len(shape) >= 2:
            highlights.append({"label": "Rows", "value": shape[0]})
            highlights.append({"label": "Columns", "value": shape[1]})
    return highlights[:6]


def _warnings_from_verification(verification: VerificationResult) -> list[str]:
    if verification.severity == "finalize_with_warning":
        return verification.reasons
    return []


def _error_block(value: str) -> str:
    if not value:
        return ""
    return f"Latest error:\n\n```text\n{_compact(value, 1000)}\n```"


def _compact(value: str, limit: int) -> str:
    cleaned = value.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}\n..."


def _summary_from_stdout(stdout: str) -> str:
    parsed = _parse_structured_text(stdout)
    if parsed is not None:
        return _summary_from_preview(parsed)
    return f"I inspected the data and captured concise output: {_compact(stdout.strip(), 300)}"


def _parse_structured_text(value: str) -> Any | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return None


def _structural_findings(preview: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    object_type = preview.get("object_type") or preview.get("type")
    display_type = _display_object_type(object_type) if object_type else ""
    if object_type and object_type != "dataframe":
        findings.append(f"Object type: {display_type}.")

    top_level_keys = preview.get("top_level_keys")
    if isinstance(top_level_keys, list) and top_level_keys:
        findings.append(f"Top-level keys: {', '.join(str(key) for key in top_level_keys[:10])}.")

    length = (
        preview.get("length")
        or preview.get("object_length")
        or preview.get("records")
        or preview.get("row_count")
        or preview.get("approximate_size")
    )
    if isinstance(length, (int, float)) and not isinstance(length, bool):
        if display_type == "dict" or top_level_keys:
            findings.append(f"Top-level keys observed: {int(length):,}.")
        else:
            findings.append(f"Records/items observed: {int(length):,}.")

    fields = _representative_fields(preview)
    if fields:
        findings.append(f"Representative fields include {', '.join(fields[:10])}.")
        findings.extend(_semantic_field_group_findings(fields))

    tables = preview.get("tables_detected")
    if isinstance(tables, list) and tables:
        names = [str(item.get("path") or item.get("name")) for item in tables if isinstance(item, Mapping)]
        if names:
            findings.append(f"Detected tables: {', '.join(names[:8])}.")

    arrays = preview.get("arrays_detected")
    if isinstance(arrays, list) and arrays:
        names = [str(item.get("path") or item.get("name")) for item in arrays if isinstance(item, Mapping)]
        if names:
            findings.append(f"Detected arrays: {', '.join(names[:8])}.")

    collections = preview.get("record_collections_detected")
    if isinstance(collections, list) and collections:
        primary = _primary_collection(preview)
        if primary:
            path = primary.get("path") or primary.get("name")
            count = primary.get("count")
            if path and isinstance(count, (int, float)) and not isinstance(count, bool):
                findings.append(f"Primary record collection: {path} ({int(count):,} records).")
        names = [str(item.get("path") or item.get("name")) for item in collections if isinstance(item, Mapping)]
        if names:
            findings.append(f"Record collections: {', '.join(names[:8])}.")

    custom = preview.get("custom_objects_detected")
    if isinstance(custom, list) and custom:
        names = [str(item.get("type") or item.get("path")) for item in custom if isinstance(item, Mapping)]
        if names:
            findings.append(f"Custom objects detected: {', '.join(_dedupe(names)[:6])}.")

    return findings


def _primary_collection(preview: Mapping[str, Any]) -> Mapping[str, Any] | None:
    likely = preview.get("likely_primary_records")
    if isinstance(likely, list):
        for item in likely:
            if isinstance(item, Mapping):
                return item
    collections = preview.get("record_collections_detected")
    if not isinstance(collections, list):
        return None
    candidates = [item for item in collections if isinstance(item, Mapping)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: item.get("count") if isinstance(item.get("count"), (int, float)) else -1,
    )


def _representative_fields(preview: Mapping[str, Any]) -> list[str]:
    candidates = (
        preview.get("sample_records")
        or preview.get("sample_examples")
        or preview.get("examples_clean")
        or preview.get("examples")
        or preview.get("rows")
        or preview.get("sample")
    )
    if isinstance(candidates, Mapping):
        return [str(key) for key in candidates.keys() if not str(key).startswith("__")]
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, Mapping):
                return [str(key) for key in item.keys() if not str(key).startswith("__")]
    columns = preview.get("columns")
    if isinstance(columns, list):
        return [str(column.get("key") if isinstance(column, Mapping) else column) for column in columns]
    collections = preview.get("record_collections_detected") or preview.get("likely_primary_records")
    if isinstance(collections, list):
        for collection in collections:
            if isinstance(collection, Mapping) and isinstance(collection.get("fields"), list):
                return [str(field) for field in collection["fields"] if not str(field).startswith("__")]
    field_groups = preview.get("field_groups")
    if isinstance(field_groups, Mapping):
        fields: list[str] = []
        for value in field_groups.values():
            if isinstance(value, list):
                fields.extend(str(item) for item in value if not str(item).startswith("__"))
        if fields:
            return _dedupe(fields)
    keys = preview.get("keys")
    if isinstance(keys, list):
        return [str(key) for key in keys]
    return []


def _semantic_summary_from_preview(preview: Mapping[str, Any]) -> str:
    fields = _representative_fields(preview)
    if not _looks_like_patent_metadata(fields):
        return ""
    length = (
        preview.get("length")
        or preview.get("object_length")
        or preview.get("records")
        or preview.get("row_count")
        or preview.get("source_total_row_count")
        or preview.get("source_row_count")
    )
    count_phrase = ""
    if isinstance(length, (int, float)) and not isinstance(length, bool):
        count_phrase = f" with **{int(length):,} records**"
    return (
        f"This appears to be a patent portfolio metadata dataset{count_phrase}. "
        "Each record represents a patent or patent-application metadata entry, including jurisdiction, "
        "document identifiers, title, parties, filing/publication dates, legal status, citation or family "
        "metrics, and classification fields when present."
    )


def _looks_like_patent_metadata(fields: list[str]) -> bool:
    normalized = {field.lower() for field in fields}
    evidence = {
        "country",
        "doc_number",
        "kind",
        "title",
        "owners",
        "assignees",
        "inventors",
        "filing_date",
        "publication_date",
        "status",
        "family_size",
        "forward_citation_count",
    }
    return len(normalized & evidence) >= 4 and ("title" in normalized or "doc_number" in normalized)


def _semantic_field_group_findings(fields: list[str]) -> list[str]:
    normalized = {field.lower(): field for field in fields}
    if not _looks_like_patent_metadata(fields):
        return []
    groups = [
        ("Identifier fields", ("country", "doc_number", "kind", "title")),
        ("Party fields", ("owners", "inventors", "assignees")),
        (
            "Date fields",
            ("filing_date", "application_date", "publication_date", "priority_date", "expiration_date"),
        ),
        (
            "Status and metrics fields",
            ("status", "is_application", "is_grant", "family_size", "forward_citation_count", "claims"),
        ),
        ("Classification fields", ("cpc_codes", "all_cpc_codes", "ipc_codes", "classification_codes")),
    ]
    findings: list[str] = []
    for label, candidates in groups:
        present = [normalized[candidate] for candidate in candidates if candidate in normalized]
        if present:
            findings.append(f"{label}: {', '.join(present[:8])}.")
    return findings


def _display_object_type(value: Any) -> str:
    if isinstance(value, str):
        return _clean_type_repr(value)
    if isinstance(value, Mapping):
        raw = value.get("repr") or value.get("name") or value.get("type")
        if isinstance(raw, str):
            return _clean_type_repr(raw)
    return _clean_type_repr(str(value))


def _clean_type_repr(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("<class '") and stripped.endswith("'>"):
        stripped = stripped.removeprefix("<class '").removesuffix("'>")
    if "." in stripped:
        stripped = stripped.split(".")[-1]
    return stripped.strip("'\"")


def _format_scalar(value: Any) -> str:
    if isinstance(value, Mapping):
        return _display_object_type(value) if "repr" in value or "type" in value else "object"
    return str(value)


def _schema_findings(preview: Mapping[str, Any]) -> list[str]:
    grouped = _schema_groups(preview)
    if not grouped:
        return []

    findings: list[str] = []
    labels = [
        ("scalar", "Scalar fields"),
        ("date", "Date-like fields"),
        ("list", "List-like fields"),
        ("numeric", "Numeric fields"),
        ("boolean", "Boolean fields"),
        ("nullable", "Nullable fields"),
    ]
    for key, label in labels:
        fields = grouped.get(key, [])
        if fields:
            findings.append(f"{label}: {', '.join(fields[:12])}.")
    return findings


def _schema_groups(preview: Mapping[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "scalar": [],
        "date": [],
        "list": [],
        "numeric": [],
        "boolean": [],
        "nullable": [],
    }

    explicit_map = {
        "scalar_fields": "scalar",
        "scalars": "scalar",
        "scalar": "scalar",
        "date_fields": "date",
        "dates": "date",
        "date": "date",
        "date_like_fields": "date",
        "list_fields": "list",
        "lists": "list",
        "list": "list",
        "list_like_fields": "list",
        "numeric_fields": "numeric",
        "numerics": "numeric",
        "boolean_fields": "boolean",
        "booleans": "boolean",
        "nullable_fields": "nullable",
    }
    for source, target in explicit_map.items():
        value = preview.get(source)
        if isinstance(value, list):
            grouped[target].extend(str(item) for item in value if not _is_wrapper_field(str(item)))
        elif isinstance(value, Mapping):
            fields = value.get("fields") or value.get("columns")
            if isinstance(fields, list):
                grouped[target].extend(str(item) for item in fields if not _is_wrapper_field(str(item)))

    for key, value in preview.items():
        if key in {
            "object_type",
            "type",
            "length",
            "object_length",
            "records",
            "row_count",
            "approximate_size",
            "sample_examples",
            "sample_records",
            "examples",
            "examples_clean",
            "rows",
            "sample",
            "columns",
            "keys",
        }:
            continue
        if isinstance(value, str):
            if _is_wrapper_field(str(key)):
                continue
            lowered = value.lower()
            if "list" in lowered:
                grouped["list"].append(str(key))
            elif "date" in lowered or "datetime" in lowered:
                grouped["date"].append(str(key))
            elif "numeric" in lowered or lowered in {"int", "float", "integer", "number"}:
                grouped["numeric"].append(str(key))
            elif "bool" in lowered:
                grouped["boolean"].append(str(key))
            elif "scalar" in lowered or lowered in {"str", "string"}:
                grouped["scalar"].append(str(key))

    return {key: _dedupe(values) for key, values in grouped.items() if values}


def _is_wrapper_field(value: str) -> bool:
    return value in {"__dict__", "__pydantic_extra__", "__pydantic_fields_set__", "__pydantic_private__", "attrs"}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def composed_answer_event(answer: ComposedAnswer) -> dict[str, Any]:
    return {
        "type": "final_answer",
        "answer": answer.markdown,
        "state_changed": answer.state_changed,
        "highlights": answer.highlights,
        "key_findings": answer.key_findings,
        "warnings": answer.warnings,
        "artifact_ids": answer.artifact_ids,
    }


def preview_json(value: Any, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(value)
    return _compact(text, limit)
