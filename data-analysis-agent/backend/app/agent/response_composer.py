from __future__ import annotations

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
    elif execution_result and execution_result.stdout.strip():
        lines.append(f"I inspected the data records and captured this output: {_compact(execution_result.stdout.strip(), 500)}")
    elif execution_result and execution_result.result_preview is not None:
        lines.append(_summary_from_preview(execution_result.result_preview))
    else:
        lines.append("I completed the analysis with the available session context.")

    if key_findings:
        lines.extend(["", "### Key findings"])
        lines.extend(f"- {finding}" for finding in key_findings[:5])

    if mutation_summary:
        lines.extend(["", f"Mutation: {mutation_summary}"])

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
        elif artifact.kind == "chart":
            chart_type = artifact.metadata.get("chart_type")
            if chart_type is None and artifact.chart_spec:
                chart_type = artifact.chart_spec.get("chart_type")
            findings.append(f"Chart `{artifact.title or artifact.name}` is a {chart_type or 'chart'} artifact.")
        elif artifact.kind == "csv":
            row_count = artifact.metadata.get("row_count") or artifact.metadata.get("rows")
            findings.append(f"CSV `{artifact.title or artifact.name}` is ready to download{f' with {row_count} rows' if row_count is not None else ''}.")

    preview = execution_result.result_preview if execution_result else None
    if preview is not None and len(findings) < 3:
        findings.extend(_findings_from_preview(preview))
    return _dedupe(findings)[:5]


def _findings_from_preview(preview: Any) -> list[str]:
    if isinstance(preview, Mapping):
        if preview.get("type") == "dataframe":
            shape = preview.get("shape")
            columns = preview.get("columns")
            findings = []
            if isinstance(shape, list) and len(shape) >= 2:
                findings.append(f"The tabular preview reports {shape[0]} rows and {shape[1]} columns.")
            if isinstance(columns, list) and columns:
                findings.append(f"Columns include {', '.join(str(column) for column in columns[:8])}.")
            return findings
        simple = []
        for key, value in list(preview.items())[:5]:
            if isinstance(value, (str, int, float, bool)) or value is None:
                simple.append(f"{key}: {value}")
            elif isinstance(value, list):
                simple.append(f"{key}: {len(value)} item{'s' if len(value) != 1 else ''}")
        return simple
    if isinstance(preview, list):
        return [f"Preview contains {len(preview)} item{'s' if len(preview) != 1 else ''}."]
    return []


def _summary_from_preview(preview: Any) -> str:
    if isinstance(preview, Mapping):
        if preview.get("type") == "dataframe":
            shape = preview.get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                return f"I inspected the data and found a DataFrame preview with {shape[0]} rows and {shape[1]} columns."
        return f"I inspected the data and extracted a structured record preview: {_compact(str(dict(list(preview.items())[:5])), 500)}"
    if isinstance(preview, list):
        return f"I inspected the data and extracted {len(preview)} preview record{'s' if len(preview) != 1 else ''}."
    return _compact(str(preview), 500)


def _highlights(
    execution_result: ExecutionResult | None,
    artifacts: list[ExecutionArtifact],
    state_changed: bool,
) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = [{"label": "State changed", "value": "Yes" if state_changed else "No"}]
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
    return f"{cleaned[:limit].rstrip()}\n... truncated ..."


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
