from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.runtime.python_executor import ExecutionResult, PythonExecutor


EXECUTE_PYTHON_TOOL = {
    "type": "function",
    "name": "execute_python",
    "description": "Execute Python code against the current session datasets. Use this for inspection, analysis, artifacts, and explicit state mutations.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in an isolated namespace. Available names: datasets (raw session datasets by safe unique key), dataset_profiles, active_dataset_profile, data (active dataset alias), pd, np, json, math, statistics, safe_attrs, object_to_record, objects_to_records, to_dataframe, preview_dataframe, preview, save_table, save_chart, save_csv, artifact_history, mutation_history, branch_history, current_branch, current_version. to_dataframe(data) and objects_to_records(data) process the full dataset by default; use limit only for explicit previews. Return useful values in the same execution with a final expression, RESULT = ..., preview(...), or an artifact helper. RESULT from previous executions is not available; recompute in the same code block or use artifact_history. Raw datasets do not have a .profile attribute. For row-preview requests, call save_table with dataset rows, not aggregate counts. For chart/plot/visualize requests, prefer save_chart(name, chart_spec, description=None) with title, chart_type, data, x, y, optional series/color/description; line-chart requests must use chart_type='line', and date/year time-series requests should prefer line. save_table(name='Title', data=data) and save_chart(name='Title', chart_spec=spec, data=rows) are also accepted.",
            },
            "mutates_state": {
                "type": "boolean",
                "description": "True only if the code intentionally changes datasets or data and should persist a new version.",
                "default": False,
            },
            "mutation_summary": {
                "type": "string",
                "description": "Short user-facing summary of the intended state change when mutates_state is true.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}

FINAL_ANSWER_TOOL = {
    "type": "function",
    "name": "final_answer",
    "description": "Send the final concise user-facing answer. Use only after necessary inspection or execution.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Concise final answer for the user.",
            },
            "state_changed": {
                "type": "boolean",
                "description": "Whether session dataset state was changed.",
                "default": False,
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}

REQUEST_CONFIRMATION_TOOL = {
    "type": "function",
    "name": "request_confirmation",
    "description": "Request confirmation before destructive or irreversible mutations.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Clear explanation of the risky mutation and what will change.",
            },
            "code": {
                "type": "string",
                "description": "Optional code that would run if confirmed.",
            },
            "mutation_summary": {
                "type": "string",
                "description": "Short summary of the mutation that needs confirmation.",
            },
        },
        "required": ["message"],
        "additionalProperties": False,
    },
}

AGENT_TOOLS = [EXECUTE_PYTHON_TOOL, FINAL_ANSWER_TOOL, REQUEST_CONFIRMATION_TOOL]


@dataclass
class ToolExecution:
    output: dict[str, Any]
    result: ExecutionResult | None = None


class AgentToolRunner:
    def __init__(
        self,
        executor: PythonExecutor,
        *,
        session_id: str,
        active_dataset_id: str | None,
        branch_name: str,
    ) -> None:
        self.executor = executor
        self.session_id = session_id
        self.active_dataset_id = active_dataset_id
        self.branch_name = branch_name

    def execute_python(self, arguments: dict[str, Any]) -> ToolExecution:
        code = str(arguments.get("code", ""))
        mutates_state = bool(arguments.get("mutates_state", False))
        mutation_summary = arguments.get("mutation_summary")
        result = self.executor.execute(
            self.session_id,
            code,
            active_dataset_id=self.active_dataset_id,
            branch_name=self.branch_name,
            mutates_state=mutates_state,
            mutation_summary=str(mutation_summary) if mutation_summary else None,
        )
        return ToolExecution(output=execution_result_for_model(result), result=result)


def execution_result_for_model(result: ExecutionResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "traceback": result.traceback,
        "result_preview": result.result_preview,
        "updated_datasets": [item.model_dump(mode="json") for item in result.updated_datasets],
        "artifacts": [item.model_dump(mode="json") for item in result.artifacts],
    }


def parse_tool_arguments(raw_arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def risk_level_for_code(code: str) -> str:
    lowered = code.lower()
    high_risk_markers = [
        ".drop(",
        ".dropna(",
        "del ",
        "delete",
        "truncate",
        "clear(",
        "rollback",
    ]
    if any(marker in lowered for marker in high_risk_markers):
        return "high"

    medium_risk_markers = [
        ".drop_duplicates(",
        "drop_duplicates",
        "dedup",
        "deduplicate",
        "remove(",
        "pop(",
        "overwrite",
        "normalize",
        "standardize",
        "reshape",
        "pivot",
        "melt(",
        ".query(",
        "data = data[",
        "data=data[",
        "data = df[",
        "data=df[",
        "datasets[",
    ]
    if any(marker in lowered for marker in medium_risk_markers):
        return "medium"

    return "low"


def looks_destructive(code: str) -> bool:
    lowered = code.lower()
    destructive_markers = [
        ".drop(",
        ".dropna(",
        ".drop_duplicates(",
        "drop_duplicates",
        "dedup",
        "deduplicate",
        "del ",
        "remove(",
        "pop(",
        "overwrite",
        "delete",
        "truncate",
        "clear(",
        "normalize",
        "standardize",
        "reshape",
        "pivot",
        "melt(",
        ".query(",
        "data = data[",
        "data=data[",
        "data = df[",
        "data=df[",
    ]
    return any(marker in lowered for marker in destructive_markers)
