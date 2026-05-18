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
                "description": "Python code to execute. Available names: datasets, data, pd, np, json, math, statistics, preview, save_table, save_chart, save_csv.",
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


def looks_destructive(code: str) -> bool:
    lowered = code.lower()
    destructive_markers = [
        ".drop(",
        "del ",
        "remove(",
        "pop(",
        "overwrite",
        "delete",
        "truncate",
        "clear(",
    ]
    return any(marker in lowered for marker in destructive_markers)
