from __future__ import annotations

import json
from collections.abc import Protocol
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import AGENT_TOOLS, parse_tool_arguments
from app.agent.types import AgentAction
from app.core.config import get_settings


@dataclass
class AgentToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentModelResponse:
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    final_text: str | None = None
    raw_output_items: list[dict[str, Any]] = field(default_factory=list)


class AgentModelClient(Protocol):
    def create_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentModelResponse:
        ...


class OpenAIResponsesClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self.api_key = api_key or settings.openai_api_key
        self.client: OpenAI | None = None

    def create_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentModelResponse:
        if self.client is None:
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured")
            self.client = OpenAI(api_key=self.api_key)

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
        )

        tool_calls: list[AgentToolCall] = []
        raw_items: list[dict[str, Any]] = []
        for item in getattr(response, "output", []) or []:
            raw_items.append(_model_dump(item))
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(
                    AgentToolCall(
                        id=str(getattr(item, "call_id", getattr(item, "id", ""))),
                        name=str(getattr(item, "name", "")),
                        arguments=parse_tool_arguments(getattr(item, "arguments", "{}")),
                    )
                )

        return AgentModelResponse(
            tool_calls=tool_calls,
            final_text=getattr(response, "output_text", None) or _extract_text(response),
            raw_output_items=raw_items,
        )


class CodingAgent:
    """LLM planner that chooses the next minimal tool action."""

    def __init__(self, model_client: AgentModelClient) -> None:
        self.model_client = model_client

    def next_action(
        self,
        *,
        input_items: list[dict[str, Any]],
        verifier_feedback: str | None = None,
    ) -> AgentModelResponse:
        if verifier_feedback:
            input_items = [
                *input_items,
                {
                    "role": "user",
                    "content": (
                        "Verifier feedback for the next attempt:\n"
                        f"{verifier_feedback}\n\n"
                        "Retry only if needed, using the same minimal tool surface."
                    ),
                },
            ]
        return self.model_client.create_response(
            instructions=SYSTEM_PROMPT,
            input_items=input_items,
            tools=AGENT_TOOLS,
        )

    @staticmethod
    def action_from_tool_call(tool_call: AgentToolCall) -> AgentAction:
        if tool_call.name == "execute_python":
            return AgentAction(
                kind="execute_python",
                code=str(tool_call.arguments.get("code", "")),
                mutates_state=bool(tool_call.arguments.get("mutates_state", False)),
                mutation_summary=(
                    str(tool_call.arguments.get("mutation_summary"))
                    if tool_call.arguments.get("mutation_summary")
                    else None
                ),
                metadata={"tool_call_id": tool_call.id},
            )
        if tool_call.name == "request_confirmation":
            return AgentAction(
                kind="request_confirmation",
                code=str(tool_call.arguments.get("code")) if tool_call.arguments.get("code") else None,
                mutates_state=True,
                mutation_summary=(
                    str(tool_call.arguments.get("mutation_summary") or tool_call.arguments.get("message"))
                    if (tool_call.arguments.get("mutation_summary") or tool_call.arguments.get("message"))
                    else None
                ),
                metadata={"tool_call_id": tool_call.id, "message": tool_call.arguments.get("message")},
            )
        return AgentAction(
            kind="final_answer",
            answer=str(tool_call.arguments.get("answer", "")),
            metadata={"tool_call_id": tool_call.id, **tool_call.arguments},
        )


class FakeAgentModelClient:
    """Deterministic model double for tests and local demos without API calls."""

    def __init__(self) -> None:
        self.recovery_started = False

    def create_response(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentModelResponse:
        del instructions, tools
        last_item = input_items[-1] if input_items else {}
        if last_item.get("type") == "function_call_output":
            output = parse_tool_arguments(str(last_item.get("output", "{}")))
            verifier = output.get("verifier") if isinstance(output.get("verifier"), dict) else {}
            retry_instruction = str(verifier.get("retry_instruction") or "")
            if "save_table" in retry_instruction:
                return _fake_tool_response(
                    "execute_python",
                    {"code": "df = to_dataframe(data, limit=None)\nsave_table('Preview table', df.head(5))\npreview({'rows': len(df), 'columns': list(df.columns), 'source_row_count': len(df), 'analyzed_row_count': len(df)})"},
                )
            if "save_chart" in retry_instruction:
                return _fake_tool_response("execute_python", {"code": _fake_chart_code()})
            if "save_csv" in retry_instruction:
                return _fake_tool_response("execute_python", {"code": "save_csv('Export', to_dataframe(data))"})
            if output.get("ok") is False and self.recovery_started:
                return _fake_tool_response(
                    "execute_python",
                    {
                        "code": "print(type(data).__name__)\npreview(data)",
                        "mutates_state": False,
                    },
                )

            state_changed = bool(output.get("updated_datasets"))
            artifact_count = len(output.get("artifacts", []) or [])
            if state_changed:
                answer = "Saved the requested dataset mutation as a new version."
            elif artifact_count:
                answer = f"Created {artifact_count} artifact{'s' if artifact_count != 1 else ''}."
            else:
                answer = "I inspected the dataset and summarized its structure."
            return _fake_tool_response("final_answer", {"answer": answer, "state_changed": state_changed})

        prompt = _latest_user_request(input_items).lower()
        if "most important identifier" in prompt or "ambiguous" in prompt:
            return _fake_tool_response(
                "final_answer",
                {
                    "answer": (
                        "I need one clarification before changing data: which identifier field "
                        "should define the missing-value drop?"
                    ),
                    "state_changed": False,
                },
            )

        if "error recovery" in prompt or "retry path" in prompt:
            self.recovery_started = True
            return _fake_tool_response(
                "execute_python",
                {"code": "preview(data['__definitely_missing_for_retry__'])", "mutates_state": False},
            )

        if any(term in prompt for term in ("visualize", "chart", "plot")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": _fake_chart_code(prompt),
                    "mutates_state": False,
                },
            )

        if any(term in prompt for term in ("table", "rows", "preview", "top", "breakdown")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": "df = to_dataframe(data, limit=None)\nsave_table('Preview table', df.head(5))\npreview({'rows': len(df), 'columns': list(df.columns), 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
                    "mutates_state": False,
                },
            )

        if any(term in prompt for term in ("compare", "join", "multi-dataset", "multiple datasets")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "print(sorted(datasets.keys()))",
                            "summary = {key: {'type': type(value).__name__, 'shape': getattr(value, 'shape', None)} for key, value in datasets.items()}",
                            "preview(summary)",
                        ]
                    ),
                    "mutates_state": False,
                },
            )

        if any(term in prompt for term in ("mutate", "mutation", "add test column")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "if isinstance(data, pd.DataFrame):",
                            "    data = data.copy()",
                            "    data['__agent_test_row_number'] = range(len(data))",
                            "else:",
                            "    datasets[next(iter(datasets))] = data",
                            "preview(data)",
                        ]
                    ),
                    "mutates_state": True,
                    "mutation_summary": "Add agent test row number column",
                },
            )

        return _fake_tool_response(
            "execute_python",
            {
                "code": "\n".join(
                    [
                        "summary = {'type': type(data).__name__, 'shape': getattr(data, 'shape', None), 'length': len(data) if hasattr(data, '__len__') else None}",
                        "print(summary)",
                        "preview(summary)",
                    ]
                ),
                "mutates_state": False,
            },
        )


def _fake_tool_response(name: str, arguments: dict[str, Any]) -> AgentModelResponse:
    call_id = f"fake-{name}-{id(arguments)}"
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


def _latest_user_content(input_items: list[dict[str, Any]]) -> str:
    for item in reversed(input_items):
        if item.get("role") == "user":
            return str(item.get("content", ""))
    return ""


def _latest_user_request(input_items: list[dict[str, Any]]) -> str:
    content = _latest_user_content(input_items)
    marker = "User request:"
    if marker not in content:
        return content
    request = content.split(marker, maxsplit=1)[1]
    return request.split("Respond by using", maxsplit=1)[0].strip()


def _fake_chart_code(prompt: str = "") -> str:
    request_text = json.dumps(prompt)
    return "\n".join(
        [
            f"request_text = {request_text}",
            "df = to_dataframe(data, limit=None)",
            "if len(df) == 0:",
            "    rows = [{'name': key, 'size': len(value) if hasattr(value, '__len__') else 1} for key, value in datasets.items()]",
            "    x, y, title = 'name', 'size', 'Dataset sizes'",
            "elif 'filing_date' in df.columns and ('year' in request_text or 'filing' in request_text):",
            "    dates = pd.to_datetime(df['filing_date'], errors='coerce')",
            "    grouped = dates.dropna().dt.year.value_counts().sort_index().reset_index()",
            "    grouped.columns = ['filing_year', 'record_count']",
            "    rows = grouped.to_dict('records')",
            "    x, y, title = 'filing_year', 'record_count', 'Filings by year'",
            "elif 'country' in df.columns:",
            "    grouped = df['country'].astype(str).value_counts().head(15).reset_index()",
            "    grouped.columns = ['country', 'record_count']",
            "    rows = grouped.to_dict('records')",
            "    x, y, title = 'country', 'record_count', 'Patent records by country'",
            "else:",
            "    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()",
            "    label_cols = [col for col in df.columns if col not in numeric_cols]",
            "    y = numeric_cols[0] if numeric_cols else 'record_count'",
            "    x = label_cols[0] if label_cols else 'row'",
            "    if numeric_cols and label_cols:",
            "        rows = df.groupby(x, dropna=False, as_index=False)[y].sum().head(10).to_dict('records')",
            "    else:",
            "        rows = df.head(10).reset_index().rename(columns={'index': 'row'}).to_dict('records')",
            "        if 'record_count' not in rows[0]:",
            "            rows = [{'row': row.get('row', index), 'record_count': 1} for index, row in enumerate(rows)]",
            "        x, y = 'row', 'record_count'",
            "    title = 'Dataset chart'",
            "if not rows:",
            "    rows = [{'name': key, 'size': len(value) if hasattr(value, '__len__') else 1} for key, value in datasets.items()]",
            "    x, y, title = 'name', 'size', 'Dataset sizes'",
            "save_chart('Fake agent chart', {'title': 'Fake agent chart', 'chart_type': 'bar', 'data': rows, 'x': x, 'y': y, 'description': title + ' generated from the full active dataset'})",
            "save_table('Chart source data', rows)",
            "preview({'rows': rows[:20], 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
        ]
    )


def _model_dump(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if isinstance(item, dict):
        return item
    return dict(item)


def _extract_text(response: Any) -> str | None:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts) if parts else None
