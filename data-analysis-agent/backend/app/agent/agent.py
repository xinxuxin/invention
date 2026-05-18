from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.agent.prompts import SYSTEM_PROMPT, build_context_prompt
from app.agent.tools import AGENT_TOOLS, AgentToolRunner, looks_destructive, parse_tool_arguments
from app.core.config import get_settings
from app.models.entities import Artifact, Branch, Dataset, VersionNode
from app.runtime.python_executor import PythonExecutor

MAX_AGENT_STEPS = 8
MAX_EXECUTION_ATTEMPTS = 3


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str


class ChatStreamRequest(BaseModel):
    message: str = Field(min_length=1)
    active_dataset_id: str | None = None
    branch_name: str = "main"
    conversation_history: list[ChatHistoryMessage] = Field(default_factory=list)
    confirmed: bool = False


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
    def __init__(
        self,
        db: Session,
        *,
        model_client: AgentModelClient | None = None,
        executor: PythonExecutor | None = None,
    ) -> None:
        self.db = db
        self.model_client = model_client or OpenAIResponsesClient()
        self.executor = executor or PythonExecutor(db)

    def stream(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> Generator[dict[str, Any], None, None]:
        yield {"type": "message_started"}
        yield {"type": "trace", "message": "Reading the current session context..."}

        try:
            context = self._build_context(
                session_id=session_id,
                active_dataset_id=request.active_dataset_id,
                branch_name=request.branch_name,
                history=request.conversation_history,
            )
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            yield {"type": "message_done"}
            return

        if context["datasets"]:
            yield {"type": "trace", "message": "Inspecting the active dataset structure..."}
        else:
            yield {"type": "trace", "message": "No datasets are uploaded yet; preparing a direct response..."}

        input_items = self._initial_input_items(context, request)
        tool_runner = AgentToolRunner(
            self.executor,
            session_id=session_id,
            active_dataset_id=request.active_dataset_id,
            branch_name=request.branch_name,
        )
        failed_execution_attempts = 0
        state_changed = False

        for _ in range(MAX_AGENT_STEPS):
            try:
                model_response = self.model_client.create_response(
                    instructions=SYSTEM_PROMPT,
                    input_items=input_items,
                    tools=AGENT_TOOLS,
                )
            except Exception as exc:
                yield {"type": "error", "message": str(exc)}
                yield {"type": "message_done"}
                return
            input_items.extend(model_response.raw_output_items)

            if model_response.tool_calls:
                for tool_call in model_response.tool_calls:
                    arguments = tool_call.arguments
                    if tool_call.name == "execute_python":
                        if failed_execution_attempts >= MAX_EXECUTION_ATTEMPTS:
                            yield {
                                "type": "error",
                                "message": "Maximum Python retry attempts reached.",
                            }
                            yield {"type": "message_done"}
                            return

                        code = str(arguments.get("code", ""))
                        mutates_state = bool(arguments.get("mutates_state", False))

                        if (
                            mutates_state
                            and looks_destructive(code)
                            and not request.confirmed
                        ):
                            yield {
                                "type": "confirmation_required",
                                "message": "This request appears to mutate or remove data. Please confirm before I run it.",
                                "code": code,
                                "mutation_summary": arguments.get("mutation_summary"),
                            }
                            yield {"type": "message_done"}
                            return

                        yield {
                            "type": "trace",
                            "message": "Running a Python analysis over the current branch...",
                        }
                        yield {"type": "code_started", "code": code}
                        tool_execution = tool_runner.execute_python(arguments)
                        result = tool_execution.result
                        if result is None:
                            continue

                        state_changed = state_changed or bool(result.updated_datasets)
                        for artifact in result.artifacts:
                            yield {
                                "type": "artifact_created",
                                "artifact": artifact.model_dump(mode="json"),
                            }

                        yield {
                            "type": "code_result_summary",
                            "ok": result.ok,
                            "stdout": result.stdout[:1000],
                            "stderr": result.stderr[:1000],
                            "traceback": _short_traceback(result.traceback),
                            "result_preview": result.result_preview,
                            "updated_datasets": [
                                item.model_dump(mode="json") for item in result.updated_datasets
                            ],
                        }

                        if not result.ok:
                            failed_execution_attempts += 1

                        if not result.ok and failed_execution_attempts < MAX_EXECUTION_ATTEMPTS:
                            yield {
                                "type": "trace",
                                "message": "The Python attempt failed; retrying with the traceback context...",
                            }
                        elif not result.ok:
                            yield {
                                "type": "trace",
                                "message": "The Python attempts failed; preparing a concise explanation...",
                            }

                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.id,
                                "output": json.dumps(tool_execution.output, default=str),
                            }
                        )
                    elif tool_call.name == "request_confirmation":
                        yield {
                            "type": "confirmation_required",
                            "message": str(arguments.get("message", "Please confirm this mutation.")),
                            "code": arguments.get("code"),
                            "mutation_summary": arguments.get("mutation_summary"),
                        }
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.id,
                                "output": json.dumps({"confirmation_required": True}),
                            }
                        )
                        yield {"type": "message_done"}
                        return
                    elif tool_call.name == "final_answer":
                        answer = str(arguments.get("answer", ""))
                        yield {
                            "type": "final_answer",
                            "answer": answer,
                            "state_changed": bool(arguments.get("state_changed", False)) or state_changed,
                        }
                        yield {"type": "message_done"}
                        return

                continue

            if model_response.final_text:
                yield {
                    "type": "final_answer",
                    "answer": model_response.final_text,
                    "state_changed": state_changed,
                }
                yield {"type": "message_done"}
                return

            yield {"type": "error", "message": "The model returned no usable tool call or answer."}
            yield {"type": "message_done"}
            return

        yield {"type": "error", "message": "Agent stopped after reaching the step limit."}
        yield {"type": "message_done"}

    def _initial_input_items(
        self,
        context: dict[str, Any],
        request: ChatStreamRequest,
    ) -> list[dict[str, Any]]:
        history_items = [
            {"role": item.role, "content": item.content}
            for item in request.conversation_history[-10:]
        ]
        context_prompt = build_context_prompt(
            json.dumps(context, ensure_ascii=False, indent=2, default=str),
            request.message,
        )
        return [*history_items, {"role": "user", "content": context_prompt}]

    def _build_context(
        self,
        *,
        session_id: str,
        active_dataset_id: str | None,
        branch_name: str,
        history: Sequence[ChatHistoryMessage],
    ) -> dict[str, Any]:
        branch = self.db.exec(
            select(Branch).where(Branch.session_id == session_id).where(Branch.name == branch_name)
        ).first()
        if branch is None:
            raise ValueError(f"Branch not found: {branch_name}")

        datasets = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
        active_dataset = _active_dataset(datasets, active_dataset_id)
        version_ids = [dataset.current_version_id for dataset in datasets if dataset.current_version_id]
        versions = (
            list(self.db.exec(select(VersionNode).where(VersionNode.id.in_(version_ids))).all())
            if version_ids
            else []
        )
        version_by_id = {version.id: version for version in versions}
        artifacts = list(self.db.exec(select(Artifact).where(Artifact.session_id == session_id)).all())

        return {
            "session_id": session_id,
            "active_branch": {"id": branch.id, "name": branch.name},
            "active_dataset_id": active_dataset.id if active_dataset else None,
            "datasets": [
                {
                    "id": dataset.id,
                    "filename": dataset.original_filename,
                    "object_type": dataset.object_type,
                    "module": dataset.module,
                    "profile": dataset.profile,
                    "current_version": _version_summary(version_by_id.get(dataset.current_version_id)),
                }
                for dataset in datasets
            ],
            "artifacts": [
                {
                    "id": artifact.id,
                    "name": artifact.name,
                    "kind": artifact.kind,
                    "metadata": artifact.artifact_metadata,
                    "created_at": artifact.created_at.isoformat(),
                }
                for artifact in artifacts
            ],
            "conversation_history_count": len(history),
        }


def _active_dataset(datasets: Sequence[Dataset], active_dataset_id: str | None) -> Dataset | None:
    if not datasets:
        return None
    if active_dataset_id is None:
        return datasets[0]
    return next((dataset for dataset in datasets if dataset.id == active_dataset_id), None)


def _version_summary(version: VersionNode | None) -> dict[str, Any] | None:
    if version is None:
        return None
    return {
        "id": version.id,
        "label": version.label,
        "parent_id": version.parent_id,
        "created_at": version.created_at.isoformat(),
    }


def _short_traceback(value: str | None) -> str | None:
    if not value:
        return None
    lines = value.strip().splitlines()
    return "\n".join(lines[-8:])


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
