from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.agent.prompts import SYSTEM_PROMPT, build_context_prompt
from app.agent.tools import AGENT_TOOLS, AgentToolRunner, looks_destructive, parse_tool_arguments, risk_level_for_code
from app.core.config import get_settings
from app.models.entities import AnalysisSession, Artifact, Branch, Dataset, PendingConfirmation, VersionNode, new_id, utc_now
from app.runtime.python_executor import ExecutionResult, PythonExecutor
from app.services.versioning import (
    active_branch,
    apply_version_to_dataset,
    checkout_branch,
    dataset_key,
    latest_versions_for_branch,
    sync_branch_pointer,
)

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
                    "code": _fake_chart_code(),
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

        shortcut_events = self._history_shortcut(session_id, request)
        if shortcut_events is not None:
            for event in shortcut_events:
                yield event
            yield {"type": "message_done"}
            return

        input_items = self._initial_input_items(context, request)
        tool_runner = AgentToolRunner(
            self.executor,
            session_id=session_id,
            active_dataset_id=request.active_dataset_id,
            branch_name=request.branch_name,
        )
        failed_execution_attempts = 0
        state_changed = False
        last_execution_result: ExecutionResult | None = None
        informative_execution_result: ExecutionResult | None = None

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
                                "type": "trace",
                                "message": "Python kept failing; preparing the clearest answer from the last traceback...",
                            }
                            yield {
                                "type": "final_answer",
                                "answer": _fallback_answer_from_execution(last_execution_result),
                                "state_changed": state_changed,
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
                            confirmation = self._create_pending_confirmation(
                                session_id=session_id,
                                request=request,
                                code=code,
                                tool_arguments=arguments,
                                input_items=input_items,
                                tool_call_id=tool_call.id,
                            )
                            yield {
                                "type": "confirmation_required",
                                "confirmation_id": confirmation.id,
                                "message": "This request appears to mutate or remove data. Please confirm before I run it.",
                                "code": code,
                                "mutation_summary": confirmation.operation_summary,
                                "operation_summary": confirmation.operation_summary,
                                "risk_level": confirmation.risk_level,
                                "affected_dataset_ids": confirmation.affected_dataset_ids,
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

                        last_execution_result = result
                        if _execution_has_user_value(result) or informative_execution_result is None:
                            informative_execution_result = result
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
                        else:
                            failed_execution_attempts = 0

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
                            yield {
                                "type": "final_answer",
                                "answer": _fallback_answer_from_execution(result),
                                "state_changed": state_changed,
                            }
                            yield {"type": "message_done"}
                            return

                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.id,
                                "output": json.dumps(tool_execution.output, default=str),
                            }
                        )
                    elif tool_call.name == "request_confirmation":
                        code = arguments.get("code")
                        confirmation = None
                        if code:
                            confirmation = self._create_pending_confirmation(
                                session_id=session_id,
                                request=request,
                                code=str(code),
                                tool_arguments={
                                    "code": str(code),
                                    "mutates_state": True,
                                    "mutation_summary": arguments.get("mutation_summary")
                                    or arguments.get("message"),
                                },
                                input_items=input_items,
                                tool_call_id=tool_call.id,
                            )
                        yield {
                            "type": "confirmation_required",
                            "confirmation_id": confirmation.id if confirmation else None,
                            "message": str(arguments.get("message", "Please confirm this mutation.")),
                            "code": code,
                            "mutation_summary": (
                                confirmation.operation_summary
                                if confirmation
                                else arguments.get("mutation_summary")
                            ),
                            "operation_summary": (
                                confirmation.operation_summary
                                if confirmation
                                else arguments.get("mutation_summary")
                            ),
                            "risk_level": confirmation.risk_level if confirmation else "medium",
                            "affected_dataset_ids": confirmation.affected_dataset_ids if confirmation else [],
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

        yield {
            "type": "trace",
            "message": "The agent reached its internal step budget; summarizing the latest execution instead...",
        }
        yield {
            "type": "final_answer",
            "answer": _fallback_answer_from_execution(
                informative_execution_result or last_execution_result,
                step_limited=True,
            ),
            "state_changed": state_changed,
        }
        yield {"type": "message_done"}

    def _create_pending_confirmation(
        self,
        *,
        session_id: str,
        request: ChatStreamRequest,
        code: str,
        tool_arguments: dict[str, Any],
        input_items: list[dict[str, Any]],
        tool_call_id: str,
    ) -> PendingConfirmation:
        operation_summary = str(
            tool_arguments.get("mutation_summary")
            or "Risky dataset mutation"
        )
        confirmation = PendingConfirmation(
            session_id=session_id,
            proposed_code=code,
            operation_summary=operation_summary,
            affected_dataset_ids=self._affected_dataset_ids(session_id, code, request.active_dataset_id),
            risk_level=risk_level_for_code(code),
            active_dataset_id=request.active_dataset_id,
            branch_name=request.branch_name,
            tool_arguments=tool_arguments,
            model_input_items=input_items,
            tool_call_id=tool_call_id,
            original_message=request.message,
        )
        self.db.add(confirmation)
        self.db.commit()
        self.db.refresh(confirmation)
        return confirmation

    def _affected_dataset_ids(
        self,
        session_id: str,
        code: str,
        active_dataset_id: str | None,
    ) -> list[str]:
        session = self.db.get(AnalysisSession, session_id)
        resolved_active_dataset_id = active_dataset_id or (session.active_dataset_id if session else None)
        datasets = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
        lowered = code.lower()
        affected: list[str] = []

        for dataset in datasets:
            key = dataset_key(dataset).lower()
            if dataset.id.lower() in lowered or key in lowered:
                affected.append(dataset.id)

        if resolved_active_dataset_id and (
            "data" in lowered or not affected
        ):
            affected.append(resolved_active_dataset_id)

        if "datasets" in lowered and not affected:
            affected.extend(dataset.id for dataset in datasets)

        return list(dict.fromkeys(affected))

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
        session = self.db.get(AnalysisSession, session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        if branch_name == "main" and session.active_branch_id:
            branch = active_branch(session, self.db)
        else:
            branch = self.db.exec(
                select(Branch).where(Branch.session_id == session_id).where(Branch.name == branch_name)
            ).first()
            if branch is None:
                raise ValueError(f"Branch not found: {branch_name}")

        datasets = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
        resolved_active_dataset_id = active_dataset_id or session.active_dataset_id
        active_dataset = _active_dataset(datasets, resolved_active_dataset_id)
        version_ids = [dataset.current_version_id for dataset in datasets if dataset.current_version_id]
        versions = (
            list(self.db.exec(select(VersionNode).where(VersionNode.id.in_(version_ids))).all())
            if version_ids
            else []
        )
        version_by_id = {version.id: version for version in versions}
        artifacts = list(self.db.exec(select(Artifact).where(Artifact.session_id == session_id)).all())
        branches = list(self.db.exec(select(Branch).where(Branch.session_id == session_id)).all())
        versions = list(
            self.db.exec(
                select(VersionNode)
                .where(VersionNode.dataset_id.in_([dataset.id for dataset in datasets]))
                .order_by(VersionNode.created_at)
            ).all()
        ) if datasets else []

        return {
            "session_id": session_id,
            "active_branch": {"id": branch.id, "name": branch.name},
            "dataset_keys": [dataset_key(dataset) for dataset in datasets],
            "branches": [
                {
                    "id": item.id,
                    "name": item.name,
                    "current_version_id": item.current_version_id,
                    "root_version_id": item.root_version_id,
                }
                for item in branches
            ],
            "active_dataset_id": active_dataset.id if active_dataset else None,
            "active_dataset_key": dataset_key(active_dataset) if active_dataset else None,
            "datasets": [
                {
                    "id": dataset.id,
                    "key": dataset_key(dataset),
                    "filename": dataset.original_filename,
                    "object_type": dataset.object_type,
                    "module": dataset.module,
                    "profile": dataset.profile,
                    "current_version": _version_summary(version_by_id.get(dataset.current_version_id)),
                }
                for dataset in datasets
            ],
            "history": [
                {
                    "id": version.id,
                    "dataset_id": version.dataset_id,
                    "branch_id": version.branch_id,
                    "parent_version_id": version.parent_version_id,
                    "label": version.label,
                    "mutation_summary": version.mutation_summary,
                    "created_at": version.created_at.isoformat(),
                }
                for version in versions[-20:]
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

    def _history_shortcut(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> list[dict[str, Any]] | None:
        message = request.message.lower()
        if not any(
            phrase in message
            for phrase in (
                "rollback",
                "roll back",
                "go back",
                "switch branch",
                "checkout",
                "check out",
                "fork",
                "create a branch",
                "compare this branch",
                "what changed",
            )
        ):
            return None

        session = self.db.get(AnalysisSession, session_id)
        if session is None:
            return [{"type": "error", "message": f"Session not found: {session_id}"}]

        try:
            branch = active_branch(session, self.db)
        except Exception as exc:
            return [{"type": "error", "message": str(exc)}]

        if "compare this branch" in message and "main" in message:
            return self._compare_branch_to_main(session_id, branch)

        if "what changed" in message:
            return self._describe_last_change(branch)

        target_branch = _branch_named_in_message(
            message,
            list(self.db.exec(select(Branch).where(Branch.session_id == session_id)).all()),
        )
        if target_branch and any(phrase in message for phrase in ("switch", "checkout", "check out")):
            datasets = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
            checkout_branch(session, target_branch, datasets, self.db)
            return [
                {"type": "trace", "message": f"Checking out branch '{target_branch.name}'..."},
                {
                    "type": "final_answer",
                    "answer": f"Checked out branch '{target_branch.name}'. Future analysis will use that branch state.",
                    "state_changed": True,
                },
            ]

        if "fork" in message or "create a branch" in message:
            source = self._version_for_fork_request(branch, message)
            if source is None:
                return [{"type": "error", "message": "I could not find a version to fork from."}]
            name = _branch_name_from_message(request.message) or f"branch-{new_id()[:6]}"
            if _branch_exists(session_id, name, self.db):
                name = f"{name}-{new_id()[:4]}"
            new_branch = Branch(session_id=session_id, name=name)
            self.db.add(new_branch)
            self.db.commit()
            self.db.refresh(new_branch)
            forked = self._copy_version(
                source=source,
                branch=new_branch,
                parent_version_id=None,
                mutation_summary=f"Forked from: {source.mutation_summary or source.label}",
            )
            new_branch.root_version_id = forked.id
            new_branch.current_version_id = forked.id
            session.active_branch_id = new_branch.id
            session.updated_at = utc_now()
            dataset = self.db.get(Dataset, forked.dataset_id)
            if dataset is not None:
                apply_version_to_dataset(dataset, forked)
                self.db.add(dataset)
            self.db.add(new_branch)
            self.db.add(session)
            self.db.commit()
            return [
                {"type": "trace", "message": f"Forking version {source.id[:8]} into branch '{name}'..."},
                {
                    "type": "final_answer",
                    "answer": f"Created and checked out branch '{name}' from the requested version.",
                    "state_changed": True,
                },
            ]

        if "rollback" in message or "roll back" in message or "go back" in message:
            target = self._version_for_rollback_request(branch, message)
            if target is None:
                return [{"type": "error", "message": "There is no earlier version to roll back to."}]
            if not request.confirmed:
                confirmation = PendingConfirmation(
                    session_id=session_id,
                    proposed_code=f"# Rollback to version {target.id}",
                    operation_summary=f"Rollback to: {target.mutation_summary or target.label}",
                    affected_dataset_ids=[target.dataset_id],
                    risk_level="high",
                    active_dataset_id=target.dataset_id,
                    branch_name=branch.name,
                    tool_arguments={"operation_kind": "rollback", "version_id": target.id},
                    original_message=request.message,
                )
                self.db.add(confirmation)
                self.db.commit()
                self.db.refresh(confirmation)
                return [
                    {
                        "type": "confirmation_required",
                        "confirmation_id": confirmation.id,
                        "message": "Rolling back changes the active dataset state. Please confirm before I restore this version.",
                        "code": confirmation.proposed_code,
                        "mutation_summary": confirmation.operation_summary,
                        "operation_summary": confirmation.operation_summary,
                        "risk_level": confirmation.risk_level,
                        "affected_dataset_ids": confirmation.affected_dataset_ids,
                    }
                ]
            current = self.db.get(VersionNode, branch.current_version_id) if branch.current_version_id else None
            rollback = self._copy_version(
                source=target,
                branch=branch,
                parent_version_id=current.id if current else None,
                mutation_summary=f"Rollback to: {target.mutation_summary or target.label}",
            )
            dataset = self.db.get(Dataset, rollback.dataset_id)
            if dataset is None:
                return [{"type": "error", "message": "Rollback target dataset is missing."}]
            apply_version_to_dataset(dataset, rollback)
            sync_branch_pointer(branch, rollback)
            session.active_branch_id = branch.id
            session.updated_at = utc_now()
            self.db.add(dataset)
            self.db.add(branch)
            self.db.add(session)
            self.db.commit()
            return [
                {"type": "trace", "message": "Restoring the requested version as a new history node..."},
                {
                    "type": "final_answer",
                    "answer": f"Rolled back to '{target.mutation_summary or target.label}' on branch '{branch.name}'.",
                    "state_changed": True,
                },
            ]

        return None

    def _version_for_rollback_request(self, branch: Branch, message: str) -> VersionNode | None:
        current = self.db.get(VersionNode, branch.current_version_id) if branch.current_version_id else None
        if current is None:
            return None
        if "original" in message:
            return self.db.get(VersionNode, branch.root_version_id) if branch.root_version_id else current
        if "one step" in message or "previous" in message or "last" in message:
            return self.db.get(VersionNode, current.parent_version_id) if current.parent_version_id else None
        return current

    def _version_for_fork_request(self, branch: Branch, message: str) -> VersionNode | None:
        current = self.db.get(VersionNode, branch.current_version_id) if branch.current_version_id else None
        versions = list(
            self.db.exec(
                select(VersionNode).where(VersionNode.branch_id == branch.id).order_by(VersionNode.created_at)
            ).all()
        )
        if "before" not in message:
            return current
        terms = [term for term in ("null", "drop", "filter", "keep", "remove") if term in message]
        for version in versions:
            summary = (version.mutation_summary or version.label).lower()
            if any(term in summary for term in terms):
                return self.db.get(VersionNode, version.parent_version_id) if version.parent_version_id else version
        return current

    def _compare_branch_to_main(self, session_id: str, branch: Branch) -> list[dict[str, Any]]:
        main = self.db.exec(
            select(Branch).where(Branch.session_id == session_id).where(Branch.name == "main")
        ).first()
        if main is None:
            return [{"type": "error", "message": "Main branch is missing."}]
        branch_versions = latest_versions_for_branch(branch.id, self.db)
        main_versions = latest_versions_for_branch(main.id, self.db)
        lines: list[str] = []
        datasets = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
        for dataset in datasets:
            left = branch_versions.get(dataset.id)
            right = main_versions.get(dataset.id)
            left_shape = left.profile.get("shape") if left else None
            right_shape = right.profile.get("shape") if right else None
            lines.append(
                f"{dataset.original_filename}: {branch.name} shape {left_shape}; main shape {right_shape}."
            )
        answer = "\n".join(lines) if lines else "There are no datasets to compare yet."
        return [
            {"type": "trace", "message": f"Comparing branch '{branch.name}' to main..."},
            {"type": "final_answer", "answer": answer, "state_changed": False},
        ]

    def _describe_last_change(self, branch: Branch) -> list[dict[str, Any]]:
        current = self.db.get(VersionNode, branch.current_version_id) if branch.current_version_id else None
        if current is None:
            return [{"type": "final_answer", "answer": "There is no version history yet.", "state_changed": False}]
        parent = self.db.get(VersionNode, current.parent_version_id) if current.parent_version_id else None
        if parent is None:
            answer = f"The current version is the root version: {current.mutation_summary or current.label}."
        else:
            answer = (
                f"Last mutation: {current.mutation_summary or current.label}. "
                f"Previous version: {parent.mutation_summary or parent.label}."
            )
        return [
            {"type": "trace", "message": "Reading the latest version transition..."},
            {"type": "final_answer", "answer": answer, "state_changed": False},
        ]

    def _copy_version(
        self,
        *,
        source: VersionNode,
        branch: Branch,
        parent_version_id: str | None,
        mutation_summary: str,
    ) -> VersionNode:
        version = VersionNode(
            id=new_id(),
            dataset_id=source.dataset_id,
            branch_id=branch.id,
            parent_version_id=parent_version_id,
            label="branch",
            snapshot_path=source.snapshot_path,
            mutation_summary=mutation_summary,
            profile=source.profile,
        )
        self.db.add(version)
        self.db.commit()
        self.db.refresh(version)
        return version


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
        "parent_version_id": version.parent_version_id,
        "mutation_summary": version.mutation_summary,
        "created_by_message_id": version.created_by_message_id,
        "created_at": version.created_at.isoformat(),
    }


def _branch_named_in_message(message: str, branches: Sequence[Branch]) -> Branch | None:
    for branch in branches:
        if branch.name.lower() in message:
            return branch
    return None


def _branch_name_from_message(message: str) -> str | None:
    words = message.replace("'", " ").replace('"', " ").split()
    for marker in ("called", "named"):
        if marker in [word.lower() for word in words]:
            index = [word.lower() for word in words].index(marker)
            if index + 1 < len(words):
                return _clean_branch_name(words[index + 1])
    if "branch" in [word.lower() for word in words]:
        index = [word.lower() for word in words].index("branch")
        if index + 1 < len(words) and words[index + 1].lower() not in {"where", "from", "that"}:
            return _clean_branch_name(words[index + 1])
    return None


def _clean_branch_name(value: str) -> str:
    cleaned = "".join(character for character in value.strip() if character.isalnum() or character in {"-", "_"})
    return cleaned[:80] or f"branch-{new_id()[:6]}"


def _branch_exists(session_id: str, name: str, db: Session) -> bool:
    return (
        db.exec(select(Branch).where(Branch.session_id == session_id).where(Branch.name == name)).first()
        is not None
    )


def _short_traceback(value: str | None) -> str | None:
    if not value:
        return None
    lines = value.strip().splitlines()
    return "\n".join(lines[-8:])


def _fallback_answer_from_execution(
    result: ExecutionResult | None,
    *,
    step_limited: bool = False,
) -> str:
    if result is None:
        if step_limited:
            return (
                "I reached the internal step budget before a Python result was available. "
                "No state changes were saved."
            )
        return "I could not complete the Python execution. No state changes were saved."

    if not result.ok:
        error_text = result.stderr or _short_traceback(result.traceback) or "Unknown Python execution error."
        return (
            "I could not complete the Python analysis after retrying. "
            "The latest Python error was:\n\n"
            f"{_truncate_for_answer(error_text)}\n\n"
            "No new mutation was saved unless a previous step explicitly reported a saved version."
        )

    parts = [
        (
            "The agent reached its internal step budget before producing a polished final response. "
            "Here is the most useful execution result available."
        )
        if step_limited
        else "The latest Python execution completed successfully."
    ]
    if result.stdout:
        parts.append(f"Latest output:\n{_truncate_for_answer(result.stdout)}")
    if result.result_preview is not None:
        parts.append(f"Latest preview:\n{_truncate_for_answer(_json_preview(result.result_preview))}")
    if result.artifacts:
        parts.append(
            f"Created {len(result.artifacts)} artifact{'s' if len(result.artifacts) != 1 else ''}."
        )
    if result.updated_datasets:
        parts.append(
            f"Saved {len(result.updated_datasets)} dataset version"
            f"{'s' if len(result.updated_datasets) != 1 else ''}."
        )
    else:
        parts.append("No dataset mutation was saved.")
    return "\n\n".join(parts)


def _json_preview(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def _truncate_for_answer(value: str, limit: int = 1200) -> str:
    cleaned = value.strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}\n... truncated ..."


def _execution_has_user_value(result: ExecutionResult) -> bool:
    return bool(
        result.stdout.strip()
        or result.result_preview is not None
        or result.artifacts
        or result.updated_datasets
        or (not result.ok and (result.stderr.strip() or result.traceback))
    )


def _model_dump(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if isinstance(item, dict):
        return item
    return dict(item)


def _fake_tool_response(name: str, arguments: dict[str, Any]) -> AgentModelResponse:
    call_id = f"fake-{name}-{new_id()}"
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


def _fake_chart_code() -> str:
    return "\n".join(
        [
            "if isinstance(data, pd.DataFrame) and len(data) > 0:",
            "    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()",
            "    label_cols = [col for col in data.columns if col not in numeric_cols]",
            "    y = numeric_cols[0] if numeric_cols else data.columns[0]",
            "    x = label_cols[0] if label_cols else data.columns[0]",
            "    if x != y and numeric_cols:",
            "        rows = data.groupby(x, dropna=False, as_index=False)[y].sum().head(10).to_dict('records')",
            "    else:",
            "        rows = data.head(10).reset_index().rename(columns={'index': 'row'}).to_dict('records')",
            "        x = 'row'",
            "    save_chart('Fake agent chart', {'title': 'Fake agent chart', 'chart_type': 'bar', 'data': rows, 'x': x, 'y': y, 'description': 'Deterministic test chart'})",
            "    preview(rows)",
            "else:",
            "    rows = [{'name': key, 'size': len(value) if hasattr(value, '__len__') else 1} for key, value in datasets.items()]",
            "    save_chart('Fake agent chart', {'title': 'Fake agent chart', 'chart_type': 'bar', 'data': rows, 'x': 'name', 'y': 'size', 'description': 'Dataset sizes'})",
            "    preview(rows)",
        ]
    )


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
