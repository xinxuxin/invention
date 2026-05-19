from __future__ import annotations

import json
import re
import time
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.agent.prompts import SYSTEM_PROMPT, build_context_prompt
from app.agent.response_composer import ResponseComposer, composed_answer_event
from app.agent.tools import AGENT_TOOLS, AgentToolRunner, looks_destructive, parse_tool_arguments, risk_level_for_code
from app.agent.verifier import ResultVerifier, merge_verification_results, verifier_trace_message
from app.agent.llm_verifier import LLMVerifier
from app.core.config import get_settings
from app.models.entities import AnalysisSession, Artifact, Branch, Dataset, PendingConfirmation, VersionNode, new_id, utc_now
from app.runtime.python_executor import ExecutionArtifact, ExecutionResult, PythonExecutor, object_to_record, to_dataframe
from app.services.versioning import (
    active_branch,
    apply_version_to_dataset,
    checkout_branch,
    dataset_key,
    latest_versions_for_branch,
    sync_branch_pointer,
)
from app.storage.files import load_pickle

MAX_AGENT_STEPS = get_settings().agent_max_steps
MAX_EXECUTION_ATTEMPTS = get_settings().agent_max_retries


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
            prompt = _latest_user_request(input_items).lower()
            output = parse_tool_arguments(str(last_item.get("output", "{}")))
            verifier = output.get("verifier") if isinstance(output.get("verifier"), dict) else {}
            retry_instruction = str(verifier.get("retry_instruction") or "")
            if "save_table" in retry_instruction:
                return _fake_tool_response(
                    "execute_python",
                    {
                        "code": _fake_table_code(prompt),
                        "mutates_state": False,
                    },
                )
            if "save_chart" in retry_instruction:
                return _fake_tool_response(
                    "execute_python",
                    {
                        "code": _fake_chart_code(prompt),
                        "mutates_state": False,
                    },
                )
            if "save_csv" in retry_instruction:
                return _fake_tool_response(
                    "execute_python",
                    {
                        "code": _fake_export_code(prompt, name="Verified export"),
                        "mutates_state": False,
                    },
                )
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

        if "missing title" in prompt or "missing titles" in prompt:
            return _fake_tool_response(
                "execute_python",
                {
                    "code": "\n".join(
                        [
                            "df = to_dataframe(data)",
                            "if 'title' in df.columns:",
                            "    data = df[df['title'].notna() & (df['title'].astype(str).str.strip() != '')].copy()",
                            "else:",
                            "    data = df.copy()",
                            "preview({'rows_after': len(data), 'columns': list(data.columns)})",
                        ]
                    ),
                    "mutates_state": True,
                    "mutation_summary": "Remove records with missing title values",
                },
            )

        if any(term in prompt for term in ("visualize", "chart", "plot")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": _fake_chart_code(prompt),
                    "mutates_state": False,
                },
            )

        if "schema" in prompt:
            return _fake_tool_response(
                "execute_python",
                {
                    "code": _fake_schema_code(),
                    "mutates_state": False,
                },
            )

        if any(term in prompt for term in ("export", "csv", "download")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": _fake_export_code(prompt),
                    "mutates_state": False,
                },
            )

        if any(term in prompt for term in ("table", "rows", "preview", "top", "breakdown", "count per")):
            return _fake_tool_response(
                "execute_python",
                {
                    "code": _fake_table_code(prompt),
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
                        "df = to_dataframe(data, limit=None)",
                        "fields = [str(column) for column in list(df.columns)[:20]]",
                        "summary = {'object_type': type(data).__name__, 'shape': getattr(data, 'shape', None), 'length': len(data) if hasattr(data, '__len__') else None, 'representative_fields': fields, 'sample_records': df.head(3).to_dict('records'), 'source_row_count': len(df), 'analyzed_row_count': len(df)}",
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
        verifier: ResultVerifier | None = None,
        llm_verifier: LLMVerifier | None = None,
        response_composer: ResponseComposer | None = None,
    ) -> None:
        self.db = db
        self.model_client = model_client or OpenAIResponsesClient()
        self.executor = executor or PythonExecutor(db)
        self.verifier = verifier or ResultVerifier()
        self.llm_verifier = llm_verifier or LLMVerifier()
        self.response_composer = response_composer or ResponseComposer()

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

        delete_events = self._delete_mutation_shortcut(session_id, request)
        if delete_events is not None:
            for event in delete_events:
                yield event
            yield {"type": "message_done"}
            return

        clarification = _clarification_for_ambiguous_destructive_request(request.message)
        if clarification:
            yield {"type": "trace", "message": "Clarification needed before making a destructive data change."}
            yield {
                "type": "final_answer",
                "answer": clarification,
                "state_changed": False,
                "highlights": [],
                "warnings": ["Clarification is required before a potentially destructive mutation."],
            }
            yield {"type": "message_done"}
            return

        shortcut_events = self._history_shortcut(session_id, request)
        if shortcut_events is not None:
            for event in shortcut_events:
                yield event
            yield {"type": "message_done"}
            return

        if self._runtime_shortcuts_enabled():
            analysis_shortcut_events = self._analysis_shortcut(session_id, request)
            if analysis_shortcut_events is not None:
                for event in analysis_shortcut_events:
                    yield event
                yield {"type": "message_done"}
                return

            chart_shortcut_events = self._chart_shortcut(session_id, request)
            if chart_shortcut_events is not None:
                for event in chart_shortcut_events:
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
        settings = get_settings()
        max_steps = settings.agent_max_steps
        max_retries = settings.agent_max_retries
        turn_started_at = time.monotonic()
        failed_execution_attempts = 0
        verifier_retry_attempts = 0
        verifier_feedback: str | None = None
        state_changed = False
        last_execution_result: ExecutionResult | None = None
        informative_execution_result: ExecutionResult | None = None
        verified_artifacts_for_message: list[Any] = []
        latest_pending_artifacts: list[Any] = []
        retry_reason_history: dict[str, int] = {}
        llm_calls_this_turn = 0

        for step_index in range(max_steps):
            try:
                model_response = self.model_client.create_response(
                    instructions=SYSTEM_PROMPT,
                    input_items=(
                        [
                            *input_items,
                            {
                                "role": "user",
                                "content": (
                                    "Verifier feedback for the next attempt:\n"
                                    f"{verifier_feedback}\n\n"
                                    "Use the same minimal tools and fix only the missing requirement."
                                ),
                            },
                        ]
                        if verifier_feedback
                        else input_items
                    ),
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
                        if failed_execution_attempts >= max_retries:
                            yield {
                                "type": "trace",
                                "message": "Python kept failing; preparing the clearest answer from the last traceback...",
                            }
                            answer = self.response_composer.compose_failure(
                                execution_result=last_execution_result,
                                verification=merge_verification_results(
                                    self.verifier.verify(
                                        user_message=request.message,
                                        execution_result=last_execution_result,
                                        artifacts_created_this_turn=[],
                                        all_artifacts_for_message=verified_artifacts_for_message,
                                        current_step=step_index,
                                        max_steps=max_steps,
                                        retries_remaining=False,
                                        state_changed=state_changed,
                                    ),
                                    None,
                                    min_confidence=settings.llm_verifier_min_confidence,
                                ),
                                state_changed=state_changed,
                            )
                            yield composed_answer_event(answer)
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
                            yield self._confirmation_event(
                                confirmation,
                                message="This request appears to mutate or remove data. Please confirm before I run it.",
                            )
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
                        latest_pending_artifacts = result.artifacts

                        yield {
                            "type": "code_result_summary",
                            "ok": result.ok,
                            "stdout": result.stdout[:1000],
                            "stderr": result.stderr[:1000],
                            "traceback": _short_traceback(result.traceback),
                            "result_summary": _execution_result_summary(result),
                            "result_preview": result.result_preview,
                            "updated_datasets": [
                                item.model_dump(mode="json") for item in result.updated_datasets
                            ],
                        }

                        if not result.ok:
                            failed_execution_attempts += 1
                        else:
                            failed_execution_attempts = 0

                        deterministic = self.verifier.verify(
                            user_message=request.message,
                            execution_result=result,
                            artifacts_created_this_turn=result.artifacts,
                            all_artifacts_for_message=verified_artifacts_for_message,
                            current_step=step_index + 1,
                            max_steps=max_steps,
                            retries_remaining=(
                                failed_execution_attempts < max_retries
                                and verifier_retry_attempts < max_retries
                            ),
                            state_changed=state_changed,
                            confirmation_status="approved" if request.confirmed else None,
                            latest_code=code,
                        )
                        repeated_reason = _verification_reason_key(deterministic)
                        repeated_retry = (
                            deterministic.severity == "retry"
                            and retry_reason_history.get(repeated_reason, 0) >= 1
                        )
                        llm_verification, llm_trace = self.llm_verifier.verify_if_allowed(
                            user_message=request.message,
                            context=context,
                            execution_result=result,
                            artifacts=[*verified_artifacts_for_message, *result.artifacts],
                            state_changed=state_changed,
                            latest_code=code,
                            deterministic_result=deterministic,
                            current_step=step_index + 1,
                            turn_started_at=turn_started_at,
                            force=repeated_retry,
                            calls_this_turn=llm_calls_this_turn,
                        )
                        if llm_verification is not None:
                            llm_calls_this_turn += 1
                        if llm_trace:
                            yield {"type": "trace", "message": llm_trace}
                        verification = merge_verification_results(
                            deterministic,
                            llm_verification,
                            min_confidence=settings.llm_verifier_min_confidence,
                        )
                        if verification.severity == "retry" and (
                            failed_execution_attempts < max_retries
                            and verifier_retry_attempts < max_retries
                        ):
                            reason_key = _verification_reason_key(verification)
                            retry_reason_history[reason_key] = retry_reason_history.get(reason_key, 0) + 1
                            if retry_reason_history[reason_key] >= 2 and _artifact_matches_request(
                                request.message, result.artifacts
                            ):
                                verification = verification.model_copy(
                                    update={
                                        "passed": True,
                                        "severity": "finalize_with_warning",
                                        "should_finalize": True,
                                        "reasons": [
                                            "Repeated verifier retry matched the latest artifact; finalizing with the best matching result."
                                        ],
                                    }
                                )
                            else:
                                verifier_retry_attempts += 1
                                verifier_feedback = verification.retry_instruction
                                _mark_artifacts_status(self.db, result.artifacts, "discarded")
                                yield {
                                    "type": "verifier_result",
                                    "message": verifier_trace_message(verification),
                                    "passed": verification.passed,
                                    "severity": verification.severity,
                                    "source": verification.source,
                                    "reasons": verification.reasons,
                                }
                                if not result.ok:
                                    yield {
                                        "type": "trace",
                                        "message": "The Python attempt failed; retrying with the traceback context...",
                                    }
                                yield {
                                    "type": "trace",
                                    "message": verifier_trace_message(verification),
                                }
                                input_items.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": tool_call.id,
                                        "output": json.dumps(
                                            {
                                                **tool_execution.output,
                                                "verifier": verification.model_dump(mode="json"),
                                            },
                                            default=str,
                                        ),
                                    }
                                )
                                continue

                        yield {
                            "type": "verifier_result",
                            "message": verifier_trace_message(verification),
                            "passed": verification.passed,
                            "severity": verification.severity,
                            "source": verification.source,
                            "reasons": verification.reasons,
                        }

                        if verification.severity == "fail" or not result.ok:
                            yield {
                                "type": "trace",
                                "message": "Verifier could not approve the result; composing a clear explanation...",
                            }
                            answer = self.response_composer.compose_failure(
                                execution_result=result,
                                verification=verification,
                                state_changed=state_changed,
                            )
                            yield composed_answer_event(answer)
                            yield {"type": "message_done"}
                            return

                        if verification.should_finalize or verification.passed:
                            verified_artifacts_for_message = dedupe_artifacts_for_message(
                                [
                                    *verified_artifacts_for_message,
                                    *[_artifact_with_status(artifact, "verified") for artifact in result.artifacts],
                                ]
                            )
                            _mark_artifacts_status(self.db, verified_artifacts_for_message, "verified")
                            for artifact in verified_artifacts_for_message:
                                yield _artifact_created_event(artifact)
                            yield {
                                "type": "trace",
                                "message": "Found enough information; verified result; composing final answer...",
                            }
                            answer = self.response_composer.compose(
                                user_message=request.message,
                                execution_result=result,
                                artifacts=verified_artifacts_for_message,
                                verification=verification,
                                state_changed=state_changed,
                                mutation_summary=arguments.get("mutation_summary"),
                            )
                            yield composed_answer_event(answer)
                            yield {"type": "message_done"}
                            return

                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.id,
                                "output": json.dumps(
                                    {
                                        **tool_execution.output,
                                        "verifier": verification.model_dump(mode="json"),
                                    },
                                    default=str,
                                ),
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
                        if confirmation:
                            yield self._confirmation_event(
                                confirmation,
                                message=str(arguments.get("message", "Please confirm this mutation.")),
                            )
                        else:
                            yield {
                                "type": "confirmation_required",
                                "confirmation_id": None,
                                "title": "Confirm dataset mutation",
                                "message": str(arguments.get("message", "Please confirm this mutation.")),
                                "code": code,
                                "proposed_code": code,
                                "mutation_summary": arguments.get("mutation_summary"),
                                "operation_summary": arguments.get("mutation_summary"),
                                "risk_level": "medium",
                                "affected_dataset_ids": [],
                                "state_impact": "This will create a new version if applied.",
                                "reversible": True,
                                "rollback_note": "You can rollback to the previous version from mutation history.",
                                "confirm_label": "Apply change",
                                "cancel_label": "Cancel",
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
                        verification = self.verifier.verify(
                            user_message=request.message,
                            execution_result=informative_execution_result or last_execution_result,
                            artifacts_created_this_turn=[],
                            all_artifacts_for_message=verified_artifacts_for_message,
                            current_step=step_index + 1,
                            max_steps=max_steps,
                            retries_remaining=verifier_retry_attempts < max_retries,
                            state_changed=bool(arguments.get("state_changed", False)) or state_changed,
                            final_answer_draft=answer,
                        )
                        if verification.severity == "retry" and verifier_retry_attempts < max_retries:
                            verifier_retry_attempts += 1
                            verifier_feedback = verification.retry_instruction
                            yield {"type": "trace", "message": verifier_trace_message(verification)}
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_call.id,
                                    "output": json.dumps({"verifier": verification.model_dump(mode="json")}),
                                }
                            )
                            continue

                        composed = self.response_composer.compose(
                            user_message=request.message,
                            execution_result=informative_execution_result or last_execution_result,
                            artifacts=verified_artifacts_for_message,
                            verification=verification,
                            state_changed=bool(arguments.get("state_changed", False)) or state_changed,
                            mutation_summary=None,
                        )
                        if answer and not verified_artifacts_for_message and informative_execution_result is None:
                            composed.markdown = answer
                        yield composed_answer_event(composed)
                        yield {"type": "message_done"}
                        return

                continue

            if model_response.final_text:
                verification = self.verifier.verify(
                    user_message=request.message,
                    execution_result=informative_execution_result or last_execution_result,
                    artifacts_created_this_turn=[],
                    all_artifacts_for_message=verified_artifacts_for_message,
                    current_step=step_index + 1,
                    max_steps=max_steps,
                    retries_remaining=False,
                    state_changed=state_changed,
                    final_answer_draft=model_response.final_text,
                )
                composed = self.response_composer.compose(
                    user_message=request.message,
                    execution_result=informative_execution_result or last_execution_result,
                    artifacts=verified_artifacts_for_message,
                    verification=verification,
                    state_changed=state_changed,
                )
                if not verified_artifacts_for_message and informative_execution_result is None:
                    composed.markdown = model_response.final_text
                yield composed_answer_event(composed)
                yield {"type": "message_done"}
                return

            yield {"type": "error", "message": "The model returned no usable tool call or answer."}
            yield {"type": "message_done"}
            return

        yield {
            "type": "trace",
            "message": "The agent reached its internal step budget; summarizing the latest execution instead...",
        }
        verification = self.verifier.verify(
            user_message=request.message,
            execution_result=informative_execution_result or last_execution_result,
            artifacts_created_this_turn=[],
            all_artifacts_for_message=verified_artifacts_for_message,
            current_step=max_steps,
            max_steps=max_steps,
            retries_remaining=False,
            state_changed=state_changed,
        )
        verification = verification.model_copy(
            update={
                "severity": "finalize_with_warning" if informative_execution_result or verified_artifacts_for_message or latest_pending_artifacts else "fail",
                "should_finalize": bool(informative_execution_result or verified_artifacts_for_message or latest_pending_artifacts),
                "reasons": [
                    "Step budget reached; using the best verified result available."
                    if informative_execution_result or verified_artifacts_for_message or latest_pending_artifacts
                    else "Step budget reached before a useful result was available."
                ],
            }
        )
        fallback_artifacts = verified_artifacts_for_message or [
            _artifact_with_status(artifact, "pending_verification") for artifact in latest_pending_artifacts[-1:]
        ]
        if informative_execution_result or fallback_artifacts:
            fallback_artifacts = dedupe_artifacts_for_message(fallback_artifacts, include_pending=True)
            for artifact in fallback_artifacts:
                yield _artifact_created_event(artifact)
            answer = self.response_composer.compose(
                user_message=request.message,
                execution_result=informative_execution_result or last_execution_result,
                artifacts=fallback_artifacts,
                verification=verification,
                state_changed=state_changed,
            )
        else:
            answer = self.response_composer.compose_failure(
                execution_result=last_execution_result,
                verification=verification,
                state_changed=state_changed,
            )
        yield composed_answer_event(answer)
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

    def _confirmation_event(
        self,
        confirmation: PendingConfirmation,
        *,
        message: str,
    ) -> dict[str, Any]:
        dataset_name = self._confirmation_dataset_name(confirmation)
        summary = confirmation.operation_summary or "Apply the proposed dataset mutation."
        return {
            "type": "confirmation_required",
            "confirmation_id": confirmation.id,
            "title": "Confirm dataset mutation",
            "message": message,
            "code": confirmation.proposed_code,
            "proposed_code": confirmation.proposed_code,
            "mutation_summary": summary,
            "operation_summary": summary,
            "dataset_name": dataset_name,
            "risk_level": confirmation.risk_level,
            "expected_effect": _expected_effect(summary, dataset_name, confirmation.tool_arguments),
            "affected_count": confirmation.tool_arguments.get("affected_count"),
            "current_row_count": confirmation.tool_arguments.get("current_row_count"),
            "new_row_count": confirmation.tool_arguments.get("new_row_count"),
            "state_impact": "This will create a new version on the current branch.",
            "reversible": True,
            "rollback_note": "You can rollback to the previous version from mutation history.",
            "affected_dataset_ids": confirmation.affected_dataset_ids,
            "confirm_label": "Apply change",
            "cancel_label": "Cancel",
        }

    def _confirmation_dataset_name(self, confirmation: PendingConfirmation) -> str:
        dataset_id = confirmation.active_dataset_id or (
            confirmation.affected_dataset_ids[0] if confirmation.affected_dataset_ids else None
        )
        dataset = self.db.get(Dataset, dataset_id) if dataset_id else None
        if dataset is None:
            return "Current dataset"
        return dataset.dataset_key or dataset.original_filename

    def _delete_mutation_shortcut(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> list[dict[str, Any]] | None:
        lowered = request.message.lower().strip()
        last_match = re.search(r"\b(?:delete|remove|drop)\s+last\s+(\d+)\s+(?:entries|rows|records)\b", lowered)
        if last_match:
            return self._confirm_delete_last_n(session_id, request, int(last_match.group(1)))

        if (
            any(term in lowered for term in ("delete", "remove", "drop"))
            and any(term in lowered for term in ("empty title", "missing title", "missing titles", "blank title", "null title"))
        ):
            return self._confirm_delete_empty_title(session_id, request)

        return None

    def _confirm_delete_last_n(
        self,
        session_id: str,
        request: ChatStreamRequest,
        count: int,
    ) -> list[dict[str, Any]]:
        dataset = self._active_dataset_for_request(session_id, request.active_dataset_id)
        if dataset is None:
            return [{"type": "error", "message": "No active dataset is available for deletion."}]
        value = load_pickle(Path(dataset.current_snapshot_path))
        current_count = len(value) if hasattr(value, "__len__") else len(to_dataframe(value, limit=None))
        delete_count = min(count, current_count)
        new_count = max(0, current_count - delete_count)
        code = "\n".join(
            [
                f"delete_count = {delete_count}",
                "current_count = len(data) if hasattr(data, '__len__') else len(to_dataframe(data, limit=None))",
                "if delete_count <= 0:",
                "    RESULT = {'full_scan': True, 'deleted_count': 0, 'current_row_count': current_count, 'new_row_count': current_count}",
                "elif isinstance(data, pd.DataFrame):",
                "    data = data.iloc[:-delete_count].copy()",
                "elif isinstance(data, list):",
                "    data = data[:-delete_count]",
                "elif isinstance(data, tuple):",
                "    data = list(data[:-delete_count])",
                "else:",
                "    frame = to_dataframe(data, limit=None)",
                "    data = frame.iloc[:-delete_count].copy()",
                "RESULT = {'full_scan': True, 'deleted_count': delete_count, 'current_row_count': current_count, 'new_row_count': len(data)}",
            ]
        )
        confirmation = self._direct_confirmation(
            session_id=session_id,
            request=request,
            dataset=dataset,
            code=code,
            operation_summary=f"Delete the last {delete_count:,} records from the current working dataset",
            risk_level="high",
            metadata={
                "operation_kind": "delete_last_n",
                "delete_count": delete_count,
                "current_row_count": current_count,
                "new_row_count": new_count,
                "affected_count": delete_count,
            },
        )
        return [
            {"type": "trace", "message": "Confirmation required before deleting records."},
            self._confirmation_event(
                confirmation,
                message="Deleting records changes the active dataset state. Please confirm before I apply it.",
            ),
        ]

    def _confirm_delete_empty_title(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> list[dict[str, Any]]:
        dataset = self._active_dataset_for_request(session_id, request.active_dataset_id)
        if dataset is None:
            return [{"type": "error", "message": "No active dataset is available for deletion."}]
        value = load_pickle(Path(dataset.current_snapshot_path))
        current_count, affected_count, has_title = _scan_empty_title(value)
        if not has_title:
            return [
                {"type": "trace", "message": "Scanned the full dataset for a title field..."},
                {
                    "type": "final_answer",
                    "answer": "I scanned the full dataset and could not find a `title` field to evaluate.\n\n**State changed:** No",
                    "state_changed": False,
                },
            ]
        if affected_count == 0:
            return [
                {"type": "trace", "message": "Scanned the full dataset for empty titles..."},
                {
                    "type": "final_answer",
                    "answer": (
                        f"I scanned all {current_count:,} records and found **0** records with missing, "
                        "null, or blank `title` values.\n\n**State changed:** No"
                    ),
                    "state_changed": False,
                },
            ]

        new_count = current_count - affected_count
        code = "\n".join(
            [
                "def _empty_title(record):",
                "    value = record.get('title')",
                "    return value is None or (isinstance(value, str) and value.strip() == '')",
                "if isinstance(data, pd.DataFrame):",
                "    frame = data.copy()",
                "    title_values = frame['title'] if 'title' in frame.columns else pd.Series([], dtype=object)",
                "    keep_mask = ~(title_values.isna() | title_values.astype(str).str.strip().eq(''))",
                "    current_count = len(frame)",
                "    affected_count = current_count - int(keep_mask.sum())",
                "    data = frame.loc[keep_mask].copy()",
                "else:",
                "    records = objects_to_records(data, limit=None)",
                "    keep_mask = [not _empty_title(record) for record in records]",
                "    current_count = len(records)",
                "    affected_count = current_count - sum(keep_mask)",
                "    if isinstance(data, list):",
                "        data = [item for item, keep in zip(data, keep_mask) if keep]",
                "    elif isinstance(data, tuple):",
                "        data = [item for item, keep in zip(data, keep_mask) if keep]",
                "    else:",
                "        frame = to_dataframe(data, limit=None)",
                "        data = frame.loc[keep_mask].copy()",
                "RESULT = {'full_scan': True, 'affected_count': affected_count, 'current_row_count': current_count, 'new_row_count': len(data)}",
            ]
        )
        confirmation = self._direct_confirmation(
            session_id=session_id,
            request=request,
            dataset=dataset,
            code=code,
            operation_summary="Remove records where `title` is missing, null, or blank",
            risk_level="medium",
            metadata={
                "operation_kind": "delete_empty_title",
                "current_row_count": current_count,
                "new_row_count": new_count,
                "affected_count": affected_count,
            },
        )
        return [
            {"type": "trace", "message": "Scanned the full dataset and found records that match the delete rule."},
            self._confirmation_event(
                confirmation,
                message="Deleting records with empty titles changes the active dataset state. Please confirm before I apply it.",
            ),
        ]

    def _active_dataset_for_request(self, session_id: str, active_dataset_id: str | None) -> Dataset | None:
        session = self.db.get(AnalysisSession, session_id)
        resolved_id = active_dataset_id or (session.active_dataset_id if session else None)
        if resolved_id:
            dataset = self.db.get(Dataset, resolved_id)
            if dataset is not None and dataset.session_id == session_id:
                return dataset
        return self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).first()

    def _direct_confirmation(
        self,
        *,
        session_id: str,
        request: ChatStreamRequest,
        dataset: Dataset,
        code: str,
        operation_summary: str,
        risk_level: str,
        metadata: dict[str, Any],
    ) -> PendingConfirmation:
        confirmation = PendingConfirmation(
            session_id=session_id,
            proposed_code=code,
            operation_summary=operation_summary,
            affected_dataset_ids=[dataset.id],
            risk_level=risk_level,
            active_dataset_id=dataset.id,
            branch_name=request.branch_name,
            tool_arguments={"mutates_state": True, "mutation_summary": operation_summary, **metadata},
            model_input_items=[],
            tool_call_id=None,
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

    def _chart_shortcut(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> list[dict[str, Any]] | None:
        code = _common_chart_code(request.message)
        if code is None:
            return None

        events: list[dict[str, Any]] = [
            {"type": "trace", "message": "Preparing a chart from the current dataset..."},
            {"type": "code_started", "code": code},
        ]
        result = self.executor.execute(
            session_id,
            code,
            active_dataset_id=request.active_dataset_id,
            branch_name=request.branch_name,
            mutates_state=False,
        )
        events.append(
            {
                "type": "code_result_summary",
                "ok": result.ok,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:1000],
                "traceback": _short_traceback(result.traceback),
                "result_summary": _execution_result_summary(result),
                "result_preview": result.result_preview,
                "updated_datasets": [],
            }
        )
        verification = self.verifier.verify(
            user_message=request.message,
            execution_result=result,
            artifacts_created_this_turn=result.artifacts,
            all_artifacts_for_message=[],
            current_step=1,
            max_steps=get_settings().agent_max_steps,
            retries_remaining=False,
            state_changed=False,
            latest_code=code,
        )
        events.append(
            {
                "type": "verifier_result",
                "message": verifier_trace_message(verification),
                "passed": verification.passed,
                "severity": verification.severity,
                "source": verification.source,
                "reasons": verification.reasons,
            }
        )

        if not result.ok or verification.severity in {"retry", "fail"}:
            answer = self.response_composer.compose_failure(
                execution_result=result,
                verification=verification,
                state_changed=False,
            )
            events.append(composed_answer_event(answer))
            return events

        artifacts = dedupe_artifacts_for_message(
            [_artifact_with_status(artifact, "verified") for artifact in result.artifacts]
        )
        _mark_artifacts_status(self.db, artifacts, "verified")
        for artifact in artifacts:
            events.append(_artifact_created_event(artifact))
        events.append({"type": "trace", "message": "Chart data verified; composing final answer..."})
        answer = self.response_composer.compose(
            user_message=request.message,
            execution_result=result,
            artifacts=artifacts,
            verification=verification,
            state_changed=False,
            mutation_summary=None,
        )
        events.append(composed_answer_event(answer))
        return events

    def _runtime_shortcuts_enabled(self) -> bool:
        return isinstance(self.model_client, OpenAIResponsesClient | FakeAgentModelClient)

    def _analysis_shortcut(
        self,
        session_id: str,
        request: ChatStreamRequest,
    ) -> list[dict[str, Any]] | None:
        code = _common_table_export_code(request.message)
        if code is None:
            return None

        events: list[dict[str, Any]] = [
            {"type": "trace", "message": "Preparing a structured table or export from the current dataset..."},
            {"type": "code_started", "code": code},
        ]
        result = self.executor.execute(
            session_id,
            code,
            active_dataset_id=request.active_dataset_id,
            branch_name=request.branch_name,
            mutates_state=False,
        )
        events.append(
            {
                "type": "code_result_summary",
                "ok": result.ok,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:1000],
                "traceback": _short_traceback(result.traceback),
                "result_summary": _execution_result_summary(result),
                "result_preview": result.result_preview,
                "updated_datasets": [],
            }
        )
        verification = self.verifier.verify(
            user_message=request.message,
            execution_result=result,
            artifacts_created_this_turn=result.artifacts,
            all_artifacts_for_message=[],
            current_step=1,
            max_steps=get_settings().agent_max_steps,
            retries_remaining=False,
            state_changed=False,
            latest_code=code,
        )
        events.append(
            {
                "type": "verifier_result",
                "message": verifier_trace_message(verification),
                "passed": verification.passed,
                "severity": verification.severity,
                "source": verification.source,
                "reasons": verification.reasons,
            }
        )
        if not result.ok or verification.severity in {"retry", "fail"}:
            answer = self.response_composer.compose_failure(
                execution_result=result,
                verification=verification,
                state_changed=False,
            )
            events.append(composed_answer_event(answer))
            return events

        artifacts = dedupe_artifacts_for_message(
            [_artifact_with_status(artifact, "verified") for artifact in result.artifacts]
        )
        _mark_artifacts_status(self.db, artifacts, "verified")
        for artifact in artifacts:
            events.append(_artifact_created_event(artifact))
        events.append({"type": "trace", "message": "Structured result verified; composing final answer..."})
        answer = self.response_composer.compose(
            user_message=request.message,
            execution_result=result,
            artifacts=artifacts,
            verification=verification,
            state_changed=False,
            mutation_summary=None,
        )
        events.append(composed_answer_event(answer))
        return events

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
                "mutation history",
                "branch history",
                "rollback history",
                "current branch",
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

        if any(phrase in message for phrase in ("mutation history", "branch history", "rollback history", "current branch")):
            return self._describe_mutation_history(session_id, branch)

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
                    self._confirmation_event(
                        confirmation,
                        message="Rolling back changes the active dataset state. Please confirm before I restore this version.",
                    )
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

    def _describe_mutation_history(self, session_id: str, branch: Branch) -> list[dict[str, Any]]:
        versions = list(
            self.db.exec(
                select(VersionNode).where(VersionNode.branch_id == branch.id).order_by(VersionNode.created_at)
            ).all()
        )
        if not versions:
            answer = (
                f"Current branch: `{branch.name}`.\n\n"
                "There is no saved mutation history yet.\n\n"
                "**State changed:** No"
            )
            return [
                {"type": "trace", "message": "Reading branch mutation history..."},
                {"type": "final_answer", "answer": answer, "state_changed": False},
            ]

        datasets = {
            dataset.id: dataset
            for dataset in self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all()
        }
        lines = [f"Current branch: `{branch.name}`.", "", "Mutation history:"]
        for index, version in enumerate(versions[-12:], start=max(1, len(versions) - 11)):
            dataset = datasets.get(version.dataset_id)
            dataset_name = dataset.dataset_key or dataset.original_filename if dataset else version.dataset_id[:8]
            summary = version.mutation_summary or version.label or "Version saved"
            lines.append(f"{index}. `{dataset_name}` - {summary} ({version.created_at.isoformat()})")
        lines.extend(["", "**State changed:** No"])
        return [
            {"type": "trace", "message": "Reading branch mutation history..."},
            {"type": "final_answer", "answer": "\n".join(lines), "state_changed": False},
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


def _clarification_for_ambiguous_destructive_request(message: str) -> str | None:
    lowered = message.lower()
    if "most important identifier" in lowered:
        return (
            "I need one clarification before changing data: which identifier field should define "
            "the missing-value drop?\n\n**State changed:** No"
        )
    ambiguous_phrases = (
        "clean this dataset",
        "remove bad records",
        "drop bad records",
        "delete bad rows",
        "fix the data",
        "drop everything irrelevant",
        "remove irrelevant records",
        "clean up the data",
    )
    if any(phrase in lowered for phrase in ambiguous_phrases):
        return (
            "I need one clarification before making a destructive data change: what exact rule "
            "should define a bad record?\n\n"
            "Options I can apply:\n"
            "1. Remove records with missing title.\n"
            "2. Remove duplicate records by country/doc_number/kind/title.\n"
            "3. Remove invalid date records.\n"
            "4. Remove records with missing country/doc_number.\n"
            "5. Filter by status, country, or date.\n\n"
            "Please choose the rule you want me to apply.\n\n"
            "**State changed:** No"
        )
    return None


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


def _expected_effect(summary: str, dataset_name: str, metadata: Mapping[str, Any] | None = None) -> str:
    cleaned = summary.rstrip(".")
    metadata = metadata or {}
    current_count = metadata.get("current_row_count")
    new_count = metadata.get("new_row_count")
    affected_count = metadata.get("affected_count")
    if isinstance(current_count, int) and isinstance(new_count, int):
        count_text = f" Row count will change from {current_count:,} to {new_count:,}."
        if isinstance(affected_count, int):
            count_text += f" Affected records: {affected_count:,}."
    else:
        count_text = ""
    if cleaned:
        return f"{cleaned}. The working state for `{dataset_name}` will be updated if you approve.{count_text}"
    return f"The working state for `{dataset_name}` will be updated if you approve.{count_text}"


def _scan_empty_title(value: Any) -> tuple[int, int, bool]:
    if isinstance(value, pd.DataFrame):
        current_count = int(len(value))
        if "title" not in value.columns:
            return current_count, 0, False
        title_values = value["title"]
        affected_count = int((title_values.isna() | title_values.astype(str).str.strip().eq("")).sum())
        return current_count, affected_count, True

    if isinstance(value, (list, tuple)):
        current_count = len(value)
        has_title = False
        affected_count = 0
        for item in value:
            record = object_to_record(item)
            if "title" in record:
                has_title = True
                title = record.get("title")
                if title is None or (isinstance(title, str) and title.strip() == ""):
                    affected_count += 1
        return current_count, affected_count, has_title

    frame = to_dataframe(value, limit=None)
    current_count = int(len(frame))
    if "title" not in frame.columns:
        return current_count, 0, False
    title_values = frame["title"]
    affected_count = int((title_values.isna() | title_values.astype(str).str.strip().eq("")).sum())
    return current_count, affected_count, True


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
            f"Created {len(result.artifacts)} artifact{'s' if len(result.artifacts) != 1 else ''}; "
            "the structured output is shown in the chat."
        )
    if result.updated_datasets:
        parts.append(
            f"Saved {len(result.updated_datasets)} dataset version"
            f"{'s' if len(result.updated_datasets) != 1 else ''}."
        )
    else:
        parts.append("No dataset mutation was saved.")
    return "\n\n".join(parts)


def _should_finalize_after_execution(user_message: str, result: ExecutionResult) -> bool:
    if not result.ok:
        return False
    if result.artifacts:
        return True
    if not _execution_has_user_value(result):
        return False
    return _is_inspection_like_request(user_message)


def _is_inspection_like_request(message: str) -> bool:
    lowered = message.lower()
    markers = [
        "what is in",
        "what's in",
        "what is this",
        "summarize",
        "summary",
        "schema",
        "preview",
        "sample",
        "inspect",
        "describe",
        "columns",
        "fields",
        "tabular",
        "convert this dataset",
    ]
    return any(marker in lowered for marker in markers)


def _answer_from_useful_execution(result: ExecutionResult) -> str:
    parts = ["I inspected the data and found enough information to answer."]
    if result.result_preview is not None:
        parts.append(f"Result preview:\n{_truncate_for_answer(_json_preview(result.result_preview), limit=2200)}")
    elif result.stdout:
        parts.append(f"Output:\n{_truncate_for_answer(result.stdout, limit=1800)}")
    if result.artifacts:
        parts.append(
            f"Created {len(result.artifacts)} artifact{'s' if len(result.artifacts) != 1 else ''}; "
            "the structured output is shown in the chat."
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


def _artifact_with_status(artifact: ExecutionArtifact, status: str) -> ExecutionArtifact:
    metadata = dict(artifact.metadata)
    metadata["status"] = status
    semantic_key = artifact.semantic_key or str(metadata.get("semantic_key") or _artifact_semantic_key(artifact))
    metadata["semantic_key"] = semantic_key
    return artifact.model_copy(update={"metadata": metadata, "status": status, "semantic_key": semantic_key})


def _common_chart_code(message: str) -> str | None:
    prompt = message.lower()
    if not any(marker in prompt for marker in ("chart", "plot", "graph", "visualize", "visualization")):
        return None
    request_text = json.dumps(prompt)
    has_country_intent = "country" in prompt or "countries" in prompt
    if "active_users" in prompt or ("daily" in prompt and "active" in prompt):
        title = "Daily Active Users"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = data.get('daily_metrics') if isinstance(data, dict) and 'daily_metrics' in data else to_dataframe(data, limit=None)",
                "date_col = next((c for c in df.columns if str(c).lower() in {'date', 'day'} or 'date' in str(c).lower()), None)",
                "value_col = next((c for c in df.columns if str(c).lower() == 'active_users' or 'active' in str(c).lower()), None)",
                "if date_col is None or value_col is None:",
                "    raise ValueError('Could not find date and active_users fields for daily active users chart.')",
                "rows = df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: 'active_users'}).to_dict('records')",
                "save_table('Daily active users', rows, description='Underlying daily active users data.')",
                f"save_chart('{title}', {{'title': '{title}', 'chart_type': 'line', 'data': rows, 'x': 'date', 'y': 'active_users'}})",
                "RESULT = {'chart': 'Daily Active Users', 'rows': len(rows)}",
            ]
        )
    if "latency" in prompt and "error_rate" in prompt:
        title = "Latency vs Error Rate"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = data.get('daily_metrics') if isinstance(data, dict) and 'daily_metrics' in data else to_dataframe(data, limit=None)",
                "x_col = next((c for c in df.columns if 'latency' in str(c).lower()), None)",
                "y_col = next((c for c in df.columns if 'error_rate' in str(c).lower() or str(c).lower() == 'error'), None)",
                "if x_col is None or y_col is None:",
                "    raise ValueError('Could not find latency and error_rate fields for scatter chart.')",
                "rows = df[[x_col, y_col]].rename(columns={x_col: 'latency_ms_p95', y_col: 'error_rate'}).to_dict('records')",
                "save_table('Latency vs error rate', rows, description='Underlying data for latency/error scatter chart.')",
                f"save_chart('{title}', {{'title': '{title}', 'chart_type': 'scatter', 'data': rows, 'x': 'latency_ms_p95', 'y': 'error_rate'}})",
                "RESULT = {'chart': 'Latency vs Error Rate', 'rows': len(rows)}",
            ]
        )
    if "revenue" in prompt and has_country_intent:
        title = "Revenue by Country"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "frames = {}",
                "for collection in find_record_collections(data, max_depth=4):",
                "    path = collection.get('path', '')",
                "    try:",
                "        frames[path] = to_dataframe(get_path(data, path), limit=None)",
                "    except Exception:",
                "        rows_at_path = flatten_records_at_path(data, path)",
                "        if rows_at_path:",
                "            frames[path] = pd.DataFrame(rows_at_path)",
                "country_frame = next((frame for frame in frames.values() if 'user_id' in frame.columns and 'country' in frame.columns), None)",
                "revenue_frame = next((frame for frame in frames.values() if 'user_id' in frame.columns and any(('revenue' in str(c).lower() or 'total' in str(c).lower()) for c in frame.columns)), None)",
                "if country_frame is not None and revenue_frame is not None:",
                "    revenue_col = next(c for c in revenue_frame.columns if 'revenue' in str(c).lower() or 'total' in str(c).lower())",
                "    df = revenue_frame.merge(country_frame[['user_id', 'country']], on='user_id', how='left')",
                "    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)",
                "    grouped = df.groupby('country', dropna=False, as_index=False)[revenue_col].sum()",
                "    grouped.columns = ['country', 'total_revenue']",
                "    rows = grouped.sort_values('total_revenue', ascending=False).to_dict('records')",
                "else:",
                "    candidate_rows = []",
                "    for collection in find_record_collections(data, max_depth=5):",
                "        collection_rows = flatten_records_at_path(data, collection.get('path', ''))",
                "        if collection_rows and any('country' in row for row in collection_rows):",
                "            candidate_rows.extend(collection_rows)",
                "    df = pd.DataFrame(candidate_rows)",
                "    if df.empty or 'country' not in df.columns:",
                "        df = to_dataframe(data, limit=None)",
                "    revenue_col = next((c for c in df.columns if str(c).lower() in {'gross_revenue','total_revenue','order_total','revenue','line_total'} or 'revenue' in str(c).lower() or 'total' in str(c).lower()), None)",
                "    if revenue_col is None or 'country' not in df.columns:",
                "        raise ValueError('Could not find country and revenue fields in the discovered record collections.')",
                "    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce').fillna(0)",
                "    grouped = df.groupby('country', dropna=False, as_index=False)[revenue_col].sum()",
                "    grouped.columns = ['country', 'total_revenue']",
                "    rows = grouped.sort_values('total_revenue', ascending=False).to_dict('records')",
                "save_table('Revenue by country', rows, description='Revenue aggregated by country from discovered records.')",
                f"save_chart('{title}', {{'title': '{title}', 'chart_type': 'bar', 'data': rows, 'x': 'country', 'y': 'total_revenue'}})",
                "RESULT = {'chart': 'Revenue by Country', 'rows': len(rows), 'source_row_count': len(df), 'analyzed_row_count': len(df)}",
            ]
        )
    if has_country_intent and any(marker in prompt for marker in ("pie", "bar", "chart", "distribution")):
        chart_type = "pie" if "pie" in prompt else "bar"
        title = "Patent Records by Country"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "if 'country' not in df.columns:",
                "    rows = []",
                "    for collection in find_record_collections(data, max_depth=5):",
                "        rows.extend(flatten_records_at_path(data, collection.get('path', '')))",
                "    df = pd.DataFrame(rows)",
                "if 'country' not in df.columns:",
                "    raise ValueError(\"Cannot create a country chart because no 'country' field was found in top-level or nested records.\")",
                "counts = df['country'].fillna('Unknown').astype(str).value_counts().reset_index()",
                "counts.columns = ['country', 'record_count']",
                "counts.attrs['source_row_count'] = len(df)",
                "counts.attrs['source_total_row_count'] = len(df)",
                "counts.attrs['analyzed_row_count'] = len(df)",
                "rows = counts.to_dict('records')",
                "save_table('Country distribution', counts, description='Full-dataset records by country used for the chart.')",
                (
                    f"save_chart('{title}', {{'title': '{title}', 'chart_type': '{chart_type}', "
                    "'data': rows, 'x': 'country', 'y': 'record_count', "
                    "'description': 'Distribution of patent records by country.'})"
                ),
                (
                    f"RESULT = {{'chart': '{title}', 'chart_type': '{chart_type}', 'row_count': len(rows), "
                    "'source_row_count': len(df), 'analyzed_row_count': len(df)}"
                ),
            ]
        )
    if any(marker in prompt for marker in ("filing", "year", "date")):
        chart_type = "line" if "line" in prompt or "line chart or bar chart" in prompt else "bar"
        title = "Patent Filings by Year"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "if 'filing_date' not in df.columns:",
                "    raise ValueError(\"Cannot create a filings-by-year chart because the active dataset has no 'filing_date' field.\")",
                "dates = pd.to_datetime(df['filing_date'], errors='coerce')",
                "counts = dates.dropna().dt.year.value_counts().sort_index().reset_index()",
                "counts.columns = ['filing_year', 'filing_count']",
                "counts.attrs['source_row_count'] = len(df)",
                "counts.attrs['source_total_row_count'] = len(df)",
                "counts.attrs['analyzed_row_count'] = len(df)",
                "rows = counts.to_dict('records')",
                "save_table('Filings by year (filing_date)', counts, description='Full-dataset filing counts by year.')",
                (
                    f"save_chart('{title}', {{'title': '{title}', 'chart_type': '{chart_type}', "
                    "'data': rows, 'x': 'filing_year', 'y': 'filing_count', "
                    "'description': 'Patent filing counts by filing year.'})"
                ),
                (
                    "RESULT = {'filing_date_min': dates.min().date().isoformat() if dates.notna().any() else None, "
                    "'filing_date_max': dates.max().date().isoformat() if dates.notna().any() else None, "
                    f"'chart': '{title}', 'chart_type': '{chart_type}', 'row_count': len(rows), "
                    "'source_row_count': len(df), 'analyzed_row_count': len(df)}"
                ),
            ]
        )
    if "temperature" in prompt and ("hour" in prompt or "time" in prompt):
        title = "Average Temperature by Hour"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = []",
                "for collection in find_record_collections(data, max_depth=5):",
                "    rows.extend(flatten_records_at_path(data, collection.get('path', '')))",
                "df = pd.DataFrame(rows)",
                "temp_col = next((c for c in df.columns if 'temperature' in str(c).lower()), None)",
                "time_col = next((c for c in df.columns if 'timestamp' in str(c).lower() or 'time' in str(c).lower()), None)",
                "if temp_col is None or time_col is None:",
                "    raise ValueError('Could not find temperature and timestamp fields in discovered records.')",
                "times = pd.to_datetime(df[time_col], errors='coerce')",
                "work = pd.DataFrame({'hour': times.dt.hour, 'temperature': pd.to_numeric(df[temp_col], errors='coerce')}).dropna()",
                "grouped = work.groupby('hour', as_index=False)['temperature'].mean()",
                "grouped.columns = ['hour', 'average_temperature']",
                "rows = grouped.to_dict('records')",
                "save_table('Average temperature by hour', rows, description='Average temperature grouped by hour from discovered readings.')",
                f"save_chart('{title}', {{'title': '{title}', 'chart_type': 'line', 'data': rows, 'x': 'hour', 'y': 'average_temperature'}})",
                "RESULT = {'chart': 'Average Temperature by Hour', 'rows': len(rows)}",
            ]
        )
    if "status" in prompt:
        title = "Status Distribution"
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "if 'status' not in df.columns:",
                "    raise ValueError(\"Cannot create a status chart because the active dataset has no 'status' field.\")",
                "counts = df['status'].fillna('Unknown').astype(str).value_counts().reset_index()",
                "counts.columns = ['status', 'record_count']",
                "counts.attrs['source_row_count'] = len(df)",
                "counts.attrs['source_total_row_count'] = len(df)",
                "counts.attrs['analyzed_row_count'] = len(df)",
                "rows = counts.to_dict('records')",
                "chart_type = 'pie' if 'pie' in request_text else 'bar'",
                "save_table('Status distribution', counts, description='Full-dataset records by status used for the chart.')",
                (
                    f"save_chart('{title}', {{'title': '{title}', 'chart_type': chart_type, "
                    "'data': rows, 'x': 'status', 'y': 'record_count', "
                    "'description': 'Distribution of records by status.'})"
                ),
                (
                    f"RESULT = {{'chart': '{title}', 'row_count': len(rows), "
                    "'source_row_count': len(df), 'analyzed_row_count': len(df)}"
                ),
            ]
        )
    title = "Dataset chart"
    return "\n".join(
        [
            f"request_text = {request_text}",
            "df = to_dataframe(data, limit=None)",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()",
            "metric_cols = [col for col in numeric_cols if not any(marker in str(col).lower() for marker in ('id', 'number', 'doc', 'patent', 'application'))]",
            "label_cols = [col for col in df.columns if col not in numeric_cols]",
            "y = metric_cols[0] if metric_cols else (numeric_cols[0] if numeric_cols else 'record_count')",
            "x = label_cols[0] if label_cols else 'row'",
            "if metric_cols and label_cols:",
            "    df[y] = pd.to_numeric(df[y], errors='coerce').fillna(0)",
            "    rows = df.groupby(x, dropna=False, as_index=False)[y].sum().head(10).to_dict('records')",
            "else:",
            "    rows = df.head(10).reset_index().rename(columns={'index': 'row'}).to_dict('records')",
            "    if rows and 'record_count' not in rows[0]:",
            "        rows = [{'row': row.get('row', index), 'record_count': 1} for index, row in enumerate(rows)]",
            "    x, y = 'row', 'record_count'",
            "if not rows:",
            "    rows = [{'name': key, 'size': len(value) if hasattr(value, '__len__') else 1} for key, value in datasets.items()]",
            "    x, y = 'name', 'size'",
            f"save_chart('{title}', {{'title': '{title}', 'chart_type': 'bar', 'data': rows, 'x': x, 'y': y, 'description': 'Generic chart generated from the active dataset.'}})",
            "save_table('Chart source data', rows, description='Underlying data for the chart')",
            "RESULT = {'chart': 'Dataset chart', 'rows': len(rows)}",
        ]
    )


def _common_table_export_code(message: str) -> str | None:
    prompt = message.lower()
    request_text = json.dumps(prompt)
    if any(marker in prompt for marker in ("chart", "plot", "graph", "visualize", "visualization")):
        return None
    if "export" in prompt and ("top-5" in prompt or "top 5" in prompt or "that top" in prompt):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = None",
                "for artifact in artifact_history[::-1]:",
                "    title = str(artifact.get('title') or artifact.get('name') or '').lower()",
                "    metadata = artifact.get('metadata') if isinstance(artifact.get('metadata'), dict) else {}",
                "    candidate_rows = artifact.get('rows') or metadata.get('rows')",
                "    if artifact.get('kind') == 'table' and isinstance(candidate_rows, list) and candidate_rows:",
                "        rows = candidate_rows[:5]",
                "        break",
                "if rows is None:",
                "    df = to_dataframe(data, limit=None)",
                "    if 'country' in df.columns:",
                "        table = df['country'].fillna('Unknown').astype(str).value_counts().head(5).reset_index()",
                "        table.columns = ['country', 'record_count']",
                "        rows = table.to_dict('records')",
                "    else:",
                "        rows = to_dataframe(data, limit=None).head(5).to_dict('records')",
                "save_csv('Top-5 table export', rows)",
                "RESULT = {'csv_created': True, 'row_count': len(rows), 'state_changed': False}",
            ]
        )
    if "export" in prompt and ("top 10" in prompt or "top-10" in prompt):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "if 'country' in df.columns:",
                "    table = df['country'].fillna('Unknown').astype(str).value_counts().head(10).reset_index()",
                "    table.columns = ['country', 'record_count']",
                "    rows = table.to_dict('records')",
                "else:",
                "    rows = df.head(10).to_dict('records')",
                "save_csv('Top 10 export', rows=rows)",
                "RESULT = {'csv_created': True, 'row_count': len(rows), 'state_changed': False}",
            ]
        )
    if "export" in prompt and "alert" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = flatten_records_at_path(data, 'alerts') or flatten_records_at_path(data, 'sensors.readings.alerts')",
                "columns = ['sensor_id', 'timestamp', 'alert_type', 'severity', 'value']",
                "export_rows = [{key: row.get(key) for key in columns} for row in rows]",
                "save_csv('alerts_export.csv', rows=export_rows)",
                "RESULT = {'csv_created': True, 'row_count': len(export_rows), 'columns': columns, 'state_changed': False}",
            ]
        )
    if "export" in prompt and "risk flag" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = flatten_records_at_path(data, 'risk_flags') or flatten_records_at_path(data, 'customers.events.risk_flags')",
                "columns = ['customer_id', 'event_id', 'flag_type', 'severity']",
                "export_rows = [{key: row.get(key) for key in columns} for row in rows]",
                "save_csv('Risk Flags Export', rows=export_rows)",
                "RESULT = {'csv_created': True, 'row_count': len(export_rows), 'columns': columns, 'state_changed': False}",
            ]
        )
    if "top-level element" in prompt and "type" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "items = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else [data]",
                "rows = []",
                "for i, item in enumerate(items):",
                "    info = inspect_object(item, max_depth=1)",
                "    rows.append({'element_index': i, 'type': info.get('type'), 'structure': info.get('kind'), 'shape': info.get('shape'), 'length': info.get('length')})",
                "save_table('Top-level element types', rows, description='Each top-level element and its type.')",
                "RESULT = {'object_type': type(data).__name__, 'length': len(items), 'top_level_elements': rows}",
            ]
        )
    if "nested records" in prompt and "field" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = []",
                "for collection in find_record_collections(data, max_depth=5):",
                "    rows.append({'record_path': collection.get('path'), 'record_kind': collection.get('kind'), 'record_count': collection.get('count'), 'common_fields': ', '.join([str(field) for field in collection.get('fields', [])[:20]])})",
                "save_table('Nested record collections and fields', rows, description='Detected nested record collections and their common fields.')",
                "RESULT = {'record_collections': rows, 'field_count': sum(len(str(row.get('common_fields') or '').split(',')) for row in rows)}",
            ]
        )
    if "join key" in prompt or "join keys" in prompt or "possible join" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = []",
                "for name, obj in datasets.items():",
                "    summary = summarize_structure(obj)",
                "    fields = []",
                "    for collection in summary.get('record_collections_detected', []):",
                "        fields.extend(collection.get('fields', []) if isinstance(collection, dict) else [])",
                "    key_like = []",
                "    for field in fields:",
                "        lowered = str(field).lower()",
                "        if lowered == 'id' or lowered.endswith('_id') or lowered.endswith('_key') or lowered in {'customer_id', 'user_id', 'sensor_id', 'event_id'}:",
                "            key_like.append(str(field))",
                "    rows.append({'dataset_name': name, 'candidate_join_keys': ', '.join(sorted(set(key_like))) or 'none detected', 'record_collections': ', '.join([str(c.get('path')) for c in summary.get('record_collections_detected', [])[:4] if isinstance(c, dict)])})",
                "save_table('Possible join keys', rows, description='Key-like fields detected in each uploaded dataset. No join was performed.')",
                "RESULT = {'join_key_candidates': rows, 'state_changed': False, 'note': 'No join was performed.'}",
            ]
        )
    if any(marker in prompt for marker in ("what is in this file", "what's in this file", "what does this dataset contain", "summarize this dataset", "explain the structure", "top-level keys", "object types")):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "RESULT = summarize_structure(data)",
            ]
        )
    if "schema" in prompt and ("scalar" in prompt or "date" in prompt or "list" in prompt):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "scalar_fields = []",
                "date_fields = []",
                "list_fields = []",
                "numeric_fields = []",
                "boolean_fields = []",
                "for col in df.columns:",
                "    values = [v for v in df[col].dropna().head(50).tolist()]",
                "    if not values:",
                "        continue",
                "    if any(isinstance(v, (list, tuple, set)) for v in values):",
                "        list_fields.append(str(col)); continue",
                "    converted = pd.to_datetime(df[col], errors='coerce')",
                "    if converted.notna().sum() >= max(1, min(5, len(values)) // 2) and ('date' in str(col).lower() or 'time' in str(col).lower() or str(col).lower().endswith('_at')):",
                "        date_fields.append(str(col)); continue",
                "    if pd.api.types.is_bool_dtype(df[col]):",
                "        boolean_fields.append(str(col)); continue",
                "    if pd.api.types.is_numeric_dtype(df[col]):",
                "        numeric_fields.append(str(col)); continue",
                "    scalar_fields.append(str(col))",
                "RESULT = {'scalar_fields': scalar_fields, 'date_fields': date_fields, 'list_fields': list_fields, 'numeric_fields': numeric_fields, 'boolean_fields': boolean_fields, 'all_fields': [str(c) for c in df.columns]}",
            ]
        )
    if all(field in prompt for field in ("customer_id", "country", "segment", "joined_at", "churn_risk")):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = []",
                "for row in flatten_records_at_path(data, 'customers') or objects_to_records(get_path(data, 'customers')):",
                "    rows.append({key: row.get(key) for key in ['customer_id', 'country', 'segment', 'joined_at', 'churn_risk']})",
                "save_table('Customer preview', rows[:50], description='Customer-level preview with requested fields.')",
                "RESULT = {'columns': ['customer_id', 'country', 'segment', 'joined_at', 'churn_risk'], 'rows': rows[:5], 'preview_row_count': min(5, len(rows)), 'source_total_row_count': len(rows)}",
            ]
        )
    if all(field in prompt for field in ("sensor_id", "timestamp", "site", "zone", "temperature_c", "vibration_g", "battery_pct")):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = flatten_records_at_path(data, 'readings')",
                "if not rows:",
                "    rows = flatten_records_at_path(data, 'sensors.readings')",
                "columns = ['sensor_id', 'timestamp', 'site', 'zone', 'temperature_c', 'vibration_g', 'battery_pct']",
                "table_rows = [{key: row.get(key) for key in columns} for row in rows]",
                "save_table('Sensor readings', table_rows, description='Normalized sensor readings with requested fields.')",
                "RESULT = {'columns': columns, 'rows': table_rows[:5], 'source_total_row_count': len(table_rows)}",
            ]
        )
    if "high vibration" in prompt and "table" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = flatten_records_at_path(data, 'readings') or flatten_records_at_path(data, 'sensors.readings')",
                "table_rows = []",
                "for row in rows:",
                "    alerts = row.get('alerts') or []",
                "    has_alert = False",
                "    severities = []",
                "    for alert in alerts if isinstance(alerts, list) else []:",
                "        alert_type = str(alert.get('alert_type') or alert.get('type') or '').lower() if isinstance(alert, dict) else str(alert).lower()",
                "        if 'vibration' in alert_type:",
                "            has_alert = True",
                "            if isinstance(alert, dict): severities.append(str(alert.get('severity') or ''))",
                "    if has_alert or float(row.get('vibration_g') or 0) >= 0.3:",
                "        table_rows.append({'sensor_id': row.get('sensor_id'), 'site': row.get('site'), 'zone': row.get('zone'), 'timestamp': row.get('timestamp'), 'vibration_g': row.get('vibration_g'), 'alert_type': 'vibration', 'severity': ', '.join([s for s in severities if s])})",
                "save_table('Sensors with high vibration alerts', table_rows, description='Readings with vibration alerts or high vibration values.')",
                "RESULT = {'columns': list(table_rows[0].keys()) if table_rows else [], 'rows': table_rows[:5], 'row_count': len(table_rows)}",
            ]
        )
    if "user_embedding_matrix" in prompt or ("embedding" in prompt and "shape" in prompt):
        return "\n".join(
            [
                f"request_text = {request_text}",
                "matrix = data.get('user_embedding_matrix') if isinstance(data, dict) else get_path(data, 'user_embedding_matrix')",
                "arr = np.asarray(matrix)",
                "RESULT = {'object': 'user_embedding_matrix', 'shape': list(arr.shape), 'mean': float(np.nanmean(arr)), 'std': float(np.nanstd(arr)), 'min': float(np.nanmin(arr)), 'max': float(np.nanmax(arr))}",
            ]
        )
    if "filing date range" in prompt or "filings by year" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "if 'filing_date' not in df.columns:",
                "    raise ValueError(\"Cannot summarize filings by year because the active dataset has no 'filing_date' field.\")",
                "dates = pd.to_datetime(df['filing_date'], errors='coerce')",
                "counts = dates.dropna().dt.year.value_counts().sort_index().reset_index()",
                "counts.columns = ['filing_year', 'filing_count']",
                "counts.attrs['source_row_count'] = len(df)",
                "counts.attrs['source_total_row_count'] = len(df)",
                "counts.attrs['analyzed_row_count'] = len(df)",
                "save_table('Filings by year (filing_date)', counts, description='Full-dataset filing counts by year.')",
                "RESULT = {'filing_date_min': dates.min().date().isoformat() if dates.notna().any() else None, 'filing_date_max': dates.max().date().isoformat() if dates.notna().any() else None, 'columns': list(counts.columns), 'rows': counts.to_dict('records'), 'source_row_count': len(df), 'analyzed_row_count': len(df)}",
            ]
        )
    if "tabular preview" in prompt or "first 5 rows" in prompt or "inferred columns" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "df = to_dataframe(data, limit=None)",
                "preview_df = df.head(5).copy()",
                "preview_df.attrs['source_row_count'] = len(df)",
                "preview_df.attrs['source_total_row_count'] = len(df)",
                "preview_df.attrs['analyzed_row_count'] = 5",
                "preview_df.attrs['preview_row_count'] = len(preview_df)",
                "preview_df.attrs['is_preview'] = True",
                "save_table('Tabular preview (first 5 rows)', preview_df, description='First 5 rows with inferred columns.')",
                "RESULT = {'columns': list(df.columns), 'rows': preview_df.to_dict('records'), 'source_total_row_count': len(df), 'preview_row_count': len(preview_df), 'is_preview': True}",
            ]
        )
    if "preview" in prompt and " with " in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "requested = []",
                "tail = request_text.split(' with ', 1)[1] if ' with ' in request_text else request_text",
                "for raw in tail.replace('.', '').replace(' and ', ',').split(','):",
                "    name = raw.strip().split()[0] if raw.strip() else ''",
                "    if name and '_' in name or name in {'country', 'segment', 'status'}:",
                "        requested.append(name)",
                "rows = []",
                "for collection in find_record_collections(data, max_depth=5):",
                "    collection_rows = flatten_records_at_path(data, collection.get('path', ''))",
                "    if not collection_rows:",
                "        continue",
                "    score = sum(1 for field in requested if field in collection_rows[0])",
                "    if score >= max(1, len(requested) // 2):",
                "        rows = collection_rows[:5]",
                "        break",
                "if not rows:",
                "    rows = to_dataframe(data, limit=None).head(5).to_dict('records')",
                "save_table('Requested preview', rows, description='Preview rows matching the requested fields where possible.')",
                "RESULT = {'columns': list(rows[0].keys()) if rows else [], 'rows': rows, 'preview_row_count': len(rows), 'is_preview': True}",
            ]
        )
    if "paid" in prompt and "refunded" in prompt and "category" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "frames = []",
                "for collection in find_record_collections(data, max_depth=4):",
                "    try:",
                "        frame = to_dataframe(get_path(data, collection.get('path','')), limit=None)",
                "    except Exception:",
                "        frame = pd.DataFrame(flatten_records_at_path(data, collection.get('path','')))",
                "    if {'status', 'category'} <= set(frame.columns):",
                "        frames.append(frame)",
                "if not frames:",
                "    raise ValueError('Could not find records with status and category fields.')",
                "df = frames[0]",
                "value_col = next((c for c in df.columns if 'revenue' in str(c).lower() or 'total' in str(c).lower()), None)",
                "if value_col:",
                "    table = df.groupby(['category', 'status'], dropna=False, as_index=False)[value_col].sum()",
                "else:",
                "    table = df.groupby(['category', 'status'], dropna=False).size().reset_index(name='count')",
                "save_table('Order status by category', table, description='Paid versus refunded orders by product category.')",
                "RESULT = {'columns': list(table.columns), 'rows': table.to_dict('records')}",
            ]
        )
    if "list all datasets" in prompt or "compare the uploaded datasets" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = []",
                "for name, obj in datasets.items():",
                "    summary = summarize_structure(obj)",
                "    rows.append({'dataset_name': name, 'object_type': summary.get('object_type'), 'row_count_or_length': summary.get('length') or (summary.get('likely_primary_records') or [{}])[0].get('count'), 'schema_style': 'tables/arrays' if summary.get('tables_detected') or summary.get('arrays_detected') else 'nested/custom/mixed', 'key_fields': ', '.join((summary.get('field_groups') or {}).get('identifier', [])[:8]), 'tables_detected': ', '.join([x.get('path','') for x in summary.get('tables_detected', [])]), 'arrays_detected': ', '.join([x.get('path','') for x in summary.get('arrays_detected', [])]), 'record_collections_detected': ', '.join([x.get('path','') for x in summary.get('record_collections_detected', [])[:5]]), 'active': obj is data})",
                "save_table('Dataset comparison', rows, description='Object type, size, and structural summary for each dataset in the session.')",
                "RESULT = {'datasets': rows}",
            ]
        )
    if "list all tables and arrays" in prompt or "tables and arrays with their shapes" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "summary = summarize_structure(data)",
                "rows = []",
                "for item in summary.get('tables_detected', []):",
                "    rows.append({'name': item.get('path'), 'path': item.get('path'), 'kind': 'table', 'type': item.get('kind'), 'shape': item.get('shape')})",
                "for item in summary.get('arrays_detected', []):",
                "    rows.append({'name': item.get('path'), 'path': item.get('path'), 'kind': 'array', 'type': item.get('type'), 'shape': item.get('shape')})",
                "save_table('Tables and arrays', rows, description='Detected tabular and ndarray-like objects with shapes.')",
                "RESULT = {'tables_and_arrays': rows}",
            ]
        )
    if "element type" in prompt and "summary table" in prompt:
        return "\n".join(
            [
                f"request_text = {request_text}",
                "rows = []",
                "for i, item in enumerate(data if isinstance(data, list) else list(data) if hasattr(data, '__iter__') and not isinstance(data, dict) else [data]):",
                "    info = inspect_object(item, max_depth=1)",
                "    rows.append({'element_index': i, 'type': info.get('type'), 'structure': info.get('kind'), 'shape': info.get('shape'), 'length': info.get('length')})",
                "save_table('Top-level element types', rows, description='Summary of each top-level element and its type.')",
                "RESULT = {'elements': rows}",
            ]
        )
    return None


def _artifact_created_event(artifact: ExecutionArtifact) -> dict[str, Any]:
    return {"type": "artifact_created", "artifact": artifact.model_dump(mode="json")}


def _mark_artifacts_status(db: Session, artifacts: Sequence[ExecutionArtifact], status: str) -> None:
    if not artifacts:
        return
    for artifact_ref in artifacts:
        artifact = db.get(Artifact, artifact_ref.id)
        if artifact is None:
            continue
        metadata = artifact.artifact_metadata if isinstance(artifact.artifact_metadata, dict) else {}
        metadata = dict(metadata)
        metadata["status"] = status
        metadata.setdefault("semantic_key", artifact_ref.semantic_key or _artifact_semantic_key(artifact_ref))
        artifact.artifact_metadata = metadata
        db.add(artifact)
    db.commit()


def dedupe_artifacts_for_message(
    artifacts: list[ExecutionArtifact],
    *,
    include_pending: bool = False,
) -> list[ExecutionArtifact]:
    kept: dict[str, ExecutionArtifact] = {}
    order: list[str] = []
    for artifact in artifacts:
        status = artifact.status or artifact.metadata.get("status") or "verified"
        if status not in {"verified", "pending_verification"}:
            continue
        if status == "pending_verification" and not include_pending:
            continue
        key = artifact.semantic_key or str(artifact.metadata.get("semantic_key") or _artifact_semantic_key(artifact))
        if key not in order:
            order.append(key)
        kept[key] = artifact
    priority = {"table": 0, "chart": 1, "csv": 2, "json": 3}
    indexed = {key: index for index, key in enumerate(order)}
    return sorted(
        (kept[key] for key in order),
        key=lambda item: (
            priority.get(item.kind, 9),
            indexed.get(item.semantic_key or str(item.metadata.get("semantic_key") or _artifact_semantic_key(item)), 999),
        ),
    )


def _artifact_semantic_key(artifact: ExecutionArtifact) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    title = f"{artifact.title or artifact.name} {metadata.get('description') or ''}".lower()
    keys = {
        str(column.get("key") or column.get("label") or "").lower()
        for column in artifact.columns
        if isinstance(column, Mapping)
    }
    if artifact.kind == "table":
        if {"filing_year", "year"} & keys and any(_is_count_key(key) for key in keys):
            return "filings_by_year_table"
        if "country" in keys and any(_is_count_key(key) for key in keys):
            return "top_countries_table"
        if "schema" in title:
            return "schema_summary_table"
        if "preview" in title or "first" in title:
            return "tabular_preview_table"
    if artifact.kind == "chart":
        spec = artifact.chart_spec or metadata.get("chart_spec") or {}
        if isinstance(spec, Mapping):
            x = str(spec.get("x") or "").lower()
            y = str(spec.get("y") or "").lower()
            if x in {"filing_year", "year"} and _is_count_key(y):
                return "filings_by_year_chart"
            if x == "country":
                return "country_distribution_chart"
    if artifact.kind == "csv":
        return "csv_export"
    normalized = "".join(character if character.isalnum() else "_" for character in (artifact.title or artifact.name).lower())
    return f"{artifact.kind}_{normalized.strip('_') or artifact.id}"


def _is_count_key(key: str) -> bool:
    return key in {"count", "record_count", "patent_count", "filing_count"} or key.endswith("_count")


def _artifact_matches_request(message: str, artifacts: list[ExecutionArtifact]) -> bool:
    lowered = message.lower()
    keys = {_artifact_semantic_key(artifact) for artifact in artifacts}
    if "filing" in lowered and ("filings_by_year_table" in keys or "filings_by_year_chart" in keys):
        return True
    if "country" in lowered and ("top_countries_table" in keys or "country_distribution_chart" in keys):
        return True
    if any(marker in lowered for marker in ("preview", "first 5 rows", "inferred columns")) and "tabular_preview_table" in keys:
        return True
    if "chart" in lowered and any(artifact.kind == "chart" for artifact in artifacts):
        return True
    if "table" in lowered and any(artifact.kind == "table" for artifact in artifacts):
        return True
    return False


def _verification_reason_key(verification: Any) -> str:
    if verification.reasons:
        return str(verification.reasons[0]).lower()
    return str(verification.retry_instruction or verification.severity).lower()


def _execution_result_summary(result: ExecutionResult) -> dict[str, Any]:
    preview = result.result_preview
    summary: dict[str, Any] = {
        "ok": result.ok,
        "artifact_count": len(result.artifacts),
        "state_changed": bool(result.updated_datasets),
    }
    if isinstance(preview, Mapping):
        summary["result_type"] = preview.get("type") or "object"
        if isinstance(preview.get("shape"), list):
            summary["shape"] = preview.get("shape")
        if isinstance(preview.get("rows"), list):
            summary["preview_rows"] = len(preview["rows"])
        if preview.get("source_total_row_count") is not None:
            summary["source_total_row_count"] = preview.get("source_total_row_count")
    elif isinstance(preview, list):
        summary["result_type"] = "list"
        summary["preview_items"] = len(preview)
    elif preview is not None:
        summary["result_type"] = type(preview).__name__
    return summary


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
            "    grouped.columns = ['filing_year', 'filing_count']",
            "    rows = grouped.to_dict('records')",
            "    x, y, title = 'filing_year', 'filing_count', 'Patent filings by year'",
            "elif 'country' in df.columns:",
            "    grouped = df['country'].astype(str).value_counts().head(15).reset_index()",
            "    grouped.columns = ['country', 'record_count']",
            "    rows = grouped.to_dict('records')",
            "    x, y, title = 'country', 'record_count', 'Patent records by country'",
            "else:",
            "    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()",
            "    metric_cols = [col for col in numeric_cols if not any(marker in str(col).lower() for marker in ('id', 'number', 'doc', 'patent', 'application'))]",
            "    label_cols = [col for col in df.columns if col not in numeric_cols]",
            "    y = metric_cols[0] if metric_cols else (numeric_cols[0] if numeric_cols else 'record_count')",
            "    x = label_cols[0] if label_cols else 'row'",
            "    if metric_cols and label_cols:",
            "        df[y] = pd.to_numeric(df[y], errors='coerce').fillna(0)",
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
            "if 'pie chart' in request_text and x == 'country':",
            "    chart_type = 'pie'",
            "elif ('line chart' in request_text or ('line chart or bar chart' in request_text and ('filing' in request_text or 'year' in request_text or 'date' in request_text))) and x in {'filing_year', 'year'}:",
            "    chart_type = 'line'",
            "else:",
            "    chart_type = 'bar'",
            "save_chart(title, {'title': title, 'chart_type': chart_type, 'data': rows, 'x': x, 'y': y, 'description': title + ' generated from the full active dataset'})",
            "save_table('Chart source data', rows, description='Underlying data for the chart')",
            "preview({'rows': rows[:20], 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
        ]
    )


def _fake_table_code(prompt: str = "") -> str:
    request_text = json.dumps(prompt)
    return "\n".join(
        [
            f"request_text = {request_text}",
            "df = to_dataframe(data, limit=None)",
            "if 'filing_date' in df.columns and ('filing date range' in request_text or 'filings by year' in request_text):",
            "    dates = pd.to_datetime(df['filing_date'], errors='coerce')",
            "    grouped = dates.dropna().dt.year.value_counts().sort_index().reset_index()",
            "    grouped.columns = ['filing_year', 'filing_count']",
            "    grouped.attrs['source_row_count'] = len(df)",
            "    grouped.attrs['analyzed_row_count'] = len(df)",
            "    save_table('Patent filings by year', grouped, description='Full-dataset filing counts by year')",
            "    RESULT = {'filing_date_min': dates.min().date().isoformat() if dates.notna().any() else None, 'filing_date_max': dates.max().date().isoformat() if dates.notna().any() else None, 'columns': list(grouped.columns), 'rows': grouped.to_dict('records'), 'source_row_count': len(df), 'analyzed_row_count': len(df)}",
            "elif 'top' in request_text and 'country' in df.columns:",
            "    rows = df['country'].astype(str).value_counts().head(10).reset_index()",
            "    rows.columns = ['country', 'record_count']",
            "    rows.attrs['source_row_count'] = len(df)",
            "    rows.attrs['analyzed_row_count'] = len(df)",
            "    save_table('Top countries', rows, description='Records by country')",
            "    preview({'columns': list(rows.columns), 'rows': rows.to_dict('records'), 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
            "else:",
            "    preview_df = df.head(5)",
            "    save_table('Tabular preview', preview_df, description='First rows converted with generic helpers')",
            "    preview({'columns': list(df.columns), 'rows': preview_df.to_dict('records'), 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
        ]
    )


def _fake_export_code(prompt: str = "", *, name: str = "Fake agent export") -> str:
    request_text = json.dumps(prompt)
    artifact_name = json.dumps(name)
    return "\n".join(
        [
            f"request_text = {request_text}",
            f"artifact_name = {artifact_name}",
            "df = to_dataframe(data, limit=None)",
            "if 'top' in request_text and 'country' in df.columns:",
            "    export_df = df['country'].astype(str).value_counts().head(5).reset_index()",
            "    export_df.columns = ['country', 'record_count']",
            "    export_df.attrs['source_row_count'] = len(df)",
            "    export_df.attrs['analyzed_row_count'] = len(df)",
            "    save_csv('Top 5 countries', export_df)",
            "    preview({'rows': export_df.to_dict('records'), 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
            "else:",
            "    save_csv(artifact_name, df)",
            "    preview({'rows': len(df), 'columns': list(df.columns), 'source_row_count': len(df), 'analyzed_row_count': len(df)})",
        ]
    )


def _fake_schema_code() -> str:
    return "\n".join(
        [
            "df = to_dataframe(data, limit=None)",
            "records = df.head(20).to_dict('records')",
            "schema = {'scalar_fields': [], 'date_fields': [], 'list_like_fields': []}",
            "for column in df.columns:",
            "    sample = next((row.get(column) for row in records if row.get(column) is not None), None)",
            "    if isinstance(sample, list):",
            "        schema['list_like_fields'].append(str(column))",
            "    elif hasattr(sample, 'isoformat') or 'date' in str(column).lower():",
            "        schema['date_fields'].append(str(column))",
            "    else:",
            "        schema['scalar_fields'].append(str(column))",
            "RESULT = schema",
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
