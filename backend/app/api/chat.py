from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator
from typing import Any
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.agent.agent import AgentModelClient, ChatStreamRequest, CodingAgent, FakeAgentModelClient, OpenAIResponsesClient
from app.core.config import get_settings, resolved_agent_mode
from app.db.session import get_session
from app.models.entities import AnalysisSession, Artifact, ChatMessage, PendingConfirmation, utc_now

router = APIRouter(prefix="/api/sessions", tags=["chat"])
logger = logging.getLogger(__name__)


def get_model_client() -> AgentModelClient:
    if resolved_agent_mode(get_settings()) == "fake":
        return FakeAgentModelClient()
    return OpenAIResponsesClient()


@router.post("/{session_id}/chat/stream")
def stream_chat(
    session_id: str,
    request: ChatStreamRequest,
    db: Annotated[Session, Depends(get_session)],
    model_client: Annotated[AgentModelClient, Depends(get_model_client)],
) -> StreamingResponse:
    agent = CodingAgent(db, model_client=model_client)
    user_message, assistant_message = _create_chat_messages(db, session_id, request)
    return StreamingResponse(
        _sse_events(
            agent.stream(session_id, request),
            db=db,
            session_id=session_id,
            assistant_message=assistant_message,
            user_message=user_message,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_events(
    events: Generator[dict, None, None],
    *,
    db: Session,
    session_id: str,
    assistant_message: ChatMessage,
    user_message: ChatMessage,
) -> Generator[str, None, None]:
    run_id = str(uuid.uuid4())
    try:
        for event in events:
            _persist_stream_event(db, session_id, assistant_message, event, run_id)
            event_type = str(event.get("type", "message"))
            yield f"event: {event_type}\n"
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    except (GeneratorExit, BrokenPipeError, ConnectionResetError):
        logger.info("Chat stream client disconnected", extra={"run_id": run_id})
        raise
    except Exception:
        logger.exception("Chat stream failed", extra={"run_id": run_id})
        error_event = {
            "type": "error",
            "message": "The assistant hit a server-side error while preparing this response. No dataset state was changed.",
            "run_id": run_id,
        }
        final_event = {
            "type": "final_answer",
            "answer": (
                "I hit a server-side error while preparing this response. "
                "No dataset state was changed. The run id is "
                f"`{run_id}`."
            ),
            "state_changed": False,
        }
        for event in (error_event, final_event, {"type": "message_done"}):
            _persist_stream_event(db, session_id, assistant_message, event, run_id)
            event_type = str(event.get("type", "message"))
            yield f"event: {event_type}\n"
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _create_chat_messages(
    db: Session,
    session_id: str,
    request: ChatStreamRequest,
) -> tuple[ChatMessage, ChatMessage]:
    analysis_session = db.get(AnalysisSession, session_id)
    if analysis_session is None:
        # Let the agent produce the user-facing not-found event, but keep persistence
        # side-effect free for unknown sessions.
        return (
            ChatMessage(session_id=session_id, role="user", content=request.message),
            ChatMessage(session_id=session_id, role="assistant", status="streaming"),
        )

    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.message,
        status="done",
    )
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content="",
        status="streaming",
    )
    if not analysis_session.name or analysis_session.name == "Demo workspace":
        analysis_session.name = _title_from_message(request.message)
    analysis_session.updated_at = utc_now()
    db.add(user_message)
    db.add(assistant_message)
    db.add(analysis_session)
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)
    return user_message, assistant_message


def _persist_stream_event(
    db: Session,
    session_id: str,
    assistant_message: ChatMessage,
    event: dict[str, Any],
    run_id: str,
) -> None:
    if not assistant_message.id or db.get(AnalysisSession, session_id) is None:
        return

    event_type = str(event.get("type", "message"))
    if event_type == "message_started":
        event["message_id"] = assistant_message.id
        assistant_message.status = "streaming"
    elif event_type == "artifact_created":
        artifact_payload = event.get("artifact")
        artifact_id = artifact_payload.get("id") if isinstance(artifact_payload, dict) else None
        if isinstance(artifact_id, str):
            assistant_message.artifact_ids = _append_unique(assistant_message.artifact_ids, artifact_id)
            _attach_artifact_to_message(db, artifact_id, assistant_message.id)
            if isinstance(artifact_payload, dict):
                artifact_payload["source_message_id"] = assistant_message.id
                metadata = artifact_payload.get("metadata")
                if isinstance(metadata, dict):
                    metadata["source_message_id"] = assistant_message.id
    elif event_type == "final_answer":
        assistant_message.status = "done"
        assistant_message.final_answer = str(event.get("answer") or "")
        assistant_message.highlights = _json_list(event.get("highlights"))
        assistant_message.key_findings = [str(item) for item in _json_list(event.get("key_findings"))]
        assistant_message.warnings = [str(item) for item in _json_list(event.get("warnings"))]
        assistant_message.state_changed = bool(event.get("state_changed"))
        assistant_message.pending_action = None
        assistant_message.artifact_ids = _append_many_unique(
            assistant_message.artifact_ids,
            [item for item in _json_list(event.get("artifact_ids")) if isinstance(item, str)],
        )
    elif event_type == "confirmation_required":
        assistant_message.status = "waiting_confirmation"
        event["assistant_message_id"] = assistant_message.id
        assistant_message.pending_action = {
            "type": "confirmation_required",
            **{
                key: event.get(key)
                for key in (
                    "confirmation_id",
                    "title",
                    "message",
                    "code",
                    "proposed_code",
                    "mutation_summary",
                    "operation_summary",
                    "dataset_name",
                    "expected_effect",
                    "affected_count",
                    "current_row_count",
                    "new_row_count",
                    "state_impact",
                    "reversible",
                    "rollback_note",
                    "confirm_label",
                    "cancel_label",
                    "risk_level",
                    "affected_dataset_ids",
                    "required_confirmation_phrase",
                )
                if key in event
            },
        }
        confirmation_id = event.get("confirmation_id")
        if isinstance(confirmation_id, str):
            confirmation = db.get(PendingConfirmation, confirmation_id)
            if confirmation is not None:
                tool_arguments = dict(confirmation.tool_arguments or {})
                tool_arguments["assistant_message_id"] = assistant_message.id
                confirmation.tool_arguments = tool_arguments
                db.add(confirmation)
    elif event_type == "clarification_required":
        assistant_message.status = "waiting_clarification"
        assistant_message.final_answer = str(event.get("message") or "")
        assistant_message.pending_action = {
            "type": "clarification_required",
            "title": event.get("title"),
            "message": event.get("message"),
            "options": event.get("options") if isinstance(event.get("options"), list) else [],
        }
    elif event_type == "message_done":
        if assistant_message.status == "streaming":
            assistant_message.status = "done"
    elif event_type == "error":
        assistant_message.status = "error"

    trace = _trace_event_from_stream(event, run_id)
    if trace is not None:
        assistant_message.trace_events = [*_json_list(assistant_message.trace_events), trace]

    assistant_message.updated_at = utc_now()
    session = db.get(AnalysisSession, session_id)
    if session is not None:
        session.updated_at = utc_now()
        db.add(session)
    db.add(assistant_message)
    db.commit()


def _trace_event_from_stream(event: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    event_type = str(event.get("type", "message"))
    if event_type in {
        "message_started",
        "final_answer",
        "message_done",
        "artifact_created",
        "confirmation_required",
        "clarification_required",
    }:
        return None
    trace: dict[str, Any] = {"id": str(uuid.uuid4()), "type": event_type}
    for key in (
        "message",
        "code",
        "ok",
        "stdout",
        "stderr",
        "traceback",
        "result_summary",
        "result_preview",
        "updated_datasets",
        "severity",
        "source",
    ):
        if key in event:
            trace[key] = _compact_trace_value(event[key])
    if event_type == "error":
        trace["run_id"] = run_id
    return trace


def _attach_artifact_to_message(db: Session, artifact_id: str, message_id: str) -> None:
    artifact = db.get(Artifact, artifact_id)
    if artifact is None:
        return
    metadata = dict(artifact.artifact_metadata or {})
    metadata["source_message_id"] = message_id
    artifact.artifact_metadata = metadata
    db.add(artifact)


def _append_unique(values: object, value: str) -> list[str]:
    return _append_many_unique(values, [value])


def _append_many_unique(values: object, additions: list[str]) -> list[str]:
    result = [str(item) for item in _json_list(values) if isinstance(item, str)]
    for item in additions:
        if item not in result:
            result.append(item)
    return result


def _json_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _compact_trace_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:4000]
    return value


def _title_from_message(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    return cleaned[:72] or "Data analysis session"
