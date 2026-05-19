from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.agent.agent import AgentModelClient, ChatStreamRequest, CodingAgent, FakeAgentModelClient, OpenAIResponsesClient
from app.core.config import get_settings, resolved_agent_mode
from app.db.session import get_session

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
    return StreamingResponse(
        _sse_events(agent.stream(session_id, request)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_events(events: Generator[dict, None, None]) -> Generator[str, None, None]:
    run_id = str(uuid.uuid4())
    try:
        for event in events:
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
            event_type = str(event.get("type", "message"))
            yield f"event: {event_type}\n"
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
