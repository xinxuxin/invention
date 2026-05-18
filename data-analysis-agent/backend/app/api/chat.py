from __future__ import annotations

import json
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.agent.agent import ChatStreamRequest, CodingAgent, OpenAIResponsesClient
from app.db.session import get_session

router = APIRouter(prefix="/api/sessions", tags=["chat"])


def get_model_client() -> OpenAIResponsesClient:
    return OpenAIResponsesClient()


@router.post("/{session_id}/chat/stream")
def stream_chat(
    session_id: str,
    request: ChatStreamRequest,
    db: Annotated[Session, Depends(get_session)],
    model_client: Annotated[OpenAIResponsesClient, Depends(get_model_client)],
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
    for event in events:
        event_type = str(event.get("type", "message"))
        yield f"event: {event_type}\n"
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
