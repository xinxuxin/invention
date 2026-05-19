from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.agent.agent import OpenAIResponsesClient
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import AGENT_TOOLS, execution_result_for_model
from app.api.chat import get_model_client
from app.db.session import get_session
from app.models.entities import AnalysisSession, Branch, Dataset, PendingConfirmation, VersionNode, new_id, utc_now
from app.runtime.python_executor import ExecutionResult, PythonExecutor, object_to_record
from app.schemas.confirmation import ConfirmationActionResponse, ConfirmationRead
from app.services.introspection import introspect_object
from app.services.versioning import apply_version_to_dataset, latest_versions_for_branch, sync_branch_pointer
from app.storage.files import load_pickle, save_snapshot

router = APIRouter(prefix="/api/sessions", tags=["confirmations"])


@router.post(
    "/{session_id}/confirmations/{confirmation_id}/approve",
    response_model=ConfirmationActionResponse,
)
def approve_confirmation(
    session_id: str,
    confirmation_id: str,
    db: Annotated[Session, Depends(get_session)],
    model_client: Annotated[OpenAIResponsesClient, Depends(get_model_client)],
) -> ConfirmationActionResponse:
    confirmation = _pending_confirmation_or_404(session_id, confirmation_id, db)
    if confirmation.tool_arguments.get("operation_kind") == "rollback":
        return _approve_rollback_confirmation(confirmation, db)
    if confirmation.tool_arguments.get("operation_kind") in {"delete_last_n", "delete_empty_title"}:
        return _approve_direct_mutation_confirmation(confirmation, db)

    events: list[dict[str, Any]] = [
        {"type": "trace", "message": "Confirmation approved; applying the proposed mutation..."},
        {"type": "code_started", "code": confirmation.proposed_code},
    ]
    executor = PythonExecutor(db)
    result = executor.execute(
        session_id,
        confirmation.proposed_code,
        active_dataset_id=confirmation.active_dataset_id,
        branch_name=confirmation.branch_name,
        mutates_state=True,
        mutation_summary=confirmation.operation_summary,
        created_by_message_id=confirmation.id,
    )
    for artifact in result.artifacts:
        events.append({"type": "artifact_created", "artifact": artifact.model_dump(mode="json")})
    events.append(_code_result_event(result))

    confirmation.status = "approved" if result.ok else "failed"
    confirmation.resolved_at = utc_now()
    db.add(confirmation)
    db.commit()
    db.refresh(confirmation)

    events.append(_final_answer_event(confirmation, result, model_client))
    events.append({"type": "message_done"})
    return ConfirmationActionResponse(
        confirmation=_confirmation_read(confirmation),
        events=events,
        result=execution_result_for_model(result),
    )


@router.post(
    "/{session_id}/confirmations/{confirmation_id}/reject",
    response_model=ConfirmationActionResponse,
)
def reject_confirmation(
    session_id: str,
    confirmation_id: str,
    db: Annotated[Session, Depends(get_session)],
) -> ConfirmationActionResponse:
    confirmation = _pending_confirmation_or_404(session_id, confirmation_id, db)
    confirmation.status = "rejected"
    confirmation.resolved_at = utc_now()
    db.add(confirmation)
    db.commit()
    db.refresh(confirmation)

    events = [
        {
            "type": "final_answer",
            "answer": (
                "Canceled. I did not run the proposed mutation, and the dataset state was left unchanged."
            ),
            "state_changed": False,
        },
        {"type": "message_done"},
    ]
    return ConfirmationActionResponse(
        confirmation=_confirmation_read(confirmation),
        events=events,
        result=None,
    )


def _pending_confirmation_or_404(
    session_id: str,
    confirmation_id: str,
    db: Session,
) -> PendingConfirmation:
    confirmation = db.get(PendingConfirmation, confirmation_id)
    if confirmation is None or confirmation.session_id != session_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Confirmation not found")
    if confirmation.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Confirmation is no longer pending")
    return confirmation


def _confirmation_read(confirmation: PendingConfirmation) -> ConfirmationRead:
    return ConfirmationRead(
        id=confirmation.id,
        session_id=confirmation.session_id,
        proposed_code=confirmation.proposed_code,
        operation_summary=confirmation.operation_summary,
        affected_dataset_ids=confirmation.affected_dataset_ids,
        risk_level=confirmation.risk_level,
        status=confirmation.status,
        active_dataset_id=confirmation.active_dataset_id,
        branch_name=confirmation.branch_name,
        created_at=confirmation.created_at,
        resolved_at=confirmation.resolved_at,
    )


def _approve_rollback_confirmation(
    confirmation: PendingConfirmation,
    db: Session,
) -> ConfirmationActionResponse:
    session = db.get(AnalysisSession, confirmation.session_id)
    version_id = confirmation.tool_arguments.get("version_id")
    target = db.get(VersionNode, str(version_id)) if version_id else None
    if session is None or target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rollback target not found")

    dataset = db.get(Dataset, target.dataset_id)
    branch = db.exec(
        select(Branch)
        .where(Branch.session_id == confirmation.session_id)
        .where(Branch.name == confirmation.branch_name)
    ).first()
    if dataset is None or dataset.session_id != confirmation.session_id or branch is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rollback target not found")

    rollback = VersionNode(
        id=new_id(),
        dataset_id=target.dataset_id,
        branch_id=branch.id,
        parent_version_id=branch.current_version_id or dataset.current_version_id,
        label="rollback",
        snapshot_path=target.snapshot_path,
        mutation_summary=confirmation.operation_summary,
        created_by_message_id=confirmation.id,
        profile=target.profile,
    )
    db.add(rollback)
    apply_version_to_dataset(dataset, rollback)
    sync_branch_pointer(branch, rollback)
    session.active_branch_id = branch.id
    session.updated_at = utc_now()
    confirmation.status = "approved"
    confirmation.resolved_at = utc_now()

    db.add(dataset)
    db.add(branch)
    db.add(session)
    db.add(confirmation)
    db.commit()
    db.refresh(confirmation)
    db.refresh(rollback)

    events: list[dict[str, Any]] = [
        {"type": "trace", "message": "Confirmation approved; restoring the requested version..."},
        {
            "type": "final_answer",
            "answer": f"Applied: {confirmation.operation_summary}. The rollback was saved as a new version.",
            "state_changed": True,
        },
        {"type": "message_done"},
    ]
    return ConfirmationActionResponse(
        confirmation=_confirmation_read(confirmation),
        events=events,
        result={
            "updated_datasets": [
                {
                    "dataset_id": dataset.id,
                    "version_id": rollback.id,
                    "mutation_summary": confirmation.operation_summary,
                }
            ]
        },
    )


def _approve_direct_mutation_confirmation(
    confirmation: PendingConfirmation,
    db: Session,
) -> ConfirmationActionResponse:
    dataset = db.get(Dataset, confirmation.active_dataset_id or "")
    session = db.get(AnalysisSession, confirmation.session_id)
    branch = db.exec(
        select(Branch)
        .where(Branch.session_id == confirmation.session_id)
        .where(Branch.name == confirmation.branch_name)
    ).first()
    if dataset is None or session is None or branch is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Confirmation target not found")

    try:
        current_value = load_pickle(Path(dataset.current_snapshot_path))
        operation_kind = str(confirmation.tool_arguments.get("operation_kind"))
        if operation_kind == "delete_last_n":
            new_value, preview = _delete_last_n_value(current_value, int(confirmation.tool_arguments.get("delete_count") or 0))
        elif operation_kind == "delete_empty_title":
            new_value, preview = _delete_empty_title_value(current_value)
        else:
            raise ValueError(f"Unsupported direct mutation: {operation_kind}")

        updated_dataset = _persist_direct_mutation(
            db,
            session=session,
            branch=branch,
            dataset=dataset,
            value=new_value,
            mutation_summary=confirmation.operation_summary,
            created_by_message_id=confirmation.id,
        )
        confirmation.status = "approved"
        confirmation.resolved_at = utc_now()
        db.add(confirmation)
        db.commit()
        db.refresh(confirmation)

        result_payload = {
            "ok": True,
            "stdout": "",
            "stderr": "",
            "traceback": None,
            "result_preview": preview,
            "updated_datasets": [updated_dataset],
        }
        events = [
            {"type": "trace", "message": "Confirmation approved; applying the optimized dataset mutation..."},
            {
                "type": "code_result_summary",
                **result_payload,
            },
            {
                "type": "final_answer",
                "answer": _direct_mutation_answer(confirmation.operation_summary, preview),
                "state_changed": True,
            },
            {"type": "message_done"},
        ]
        return ConfirmationActionResponse(
            confirmation=_confirmation_read(confirmation),
            events=events,
            result=result_payload,
        )
    except Exception as exc:
        confirmation.status = "failed"
        confirmation.resolved_at = utc_now()
        db.add(confirmation)
        db.commit()
        db.refresh(confirmation)
        events = [
            {
                "type": "final_answer",
                "answer": (
                    f"I tried to apply '{confirmation.operation_summary}', but the optimized mutation failed: "
                    f"{exc}. No new dataset version was saved."
                ),
                "state_changed": False,
            },
            {"type": "message_done"},
        ]
        return ConfirmationActionResponse(
            confirmation=_confirmation_read(confirmation),
            events=events,
            result={"ok": False, "stderr": str(exc), "updated_datasets": []},
        )


def _delete_last_n_value(value: Any, delete_count: int) -> tuple[Any, dict[str, Any]]:
    current_count = _safe_len(value)
    count = max(0, min(delete_count, current_count))
    if isinstance(value, pd.DataFrame):
        new_value = value.iloc[:-count].copy() if count else value.copy()
    elif isinstance(value, list):
        new_value = value[:-count] if count else list(value)
    elif isinstance(value, tuple):
        new_value = value[:-count] if count else tuple(value)
    else:
        frame = pd.DataFrame([object_to_record(item) for item in value]) if _is_iterable_records(value) else None
        if frame is None:
            raise ValueError("This object type does not support deleting the last records safely")
        new_value = frame.iloc[:-count].copy() if count else frame.copy()
    return new_value, {
        "full_scan": True,
        "deleted_count": count,
        "affected_count": count,
        "current_row_count": current_count,
        "new_row_count": _safe_len(new_value),
    }


def _delete_empty_title_value(value: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        current_count = int(len(value))
        if "title" not in value.columns:
            raise ValueError("No title field exists on this dataset")
        mask = ~(value["title"].isna() | value["title"].astype(str).str.strip().eq(""))
        affected_count = current_count - int(mask.sum())
        new_value = value.loc[mask].copy()
    elif isinstance(value, list):
        current_count = len(value)
        keep_mask = [not _empty_title_from_item(item) for item in value]
        affected_count = current_count - sum(keep_mask)
        new_value = [item for item, keep in zip(value, keep_mask, strict=False) if keep]
    elif isinstance(value, tuple):
        current_count = len(value)
        keep_mask = [not _empty_title_from_item(item) for item in value]
        affected_count = current_count - sum(keep_mask)
        new_value = tuple(item for item, keep in zip(value, keep_mask, strict=False) if keep)
    else:
        raise ValueError("This object type does not support title-based deletion safely")

    return new_value, {
        "full_scan": True,
        "affected_count": int(affected_count),
        "current_row_count": int(current_count),
        "new_row_count": _safe_len(new_value),
    }


def _empty_title_from_item(item: Any) -> bool:
    record = object_to_record(item)
    value = record.get("title")
    return value is None or (isinstance(value, str) and value.strip() == "")


def _safe_len(value: Any) -> int:
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        return 1


def _is_iterable_records(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _persist_direct_mutation(
    db: Session,
    *,
    session: AnalysisSession,
    branch: Branch,
    dataset: Dataset,
    value: Any,
    mutation_summary: str,
    created_by_message_id: str,
) -> dict[str, Any]:
    version_id = new_id()
    snapshot_path = save_snapshot(session.id, dataset.id, version_id, value)
    profile = introspect_object(value)
    profile["_mutation"] = {"summary": mutation_summary}
    branch_versions = latest_versions_for_branch(branch.id, db)
    version = VersionNode(
        id=version_id,
        dataset_id=dataset.id,
        branch_id=branch.id,
        parent_version_id=(branch_versions.get(dataset.id).id if branch_versions.get(dataset.id) else dataset.current_version_id),
        label="confirmed mutation",
        snapshot_path=str(snapshot_path),
        mutation_summary=mutation_summary,
        created_by_message_id=created_by_message_id,
        profile=profile,
    )
    sync_branch_pointer(branch, version)
    dataset.current_version_id = version_id
    dataset.current_snapshot_path = str(snapshot_path)
    dataset.profile = profile
    dataset.object_type = profile.get("object_type", type(value).__qualname__)
    dataset.module = profile.get("module")
    dataset.updated_at = utc_now()
    session.updated_at = utc_now()
    db.add(version)
    db.add(branch)
    db.add(dataset)
    db.add(session)
    db.commit()
    db.refresh(dataset)
    return {
        "dataset_id": dataset.id,
        "key": dataset.dataset_key,
        "version_id": version_id,
        "profile": profile,
        "mutation_summary": mutation_summary,
    }


def _direct_mutation_answer(summary: str, preview: dict[str, Any]) -> str:
    current_count = preview.get("current_row_count")
    new_count = preview.get("new_row_count")
    affected_count = preview.get("affected_count") or preview.get("deleted_count")
    if isinstance(current_count, int) and isinstance(new_count, int):
        return (
            f"Applied: {summary}. Affected records: {affected_count}. "
            f"Row count changed from {current_count:,} to {new_count:,}.\n\n"
            "**State changed:** Yes"
        )
    return f"Applied: {summary}. Saved a new dataset version.\n\n**State changed:** Yes"


def _code_result_event(result: ExecutionResult) -> dict[str, Any]:
    return {
        "type": "code_result_summary",
        "ok": result.ok,
        "stdout": result.stdout[:1000],
        "stderr": result.stderr[:1000],
        "traceback": _short_traceback(result.traceback),
        "result_preview": result.result_preview,
        "updated_datasets": [item.model_dump(mode="json") for item in result.updated_datasets],
    }


def _final_answer_event(
    confirmation: PendingConfirmation,
    result: ExecutionResult,
    model_client: OpenAIResponsesClient,
) -> dict[str, Any]:
    model_answer = _model_final_answer(confirmation, result, model_client)
    if model_answer:
        return {
            "type": "final_answer",
            "answer": model_answer,
            "state_changed": bool(result.updated_datasets),
        }

    if result.ok:
        version_count = len(result.updated_datasets)
        return {
            "type": "final_answer",
            "answer": (
                f"Applied: {confirmation.operation_summary}. "
                f"Saved {version_count} new dataset version{'s' if version_count != 1 else ''}."
            ),
            "state_changed": bool(result.updated_datasets),
        }

    return {
        "type": "final_answer",
        "answer": (
            f"I tried to apply '{confirmation.operation_summary}', but the Python execution failed. "
            "No new dataset version was saved."
        ),
        "state_changed": False,
    }


def _model_final_answer(
    confirmation: PendingConfirmation,
    result: ExecutionResult,
    model_client: OpenAIResponsesClient,
) -> str | None:
    if not confirmation.model_input_items or not confirmation.tool_call_id:
        return None

    input_items = list(confirmation.model_input_items)
    input_items.append(
        {
            "type": "function_call_output",
            "call_id": confirmation.tool_call_id,
            "output": json.dumps(execution_result_for_model(result), default=str),
        }
    )
    try:
        response = model_client.create_response(
            instructions=SYSTEM_PROMPT,
            input_items=input_items,
            tools=AGENT_TOOLS,
        )
    except Exception:
        return None

    for tool_call in response.tool_calls:
        if tool_call.name == "final_answer":
            answer = tool_call.arguments.get("answer")
            return str(answer) if answer else None

    return response.final_text


def _short_traceback(traceback: str | None, limit: int = 1200) -> str | None:
    if traceback is None:
        return None
    if len(traceback) <= limit:
        return traceback
    return traceback[-limit:]
