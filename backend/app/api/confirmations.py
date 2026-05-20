from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlmodel import Session, select

from app.agent.agent import OpenAIResponsesClient
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import AGENT_TOOLS, execution_result_for_model
from app.api.chat import get_model_client
from app.db.session import get_session
from app.models.entities import AnalysisSession, Artifact, Branch, ChatMessage, Dataset, PendingConfirmation, VersionNode, new_id, utc_now
from app.runtime.python_executor import ExecutionResult, PythonExecutor, fast_get_field, object_to_record
from app.schemas.confirmation import ConfirmationActionResponse, ConfirmationRead
from app.services.introspection import introspect_object
from app.services.mutation_intents import normalize_country_value, parse_country_filter_mutation
from app.services.optimized_mutations import MutationSpec, apply_mutation_spec
from app.services.optimized_mutations import parse_mutation_request
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
    payload: dict[str, Any] = Body(default_factory=dict),
) -> ConfirmationActionResponse:
    confirmation = _pending_confirmation_or_404(session_id, confirmation_id, db)
    inferred_direct_args = _infer_direct_mutation_arguments(confirmation, db)
    if inferred_direct_args:
        confirmation.tool_arguments = {**dict(confirmation.tool_arguments or {}), **inferred_direct_args}
        db.add(confirmation)
        db.commit()
        db.refresh(confirmation)
    if confirmation.tool_arguments.get("operation_kind") == "rollback":
        return _approve_rollback_confirmation(confirmation, db)
    if confirmation.tool_arguments.get("operation_kind") in {
        "delete_first_n",
        "delete_last_n",
        "delete_empty_title",
        "delete_all_records",
        "add_filing_year",
        "filter_by_field",
        "remove_battery_below",
        "mutation_spec",
    }:
        _validate_required_phrase(confirmation, payload)
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
    _persist_confirmation_events(db, confirmation, events)
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
    _persist_confirmation_events(db, confirmation, events)
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


def _infer_direct_mutation_arguments(confirmation: PendingConfirmation, db: Session) -> dict[str, Any] | None:
    existing_kind = confirmation.tool_arguments.get("operation_kind")
    if isinstance(existing_kind, str) and existing_kind:
        return None

    text = " ".join(
        str(value or "")
        for value in (
            confirmation.original_message,
            confirmation.operation_summary,
            confirmation.proposed_code,
        )
    ).lower()

    if "filing_year" in text and "filing_date" in text:
        return {
            "operation_kind": "add_filing_year",
            "mutation_summary": confirmation.operation_summary
            or "Add derived field `filing_year` based on `filing_date`",
        }

    country_filter = parse_country_filter_mutation(text)
    if country_filter is not None:
        keep_value = country_filter.keep_value
        return {
            "operation_kind": "filter_by_field",
            "field": "country",
            "operator": "eq",
            "keep_value": keep_value,
            "delete_inverse": True,
            "mutation_summary": confirmation.operation_summary
            or f"Delete all non-{keep_value} records and keep only {keep_value} records",
        }

    if confirmation.active_dataset_id:
        dataset = db.get(Dataset, confirmation.active_dataset_id)
        if dataset is not None:
            try:
                value = load_pickle(Path(dataset.current_snapshot_path))
                outcome = parse_mutation_request(
                    confirmation.original_message or confirmation.operation_summary or text,
                    value,
                    target_dataset_id=dataset.id,
                    target_dataset_name=dataset.dataset_key or dataset.original_filename,
                )
            except Exception:
                outcome = None
            if outcome is not None and outcome.spec is not None:
                return {
                    "operation_kind": "mutation_spec",
                    "mutation_spec": outcome.spec.to_dict(),
                    "mutation_summary": outcome.spec.human_summary or confirmation.operation_summary,
                }

    first_match = re.search(r"\b(?:delete|remove|drop)\s+first\s+(\d+)\s+(?:entries|rows|records)\b", text)
    if first_match:
        delete_count = int(first_match.group(1))
        return {
            "operation_kind": "delete_first_n",
            "delete_count": delete_count,
            "mutation_summary": confirmation.operation_summary
            or f"Delete the first {delete_count:,} records from the current working dataset",
        }

    last_match = re.search(r"\b(?:delete|remove|drop)\s+last\s+(\d+)\s+(?:entries|rows|records)\b", text)
    if last_match:
        delete_count = int(last_match.group(1))
        return {
            "operation_kind": "delete_last_n",
            "delete_count": delete_count,
            "mutation_summary": confirmation.operation_summary
            or f"Delete the last {delete_count:,} records from the current working dataset",
        }

    return None


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
    _persist_confirmation_events(db, confirmation, events)
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
        if operation_kind == "delete_first_n":
            new_value, preview = _delete_first_n_value(current_value, int(confirmation.tool_arguments.get("delete_count") or 0))
        elif operation_kind == "delete_last_n":
            new_value, preview = _delete_last_n_value(current_value, int(confirmation.tool_arguments.get("delete_count") or 0))
        elif operation_kind == "delete_empty_title":
            new_value, preview = _delete_empty_title_value(current_value)
        elif operation_kind == "delete_all_records":
            new_value, preview = _delete_all_records_value(current_value)
        elif operation_kind == "add_filing_year":
            new_value, preview = _add_filing_year_value(current_value)
        elif operation_kind == "filter_by_field":
            new_value, preview = _filter_by_field_value(
                current_value,
                str(confirmation.tool_arguments.get("field") or ""),
                str(confirmation.tool_arguments.get("keep_value") or ""),
            )
        elif operation_kind == "remove_battery_below":
            new_value, preview = _remove_battery_below_value(
                current_value,
                float(confirmation.tool_arguments.get("threshold") or 0),
            )
        elif operation_kind == "mutation_spec":
            spec_payload = confirmation.tool_arguments.get("mutation_spec")
            if not isinstance(spec_payload, Mapping):
                raise ValueError("Missing optimized mutation spec")
            new_value, preview = apply_mutation_spec(current_value, MutationSpec.from_mapping(spec_payload))
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
        _persist_confirmation_events(db, confirmation, events)
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
        _persist_confirmation_events(db, confirmation, events)
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


def _delete_first_n_value(value: Any, delete_count: int) -> tuple[Any, dict[str, Any]]:
    current_count = _safe_len(value)
    count = max(0, min(delete_count, current_count))
    if isinstance(value, pd.DataFrame):
        new_value = value.iloc[count:].copy() if count else value.copy()
    elif isinstance(value, list):
        new_value = value[count:] if count else list(value)
    elif isinstance(value, tuple):
        new_value = value[count:] if count else tuple(value)
    else:
        frame = pd.DataFrame([object_to_record(item) for item in value]) if _is_iterable_records(value) else None
        if frame is None:
            raise ValueError("This object type does not support deleting the first records safely")
        new_value = frame.iloc[count:].copy() if count else frame.copy()
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


def _delete_all_records_value(value: Any) -> tuple[Any, dict[str, Any]]:
    current_count = _safe_len(value)
    if isinstance(value, pd.DataFrame):
        new_value = value.iloc[0:0].copy()
    elif isinstance(value, list):
        new_value = []
    elif isinstance(value, tuple):
        new_value = tuple()
    else:
        raise ValueError("This object type does not support deleting all records safely")
    return new_value, {
        "full_scan": True,
        "affected_count": int(current_count),
        "deleted_count": int(current_count),
        "current_row_count": int(current_count),
        "new_row_count": 0,
    }


def _add_filing_year_value(value: Any) -> tuple[Any, dict[str, Any]]:
    def filing_year(item: Any) -> int | None:
        record = object_to_record(item)
        raw = record.get("filing_date")
        if raw is None:
            return None
        if hasattr(raw, "year"):
            try:
                return int(raw.year)
            except (TypeError, ValueError):
                return None
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return None
        return int(parsed.year)

    if isinstance(value, pd.DataFrame):
        current_count = int(len(value))
        if "filing_date" not in value.columns:
            raise ValueError("No filing_date field exists on this dataset")
        new_value = value.copy()
        new_value["filing_year"] = pd.to_datetime(new_value["filing_date"], errors="coerce").dt.year.astype("Int64")
        derived_count = int(new_value["filing_year"].notna().sum())
    elif isinstance(value, list):
        current_count = len(value)
        new_value = [_copy_with_field(item, "filing_year", filing_year(item)) for item in value]
        derived_count = sum(1 for item in new_value if object_to_record(item).get("filing_year") is not None)
    elif isinstance(value, tuple):
        current_count = len(value)
        new_value = tuple(_copy_with_field(item, "filing_year", filing_year(item)) for item in value)
        derived_count = sum(1 for item in new_value if object_to_record(item).get("filing_year") is not None)
    else:
        raise ValueError("This object type does not support adding filing_year safely")

    return new_value, {
        "full_scan": True,
        "derived_field": "filing_year",
        "derived_non_null_count": int(derived_count),
        "affected_count": int(current_count),
        "current_row_count": int(current_count),
        "new_row_count": _safe_len(new_value),
    }


def _copy_with_field(item: Any, field: str, value: Any) -> Any:
    if isinstance(item, dict):
        copied = dict(item)
        copied[field] = value
        return copied
    try:
        import copy

        copied = copy.copy(item)
        setattr(copied, field, value)
        return copied
    except Exception:
        record = object_to_record(item)
        record[field] = value
    return record


def _filter_by_field_value(value: Any, field: str, keep_value: str) -> tuple[Any, dict[str, Any]]:
    if field != "country":
        raise ValueError(f"Unsupported optimized filter field: {field}")

    def keep(item: Any) -> bool:
        return normalize_country_value(fast_get_field(item, field)) == keep_value

    if isinstance(value, pd.DataFrame):
        current_count = int(len(value))
        if field not in value.columns:
            raise ValueError(f"No {field} field exists on this dataset")
        normalized = value[field].map(normalize_country_value)
        keep_mask = normalized.eq(keep_value)
        removed_counts = normalized.loc[~keep_mask].fillna("Unknown").value_counts(dropna=False).to_dict()
        new_value = value.loc[keep_mask].copy()
        affected_count = current_count - int(keep_mask.sum())
    elif isinstance(value, list):
        current_count = len(value)
        kept_items = []
        removed_counts: dict[str, int] = {}
        for item in value:
            normalized = normalize_country_value(fast_get_field(item, field))
            if normalized == keep_value:
                kept_items.append(item)
            else:
                key = normalized or "Unknown"
                removed_counts[key] = removed_counts.get(key, 0) + 1
        new_value = kept_items
        affected_count = current_count - len(kept_items)
    elif isinstance(value, tuple):
        current_count = len(value)
        kept_items = []
        removed_counts = {}
        for item in value:
            normalized = normalize_country_value(fast_get_field(item, field))
            if normalized == keep_value:
                kept_items.append(item)
            else:
                key = normalized or "Unknown"
                removed_counts[key] = removed_counts.get(key, 0) + 1
        new_value = tuple(kept_items)
        affected_count = current_count - len(kept_items)
    else:
        raise ValueError("This object type does not support optimized field filtering safely")

    return new_value, {
        "full_scan": True,
        "field": field,
        "keep_value": keep_value,
        "delete_inverse": True,
        "affected_count": int(affected_count),
        "current_row_count": int(current_count),
        "new_row_count": _safe_len(new_value),
        "removed_value_counts": {str(key): int(count) for key, count in removed_counts.items()},
    }


def _remove_battery_below_value(value: Any, threshold: float) -> tuple[Any, dict[str, Any]]:
    def keep(item: Any) -> bool:
        record = object_to_record(item)
        try:
            return float(record.get("battery_pct")) >= threshold
        except (TypeError, ValueError):
            return True

    if isinstance(value, pd.DataFrame):
        current_count = int(len(value))
        if "battery_pct" not in value.columns:
            raise ValueError("No battery_pct field exists on this dataset")
        values = pd.to_numeric(value["battery_pct"], errors="coerce")
        mask = values.isna() | (values >= threshold)
        affected_count = current_count - int(mask.sum())
        new_value = value.loc[mask].copy()
    elif hasattr(value, "readings"):
        readings = list(getattr(value, "readings"))
        current_count = len(readings)
        kept = [item for item in readings if keep(item)]
        affected_count = current_count - len(kept)
        setattr(value, "readings", kept)
        new_value = value
    elif isinstance(value, list):
        current_count = len(value)
        new_value = [item for item in value if keep(item)]
        affected_count = current_count - len(new_value)
    elif isinstance(value, tuple):
        current_count = len(value)
        new_value = tuple(item for item in value if keep(item))
        affected_count = current_count - len(new_value)
    else:
        raise ValueError("This object type does not support battery-based filtering safely")

    return new_value, {
        "full_scan": True,
        "affected_count": int(affected_count),
        "current_row_count": int(current_count),
        "new_row_count": _safe_len(getattr(new_value, "readings", new_value)),
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


def _validate_required_phrase(confirmation: PendingConfirmation, payload: dict[str, Any]) -> None:
    phrase = confirmation.tool_arguments.get("required_confirmation_phrase")
    if not isinstance(phrase, str) or not phrase.strip():
        return
    supplied = payload.get("confirmation_phrase") or payload.get("phrase")
    if str(supplied or "").strip().lower() != phrase.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"This high-risk mutation requires the confirmation phrase: {phrase}",
        )


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
        if current_count == new_count:
            return (
                f"Applied: {summary}. Updated records: {affected_count}. "
                f"Row count remains {new_count:,}.\n\n"
                "**State changed:** Yes"
            )
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


def _persist_confirmation_events(
    db: Session,
    confirmation: PendingConfirmation,
    events: list[dict[str, Any]],
) -> None:
    assistant_message_id = (confirmation.tool_arguments or {}).get("assistant_message_id")
    if not isinstance(assistant_message_id, str):
        return
    message = db.get(ChatMessage, assistant_message_id)
    if message is None:
        return

    trace_events = list(message.trace_events or [])
    artifact_ids = [item for item in (message.artifact_ids or []) if isinstance(item, str)]
    for event in events:
        event_type = str(event.get("type", "message"))
        if event_type == "artifact_created":
            artifact_payload = event.get("artifact")
            artifact_id = artifact_payload.get("id") if isinstance(artifact_payload, dict) else None
            if isinstance(artifact_id, str) and artifact_id not in artifact_ids:
                artifact_ids.append(artifact_id)
                artifact = db.get(Artifact, artifact_id)
                if artifact is not None:
                    metadata = dict(artifact.artifact_metadata or {})
                    metadata["source_message_id"] = message.id
                    artifact.artifact_metadata = metadata
                    db.add(artifact)
        elif event_type == "final_answer":
            message.status = "done"
            message.final_answer = str(event.get("answer") or "")
            message.state_changed = bool(event.get("state_changed"))
            message.pending_action = None
        elif event_type == "message_done":
            if message.status == "streaming":
                message.status = "done"
        elif event_type in {"trace", "code_started", "code_result_summary", "error"}:
            trace_events.append(
                {
                    "id": new_id(),
                    "type": event_type,
                    **{
                        key: value
                        for key, value in event.items()
                        if key
                        in {
                            "message",
                            "code",
                            "ok",
                            "stdout",
                            "stderr",
                            "traceback",
                            "result_preview",
                            "updated_datasets",
                        }
                    },
                }
            )

    message.artifact_ids = artifact_ids
    message.trace_events = trace_events
    message.updated_at = utc_now()
    session = db.get(AnalysisSession, confirmation.session_id)
    if session is not None:
        session.updated_at = utc_now()
        db.add(session)
    db.add(message)
    db.commit()


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
