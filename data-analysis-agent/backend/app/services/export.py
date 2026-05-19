from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.models.entities import AnalysisSession, Artifact, Dataset, VersionNode
from app.services.artifacts import persist_artifact
from app.storage.files import load_pickle


@dataclass
class ExportResult:
    artifact: Artifact | None
    message: str
    ok: bool


class ExportConversionError(ValueError):
    pass


def export_dataset_csv(
    db: Session,
    *,
    session_id: str,
    dataset_id: str | None,
    version_id: str | None,
    name: str | None,
) -> ExportResult:
    session = db.get(AnalysisSession, session_id)
    if session is None:
        raise ValueError("Session not found")

    dataset = _resolve_dataset(db, session, dataset_id)
    version = _resolve_version(db, dataset, version_id)
    value = load_pickle(Path(version.snapshot_path))

    try:
        frame = object_to_dataframe(value)
    except ExportConversionError as exc:
        json_artifact = _persist_json_fallback(db, session_id, dataset, version, value, name)
        return ExportResult(
            artifact=json_artifact,
            message=f"{exc}. I saved a JSON artifact internally, but could not create a useful CSV.",
            ok=False,
        )

    export_name = name or f"{dataset.original_filename}-export"
    artifact = persist_artifact(
        db,
        session_id=session_id,
        dataset_id=dataset.id,
        version_id=version.id,
        name=export_name,
        kind="csv",
        content=frame.to_csv(index=False),
        metadata={
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns.tolist()],
            "source": "current_dataset" if version_id is None else "version",
            "dataset_id": dataset.id,
            "dataset_filename": dataset.original_filename,
            "version_id": version.id,
        },
    )
    return ExportResult(
        artifact=artifact,
        message=f"Exported {len(frame)} rows from {dataset.original_filename}.",
        ok=True,
    )


def object_to_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        name = value.name if value.name is not None else "value"
        return value.rename(name).reset_index()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return pd.DataFrame({"value": [value.item()]})
        if value.ndim == 1:
            return pd.DataFrame({"value": value.tolist()})
        if value.ndim == 2:
            return pd.DataFrame(value)
        flat = value.reshape((value.shape[0], -1))
        columns = [f"value_{index}" for index in range(flat.shape[1])]
        return pd.DataFrame(flat, columns=columns)

    if isinstance(value, Mapping):
        return _mapping_to_dataframe(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _sequence_to_dataframe(value)

    attrs = _public_attrs(value)
    if attrs:
        return pd.json_normalize([attrs])

    raise ExportConversionError(f"Object of type {type(value).__qualname__} is not tabular")


def _mapping_to_dataframe(value: Mapping[Any, Any]) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()

    if all(_is_scalar(item) for item in value.values()):
        return pd.DataFrame([{str(key): _json_safe(item) for key, item in value.items()}])

    if all(isinstance(item, Mapping) for item in value.values()):
        rows = []
        for key, item in value.items():
            row = {"key": str(key)}
            row.update({str(child_key): _json_safe(child_value) for child_key, child_value in item.items()})
            rows.append(row)
        return pd.json_normalize(rows)

    if all(isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) for item in value.values()):
        try:
            return pd.DataFrame(value)
        except ValueError:
            pass

    return pd.json_normalize(_json_safe(value))


def _sequence_to_dataframe(value: Sequence[Any]) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()

    if all(isinstance(item, Mapping) for item in value):
        return pd.json_normalize([_json_safe(item) for item in value])

    if all(_is_scalar(item) for item in value):
        return pd.DataFrame({"value": list(value)})

    normalized_items = []
    for item in value:
        if isinstance(item, Mapping):
            normalized_items.append(_json_safe(item))
        else:
            attrs = _public_attrs(item)
            normalized_items.append(attrs if attrs else {"value": _json_safe(item)})

    return pd.json_normalize(normalized_items)


def _resolve_dataset(db: Session, session: AnalysisSession, dataset_id: str | None) -> Dataset:
    if dataset_id:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None or dataset.session_id != session.id:
            raise ValueError("Dataset not found")
        return dataset

    if session.active_dataset_id:
        dataset = db.get(Dataset, session.active_dataset_id)
        if dataset is not None and dataset.session_id == session.id:
            return dataset

    dataset = db.exec(select(Dataset).where(Dataset.session_id == session.id)).first()
    if dataset is None:
        raise ValueError("No dataset available to export")
    return dataset


def _resolve_version(db: Session, dataset: Dataset, version_id: str | None) -> VersionNode:
    resolved_id = version_id or dataset.current_version_id
    if resolved_id is None:
        raise ValueError("Dataset has no current version")

    version = db.get(VersionNode, resolved_id)
    if version is None or version.dataset_id != dataset.id:
        raise ValueError("Version not found")
    return version


def _persist_json_fallback(
    db: Session,
    session_id: str,
    dataset: Dataset,
    version: VersionNode,
    value: Any,
    name: str | None,
) -> Artifact:
    payload = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, default=str)
    return persist_artifact(
        db,
        session_id=session_id,
        dataset_id=dataset.id,
        version_id=version.id,
        name=name or f"{dataset.original_filename}-json-fallback",
        kind="json",
        content=payload,
        metadata={
            "source": "csv_conversion_fallback",
            "dataset_id": dataset.id,
            "dataset_filename": dataset.original_filename,
            "version_id": version.id,
        },
    )


def _public_attrs(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return {}
    try:
        attrs = vars(value)
    except TypeError:
        return {}
    return {
        str(key): _json_safe(item)
        for key, item in attrs.items()
        if not str(key).startswith("_") and not callable(item)
    }


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bytes, bytearray, int, float, bool, np.generic))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if _is_scalar(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    attrs = _public_attrs(value)
    return attrs if attrs else repr(value)
