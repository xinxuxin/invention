from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import multiprocessing as mp
import socket
import statistics
import traceback as traceback_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.models.entities import AnalysisSession, Artifact, Branch, Dataset, VersionNode, new_id, utc_now
from app.services.introspection import introspect_object
from app.storage.files import artifacts_root, load_pickle, save_snapshot

DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_MAX_STDOUT = 20_000
DEFAULT_MAX_PREVIEW = 12_000
MAX_ARTIFACT_BYTES = 2_000_000


class ExecutionArtifact(BaseModel):
    id: str
    name: str
    kind: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdatedDataset(BaseModel):
    dataset_id: str
    key: str
    version_id: str
    profile: dict[str, Any]
    mutation_summary: str


class ExecutionResult(BaseModel):
    ok: bool
    stdout: str
    stderr: str
    traceback: str | None
    result_preview: dict[str, Any] | list[Any] | str | int | float | bool | None
    updated_datasets: list[UpdatedDataset] = Field(default_factory=list)
    artifacts: list[ExecutionArtifact] = Field(default_factory=list)


@dataclass
class LoadedDataset:
    row: Dataset
    key: str
    value: Any
    fingerprint: str


class PythonExecutor:
    def __init__(
        self,
        db: Session,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_stdout_length: int = DEFAULT_MAX_STDOUT,
        max_preview_length: int = DEFAULT_MAX_PREVIEW,
    ) -> None:
        self.db = db
        self.timeout_seconds = timeout_seconds
        self.max_stdout_length = max_stdout_length
        self.max_preview_length = max_preview_length

    def execute(
        self,
        session_id: str,
        code: str,
        *,
        active_dataset_id: str | None = None,
        branch_name: str = "main",
        mutates_state: bool = False,
        mutation_summary: str | None = None,
    ) -> ExecutionResult:
        analysis_session = self.db.get(AnalysisSession, session_id)
        if analysis_session is None:
            raise ValueError(f"Session not found: {session_id}")

        branch = self._get_branch(session_id, branch_name)
        loaded = self._load_session_datasets(session_id)
        active = self._resolve_active_dataset(loaded, active_dataset_id)

        child_payload = self._run_in_child(code, loaded, active)
        if child_payload.get("timed_out"):
            return ExecutionResult(
                ok=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds} seconds.",
                traceback=None,
                result_preview=None,
            )

        stdout = _truncate_text(child_payload.get("stdout", ""), self.max_stdout_length)
        stderr = _truncate_text(child_payload.get("stderr", ""), self.max_stdout_length)
        result_preview = _limit_preview(child_payload.get("result_preview"), self.max_preview_length)

        if not child_payload.get("ok", False):
            return ExecutionResult(
                ok=False,
                stdout=stdout,
                stderr=stderr,
                traceback=child_payload.get("traceback"),
                result_preview=result_preview,
            )

        artifacts = self._persist_artifacts(
            session_id=session_id,
            artifact_payloads=child_payload.get("artifacts", []),
        )
        updated_datasets: list[UpdatedDataset] = []
        if mutates_state:
            updated_datasets = self._persist_mutations(
                session=analysis_session,
                branch=branch,
                loaded=loaded,
                returned_datasets=child_payload.get("datasets", {}),
                returned_data=child_payload.get("data"),
                active=active,
                mutation_summary=mutation_summary or "Python execution mutation",
            )

        return ExecutionResult(
            ok=True,
            stdout=stdout,
            stderr=stderr,
            traceback=None,
            result_preview=result_preview,
            updated_datasets=updated_datasets,
            artifacts=artifacts,
        )

    def _get_branch(self, session_id: str, branch_name: str) -> Branch:
        branch = self.db.exec(
            select(Branch).where(Branch.session_id == session_id).where(Branch.name == branch_name)
        ).first()
        if branch is None:
            raise ValueError(f"Branch not found: {branch_name}")

        return branch

    def _load_session_datasets(self, session_id: str) -> list[LoadedDataset]:
        rows = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
        keys = _dataset_keys(rows)

        loaded: list[LoadedDataset] = []
        for row in rows:
            value = load_pickle(Path(row.current_snapshot_path))
            loaded.append(
                LoadedDataset(
                    row=row,
                    key=keys[row.id],
                    value=value,
                    fingerprint=_fingerprint(value),
                )
            )

        return loaded

    def _resolve_active_dataset(
        self,
        loaded: list[LoadedDataset],
        active_dataset_id: str | None,
    ) -> LoadedDataset | None:
        if not loaded:
            return None

        if active_dataset_id is None:
            return loaded[0]

        for dataset in loaded:
            if dataset.row.id == active_dataset_id:
                return dataset

        raise ValueError(f"Active dataset not found: {active_dataset_id}")

    def _run_in_child(
        self,
        code: str,
        loaded: list[LoadedDataset],
        active: LoadedDataset | None,
    ) -> dict[str, Any]:
        context = _mp_context()
        queue: mp.Queue[bytes] = context.Queue()
        datasets = {dataset.key: dataset.value for dataset in loaded}
        active_key = active.key if active else None

        process = context.Process(
            target=_child_execute,
            args=(queue, code, datasets, active_key, self.max_stdout_length, self.max_preview_length),
        )
        process.start()
        process.join(self.timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(1)
            return {"timed_out": True}

        try:
            payload_bytes = queue.get_nowait()
        except Empty:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "Execution process exited without returning a result.",
                "traceback": None,
                "result_preview": None,
            }

        return cloudpickle.loads(payload_bytes)

    def _persist_mutations(
        self,
        session: AnalysisSession,
        branch: Branch,
        loaded: list[LoadedDataset],
        returned_datasets: Mapping[str, Any],
        returned_data: Any,
        active: LoadedDataset | None,
        mutation_summary: str,
    ) -> list[UpdatedDataset]:
        updated: list[UpdatedDataset] = []
        values_by_dataset_id: dict[str, tuple[str, Any]] = {}

        for dataset in loaded:
            if dataset.key in returned_datasets:
                values_by_dataset_id[dataset.row.id] = (dataset.key, returned_datasets[dataset.key])

        if (
            active is not None
            and returned_data is not None
            and _fingerprint(returned_data) != active.fingerprint
        ):
            values_by_dataset_id[active.row.id] = (active.key, returned_data)

        for dataset in loaded:
            candidate = values_by_dataset_id.get(dataset.row.id)
            if candidate is None:
                continue

            key, value = candidate
            if _fingerprint(value) == dataset.fingerprint:
                continue

            version_id = new_id()
            snapshot_path = save_snapshot(session.id, dataset.row.id, version_id, value)
            profile = introspect_object(value)
            profile["_mutation"] = {"summary": mutation_summary}

            version = VersionNode(
                id=version_id,
                dataset_id=dataset.row.id,
                branch_id=branch.id,
                parent_id=dataset.row.current_version_id,
                label="python execution",
                snapshot_path=str(snapshot_path),
                profile=profile,
            )
            dataset.row.current_version_id = version_id
            dataset.row.current_snapshot_path = str(snapshot_path)
            dataset.row.profile = profile
            dataset.row.object_type = profile.get("object_type", type(value).__qualname__)
            dataset.row.module = profile.get("module")
            dataset.row.updated_at = utc_now()
            session.updated_at = utc_now()

            self.db.add(version)
            self.db.add(dataset.row)
            self.db.add(session)
            updated.append(
                UpdatedDataset(
                    dataset_id=dataset.row.id,
                    key=key,
                    version_id=version_id,
                    profile=profile,
                    mutation_summary=mutation_summary,
                )
            )

        if updated:
            self.db.commit()

        return updated

    def _persist_artifacts(
        self,
        session_id: str,
        artifact_payloads: Sequence[Mapping[str, Any]],
    ) -> list[ExecutionArtifact]:
        persisted: list[ExecutionArtifact] = []
        root = artifacts_root(session_id)

        for payload in artifact_payloads:
            artifact_id = new_id()
            name = str(payload.get("name") or f"artifact-{artifact_id}")
            kind = str(payload.get("kind") or "artifact")
            extension = _artifact_extension(kind)
            path = root / f"{artifact_id}-{_safe_artifact_name(name)}.{extension}"
            content = payload.get("content", "")
            if isinstance(content, bytes):
                path.write_bytes(content[:MAX_ARTIFACT_BYTES])
            else:
                path.write_text(str(content)[:MAX_ARTIFACT_BYTES], encoding="utf-8")

            metadata = _json_safe(payload.get("metadata", {}))
            artifact = Artifact(
                id=artifact_id,
                session_id=session_id,
                name=name,
                kind=kind,
                path=str(path),
                artifact_metadata=metadata if isinstance(metadata, dict) else {"metadata": metadata},
            )
            self.db.add(artifact)
            persisted.append(
                ExecutionArtifact(
                    id=artifact.id,
                    name=artifact.name,
                    kind=artifact.kind,
                    path=artifact.path,
                    metadata=artifact.artifact_metadata,
                )
            )

        if persisted:
            self.db.commit()

        return persisted


def _child_execute(
    queue: mp.Queue[bytes],
    code: str,
    datasets: dict[str, Any],
    active_key: str | None,
    max_stdout_length: int,
    max_preview_length: int,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    artifacts: list[dict[str, Any]] = []
    preview_value: Any = None
    data = datasets.get(active_key) if active_key else None

    def preview(obj: Any) -> Any:
        nonlocal preview_value
        preview_value = obj
        return obj

    helpers = _artifact_helpers(artifacts)
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "datasets": datasets,
        "data": data,
        "pd": pd,
        "np": np,
        "json": json,
        "math": math,
        "statistics": statistics,
        "save_table": helpers["save_table"],
        "save_chart": helpers["save_chart"],
        "save_csv": helpers["save_csv"],
        "preview": preview,
    }

    _block_network()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "<agent_code>", "exec"), namespace)

        data = namespace.get("data")
        if preview_value is None and "_result" in namespace:
            preview_value = namespace["_result"]

        payload = {
            "ok": True,
            "stdout": _truncate_text(stdout.getvalue(), max_stdout_length),
            "stderr": _truncate_text(stderr.getvalue(), max_stdout_length),
            "traceback": None,
            "result_preview": _limit_preview(_preview(preview_value), max_preview_length),
            "datasets": namespace["datasets"],
            "data": data,
            "artifacts": artifacts,
        }
    except Exception:
        payload = {
            "ok": False,
            "stdout": _truncate_text(stdout.getvalue(), max_stdout_length),
            "stderr": _truncate_text(stderr.getvalue(), max_stdout_length),
            "traceback": traceback_module.format_exc(),
            "result_preview": _limit_preview(_preview(preview_value), max_preview_length),
            "datasets": datasets,
            "data": data,
            "artifacts": artifacts,
        }

    queue.put(cloudpickle.dumps(payload))


def _artifact_helpers(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    def save_table(name: str, dataframe_or_records: Any) -> dict[str, str]:
        records = _records_from_table(dataframe_or_records)
        content = json.dumps(records, ensure_ascii=False, indent=2)
        artifacts.append(
            {
                "kind": "table",
                "name": name,
                "content": content,
                "metadata": {
                    "rows": len(records),
                    "columns": list(records[0].keys()) if records else [],
                },
            }
        )
        return {"kind": "table", "name": name}

    def save_chart(name: str, chart_spec: Any) -> dict[str, str]:
        safe_spec = _json_safe(chart_spec)
        artifacts.append(
            {
                "kind": "chart",
                "name": name,
                "content": json.dumps(safe_spec, ensure_ascii=False, indent=2),
                "metadata": {
                    "spec_type": type(chart_spec).__qualname__,
                },
            }
        )
        return {"kind": "chart", "name": name}

    def save_csv(name: str, dataframe_or_records: Any) -> dict[str, str]:
        frame = _frame_from_records(dataframe_or_records)
        artifacts.append(
            {
                "kind": "csv",
                "name": name,
                "content": frame.to_csv(index=False),
                "metadata": {
                    "rows": int(len(frame)),
                    "columns": [str(column) for column in frame.columns.tolist()],
                },
            }
        )
        return {"kind": "csv", "name": name}

    return {"save_table": save_table, "save_chart": save_chart, "save_csv": save_csv}


def _safe_builtins() -> dict[str, Any]:
    blocked_imports = {
        "aiohttp",
        "ftplib",
        "http",
        "httpx",
        "requests",
        "socket",
        "subprocess",
        "telnetlib",
        "urllib",
        "webbrowser",
    }

    def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        root = name.split(".", maxsplit=1)[0]
        if root in blocked_imports:
            raise ImportError(f"Import of network/process module '{root}' is blocked in executor")
        return __import__(name, globals_, locals_, fromlist, level)

    safe_names = [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "repr",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "Exception",
        "ValueError",
        "RuntimeError",
        "TypeError",
        "KeyError",
        "IndexError",
    ]
    builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe = {name: builtins_dict[name] for name in safe_names}
    safe["__import__"] = guarded_import
    return safe


def _block_network() -> None:
    def blocked(*_: Any, **__: Any) -> None:
        raise PermissionError("Network access is blocked in the Python executor")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]


def _records_from_table(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))

    if isinstance(value, Mapping):
        return [_json_safe(dict(value))]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) if isinstance(item, Mapping) else {"value": _json_safe(item)} for item in value]

    return [{"value": _json_safe(value)}]


def _frame_from_records(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, pd.Series):
        return value.to_frame()

    if isinstance(value, Mapping):
        return pd.DataFrame([value])

    return pd.DataFrame(value)


def _preview(value: Any) -> Any:
    if value is None:
        return None

    try:
        return introspect_object(value)
    except Exception:
        return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(_json_safe(key)): _json_safe(item_value) for key, item_value in value.items()}

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_json_safe(item) for item in value]

    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


def _limit_preview(value: Any, max_length: int) -> Any:
    try:
        encoded = json.dumps(_json_safe(value), ensure_ascii=False)
    except Exception:
        encoded = json.dumps(str(value), ensure_ascii=False)

    if len(encoded) <= max_length:
        return _json_safe(value)

    return {
        "truncated": True,
        "preview": encoded[:max_length],
    }


def _truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value

    return f"{value[:max_length]}\n...[truncated]"


def _fingerprint(value: Any) -> str:
    try:
        payload = cloudpickle.dumps(value)
    except Exception:
        payload = repr(value).encode("utf-8", errors="replace")

    return hashlib.sha256(payload).hexdigest()


def _dataset_keys(rows: Sequence[Dataset]) -> dict[str, str]:
    names = [row.original_filename for row in rows]
    duplicate_names = {name for name in names if names.count(name) > 1}
    keys: dict[str, str] = {}
    used: set[str] = set()

    for row in rows:
        base = row.id if row.original_filename in duplicate_names else row.original_filename
        key = base
        if key in used:
            stem = Path(base).stem or "dataset"
            suffix = 2
            while f"{stem}-{suffix}" in used:
                suffix += 1
            key = f"{stem}-{suffix}"

        keys[row.id] = key
        used.add(key)

    return keys


def _artifact_extension(kind: str) -> str:
    if kind == "csv":
        return "csv"
    if kind in {"table", "chart"}:
        return "json"
    return "txt"


def _safe_artifact_name(name: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in name)
    return sanitized.strip("-")[:80] or "artifact"


def _mp_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - fork is unavailable on some platforms
        return mp.get_context()
