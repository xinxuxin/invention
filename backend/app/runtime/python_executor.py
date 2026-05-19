from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import socket
import statistics
import traceback as traceback_module
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from queue import Empty
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.models.entities import AnalysisSession, Artifact, Branch, Dataset, VersionNode, new_id, utc_now
from app.schemas.artifact import ChartArtifactSpec
from app.services.artifacts import persist_artifact
from app.services.introspection import introspect_object
from app.services.versioning import dataset_key, latest_versions_for_branch, sync_branch_pointer
from app.storage.files import load_pickle, save_snapshot

DEFAULT_TIMEOUT_SECONDS = 5
DEFAULT_MAX_STDOUT = 20_000
DEFAULT_MAX_PREVIEW = 12_000
MAX_CHART_ROWS = 500
MAX_CATEGORY_CHART_ROWS = 50
MAX_TABLE_INLINE_ROWS = 50
MAX_TABLE_COLUMNS = 30
MAX_TABLE_STORED_ROWS = 5_000
MAX_TABLE_CELL_LENGTH = 200
MAX_PREVIEW_ROWS = 20
MAX_PREVIEW_ITEMS = 50
MAX_PREVIEW_STRING_LENGTH = 1000
MAX_PREVIEW_DEPTH = 4
WRAPPER_KEYS = {
    "__dict__",
    "__pydantic_extra__",
    "__pydantic_fields_set__",
    "__pydantic_private__",
}


class ExecutionArtifact(BaseModel):
    id: str
    name: str
    kind: str
    type: str | None = None
    title: str | None = None
    description: str | None = None
    columns: list[dict[str, Any]] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    chart_spec: dict[str, Any] | None = None
    download_url: str | None = None
    source_message_id: str | None = None
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    status: str | None = None
    semantic_key: str | None = None


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
        created_by_message_id: str | None = None,
    ) -> ExecutionResult:
        analysis_session = self.db.get(AnalysisSession, session_id)
        if analysis_session is None:
            raise ValueError(f"Session not found: {session_id}")

        branch = self._get_branch(analysis_session, branch_name)
        loaded = self._load_session_datasets(session_id, branch)
        active = self._resolve_active_dataset(loaded, active_dataset_id or analysis_session.active_dataset_id)

        runtime_metadata = self._runtime_metadata(session_id, branch, loaded, active)
        child_payload = self._run_in_child(code, loaded, active, runtime_metadata, return_state=mutates_state)
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

        artifact_payloads = list(child_payload.get("artifacts", []))
        auto_table_payload = _auto_table_artifact_from_preview(result_preview, artifact_payloads)
        if auto_table_payload is not None:
            artifact_payloads.append(auto_table_payload)

        artifacts = self._persist_artifacts(
            session_id=session_id,
            artifact_payloads=artifact_payloads,
            source_message_id=created_by_message_id,
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
                created_by_message_id=created_by_message_id,
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

    def _get_branch(self, session: AnalysisSession, branch_name: str) -> Branch:
        branch = None
        if branch_name == "main" and session.active_branch_id:
            branch = self.db.get(Branch, session.active_branch_id)

        if branch is None:
            branch = self.db.exec(
                select(Branch).where(Branch.session_id == session.id).where(Branch.name == branch_name)
            ).first()

        if branch is None:
            raise ValueError(f"Branch not found: {branch_name}")

        return branch

    def _load_session_datasets(self, session_id: str, branch: Branch) -> list[LoadedDataset]:
        rows = list(self.db.exec(select(Dataset).where(Dataset.session_id == session_id)).all())
        keys = _dataset_keys(rows)
        branch_versions = latest_versions_for_branch(branch.id, self.db)

        loaded: list[LoadedDataset] = []
        for row in rows:
            version = branch_versions.get(row.id)
            snapshot_path = version.snapshot_path if version else row.current_snapshot_path
            value = load_pickle(Path(snapshot_path))
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
        runtime_metadata: dict[str, Any],
        return_state: bool,
    ) -> dict[str, Any]:
        context = _mp_context()
        queue: mp.Queue[bytes] = context.Queue()
        datasets = {dataset.key: dataset.value for dataset in loaded}
        dataset_profiles = {dataset.key: _json_safe(dataset.row.profile) for dataset in loaded}
        active_key = active.key if active else None
        active_dataset_profile = dataset_profiles.get(active_key) if active_key else None

        process = context.Process(
            target=_child_execute,
            args=(
                queue,
                code,
                datasets,
                dataset_profiles,
                active_dataset_profile,
                runtime_metadata,
                active_key,
                self.max_stdout_length,
                self.max_preview_length,
                return_state,
            ),
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

    def _runtime_metadata(
        self,
        session_id: str,
        branch: Branch,
        loaded: list[LoadedDataset],
        active: LoadedDataset | None,
    ) -> dict[str, Any]:
        dataset_ids = [dataset.row.id for dataset in loaded]
        versions = (
            list(
                self.db.exec(
                    select(VersionNode)
                    .where(VersionNode.dataset_id.in_(dataset_ids))
                    .order_by(VersionNode.created_at)
                ).all()
            )
            if dataset_ids
            else []
        )
        branches = list(self.db.exec(select(Branch).where(Branch.session_id == session_id)).all())
        artifacts = list(
            self.db.exec(select(Artifact).where(Artifact.session_id == session_id).order_by(Artifact.created_at)).all()
        )
        current = self.db.get(VersionNode, active.row.current_version_id) if active and active.row.current_version_id else None
        return _json_safe(
            {
                "artifact_history": [
                    {
                        "id": artifact.id,
                        "name": artifact.name,
                        "kind": artifact.kind,
                        "metadata": artifact.artifact_metadata,
                        "created_at": artifact.created_at.isoformat(),
                    }
                    for artifact in artifacts[-50:]
                ],
                "mutation_history": [
                    {
                        "version_id": version.id,
                        "dataset_id": version.dataset_id,
                        "branch_id": version.branch_id,
                        "summary": version.mutation_summary or version.label,
                        "created_at": version.created_at.isoformat(),
                        "row_count": _profile_row_count(version.profile),
                    }
                    for version in versions[-50:]
                ],
                "branch_history": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "current_version_id": item.current_version_id,
                        "root_version_id": item.root_version_id,
                        "created_at": item.created_at.isoformat(),
                    }
                    for item in branches
                ],
                "current_branch": {
                    "id": branch.id,
                    "name": branch.name,
                    "current_version_id": branch.current_version_id,
                    "root_version_id": branch.root_version_id,
                },
                "current_version": _version_metadata(current),
            }
        )

    def _persist_mutations(
        self,
        session: AnalysisSession,
        branch: Branch,
        loaded: list[LoadedDataset],
        returned_datasets: Mapping[str, Any],
        returned_data: Any,
        active: LoadedDataset | None,
        mutation_summary: str,
        created_by_message_id: str | None,
    ) -> list[UpdatedDataset]:
        updated: list[UpdatedDataset] = []
        values_by_dataset_id: dict[str, tuple[str, Any]] = {}
        branch_versions = latest_versions_for_branch(branch.id, self.db)

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
                parent_version_id=(
                    branch_versions.get(dataset.row.id).id
                    if branch_versions.get(dataset.row.id)
                    else dataset.row.current_version_id
                ),
                label="python execution",
                snapshot_path=str(snapshot_path),
                mutation_summary=mutation_summary,
                created_by_message_id=created_by_message_id,
                profile=profile,
            )
            sync_branch_pointer(branch, version)
            dataset.row.current_version_id = version_id
            dataset.row.current_snapshot_path = str(snapshot_path)
            dataset.row.profile = profile
            dataset.row.object_type = profile.get("object_type", type(value).__qualname__)
            dataset.row.module = profile.get("module")
            dataset.row.updated_at = utc_now()
            session.updated_at = utc_now()

            self.db.add(version)
            self.db.add(branch)
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
        source_message_id: str | None = None,
    ) -> list[ExecutionArtifact]:
        persisted: list[ExecutionArtifact] = []
        for payload in artifact_payloads:
            if not isinstance(payload, Mapping):
                continue
            raw_name = payload.get("name")
            raw_kind = payload.get("kind")
            name = str(raw_name) if isinstance(raw_name, (str, int, float, bool)) else "artifact"
            kind = str(raw_kind) if isinstance(raw_kind, (str, int, float, bool)) else "artifact"
            content = payload.get("content", "")
            metadata = _json_safe(payload.get("metadata", {}))
            if isinstance(metadata, dict):
                metadata.setdefault("type", kind)
                metadata.setdefault("title", name)
                metadata.setdefault("status", "pending_verification")
                metadata.setdefault("semantic_key", _semantic_key(kind, name, metadata))
                if source_message_id:
                    metadata.setdefault("source_message_id", source_message_id)
            artifact = persist_artifact(
                self.db,
                name=name,
                kind=kind,
                session_id=session_id,
                content=_artifact_content_for_storage(content),
                metadata=metadata if isinstance(metadata, dict) else {"metadata": metadata},
            )
            persisted.append(_execution_artifact_read(artifact, session_id))

        return persisted


def _child_execute(
    queue: mp.Queue[bytes],
    code: str,
    datasets: dict[str, Any],
    dataset_profiles: dict[str, Any],
    active_dataset_profile: dict[str, Any] | None,
    runtime_metadata: dict[str, Any],
    active_key: str | None,
    max_stdout_length: int,
    max_preview_length: int,
    return_state: bool,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    artifacts: list[dict[str, Any]] = []
    preview_value: Any = None
    preview_called = False
    data = datasets.get(active_key) if active_key else None

    def preview(obj: Any) -> Any:
        nonlocal preview_called, preview_value
        preview_called = True
        preview_value = obj
        return obj

    helpers = _artifact_helpers(artifacts)
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "datasets": datasets,
        "dataset_profiles": dataset_profiles,
        "active_dataset_profile": active_dataset_profile,
        "artifact_history": runtime_metadata.get("artifact_history", []),
        "artifacts": runtime_metadata.get("artifact_history", []),
        "current_message_artifacts": artifacts,
        "mutation_history": runtime_metadata.get("mutation_history", []),
        "branch_history": runtime_metadata.get("branch_history", []),
        "current_branch": runtime_metadata.get("current_branch"),
        "current_version": runtime_metadata.get("current_version"),
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
        "safe_attrs": safe_attrs,
        "object_to_record": object_to_record,
        "objects_to_records": objects_to_records,
        "to_dataframe": to_dataframe,
        "preview_dataframe": preview_dataframe,
        "inspect_object": inspect_object,
        "summarize_structure": summarize_structure,
        "find_record_collections": find_record_collections,
        "get_path": get_path,
        "flatten_records_at_path": flatten_records_at_path,
    }

    _block_network()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(_compile_agent_code(code), namespace)

        data = namespace.get("data")
        result_value = _execution_result_value(namespace, preview_value, preview_called)

        payload = {
            "ok": True,
            "stdout": _truncate_text(stdout.getvalue(), max_stdout_length),
            "stderr": _truncate_text(stderr.getvalue(), max_stdout_length),
            "traceback": None,
            "result_preview": _limit_preview(_preview(result_value), max_preview_length),
            "datasets": namespace["datasets"] if return_state else {},
            "data": data if return_state else None,
            "artifacts": artifacts,
        }
    except Exception:
        payload = {
            "ok": False,
            "stdout": _truncate_text(stdout.getvalue(), max_stdout_length),
            "stderr": _truncate_text(stderr.getvalue(), max_stdout_length),
            "traceback": traceback_module.format_exc(),
            "result_preview": _limit_preview(_preview(preview_value), max_preview_length),
            "datasets": datasets if return_state else {},
            "data": data if return_state else None,
            "artifacts": artifacts,
        }

    queue.put(cloudpickle.dumps(payload))


def _compile_agent_code(code: str) -> Any:
    module = ast.parse(code, filename="<agent_code>", mode="exec")
    if module.body and isinstance(module.body[-1], ast.Expr):
        final_expression = module.body[-1]
        assignment = ast.Assign(
            targets=[ast.Name(id="_agent_result", ctx=ast.Store())],
            value=final_expression.value,
        )
        ast.copy_location(assignment, final_expression)
        module.body[-1] = assignment
        ast.fix_missing_locations(module)
    return compile(module, "<agent_code>", "exec")


def _execution_result_value(namespace: Mapping[str, Any], preview_value: Any, preview_called: bool) -> Any:
    if "RESULT" in namespace:
        return namespace["RESULT"]
    if "_agent_result" in namespace:
        return namespace["_agent_result"]
    if "_result" in namespace:
        return namespace["_result"]
    if preview_called:
        return preview_value
    return None


def _artifact_helpers(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    def save_table(*args: Any, description: str | None = None, **kwargs: Any) -> dict[str, str]:
        kw_name = kwargs.pop("name", None)
        kw_data = kwargs.pop(
            "data",
            kwargs.pop("rows", kwargs.pop("dataframe", kwargs.pop("dataframe_or_records", None))),
        )
        if kwargs:
            raise ValueError(f"save_table got unsupported keyword argument(s): {', '.join(sorted(kwargs))}")

        if len(args) >= 2:
            name = str(args[0])
            dataframe_or_records = args[1]
            if len(args) >= 3 and description is None:
                description = str(args[2]) if args[2] is not None else None
        elif len(args) == 1 and kw_data is not None:
            name = str(args[0])
            dataframe_or_records = kw_data
        elif kw_name is not None and kw_data is not None:
            name = str(kw_name)
            dataframe_or_records = kw_data
        else:
            raise ValueError("save_table requires a table name and data")

        artifacts.append(_table_artifact_payload(name, dataframe_or_records, description=description))
        return {"kind": "table", "name": name}

    def save_chart(*args: Any, description: str | None = None, **kwargs: Any) -> dict[str, str]:
        kw_name = kwargs.pop("name", None)
        kw_spec = kwargs.pop("chart_spec", None)
        kw_data = kwargs.pop("data", None)
        if kwargs:
            raise ValueError(f"save_chart got unsupported keyword argument(s): {', '.join(sorted(kwargs))}")

        if len(args) == 1:
            chart_spec = args[0]
            if not isinstance(chart_spec, Mapping):
                raise ValueError("save_chart(chart_spec) requires a mapping chart spec")
            name = str(chart_spec.get("title") or "Chart")
        elif len(args) >= 2:
            name = str(args[0])
            chart_spec = args[1]
            if len(args) >= 3 and description is None:
                description = str(args[2]) if args[2] is not None else None
        elif kw_spec is not None:
            chart_spec = kw_spec
            name = str(kw_name or (chart_spec.get("title") if isinstance(chart_spec, Mapping) else None) or "Chart")
        else:
            raise ValueError("save_chart requires chart_spec or name and chart_spec")

        if kw_name is not None:
            name = str(kw_name)
        if kw_data is not None:
            if not isinstance(chart_spec, Mapping):
                raise ValueError("save_chart data= requires chart_spec to be a mapping")
            chart_spec = dict(chart_spec)
            if not chart_spec.get("data"):
                chart_spec["data"] = kw_data

        safe_spec = _validated_chart_spec(name, chart_spec, description=description)
        artifacts.append(
            {
                "kind": "chart",
                "name": name,
                "content": {
                    "type": "chart",
                    "title": safe_spec["title"],
                    "description": safe_spec.get("description"),
                    "chart_spec": safe_spec,
                },
                "metadata": {
                    "type": "chart",
                    "title": safe_spec["title"],
                    "description": safe_spec.get("description"),
                    "chart_type": safe_spec["chart_type"],
                    "row_count": len(safe_spec["data"]),
                    "x": safe_spec["x"],
                    "y": safe_spec["y"],
                    "sampled": bool(safe_spec.get("_sampling")),
                    "chart_spec": safe_spec,
                },
            }
        )
        return {"kind": "chart", "name": name}

    def save_csv(*args: Any, **kwargs: Any) -> dict[str, str]:
        kw_name = kwargs.pop("name", None)
        kw_data = kwargs.pop(
            "data",
            kwargs.pop("rows", kwargs.pop("dataframe", kwargs.pop("dataframe_or_records", None))),
        )
        if kwargs:
            raise ValueError(f"save_csv got unsupported keyword argument(s): {', '.join(sorted(kwargs))}")
        if len(args) >= 2:
            name = str(args[0])
            dataframe_or_records = args[1]
        elif len(args) == 1 and kw_name is not None and kw_data is None:
            name = str(kw_name)
            dataframe_or_records = args[0]
        elif len(args) == 1 and kw_data is not None:
            name = str(args[0])
            dataframe_or_records = kw_data
        elif len(args) == 1:
            name = str(kw_name or "CSV export")
            dataframe_or_records = args[0]
        elif kw_data is not None:
            name = str(kw_name or "CSV export")
            dataframe_or_records = kw_data
        else:
            raise ValueError("save_csv requires data to export")
        frame = _prepare_csv_frame(_frame_from_records(dataframe_or_records))
        artifacts.append(
            {
                "kind": "csv",
                "name": name,
                "content": frame.to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC),
                "metadata": {
                    "type": "csv",
                    "title": name,
                    "rows": int(len(frame)),
                    "row_count": int(len(frame)),
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
        "callable",
        "dir",
        "dict",
        "enumerate",
        "filter",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
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
        "vars",
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


def safe_attrs(obj: Any) -> dict[str, Any]:
    attrs = getattr(obj, "attrs", None)
    if isinstance(attrs, Mapping) and isinstance(attrs.get("__dict__"), Mapping):
        return dict(attrs["__dict__"])

    object_dict = getattr(obj, "__dict__", None)
    if isinstance(object_dict, Mapping):
        nested_attrs = object_dict.get("attrs")
        if isinstance(nested_attrs, Mapping) and isinstance(nested_attrs.get("__dict__"), Mapping):
            return dict(nested_attrs["__dict__"])
        if isinstance(object_dict.get("__dict__"), Mapping):
            return dict(object_dict["__dict__"])

    dumped = _pydantic_dump(obj)
    if dumped is not None:
        return _unwrap_record_mapping(dumped)

    if isinstance(obj, Mapping):
        return _unwrap_record_mapping(obj)

    if isinstance(object_dict, Mapping):
        return _unwrap_record_mapping(object_dict)

    return {"repr": _safe_repr(obj)}


def object_to_record(obj: Any) -> dict[str, Any]:
    record = _domain_record(safe_attrs(obj))
    return _record_safe(record, iso_dates=True)


def objects_to_records(items: Any, limit: int | None = None) -> list[dict[str, Any]]:
    return _objects_to_records(items, limit=limit, iso_dates=True)


def _objects_to_records(items: Any, limit: int | None = None, *, iso_dates: bool) -> list[dict[str, Any]]:
    max_items = None if limit is None else max(0, limit)
    if isinstance(items, Mapping) or isinstance(items, (str, bytes, bytearray)):
        return [_record_safe(_domain_record(safe_attrs(items)), iso_dates=iso_dates)]

    try:
        iterator = iter(items)
    except TypeError:
        return [_record_safe(_domain_record(safe_attrs(items)), iso_dates=iso_dates)]

    records: list[dict[str, Any]] = []
    for index, item in enumerate(iterator):
        if max_items is not None and index >= max_items:
            break
        records.append(_record_safe(_domain_record(safe_attrs(item)), iso_dates=iso_dates))
    return records


def to_dataframe(obj: Any, limit: int | None = None) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.head(limit) if limit is not None else obj

    if isinstance(obj, pd.Series):
        series = obj.head(limit) if limit is not None else obj
        return series.to_frame()

    if isinstance(obj, np.ndarray):
        array = obj[:limit] if limit is not None and obj.ndim > 0 else obj
        return pd.DataFrame(array)

    if isinstance(obj, Mapping):
        record = _record_safe(_domain_record(safe_attrs(obj)), iso_dates=False)
        if isinstance(record, Mapping) and _looks_like_column_dict(record):
            return pd.DataFrame({str(key): list(value) for key, value in record.items()})
        return pd.DataFrame([record])

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return pd.DataFrame(_objects_to_records(obj, limit=limit, iso_dates=False))

    try:
        return pd.DataFrame(_objects_to_records(obj, limit=limit, iso_dates=False))
    except Exception:
        return pd.DataFrame([{"repr": _safe_repr(obj)}])


def preview_dataframe(obj: Any, limit: int = MAX_PREVIEW_ROWS) -> pd.DataFrame:
    source_total = _source_row_count(obj)
    frame = to_dataframe(obj, limit=limit)
    preview_count = int(len(frame))
    if source_total is None:
        source_total = preview_count
    frame.attrs["source_total_row_count"] = int(source_total)
    frame.attrs["source_row_count"] = int(source_total)
    frame.attrs["analyzed_row_count"] = preview_count
    frame.attrs["preview_row_count"] = preview_count
    frame.attrs["is_preview"] = True
    return frame


def inspect_object(obj: Any, max_depth: int = 3, max_items: int = 5) -> dict[str, Any]:
    """Return a compact JSON-safe structural summary for arbitrary Python objects."""
    summary = _inspect_object(obj, path="", depth=0, max_depth=max_depth, max_items=max_items)
    if isinstance(summary, dict):
        summary["possible_record_collections"] = find_record_collections(obj, max_depth=max_depth + 1)[:20]
    return summary


def summarize_structure(obj: Any) -> dict[str, Any]:
    inspected = inspect_object(obj, max_depth=3, max_items=5)
    collections = find_record_collections(obj, max_depth=4)
    tables = [item for item in collections if item.get("kind") == "dataframe"]
    arrays = _array_summaries(obj)
    custom_objects = _custom_object_summaries(obj)
    return {
        "object_type": inspected.get("type"),
        "length": inspected.get("length"),
        "top_level_keys": inspected.get("keys", []),
        "top_level_items": inspected.get("item_types", []),
        "tables_detected": [_compact_collection_summary(item) for item in tables],
        "arrays_detected": arrays,
        "record_collections_detected": [_compact_collection_summary(item) for item in collections],
        "custom_objects_detected": custom_objects,
        "likely_primary_records": [_compact_collection_summary(item) for item in _likely_primary_records(collections)],
        "field_groups": _field_groups_from_collections(collections),
    }


def find_record_collections(obj: Any, max_depth: int = 4) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[int] = set()
    _find_record_collections(obj, path="", depth=0, max_depth=max_depth, found=found, seen=seen)
    return _dedupe_collection_summaries(found)


def get_path(obj: Any, path: str) -> Any:
    value: Any = obj
    for token in _path_tokens(path):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value = [_get_child(item, token) for item in value]
            value = [item for item in value if item is not _MISSING]
        else:
            value = _get_child(value, token)
        if value is _MISSING:
            raise KeyError(f"Path not found: {path}")
    return value


def flatten_records_at_path(obj: Any, path: str, parent_fields: Sequence[str] | None = None) -> list[dict[str, Any]]:
    tokens = _path_tokens(path)
    if not tokens:
        return objects_to_records(obj, limit=None)
    rows: list[dict[str, Any]] = []
    _flatten_path(obj, tokens, parent={}, rows=rows, parent_fields=set(parent_fields or []))
    if rows:
        return rows

    # Helpful fallback: allow "readings" to match "sensors.readings".
    for collection in find_record_collections(obj, max_depth=5):
        collection_path = str(collection.get("path") or "")
        if collection_path == path or collection_path.endswith(f".{path}"):
            try:
                return flatten_records_at_path(obj, collection_path, parent_fields=parent_fields)
            except Exception:
                continue
    return rows


class _Missing:
    pass


_MISSING = _Missing()


def _inspect_object(obj: Any, *, path: str, depth: int, max_depth: int, max_items: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": path, "type": type(obj).__name__, "module": type(obj).__module__}
    length = _safe_len(obj)
    if length is not None:
        summary["length"] = length

    if isinstance(obj, pd.DataFrame):
        summary.update(
            {
                "kind": "dataframe",
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns": [str(column) for column in obj.columns.tolist()],
                "dtypes": {str(column): str(dtype) for column, dtype in obj.dtypes.items()},
                "sample": _json_safe(obj.head(max_items).to_dict(orient="records"), max_items=max_items),
            }
        )
        return summary

    if isinstance(obj, pd.Series):
        summary.update({"kind": "series", "shape": [int(len(obj))], "dtype": str(obj.dtype)})
        return summary

    if isinstance(obj, np.ndarray):
        summary.update(
            {
                "kind": "ndarray",
                "shape": [int(item) for item in obj.shape],
                "dtype": str(obj.dtype),
                "sample": _json_safe(obj.reshape(-1)[:max_items].tolist(), max_items=max_items),
            }
        )
        return summary

    if depth >= max_depth:
        summary["sample"] = _json_safe(obj, max_depth=1, max_items=max_items)
        return summary

    if isinstance(obj, Mapping):
        keys = [str(key) for key in list(obj.keys())[:max_items]]
        summary["kind"] = "dict"
        summary["keys"] = keys
        summary["children"] = [
            _inspect_object(value, path=_join_path(path, str(key)), depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for key, value in list(obj.items())[:max_items]
        ]
        return summary

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        items = list(obj[:max_items]) if isinstance(obj, (list, tuple)) else list(obj)[:max_items]
        summary["kind"] = "sequence"
        summary["item_types"] = _item_type_counts(items)
        summary["sample"] = _json_safe(items, max_depth=2, max_items=max_items)
        if items:
            summary["children"] = [
                _inspect_object(item, path=f"{path}[{index}]" if path else f"[{index}]", depth=depth + 1, max_depth=max_depth, max_items=max_items)
                for index, item in enumerate(items[:max_items])
            ]
        return summary

    attrs = _public_attrs(obj)
    if attrs:
        summary["kind"] = "custom_object"
        summary["attrs"] = list(attrs.keys())[:max_items]
        summary["children"] = [
            _inspect_object(value, path=_join_path(path, key), depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for key, value in list(attrs.items())[:max_items]
        ]
        return summary

    summary["sample"] = _json_safe(obj, max_depth=1, max_items=max_items)
    return summary


def _find_record_collections(
    obj: Any,
    *,
    path: str,
    depth: int,
    max_depth: int,
    found: list[dict[str, Any]],
    seen: set[int],
) -> None:
    if depth > max_depth:
        return
    object_id = id(obj)
    if object_id in seen:
        return
    seen.add(object_id)

    if isinstance(obj, pd.DataFrame):
        found.append(
            {
                "path": path or "<root>",
                "kind": "dataframe",
                "count": int(len(obj)),
                "fields": [str(column) for column in obj.columns.tolist()],
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "sample": _json_safe(obj.head(3).to_dict(orient="records"), max_items=3),
            }
        )
        return

    if _is_record_sequence(obj):
        records = objects_to_records(obj, limit=5)
        fields = _fields_from_records(records)
        found.append(
            {
                "path": path or "<root>",
                "kind": _record_sequence_kind(obj),
                "count": _safe_len(obj) or len(records),
                "fields": fields,
                "sample": _json_safe(records[:3], max_items=3),
            }
        )

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            _find_record_collections(value, path=_join_path(path, str(key)), depth=depth + 1, max_depth=max_depth, found=found, seen=seen)
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in list(obj)[:MAX_PREVIEW_ITEMS]:
            attrs = safe_attrs(item)
            if attrs.keys() == {"repr"}:
                continue
            for key, value in attrs.items():
                if _looks_like_collection(value):
                    _find_record_collections(value, path=_join_path(path, str(key)), depth=depth + 1, max_depth=max_depth, found=found, seen=seen)
        return

    attrs = _public_attrs(obj)
    for key, value in attrs.items():
        if _looks_like_collection(value):
            _find_record_collections(value, path=_join_path(path, key), depth=depth + 1, max_depth=max_depth, found=found, seen=seen)


def _flatten_path(
    value: Any,
    tokens: list[str],
    *,
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    parent_fields: set[str],
) -> None:
    if not tokens:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                record = _flatten_record_leaves(object_to_record(item))
                rows.append({**parent, **record})
        else:
            rows.append({**parent, **_flatten_record_leaves(object_to_record(value))})
        return

    token = tokens[0]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _flatten_path(item, tokens, parent=_parent_record(parent, item, parent_fields), rows=rows, parent_fields=parent_fields)
        return

    child = _get_child(value, token)
    if child is _MISSING:
        return
    _flatten_path(child, tokens[1:], parent=_parent_record(parent, value, parent_fields), rows=rows, parent_fields=parent_fields)


def _parent_record(parent: dict[str, Any], value: Any, parent_fields: set[str]) -> dict[str, Any]:
    record = _flatten_record_leaves(object_to_record(value))
    output = dict(parent)
    useful_keys = parent_fields or {
        key
        for key in record
        if key.endswith("_id")
        or key
        in {
            "id",
            "customer_id",
            "event_id",
            "sensor_id",
            "timestamp",
            "country",
            "site",
            "zone",
            "segment",
            "joined_at",
            "churn_risk",
        }
    }
    for key in useful_keys:
        if key in record and key not in output:
            output[key] = record[key]
    return output


def _flatten_record_leaves(record: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    def add(key: str, value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                add(str(child_key), child_value, f"{prefix}{key}_" if prefix or key else "")
            return
        final_key = key
        if final_key in output:
            final_key = f"{prefix}{key}".strip("_") or key
        output[final_key] = value

    for key, value in record.items():
        add(str(key), value)
    return output


def _path_tokens(path: str) -> list[str]:
    return [part.replace("[]", "").strip() for part in path.split(".") if part.replace("[]", "").strip()]


def _get_child(value: Any, token: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(token, _MISSING)
    attrs = _public_attrs(value)
    if token in attrs:
        return attrs[token]
    return _MISSING


def _public_attrs(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return asdict(obj)
        except Exception:
            pass
    attrs = safe_attrs(obj)
    if attrs.keys() != {"repr"}:
        return {str(key): value for key, value in attrs.items() if not str(key).startswith("_")}
    return {}


def _is_record_sequence(obj: Any) -> bool:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, bytearray)):
        return False
    if not obj:
        return False
    sample = list(obj)[:5]
    records = [object_to_record(item) for item in sample]
    return any(record.keys() != {"repr"} and len(record) >= 2 for record in records)


def _record_sequence_kind(obj: Any) -> str:
    sample = list(obj)[:5]
    if all(isinstance(item, Mapping) for item in sample):
        return "list[dict]"
    if all(is_dataclass(item) for item in sample):
        return "list[dataclass]"
    return "list[object]"


def _looks_like_collection(value: Any) -> bool:
    return isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, Mapping, list, tuple))


def _fields_from_records(records: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields and not str(key).startswith("__"):
                fields.append(str(key))
    return fields


def _item_type_counts(items: Sequence[Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        key = type(item).__name__
        counts[key] = counts.get(key, 0) + 1
    return [{"type": key, "count": value} for key, value in counts.items()]


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        return None


def _join_path(prefix: str, key: str) -> str:
    if not prefix or prefix == "<root>":
        return key
    return f"{prefix}.{key}"


def _dedupe_collection_summaries(collections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for item in collections:
        key = (str(item.get("path")), str(item.get("kind")))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _array_summaries(obj: Any) -> list[dict[str, Any]]:
    arrays: list[dict[str, Any]] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 3:
            return
        if isinstance(value, np.ndarray):
            arrays.append({"path": path or "<root>", "type": "ndarray", "shape": [int(item) for item in value.shape], "dtype": str(value.dtype)})
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, _join_path(path, str(key)), depth + 1)
        else:
            for key, child in _public_attrs(value).items():
                visit(child, _join_path(path, key), depth + 1)

    visit(obj, "", 0)
    return arrays


def _custom_object_summaries(obj: Any) -> list[dict[str, Any]]:
    custom: list[dict[str, Any]] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if depth > 3:
            return
        if _is_custom_object(value):
            attrs = _public_attrs(value)
            custom.append({"path": path or "<root>", "type": type(value).__name__, "fields": list(attrs.keys())[:20]})
            for key, child in attrs.items():
                visit(child, _join_path(path, key), depth + 1)
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, _join_path(path, str(key)), depth + 1)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in list(value)[:5]:
                visit(item, path, depth + 1)

    visit(obj, "", 0)
    return custom


def _is_custom_object(value: Any) -> bool:
    return not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
            int,
            float,
            bool,
            type(None),
            datetime,
            date,
            Mapping,
            list,
            tuple,
            set,
            pd.DataFrame,
            pd.Series,
            np.ndarray,
        ),
    )


def _compact_collection_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("path", "kind", "count", "fields", "shape"):
        if key in item:
            value = item[key]
            if key == "fields" and isinstance(value, list):
                compact[key] = [str(field) for field in value[:40]]
            else:
                compact[key] = value
    sample = item.get("sample")
    if isinstance(sample, list) and sample and isinstance(sample[0], Mapping):
        compact["sample_fields"] = list(sample[0].keys())[:20]
    return compact


def _likely_primary_records(collections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(collections, key=lambda item: int(item.get("count") or 0), reverse=True)
    return [dict(item) for item in ranked[:3]]


def _field_groups_from_collections(collections: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups = {"identifier": [], "date": [], "list_like": [], "numeric": [], "text": []}
    for collection in collections[:5]:
        for field in collection.get("fields", []) if isinstance(collection.get("fields"), list) else []:
            lowered = str(field).lower()
            if lowered.endswith("_id") or lowered in {"id", "country", "kind", "doc_number"}:
                groups["identifier"].append(str(field))
            elif "date" in lowered or "time" in lowered or lowered.endswith("_at"):
                groups["date"].append(str(field))
            elif lowered.endswith("s") or lowered in {"items", "events", "alerts", "readings"}:
                groups["list_like"].append(str(field))
            elif any(marker in lowered for marker in ("count", "total", "rate", "score", "pct", "amount", "revenue")):
                groups["numeric"].append(str(field))
            else:
                groups["text"].append(str(field))
    return {key: _dedupe([str(item) for item in values])[:20] for key, values in groups.items() if values}


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _pydantic_dump(obj: Any) -> dict[str, Any] | None:
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass

    dict_method = getattr(obj, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass

    return None


def _unwrap_record_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    inner = raw.get("__dict__")
    if isinstance(inner, Mapping) and (_has_wrapper_keys(raw) or len(raw) == 1):
        return dict(inner)
    return raw


def _domain_record(value: Mapping[str, Any]) -> dict[str, Any]:
    unwrapped = _unwrap_record_mapping(value)
    if set(unwrapped.keys()) == {"__dict__"} and isinstance(unwrapped.get("__dict__"), Mapping):
        unwrapped = dict(unwrapped["__dict__"])

    non_wrapper = {key: item for key, item in unwrapped.items() if key not in WRAPPER_KEYS}
    if non_wrapper:
        return non_wrapper
    return dict(unwrapped)


def _has_wrapper_keys(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in WRAPPER_KEYS)


def _records_from_table(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return [_normalize_table_record(row) for row in value.to_dict(orient="records")]

    if isinstance(value, Mapping):
        frame = to_dataframe(value)
        return [_normalize_table_record(row) for row in frame.to_dict(orient="records")]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return objects_to_records(value, limit=None)

    return [object_to_record(value)]


def _frame_from_records(value: Any) -> pd.DataFrame:
    return to_dataframe(value)


def _table_artifact_payload(
    name: str,
    data: Any,
    *,
    description: str | None = None,
    row_count_override: int | None = None,
    columns_override: Sequence[Any] | None = None,
) -> dict[str, Any]:
    document = _table_document(
        name,
        data,
        description=description,
        row_count_override=row_count_override,
        columns_override=columns_override,
    )
    return {
        "kind": "table",
        "name": name,
        "content": document,
        "metadata": {
            "type": "table",
            "title": document["title"],
            "description": document.get("description"),
            "columns": document["columns"],
            "rows": document["preview_rows"],
            "row_count": document["row_count"],
            "column_count": document["column_count"],
            "total_column_count": document["total_column_count"],
            "display_column_count": document["display_column_count"],
            "preview_column_count": document["preview_column_count"],
            "hidden_columns": document["hidden_columns"],
            "display_columns": document["display_columns"],
            "source_row_count": document["source_row_count"],
            "source_total_row_count": document["source_total_row_count"],
            "analyzed_row_count": document["analyzed_row_count"],
            "preview_row_count": document["preview_row_count"],
            "is_preview": document["is_preview"],
            "is_sampled": document["is_sampled"],
            "stored_row_count": document["stored_row_count"],
            "stored_column_count": document["stored_column_count"],
            "csv_download_available": True,
        },
    }


def _table_document(
    name: str,
    data: Any,
    *,
    description: str | None,
    row_count_override: int | None = None,
    columns_override: Sequence[Any] | None = None,
) -> dict[str, Any]:
    frame = _frame_for_table(data)
    source_row_count = _source_row_count(data)
    analyzed_row_count = _dataframe_attr_int(frame, "analyzed_row_count") or source_row_count or int(len(frame))
    source_row_count = _dataframe_attr_int(frame, "source_row_count") or source_row_count or analyzed_row_count
    source_total_row_count = _dataframe_attr_int(frame, "source_total_row_count") or source_row_count
    preview_attr_count = _dataframe_attr_int(frame, "preview_row_count")
    is_preview = bool(frame.attrs.get("is_preview"))
    frame = frame.copy()
    frame.columns = _unique_column_names(frame.columns)

    full_columns = [str(column) for column in frame.columns]
    if columns_override is not None:
        ordered_columns = [str(column) for column in columns_override]
        for column in ordered_columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[ordered_columns]
        full_columns = ordered_columns

    row_count = row_count_override if row_count_override is not None else int(len(frame))
    stored_frame = frame.head(MAX_TABLE_STORED_ROWS)
    rows = [_normalize_table_record(row) for row in stored_frame.to_dict(orient="records")]
    columns = _table_columns(frame, rows)
    display_columns = columns[:MAX_TABLE_COLUMNS]
    display_column_keys = [column["key"] for column in display_columns]
    preview_rows = [
        {key: row.get(key) for key in display_column_keys if key in row}
        for row in rows[:MAX_TABLE_INLINE_ROWS]
    ]
    hidden_columns = full_columns[MAX_TABLE_COLUMNS:]

    return {
        "type": "table",
        "title": str(name),
        "description": description,
        "columns": columns,
        "display_columns": display_columns,
        "rows": rows,
        "preview_rows": preview_rows,
        "row_count": int(row_count),
        "source_row_count": int(source_row_count),
        "source_total_row_count": int(source_total_row_count),
        "analyzed_row_count": int(analyzed_row_count),
        "preview_row_count": int(preview_attr_count or len(preview_rows)),
        "is_preview": is_preview,
        "is_sampled": bool(analyzed_row_count < source_row_count),
        "stored_row_count": len(rows),
        "stored_column_count": len(columns),
        "column_count": len(columns),
        "total_column_count": len(columns),
        "display_column_count": len(display_columns),
        "preview_column_count": len(display_columns),
        "hidden_columns": hidden_columns,
        "truncated": bool(row_count > len(rows) or hidden_columns),
    }


def _frame_for_table(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        name = value.name if value.name is not None else "value"
        return value.to_frame(name=str(name))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return pd.DataFrame(_objects_to_records(value, limit=None, iso_dates=False))
    return to_dataframe(value)


def _source_row_count(value: Any) -> int | None:
    if isinstance(value, pd.DataFrame):
        return int(len(value))
    if isinstance(value, pd.Series):
        return int(len(value))
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return int(value.shape[0])
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        return None


def _semantic_key(kind: str, name: str, metadata: Mapping[str, Any]) -> str:
    lowered = f"{name} {metadata.get('title') or ''} {metadata.get('description') or ''}".lower()
    if kind == "csv":
        return "csv_export"
    if kind == "table":
        keys = {
            str(column.get("key") or column.get("label") or "").lower()
            for column in metadata.get("columns", [])
            if isinstance(column, Mapping)
        }
        if {"filing_year", "year"} & keys and any(key in keys for key in {"filing_count", "count", "record_count"}):
            return "filings_by_year_table"
        if "country" in keys and any(key in keys for key in {"count", "record_count", "patent_count"}):
            return "top_countries_table"
        if "schema" in lowered:
            return "schema_summary_table"
        if "preview" in lowered or "first" in lowered:
            return "tabular_preview_table"
    if kind == "chart":
        chart_spec = metadata.get("chart_spec")
        spec = chart_spec if isinstance(chart_spec, Mapping) else {}
        x = str(spec.get("x") or metadata.get("x") or "").lower()
        y = str(spec.get("y") or metadata.get("y") or "").lower()
        if x in {"filing_year", "year"} and y in {"filing_count", "count", "record_count"}:
            return "filings_by_year_chart"
        if x == "country":
            return "country_distribution_chart"
    normalized_title = "".join(character if character.isalnum() else "_" for character in name.lower())
    return f"{kind}_{normalized_title.strip('_') or 'artifact'}"


def _dataframe_attr_int(frame: pd.DataFrame, key: str) -> int | None:
    value = frame.attrs.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _unique_column_names(columns: Sequence[Any]) -> list[str]:
    used: dict[str, int] = {}
    names: list[str] = []
    for column in columns:
        base = str(column)
        count = used.get(base, 0)
        used[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    return names


def _table_columns(frame: pd.DataFrame, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    columns: list[dict[str, str]] = []
    for column in list(frame.columns):
        columns.append(
            {
                "key": str(column),
                "label": str(column),
                "type": _infer_table_column_type(str(column), rows),
            }
        )
    return columns


def _infer_table_column_type(column: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if _is_identifier_column(column):
        return "identifier"
    for row in rows[:MAX_TABLE_INLINE_ROWS]:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "number"
        if isinstance(value, list):
            return "list"
        if isinstance(value, Mapping):
            return "object"
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return "text"
            return "date"
    return "unknown"


def _normalize_table_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _table_cell_safe(value, column=str(key)) for key, value in record.items()}


def _table_cell_safe(value: Any, *, column: str | None = None, depth: int = 0) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _table_cell_safe(value.item(), column=column, depth=depth + 1)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        parsed = _parse_structured_string(value)
        if parsed is not value:
            return _table_cell_safe(parsed, column=column, depth=depth + 1)
        return _truncate_preview_string(value, MAX_TABLE_CELL_LENGTH)
    if isinstance(value, Decimal):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    if isinstance(value, (int, bool)):
        if column is not None and _is_identifier_column(column):
            return str(value)
        return value
    if isinstance(value, float) and column is not None and _is_identifier_column(column):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, Mapping):
        if depth >= MAX_PREVIEW_DEPTH:
            return _safe_repr(value, MAX_TABLE_CELL_LENGTH)
        return {
            str(key): _table_cell_safe(item_value, column=str(key), depth=depth + 1)
            for key, item_value in list(value.items())[:MAX_PREVIEW_ITEMS]
        }
    if isinstance(value, (list, tuple, set)):
        if depth >= MAX_PREVIEW_DEPTH:
            return _safe_repr(value, MAX_TABLE_CELL_LENGTH)
        return [_table_cell_safe(item, depth=depth + 1) for item in list(value)[:MAX_PREVIEW_ITEMS]]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    attrs = safe_attrs(value)
    if attrs.keys() != {"repr"}:
        return _table_cell_safe(_domain_record(attrs), column=column, depth=depth + 1)
    return _safe_repr(value, MAX_TABLE_CELL_LENGTH)


def _is_identifier_column(column: str) -> bool:
    lowered = column.lower()
    return (
        lowered in {"id", "country", "kind", "doc_number", "patent_number", "application_number", "publication_number"}
        or lowered.endswith("_id")
        or lowered.endswith("_number")
    )


def _parse_structured_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, (list, tuple, set, dict)):
        return parsed
    return value


def _prepare_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared.columns = _unique_column_names(prepared.columns)
    for column in prepared.columns:
        column_name = str(column)
        prepared[column] = prepared[column].map(lambda value, column_name=column_name: _csv_cell_safe(value, column_name))
    return prepared


def _csv_cell_safe(value: Any, column: str) -> Any:
    value = _table_cell_safe(value, column=column)
    if value is None:
        return ""
    if _is_identifier_column(column):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _looks_like_column_dict(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    lengths: set[int] = set()
    for item in value.values():
        if isinstance(item, np.ndarray):
            if item.ndim == 0:
                return False
            lengths.add(int(item.shape[0]))
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            lengths.add(len(item))
            continue
        return False
    return len(lengths) == 1


def _auto_table_artifact_from_preview(
    result_preview: Any,
    artifact_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if artifact_payloads:
        return None
    if not isinstance(result_preview, Mapping):
        return None

    preview_type = result_preview.get("type")
    rows = result_preview.get("rows")
    columns = result_preview.get("columns")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        return None

    if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes, bytearray)):
        columns_override = [str(column) for column in columns]
    else:
        columns_override = None

    row_count_override = None
    shape = result_preview.get("shape")
    if preview_type == "dataframe" and isinstance(shape, Sequence) and shape:
        try:
            row_count_override = int(shape[0])
        except (TypeError, ValueError):
            row_count_override = None

    return _table_artifact_payload(
        "Result preview table",
        list(rows),
        description="Structured table generated from the Python result preview.",
        row_count_override=row_count_override,
        columns_override=columns_override,
    )


def _artifact_content_for_storage(content: Any) -> str | bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content
    return json.dumps(_artifact_safe(content), ensure_ascii=False, indent=2)


def _execution_artifact_read(artifact: Any, session_id: str) -> ExecutionArtifact:
    metadata = artifact.artifact_metadata if isinstance(artifact.artifact_metadata, dict) else {}
    columns = metadata.get("columns")
    rows = metadata.get("rows")
    chart_spec = metadata.get("chart_spec")
    title = metadata.get("title")
    description = metadata.get("description")
    source_message_id = metadata.get("source_message_id")
    return ExecutionArtifact(
        id=artifact.id,
        name=artifact.name,
        kind=artifact.kind,
        type=str(metadata.get("type") or artifact.kind),
        title=str(title) if title is not None else artifact.name,
        description=str(description) if description is not None else None,
        columns=_artifact_column_defs(columns),
        rows=rows if isinstance(rows, list) else [],
        chart_spec=chart_spec if isinstance(chart_spec, dict) else None,
        download_url=f"/api/sessions/{session_id}/artifacts/{artifact.id}/download",
        source_message_id=str(source_message_id) if source_message_id is not None else None,
        path=artifact.path,
        metadata=metadata,
        created_at=artifact.created_at.isoformat(),
        status=str(metadata.get("status")) if metadata.get("status") else None,
        semantic_key=str(metadata.get("semantic_key")) if metadata.get("semantic_key") else None,
    )


def _artifact_column_defs(columns: Any) -> list[dict[str, Any]]:
    if not isinstance(columns, list):
        return []
    normalized: list[dict[str, Any]] = []
    for column in columns:
        if isinstance(column, Mapping):
            key = str(column.get("key") or column.get("name") or column.get("label") or "")
            if key:
                normalized.append(
                    {
                        "key": key,
                        "label": str(column.get("label") or key),
                        "type": str(column.get("type") or "value"),
                    }
                )
        elif column is not None:
            key = str(column)
            normalized.append({"key": key, "label": key, "type": "value"})
    return normalized


def _artifact_safe(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _artifact_safe(value.item(), depth=depth + 1)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if depth >= MAX_PREVIEW_DEPTH + 2:
        return _safe_repr(value, MAX_TABLE_CELL_LENGTH)
    if isinstance(value, Mapping):
        return {
            str(key): _artifact_safe(item_value, depth=depth + 1)
            for key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_artifact_safe(item, depth=depth + 1) for item in list(value)[:MAX_TABLE_STORED_ROWS]]
    if isinstance(value, set):
        return [_artifact_safe(item, depth=depth + 1) for item in list(value)[:MAX_TABLE_STORED_ROWS]]
    if isinstance(value, pd.DataFrame):
        return _artifact_safe(value.to_dict(orient="records"), depth=depth + 1)
    if isinstance(value, pd.Series):
        return _artifact_safe(value.tolist(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        return _artifact_safe(value.tolist(), depth=depth + 1)
    attrs = safe_attrs(value)
    if attrs.keys() != {"repr"}:
        return _artifact_safe(_domain_record(attrs), depth=depth + 1)
    return attrs["repr"]


def _validated_chart_spec(name: str, chart_spec: Any, description: str | None = None) -> dict[str, Any]:
    raw = dict(chart_spec) if isinstance(chart_spec, Mapping) else _json_safe(chart_spec)
    if not isinstance(raw, Mapping):
        raise ValueError("Chart spec must be a mapping with chart_type, data, x, and y")

    data_value = None
    for data_key in ("data", "values", "rows", "records"):
        if data_key in raw and raw.get(data_key) is not None:
            data_value = raw.get(data_key)
            break
    if data_value is None:
        for candidate_key in ("dataset", "source", "table"):
            candidate = raw.get(candidate_key)
            if isinstance(candidate, (pd.DataFrame, pd.Series, np.ndarray, Mapping, Sequence)) and not isinstance(
                candidate, (str, bytes, bytearray)
            ):
                data_value = candidate
                break
    data = _chart_rows_from_value(data_value)

    if not data:
        got_type = type(data_value).__name__ if data_value is not None else "missing"
        raise ValueError(f"chart_spec.data must be a non-empty list of dict rows; got {got_type}")

    chart_type = _normalize_chart_type(raw.get("chart_type") or raw.get("type") or raw.get("mark") or "bar")
    if chart_type not in {"bar", "line", "pie", "scatter", "area"}:
        raise ValueError(f"Unsupported chart_type '{chart_type}'")

    x = raw.get("x") or _encoding_field(raw, "x")
    y = raw.get("y") or _encoding_field(raw, "y")
    x_key, y_key = _infer_chart_fields(data, str(chart_type), str(x) if x else None, str(y) if y else None)

    if not all(x_key in row for row in data):
        raise ValueError(f"Chart x field '{x_key}' is missing from one or more rows")
    if not all(y_key in row for row in data):
        raise ValueError(f"Chart y field '{y_key}' is missing from one or more rows")

    data = _coerce_chart_y_values(data, y_key, chart_type)
    data, sampling = _reduce_chart_data(data, chart_type, x_key, y_key)
    spec = ChartArtifactSpec(
        id=str(raw.get("id") or new_id()),
        title=str(raw.get("title") or name),
        chart_type=chart_type,  # type: ignore[arg-type]
        data=[_json_safe(dict(row)) for row in data],
        x=x_key,
        y=y_key,
        color=str(raw["color"]) if raw.get("color") else None,
        description=str(description or raw["description"]) if (description or raw.get("description")) else None,
    ).model_dump(exclude_none=True)
    if raw.get("series"):
        spec["series"] = str(raw["series"])
    if sampling:
        spec["_sampling"] = sampling
    return spec


def _chart_rows_from_value(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [_normalize_table_record(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _records_from_table(value.reset_index())
    if isinstance(value, np.ndarray):
        return _records_from_table(value)
    if isinstance(value, Mapping):
        return _records_from_table(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                rows.append(_normalize_table_record(dict(item)))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                rows.append({f"value_{i}": _table_cell_safe(cell) for i, cell in enumerate(item)})
            else:
                record = object_to_record(item)
                if record.keys() == {"repr"}:
                    rows.append({"index": index, "value": _table_cell_safe(item)})
                else:
                    rows.append(record)
        return rows
    return []


def _coerce_chart_y_values(rows: list[dict[str, Any]], y_key: str, chart_type: str) -> list[dict[str, Any]]:
    if chart_type not in {"bar", "line", "area", "scatter", "pie"}:
        return rows
    coerced: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        value = next_row.get(y_key)
        if isinstance(value, Decimal):
            numeric = float(value)
            next_row[y_key] = int(numeric) if numeric.is_integer() else numeric
        elif isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    numeric = float(stripped)
                    next_row[y_key] = int(numeric) if numeric.is_integer() else numeric
                except ValueError:
                    pass
        coerced.append(next_row)
    return coerced


def _normalize_chart_type(value: Any) -> str:
    text = str(value or "bar").strip().lower().replace("_", " ").replace("-", " ")
    synonyms = {
        "bar chart": "bar",
        "column": "bar",
        "column chart": "bar",
        "line chart": "line",
        "area chart": "area",
        "pie chart": "pie",
        "scatter chart": "scatter",
        "scatter plot": "scatter",
    }
    return synonyms.get(text, text.split()[0] if text.endswith(" chart") and text.split() else text)


def _encoding_field(raw: Mapping[str, Any], channel: str) -> str | None:
    encoding = raw.get("encoding")
    if not isinstance(encoding, Mapping):
        return None
    value = encoding.get(channel)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("field"), str):
        return value["field"]
    return None


def _infer_chart_fields(
    data: Sequence[Mapping[str, Any]],
    chart_type: str,
    x: str | None,
    y: str | None,
) -> tuple[str, str]:
    keys = list(data[0].keys())
    if not keys:
        raise ValueError("Chart rows must have at least one field")

    if x and y:
        return x, y

    numeric_keys = [
        key
        for key in keys
        if any(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in data)
    ]
    categorical_keys = [key for key in keys if key not in numeric_keys]

    inferred_y = y or (numeric_keys[1] if chart_type == "scatter" and len(numeric_keys) > 1 else numeric_keys[0] if numeric_keys else keys[-1])
    inferred_x = x or (
        numeric_keys[0]
        if chart_type == "scatter" and numeric_keys and numeric_keys[0] != inferred_y
        else categorical_keys[0]
        if categorical_keys
        else next((key for key in keys if key != inferred_y), keys[0])
    )
    return str(inferred_x), str(inferred_y)


def _reduce_chart_data(
    data: Sequence[Mapping[str, Any]],
    chart_type: str,
    x: str,
    y: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows = [dict(row) for row in data]
    original_count = len(rows)

    if chart_type in {"bar", "pie"} and original_count > MAX_CATEGORY_CHART_ROWS:
        frame = pd.DataFrame(rows)
        if x in frame.columns and y in frame.columns and pd.api.types.is_numeric_dtype(frame[y]):
            grouped = frame.groupby(x, dropna=False, as_index=False)[y].sum()
            grouped = grouped.sort_values(y, key=lambda series: series.abs(), ascending=False)
            reduced = [
                _json_safe(dict(row))
                for row in grouped.head(MAX_CATEGORY_CHART_ROWS).to_dict(orient="records")
            ]
            return reduced, {
                "method": "aggregate_top_categories",
                "original_rows": original_count,
                "saved_rows": len(reduced),
            }

        reduced = rows[:MAX_CATEGORY_CHART_ROWS]
        return reduced, {
            "method": "head",
            "original_rows": original_count,
            "saved_rows": len(reduced),
        }

    if original_count > MAX_CHART_ROWS:
        step = max(1, math.ceil(original_count / MAX_CHART_ROWS))
        reduced = rows[::step][:MAX_CHART_ROWS]
        return reduced, {
            "method": "even_sample",
            "original_rows": original_count,
            "saved_rows": len(reduced),
        }

    return rows, None


def _preview(value: Any) -> Any:
    return _json_safe(value)


def _json_safe(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = MAX_PREVIEW_DEPTH,
    max_items: int = MAX_PREVIEW_ITEMS,
    max_string_length: int = MAX_PREVIEW_STRING_LENGTH,
) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return _truncate_preview_string(value, max_string_length) if isinstance(value, str) else value

    if isinstance(value, type):
        return _type_name(value)

    if isinstance(value, Decimal):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, np.generic):
        return _json_safe(
            value.item(),
            depth=depth + 1,
            max_depth=max_depth,
            max_items=max_items,
            max_string_length=max_string_length,
        )

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if depth >= max_depth:
        return _safe_repr(value, max_string_length)

    if isinstance(value, pd.DataFrame):
        frame = value.head(MAX_PREVIEW_ROWS)
        source_total_row_count = int(value.attrs.get("source_total_row_count") or value.attrs.get("source_row_count") or len(value))
        analyzed_row_count = int(value.attrs.get("analyzed_row_count") or len(value))
        preview_row_count = int(value.attrs.get("preview_row_count") or len(frame))
        return {
            "type": "dataframe",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(column) for column in value.columns.tolist()],
            "source_row_count": source_total_row_count,
            "source_total_row_count": source_total_row_count,
            "analyzed_row_count": analyzed_row_count,
            "preview_row_count": preview_row_count,
            "is_preview": bool(value.attrs.get("is_preview")),
            "rows": _json_safe(
                frame.to_dict(orient="records"),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=MAX_PREVIEW_ROWS,
                max_string_length=max_string_length,
            ),
        }

    if isinstance(value, pd.Series):
        series = value.head(MAX_PREVIEW_ROWS)
        return {
            "type": "series",
            "name": str(value.name) if value.name is not None else None,
            "length": int(len(value)),
            "dtype": str(value.dtype),
            "source_row_count": int(len(value)),
            "analyzed_row_count": int(len(value)),
            "values": _json_safe(
                series.tolist(),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=MAX_PREVIEW_ROWS,
                max_string_length=max_string_length,
            ),
        }

    if isinstance(value, np.ndarray):
        sample = value.reshape(-1)[:max_items].tolist()
        return {
            "type": "ndarray",
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "source_row_count": int(value.shape[0]) if value.ndim > 0 else 1,
            "analyzed_row_count": int(value.shape[0]) if value.ndim > 0 else 1,
            "sample": _json_safe(
                sample,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            ),
        }

    if isinstance(value, Mapping):
        safe_items: dict[str, Any] = {}
        for index, (key, item_value) in enumerate(value.items()):
            if index >= max_items:
                safe_items["_truncated"] = f"Showing first {max_items} items"
                break
            safe_items[str(_json_safe(key, depth=depth + 1, max_depth=max_depth, max_items=max_items))] = _json_safe(
                item_value,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
        return safe_items

    if isinstance(value, (list, tuple)):
        item_limit = MAX_PREVIEW_ROWS if _looks_like_records(value) else max_items
        return [
            _json_safe(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in list(value)[:item_limit]
        ]

    if isinstance(value, set):
        return [
            _json_safe(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in list(value)[:max_items]
        ]

    if hasattr(value, "tolist"):
        try:
            return _json_safe(
                value.tolist(),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
        except Exception:
            pass

    attrs = safe_attrs(value)
    if attrs.keys() != {"repr"}:
        return {
            "type": type(value).__qualname__,
            "module": getattr(type(value), "__module__", None),
            "repr": _safe_repr(value, max_string_length),
            "attrs": _json_safe(
                attrs,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            ),
        }

    return attrs["repr"]


def _looks_like_records(value: Sequence[Any]) -> bool:
    sample = list(value[: min(len(value), MAX_PREVIEW_ROWS)]) if hasattr(value, "__len__") else []
    return bool(sample) and all(isinstance(item, Mapping) for item in sample)


def _record_safe(value: Any, *, depth: int = 0, iso_dates: bool) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat() if iso_dates else value

    if depth >= MAX_PREVIEW_DEPTH:
        return _safe_repr(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [
            _record_safe(item, depth=depth + 1, iso_dates=iso_dates)
            for item in value.reshape(-1)[:MAX_PREVIEW_ITEMS].tolist()
        ]

    if isinstance(value, Mapping):
        return {
            str(key): _record_safe(item_value, depth=depth + 1, iso_dates=iso_dates)
            for key, item_value in list(value.items())[:MAX_PREVIEW_ITEMS]
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _record_safe(item, depth=depth + 1, iso_dates=iso_dates)
            for item in list(value)[:MAX_PREVIEW_ITEMS]
        ]

    attrs = safe_attrs(value)
    if attrs.keys() != {"repr"}:
        return _record_safe(_domain_record(attrs), depth=depth + 1, iso_dates=iso_dates)

    return attrs["repr"]


def _safe_repr(value: Any, max_length: int = MAX_PREVIEW_STRING_LENGTH) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__qualname__}>"
    return _truncate_preview_string(text, max_length)


def _type_name(value: type[Any]) -> str:
    if value.__module__ == "builtins":
        return value.__qualname__
    return f"{value.__module__}.{value.__qualname__}"


def _profile_row_count(profile: Mapping[str, Any] | None) -> int | None:
    if not profile:
        return None
    shape = profile.get("shape")
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and shape:
        first = shape[0]
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            return int(first)
    length = profile.get("length")
    if isinstance(length, (int, float)) and not isinstance(length, bool):
        return int(length)
    return None


def _version_metadata(version: VersionNode | None) -> dict[str, Any] | None:
    if version is None:
        return None
    return {
        "version_id": version.id,
        "dataset_id": version.dataset_id,
        "branch_id": version.branch_id,
        "parent_version_id": version.parent_version_id,
        "summary": version.mutation_summary or version.label,
        "created_at": version.created_at.isoformat(),
        "row_count": _profile_row_count(version.profile),
    }


def _truncate_preview_string(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}...[truncated]"


def _limit_preview(value: Any, max_length: int) -> Any:
    safe_value = _json_safe(value)
    try:
        encoded = json.dumps(safe_value, ensure_ascii=False)
    except Exception:
        return _safe_repr(value)

    if len(encoded) <= max_length:
        return safe_value

    return {
        "truncated": True,
        "value": _json_safe(
            value,
            max_depth=max(1, MAX_PREVIEW_DEPTH - 2),
            max_items=10,
            max_string_length=300,
        ),
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
    names = [dataset_key(row) for row in rows]
    duplicate_names = {name for name in names if names.count(name) > 1}
    keys: dict[str, str] = {}
    used: set[str] = set()

    for row in rows:
        row_key = dataset_key(row)
        base = row.id if row_key in duplicate_names else row_key
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


def _mp_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - fork is unavailable on some platforms
        return mp.get_context()
