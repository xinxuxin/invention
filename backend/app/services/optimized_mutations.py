from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
import math
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.runtime.python_executor import fast_get_field as runtime_fast_get_field
from app.runtime.python_executor import find_record_collections, object_to_record
from app.services.mutation_intents import normalize_country_value


MutationKind = Literal[
    "filter_records",
    "filter_records_at_path",
    "delete_first_n",
    "delete_last_n",
    "remove_missing_field",
    "drop_duplicates",
    "add_derived_field",
    "replace_collection_at_path",
    "delete_all_records",
]
Operator = Literal[
    "eq",
    "ne",
    "in",
    "not_in",
    "lt",
    "lte",
    "gt",
    "gte",
    "is_missing",
    "is_not_missing",
    "contains",
    "not_contains",
    "regex",
]
MutationMode = Literal["keep_matching", "delete_matching"]


@dataclass
class MutationSpec:
    kind: MutationKind
    target_dataset_id: str | None = None
    target_dataset_name: str | None = None
    target_path: str | None = None
    field_path: str | None = None
    field_paths: list[str] = field(default_factory=list)
    operator: Operator | None = None
    value: Any = None
    mode: MutationMode = "delete_matching"
    requires_confirmation: bool = True
    confidence: float = 0.0
    confidence_reasons: list[str] = field(default_factory=list)
    ambiguity_reasons: list[str] = field(default_factory=list)
    human_summary: str = ""
    reversible: bool = True
    created_by: str = "controller_preflight"
    source_prompt: str = ""
    estimated_current_count: int | None = None
    estimated_affected_count: int | None = None
    estimated_new_count: int | None = None
    count: int | None = None
    derived_field: str | None = None
    source_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MutationSpec":
        data = dict(value)
        data["field_paths"] = list(data.get("field_paths") or [])
        data["confidence_reasons"] = list(data.get("confidence_reasons") or [])
        data["ambiguity_reasons"] = list(data.get("ambiguity_reasons") or [])
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})


@dataclass
class MutationImpact:
    current_count: int
    affected_count: int
    new_count: int
    matching_count: int
    removed_value_counts: dict[str, int] = field(default_factory=dict)
    kept_value_counts: dict[str, int] = field(default_factory=dict)
    removed_examples: list[dict[str, Any]] = field(default_factory=list)
    kept_examples: list[dict[str, Any]] = field(default_factory=list)
    target_path: str | None = None
    field_path: str | None = None

    def to_preview(self) -> dict[str, Any]:
        return {
            "full_scan": True,
            "target_path": self.target_path,
            "field_path": self.field_path,
            "affected_count": int(self.affected_count),
            "matching_count": int(self.matching_count),
            "current_row_count": int(self.current_count),
            "new_row_count": int(self.new_count),
            "removed_value_counts": {str(key): int(value) for key, value in self.removed_value_counts.items()},
            "kept_value_counts": {str(key): int(value) for key, value in self.kept_value_counts.items()},
            "removed_examples": self.removed_examples[:3],
            "kept_examples": self.kept_examples[:3],
        }


@dataclass
class RecordCollectionCandidate:
    path: str | None
    kind: str
    count: int
    fields: list[str]
    sample: list[Any] = field(default_factory=list)
    parent_fields: list[str] = field(default_factory=list)
    score: float = 0.0
    supports_mutation: bool = True
    original_type: str = ""


@dataclass
class FieldCandidate:
    collection_path: str | None
    field_path: str
    field_name: str
    type: str
    value_examples: list[Any] = field(default_factory=list)
    unique_values_sample: list[Any] = field(default_factory=list)
    non_null_count: int = 0
    unique_count: int = 0
    score: float = 0.0
    semantic_tags: list[str] = field(default_factory=list)


@dataclass
class MutationParseOutcome:
    spec: MutationSpec | None = None
    clarification: dict[str, Any] | None = None
    reason: str | None = None


_MISSING = object()


def fast_get_field(obj: Any, field: str) -> Any:
    """Cheap generic field access that avoids full object serialization."""

    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        return obj.get(field, None)
    if isinstance(obj, Mapping):
        return obj.get(field)
    if is_dataclass(obj) and not isinstance(obj, type) and hasattr(obj, field):
        return getattr(obj, field, None)
    if hasattr(obj, field):
        return getattr(obj, field, None)
    value = runtime_fast_get_field(obj, field)
    if value is not None:
        return value
    object_dict = getattr(obj, "__dict__", None)
    if isinstance(object_dict, Mapping):
        return object_dict.get(field)
    return None


def fast_get_path(obj: Any, path: str | None) -> Any:
    if path is None or path == "" or path == "<root>":
        return obj
    value: Any = obj
    for token in _path_tokens(path):
        if isinstance(value, pd.DataFrame):
            if token not in value.columns:
                return None
            value = value[token]
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            children: list[Any] = []
            for item in value:
                child = fast_get_field(item, token)
                if child is None:
                    continue
                if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
                    children.extend(list(child))
                else:
                    children.append(child)
            value = children
            continue
        value = fast_get_field(value, token)
        if value is None:
            return None
    return value


def fast_set_path(root: Any, path: str | None, value: Any) -> Any:
    if path is None or path == "" or path == "<root>":
        return value
    tokens = _path_tokens(path)
    if not tokens:
        return value
    copied = _copy_container(root)
    _set_child_path(copied, tokens, value)
    return copied


def get_collection_at_path(root: Any, target_path: str | None) -> Any:
    return fast_get_path(root, target_path)


def replace_collection_at_path(root: Any, target_path: str | None, new_collection: Any) -> Any:
    return fast_set_path(root, target_path, new_collection)


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except Exception:
        return False


def compare_value(actual: Any, operator: str | None, expected: Any) -> bool:
    op = operator or "eq"
    if op == "is_missing":
        return is_missing_value(actual)
    if op == "is_not_missing":
        return not is_missing_value(actual)
    if op in {"lt", "lte", "gt", "gte"}:
        actual_num = _to_number(actual)
        expected_num = _to_number(expected)
        if actual_num is None or expected_num is None:
            return False
        if op == "lt":
            return actual_num < expected_num
        if op == "lte":
            return actual_num <= expected_num
        if op == "gt":
            return actual_num > expected_num
        return actual_num >= expected_num
    if op in {"contains", "not_contains"}:
        matched = _contains(actual, expected)
        return matched if op == "contains" else not matched
    if op in {"in", "not_in"}:
        expected_values = expected if isinstance(expected, list) else [expected]
        matched = any(_values_equal(actual, item) for item in expected_values)
        return matched if op == "in" else not matched
    if op == "regex":
        try:
            return re.search(str(expected), str(actual or ""), flags=re.IGNORECASE) is not None
        except re.error:
            return False
    matched = _values_equal(actual, expected)
    return not matched if op == "ne" else matched


def discover_record_collections(data: Any) -> list[RecordCollectionCandidate]:
    candidates: list[RecordCollectionCandidate] = []
    if isinstance(data, (list, tuple)) and data:
        records = _sample_records(data, limit=5)
        fields = list(records[0].keys()) if records else []
        if fields:
            candidates.append(
                RecordCollectionCandidate(
                    path=None,
                    kind="list[dict]" if all(isinstance(item, Mapping) for item in data[:5]) else "list[object]",
                    count=len(data),
                    fields=fields,
                    sample=records[:3],
                    score=3.0,
                    supports_mutation=True,
                    original_type=type(data).__name__,
                )
            )
    for item in find_record_collections(data, max_depth=5):
        path = str(item.get("path") or "")
        normalized_path = None if path in {"", "<root>"} else path
        fields = [str(field) for field in item.get("fields") or []]
        kind = str(item.get("kind") or type(get_collection_at_path(data, normalized_path)).__name__)
        count = int(item.get("count") or 0)
        score = 1.0 + min(count, 1000) / 1000
        if normalized_path is None:
            score += 1.5
        if normalized_path in {"customers", "readings", "orders", "tables.orders"}:
            score += 1.0
        candidates.append(
            RecordCollectionCandidate(
                path=normalized_path,
                kind=kind,
                count=count,
                fields=fields,
                sample=list(item.get("sample") or []),
                score=score,
                supports_mutation=_supports_collection_mutation(data, normalized_path),
                original_type=kind,
            )
        )
    if isinstance(data, pd.DataFrame) and not any(candidate.path is None for candidate in candidates):
        candidates.insert(
            0,
            RecordCollectionCandidate(
                path=None,
                kind="dataframe",
                count=int(len(data)),
                fields=[str(column) for column in data.columns.tolist()],
                sample=data.head(3).to_dict(orient="records"),
                score=3.0,
                supports_mutation=True,
                original_type="DataFrame",
            ),
        )
    _discover_nested_mapping_sequences(data, candidates, path="", depth=0, max_depth=5)
    return _dedupe_candidates(candidates)


def discover_filter_fields(data: Any, collections: list[RecordCollectionCandidate] | None = None) -> list[FieldCandidate]:
    fields: list[FieldCandidate] = []
    for collection in collections or discover_record_collections(data):
        records = _sample_records(get_collection_at_path(data, collection.path), limit=25)
        field_values: dict[str, list[Any]] = {}
        for record in records:
            for path, value in _flatten_mapping(record).items():
                field_values.setdefault(path, []).append(value)
        for path, values in field_values.items():
            non_missing = [value for value in values if not is_missing_value(value)]
            unique = _unique_values(non_missing)
            fields.append(
                FieldCandidate(
                    collection_path=collection.path,
                    field_path=path,
                    field_name=path.split(".")[-1],
                    type=_infer_field_type(non_missing),
                    value_examples=non_missing[:5],
                    unique_values_sample=unique[:10],
                    non_null_count=len(non_missing),
                    unique_count=len(unique),
                    score=_field_score(path, unique),
                    semantic_tags=_semantic_tags(path, unique),
                )
            )
    return fields


def parse_mutation_request(
    message: str,
    data: Any,
    *,
    target_dataset_id: str | None = None,
    target_dataset_name: str | None = None,
) -> MutationParseOutcome:
    lowered = _normalize_text(message)
    if _is_ambiguous_cleanup(lowered):
        return MutationParseOutcome(clarification=_clarification("Choose a cleaning rule", "what exact rule should define the records to remove?"))
    if not _looks_like_mutation_request(lowered):
        return MutationParseOutcome(reason="not_mutation")
    if "non active" in lowered or re.search(r"\b(?:bad|irrelevant|low quality)\b", lowered):
        return MutationParseOutcome(clarification=_clarification("Clarify the filter", "which field and value should define the records to keep or remove?"))

    collections = discover_record_collections(data)
    fields = discover_filter_fields(data, collections)

    spec = _parse_positional_delete(lowered)
    if spec is None:
        spec = _parse_full_delete(lowered)
    if spec is None:
        spec = _parse_derived_field(lowered, fields)
    if spec is None:
        spec = _parse_duplicate_drop(lowered, fields)
    if spec is None:
        spec = _parse_missing_field(lowered, fields)
    if spec is None:
        spec = _parse_numeric_filter(lowered, fields)
    if spec is None:
        spec = _parse_discrete_filter(lowered, fields)

    if spec is None:
        return MutationParseOutcome(reason="unsupported_or_unclear")

    spec.target_dataset_id = target_dataset_id
    spec.target_dataset_name = target_dataset_name
    spec.source_prompt = message
    if spec.confidence < 0.7:
        return MutationParseOutcome(
            clarification=_clarification("Clarify the mutation", "; ".join(spec.ambiguity_reasons) or "the target field is ambiguous."),
            reason="low_confidence",
        )
    return MutationParseOutcome(spec=spec)


def looks_like_optimized_mutation_prompt(message: str) -> bool:
    lowered = _normalize_text(message)
    if _is_ambiguous_cleanup(lowered):
        return True
    return any(
        re.search(pattern, lowered)
        for pattern in (
            r"\b(?:delete|remove|drop)\s+(?:all\s+)?non[-\s]+[a-zA-Z]",
            r"\bkeep only\s+",
            r"\b(?:delete|remove|drop)\s+first\s+\d+\s+(?:entries|rows|records)\b",
            r"\b(?:delete|remove|drop)\s+last\s+\d+\s+(?:entries|rows|records)\b",
            r"\b(?:delete|remove|drop).*(?:missing|blank|empty|null)\s+[a-zA-Z_]",
            r"\b(?:drop|delete|remove)\s+duplicate",
            r"\b(?:delete|remove|drop)\s+[a-zA-Z][\w-]*\s+(?:orders|records|rows|entries|customers|readings|patents)\b",
            r"[a-zA-Z_][\w.]*\s+(?:below|under|less than|above|over|greater than)\s+-?\d",
            r"\badd\s+(?:a\s+)?(?:derived\s+)?field\s+",
            r"\bdelete everything\b",
            r"\b(?:clear|wipe)\s+(?:the\s+)?dataset\b",
        )
    )


def analyze_mutation_impact(data: Any, spec: MutationSpec) -> MutationImpact:
    if spec.kind in {"delete_first_n", "delete_last_n", "delete_all_records"}:
        collection = get_collection_at_path(data, spec.target_path)
        current_count = _collection_len(collection)
        if spec.kind == "delete_all_records":
            affected = current_count
        else:
            affected = min(max(int(spec.count or 0), 0), current_count)
        return MutationImpact(
            current_count=current_count,
            affected_count=affected,
            new_count=max(0, current_count - affected),
            matching_count=affected,
            target_path=spec.target_path,
            field_path=spec.field_path,
        )
    if spec.kind == "add_derived_field":
        collection = get_collection_at_path(data, spec.target_path)
        count = _collection_len(collection)
        return MutationImpact(current_count=count, affected_count=count, new_count=count, matching_count=count, target_path=spec.target_path, field_path=spec.source_field)

    collection = get_collection_at_path(data, spec.target_path)
    records = _iter_collection_items(collection)
    current_count = len(records)
    matching_items: list[Any] = []
    removed_counter: Counter[str] = Counter()
    kept_counter: Counter[str] = Counter()
    removed_examples: list[dict[str, Any]] = []
    kept_examples: list[dict[str, Any]] = []

    for item in records:
        matches = _item_matches(item, spec)
        removed = matches if spec.mode == "delete_matching" else not matches
        raw_value = fast_get_path(item, spec.field_path) if spec.field_path else None
        key = _value_key(raw_value)
        if removed:
            removed_counter[key] += 1
            if len(removed_examples) < 3:
                removed_examples.append(_compact_record(item))
        else:
            kept_counter[key] += 1
            if len(kept_examples) < 3:
                kept_examples.append(_compact_record(item))
        if matches:
            matching_items.append(item)

    affected_count = sum(removed_counter.values())
    return MutationImpact(
        current_count=current_count,
        affected_count=affected_count,
        new_count=current_count - affected_count,
        matching_count=len(matching_items),
        removed_value_counts=dict(removed_counter),
        kept_value_counts=dict(kept_counter),
        removed_examples=removed_examples,
        kept_examples=kept_examples,
        target_path=spec.target_path,
        field_path=spec.field_path,
    )


def apply_mutation_spec(data: Any, spec: MutationSpec) -> tuple[Any, dict[str, Any]]:
    impact = analyze_mutation_impact(data, spec)
    if spec.kind == "delete_first_n":
        new_data = _slice_collection_at_path(data, spec.target_path, start=min(int(spec.count or 0), impact.current_count), end=None)
    elif spec.kind == "delete_last_n":
        count = min(int(spec.count or 0), impact.current_count)
        end = -count if count else None
        new_data = _slice_collection_at_path(data, spec.target_path, start=0, end=end)
    elif spec.kind == "delete_all_records":
        new_data = replace_collection_at_path(data, spec.target_path, _empty_like_collection(get_collection_at_path(data, spec.target_path)))
    elif spec.kind in {"filter_records", "filter_records_at_path", "remove_missing_field"}:
        new_data = _filter_collection_tree(data, spec.target_path, lambda item: _keep_item(item, spec))
    elif spec.kind == "drop_duplicates":
        new_data = _drop_duplicates_at_path(data, spec)
    elif spec.kind == "add_derived_field":
        new_data = _add_derived_field_at_path(data, spec)
    else:
        raise ValueError(f"Unsupported optimized mutation kind: {spec.kind}")

    preview = impact.to_preview()
    preview.update(
        {
            "operation_kind": spec.kind,
            "human_summary": spec.human_summary,
            "field_path": spec.field_path,
            "operator": spec.operator,
            "value": spec.value,
        }
    )
    return new_data, preview


def operation_summary_for_spec(spec: MutationSpec) -> str:
    if spec.human_summary:
        return spec.human_summary
    path = f" at `{spec.target_path}`" if spec.target_path else ""
    if spec.kind == "delete_first_n":
        suffix = path or " from the current working dataset"
        return f"Delete the first {int(spec.count or 0):,} records{suffix}"
    if spec.kind == "delete_last_n":
        suffix = path or " from the current working dataset"
        return f"Delete the last {int(spec.count or 0):,} records{suffix}"
    if spec.kind == "delete_all_records":
        suffix = path or " from the current working dataset"
        return f"Delete all records{suffix}"
    if spec.kind == "add_derived_field":
        return f"Add derived field `{spec.derived_field}` based on `{spec.source_field}`{path}"
    if spec.kind == "drop_duplicates":
        fields = ", ".join(f"`{field}`" for field in spec.field_paths)
        return f"Drop duplicate records by {fields}{path}"
    action = "Keep" if spec.mode == "keep_matching" else "Remove"
    return f"{action} records{path} where `{spec.field_path}` {spec.operator} {spec.value!r}"


def pseudocode_for_spec(spec: MutationSpec) -> str:
    return "\n".join(
        [
            f"target_path = {spec.target_path!r}",
            f"field_path = {spec.field_path!r}",
            f"operator = {spec.operator!r}",
            f"value = {spec.value!r}",
            f"mode = {spec.mode!r}",
            "scan all records at target_path using fast path accessors",
            "create a new version with the filtered/transformed collection after confirmation",
        ]
    )


def confirmation_metadata_for_spec(spec: MutationSpec, impact: MutationImpact) -> dict[str, Any]:
    spec.estimated_current_count = impact.current_count
    spec.estimated_affected_count = impact.affected_count
    spec.estimated_new_count = impact.new_count
    return {
        "operation_kind": "mutation_spec",
        "mutation_spec": spec.to_dict(),
        "target_path": spec.target_path,
        "field": spec.field_path,
        "field_path": spec.field_path,
        "operator": spec.operator,
        "value": spec.value,
        "mode": spec.mode,
        "keep_value": spec.value if spec.mode == "keep_matching" and spec.operator == "eq" else None,
        "current_row_count": impact.current_count,
        "new_row_count": impact.new_count,
        "affected_count": impact.affected_count,
        "removed_value_counts": impact.removed_value_counts,
        "kept_value_counts": impact.kept_value_counts,
    }


def _parse_positional_delete(message: str) -> MutationSpec | None:
    first = re.search(r"\b(?:delete|remove|drop)\s+first\s+(\d+)\s+(?:entries|rows|records)\b", message)
    if first:
        count = int(first.group(1))
        return MutationSpec(
            kind="delete_first_n",
            count=count,
            confidence=0.98,
            human_summary=f"Delete the first {count:,} records from the current working dataset",
        )
    last = re.search(r"\b(?:delete|remove|drop)\s+last\s+(\d+)\s+(?:entries|rows|records)\b", message)
    if last:
        count = int(last.group(1))
        return MutationSpec(
            kind="delete_last_n",
            count=count,
            confidence=0.98,
            human_summary=f"Delete the last {count:,} records from the current working dataset",
        )
    return None


def _parse_full_delete(message: str) -> MutationSpec | None:
    if any(
        phrase in message
        for phrase in (
            "delete everything",
            "remove all records",
            "clear the dataset",
            "wipe this dataset",
            "delete all rows",
            "delete all records",
            "drop all data",
            "drop all rows",
            "drop all records",
        )
    ):
        return MutationSpec(
            kind="delete_all_records",
            confidence=0.99,
            human_summary="Delete all records from the current working dataset",
            ambiguity_reasons=[],
        )
    return None


def _parse_derived_field(message: str, fields: list[FieldCandidate]) -> MutationSpec | None:
    match = re.search(r"\badd\s+(?:a\s+)?(?:derived\s+)?field\s+(?:called\s+)?`?([a-zA-Z_][\w]*)`?\s+based on\s+`?([a-zA-Z_][\w]*)`?", message)
    if not match:
        match = re.search(r"\badd\s+`?([a-zA-Z_][\w]*_year)`?.*based on\s+`?([a-zA-Z_][\w]*_date)`?", message)
    if not match:
        match = re.search(r"\badd\s+`?([a-zA-Z_][\w]*)`?\s+based on\s+`?([a-zA-Z_][\w]*)`?", message)
    if not match:
        return None
    derived_field, source_field = match.group(1), match.group(2)
    source = _best_field(fields, source_field)
    if source is None:
        return MutationSpec(
            kind="add_derived_field",
            derived_field=derived_field,
            source_field=source_field,
            confidence=0.45,
            ambiguity_reasons=[f"Could not find source field `{source_field}`."],
        )
    return MutationSpec(
        kind="add_derived_field",
        target_path=source.collection_path,
        derived_field=derived_field,
        source_field=source.field_path,
        confidence=0.95,
        human_summary=f"Add derived field `{derived_field}` based on `{source.field_path}`",
    )


def _parse_duplicate_drop(message: str, fields: list[FieldCandidate]) -> MutationSpec | None:
    if "duplicate" not in message or not any(verb in message for verb in ("drop", "delete", "remove")):
        return None
    by_match = re.search(r"\bby\s+(.+)$", message)
    if not by_match:
        return MutationSpec(kind="drop_duplicates", confidence=0.4, ambiguity_reasons=["Duplicate key fields were not specified."])
    requested = [part.strip(" `.") for part in re.split(r",|\band\b", by_match.group(1)) if part.strip(" `.")]
    resolved: list[FieldCandidate] = []
    for field_name in requested:
        candidate = _best_field(fields, field_name)
        if candidate is not None:
            resolved.append(candidate)
    if not resolved:
        return MutationSpec(kind="drop_duplicates", confidence=0.4, ambiguity_reasons=["Could not resolve duplicate key fields."])
    target_path = resolved[0].collection_path
    field_paths = [item.field_path for item in resolved if item.collection_path == target_path]
    confidence = 0.9 if len(field_paths) == len(requested) else 0.65
    return MutationSpec(
        kind="drop_duplicates",
        target_path=target_path,
        field_paths=field_paths,
        confidence=confidence,
        human_summary="Drop duplicate records by " + ", ".join(f"`{field}`" for field in field_paths),
        ambiguity_reasons=[] if confidence >= 0.7 else ["Some duplicate key fields were not found."],
    )


def _parse_missing_field(message: str, fields: list[FieldCandidate]) -> MutationSpec | None:
    if not any(term in message for term in ("missing", "blank", "empty", "null")):
        return None
    if not any(verb in message for verb in ("delete", "remove", "drop")):
        return None
    match = re.search(r"(?:missing|blank|empty|null)\s+`?([a-zA-Z_][\w.]*)`?", message)
    if not match:
        match = re.search(r"`?([a-zA-Z_][\w.]*)`?\s+(?:is\s+)?(?:missing|blank|empty|null)", message)
    field_name = match.group(1) if match else ""
    candidate = _best_field(fields, field_name) if field_name else None
    if candidate is None:
        return MutationSpec(kind="remove_missing_field", confidence=0.45, ambiguity_reasons=["Could not identify the missing-value field."])
    return MutationSpec(
        kind="remove_missing_field",
        target_path=candidate.collection_path,
        field_path=candidate.field_path,
        operator="is_missing",
        mode="delete_matching",
        confidence=0.93,
        human_summary=f"Remove records where `{candidate.field_path}` is missing",
    )


def _parse_numeric_filter(message: str, fields: list[FieldCandidate]) -> MutationSpec | None:
    patterns = [
        (r"`?([a-zA-Z_][\w.]*)`?\s+(?:is\s+)?(?:below|under|less than)\s+(-?\d+(?:\.\d+)?)", "gte", "keep_matching"),
        (r"`?([a-zA-Z_][\w.]*)`?\s+(?:is\s+)?(?:above|over|greater than)\s+(-?\d+(?:\.\d+)?)", "lte", "keep_matching"),
        (r"`?([a-zA-Z_][\w.]*)`?\s*<\s*(-?\d+(?:\.\d+)?)", "gte", "keep_matching"),
        (r"`?([a-zA-Z_][\w.]*)`?\s*>\s*(-?\d+(?:\.\d+)?)", "lte", "keep_matching"),
        (r"`?([a-zA-Z_][\w.]*)`?\s*>=\s*(-?\d+(?:\.\d+)?)", "gte", "keep_matching"),
        (r"`?([a-zA-Z_][\w.]*)`?\s*<=\s*(-?\d+(?:\.\d+)?)", "lte", "keep_matching"),
    ]
    for pattern, operator, mode in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        field_name = match.group(1)
        value = float(match.group(2))
        candidate = _best_field(fields, field_name)
        if candidate is None:
            return MutationSpec(kind="filter_records_at_path", field_path=field_name, confidence=0.45, ambiguity_reasons=[f"Could not find field `{field_name}`."])
        verb = "Keep" if mode == "keep_matching" else "Remove"
        return MutationSpec(
            kind="filter_records_at_path",
            target_path=candidate.collection_path,
            field_path=candidate.field_path,
            operator=operator,  # keep threshold side by converting below/above into retained records
            value=value,
            mode=mode,
            confidence=0.92,
            human_summary=f"{verb} records where `{candidate.field_path}` is {operator} {value:g}",
        )
    return None


def _parse_discrete_filter(message: str, fields: list[FieldCandidate]) -> MutationSpec | None:
    explicit = _parse_explicit_field_filter(message, fields)
    if explicit is not None:
        return explicit

    # "delete all non Japan entries" / "keep only enterprise customers" / "drop failed orders"
    keep_mode = "keep_matching" if any(phrase in message for phrase in ("keep only", "filter to only", "filter the dataset to only", "only ")) else None
    delete_inverse = re.search(r"\b(?:delete|remove|drop)\s+(?:all\s+)?non[-\s]+([a-zA-Z][\w-]*)", message)
    if delete_inverse:
        keep_mode = "keep_matching"
        value_text = delete_inverse.group(1)
    else:
        keep_match = re.search(r"\bkeep only\s+(.+?)(?:\s+(?:records|rows|entries|customers|orders|readings|patents|dataset)|$)", message)
        value_text = keep_match.group(1).strip() if keep_match else ""
    if not value_text:
        from_match = re.search(r"\bkeep only\s+(?:records|rows|entries|readings|sensors)\s+from\s+(.+)$", message)
        if from_match:
            value_text = from_match.group(1).strip()
            keep_mode = "keep_matching"
    if not value_text:
        drop_match = re.search(r"\b(?:drop|delete|remove)\s+([a-zA-Z][\w-]*)\s+(orders|records|rows|entries|customers|readings|patents)\b", message)
        if drop_match:
            value_text = drop_match.group(1)
            keep_mode = "delete_matching"
    if not value_text:
        return None

    target_hint = _target_hint(message)
    field = _field_for_value(value_text, fields, target_hint=target_hint, message=message)
    if field is None:
        return MutationSpec(kind="filter_records_at_path", confidence=0.45, ambiguity_reasons=[f"Could not identify which field contains `{value_text}`."])
    value = _resolve_value_alias(value_text, field)
    operator = "eq"
    mode: MutationMode = "keep_matching" if keep_mode == "keep_matching" else "delete_matching"
    action = "Keep" if mode == "keep_matching" else "Remove"
    human_summary = f"{action} records where `{field.field_path}` is `{value}`"
    if delete_inverse and "country" in field.semantic_tags:
        human_summary = f"Delete all non-{value} records and keep only {value} records"
    return MutationSpec(
        kind="filter_records_at_path" if field.collection_path else "filter_records",
        target_path=field.collection_path,
        field_path=field.field_path,
        operator=operator,
        value=value,
        mode=mode,
        confidence=0.9,
        human_summary=human_summary,
    )


def _parse_explicit_field_filter(message: str, fields: list[FieldCandidate]) -> MutationSpec | None:
    patterns = [
        (r"`?([a-zA-Z_][\w.]*)`?\s*(?:!=|<>|\bis\s+not\b|\bnot\s+in\b)\s*`?([a-zA-Z0-9_. -]+)`?", "ne", "delete_matching"),
        (r"`?([a-zA-Z_][\w.]*)`?\s*(?:==|=|\bis\b)\s*`?([a-zA-Z0-9_. -]+)`?", "eq", "keep_matching"),
        (r"where\s+`?([a-zA-Z_][\w.]*)`?\s+(?:is\s+)?`?([a-zA-Z0-9_. -]+)`?", "eq", "delete_matching"),
    ]
    for pattern, operator, default_mode in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        requested_field, value_text = match.group(1).strip(), match.group(2).strip(" .`")
        field = _best_field(fields, requested_field)
        if field is None:
            return MutationSpec(kind="filter_records_at_path", field_path=requested_field, confidence=0.45, ambiguity_reasons=[f"Could not find field `{requested_field}`."])
        value = _resolve_value_alias(value_text, field)
        mode: MutationMode = "keep_matching" if ("keep" in message or "filter" in message or default_mode == "keep_matching") else "delete_matching"
        if operator == "ne" and any(term in message for term in ("drop", "delete", "remove")):
            # Removing records where country != JP is equivalent to keeping JP.
            operator = "eq"
            mode = "keep_matching"
        return MutationSpec(
            kind="filter_records_at_path" if field.collection_path else "filter_records",
            target_path=field.collection_path,
            field_path=field.field_path,
            operator=operator,
            value=value,
            mode=mode,
            confidence=0.92,
            human_summary=operation_summary_for_spec(
                MutationSpec(kind="filter_records", field_path=field.field_path, operator=operator, value=value, mode=mode)
            ),
        )
    return None


def _item_matches(item: Any, spec: MutationSpec) -> bool:
    if spec.kind == "drop_duplicates":
        return False
    actual = fast_get_path(item, spec.field_path)
    return compare_value(actual, spec.operator, spec.value)


def _keep_item(item: Any, spec: MutationSpec) -> bool:
    matches = _item_matches(item, spec)
    return not matches if spec.mode == "delete_matching" else matches


def _filter_collection_tree(root: Any, target_path: str | None, predicate: Any) -> Any:
    tokens = _path_tokens(target_path)
    if not tokens:
        collection = get_collection_at_path(root, None)
        return _filter_collection(collection, predicate)
    return _filter_at_tokens(root, tokens, predicate)


def _filter_at_tokens(value: Any, tokens: list[str], predicate: Any) -> Any:
    if not tokens:
        return _filter_collection(value, predicate)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return type(value)(_filter_at_tokens(item, tokens, predicate) for item in value) if isinstance(value, tuple) else [_filter_at_tokens(item, tokens, predicate) for item in value]
    token = tokens[0]
    child = fast_get_field(value, token)
    if child is None:
        raise ValueError(f"Path not found for optimized mutation: {'.'.join(tokens)}")
    copied = _copy_container(value)
    updated_child = _filter_at_tokens(child, tokens[1:], predicate)
    _set_child(copied, token, updated_child)
    return copied


def _filter_collection(collection: Any, predicate: Any) -> Any:
    if isinstance(collection, pd.DataFrame):
        mask = [bool(predicate(row)) for _, row in collection.iterrows()]
        return collection.loc[mask].copy()
    if isinstance(collection, tuple):
        return tuple(item for item in collection if predicate(item))
    if isinstance(collection, list):
        return [item for item in collection if predicate(item)]
    if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray)):
        return [item for item in collection if predicate(item)]
    raise ValueError("Target collection does not support optimized filtering")


def _slice_collection_at_path(data: Any, target_path: str | None, *, start: int | None, end: int | None) -> Any:
    collection = get_collection_at_path(data, target_path)
    if isinstance(collection, pd.DataFrame):
        sliced = collection.iloc[start:end].copy()
    elif isinstance(collection, tuple):
        sliced = collection[slice(start, end)]
    elif isinstance(collection, list):
        sliced = collection[slice(start, end)]
    elif isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray)):
        sliced = list(collection)[slice(start, end)]
    else:
        raise ValueError("Target collection does not support positional deletion")
    return replace_collection_at_path(data, target_path, sliced)


def _drop_duplicates_at_path(data: Any, spec: MutationSpec) -> Any:
    seen: set[tuple[Any, ...]] = set()

    def keep(item: Any) -> bool:
        key = tuple(_hashable(fast_get_path(item, field)) for field in spec.field_paths)
        if key in seen:
            return False
        seen.add(key)
        return True

    return _filter_collection_tree(data, spec.target_path, keep)


def _add_derived_field_at_path(data: Any, spec: MutationSpec) -> Any:
    collection = get_collection_at_path(data, spec.target_path)
    derived_field = spec.derived_field or ""
    source_field = spec.source_field or ""
    if not derived_field or not source_field:
        raise ValueError("Derived field mutation requires derived_field and source_field")
    if isinstance(collection, pd.DataFrame):
        new_collection = collection.copy()
        source_column = source_field.split(".")[-1]
        if source_column not in new_collection.columns:
            source_column = source_field
        if source_column not in new_collection.columns:
            raise ValueError(f"Source field not found: {source_field}")
        if derived_field.endswith("_year"):
            new_collection[derived_field] = pd.to_datetime(new_collection[source_column], errors="coerce").dt.year.astype("Int64")
        else:
            new_collection[derived_field] = new_collection[source_column]
        return replace_collection_at_path(data, spec.target_path, new_collection)

    def mutate_item(item: Any) -> Any:
        raw = fast_get_path(item, source_field)
        value = _derive_value(raw, derived_field)
        return _copy_with_path(item, derived_field, value)

    return _map_collection_tree(data, spec.target_path, mutate_item)


def _map_collection_tree(root: Any, target_path: str | None, mapper: Any) -> Any:
    collection = get_collection_at_path(root, target_path)
    if isinstance(collection, tuple):
        mapped = tuple(mapper(item) for item in collection)
    elif isinstance(collection, list):
        mapped = [mapper(item) for item in collection]
    elif isinstance(collection, pd.DataFrame):
        mapped = collection.apply(mapper, axis=1)
    else:
        raise ValueError("Target collection does not support derived-field mutation")
    return replace_collection_at_path(root, target_path, mapped)


def _copy_with_path(item: Any, path: str, value: Any) -> Any:
    copied = _copy_container(item)
    tokens = _path_tokens(path)
    if not tokens:
        return value
    _set_child_path(copied, tokens, value)
    return copied


def _derive_value(raw: Any, derived_field: str) -> Any:
    if derived_field.endswith("_year"):
        if raw is None:
            return None
        if hasattr(raw, "year"):
            try:
                return int(raw.year)
            except Exception:
                return None
        parsed = pd.to_datetime(raw, errors="coerce")
        return None if pd.isna(parsed) else int(parsed.year)
    return raw


def _copy_container(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    try:
        return copy.copy(value)
    except Exception:
        return object_to_record(value)


def _set_child_path(root: Any, tokens: list[str], value: Any) -> None:
    current = root
    for token in tokens[:-1]:
        child = fast_get_field(current, token)
        if child is None:
            child = {}
            _set_child(current, token, child)
        current = child
    _set_child(current, tokens[-1], value)


def _set_child(obj: Any, field_name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[field_name] = value
        return
    if isinstance(obj, pd.Series):
        obj[field_name] = value
        return
    setattr(obj, field_name, value)


def _iter_collection_items(collection: Any) -> list[Any]:
    if collection is None:
        return []
    if isinstance(collection, pd.DataFrame):
        return [row for _, row in collection.iterrows()]
    if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray)):
        return list(collection)
    return [collection]


def _collection_len(collection: Any) -> int:
    if collection is None:
        return 0
    try:
        return int(len(collection))  # type: ignore[arg-type]
    except Exception:
        return 1


def _empty_like_collection(collection: Any) -> Any:
    if isinstance(collection, pd.DataFrame):
        return collection.iloc[0:0].copy()
    if isinstance(collection, tuple):
        return tuple()
    if isinstance(collection, list):
        return []
    if isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray)):
        return []
    raise ValueError("Target collection does not support empty replacement")


def _sample_records(collection: Any, limit: int = 25) -> list[dict[str, Any]]:
    if collection is None:
        return []
    if isinstance(collection, pd.DataFrame):
        return [_flatten_mapping(record) for record in collection.head(limit).to_dict(orient="records")]
    items = _iter_collection_items(collection)[:limit]
    return [_flatten_mapping(object_to_record(item)) for item in items]


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, Mapping):
            output.update(_flatten_mapping(child, path))
        else:
            output[path] = child
    return output


def _best_field(fields: list[FieldCandidate], requested: str | None, *, target_hint: str | None = None) -> FieldCandidate | None:
    if not requested:
        return None
    normalized = _field_token(requested)
    normalized_singular = normalized[:-1] if normalized.endswith("s") else normalized
    candidates = []
    for candidate in fields:
        field_tokens = {_field_token(candidate.field_path), _field_token(candidate.field_name), candidate.field_path.lower(), candidate.field_name.lower()}
        score = candidate.score
        if normalized in field_tokens or normalized_singular in field_tokens:
            score += 10
        elif normalized and any(
            token.endswith(normalized)
            or normalized.endswith(token)
            or token.endswith(normalized_singular)
            or normalized_singular.endswith(token)
            for token in field_tokens
        ):
            score += 5
        else:
            continue
        if target_hint and candidate.collection_path and target_hint in candidate.collection_path.lower():
            score += 3
        candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _field_for_value(value_text: str, fields: list[FieldCandidate], *, target_hint: str | None, message: str) -> FieldCandidate | None:
    semantic = _semantic_from_value(value_text, message)
    candidates: list[tuple[float, FieldCandidate]] = []
    for candidate in fields:
        score = candidate.score
        if target_hint and candidate.collection_path and target_hint in candidate.collection_path.lower():
            score += 4
        if target_hint and candidate.field_path.lower().startswith(target_hint.lower()):
            score += 1
        if semantic and semantic in candidate.semantic_tags:
            score += 8
        resolved = _resolve_value_alias(value_text, candidate)
        if any(_values_equal(resolved, observed) for observed in candidate.unique_values_sample):
            score += 10
        if score > 8:
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_value_alias(value_text: str, field: FieldCandidate) -> Any:
    text = value_text.strip(" `.'\"")
    lowered = _normalize_text(text)
    if "country" in field.semantic_tags:
        return normalize_country_value(text)
    if "status" in field.semantic_tags:
        if lowered == "granted":
            for observed in field.unique_values_sample:
                if "grant" in str(observed).lower():
                    return observed
        return _match_observed(text, field.unique_values_sample)
    if "segment" in field.semantic_tags or "plan" in field.semantic_tags:
        return _match_observed(text, field.unique_values_sample)
    if "site" in field.semantic_tags:
        aliases = {"plant a": "Plant-A", "plant-a": "Plant-A", "plant b": "Plant-B", "plant-b": "Plant-B"}
        return _match_observed(aliases.get(lowered, text), field.unique_values_sample)
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    return _match_observed(text, field.unique_values_sample)


def _match_observed(value_text: Any, observed_values: list[Any]) -> Any:
    for observed in observed_values:
        if _values_equal(value_text, observed):
            return observed
    return value_text


def _semantic_from_value(value_text: str, message: str) -> str | None:
    lowered = _normalize_text(value_text)
    if normalize_country_value(value_text) in {"JP", "CN", "US", "EP", "WO", "CA", "KR"} or "country" in message:
        return "country"
    if lowered in {"granted", "failed", "refunded", "paid", "active"} or "status" in message:
        return "status"
    if lowered in {"enterprise", "pro", "free", "starter", "consumer", "startup"}:
        return "segment"
    if "plant" in lowered or "site" in message:
        return "site"
    return None


def _target_hint(message: str) -> str | None:
    for hint in ("orders", "users", "customers", "readings", "sensors", "daily_metrics"):
        if hint in message:
            return hint
    return None


def _semantic_tags(path: str, values: list[Any]) -> list[str]:
    lowered = path.lower()
    tags: list[str] = []
    if "country" in lowered or "jurisdiction" in lowered:
        tags.append("country")
    if "status" in lowered:
        tags.append("status")
    if "date" in lowered or lowered.endswith("_at"):
        tags.append("date")
    if "title" in lowered:
        tags.append("title")
    if lowered.endswith("_id") or lowered == "id" or "number" in lowered:
        tags.append("id")
    if "category" in lowered:
        tags.append("category")
    if "segment" in lowered:
        tags.append("segment")
    if "plan" in lowered:
        tags.append("plan")
    if "site" in lowered:
        tags.append("site")
    if "battery" in lowered:
        tags.extend(["battery", "metric"])
    if "metric" in lowered or any(isinstance(value, (int, float, np.number)) for value in values):
        tags.append("metric")
    if "alert" in lowered:
        tags.append("alert")
    if "priority" in lowered:
        tags.append("priority")
    return list(dict.fromkeys(tags))


def _field_score(path: str, values: list[Any]) -> float:
    tags = _semantic_tags(path, values)
    return 1.0 + len(tags) * 0.5


def _infer_field_type(values: list[Any]) -> str:
    if not values:
        return "unknown"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in values):
        return "number"
    if all(isinstance(value, (list, tuple, set)) for value in values):
        return "list"
    return "text"


def _unique_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if any(_values_equal(value, existing) for existing in unique):
            continue
        unique.append(value)
        if len(unique) >= 50:
            break
    return unique


def _values_equal(left: Any, right: Any) -> bool:
    if is_missing_value(left) and is_missing_value(right):
        return True
    if isinstance(left, str) or isinstance(right, str):
        return _normalize_text(str(left)) == _normalize_text(str(right))
    return left == right


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        return _normalize_text(str(expected)) in _normalize_text(actual)
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
        return any(_values_equal(item, expected) for item in actual)
    return False


def _to_number(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return float(value)
    except Exception:
        return None


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, dict, set)):
        return repr(value)
    return value


def _compact_record(item: Any) -> dict[str, Any]:
    record = object_to_record(item)
    compact: dict[str, Any] = {}
    for key, value in record.items():
        if str(key).startswith("__"):
            continue
        compact[str(key)] = _compact_value(value)
        if len(compact) >= 8:
            break
    return compact


def _compact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _compact_value(child) for key, child in list(value.items())[:4]}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact_value(item) for item in list(value)[:3]]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _value_key(value: Any) -> str:
    if is_missing_value(value):
        return "Missing"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return type(value).__name__


def _looks_like_mutation_request(message: str) -> bool:
    return any(
        marker in message
        for marker in (
            "delete",
            "remove",
            "drop",
            "keep only",
            "filter",
            "persist",
            "clear",
            "wipe",
            "add ",
        )
    )


def _is_ambiguous_cleanup(message: str) -> bool:
    return any(
        phrase in message
        for phrase in (
            "clean this dataset",
            "remove bad records",
            "drop bad records",
            "delete bad rows",
            "fix the data",
            "drop everything irrelevant",
            "remove irrelevant records",
            "clean up the data",
        )
    )


def _clarification(title: str, message: str) -> dict[str, Any]:
    return {
        "title": title,
        "message": f"I need one clarification before making a destructive data change: {message}",
        "options": [
            {"id": "missing_values", "label": "Remove missing values", "message": "Remove records where a specific field is missing. Ask me which field first."},
            {"id": "duplicates", "label": "Remove duplicates", "message": "Remove duplicate records based on specified fields, but ask for confirmation first."},
            {"id": "filter_field", "label": "Filter by field", "message": "Filter records by a field and value. Ask me which field and value first."},
        ],
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _field_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _path_tokens(path: str | None) -> list[str]:
    return [part.replace("[]", "").strip() for part in str(path or "").split(".") if part.replace("[]", "").strip()]


def _dedupe_candidates(candidates: list[RecordCollectionCandidate]) -> list[RecordCollectionCandidate]:
    seen: set[str | None] = set()
    output: list[RecordCollectionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.path in seen:
            continue
        seen.add(candidate.path)
        output.append(candidate)
    return output


def _discover_nested_mapping_sequences(
    value: Any,
    candidates: list[RecordCollectionCandidate],
    *,
    path: str,
    depth: int,
    max_depth: int,
) -> None:
    if depth > max_depth:
        return
    if isinstance(value, pd.DataFrame):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _discover_nested_mapping_sequences(child, candidates, path=_join_path(path, str(key)), depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if value and all(isinstance(item, Mapping) for item in list(value)[:5]):
            records = [_flatten_mapping(item) for item in list(value)[:5]]
            fields = sorted({field_name for record in records for field_name in record})
            if fields:
                candidates.append(
                    RecordCollectionCandidate(
                        path=path or None,
                        kind="list[dict]",
                        count=len(value),
                        fields=fields,
                        sample=records[:3],
                        score=2.5,
                        supports_mutation=True,
                        original_type="list",
                    )
                )
        for item in list(value)[:5]:
            _discover_nested_mapping_sequences(item, candidates, path=path, depth=depth + 1, max_depth=max_depth)
        return
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, Mapping):
        for key, child in attrs.items():
            if str(key).startswith("_"):
                continue
            _discover_nested_mapping_sequences(child, candidates, path=_join_path(path, str(key)), depth=depth + 1, max_depth=max_depth)


def _join_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _supports_collection_mutation(data: Any, path: str | None) -> bool:
    try:
        collection = get_collection_at_path(data, path)
    except Exception:
        return False
    return isinstance(collection, (pd.DataFrame, list, tuple)) or (
        isinstance(collection, Sequence) and not isinstance(collection, (str, bytes, bytearray))
    )
