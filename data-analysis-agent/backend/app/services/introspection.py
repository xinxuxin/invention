from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, asdict
from sys import getsizeof
from typing import Any

import numpy as np
import pandas as pd

MAX_DEPTH = 3
MAX_ITEMS = 5
MAX_REPR = 500
MAX_ATTRS = 20


def introspect_object(value: object) -> dict[str, Any]:
    warnings: list[str] = []
    value_type = type(value)

    profile: dict[str, Any] = {
        "object_type": value_type.__qualname__,
        "module": value_type.__module__,
        "repr_preview": _safe_repr(value, warnings),
        "approximate_size": _approximate_size(value, warnings=warnings),
        "warnings": warnings,
    }

    shape = getattr(value, "shape", None)
    if shape is not None:
        profile["shape"] = _json_safe(shape)

    if isinstance(value, pd.DataFrame):
        profile.update(_profile_dataframe(value, warnings))
    elif isinstance(value, pd.Series):
        profile.update(_profile_series(value, warnings))
    elif isinstance(value, np.ndarray):
        profile.update(_profile_ndarray(value, warnings))
    elif isinstance(value, Mapping):
        profile.update(_profile_mapping(value, warnings))
    elif _is_sequence(value):
        profile.update(_profile_sequence(value, warnings))

    public_attrs = _public_attributes(value, warnings)
    if public_attrs:
        profile["public_attributes"] = public_attrs

    profile["nested_summary"] = _nested_summary(value, depth=0, warnings=warnings)
    return _json_safe(profile)


def _profile_dataframe(value: pd.DataFrame, warnings: list[str]) -> dict[str, Any]:
    return {
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "columns": [_json_safe(column) for column in value.columns.tolist()],
        "dtypes": {str(column): str(dtype) for column, dtype in value.dtypes.items()},
        "index_preview": _json_safe(value.index[:MAX_ITEMS].tolist()),
        "sample_rows": _dataframe_records(value, warnings),
        "length": int(len(value)),
    }


def _profile_series(value: pd.Series, warnings: list[str]) -> dict[str, Any]:
    return {
        "shape": [int(value.shape[0])],
        "dtype": str(value.dtype),
        "name": _json_safe(value.name),
        "index_preview": _json_safe(value.index[:MAX_ITEMS].tolist()),
        "sample_items": _json_safe(value.head(MAX_ITEMS).tolist()),
        "length": int(len(value)),
    }


def _profile_ndarray(value: np.ndarray, warnings: list[str]) -> dict[str, Any]:
    return {
        "shape": [int(part) for part in value.shape],
        "dtype": str(value.dtype),
        "sample": _json_safe(_ndarray_sample(value, warnings)),
        "length": int(len(value)) if value.ndim > 0 else 1,
    }


def _profile_mapping(value: Mapping[Any, Any], warnings: list[str]) -> dict[str, Any]:
    items = list(value.items())[:MAX_ITEMS]
    return {
        "keys": [_json_safe(key) for key in list(value.keys())[:MAX_ITEMS]],
        "length": len(value),
        "sample_items": [
            {"key": _json_safe(key), "value": _preview_value(item_value, warnings)}
            for key, item_value in items
        ],
    }


def _profile_sequence(value: Sequence[Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "length": len(value),
        "sample_items": [_preview_value(item, warnings) for item in list(value[:MAX_ITEMS])],
        "keys": _common_mapping_keys(value),
    }


def _dataframe_records(value: pd.DataFrame, warnings: list[str]) -> list[dict[str, Any]]:
    try:
        safe_df = value.head(MAX_ITEMS).replace({np.nan: None})
        return _json_safe(safe_df.to_dict(orient="records"))
    except Exception as exc:  # pragma: no cover - defensive for exotic extension arrays
        warnings.append(f"Unable to create DataFrame sample rows: {exc}")
        return []


def _ndarray_sample(value: np.ndarray, warnings: list[str]) -> Any:
    try:
        if value.ndim == 0:
            return value.item()

        return value.reshape(-1)[:MAX_ITEMS].tolist()
    except Exception as exc:  # pragma: no cover - defensive for object arrays
        warnings.append(f"Unable to create ndarray sample: {exc}")
        return []


def _nested_summary(value: object, depth: int, warnings: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(value).__qualname__,
        "module": type(value).__module__,
    }

    shape = getattr(value, "shape", None)
    if shape is not None:
        summary["shape"] = _json_safe(shape)

    if depth >= MAX_DEPTH:
        summary["preview"] = _safe_repr(value, warnings, limit=120)
        return summary

    if isinstance(value, pd.DataFrame):
        summary["kind"] = "dataframe"
        summary["columns"] = [_json_safe(column) for column in value.columns[:MAX_ITEMS].tolist()]
        summary["length"] = int(len(value))
        return summary

    if isinstance(value, pd.Series):
        summary["kind"] = "series"
        summary["dtype"] = str(value.dtype)
        summary["length"] = int(len(value))
        return summary

    if isinstance(value, np.ndarray):
        summary["kind"] = "ndarray"
        summary["dtype"] = str(value.dtype)
        summary["sample"] = _json_safe(_ndarray_sample(value, warnings))
        return summary

    if isinstance(value, Mapping):
        summary["kind"] = "mapping"
        summary["length"] = len(value)
        summary["children"] = [
            {
                "key": _json_safe(key),
                "value": _nested_summary(item_value, depth + 1, warnings),
            }
            for key, item_value in list(value.items())[:MAX_ITEMS]
        ]
        return summary

    if _is_sequence(value):
        summary["kind"] = "sequence"
        summary["length"] = len(value)
        summary["items"] = [_nested_summary(item, depth + 1, warnings) for item in list(value[:MAX_ITEMS])]
        return summary

    attrs = _public_attributes(value, warnings, max_attrs=MAX_ITEMS)
    if attrs:
        summary["kind"] = "object"
        summary["attributes"] = {
            key: _nested_summary(attr_value, depth + 1, warnings)
            for key, attr_value in attrs.items()
        }
        return summary

    summary["preview"] = _safe_repr(value, warnings, limit=120)
    return summary


def _public_attributes(value: object, warnings: list[str], max_attrs: int = MAX_ATTRS) -> dict[str, Any]:
    if _is_scalar(value) or isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, Mapping)):
        return {}

    try:
        if is_dataclass(value) and not isinstance(value, type):
            raw_attrs = asdict(value)
        elif hasattr(value, "__dict__"):
            raw_attrs = vars(value)
        else:
            raw_attrs = {
                name: getattr(value, name)
                for name in dir(value)
                if not name.startswith("_") and not callable(getattr(value, name, None))
            }
    except Exception as exc:
        warnings.append(f"Unable to inspect public attributes: {exc}")
        return {}

    attrs: dict[str, Any] = {}
    for key, attr_value in raw_attrs.items():
        if str(key).startswith("_") or callable(attr_value):
            continue
        attrs[str(key)] = _preview_value(attr_value, warnings)
        if len(attrs) >= max_attrs:
            break

    return attrs


def _common_mapping_keys(value: Sequence[Any]) -> list[Any]:
    mapping_items = [item for item in value[:MAX_ITEMS] if isinstance(item, Mapping)]
    if not mapping_items:
        return []

    ordered_keys: list[Any] = []
    seen = set()
    for item in mapping_items:
        for key in item.keys():
            marker = repr(key)
            if marker not in seen:
                seen.add(marker)
                ordered_keys.append(_json_safe(key))
    return ordered_keys[:MAX_ITEMS]


def _preview_value(value: object, warnings: list[str]) -> Any:
    if _is_scalar(value):
        return _json_safe(value)

    if isinstance(value, np.ndarray):
        return {
            "type": type(value).__qualname__,
            "shape": [int(part) for part in value.shape],
            "dtype": str(value.dtype),
            "sample": _json_safe(_ndarray_sample(value, warnings)),
        }

    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [_json_safe(column) for column in value.columns[:MAX_ITEMS].tolist()],
        }

    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "shape": [int(value.shape[0])],
            "dtype": str(value.dtype),
            "sample_items": _json_safe(value.head(MAX_ITEMS).tolist()),
        }

    if isinstance(value, Mapping):
        return {
            "type": type(value).__qualname__,
            "length": len(value),
            "keys": [_json_safe(key) for key in list(value.keys())[:MAX_ITEMS]],
        }

    if _is_sequence(value):
        return {
            "type": type(value).__qualname__,
            "length": len(value),
            "sample_items": [_preview_value(item, warnings) for item in list(value[:MAX_ITEMS])],
        }

    return {
        "type": type(value).__qualname__,
        "module": type(value).__module__,
        "repr_preview": _safe_repr(value, warnings, limit=160),
    }


def _approximate_size(value: object, warnings: list[str]) -> int | None:
    seen: set[int] = set()

    def walk(item: object, depth: int) -> int:
        item_id = id(item)
        if item_id in seen:
            return 0
        seen.add(item_id)

        try:
            size = getsizeof(item)
        except Exception:
            return 0

        if depth >= MAX_DEPTH:
            return size

        if isinstance(item, Mapping):
            return size + sum(walk(key, depth + 1) + walk(val, depth + 1) for key, val in item.items())

        if _is_sequence(item):
            return size + sum(walk(child, depth + 1) for child in item[:MAX_ITEMS])

        if hasattr(item, "__dict__") and not isinstance(item, type):
            return size + walk(vars(item), depth + 1)

        return size

    try:
        return walk(value, 0)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.append(f"Unable to approximate object size: {exc}")
        return None


def _safe_repr(value: object, warnings: list[str], limit: int = MAX_REPR) -> str:
    try:
        preview = repr(value)
    except Exception as exc:
        warnings.append(f"Unable to create repr preview: {exc}")
        return "<repr unavailable>"

    if len(preview) > limit:
        return f"{preview[:limit]}..."

    return preview


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

    if isinstance(value, pd.Timedelta):
        return str(value)

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


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, bytes, bytearray, int, float, bool, np.generic))
