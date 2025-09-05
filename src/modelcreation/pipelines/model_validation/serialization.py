from __future__ import annotations

"""Reusable serialization helpers for analysis artifact structures.

This module centralizes logic for converting raw analysis objects, pandas
DataFrames, nested dicts, and custom analysis objects exposing
`get_data_and_metadata()` into JSON-serializable dictionaries for logging.
"""
from typing import Any, Dict
import logging

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore
try:  # Lightweight optional pandas handling
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None  # type: ignore

__all__ = ["serialize_tables"]


def _serialize_data(obj: Any) -> Any:
    """Recursively convert heterogeneous objects into JSON-friendly primitives.

    Handling:
    - pandas DataFrame/Series: list-of-records if possible else dict
    - dict: recurse
    - numpy arrays: tolist()
    - numpy scalars: item()
    - pandas Index: list
    - set / frozenset: list
    - tuples/lists: recurse
    - unsupported custom objects: repr(obj) as last resort
    """
    # Fast path for simple builtins
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # pandas-like structures
    if hasattr(obj, "to_dict") and not isinstance(obj, dict):
        try:
            return obj.to_dict(orient="records")  # type: ignore[arg-type]
        except TypeError:
            try:
                return obj.to_dict()  # type: ignore[attr-defined]
            except Exception:
                return repr(obj)

    # numpy arrays
    if _np is not None and isinstance(obj, getattr(_np, 'ndarray', ())):  # type: ignore[arg-type]
        return obj.tolist()
    # numpy scalar
    if _np is not None and isinstance(obj, tuple(getattr(_np, t, ()) for t in [
        'integer', 'floating', 'bool_', 'number'
    ])):
        try:
            return obj.item()  # type: ignore[attr-defined]
        except Exception:
            return float(obj) if hasattr(obj, '__float__') else repr(obj)

    # pandas Index
    if _pd is not None and isinstance(obj, getattr(_pd, 'Index', ())):  # type: ignore[arg-type]
        return list(obj)

    # Mapping
    if isinstance(obj, dict):
        return {k: _serialize_data(v) for k, v in obj.items()}

    # Set-like
    if isinstance(obj, (set, frozenset)):
        return [_serialize_data(v) for v in obj]

    # Sequence (avoid treating strings again)
    if isinstance(obj, (list, tuple)):
        return [_serialize_data(v) for v in obj]

    # Objects with tolist()
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()  # type: ignore[attr-defined]
        except Exception as exc:
            logging.debug("tolist() serialization failed for %r: %s", obj, exc)

    # Dataclasses / objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {k: _serialize_data(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            return repr(obj)

    return repr(obj)


def _expand_analysis_obj(obj: Any) -> Any:
    """Expand objects exposing get_data_and_metadata() into serializable dicts."""
    if hasattr(obj, "get_data_and_metadata"):
        meta = obj.get_data_and_metadata()
        ser_meta: Dict[str, Any] = {}
        for mk, mv in meta.items():
            entry = dict(mv)
            # Serialize every field inside the metadata entry (not only 'data')
            for ek, ev in list(entry.items()):
                entry[ek] = _serialize_data(ev)
            ser_meta[mk] = entry
        return ser_meta
    return obj


def serialize_tables(tables: Any) -> Dict[str, Any]:
    """Normalize/serialize heterogeneous "tables" payload to a JSON-safe dict.

    Accepts:
        - dicts of analysis objects or DataFrames
        - raw analysis objects with get_data_and_metadata
        - dict containers with keys like *_df / *_data / statistics_df
        - plain pandas objects

    Returns a dict suitable for json.dump.
    """
    out: Dict[str, Any] = {}
    if tables is None:
        return out

    # Direct mapping path
    if isinstance(tables, dict):
        for key, value in tables.items():
            expanded = _expand_analysis_obj(value)
            if isinstance(expanded, dict) and any(
                k.endswith("_df") or k.endswith("_data") or k == "statistics_df" for k in expanded.keys()
            ):
                serial_container = {}
                for ck, cv in expanded.items():
                    serial_container[ck] = _serialize_data(cv)
                out[key] = serial_container
            else:
                out[key] = _serialize_data(expanded)

        # Deep sanitize everything to ensure JSON safety
        def _deep(obj: Any) -> Any:
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, list):
                return [_deep(v) for v in obj]
            if isinstance(obj, dict):
                return {str(k): _deep(v) for k, v in obj.items()}
            return _serialize_data(obj)

        return {k: _deep(v) for k, v in out.items()}

    # Single pandas-like object
    if hasattr(tables, "to_dict"):
        return _serialize_data(tables)

    raise TypeError("tables must be dict-like, an analysis object, or have to_dict()")
