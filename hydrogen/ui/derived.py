"""Client-side derived variables — aggregations over streamed data.

A *derived* variable is a synthetic watch key (``__derived__:<id>``) whose values
are computed on the GUI pump thread from data already on the host variable
stream.  No model rebuild is required.

Supported reductions:

  * **structural** (``axis="instances"``) — ``sum`` / ``mean`` / ``max`` /
    ``min`` / ``std`` across either an explicit ``sources`` list or every
    stream column matching a ``pattern`` (host suffix expansion via
    :meth:`~hydrogen.service.client.Stream.series_values`).
  * **temporal** (``axis="time"``) — ``integral`` (cumulative trapezoid),
    ``cumsum``, or ``abs`` (element-wise absolute value) along the run's time
    axis for a single ``sources[0]`` series.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import numpy as np

__all__ = [
    "DERIVED_PREFIX",
    "STRUCTURAL_OPS",
    "TEMPORAL_OPS",
    "is_derived",
    "make_derived_id",
    "make_derived_payload",
    "unit_for_agg",
    "compute_series",
    "filter_leaves_regex",
]

DERIVED_PREFIX = "__derived__:"

#: Structural (per-time-step) reductions across instances / explicit sources.
STRUCTURAL_OPS = ("sum", "mean", "max", "min", "std")

#: Temporal reductions / transforms along the time axis of one series.
TEMPORAL_OPS = ("integral", "cumsum", "abs")


def is_derived(full: str) -> bool:
    return isinstance(full, str) and full.startswith(DERIVED_PREFIX)


def make_derived_id() -> str:
    return f"{DERIVED_PREFIX}{uuid.uuid4().hex[:12]}"


def unit_for_agg(op: str, base_unit: str) -> str:
    """Best-effort result unit for an aggregation."""
    u = (base_unit or "").strip()
    if op == "integral":
        if u.endswith("/s"):
            return u[:-2]                    # kg/s -> kg
        if u:
            return f"{u}·s"
        return "s"
    return u


def _op_label(op: str) -> str:
    return {"integral": "∫", "cumsum": "Σ", "std": "σ", "abs": "|x|"}.get(op, op)


def make_derived_payload(
    *,
    op: str,
    axis: str,
    label: str,
    unit: str = "",
    description: str = "",
    sources: list[str] | None = None,
    pattern: str | None = None,
) -> dict:
    """Build a drag / row payload for a derived variable."""
    agg: dict[str, Any] = {"op": op, "axis": axis}
    if pattern:
        agg["pattern"] = pattern
    if sources:
        agg["sources"] = list(sources)
    return {
        "full": make_derived_id(),
        "label": label,
        "name": label.rsplit(".", 1)[-1],
        "unit": unit,
        "description": description or label,
        "kind": "derived",
        "value": None,
        "agg": agg,
    }


def _slice_pair(t, y, start_index: int):
    if t is None or y is None:
        return np.empty(0), np.empty(0)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(t), len(y))
    if n == 0:
        return np.empty(0), np.empty(0)
    i = max(0, min(int(start_index), n))
    return t[i:n], y[i:n]


def _reduce_instances(op: str, mat: np.ndarray) -> np.ndarray:
    """``mat`` is 2-D (rows=time, cols=instances) or 1-D."""
    if mat.ndim == 1:
        return mat
    if mat.size == 0:
        return np.empty(0)
    if op == "sum":
        return mat.sum(axis=1)
    if op == "mean":
        return mat.mean(axis=1)
    if op == "max":
        return mat.max(axis=1)
    if op == "min":
        return mat.min(axis=1)
    if op == "std":
        return mat.std(axis=1)
    raise ValueError(f"unknown structural op {op!r}")


def _cumulative_trapezoid(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    if y.size == 1:
        return np.zeros(1)
    dt = np.diff(t)
    mids = 0.5 * (y[:-1] + y[1:])
    inc = mids * dt
    return np.concatenate([[0.0], np.cumsum(inc)])


def _temporal(op: str, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if op == "integral":
        return _cumulative_trapezoid(y, t)
    if op == "cumsum":
        return np.cumsum(y)
    if op == "abs":
        return np.abs(y)
    raise ValueError(f"unknown temporal op {op!r}")


def compute_series(
    agg: dict,
    *,
    time_array,
    start_index: int,
    series_handles: dict[str, Any],
    values_handles: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one derived spec against live stream handles.

    ``series_handles`` maps recorded full names -> 1-D handles.
    ``values_handles`` maps pattern strings -> 2-D ``series_values`` handles.
    """
    op = agg.get("op", "sum")
    axis = agg.get("axis", "instances")

    if axis == "time":
        sources = agg.get("sources") or []
        if not sources:
            return np.empty(0), np.empty(0)
        h = series_handles.get(sources[0])
        if h is None:
            return np.empty(0), np.empty(0)
        t, y = _slice_pair(time_array, h.array, start_index)
        if t.size == 0:
            return t, y
        return t, _temporal(op, y, t)

    # Structural reduction at each time step.
    pattern = agg.get("pattern")
    sources = agg.get("sources")
    t_slice = None
    if pattern:
        h = values_handles.get(pattern)
        if h is None:
            return np.empty(0), np.empty(0)
        mat = np.asarray(h.array, dtype=float)
        t_arr = np.asarray(time_array, dtype=float) if time_array is not None else None
        if t_arr is None or mat.size == 0:
            return np.empty(0), np.empty(0)
        n = min(len(t_arr), mat.shape[0] if mat.ndim == 2 else len(mat))
        i = max(0, min(int(start_index), n))
        t_slice = t_arr[i:n]
        mat = mat[i:n]
        y_out = _reduce_instances(op, mat)
        return t_slice, y_out

    if not sources:
        return np.empty(0), np.empty(0)
    cols = []
    for src in sources:
        h = series_handles.get(src)
        if h is None:
            continue
        cols.append(np.asarray(h.array, dtype=float))
    if not cols:
        return np.empty(0), np.empty(0)
    n = min(len(c) for c in cols)
    t_arr = np.asarray(time_array, dtype=float) if time_array is not None else None
    if t_arr is not None:
        n = min(n, len(t_arr))
    i = max(0, min(int(start_index), n))
    if t_arr is not None:
        t_slice = t_arr[i:n]
    else:
        t_slice = np.arange(i, n, dtype=float)
    mat = np.column_stack([c[i:n] for c in cols])
    return t_slice, _reduce_instances(op, mat)


def filter_leaves_regex(leaves: list[dict], pattern: str) -> list[dict]:
    """Filter leaf payloads (each with ``full`` / ``label`` / ``description``)
    by a regex; invalid patterns fall back to substring match."""
    text = (pattern or "").strip()
    if not text:
        return list(leaves)
    try:
        rx = re.compile(text, re.IGNORECASE)
        def match(hay):
            return rx.search(hay) is not None
    except re.error:
        needle = text.lower()
        def match(hay):
            return needle in hay.lower()
    out = []
    for leaf in leaves:
        hay = f"{leaf.get('full', '')} {leaf.get('label', '')} " \
              f"{leaf.get('description', '')} {leaf.get('unit', '')}"
        if match(hay):
            out.append(leaf)
    return out
