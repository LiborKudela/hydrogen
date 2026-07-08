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
  * **formula** (``axis="formula"``) — the unified path.  An arbitrary
    expression over named inputs, where each input is either a scalar
    ``variables`` entry (``alias -> recorded full name``) or a ``groups`` entry
    (``alias -> {regex, scope}``) that resolves to a 2-D matrix of every matching
    column, re-resolved against the live model each run.  The grammar offers
    element-wise math (``sqrt``, ``exp``, ``v1 - v2`` ...), instance reducers
    that collapse a group's columns per time step (``sum`` / ``mean`` / ``max`` /
    ``min`` / ``std``), and temporal transforms along the run's time axis
    (``integral`` = cumulative trapezoid, ``cumsum``).  So ``v1 - v2``,
    ``sum(g1)``, ``integral(sum(g1))`` and ``sqrt(v1**2 + v2**2)`` are all
    formulas.  The expression is parsed and whitelisted (no attribute access,
    indexing or builtins) so it can be evaluated safely on the pump thread.

The ``instances`` / ``time`` axes remain understood for loading older projects.
"""

from __future__ import annotations

import ast
import re
import uuid
from typing import Any

import numpy as np

__all__ = [
    "DERIVED_PREFIX",
    "STRUCTURAL_OPS",
    "TEMPORAL_OPS",
    "FORMULA_FUNCS",
    "FORMULA_REDUCERS",
    "FORMULA_TEMPORAL",
    "is_derived",
    "make_derived_id",
    "make_derived_payload",
    "unit_for_agg",
    "compute_series",
    "compile_formula",
    "evaluate_formula",
    "filter_leaves_regex",
    "resolve_regex_names",
]

DERIVED_PREFIX = "__derived__:"

#: Structural (per-time-step) reductions across instances / explicit sources.
STRUCTURAL_OPS = ("sum", "mean", "max", "min", "std")

#: Temporal reductions / transforms along the time axis of one series.
TEMPORAL_OPS = ("integral", "cumsum", "abs")

#: Whitelisted callables available inside a custom formula (all element-wise /
#: broadcasting ``numpy`` ufuncs; no I/O, no attribute access).
FORMULA_FUNCS: dict[str, Any] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "arctan2": np.arctan2, "hypot": np.hypot,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "expm1": np.expm1, "log": np.log, "log10": np.log10,
    "log2": np.log2, "log1p": np.log1p, "sqrt": np.sqrt, "cbrt": np.cbrt,
    "square": np.square, "power": np.power, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil, "round": np.round, "clip": np.clip,
    "minimum": np.minimum, "maximum": np.maximum, "mod": np.mod,
    "where": np.where,
}

#: Whitelisted constants.
FORMULA_CONSTS: dict[str, Any] = {
    "pi": np.pi, "e": np.e, "inf": np.inf, "nan": np.nan,
}


def _reduce_instances_fn(a, op: str):
    """Collapse a group's instance axis (columns) per time step.

    A 2-D input (rows=time, cols=instances) is reduced over ``axis=1``; a plain
    1-D series (a single-instance / scalar input) passes through unchanged, so
    ``sum(v1)`` is just ``v1``.
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim <= 1:
        return arr
    if arr.size == 0:
        return np.empty(0)
    return {
        "sum": lambda: arr.sum(axis=1),
        "mean": lambda: arr.mean(axis=1),
        "max": lambda: arr.max(axis=1),
        "min": lambda: arr.min(axis=1),
        "std": lambda: arr.std(axis=1),
    }[op]()


#: Instance reducers (group columns -> one series per time step).
FORMULA_REDUCERS: dict[str, Any] = {
    op: (lambda a, _op=op: _reduce_instances_fn(a, _op))
    for op in ("sum", "mean", "max", "min", "std")
}

#: Temporal transforms along the run's time axis (bound to the time array at
#: evaluation time by :func:`_temporal_namespace`).
FORMULA_TEMPORAL = ("integral", "cumsum")

#: Every name that may appear as a *call* target in a formula.
_FORMULA_CALL_NAMES = (
    frozenset(FORMULA_FUNCS) | frozenset(FORMULA_REDUCERS)
    | frozenset(FORMULA_TEMPORAL)
)


def _temporal_namespace(time) -> dict[str, Any]:
    """Temporal functions closed over the current (sliced) time array."""
    t = np.asarray(time, dtype=float) if time is not None else None

    def integral(x):
        y = np.asarray(x, dtype=float)
        if t is None or y.ndim != 1:
            return np.cumsum(y) * 0.0
        n = min(len(t), len(y))
        return _cumulative_trapezoid(y[:n], t[:n])

    def cumsum(x):
        return np.cumsum(np.asarray(x, dtype=float))

    return {"integral": integral, "cumsum": cumsum}

#: AST node types allowed in a formula expression.
_FORMULA_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.BoolOp, ast.Compare, ast.IfExp, ast.Tuple, ast.List,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


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
    regex: str | None = None,
    scope: str | None = None,
    expr: str | None = None,
    variables: dict[str, str] | None = None,
    groups: dict[str, dict] | None = None,
    full: str | None = None,
) -> dict:
    """Build a drag / row payload for a derived variable.

    A structural (``axis="instances"``) aggregate is defined by a **regex**
    plus a component **scope** rather than a frozen ``sources`` list, so the set
    of variables it reduces is re-resolved against the live model on every run.
    Changing e.g. a pipe's ``n_segments`` therefore grows/shrinks the aggregate
    automatically -- no need to rebuild the derived variable.  ``sources`` /
    ``pattern`` remain supported for backward compatibility with saved projects.

    A **formula** (``axis="formula"``) carries the ``expr`` string plus its
    inputs: ``variables`` (``alias -> recorded full name``) and/or ``groups``
    (``alias -> {"regex": ..., "scope": ...}``).  Pass ``full`` to keep an
    existing derived id (i.e. when editing in place).
    """
    agg: dict[str, Any] = {"op": op, "axis": axis}
    if regex is not None:
        agg["regex"] = regex
        if scope:
            agg["scope"] = scope
    if pattern:
        agg["pattern"] = pattern
    if sources:
        agg["sources"] = list(sources)
    if expr is not None:
        agg["expr"] = expr
    if variables:
        agg["variables"] = dict(variables)
    if groups:
        agg["groups"] = {a: dict(g) for a, g in groups.items()}
    return {
        "full": full or make_derived_id(),
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


def _compute_formula(
    agg: dict, *, time_array, start_index: int,
    series_handles: dict[str, Any],
    group_arrays: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an ``axis="formula"`` spec over its scalar + group inputs.

    ``series_handles`` supplies the 1-D ``variables`` columns; ``group_arrays``
    supplies each ``groups`` alias's 2-D matrix (rows=time, cols=instances).
    Every input must have backfilled data; a missing / empty one yields an empty
    result (the value shows a dash) rather than a partial evaluation.  A scalar
    result is broadcast to the run's length, and a result that fails to reduce to
    a single series (still 2-D) is rejected.
    """
    variables = agg.get("variables") or {}
    groups = agg.get("groups") or {}
    expr = agg.get("expr") or ""
    if not expr or (not variables and not groups):
        return np.empty(0), np.empty(0)
    group_arrays = group_arrays or {}
    cols: dict[str, np.ndarray] = {}
    lengths: list[int] = []
    for alias, full in variables.items():
        h = series_handles.get(full)
        if h is None:
            return np.empty(0), np.empty(0)
        arr = np.asarray(h.array, dtype=float)
        if arr.size == 0:
            return np.empty(0), np.empty(0)
        cols[alias] = arr
        lengths.append(len(arr))
    mats: dict[str, np.ndarray] = {}
    for alias in groups:
        m = group_arrays.get(alias)
        if m is None:
            return np.empty(0), np.empty(0)
        m = np.asarray(m, dtype=float)
        if m.ndim != 2 or m.size == 0 or m.shape[0] == 0:
            return np.empty(0), np.empty(0)
        mats[alias] = m
        lengths.append(m.shape[0])
    if not lengths:
        return np.empty(0), np.empty(0)
    t_arr = np.asarray(time_array, dtype=float) if time_array is not None else None
    n = min(lengths)
    if t_arr is not None:
        n = min(n, len(t_arr))
    i = max(0, min(int(start_index), n))
    if n - i <= 0:
        return np.empty(0), np.empty(0)
    t_slice = t_arr[i:n] if t_arr is not None else np.arange(i, n, dtype=float)
    env = {alias: arr[i:n] for alias, arr in cols.items()}
    env.update({alias: m[i:n] for alias, m in mats.items()})
    try:
        code = compile_formula(expr)
        y = evaluate_formula(code, env, time=t_slice)
    except Exception:
        return np.empty(0), np.empty(0)
    y = np.asarray(y, dtype=float)
    if y.ndim == 0:
        y = np.full(n - i, float(y))
    if y.ndim != 1:
        return np.empty(0), np.empty(0)
    m = min(len(t_slice), len(y))
    return t_slice[:m], y[:m]


def compute_series(
    agg: dict,
    *,
    time_array,
    start_index: int,
    series_handles: dict[str, Any],
    values_handles: dict[str, Any],
    group_arrays: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one derived spec against live stream handles.

    ``series_handles`` maps recorded full names -> 1-D handles.
    ``values_handles`` maps pattern strings -> 2-D ``series_values`` handles.
    ``group_arrays`` maps a formula group alias -> its 2-D matrix.
    """
    op = agg.get("op", "sum")
    axis = agg.get("axis", "instances")

    if axis == "formula":
        return _compute_formula(agg, time_array=time_array,
                                start_index=start_index,
                                series_handles=series_handles,
                                group_arrays=group_arrays)

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
        col = np.asarray(h.array, dtype=float)
        # Skip a source whose column hasn't backfilled yet: including a length-0
        # column would collapse the aggregate to empty (min length 0) and blank
        # the value even though the other instances have data.  The lagging one
        # is folded in on a later frame once its samples arrive.
        if col.size == 0:
            continue
        cols.append(col)
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


def _name_in_scope(name: str, scope: str | None) -> bool:
    """True when ``name`` belongs to component ``scope`` (a dotted path segment).

    ``scope`` is a component id; a recorded variable like ``System.pipe.wall_0.T``
    is in the ``pipe`` scope.  An empty scope matches everything.
    """
    if not scope:
        return True
    return scope in name.split(".")


def resolve_regex_names(
    names: list[str], pattern: str, scope: str | None = None,
) -> list[str]:
    """Recorded variable names in ``scope`` matching ``pattern``.

    ``pattern`` is a regex (``re.search``, case-insensitive); a blank pattern
    matches every in-scope name, and an invalid regex falls back to a substring
    match.  This is the runtime counterpart of the Variables window's filter, so
    an aggregate re-expands to the current model's variable set on each run.
    """
    text = (pattern or "").strip()
    scoped = [n for n in names if _name_in_scope(n, scope)]
    if not text:
        return scoped
    try:
        rx = re.compile(text, re.IGNORECASE)
        return [n for n in scoped if rx.search(n) is not None]
    except re.error:
        needle = text.lower()
        return [n for n in scoped if needle in n.lower()]


def compile_formula(expr: str):
    """Parse + whitelist a formula expression, returning a code object.

    Raises :class:`ValueError` if the expression uses anything outside the
    allowed grammar (arithmetic / comparisons / whitelisted :data:`FORMULA_FUNCS`
    calls over plain names and numeric literals).  Rejecting attribute access,
    subscripting, dunder names and builtins keeps the ``eval`` on the pump thread
    from reaching arbitrary Python.
    """
    text = (expr or "").strip()
    if not text:
        raise ValueError("empty formula")
    tree = ast.parse(text, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _FORMULA_NODES):
            raise ValueError(
                f"disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id not in _FORMULA_CALL_NAMES:
                raise ValueError("only whitelisted functions may be called")
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("names may not start with '_'")
    return compile(tree, "<formula>", "eval")


def evaluate_formula(code, env: dict[str, Any], *, time=None):
    """Evaluate a :func:`compile_formula` code object against ``env`` (alias ->
    array), with only the whitelisted functions/constants in scope.

    ``time`` (the sliced run time array) backs the temporal transforms
    (``integral`` / ``cumsum``).  Floating-point warnings are silenced: a formula
    that hits ``sqrt`` of a negative, a divide-by-zero, etc. should quietly
    produce ``nan`` / ``inf`` rather than spam the console -- and with an empty
    ``__builtins__`` numpy's warn path can't import ``warnings`` anyway.
    """
    namespace = {"__builtins__": {}}
    namespace.update(FORMULA_FUNCS)
    namespace.update(FORMULA_REDUCERS)
    namespace.update(_temporal_namespace(time))
    namespace.update(FORMULA_CONSTS)
    namespace.update(env)
    with np.errstate(all="ignore"):
        return eval(code, namespace)  # noqa: S307 - AST-whitelisted, no builtins


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
