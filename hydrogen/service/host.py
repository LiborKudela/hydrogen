"""Host-side engine for the hydrogen service (runs under ``python -m hydrogen.service``).

Architecture (the actor model from the design discussion):

* **One model-owning thread** (the main thread on rank 0) drives every model:
  it reads commands, executes them, and -- crucially -- is the *only* thread
  that touches model state.  This is what keeps the whole thing MPI-safe.
* **One writer thread** (`_EventBus`) owns all socket *writes*.  Replies,
  status, logs, and variable-stream chunks are pushed onto a queue from the
  main thread and serialised out by the writer.  Nothing else writes to the
  socket.
* **On-demand variable streams**: a client opens a `vars_stream` (any time,
  before/during/after a run) over a chosen set of variables; the run loop
  flushes each stream's newly-recorded rows (with time points) every step, so a
  UI can chart any data at any moment.  ``scope="all"`` backfills history first,
  ``scope="new"`` streams only from the open point onward.
* **Cooperative control**: a `run` honours `stop` / `pause` / `resume` only at
  step boundaries.  The run loop polls the command socket between steps (never
  mid-solve); a pause parks the loop there (model state frozen) until a resume
  or stop arrives, so the same rule extends cleanly to MPI collectives later.

MPI seam
--------
When launched under ``mpirun -n N`` (and ``mpi4py`` is importable with N > 1),
rank 0 runs :func:`_serve_lead` (owns the socket) and ranks 1..N-1 run
:func:`_run_follower` (no socket; they act on broadcast ops in lockstep).  The
per-step solve is *not yet* domain-decomposed -- that arrives with a distributed
``FEMModel`` backend -- so today the followers mirror the lifecycle to validate
the collective control path.  With N == 1 (the default) none of this engages and
the host is a plain single process.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import selectors
import socket
import sys
import threading
import time
import warnings
import queue as _queue

import numpy as np

from ..model import Model, match_name_index, match_name_indices
from .protocol import ProtocolError, recv_msg, send_msg

# --- optional MPI ----------------------------------------------------------
try:  # pragma: no cover - exercised only under mpirun
    from mpi4py import MPI

    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except Exception:  # noqa: BLE001 - mpi4py absent or not under mpirun
    MPI = None
    _COMM = None
    _RANK = 0
    _SIZE = 1


_INSTANTIATE_OPTS = (
    "cse",
    "max_remove_trival_passes",
    "max_remove_duplicate_passes",
    "max_remove_linear_block_passes",
    "enable_blt",
    "enable_var_scaling",
)


# ---------------------------------------------------------------------------
# Outbound event bus (single writer owns the socket)
# ---------------------------------------------------------------------------


class _EventBus:
    """Serialises every host->client message through one writer thread."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._q: _queue.Queue = _queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="hydrogen-host-writer", daemon=True
        )
        self._alive = True
        self._thread.start()

    def emit(self, msg: dict):
        self._q.put(msg)

    def _run(self):
        while True:
            msg = self._q.get()
            if msg is None:
                return
            try:
                send_msg(self._sock, msg)
            except OSError:
                self._alive = False
                return

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# stdout capture -> log events (so the UI sees hydrogen's print() diagnostics)
# ---------------------------------------------------------------------------


class _StdoutForwarder:
    """Tee `sys.stdout` to the real console *and* to log events on the bus.

    hydrogen logs via bare ``print`` (``[parallel lambdify ...]``,
    ``[eval strategy ...]``), and those lines are exactly the "compiling..." /
    "solving..." feedback a UI wants.  We forward them line-by-line while a
    command runs, tagged with the active ``system_id``.
    """

    def __init__(self, bus: _EventBus, real, system_id, rank):
        self._bus = bus
        self._real = real
        self._sid = system_id
        self._rank = rank
        self._buf = ""

    def write(self, text):
        self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._bus.emit({
                    "type": "log", "system_id": self._sid, "rank": self._rank,
                    "level": "INFO", "message": line.rstrip(),
                })

    def flush(self):
        self._real.flush()


class _capture_logs:
    """Context manager: route ``print`` output to the bus for the duration."""

    def __init__(self, bus, system_id, rank):
        self._fwd = _StdoutForwarder(bus, sys.stdout, system_id, rank)

    def __enter__(self):
        self._saved = sys.stdout
        sys.stdout = self._fwd
        return self

    def __exit__(self, *exc):
        sys.stdout = self._saved
        return False


class _capture_warnings:
    """Context manager: turn Python ``warnings.warn`` calls into WARNING log
    events on the bus, attributed to the top-level component that emitted them.

    hydrogen components raise ``UserWarning`` for build/run-time modelling
    issues (e.g. a `SegmentedChannel`'s cell-Peclet check).  These go to stderr
    by default, so a GUI never sees them.  We install a ``showwarning`` hook
    that (a) keeps the console behaviour, (b) walks the call stack to find the
    *direct child of the root model* on the path to the ``warn`` call -- that is
    the canvas node responsible -- and (c) emits a structured event carrying the
    component's id so the UI can badge that node and list the message.

    Attribution needs the root model's ``components`` map, which for a fresh
    ``load_json`` only exists once construction finishes; ``root_provider`` is a
    callable returning the root model (or ``None`` while it is still being
    built).  Warnings raised before the root is known are buffered and flushed,
    resolved, on exit.
    """

    def __init__(self, bus, system_id, rank, root_provider):
        self._bus = bus
        self._sid = system_id
        self._rank = rank
        self._root_provider = root_provider
        self._pending: list = []          # (text, node_instance, where)

    def __enter__(self):
        self._saved = warnings.showwarning
        # Force "always" for UserWarning (restored on exit) so every occurrence
        # is reported -- otherwise Python's once-per-location default would badge
        # only the first of several components hitting the same modelling issue.
        # Scoped to UserWarning so unrelated Deprecation/Resource warnings keep
        # their default suppression and don't flood the run log.
        self._saved_filters = warnings.filters[:]
        warnings.filterwarnings("always", category=UserWarning)
        warnings.showwarning = self._show
        return self

    def __exit__(self, *exc):
        warnings.showwarning = self._saved
        warnings.filters[:] = self._saved_filters
        # Flush anything we couldn't attribute yet (root not built at the time).
        idmap = self._component_index()
        for text, node, where in self._pending:
            self._emit(text, idmap.get(id(node)) if node is not None else None,
                       where)
        self._pending.clear()
        return False

    # --- internals --------------------------------------------------------
    def _component_index(self) -> dict:
        """`{id(child_instance): child_name}` for the root model's direct
        children (the canvas nodes), or ``{}`` if the root isn't ready."""
        try:
            root = self._root_provider()
        except Exception:
            root = None
        comps = getattr(root, "components", None) or {}
        return {id(v): k for k, v in comps.items()}

    @staticmethod
    def _emitting_node():
        """The root model's direct child on the current call stack -- i.e. the
        node that owns the code raising the warning -- or ``None``.

        Walking outward, the distinct ``self`` Model instances are
        ``[emitter, ..., node, root]``; the root is the outermost and the node
        is the one just inside it.
        """
        chain, seen = [], set()
        f = sys._getframe()
        while f is not None:
            slf = f.f_locals.get("self")
            if isinstance(slf, Model) and id(slf) not in seen:
                seen.add(id(slf))
                chain.append(slf)
            f = f.f_back
        return chain[-2] if len(chain) >= 2 else None

    def _emit(self, text, comp_id, where):
        self._bus.emit({
            "type": "log", "system_id": self._sid, "rank": self._rank,
            "level": "WARNING", "message": text,
            "component": comp_id, "where": where,
        })

    def _show(self, message, category, filename, lineno, file=None, line=None):
        # Preserve the normal stderr output so the console still shows it.
        try:
            self._saved(message, category, filename, lineno, file, line)
        except Exception:
            pass
        # Only surface modelling warnings (UserWarning); framework noise such as
        # DeprecationWarning / ResourceWarning stays on the console only.
        if not (isinstance(category, type)
                and issubclass(category, UserWarning)):
            return
        node = self._emitting_node()
        text = f"{getattr(category, '__name__', 'Warning')}: {message}"
        where = f"{os.path.basename(filename)}:{lineno}"
        idmap = self._component_index()
        if idmap:                          # root is ready -> attribute + emit now
            self._emit(text, idmap.get(id(node)) if node is not None else None,
                       where)
        else:                              # still constructing -> resolve on exit
            self._pending.append((text, node, where))


# ---------------------------------------------------------------------------
# Per-system runtime
# ---------------------------------------------------------------------------


class _StreamState:
    """One open variable stream over a system's recorded history.

    A stream watches a fixed set of variables (resolved once to column indices)
    and ships their values + matching time points to the client in chunks.
    ``sent`` is the number of recorded rows already examined; the run loop
    flushes everything appended since the last flush.  ``every`` subsamples on a
    *global* record index (row ``i`` is emitted iff ``i % every == 0``) so the
    sampling stays consistent across chunk boundaries.

    The ``scope`` chosen at open time only decides the initial ``sent`` cursor:
    ``"all"`` starts at 0 (backfills the whole history, then streams live data),
    ``"new"`` starts at the current row count (live data only).
    """

    def __init__(self, stream_id: str, cols, every: int):
        self.id = stream_id
        self.cols = cols          # list of (key_name, column_index)
        self.every = max(1, int(every))
        self.sent = 0             # recorded rows already examined
        self.start = 0            # scope anchor (row to re-backfill from on add)


class _SystemRuntime:
    """Wraps one model with its lifecycle phase + last-solve diagnostics."""

    def __init__(self, system_id: str, model):
        self.id = system_id
        self.model = model
        self.phase = "loaded"
        self.step = 0
        self.total = 0
        self.stop_requested = False
        self.pause_requested = False
        # Last run configuration (for continue after finish / stop).
        self.run_dt = None
        self.run_steps = None
        self.run_stop_time = None
        self.run_strategy = None
        self.run_adaptive: dict = {}
        self.run_every = 1
        self.run_delay = 0.0
        self.continuable = False
        # Cached solver post-mortem from the last failed run (see
        # `Model.diagnose`), captured at the failing state so the UI can inspect
        # it even after the model has moved on.
        self.last_diagnostic: dict | None = None
        # Open variable streams, keyed by stream id (see `_StreamState`). Each
        # is flushed after every committed step in the run loop.
        self.streams: dict[str, _StreamState] = {}
        self._stream_counter = 0

    def new_stream_id(self) -> str:
        self._stream_counter += 1
        return f"{self.id}-stream-{self._stream_counter}"

    # --- diagnostics ------------------------------------------------------

    def status(self) -> dict:
        m = self.model
        t = _t_value(m)
        out = {
            "system_id": self.id,
            "phase": self.phase,
            "step": self.step,
            "total": self.total,
            "t": t,
            "stop_time": self.run_stop_time,
            "continuable": bool(self.continuable),
            "last_residual": float(getattr(m, "_last_solve_error_norm", float("nan"))),
            "last_iters": int(getattr(m, "_last_solve_iters", 0)),
        }
        rc = m.get_run_control()
        if rc:
            out["run_control"] = rc
        return out

    def _sync_continuable(self):
        """True when the model can integrate further (``t < stop_time``)."""
        if self.run_stop_time is None:
            self.continuable = False
            return
        t = _t_value(self.model)
        self.continuable = t < self.run_stop_time - 1e-9

    # --- variable resolution ---------------------------------------------

    def var_names(self) -> list:
        return list(self.model.record.get("vars_names", []))

    def param_names(self) -> list:
        """Full names of every settable Parameter (available post-instantiate).

        Parameters get their ``full_name`` (and their live solver-buffer slot)
        when the model is instantiated, so this is empty before ``instantiate``.
        """
        params = getattr(self.model, "raw_param_references", None) or []
        return [getattr(p, "full_name", None) or getattr(p, "name", "") or ""
                for p in params]

    def resolve(self, names):
        """Map requested (full or suffix) names -> (display_name, column index).

        Delegates to :meth:`Model.resolve_vars` so the host and the in-process
        API share one name-resolution rule.
        """
        return self.model.resolve_vars(names)

    def resolve_all(self, names):
        """Expand requested (full/suffix) names to *every* matching recorded
        variable -> ``(full_name, column index)``, deduped in record order.

        Mirrors :meth:`Model.series_values` (all matches) the way
        :meth:`resolve` mirrors :meth:`Model.resolve_vars` (first match), so a
        stream can watch every per-instance match of a suffix at once
        (e.g. ``"m_dot_a_leak"`` -> one column per pipe segment).  Keys are the
        full recorded names so the client can tell the matches apart.
        """
        if isinstance(names, str):
            names = [names]
        full = self.model.record.get("vars_names", [])
        out, seen = [], set()
        for req in names:
            for i in match_name_indices(full, req):
                if i in seen:
                    continue
                seen.add(i)
                out.append((full[i], i))
        return out

    def latest_row(self):
        return self.model.latest_state()


def _match_index(full_names, req):
    """Shared name-resolution (exact / dotted-suffix / bare-suffix).

    Re-exported as a thin wrapper so parameter matching (which resolves against
    a different name list than the recorded variables) uses the same rule as
    :meth:`Model.resolve_vars`.
    """
    return match_name_index(full_names, req)


def _apply_params(model, assignments):
    """Resolve ``{name: value}`` against the model's Parameters and push each.

    Names match exactly as variables do (exact, else dotted-suffix), so a UI can
    pass the ``full`` name from :meth:`list_params` or a convenient suffix like
    ``heat_0.Q_flow``.  Each value is written straight into the live solver
    buffer via ``Parameter.set_value`` (single slot, no whole-vector re-push), so
    the next solve sees it.  Returns the applied ``{resolved_full_name: value}``.

    Requires an instantiated model -- parameters acquire their ``full_name`` and
    buffer slot at ``instantiate()``.
    """
    params = list(getattr(model, "raw_param_references", None) or [])
    names = [getattr(p, "full_name", None) or getattr(p, "name", "") or ""
             for p in params]
    applied = {}
    for req, val in assignments.items():
        i = _match_index(names, req)
        if i is None:
            # ValueError (not KeyError): the engine reserves KeyError for an
            # unknown *system*, so a missing parameter must use a different type
            # to be surfaced distinctly to the client.
            raise ValueError(f"no parameter matching {req!r}")
        fval = float(val)
        params[i].set_value(fval)
        applied[names[i]] = fval
    return applied


_LIVE_ADAPTIVE_KEYS = (
    "dt_min", "dt_max", "grow", "shrink", "max_retries",
    "relaxation", "tol", "max_iter", "line_search",
)
_LIVE_STRATEGY_KEYS = ("tol_local", "atol")
_RUN_CONFIG_PHASES = frozenset({
    "running", "paused", "finished", "stopped", "initialised",
})


def _apply_run_config(rt, args: dict) -> dict:
    """Update the live run controller + persisted run config on ``rt``.

    Honoured on the next committed step (or before a :meth:`continue_run`).
    """
    phase = rt.phase
    if phase not in _RUN_CONFIG_PHASES:
        raise RuntimeError(
            f"run config cannot be changed while phase is {phase!r}")
    applied: dict = {}
    if "stop_time" in args and args["stop_time"] is not None:
        rt.run_stop_time = float(args["stop_time"])
        applied["stop_time"] = rt.run_stop_time
    adaptive = dict(args.get("adaptive") or {})
    for key in _LIVE_ADAPTIVE_KEYS:
        if key in args and args[key] is not None:
            adaptive[key] = args[key]
    if adaptive:
        rt.run_adaptive.update(adaptive)
        applied["adaptive"] = dict(rt.run_adaptive)
    strat_updates = {}
    if "strategy" in args and args["strategy"] is not None:
        strat_updates = dict(args["strategy"])
    for key in _LIVE_STRATEGY_KEYS:
        if key in args and args[key] is not None:
            strat_updates[key] = args[key]
    if strat_updates:
        base = dict(rt.run_strategy or {"name": "richardson"})
        if "name" in strat_updates and len(strat_updates) == 1:
            base["name"] = strat_updates["name"]
        else:
            base.update(strat_updates)
        rt.run_strategy = base
        applied["strategy"] = dict(base)
    ctrl_updates: dict = {}
    if rt.run_stop_time is not None:
        ctrl_updates["stop_time"] = rt.run_stop_time
    for key in _LIVE_ADAPTIVE_KEYS:
        if key in rt.run_adaptive:
            ctrl_updates[key] = rt.run_adaptive[key]
    if rt.run_strategy is not None:
        ctrl_updates["strategy"] = rt.run_strategy
    if ctrl_updates and rt.model.get_run_control():
        applied["run_control"] = rt.model.update_run_control(**ctrl_updates)
    elif ctrl_updates and phase in ("finished", "stopped", "initialised"):
        seed = {k: v for k, v in rt.run_adaptive.items()
                if k in _LIVE_ADAPTIVE_KEYS}
        rt.model._init_run_control(
            strategy=rt.run_strategy or {"name": "richardson"},
            dt=rt.run_dt,
            stop_time=rt.run_stop_time,
            **seed,
        )
        applied["run_control"] = rt.model.get_run_control()
    rt._sync_continuable()
    applied["continuable"] = rt.continuable
    return applied


def _param_assignments(args):
    """Normalise a ``set_param`` request to a ``{name: value}`` dict.

    Accepts either a single ``name``/``value`` pair or a ``params`` mapping.
    """
    if args.get("params"):
        return dict(args["params"])
    if "name" in args:
        return {args["name"]: args["value"]}
    raise ValueError("set_param needs 'name'+'value' or a 'params' mapping")


def _common_prefix_depth(split_names):
    """Number of leading path segments shared by *all* names (leaving at least
    one trailing segment on every name). Used to drop a redundant root such as
    the synthetic ``_SpecComposite`` wrapper a loaded spec sits under."""
    if not split_names:
        return 0
    depth = 0
    while True:
        if any(len(s) <= depth + 1 for s in split_names):
            break
        seg = split_names[0][depth]
        if any(s[depth] != seg for s in split_names):
            break
        depth += 1
    return depth


def _natural_key(name):
    """Sort key that orders embedded integers numerically (``seg2`` < ``seg10``)."""
    return [(1, int(tok)) if tok.isdigit() else (0, tok.lower())
            for tok in re.split(r"(\d+)", name) if tok]


def _build_var_tree(names):
    """Turn flat dotted variable names into a nested tree a UI can render.

    Each recorded name (e.g. ``wall.T_a``, ``stack.c1.Q_dot_a``) is split on
    ``.`` into path segments; interior segments become group nodes and the
    final segment a selectable leaf.  A redundant root prefix shared by every
    variable (e.g. the ``_SpecComposite`` wrapper a loaded spec lives under) is
    stripped from the *display* paths.

    Every node carries:

    * ``name``  -- the raw path segment (``""`` for the root),
    * ``path``  -- the display path (prefix stripped); **unique across the whole
      tree**, so a UI can use it directly as the node key / selection id,
    * ``leaf``  -- ``True`` for a selectable variable, ``False`` for a group.
      This is explicit (not inferred from ``children``) so the edge case where a
      recorded name is also a prefix of another (e.g. ``tank`` *and*
      ``tank.level``) stays unambiguous: such a node is a leaf *and* has
      children.
    * ``count`` -- number of selectable variables in its subtree (counting the
      node itself when it is a leaf); handy for badges / "select all".

    Group nodes (anything not a pure leaf, including the root) additionally
    carry ``children`` (a list -- possibly empty).  Leaf nodes additionally
    carry:

    * ``index`` -- their column in the recorded state / ``get_record`` rows,
    * ``full``  -- the exact recorded name (always resolves unambiguously when
      passed back to ``get_series`` / ``get_record`` / ``vars_stream``).

    Children are ordered leaves-first (a level's own, outermost variables), then
    groups (nested sub-models), each block sorted naturally (embedded integers
    compare numerically, so ``seg2`` precedes ``seg10``).
    Returns the root group node
    ``{"name": "", "path": "", "leaf": False, "count": N, "children": [...]}``.
    """
    split = [n.split(".") for n in names]
    skip = _common_prefix_depth(split)

    root = {"name": "", "path": "", "_children": {}}
    for idx, (full, parts) in enumerate(zip(names, split)):
        disp = parts[skip:]
        node = root
        path = ""
        for d, part in enumerate(disp):
            path = part if not path else f"{path}.{part}"
            child = node["_children"].get(part)
            if child is None:
                child = {"name": part, "path": path, "_children": {}}
                node["_children"][part] = child
            node = child
            if d == len(disp) - 1:
                node["index"] = idx
                node["full"] = full

    def _finish(node):
        kids = node.pop("_children")
        node["leaf"] = "index" in node
        children = [_finish(kids[k]) for k in kids]
        if children:
            # Leaves first (a level's own, outermost variables), then groups
            # (nested sub-models holding deeper variables); each block
            # natural-sorted by name.
            children.sort(key=lambda c: (not c["leaf"], _natural_key(c["name"])))
        # A non-leaf always advertises a `children` list (empty only for an
        # empty root) so the UI can treat every group uniformly; pure leaves
        # omit it so the widget knows they are not expandable.
        if children or not node["leaf"]:
            node["children"] = children
        node["count"] = (1 if node["leaf"] else 0) + sum(c["count"]
                                                          for c in children)
        return node

    return _finish(root)


def _t_value(model):
    try:
        return float(model.get_t_value())
    except Exception:  # noqa: BLE001 - before initialise t may be unset
        return 0.0


def _gather_modules(model):
    """Best-effort collection of CoolProp ``medium.modules`` for instantiate.

    Returns a merged module list (fluid systems) or ``None`` (e.g. a pure
    thermal network, which needs no symbolic property callbacks).
    """
    media, seen = [], set()

    def add(m):
        if m is not None and id(m) not in seen:
            seen.add(id(m))
            media.append(m)

    ctx = getattr(model, "_ctx", None)
    if ctx is not None:
        for m in getattr(ctx, "media", {}).values():
            add(m)

    def walk(comp):
        add(getattr(comp, "medium", None))
        for child in getattr(comp, "components", {}).values():
            walk(child)

    walk(model)

    mods = []
    for m in media:
        mm = getattr(m, "modules", None)
        if mm:
            mods.extend(mm)
    return mods or None


# ---------------------------------------------------------------------------
# Engine: command dispatch over the loaded systems
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self, bus: _EventBus, sock: socket.socket, rank: int):
        self._bus = bus
        self._sock = sock
        self._rank = rank
        self._systems: dict[str, _SystemRuntime] = {}
        self._counter = 0
        self._shutdown = False

    # --- helpers ----------------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"sys-{self._counter}"

    def _require(self, sid) -> _SystemRuntime:
        rt = self._systems.get(sid)
        if rt is None:
            raise KeyError(f"no such system {sid!r}")
        return rt

    def _reply(self, req_id, result=None):
        self._bus.emit({"type": "reply", "id": req_id, "status": "ok",
                        "result": result})

    def _reply_error(self, req_id, message, kind=None, **extra):
        msg = {"type": "reply", "id": req_id, "status": "error",
               "message": str(message), "kind": kind}
        msg.update(extra)
        self._bus.emit(msg)

    # --- top-level command handling --------------------------------------

    def handle(self, msg: dict):
        """Process one client request (called on the model-owning thread)."""
        req_id = msg.get("id")
        cmd = msg.get("cmd")
        args = msg.get("args", {}) or {}
        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                raise ValueError(f"unknown command {cmd!r}")
            handler(req_id, args)
        except KeyError as exc:
            self._reply_error(req_id, exc, kind="UnknownSystem")
        except Exception as exc:  # noqa: BLE001 - surface everything to the UI
            self._reply_error(req_id, exc, kind=type(exc).__name__)

    @property
    def should_shutdown(self):
        return self._shutdown

    # --- commands ---------------------------------------------------------

    def _cmd_load_json(self, req_id, args):
        from ..serialization import from_json

        sid = self._new_id()
        holder: dict = {}
        with _capture_logs(self._bus, sid, self._rank), \
                _capture_warnings(self._bus, sid, self._rank,
                                  lambda: holder.get("model")):
            model = from_json(args["spec"])
            holder["model"] = model        # now resolvable for buffered warnings
        self._systems[sid] = _SystemRuntime(sid, model)
        self._broadcast({"op": "load_json", "spec": args["spec"], "sid": sid})
        self._reply(req_id, sid)

    def _cmd_load_dict(self, req_id, args):
        from ..serialization import from_dict

        model = from_dict(args["spec"])
        sid = self._new_id()
        self._systems[sid] = _SystemRuntime(sid, model)
        self._broadcast({"op": "load_dict", "spec": args["spec"], "sid": sid})
        self._reply(req_id, sid)

    def _cmd_instantiate(self, req_id, args):
        rt = self._require(args["system_id"])
        opts = {k: args[k] for k in _INSTANTIATE_OPTS if k in args}
        opts.setdefault("aditional_modules", _gather_modules(rt.model))
        self._broadcast({"op": "instantiate", "sid": rt.id, "opts": opts})
        with _capture_logs(self._bus, rt.id, self._rank), \
                _capture_warnings(self._bus, rt.id, self._rank,
                                  lambda: rt.model):
            rt.model.instantiate(**opts)
        rt.phase = "instantiated"
        self._reply(req_id, {"n_vars": int(getattr(rt.model, "n_v", 0)),
                             "n_recorded": len(rt.var_names())})

    def _cmd_initialise(self, req_id, args):
        rt = self._require(args["system_id"])
        n = args.get("n", 1)
        kw = {k: args[k] for k in
              ("relaxation", "tol", "max_iter", "line_search") if k in args}
        self._broadcast({"op": "initialise", "sid": rt.id, "n": n, "kw": kw})
        with _capture_logs(self._bus, rt.id, self._rank):
            rt.model.initialise(n=n, **kw)
        rt.phase = "initialised"
        rt.step = 0
        self._reply(req_id, rt.status())

    def _cmd_step(self, req_id, args):
        rt = self._require(args["system_id"])
        dt = args["dt"]
        self._broadcast({"op": "step", "sid": rt.id, "dt": dt})
        rt.model.solve_dae_step(dt)
        rt.model.next_step()
        rt.step += 1
        self._flush_streams(rt)
        self._reply(req_id, rt.status())

    def _cmd_list_vars(self, req_id, args):
        rt = self._require(args["system_id"])
        self._reply(req_id, rt.var_names())

    def _cmd_list_params(self, req_id, args):
        rt = self._require(args["system_id"])
        self._reply(req_id, rt.param_names())

    def _cmd_set_param(self, req_id, args):
        rt = self._require(args["system_id"])
        assignments = _param_assignments(args)
        applied = _apply_params(rt.model, assignments)
        # Keep MPI followers' model copies in sync (no-op when SIZE == 1).
        self._broadcast({"op": "set_param", "sid": rt.id,
                         "assignments": assignments})
        self._reply(req_id, applied)

    def _cmd_var_tree(self, req_id, args):
        rt = self._require(args["system_id"])
        self._reply(req_id, _build_var_tree(rt.var_names()))

    def _cmd_status(self, req_id, args):
        rt = self._require(args["system_id"])
        self._reply(req_id, rt.status())

    def _cmd_diagnose(self, req_id, args):
        """Return a solver post-mortem (see `Model.diagnose`).

        After a failed run we return the diagnostic captured AT the failing
        state (highest fidelity).  Otherwise -- or when `fresh=True` is passed
        -- we compute a new one against the current state so the command is
        useful any time after `instantiate` (e.g. to inspect a stalling init).
        """
        rt = self._require(args["system_id"])
        top_k = int(args.get("top_k", 12) or 12)
        fresh = bool(args.get("fresh", False))
        if not fresh and rt.phase == "error" and rt.last_diagnostic is not None:
            self._reply(req_id, rt.last_diagnostic)
            return
        with _capture_logs(self._bus, rt.id, self._rank):
            report = rt.model.diagnose(top_k=top_k)
        rt.last_diagnostic = report
        self._reply(req_id, report)

    def _cmd_get_state(self, req_id, args):
        rt = self._require(args["system_id"])
        row = rt.latest_row()
        if row is None:
            self._reply(req_id, {})
            return
        cols = rt.resolve(args.get("vars"))
        self._reply(req_id, {name: float(row[i]) for name, i in cols})

    def _cmd_get_record(self, req_id, args):
        rt = self._require(args["system_id"])
        rec = rt.model.get_record(
            args.get("vars"), start=args.get("start", 0) or 0,
            stop=args.get("stop"), stride=args.get("stride", 1) or 1)
        self._reply(req_id, {
            "names": rec["names"],
            "time": rec["time"].tolist(),
            "rows": rec["rows"].tolist(),
        })

    def _cmd_get_series(self, req_id, args):
        rt = self._require(args["system_id"])
        rec = rt.model.get_series(
            args.get("vars"), start=args.get("start", 0) or 0,
            stop=args.get("stop"), stride=args.get("stride", 1) or 1)
        self._reply(req_id, {
            "time": rec["time"].tolist(),
            "series": {name: col.tolist() for name, col in rec["series"].items()},
        })

    # --- variable streams (on-demand, independent of `run`) ---------------

    def _cmd_vars_stream(self, req_id, args):
        """Open a stream from *outside* a run (before/after `run`).

        Mid-run opens arrive on the command socket and are handled by
        :meth:`_drain_during_run`; both funnel through
        :meth:`_handle_open_stream`.
        """
        rt = self._require(args["system_id"])
        self._handle_open_stream(rt, req_id, args)

    def _cmd_add_stream_vars(self, req_id, args):
        """Add watched columns to an open stream (lazy/dynamic watching).

        Funnels through :meth:`_handle_add_stream_vars`, like ``vars_stream``;
        mid-run requests are routed by :meth:`_drain_during_run`.
        """
        rt = self._require(args["system_id"])
        self._handle_add_stream_vars(rt, req_id, args)

    def _cmd_close_stream(self, req_id, args):
        rt = self._require(args["system_id"])
        self._handle_close_stream(rt, req_id, args)

    def _open_stream(self, rt, args) -> _StreamState:
        var_names = args.get("vars")
        if isinstance(var_names, str):
            var_names = [var_names]
        var_names = var_names or []  # empty is allowed: add columns later via
                                     # add_stream_vars (lazy/dynamic watching)
        scope = args.get("scope", "new")
        if scope not in ("new", "all"):
            raise ValueError(f"scope must be 'new' or 'all', got {scope!r}")
        # `expand`: resolve each requested name to *every* matching recorded
        # variable (keyed by full name) instead of just the first.  This lets a
        # client mirror `Model.series_values` -- watch a suffix once and pull
        # all per-instance matches -- while still only shipping those columns.
        expand = bool(args.get("expand", False))
        if not var_names:
            cols = []
        else:
            cols = rt.resolve_all(var_names) if expand else rt.resolve(var_names)
        st = _StreamState(rt.new_stream_id(), cols, args.get("every", 1) or 1)
        n_now = len(rt.model.record.get("state", []))
        st.sent = st.start = 0 if scope == "all" else n_now
        rt.streams[st.id] = st
        return st

    def _add_stream_vars(self, rt, args):
        """Append more watched columns to an already-open stream.

        Dynamic watching always *expands* (a suffix grabs every match, keyed by
        full name) so a client can defer the choice of variables to the first
        :meth:`Stream.series` / :meth:`Stream.series_values` call.  Returns
        ``(stream, added_full_names)``; ``added`` is empty when nothing new
        matched (already watched, or no match at all).
        """
        stid = args.get("stream_id")
        st = rt.streams.get(stid)
        if st is None:
            raise KeyError(f"no open stream {stid!r}")
        names = args.get("vars") or []
        if isinstance(names, str):
            names = [names]
        watched = {idx for _, idx in st.cols}
        added = []
        for full, idx in rt.resolve_all(names):
            if idx in watched:
                continue
            watched.add(idx)
            st.cols.append((full, idx))
            added.append(full)
        return st, added

    def _handle_open_stream(self, rt, req_id, args):
        st = self._open_stream(rt, args)
        self._reply(req_id, {
            "stream_id": st.id,
            "vars": [name for name, _ in st.cols],
            "scope": args.get("scope", "new"),
            "every": st.every,
        })
        # 'all' scope: backfill the existing history right away (so a UI gets
        # the full chart immediately, even if the system isn't running). 'new'
        # scope opened with sent == n_now, so there is nothing to backfill.
        if st.sent == 0:
            self._flush_stream(rt, st, initial=True)

    def _handle_add_stream_vars(self, rt, req_id, args):
        st, added = self._add_stream_vars(rt, args)
        self._reply(req_id, {
            "stream_id": st.id,
            "added": added,
            "vars": [name for name, _ in st.cols],
        })
        # Re-backfill from the stream's scope anchor so the new column(s) arrive
        # aligned with the history already shipped for the existing ones; the
        # client resets its buffer on the `initial` chunk.
        if added:
            st.sent = st.start
            self._flush_stream(rt, st, initial=True)

    def _handle_close_stream(self, rt, req_id, args):
        stid = args.get("stream_id")
        existed = rt.streams.pop(stid, None) is not None
        self._reply(req_id, {"closed": stid, "existed": existed})
        if existed:
            self._bus.emit({"type": "stream_closed", "system_id": rt.id,
                            "stream_id": stid})

    def _flush_stream(self, rt, st: _StreamState, *, initial=False):
        """Emit any recorded rows appended since this stream's last flush."""
        rec = rt.model.record
        state_all = rec.get("state", [])
        time_all = rec.get("time", [])
        n = len(state_all)
        if st.sent >= n:
            return
        times, series = [], {name: [] for name, _ in st.cols}
        for idx in range(st.sent, n):
            if idx % st.every:
                continue
            row = state_all[idx]
            times.append(float(time_all[idx]))
            for name, i in st.cols:
                series[name].append(float(row[i]))
        st.sent = n
        if times:  # nothing to send if subsampling skipped the whole window
            self._bus.emit({
                "type": "stream_data", "system_id": rt.id, "stream_id": st.id,
                "initial": initial, "time": times, "series": series,
            })

    def _flush_streams(self, rt):
        for st in list(rt.streams.values()):
            self._flush_stream(rt, st)

    def _cmd_list_systems(self, req_id, args):
        self._reply(req_id, [
            {"system_id": rt.id, "phase": rt.phase} for rt in self._systems.values()
        ])

    def _cmd_close(self, req_id, args):
        sid = args["system_id"]
        rt = self._require(sid)
        # Terminate any open streams so client-side consumers stop cleanly.
        for stid in list(rt.streams):
            self._bus.emit({"type": "stream_closed", "system_id": sid,
                            "stream_id": stid})
        self._broadcast({"op": "close", "sid": sid})
        del self._systems[sid]
        self._reply(req_id, {"closed": sid})

    def _cmd_shutdown(self, req_id, args):
        self._broadcast({"op": "shutdown"})
        self._shutdown = True
        self._reply(req_id, {"shutdown": True})

    def _cmd_update_run_config(self, req_id, args):
        rt = self._require(args["system_id"])
        try:
            applied = _apply_run_config(rt, args)
        except Exception as exc:  # noqa: BLE001
            self._reply_error(req_id, exc, kind=type(exc).__name__)
            return
        self._reply(req_id, applied)
        if rt.phase in ("finished", "stopped", "paused"):
            self._bus.emit({"type": "status", **rt.status()})

    def _cmd_continue_run(self, req_id, args):
        rt = self._require(args["system_id"])
        if not rt.continuable:
            self._reply_error(
                req_id,
                "continue_run requires stop_time > current model time "
                "(extend stop_time first)",
                kind="ValueError")
            return
        if rt.run_stop_time is None:
            self._reply_error(req_id, "no stop_time configured for this run",
                              kind="ValueError")
            return
        stream = bool(args.get("stream", True))
        every = max(1, args.get("every", rt.run_every) or rt.run_every)
        delay = max(0.0, float(args.get("delay", rt.run_delay) or rt.run_delay))
        cfg = (rt, rt.run_dt, rt.run_steps, rt.run_stop_time, rt.run_strategy,
               rt.run_adaptive, every, delay)
        if stream:
            self._reply(req_id, {"streaming": True,
                                 "stop_time": rt.run_stop_time})
            t0 = time.monotonic()
            self._run_loop(*cfg, stream=True, continue_from=rt.step)
            done = {"type": "done", "elapsed": time.monotonic() - t0}
            done.update(rt.status())
            done["type"] = "done"
            self._bus.emit(done)
        else:
            summary = self._run_loop(*cfg, stream=False, continue_from=rt.step)
            self._reply(req_id, summary)

    def _cmd_run(self, req_id, args):
        rt = self._require(args["system_id"])
        dt = args.get("dt")
        steps = args.get("steps")
        steps = int(steps) if steps is not None else None
        stop_time = args.get("stop_time")
        stop_time = float(stop_time) if stop_time is not None else None
        # `strategy` is None / "fixed" for the classic fixed-`dt` loop, or an
        # adaptive name / dict (see Model.solve_adaptive_step).  `adaptive`
        # carries the controller knobs (dt_min/dt_max/dt_start/tol/...).
        strategy = args.get("strategy")
        adaptive = args.get("adaptive") or {}
        stream = args.get("stream", True)
        # `every` throttles the cadence of `status` progress events only.
        every = max(1, args.get("every", 1) or 1)
        delay = max(0.0, float(args.get("delay", 0.0) or 0.0))

        if steps is None and stop_time is None:
            self._reply_error(
                req_id, "run requires 'steps' or 'stop_time'", kind="ValueError")
            return
        is_fixed = (strategy is None or strategy == "fixed"
                    or (isinstance(strategy, dict) and strategy.get("name") == "fixed"))
        if is_fixed and dt is None:
            self._reply_error(
                req_id, "fixed-step run requires 'dt'", kind="ValueError")
            return

        cfg = (rt, dt, steps, stop_time, strategy, adaptive, every, delay)
        if stream:
            # Ack now so the client's run() returns; status/stream events then
            # flow asynchronously while the loop advances.
            self._reply(req_id, {"streaming": True, "steps": steps,
                                 "stop_time": stop_time})
            t0 = time.monotonic()
            self._run_loop(*cfg, stream=True)
            # Carry the final run stats on `done` (step count, sim time, wall
            # time, last-solve diagnostics) so the client can print a summary.
            done = {"type": "done", "elapsed": time.monotonic() - t0}
            done.update(rt.status())
            done["type"] = "done"           # status() has no 'type'; keep 'done'
            self._bus.emit(done)
        else:
            summary = self._run_loop(*cfg, stream=False)
            self._reply(req_id, summary)

    # --- the run loop (cooperative stop + streaming) ----------------------

    def _run_loop(self, rt, dt, steps, stop_time, strategy, adaptive, every, delay,
                  *, stream, continue_from: int | None = None):
        rt.phase = "running"
        rt.total = steps if steps is not None else 0
        rt.stop_requested = False
        rt.pause_requested = False
        rt.continuable = False
        rt.last_diagnostic = None

        # Persist for continue / live config edits after this loop exits.
        rt.run_dt = dt
        rt.run_steps = steps
        rt.run_stop_time = stop_time
        rt.run_strategy = strategy
        rt.run_adaptive = dict(adaptive or {})
        rt.run_every = every
        rt.run_delay = delay

        model = rt.model
        is_fixed = (strategy is None or strategy == "fixed"
                    or (isinstance(strategy, dict) and strategy.get("name") == "fixed"))
        dt_min = adaptive.get("dt_min", 1e-9)
        dt_max = adaptive.get("dt_max")
        dt_start = adaptive.get("dt_start", dt)
        extra = {k: adaptive[k] for k in
                 ("grow", "shrink", "max_retries", "relaxation", "tol",
                  "max_iter", "line_search")
                 if k in adaptive}
        steps_gen = model.iter_run(
            stop_time=stop_time, strategy=("fixed" if is_fixed else strategy),
            dt=dt, dt_min=dt_min, dt_max=dt_max, dt_start=dt_start, **extra)

        sel = selectors.DefaultSelector()
        if self._sock is not None:
            sel.register(self._sock, selectors.EVENT_READ)

        def _done(k):
            st = rt.run_stop_time
            if st is not None and _t_value(model) >= st - 1e-9:
                return True
            if steps is not None and k >= steps:
                return True
            return False

        with _capture_logs(self._bus, rt.id, self._rank), \
                _capture_warnings(self._bus, rt.id, self._rank,
                                  lambda: rt.model):
            k = int(continue_from or 0)
            while not _done(k):
                self._drain_during_run(sel, rt)
                # Honour a pause request at this step boundary: park here (still
                # servicing resume/stop/status) until the UI resumes or stops.
                if rt.pause_requested and not rt.stop_requested:
                    self._wait_while_paused(sel, rt)
                go = self._broadcast_control(not rt.stop_requested)
                if not go:
                    break
                step_start = time.monotonic()
                # One accepted step from the shared kernel (dt selection +
                # solve + commit live inside the generator).
                try:
                    next(steps_gen)
                except StopIteration:
                    break
                except Exception as exc:  # noqa: BLE001
                    rt.phase = "error"
                    # Capture a solver post-mortem at the failing state (best
                    # fidelity here -- later commands may have moved the model).
                    try:
                        rt.last_diagnostic = rt.model.diagnose()
                    except Exception:  # noqa: BLE001 - diagnosis is best-effort
                        rt.last_diagnostic = None
                    self._emit_error(rt, exc)
                    if stream:
                        return None
                    return {"phase": "error", "step": rt.step,
                            "error": str(exc), "kind": type(exc).__name__}
                k += 1
                rt.step = k
                # Push any open variable streams (the live data channel).
                self._flush_streams(rt)
                if stream and k % every == 0:
                    self._bus.emit({"type": "status", **rt.status()})
                # Real-time pacing: throttle the loop so wall-clock ~ delay per
                # step (minus the time already spent solving). Lets a UI watch a
                # fast model evolve live instead of finishing instantly.
                if delay:
                    remaining = delay - (time.monotonic() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)

        sel.close()
        model.clear_run_control()
        rt.phase = "stopped" if rt.stop_requested else "finished"
        rt._sync_continuable()
        if not stream:
            return rt.status()
        return None

    def _drain_during_run(self, sel, rt):
        """Read any commands that arrived mid-run; only stop/status are acted on."""
        if self._sock is None:
            return
        while sel.select(timeout=0):
            try:
                msg = recv_msg(self._sock)
            except (ProtocolError, OSError):
                msg = None
            if msg is None:
                rt.stop_requested = True
                self._shutdown = True
                return
            cmd = msg.get("cmd")
            req_id = msg.get("id")
            if cmd == "stop":
                rt.stop_requested = True
                rt.pause_requested = False  # stop wins over a pending pause
                self._reply(req_id, {"stopping": rt.id})
            elif cmd == "pause":
                rt.pause_requested = True
                self._reply(req_id, {"pausing": rt.id})
            elif cmd == "resume":
                rt.pause_requested = False
                self._reply(req_id, {"resuming": rt.id})
            elif cmd == "status":
                self._reply(req_id, rt.status())
            elif cmd == "diagnose":
                # Safe at a step boundary (the run loop is parked here); state
                # is frozen, so a fresh post-mortem is well-defined.  Handy to
                # pause a struggling run and inspect the Jacobian live.
                try:
                    args = msg.get("args", {}) or {}
                    report = rt.model.diagnose(
                        top_k=int(args.get("top_k", 12) or 12))
                    rt.last_diagnostic = report
                    self._reply(req_id, report)
                except Exception as exc:  # noqa: BLE001
                    self._reply_error(req_id, exc, kind=type(exc).__name__)
            elif cmd == "list_vars":
                # Read-only introspection of the recorded-variable names, which
                # are fixed for the life of the model.  Servicing it mid-run lets
                # a UI resolve e.g. a regex aggregate's columns during the *first*
                # run instead of having to wait until the run finishes.
                self._reply(req_id, rt.var_names())
            elif cmd == "list_params":
                self._reply(req_id, rt.param_names())
            elif cmd == "vars_stream":
                self._handle_open_stream(rt, req_id, msg.get("args", {}) or {})
            elif cmd == "add_stream_vars":
                self._handle_add_stream_vars(rt, req_id, msg.get("args", {}) or {})
            elif cmd == "close_stream":
                self._handle_close_stream(rt, req_id, msg.get("args", {}) or {})
            elif cmd == "set_param":
                # Applied at this step boundary (never mid-solve), so a UI can
                # nudge a parameter live and the very next step picks it up.
                try:
                    assignments = _param_assignments(msg.get("args", {}) or {})
                    applied = _apply_params(rt.model, assignments)
                    self._broadcast({"op": "set_param", "sid": rt.id,
                                     "assignments": assignments})
                    self._reply(req_id, applied)
                except Exception as exc:  # noqa: BLE001 - surface to the UI
                    self._reply_error(req_id, exc, kind=type(exc).__name__)
            elif cmd == "update_run_config":
                try:
                    applied = _apply_run_config(
                        rt, msg.get("args", {}) or {})
                    self._reply(req_id, applied)
                except Exception as exc:  # noqa: BLE001
                    self._reply_error(req_id, exc, kind=type(exc).__name__)
            else:
                self._reply_error(
                    req_id,
                    f"system {rt.id} is running; only 'stop'/'pause'/'resume'/"
                    f"'status'/'diagnose'/'list_vars'/'list_params'/'set_param'/"
                    f"'update_run_config'/'vars_stream'/'add_stream_vars'/"
                    f"'close_stream' are accepted until it finishes",
                    kind="Busy",
                )

    def _wait_while_paused(self, sel, rt):
        """Park a running system at a step boundary until resumed or stopped.

        Blocks the run loop without advancing the model, so model state is held
        exactly where the pause landed.  We keep servicing the command socket
        (via :meth:`_drain_during_run`) so ``resume`` / ``stop`` / ``status``
        stay responsive, and emit a ``status`` event on each phase transition so
        the UI can reflect paused / running without polling.
        """
        rt.phase = "paused"
        self._bus.emit({"type": "status", **rt.status()})
        # No socket means there is no way to receive a resume (e.g. an MPI
        # follower, which never reaches this path) -- don't deadlock.
        if self._sock is None:
            rt.pause_requested = False
        while rt.pause_requested and not rt.stop_requested and not self._shutdown:
            # Block cheaply until a command arrives instead of busy-spinning;
            # the timeout bounds how quickly we notice a shutdown.
            if sel.select(timeout=0.2):
                self._drain_during_run(sel, rt)
        if not rt.stop_requested and not self._shutdown:
            rt.phase = "running"
            self._bus.emit({"type": "status", **rt.status()})

    def _emit_error(self, rt, exc):
        info = {
            "type": "error", "system_id": rt.id,
            "kind": type(exc).__name__, "message": str(exc),
        }
        for attr in ("error_norm", "iterations", "max_iterations", "tol"):
            if hasattr(exc, attr):
                try:
                    info[attr] = float(getattr(exc, attr))
                except (TypeError, ValueError):
                    info[attr] = getattr(exc, attr)
        # Attach a one-line solver post-mortem + the culprit components so the
        # UI can surface "why" without a follow-up round-trip.
        diag = getattr(rt, "last_diagnostic", None)
        if diag:
            info["diagnostic_summary"] = diag.get("summary")
            info["diagnostic_severity"] = diag.get("severity")
            info["diagnostic_cause_codes"] = diag.get("cause_codes") or []
            info["diagnostic_components"] = [
                c.get("component") for c in (diag.get("components") or [])[:3]]
            info["has_diagnostic"] = True
        self._bus.emit(info)

    # --- MPI broadcast helpers (no-ops when SIZE == 1) --------------------

    def _broadcast(self, op):
        if _COMM is not None and _SIZE > 1:
            _COMM.bcast(op, root=0)

    def _broadcast_control(self, go: bool) -> bool:
        if _COMM is not None and _SIZE > 1:
            return bool(_COMM.bcast({"op": "run_step", "go": go}, root=0)["go"])
        return go


# ---------------------------------------------------------------------------
# Follower ranks (MPI only): mirror the lead's collective ops in lockstep
# ---------------------------------------------------------------------------


def _run_follower():  # pragma: no cover - requires mpirun -n >1
    """Non-zero ranks: build/instantiate/step in lockstep with rank 0.

    No socket, no streaming.  This validates the collective control path; the
    per-step solve becomes a genuinely distributed operation once a
    domain-decomposed model backend exists.
    """
    from ..serialization import from_dict, from_json

    systems = {}
    while True:
        op = _COMM.bcast(None, root=0)
        kind = op.get("op")
        if kind == "shutdown":
            return
        if kind == "load_json":
            systems[op["sid"]] = from_json(op["spec"])
        elif kind == "load_dict":
            systems[op["sid"]] = from_dict(op["spec"])
        elif kind == "instantiate":
            systems[op["sid"]].instantiate(**op["opts"])
        elif kind == "initialise":
            systems[op["sid"]].initialise(n=op["n"], **op["kw"])
        elif kind == "step":
            m = systems[op["sid"]]
            m.solve_dae_step(op["dt"])
            m.next_step()
        elif kind == "set_param":
            _apply_params(systems[op["sid"]], op["assignments"])
        elif kind == "close":
            systems.pop(op["sid"], None)
        elif kind == "run_step":
            # Should be consumed inside the follower run loop below; ignore here.
            pass


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _serve_lead(host: str, port: int):
    """Rank 0 (or the only process): own the socket and the command loop."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    conn, _peer = listener.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    listener.close()

    bus = _EventBus(conn)
    engine = _Engine(bus, conn, _RANK)
    logging.getLogger("hydrogen").info("host ready (rank %d/%d)", _RANK, _SIZE)
    try:
        while True:
            try:
                msg = recv_msg(conn)
            except (ProtocolError, OSError):
                break
            if msg is None:
                break
            engine.handle(msg)
            if engine.should_shutdown:
                break
    finally:
        # Tell followers to exit if the client vanished without a clean shutdown.
        if _COMM is not None and _SIZE > 1 and not engine.should_shutdown:
            _COMM.bcast({"op": "shutdown"}, root=0)
        bus.close()
        try:
            conn.close()
        except OSError:
            pass


def serve(host: str = "127.0.0.1", port: int = 0):
    """Process entry point: rank 0 serves; other ranks follow."""
    if _RANK == 0:
        _serve_lead(host, port)
    else:  # pragma: no cover - requires mpirun
        _run_follower()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hydrogen.service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
