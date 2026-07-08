"""Live data pump feeding the canvas plot/table objects off the simulation.

A single :class:`LiveController` owns one variable stream on the currently-built
model (:class:`~hydrogen.ui.session.SimulationSession`) and refills every canvas
object's traces / values off it.  It doubles as the *live source* the content
widgets read: :meth:`latest` for the table's value column and :meth:`series` for
a chart's traces.

**Threading.**  Every host round-trip is *blocking* -- the host handles one
command at a time, so ``vars_stream`` / ``add_stream_vars`` / ``close_stream``
sit in ``recv`` until the host is free (which, while it is compiling a model or
grinding through a slow step, can be seconds).  Doing that on the GUI thread
freezes the window and, worse, makes the pause / stop buttons unresponsive.  So
the pump runs on its **own daemon thread**: it performs all host I/O there and
publishes a plain-array *snapshot* under a lock.  The GUI thread only ever reads
that snapshot (via :meth:`latest` / :meth:`series`) and repaints -- it never
touches the socket.  Results cross back to the GUI thread through queued Qt
signals (``phaseChanged`` for the run controls, an internal refresh request, and
an internal log batch), so all widget work still happens on the GUI thread.
"""

from __future__ import annotations

import threading

import numpy as np

from .derived import compute_series, is_derived, resolve_regex_names
from .plots import (
    PROGRESS_KEY, SPECIAL_KEYS, STEP_KEY, STEPSIZE_KEY, TIME_KEY, WALLTIME_KEY)
from .qt import QtCore, Signal

__all__ = ["LiveController"]


class LiveController(QtCore.QObject):
    """Polls the session's variable stream on a background thread and refreshes
    the registered canvas objects on the GUI thread.

    Being the always-on run monitor, it also drains the host's status/log/done
    events each cycle (refreshing the session's
    :attr:`~hydrogen.ui.session.SimulationSession.run_phase`), forwards those log
    lines to an optional sink (the open Simulate window's log panel), and emits
    :data:`phaseChanged` on every run-phase transition -- so the toolbar's
    pause/stop controls stay in sync even with no window open.
    """

    #: Emitted (queued, on the GUI thread) with the new run phase ("" when there
    #: is no active run) whenever it changes -- drives the toolbar run controls.
    phaseChanged = Signal(str)

    #: Emitted (queued, on the GUI thread) whenever the session's per-component
    #: warning set changes -- drives the canvas node warning badges.
    warningsChanged = Signal()

    #: Internal bg->GUI hand-offs (queued): repaint request + a batch of
    #: ``(message, level)`` log lines to forward to the sink.
    _refreshRequested = Signal()
    _logsCollected = Signal(object)

    def __init__(self, session, interval_ms: int = 250, parent=None):
        super().__init__(parent)
        self._session = session
        self._interval = max(0.02, interval_ms / 1000.0)
        self._contents: list = []            # content widgets to feed
        self._contents_lock = threading.Lock()
        self._stream = None
        self._sysp_key = None                # id() of the sysp the stream belongs to
        self._time = None                    # StreamHandle over time
        self._handles: dict = {}             # full_name -> StreamHandle (1-D)
        self._values_handles: dict = {}      # pattern -> StreamHandle (2-D)
        self._derived_specs: dict = {}       # derived full -> agg spec
        self._all_var_names: list = []       # every recorded name (for regex aggs)
        self._regex_handles: dict = {}       # derived full -> 2-D StreamHandle
        # derived full -> {group alias -> 2-D StreamHandle} for formula groups.
        self._formula_groups: dict = {}
        self._log_sink = None                # optional callable(msg, level)
        self._force_reconnect = False        # set by notify_model_ready (GUI thread)
        self._last_phase = None              # last phase we announced
        self._last_warn_ver = 0              # last session.warnings_version seen
        # Charts show only the *current run*: `_start_index` is where that run's
        # samples begin in the accumulated stream buffer (a re-run appends a
        # fresh t=0.. segment); `_reset_index` is a floor pinned to the current
        # run's first sample the moment a new run begins, so the previous run's
        # tail is dropped while this run's t=0 is kept.
        self._start_index = 0
        self._reset_index = 0
        # GUI-facing snapshot published by the pump thread each cycle.
        self._snap_lock = threading.Lock()
        self._snapshot: dict = {}            # full -> (t_slice, value_slice)
        self._latest: dict = {}              # full -> float | None
        # Background pump thread.
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Marshal bg-thread results back onto the GUI thread (queued, since the
        # controller lives on the GUI thread while these are emitted from the
        # pump thread).
        self._refreshRequested.connect(self._refresh_contents)
        self._logsCollected.connect(self._forward_logs)

    # --- registration (GUI thread) ---------------------------------------- #
    def set_contents(self, contents: list):
        """Register the current set of canvas object content widgets."""
        with self._contents_lock:
            self._contents = list(contents)
        for c in contents:
            c.set_live_source(self)

    def notify_model_ready(self):
        """Hint that a model was just built or a run is about to stream.

        Called from the GUI thread after a toolbar / simulate worker finishes
        building so the pump re-binds its stream on the next cycle instead of
        waiting for a phase transition.
        """
        self._force_reconnect = True

    def set_log_sink(self, sink):
        """Route host run logs to ``sink(message, level)`` (or ``None`` to drop
        them).  Set by the Simulate window while it is open."""
        self._log_sink = sink

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="hydrogen-live-pump", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def shutdown(self):
        # Just stop the pump thread; the host (and its streams) are torn down by
        # the session right after, so we avoid a blocking close_stream here.
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None

    # --- live-source contract (read by the content widgets, GUI thread) --- #
    def latest(self, full: str):
        with self._snap_lock:
            return self._latest.get(full)

    def series(self, full: str):
        with self._snap_lock:
            pair = self._snapshot.get(full)
        if pair is None:
            return np.empty(0), np.empty(0)
        return pair

    # --- background pump --------------------------------------------------- #
    def _run_loop(self):
        while not self._stop.is_set():
            try:
                self._pump_once()
            except Exception:
                # Never let a transient host/stream error kill the pump.
                pass
            self._stop.wait(self._interval)

    def _pump_once(self):
        sysp = getattr(self._session, "sysp", None)
        if sysp is None:                     # not built (or mid-rebuild) -> idle
            if self._stream is not None:
                self._close_stream()
            self._drain_run_events()
            self._start_index = 0
            self._publish_snapshot()
            self._refreshRequested.emit()
            return
        if self._force_reconnect or id(sysp) != self._sysp_key:
            self._close_stream()
            self._sysp_key = id(sysp)
            self._force_reconnect = False
        # Phase transitions (new run, etc.) may clear handles; do that before we
        # (re)open the stream and pull samples so the same pump cycle can paint.
        self._drain_run_events()
        if self._stream is None:
            self._open_stream(sysp)
        if self._stream is not None:
            self._register_needed()
            try:
                self._stream.update()
            except Exception:
                # Host can be briefly busy during build/initialise; drop the
                # stale stream handle and retry opening on the next cycle.
                self._close_stream()
        self._start_index = max(self._compute_start_index(), self._reset_index)
        self._publish_snapshot()
        self._refreshRequested.emit()

    def _publish_snapshot(self):
        """Snapshot each watched series (sliced to the current run) into plain
        arrays the GUI thread can read without touching the host or the stream."""
        snap: dict = {}
        latest: dict = {}
        t = self._time.array if self._time is not None else None
        if t is not None:
            for full, h in list(self._handles.items()):
                a = h.array
                n = min(len(t), len(a))
                i = max(0, min(self._start_index, n))
                snap[full] = (t[i:n], a[i:n])
                latest[full] = float(a[-1]) if len(a) else None
            for dfull, agg in self._derived_specs.items():
                try:
                    vhandles = self._values_handles
                    group_arrays = None
                    # A regex aggregate reduces one persistent 2-D handle (all of
                    # its matched columns), fed through the reduction path as a
                    # ``pattern``-style matrix -> ``handle.array`` reduced axis=1.
                    if agg.get("regex") is not None:
                        h = self._regex_handles.get(dfull)
                        if h is None:
                            latest[dfull] = None
                            continue
                        key = f"__rx__:{dfull}"
                        agg = {k: v for k, v in agg.items() if k != "regex"}
                        agg["pattern"] = key
                        vhandles = {**self._values_handles, key: h}
                    elif agg.get("axis") == "formula" and agg.get("groups"):
                        # Snapshot each group's 2-D handle into a plain matrix the
                        # formula evaluator reduces (sum(g1) etc.).
                        gh = self._formula_groups.get(dfull, {})
                        group_arrays = {a: np.asarray(h.array, dtype=float)
                                        for a, h in gh.items()}
                    ts, ys = compute_series(
                        agg,
                        time_array=t,
                        start_index=self._start_index,
                        series_handles=self._handles,
                        values_handles=vhandles,
                        group_arrays=group_arrays,
                    )
                    snap[dfull] = (ts, ys)
                    latest[dfull] = float(ys[-1]) if len(ys) else None
                except Exception:
                    pass
            # The current simulation time + last step size dt, for the table's
            # time / step-size rows (dt is the gap between the last two samples;
            # a run restart's backward jump is ignored).
            if len(t):
                latest[TIME_KEY] = float(t[-1])
            if len(t) >= 2:
                dt = float(t[-1] - t[-2])
                latest[STEPSIZE_KEY] = dt if dt > 0 else None
        # Run-status rows (step / progress) come from the session, not the
        # stream, so they work even for a table with no watched variables.
        latest[STEP_KEY] = float(getattr(self._session, "run_step", 0) or 0)
        prog = getattr(self._session, "run_progress", None)
        latest[PROGRESS_KEY] = None if prog is None else prog * 100.0
        latest[WALLTIME_KEY] = getattr(self._session, "run_wall_time", None)
        with self._snap_lock:
            self._snapshot = snap
            self._latest = latest

    def _drain_run_events(self):
        """Drain host run events (updating the session's run phase + collecting
        log lines) and announce any phase transition.  A transition *into*
        "running" that isn't a resume marks a new run -> blank the charts."""
        msgs: list = []
        try:
            self._session.poll_logs(
                lambda m, level="status": msgs.append((m, level)))
        except Exception:
            pass
        if msgs:
            self._logsCollected.emit(msgs)
        # Announce component-warning changes (build/run raised or cleared some)
        # so the GUI can refresh the canvas badges.
        ver = getattr(self._session, "warnings_version", 0)
        if ver != self._last_warn_ver:
            self._last_warn_ver = ver
            self.warningsChanged.emit()
        phase = self._session.run_phase
        if phase != self._last_phase:
            prev, self._last_phase = self._last_phase, phase
            if phase == "running" and prev != "paused":
                # New run: drop the previous run's samples, but keep this run's
                # own t=0.  `initialise` records that t=0 (and the first step or
                # two often stream in) *before* we observe the "running" phase,
                # so anchoring to `len(buffer)` here would skip past t=0 and make
                # the chart appear to start at the first step (e.g. 1 s).  The
                # run's true first sample is `_compute_start_index()`: 0 for a
                # fresh run, or one past the re-initialise backward jump.
                self._reset_index = self._compute_start_index()
                # NB: we deliberately keep the existing stream handles here.  The
                # stream persists across a re-run of the same model (a re-run just
                # appends a fresh t=0.. segment, sliced off by `_reset_index`), so
                # tearing the handles down every run only churned registrations
                # and re-triggered per-column backfills -- which is exactly what
                # made a live value flicker in and then drop back to a dash.  A
                # genuine rebuild swaps `sysp`, which reconnects the stream (and
                # rebuilds the handles) via `_open_stream`.
            self.phaseChanged.emit(phase or "")

    def _compute_start_index(self) -> int:
        """Index of the current run's first sample = one past the last backward
        jump in recorded time (a re-``initialise`` resets t to 0), or 0."""
        if self._time is None:
            return 0
        t = np.asarray(self._time.array)
        if t.size < 2:
            return 0
        drops = np.nonzero(np.diff(t) < 0)[0]
        return int(drops[-1]) + 1 if drops.size else 0

    def _open_stream(self, sysp):
        try:
            self._stream = sysp.vars_stream()
            self._time = self._stream.time()
            self._handles = {}
            self._values_handles = {}
            self._regex_handles = {}
            self._formula_groups = {}
            self._all_var_names = []
            # Snapshot the model's recorded variable names so regex aggregates can
            # be resolved without a host round-trip each pump cycle.  Fetched
            # lazily/retried by `_ensure_var_names` because a `list_vars` issued
            # right after (re)opening -- while the host is still busy building or
            # stepping -- can come back empty; a one-shot fetch here would then
            # leave every regex aggregate stuck on a dash.
            self._ensure_var_names(sysp)
        except Exception:
            self._stream = None
            self._time = None

    def _ensure_var_names(self, sysp=None):
        """Populate ``_all_var_names`` (the recorded-name universe) if empty.

        Safe to call every cycle: it only hits the host until it gets a
        non-empty list, so a regex aggregate re-resolves as soon as the model's
        variables become queryable, even if the first attempt raced the build.
        """
        if self._all_var_names:
            return
        if sysp is None:
            sysp = getattr(self._session, "sysp", None)
        if sysp is None:
            return
        try:
            self._all_var_names = list(sysp.list_vars())
        except Exception:
            self._all_var_names = []

    def _collect_stream_requests(self):
        """Raw series names + derived aggregation specs requested by plot objects."""
        needed: set[str] = set()
        derived: dict = {}
        with self._contents_lock:
            contents = list(self._contents)
        for c in contents:
            try:
                for full in c.variable_names():
                    if full in SPECIAL_KEYS or is_derived(full):
                        continue
                    needed.add(full)
                specs = getattr(c, "derived_specs", None)
                if callable(specs):
                    derived.update(specs())
            except Exception:
                pass
        return needed, derived

    def _register_needed(self):
        needed, derived = self._collect_stream_requests()
        self._derived_specs = derived

        for full in needed - set(self._handles):
            try:
                self._handles[full] = self._stream.series(full)
            except Exception:
                pass

        # Drop the 2-D handles of any regex aggregate / formula group no longer
        # shown, so a removed derived variable stops pinning columns.
        for dfull in list(self._regex_handles):
            if dfull not in derived:
                self._regex_handles.pop(dfull, None)
        for dfull in list(self._formula_groups):
            if dfull not in derived:
                self._formula_groups.pop(dfull, None)

        for dfull, agg in derived.items():
            axis = agg.get("axis", "instances")
            if axis == "time":
                for src in agg.get("sources") or []:
                    if src not in self._handles:
                        try:
                            self._handles[src] = self._stream.series(src)
                        except Exception:
                            pass
                continue
            if axis == "formula":
                # Watch each scalar input as a 1-D series; each group input as a
                # single 2-D handle over its regex-matched columns (re-resolved
                # against the live model, like a standalone regex aggregate).
                for src in (agg.get("variables") or {}).values():
                    if src not in self._handles:
                        try:
                            self._handles[src] = self._stream.series(src)
                        except Exception:
                            pass
                groups = agg.get("groups") or {}
                if groups:
                    gh = self._formula_groups.setdefault(dfull, {})
                    for alias, gspec in groups.items():
                        if alias in gh:
                            continue
                        self._ensure_var_names()
                        matched = resolve_regex_names(
                            self._all_var_names, gspec.get("regex", ""),
                            gspec.get("scope"))
                        if matched:
                            try:
                                gh[alias] = self._stream.series_multi(matched)
                            except Exception:
                                pass
                continue
            # Regex aggregate: resolve the pattern to its matching columns ONCE
            # per stream and watch them as a single 2-D handle (one host call,
            # one backfill).  It is held for the life of the stream and reduced
            # with `handle.array` on every frame; a rebuild opens a fresh stream
            # (clearing this), which re-resolves against the new model so the
            # instance count (e.g. n_segments) stays current.
            regex = agg.get("regex")
            if regex is not None:
                if dfull not in self._regex_handles:
                    self._ensure_var_names()   # retry if the open-time fetch was empty
                    matched = resolve_regex_names(
                        self._all_var_names, regex, agg.get("scope"))
                    if matched:
                        try:
                            self._regex_handles[dfull] = \
                                self._stream.series_multi(matched)
                        except Exception:
                            pass
                continue
            pattern = agg.get("pattern")
            if pattern and pattern not in self._values_handles:
                try:
                    self._values_handles[pattern] = self._stream.series_values(
                        pattern)
                except Exception:
                    pass
            for src in agg.get("sources") or []:
                if src not in self._handles:
                    try:
                        self._handles[src] = self._stream.series(src)
                    except Exception:
                        pass

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._time = None
        self._handles = {}
        self._values_handles = {}
        self._regex_handles = {}
        self._formula_groups = {}
        self._derived_specs = {}
        self._all_var_names = []
        self._start_index = 0
        self._reset_index = 0

    # --- GUI-thread slots (queued from the pump thread) -------------------- #
    def _forward_logs(self, msgs):
        sink = self._log_sink
        if sink is None:
            return
        for message, level in msgs:
            try:
                sink(message, level)
            except Exception:
                pass

    def _refresh_contents(self):
        with self._contents_lock:
            contents = list(self._contents)
        for c in contents:
            try:
                c.refresh_live()
            except Exception:
                pass
