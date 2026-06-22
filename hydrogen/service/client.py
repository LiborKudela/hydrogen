"""Client-side handles for driving a hydrogen host from another process.

This is the *only* piece a UI tool needs.  It is **stdlib-only** (no hydrogen
imports) so it stays light and import-cheap in the UI process::

    import hydrogen

    service = hydrogen.start_host(workers=5)        # spawns `mpirun -n 5 ...`
    system  = service.load_json(open("sys.json").read())
    system.instantiate()
    system.initialise(n=1)
    system.run(dt=2.0, steps=40, stream=True)       # advance + record
    stream = system.vars_stream()                   # watches nothing yet
    Ta = stream.series("wall.T_a")                  # live handle (registers it)
    t  = stream.time()
    while running:
        if stream.update():                         # redraw only on new rows
            chart(t.array, Ta.array)
    service.shutdown()

Design (actor / request-reply over one socket):

* `start_host` launches the host as a subprocess (directly for ``workers=1``,
  under ``mpirun -n N`` for ``workers > 1``) and connects to it, returning a
  :class:`HostService`.
* :class:`HostService` is the *connection*: it hands out :class:`SystemProxy`
  handles (one per loaded model) and owns ``shutdown``.
* :class:`SystemProxy` mirrors the model lifecycle (``instantiate`` ->
  ``initialise`` -> ``run``), in-flight run control (``stop`` / ``pause`` /
  ``resume``, honoured at step boundaries), live parameter edits
  (``set_param`` / ``set_params``, also honoured mid-run at step boundaries;
  discover names with ``list_params``) plus reads: streamed events
  (``events`` / ``poll_events``), on-demand variable streams (``vars_stream``
  -> a :class:`Stream`), point queries (``status``, ``get_state``), timeseries
  (``get_record`` row-major, ``get_series`` column-major), and variable
  discovery (``list_vars`` flat, ``var_tree`` nested for a UI picker).
* :class:`Stream` is an independent, on-demand channel over a chosen set of
  variables (open any time, several at once); poll/iterate its ``stream_data``
  chunks and ``close`` it when done.
* `_Conn` multiplexes the socket: synchronous request/reply (matched by a
  correlation id) for commands, and a background reader that demuxes streamed
  ``status`` / ``log`` / ``error`` / ``done`` events into a queue per
  ``system_id``, and ``stream_data`` / ``stream_closed`` chunks into a queue
  per ``(system_id, stream_id)``.
"""

from __future__ import annotations

import itertools
import queue
import socket
import subprocess
import sys
import threading
import time
from typing import Iterator

from .protocol import ProtocolError, recv_msg, send_msg

# Sentinel pushed onto every per-system event queue when the connection drops,
# so a blocked `events()` generator wakes up instead of hanging forever.
_CLOSED = {"type": "closed"}

# Sentinel pushed onto every open variable-stream queue when the connection
# drops, so a blocked `Stream.events()` generator wakes up and terminates.
_STREAM_CLOSED = {"type": "stream_closed"}

# Event types that terminate a `SystemProxy.events()` stream.
_TERMINAL = ("done", "closed")

# Host->client event types that belong to a variable stream (routed by
# (system_id, stream_id) rather than by system_id alone).
_STREAM_TYPES = ("stream_data", "stream_closed")


class HostError(RuntimeError):
    """Raised in the client when the host returns an error reply.

    Carries the structured ``kind`` (e.g. ``"NewtonConvergenceFailure"``) and
    any extra fields the host attached, so the UI can branch on the failure.
    """

    def __init__(self, message, kind=None, **extra):
        super().__init__(message)
        self.kind = kind
        self.extra = extra


class HostConnectionError(RuntimeError):
    """Raised when the host process dies or the socket closes unexpectedly."""


class _Conn:
    """Owns the socket: request/reply correlation + event demux by system_id."""

    def __init__(self, proc: subprocess.Popen, sock: socket.socket):
        self._proc = proc
        self._sock = sock
        self._ids = itertools.count(1)
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._waiters: dict[int, threading.Event] = {}
        self._results: dict[int, dict] = {}
        self._event_qs: dict[str, queue.Queue] = {}
        # Per-stream queues keyed by (system_id, stream_id).
        self._stream_qs: dict[tuple, queue.Queue] = {}
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_loop, name="hydrogen-host-reader", daemon=True
        )
        self._reader.start()

    # --- background reader ------------------------------------------------

    def _read_loop(self):
        try:
            while True:
                try:
                    msg = recv_msg(self._sock)
                except (ProtocolError, OSError):
                    msg = None
                if msg is None:
                    break
                if msg.get("type") == "reply":
                    self._resolve(msg)
                else:
                    self._route_event(msg)
        finally:
            self._mark_closed()

    def _resolve(self, msg):
        mid = msg.get("id")
        with self._state_lock:
            ev = self._waiters.pop(mid, None)
            if ev is not None:
                self._results[mid] = msg
        if ev is not None:
            ev.set()

    def _route_event(self, msg):
        if msg.get("type") in _STREAM_TYPES:
            key = (msg.get("system_id"), msg.get("stream_id"))
            self._stream_q(key).put(msg)
        else:
            self._event_q(msg.get("system_id")).put(msg)

    def _mark_closed(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            waiters = list(self._waiters.items())
            self._waiters.clear()
            qs = list(self._event_qs.values())
            stream_qs = list(self._stream_qs.values())
        # Wake every blocked caller / generator.
        for mid, ev in waiters:
            self._results[mid] = {
                "type": "reply", "id": mid, "status": "error",
                "message": "host connection closed", "kind": "HostConnectionError",
            }
            ev.set()
        for q in qs:
            q.put(_CLOSED)
        for q in stream_qs:
            q.put(_STREAM_CLOSED)

    # --- request/reply ----------------------------------------------------

    def call(self, cmd, *, timeout=None, **args):
        with self._state_lock:
            if self._closed:
                raise HostConnectionError("host connection is closed")
            mid = next(self._ids)
            ev = threading.Event()
            self._waiters[mid] = ev
        payload = {"id": mid, "cmd": cmd, "args": args}
        try:
            with self._send_lock:
                send_msg(self._sock, payload)
        except OSError as exc:
            raise HostConnectionError(f"failed to send {cmd!r}: {exc}") from exc
        if not ev.wait(timeout):
            raise TimeoutError(f"timed out waiting for reply to {cmd!r}")
        msg = self._results.pop(mid)
        if msg.get("status") == "error":
            raise HostError(
                msg.get("message", "host error"),
                kind=msg.get("kind"),
                **{k: v for k, v in msg.items()
                   if k not in ("type", "id", "status", "message", "kind")},
            )
        return msg.get("result")

    # --- events -----------------------------------------------------------

    def _event_q(self, sid) -> queue.Queue:
        with self._state_lock:
            q = self._event_qs.get(sid)
            if q is None:
                q = queue.Queue()
                self._event_qs[sid] = q
            return q

    def poll_events(self, sid) -> list:
        q = self._event_q(sid)
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                break
        return out

    def iter_events(self, sid, timeout=None) -> Iterator[dict]:
        q = self._event_q(sid)
        while True:
            try:
                msg = q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("timed out waiting for the next event")
            if msg.get("type") in _TERMINAL:
                return
            yield msg

    # --- variable streams -------------------------------------------------

    def _stream_q(self, key) -> queue.Queue:
        with self._state_lock:
            q = self._stream_qs.get(key)
            if q is None:
                q = queue.Queue()
                self._stream_qs[key] = q
            return q

    def poll_stream(self, key) -> list:
        q = self._stream_q(key)
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                break
        return out

    def iter_stream(self, key, timeout=None) -> Iterator[dict]:
        q = self._stream_q(key)
        while True:
            try:
                msg = q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("timed out waiting for the next stream chunk")
            if msg.get("type") == "stream_closed":
                return
            yield msg

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
        self._mark_closed()


def _match_one_name(keys, req):
    """First key matching ``req``: exact, else dotted-suffix, else bare-suffix.

    Mirrors :func:`hydrogen.model.match_name_index` so the client resolves
    names the same way the model and host do, without importing hydrogen.
    """
    if req in keys:
        return req
    for k in keys:
        if k.endswith("." + req):
            return k
    for k in keys:
        if k.endswith(req):
            return k
    return None


def _match_all_names(keys, req):
    """All keys matching ``req`` (exact if any exact match exists, else every
    dotted-/bare-suffix match) -- mirrors
    :func:`hydrogen.model.match_name_indices`.
    """
    exact = [k for k in keys if k == req]
    if exact:
        return exact
    return [k for k in keys if k.endswith("." + req) or k.endswith(req)]


class StreamHandle:
    """A live view over one :class:`Stream` selection.

    Returned by :meth:`Stream.series`, :meth:`Stream.series_values` and
    :meth:`Stream.time`.  The real numpy array lives on the :attr:`array`
    attribute and is refreshed in place by :meth:`Stream.update` -- so you keep
    the handle once and read ``handle.array`` each frame, e.g.::

        p  = stream.series("p_in")            # 1-D
        up = stream.series_values("m_dot")    # 2-D (rows=time, cols=matches)
        t  = stream.time()
        while running:
            stream.update()                   # refills every handle, one length
            redraw(t.array, p.array, up.array.sum(axis=1))

    :meth:`Stream.update` guarantees all handles share the same row count, so no
    manual length-clipping is needed.  Materialising (``p.array.sum()``,
    ``np.abs(en.array)``) returns a *new* array and never disturbs the handle.

    The handle also proxies the numpy array protocol (``np.asarray(h)``,
    ``len(h)``, ``h[i]``), so it can be handed straight to a consumer like
    matplotlib's ``set_data`` -- though reading ``.array`` explicitly is clearer.
    """

    __slots__ = ("array", "_stream", "_selector")

    def __init__(self, stream, selector, array):
        self._stream = stream
        self._selector = selector       # ("time",) | ("series", name) | ("series_values", name)
        self.array = array

    def __repr__(self):
        sel = self._selector
        label = sel[1] if len(sel) > 1 else sel[0]
        return f"<StreamHandle {label!r} shape={getattr(self.array, 'shape', None)}>"

    # numpy array protocol -- `copy` kwarg added for numpy>=2 compatibility.
    def __array__(self, dtype=None, copy=None):
        a = self.array
        if dtype is not None:
            return a.astype(dtype)
        return a.copy() if copy else a

    def __len__(self):
        return len(self.array)

    def __getitem__(self, k):
        return self.array[k]

    def _refresh(self, n):
        self.array = self._stream._read(self._selector, n)


class Stream:
    """A handle to one open variable stream on the host.

    Created by :meth:`SystemProxy.vars_stream`.  The stream starts watching
    nothing; ``series`` / ``series_values`` (or :meth:`watch`) register a name
    on the host on first use -- always expanding a suffix to every match -- so
    ``series_values`` aggregates across per-instance matches.  The full recorded
    history is backfilled before live data, so a chart is complete the moment a
    variable is requested.  :meth:`list_watched_names` reports what is watched.

    Two ways to consume it:

    * **Live array handles** (preferred) -- :meth:`time`, :meth:`series` and
      :meth:`series_values` each return a :class:`StreamHandle` whose
      :attr:`~StreamHandle.array` is refreshed together by a single
      :meth:`update` call (which also flags new rows), so every handle stays the same
      length with no manual clipping.  Only the watched columns are transferred
      (never the full record).
    * **Raw chunks** -- :meth:`poll` for a non-blocking drain or :meth:`events`
      to iterate until the stream is closed.  Each ``stream_data`` chunk is
      ``{"type": "stream_data", "time": [...], "series": {name: [...]},
      "initial": bool}`` where ``time[k]`` is the timestamp of
      ``series[name][k]`` for every watched ``name`` (``initial`` flags the
      history backfill chunk).

    Both styles share one underlying queue (:meth:`update` and :meth:`poll`
    drain the same chunks); pick one style per stream to keep things clear.
    Close it with :meth:`close` (or use it as a context manager) when done.
    """

    def __init__(self, conn: "_Conn", system_id: str, stream_id: str, vars: list):
        self._conn = conn
        self.system_id = system_id
        self.stream_id = stream_id
        self.vars = list(vars)          # full names currently watched on the host
        self._requested: set = set()    # name patterns already registered
        self._closed = False
        self._time: list = []           # accumulated time points
        self._data: dict = {}           # key -> accumulated values (aligned)
        self._handles: list = []        # live StreamHandles refreshed by update()
        self._seen_n = 0                # rows seen at the last update() (new-row flag)

    def __repr__(self):
        return f"<Stream {self.stream_id!r} vars={self.vars}>"

    @property
    def _key(self):
        return (self.system_id, self.stream_id)

    def _ingest(self, chunks: list) -> list:
        """Fold any ``stream_data`` chunks into the accumulation buffer.

        All watched columns share one ``time`` vector per chunk, so the buffers
        stay aligned with :attr:`_time` chunk-by-chunk.  An ``initial`` chunk is
        a full history (re)backfill -- emitted on open with ``scope="all"`` and
        again whenever a column is added dynamically -- so it *resets* the
        buffer before refilling, keeping every column aligned.
        """
        for ch in chunks:
            if ch.get("type") != "stream_data":
                continue
            if ch.get("initial"):
                self._time = []
                self._data = {}
            self._time.extend(ch.get("time", []))
            for name, vals in ch.get("series", {}).items():
                self._data.setdefault(name, []).extend(vals)
        return chunks

    def poll(self) -> list:
        """Non-blocking drain of stream chunks queued so far (also buffered for
        :meth:`series` / :meth:`series_values`)."""
        return self._ingest(self._conn.poll_stream(self._key))

    def events(self, timeout=None) -> Iterator[dict]:
        """Yield ``stream_data`` chunks until the stream is closed."""
        for ch in self._conn.iter_stream(self._key, timeout=timeout):
            self._ingest([ch])
            yield ch

    def __iter__(self) -> Iterator[dict]:
        return self.events()

    # --- dynamic watching -------------------------------------------------

    def watch(self, *names):
        """Register one or more name patterns on the host so they start
        streaming (each is *expanded* to every match, keyed by full name).

        Called automatically by :meth:`series` / :meth:`series_values`, so you
        rarely need it directly; it is handy to pre-register before a run.
        Returns the full names added by this call.
        """
        flat = []
        for n in names:
            flat.extend([n] if isinstance(n, str) else list(n))
        added_all = []
        for pattern in flat:
            added_all.extend(self._ensure_watched(pattern))
        return added_all

    def list_watched_names(self):
        """The full variable names currently watched (in host order).

        Grows as :meth:`series` / :meth:`series_values` / :meth:`watch`
        register patterns on a stream opened without explicit ``vars``.
        """
        return list(self.vars)

    def _ensure_watched(self, name) -> list:
        """Register ``name`` on the host if not already; return names added."""
        if self._closed or name in self._requested:
            return []
        self._requested.add(name)
        try:
            res = self._conn.call(
                "add_stream_vars", system_id=self.system_id,
                stream_id=self.stream_id, vars=[name],
            )
        except (HostError, HostConnectionError, TimeoutError):
            self._requested.discard(name)  # allow a later retry
            raise
        self.vars = res.get("vars", self.vars)
        added = res.get("added", []) or []
        if added:
            # The host emits a fresh `initial` backfill right after this reply;
            # wait until the new column(s) actually land so a handle refreshed
            # right after registration is populated (and so other handles never
            # see a frame where this column is still missing).
            self._drain_until(added)
        return added

    def _drain_until(self, names, timeout=0.5):
        """Poll until every name in ``names`` is present in the buffer (or a
        short timeout elapses)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if all(n in self._data for n in names):
                return
            time.sleep(0.005)

    # --- live array handles (refreshed together by update()) -------------

    def update(self):
        """Drain newly-arrived chunks once, refresh every handle to one
        consistent length, and return ``True`` if new rows arrived since the
        last :meth:`update` call -- a ready-made redraw guard::

            while running:
                if stream.update():
                    redraw(t.array, p.array, ...)

        This is the only place the array path polls the host, so all handles
        returned by :meth:`series` / :meth:`series_values` / :meth:`time` stay
        the same length on every frame -- read ``handle.array`` afterwards with
        no manual clipping (the reads are valid whether or not it returned
        ``True``).
        """
        self.poll()
        n = self._current_n()
        for h in self._handles:
            h._refresh(n)
        fresh = n > self._seen_n
        self._seen_n = n
        return fresh

    def _current_n(self) -> int:
        """Rows safe to expose this frame.

        The time length, clamped to the shortest column any *registered handle*
        needs.  Columns are normally all equal; this only bites in the brief
        window after a handle is registered but before its freshly-added
        column's backfill has arrived -- there we report fewer rows (possibly 0)
        so every handle stays the same length rather than one lagging at 0.
        """
        n = len(self._time)
        for key in self._handle_keys():
            if key is None:        # unmatched name -> don't stall the whole stream
                continue
            n = min(n, len(self._data.get(key, [])))
        return n

    def _handle_keys(self):
        """Every buffer key referenced by the currently registered handles."""
        keys = set()
        for h in self._handles:
            sel = h._selector
            if sel[0] == "series":
                keys.add(_match_one_name(self._buffer_keys(), sel[1]))
            elif sel[0] == "series_values":
                keys.update(_match_all_names(self._buffer_keys(), sel[1]))
        return keys

    def time(self):
        """A live :class:`StreamHandle` over the accumulated time points (1-D).
        Refresh it (and all sibling handles) with :meth:`update`."""
        return self._register(("time",))

    def series(self, name):
        """A live :class:`StreamHandle` (1-D) over the *first* watched variable
        matching ``name`` -- the streaming analogue of
        :meth:`hydrogen.Model.series`.

        Registers ``name`` on first use; read ``handle.array`` after
        :meth:`update`.
        """
        self._ensure_watched(name)
        return self._register(("series", name))

    def series_values(self, name):
        """A live :class:`StreamHandle` (2-D, rows = time, columns = matches)
        over *all* watched variables matching ``name`` -- the streaming analogue
        of :meth:`hydrogen.Model.series_values`.

        Registers ``name`` on first use; aggregate across instances with e.g.
        ``handle.array.sum(axis=1)`` after :meth:`update`.
        """
        self._ensure_watched(name)
        return self._register(("series_values", name))

    def _register(self, selector) -> "StreamHandle":
        import numpy as np
        h = StreamHandle(self, selector, np.empty(0))
        self._handles.append(h)
        if self._time:                   # data already buffered -> fill now
            h._refresh(self._current_n())
        return h

    def _read(self, selector, n):
        """Materialise the array for ``selector`` from the buffer, clipped to
        ``n`` rows.  Used by :meth:`StreamHandle._refresh`."""
        import numpy as np
        kind = selector[0]
        if kind == "time":
            return np.asarray(self._time[:n], dtype=float)
        if kind == "series":
            key = _match_one_name(self._buffer_keys(), selector[1])
            if key is None:
                return np.empty(0)
            return np.asarray(self._data.get(key, [])[:n], dtype=float)
        # series_values -> 2-D (rows = time, columns = matches)
        keys = _match_all_names(self._buffer_keys(), selector[1])
        if not keys:
            return np.empty((n, 0))
        cols = [np.asarray(self._data.get(k, [])[:n], dtype=float) for k in keys]
        return np.column_stack(cols)

    def _buffer_keys(self):
        """Watched key names, preferring the host-reported order in
        :attr:`vars` and falling back to whatever has actually arrived."""
        return self.vars or list(self._data)

    def close(self):
        """Close the stream on the host so it stops emitting chunks."""
        if self._closed:
            return None
        self._closed = True
        try:
            return self._conn.call(
                "close_stream", system_id=self.system_id, stream_id=self.stream_id
            )
        except (HostError, HostConnectionError, TimeoutError):
            return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class SystemProxy:
    """A handle to one model loaded on the host (keyed by ``system_id``)."""

    def __init__(self, conn: _Conn, system_id: str):
        self._conn = conn
        self.system_id = system_id

    def __repr__(self):
        return f"<SystemProxy {self.system_id!r}>"

    # --- lifecycle --------------------------------------------------------

    def instantiate(self, **options):
        """Compile the DAE (symbol assignment, lambdify, BLT). Heavy step."""
        return self._conn.call("instantiate", system_id=self.system_id, **options)

    def initialise(self, n=1, relaxation=1.0, tol=1e-6, max_iter=100):
        """Solve to a Newton-consistent state at t = 0."""
        return self._conn.call(
            "initialise", system_id=self.system_id, n=n,
            relaxation=relaxation, tol=tol, max_iter=max_iter,
        )

    def step(self, dt, **kw):
        """Advance a single Crank-Nicolson step and commit it."""
        return self._conn.call("step", system_id=self.system_id, dt=dt, **kw)

    def run(self, dt=None, steps=None, *, stop_time=None, strategy=None,
            adaptive=None, stream=True, every=1, delay=0.0, **kw):
        """Advance the model, mirroring :meth:`hydrogen.Model.run`.

        Give a stop condition: ``steps`` (number of accepted steps) and/or
        ``stop_time`` (integrate until the model time reaches it; the host
        clips the last step to land on it).

        ``strategy`` selects the stepping rule, identical to
        :meth:`Model.solve_adaptive_step`:

        * ``None`` / ``"fixed"`` -- classic fixed-``dt`` loop (``dt`` required).
        * ``"richardson"`` / ``"derivative_limit"`` / ``"predictor_corrector"``
          (or a ``{"name": ..., **params}`` dict) -- the host adapts ``dt``
          internally; ``dt`` is only the first target.

        ``adaptive`` is an optional dict of controller knobs forwarded to the
        host: ``dt_min``, ``dt_max``, ``dt_start``, ``grow``, ``shrink``,
        ``max_retries``, ``relaxation``, ``tol``, ``max_iter``.

        With ``stream=True`` the call returns immediately after the host
        acknowledges; ``status`` progress events then flow asynchronously
        (iterate :meth:`events` / :meth:`poll_events`) and the run can be
        controlled with :meth:`pause` / :meth:`resume` / :meth:`stop`.  With
        ``stream=False`` it blocks until the run finishes and returns a summary
        dict.

        To chart variables while a run advances, open one or more
        :meth:`vars_stream` channels -- variable data is no longer carried by
        ``run`` itself.

        ``every`` throttles the cadence of ``status`` progress events.
        ``delay`` throttles the host to roughly that many seconds of wall-clock
        per step (real-time / slow-motion playback) so a UI can watch the model
        evolve live; ``0.0`` runs as fast as possible.
        """
        return self._conn.call(
            "run", system_id=self.system_id, dt=dt, steps=steps,
            stop_time=stop_time, strategy=strategy, adaptive=adaptive,
            stream=stream, every=every, delay=delay, **kw,
        )

    def stop(self):
        """Request a cooperative stop; honoured at the next step boundary."""
        return self._conn.call("stop", system_id=self.system_id)

    def pause(self):
        """Request a cooperative pause; honoured at the next step boundary.

        The run loop parks (model state frozen where it landed) and emits a
        ``status`` event with ``phase == "paused"``.  Call :meth:`resume` to
        continue or :meth:`stop` to end the run.  Only meaningful while a
        streaming :meth:`run` is in flight.
        """
        return self._conn.call("pause", system_id=self.system_id)

    def resume(self):
        """Resume a paused run; it continues from where it was paused."""
        return self._conn.call("resume", system_id=self.system_id)

    def close(self):
        """Free this model on the host (the connection stays open)."""
        return self._conn.call("close", system_id=self.system_id)

    # --- parameters -------------------------------------------------------

    def list_params(self) -> list:
        """Full names of every settable Parameter (available post-instantiate).

        Pass any of these (or a convenient dotted suffix like ``heat_0.Q_flow``)
        to :meth:`set_param` / :meth:`set_params`.
        """
        return self._conn.call("list_params", system_id=self.system_id)

    def set_param(self, name, value):
        """Set one compile-time Parameter by name; the next solve sees it.

        ``name`` matches like a variable name (exact, else dotted-suffix). The
        value is written straight into the host's live solver buffer, so it
        takes effect on the next :meth:`step` / :meth:`run` step -- including
        mid-run (it is applied at the next step boundary, never mid-solve).
        Returns the applied ``{full_name: value}``.
        """
        return self._conn.call(
            "set_param", system_id=self.system_id, name=name, value=value)

    def set_params(self, params: dict):
        """Set several parameters at once from a ``{name: value}`` mapping.

        Same matching/semantics as :meth:`set_param`; returns the applied
        ``{full_name: value}`` for every entry.
        """
        return self._conn.call(
            "set_param", system_id=self.system_id, params=dict(params))

    # --- reads ------------------------------------------------------------

    def status(self) -> dict:
        """Lifecycle phase + last-solve diagnostics."""
        return self._conn.call("status", system_id=self.system_id)

    def list_vars(self) -> list:
        """Full names of every recorded variable (available post-instantiate)."""
        return self._conn.call("list_vars", system_id=self.system_id)

    def var_tree(self) -> dict:
        """The recorded variables as a nested tree for UI selection.

        Dotted names (``wall.T_a``, ``stack.c1.Q_dot_a``) are folded into a tree
        whose root is ``{"name": "", "path": "", "leaf": False, "count": N,
        "children": [...]}``.  Every node carries:

        * ``name``  -- raw path segment,
        * ``path``  -- display path, **unique across the tree** (use it as the
          UI node key / selection id),
        * ``leaf``  -- ``True`` for a selectable variable, ``False`` for a group
          (explicit, so a name that is also a prefix of another is handled
          correctly -- such a node is a leaf *and* has children),
        * ``count`` -- selectable variables in its subtree (for badges /
          "select all").

        Group nodes also carry a ``children`` list (groups-first, then leaves,
        each natural-sorted).  Leaf nodes also carry their column ``index`` and
        exact ``full`` name -- pass ``full`` (or ``path``) to
        :meth:`get_series`, :meth:`get_record`, or :meth:`vars_stream` to read
        the corresponding timeseries.  A redundant common root (e.g. the
        ``_SpecComposite`` wrapper a loaded spec lives under) is stripped from
        the display paths.
        """
        return self._conn.call("var_tree", system_id=self.system_id)

    def get_state(self, vars=None) -> dict:
        """Latest value of each requested variable (suffix match), or all."""
        return self._conn.call("get_state", system_id=self.system_id, vars=vars)

    def get_record(self, vars=None, start=0, stop=None, stride=1) -> dict:
        """A slice of recorded history, row-major: ``{names, time, rows}``.

        ``rows[k]`` is the vector of the selected variables at ``time[k]``.
        """
        return self._conn.call(
            "get_record", system_id=self.system_id, vars=vars,
            start=start, stop=stop, stride=stride,
        )

    def get_series(self, vars=None, start=0, stop=None, stride=1) -> dict:
        """A slice of recorded history, column-major (one array per variable):
        ``{"time": [...], "series": {name: [...]}}``.

        This is the convenient shape for plotting a chosen variable / variable
        list over time -- request the variables a user picked from
        :meth:`var_tree` (full or suffix names both match).
        """
        return self._conn.call(
            "get_series", system_id=self.system_id, vars=vars,
            start=start, stop=stop, stride=stride,
        )

    def vars_stream(self, expand=True) -> Stream:
        """Open an on-demand variable stream and return a :class:`Stream`.

        Can be called any time after the system is instantiated -- before,
        during, or after a :meth:`run` -- and several independent streams may be
        open at once, so a UI can chart any data at any moment.  The stream
        starts watching nothing; choose variables on demand via
        :meth:`Stream.series` / :meth:`Stream.series_values` (or
        :meth:`Stream.watch`), read them through their live handles
        (``handle.array`` after :meth:`Stream.update`),
        and discover what is watched with :meth:`Stream.list_watched_names`.

        The stream always backfills the full recorded history first (so a chart
        is complete the moment a variable is requested), then streams new rows.

        Parameters
        ----------
        expand : bool
            When ``True`` (default), each requested name expands to *every*
            matching recorded variable (keyed by full name), so
            :meth:`Stream.series_values` can aggregate per-instance matches of a
            suffix -- e.g. one column per pipe segment for ``"m_dot_a_leak"``.
            ``False`` keeps only the first match of each name.
        """
        res = self._conn.call(
            "vars_stream", system_id=self.system_id, vars=[],
            scope="all", every=1, expand=expand,
        )
        return Stream(self._conn, self.system_id, res["stream_id"],
                      res.get("vars", []))

    # --- event stream -----------------------------------------------------

    def events(self, timeout=None) -> Iterator[dict]:
        """Yield streamed events (``result``/``status``/``log``/``error``)
        until the run finishes (``done``) or the connection drops."""
        yield from self._conn.iter_events(self.system_id, timeout=timeout)

    def poll_events(self) -> list:
        """Non-blocking drain of any events queued so far (e.g. logs emitted
        during ``instantiate``)."""
        return self._conn.poll_events(self.system_id)


class HostService:
    """A live connection to a hydrogen host process (one ``mpirun`` job).

    Hands out :class:`SystemProxy` objects via :meth:`load_json` /
    :meth:`load_dict`; owns the process lifecycle via :meth:`shutdown`.
    Usable as a context manager.
    """

    def __init__(self, proc: subprocess.Popen, sock: socket.socket, *, workers=1):
        self._proc = proc
        self._conn = _Conn(proc, sock)
        self._systems: dict[str, SystemProxy] = {}
        self.workers = workers

    def __repr__(self):
        return f"<HostService workers={self.workers} systems={list(self._systems)}>"

    def load_json(self, text: str) -> SystemProxy:
        """Build a model on the host from a JSON spec string."""
        sid = self._conn.call("load_json", spec=text)
        return self._register(sid)

    def load_dict(self, spec: dict) -> SystemProxy:
        """Build a model on the host from a spec dict."""
        sid = self._conn.call("load_dict", spec=spec)
        return self._register(sid)

    def _register(self, sid) -> SystemProxy:
        proxy = SystemProxy(self._conn, sid)
        self._systems[sid] = proxy
        return proxy

    def list_systems(self) -> list:
        """Ask the host for the id + phase of every loaded system."""
        return self._conn.call("list_systems")

    def shutdown(self, timeout=10.0):
        """Tell the host to exit, close the socket, and reap the process."""
        try:
            self._conn.call("shutdown", timeout=timeout)
        except (HostError, HostConnectionError, TimeoutError):
            pass
        finally:
            self._conn.close()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def start_host(
    workers: int = 1,
    *,
    launcher: str = "mpirun",
    python: str | None = None,
    host: str = "127.0.0.1",
    connect_timeout: float = 60.0,
    env: dict | None = None,
    launcher_args: list | None = None,
    quiet: bool = True,
) -> HostService:
    """Start a hydrogen host process and return a connected :class:`HostService`.

    ``workers == 1`` runs the host directly (no MPI dependency required).
    ``workers > 1`` launches it under ``mpirun -n <workers>`` (override the
    program with ``launcher`` / add flags with ``launcher_args``); rank 0 owns
    the socket and the other ranks follow it.

    The host binds an ephemeral localhost port that we pick here and pass in,
    then we poll-connect until it is accepting (or the process dies / we time
    out).

    ``quiet`` (default) sends the host's own stdout to ``/dev/null`` -- its
    ``print`` diagnostics still reach the client as ``log`` events, so this just
    avoids them being echoed twice.  Set ``quiet=False`` to also see the host
    console directly.  Host stderr is always inherited so tracebacks are visible
    if it crashes during startup.
    """
    python = python or sys.executable

    # Reserve a free localhost port, then release it for the host to bind.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind((host, 0))
    port = probe.getsockname()[1]
    probe.close()

    base = [python, "-m", "hydrogen.service", "--host", host, "--port", str(port)]
    if workers > 1:
        cmd = [launcher, "-n", str(workers), *(launcher_args or []), *base]
    else:
        cmd = base

    stdout = subprocess.DEVNULL if quiet else None
    proc = subprocess.Popen(cmd, env=env, stdout=stdout)

    deadline = time.time() + connect_timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            break
        except OSError:
            if proc.poll() is not None:
                raise HostConnectionError(
                    f"hydrogen host exited (code {proc.returncode}) before "
                    f"accepting a connection; command was: {' '.join(cmd)}"
                )
            if time.time() > deadline:
                proc.kill()
                raise TimeoutError(
                    f"hydrogen host did not start within {connect_timeout}s"
                )
            time.sleep(0.1)

    # `create_connection` leaves its connect timeout (2s) on the socket. Clear
    # it so the long-lived connection blocks indefinitely: the background reader
    # sits in recv() between host messages, and a finite timeout there would
    # surface as a spurious EOF (and a dropped connection) during any idle gap
    # longer than the timeout -- e.g. a slow `plt.subplots()` over forwarded
    # X11. The timeout was only ever meant to bound the connect attempt above.
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return HostService(proc, sock, workers=workers)
