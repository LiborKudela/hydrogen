"""A long-lived simulation session that keeps a hydrogen host + an instantiated
model alive across runs, re-instantiating only when the system's *structure*
changes.

``instantiate`` (compiling the DAE) is the expensive step.  Once a model is
built it is kept on the host; subsequent runs reuse it.  A *pure* numeric
parameter can be pushed into the live model with ``set_param`` (no rebuild),
while a *structural* change -- topology (connections), media, or a parameter
flagged ``structural`` / typed object|list -- forces a fresh ``load_json`` +
``instantiate``.

:func:`structural_param_names` is the single classifier the UI shares: it drives
both the rebuild decision here and the green/red colouring in the properties
editor.
"""

from __future__ import annotations

import json
import threading
import time
from functools import lru_cache
from typing import Callable

import hydrogen as hd

__all__ = ["structural_param_names", "SimulationSession"]

#: A log sink: ``log(message, level)`` where level is one of
#: ``status`` / ``host`` / ``warning`` / ``error`` (mirrors :class:`LogPanel`).
LogFn = Callable[[str, str], None]


@lru_cache(maxsize=None)
def structural_param_names(type_name: str) -> frozenset[str]:
    """Names of the parameters whose value changes a component's *structure*.

    Sourced from the catalog's ``literals`` (constructor args flagged
    ``structural=True``, which change the emitted equation set) plus any
    object/list-valued parameter (these swap whole sub-models / topology).  Every
    other parameter is "pure" -- a numeric coefficient that can be updated on a
    live model without recompiling.
    """
    try:
        spec = hd.component_spec(type_name)
    except Exception:
        return frozenset()
    names = {lit["name"] for lit in spec.get("literals", [])}
    for p in spec.get("parameters", []):
        if p.get("type") in ("object", "list"):
            names.add(p["name"])
    return frozenset(names)


class SimulationSession:
    """Owns the host connection + the currently-built model.

    Call :meth:`ensure_built` before :meth:`run`; it decides between reusing the
    live model (optionally pushing changed pure parameters) and a full rebuild.
    The session outlives any single dialog so the model stays warm between runs;
    :meth:`shutdown` tears the host down (e.g. on app exit).
    """

    def __init__(self):
        self._service = None             # hd.HostService
        self._sysp = None                # hd.SystemProxy (the live model)
        # The proxy of a build that is currently compiling on the host.  It is
        # exposed (separately from `_sysp`, which is only set once the build
        # commits) so `poll_logs` can stream the host's `instantiate` progress
        # live while the worker thread is blocked in the call.
        self._inflight = None
        self._sig: str | None = None     # structural signature of the built model
        self._params: dict = {}          # {comp_id: {pure_name: value}} as built
        self._inst_kw: dict | None = None
        # Run lifecycle for a *streaming* run (the run keeps advancing on the
        # host after `start_run` returns; the phase is refreshed from the host's
        # status/done events by `poll_logs`).  ``None`` means no run has been
        # launched on the current model; otherwise one of the host phases
        # ("running" / "paused" / "finished" / "stopped" / "error").
        self._run_phase: str | None = None
        self._stopping = False            # a stop was requested; label the finish
        # Latest run status (from the host's throttled status/done events), for
        # the table's step / progress rows.
        self._run_step = 0
        self._run_total = 0
        self._run_t = 0.0
        self._run_stop_time: float | None = None
        # Wall-clock timing of the run: `_run_wall_t0` is the monotonic start
        # (None when no run has begun); `_run_wall_elapsed` freezes the final
        # elapsed the host reports on `done`.
        self._run_wall_t0: float | None = None
        self._run_wall_elapsed = 0.0
        self._continuable = False
        # Snapshot of the canvas model when the current run started / was paused;
        # resume/continue is only offered while this still matches.
        self._run_checkpoint: dict | None = None
        self._run_stale = False
        # Guards state transitions so a background build and a GUI-thread abort
        # don't interleave; `_gen` bumps on every abort/reset so a build that
        # finishes *after* it was aborted discards its (now-dead) result.
        self._lock = threading.RLock()
        self._gen = 0
        # Per-component modelling warnings raised by the host during build/run
        # (``{comp_id: [message, ...]}``), plus a monotonic version so a poller
        # can cheaply tell when the set changed.  Populated from host WARNING
        # log events (see :meth:`_forward_events`); read by the GUI to badge the
        # offending canvas nodes.
        self._warn_lock = threading.Lock()
        self._component_warnings: dict[str, list[str]] = {}
        self._warnings_version = 0

    # --- warnings ---------------------------------------------------------- #
    def component_warnings(self) -> dict[str, list[str]]:
        """Snapshot of ``{comp_id: [message, ...]}`` for components that raised
        a modelling warning on the host."""
        with self._warn_lock:
            return {k: list(v) for k, v in self._component_warnings.items()}

    @property
    def warnings_version(self) -> int:
        """Bumped whenever :meth:`component_warnings` would change (add / clear)
        -- lets a poller detect updates without diffing the whole map."""
        with self._warn_lock:
            return self._warnings_version

    def clear_warnings(self):
        """Drop all accumulated component warnings (e.g. on a fresh build)."""
        with self._warn_lock:
            if self._component_warnings:
                self._component_warnings = {}
                self._warnings_version += 1

    def _record_warning(self, comp_id: str, message: str):
        with self._warn_lock:
            msgs = self._component_warnings.setdefault(comp_id, [])
            if message not in msgs:            # collapse identical repeats
                msgs.append(message)
                self._warnings_version += 1

    # --- state ------------------------------------------------------------- #
    @property
    def built(self) -> bool:
        """A model is instantiated and kept alive on the host."""
        return self._sysp is not None

    @property
    def sysp(self):
        return self._sysp

    @property
    def run_phase(self) -> str | None:
        """Host phase of the streaming run on the current model, or ``None``."""
        return self._run_phase

    @property
    def run_active(self) -> bool:
        """A streaming run is advancing or parked (i.e. controllable)."""
        return self._run_phase in ("running", "paused")

    @property
    def can_continue(self) -> bool:
        """The built model can integrate further without re-initialising."""
        return (self._continuable and not self._run_stale
                and self._run_checkpoint is not None
                and self._sysp is not None
                and self._run_phase in ("finished", "stopped"))

    def set_run_checkpoint(self, system: dict, inst_kw: dict):
        """Remember the model definition the current run belongs to."""
        self._run_checkpoint = {
            "structural": self._structural_signature(system, inst_kw),
            "params": self._pure_params(system),
        }
        self._run_stale = False

    def is_checkpoint_current(self, system: dict, inst_kw: dict) -> bool:
        if self._run_stale or self._run_checkpoint is None:
            return False
        if self._structural_signature(system, inst_kw) != self._run_checkpoint["structural"]:
            return False
        return self._pure_params(system) == self._run_checkpoint["params"]

    def can_steering_resume(self, system: dict, inst_kw: dict) -> bool:
        """True when pause/resume or continue is valid for this canvas state."""
        if not self.is_checkpoint_current(system, inst_kw):
            return False
        if self._run_phase == "paused":
            return True
        return self.can_continue

    def mark_model_stale(self):
        """The canvas model no longer matches the run checkpoint."""
        if self._run_checkpoint is None and not self._run_stale:
            return
        self._run_stale = True
        self._continuable = False
        self._run_checkpoint = None

    @property
    def run_step(self) -> int:
        """Latest solver step index of the current run (0 before/at start)."""
        return self._run_step

    @property
    def run_progress(self) -> float | None:
        """Run completion as a fraction 0..1, or ``None`` if it can't be
        estimated.  Uses the step count when the run is step-bounded, otherwise
        the sim time against the target ``stop_time``."""
        if self._run_total:
            return max(0.0, min(1.0, self._run_step / self._run_total))
        if self._run_stop_time and self._run_stop_time > 0:
            return max(0.0, min(1.0, self._run_t / self._run_stop_time))
        return None

    @property
    def run_wall_time(self) -> float | None:
        """Wall-clock seconds the current run has taken -- counting live while
        it advances, frozen at the host-reported total once it ends.  ``None``
        before any run."""
        if self._run_wall_t0 is None:
            return None
        if self.run_active:
            return time.monotonic() - self._run_wall_t0
        return self._run_wall_elapsed

    # --- structural classification ----------------------------------------- #
    @staticmethod
    def _structural_signature(system: dict, inst_kw: dict) -> str:
        """A stable fingerprint of everything that, if changed, needs a rebuild:
        component types, media, structural parameter values, the wiring, and the
        instantiate options.  Pure parameter values are deliberately excluded."""
        comps = {}
        for cid, tmpl in system.get("components", {}).items():
            type_name = tmpl.get("type")
            sparams = structural_param_names(type_name)
            params = tmpl.get("params") or {}
            comps[cid] = {
                "type": type_name,
                "medium": tmpl.get("medium"),
                "structural": {k: params.get(k) for k in sorted(sparams)},
            }
        conns = sorted(
            (c.get("from"), c.get("to")) for c in system.get("connections", []))
        payload = {
            "components": comps,
            "connections": conns,
            "media": system.get("media", {}),
            "instantiate": inst_kw,
        }
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _pure_params(system: dict) -> dict:
        """``{comp_id: {pure_param_name: value}}`` -- the live-updatable knobs."""
        out: dict = {}
        for cid, tmpl in system.get("components", {}).items():
            sparams = structural_param_names(tmpl.get("type"))
            params = tmpl.get("params") or {}
            out[cid] = {k: v for k, v in params.items() if k not in sparams}
        return out

    def _changed_pure_params(self, system: dict) -> dict:
        """``{f"{comp_id}.{name}": value}`` for pure params that differ from the
        values the live model was built / last synced with."""
        new = self._pure_params(system)
        changed: dict = {}
        for cid, params in new.items():
            old = self._params.get(cid, {})
            for name, val in params.items():
                if old.get(name) != val:
                    changed[f"{cid}.{name}"] = val
        return changed

    # --- build / reuse ----------------------------------------------------- #
    def ensure_built(self, system: dict, inst_kw: dict, log: LogFn) -> str:
        """Make the live model match ``system``.

        Returns ``"built"`` if it (re)instantiated, or ``"reused"`` if it kept
        the existing model (pushing any changed pure parameters live).
        """
        sig = self._structural_signature(system, inst_kw)
        if self._sysp is None:
            self._build(system, inst_kw, sig, log)
            return "built"
        if self._run_stale:
            log("model changed since last run — re-instantiating model.",
                "status")
            self._build(system, inst_kw, sig, log)
            return "built"
        if sig != self._sig:
            log("structure changed — re-instantiating model.", "status")
            self._build(system, inst_kw, sig, log)
            return "built"

        changed = self._changed_pure_params(system)
        if not changed:
            log("no changes — reusing the live model.", "status")
            return "reused"
        if self._try_set_params(changed, log):
            self._params = self._pure_params(system)
            return "reused"
        log("a changed parameter isn't live-settable — re-instantiating.",
            "status")
        self._build(system, inst_kw, sig, log)
        return "built"

    def force_build(self, system: dict, inst_kw: dict, log: LogFn) -> str:
        """Unconditionally (re)instantiate the model, even if its structure is
        unchanged.

        Bypasses the reuse / live-parameter path :meth:`ensure_built` takes --
        use it to force a fresh compile (e.g. to pick up an external change the
        structural signature can't see).
        """
        sig = self._structural_signature(system, inst_kw)
        log("forced rebuild — re-instantiating model.", "status")
        self._build(system, inst_kw, sig, log)
        return "rebuilt"

    def initialise(self, init_kw: dict, log: LogFn) -> str:
        """Re-solve the built model to a Newton-consistent state at t=0, without
        launching a run.

        Use it to force a fresh initialise (e.g. after a live parameter change)
        so the next run starts from a clean t=0 state.
        """
        if self._sysp is None:
            raise RuntimeError("no model has been built")
        log(f"initialising with {init_kw} …", "status")
        self._sysp.initialise(**init_kw)
        self._drain(log)
        # State is back at t=0 and no run is in flight anymore.
        self._run_phase = None
        self._run_step = 0
        self._run_t = 0.0
        log("model initialised to t=0.", "status")
        return "initialised"

    def _try_set_params(self, changed: dict, log: LogFn) -> bool:
        """Push changed pure params onto the live model. Only numeric values are
        live-settable (the host writes them straight into the solver buffer); a
        non-numeric or unmatched change returns ``False`` to request a rebuild."""
        numeric: dict = {}
        for name, val in changed.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                log(f"  {name} = {val!r} is not a live-settable number.",
                    "status")
                return False
            numeric[name] = float(val)
        try:
            applied = self._sysp.set_params(numeric)
        except Exception as exc:  # unknown name etc. -> fall back to rebuild
            log(f"  live update failed: {type(exc).__name__}: {exc}", "warning")
            return False
        self._drain(log)
        log(f"  updated {len(applied)} parameter(s) live: {applied}", "status")
        return True

    def _build(self, system: dict, inst_kw: dict, sig: str, log: LogFn):
        # A (re)build produces a fresh set of warnings; drop the previous
        # model's so stale badges don't linger on the canvas.
        self.clear_warnings()
        # Snapshot the generation + service under the lock, then run the slow,
        # blocking host calls WITHOUT holding it -- otherwise abort() couldn't
        # interrupt us (it needs the lock to tear the host down).
        with self._lock:
            gen = self._gen
            self._ensure_service(log)
            if self._sysp is not None:
                try:
                    self._sysp.close()
                except Exception:
                    pass
                self._sysp = None
            service = self._service

        text = json.dumps(system)
        sysp = service.load_json(text)
        # Publish the in-flight proxy so the UI's log pump can stream the host's
        # `instantiate` diagnostics live while we block in the call below.
        with self._lock:
            self._inflight = sysp
        try:
            log(f"instantiating model with {inst_kw} … (compiling DAE)", "status")
            sysp.instantiate(**inst_kw)    # blocking; abort() kills the host -> raises

            # Commit only if we weren't aborted/reset while compiling; otherwise
            # the host is gone and this proxy is dead, so discard it.
            with self._lock:
                if self._gen != gen:
                    try:
                        sysp.close()
                    except Exception:
                        pass
                    raise RuntimeError("build cancelled")
                self._sysp = sysp
                self._sig = sig
                self._inst_kw = dict(inst_kw)
                self._params = self._pure_params(system)
        finally:
            with self._lock:
                self._inflight = None
        self._drain(log)
        log("model built and kept alive on the host.", "status")

    def _ensure_service(self, log: LogFn):
        if self._service is None:
            log("starting hydrogen host …", "status")
            self._service = hd.start_host(workers=1)

    # --- run --------------------------------------------------------------- #
    def start_run(self, init_kw: dict, run_kw: dict, log: LogFn):
        """Initialise to t=0 and launch a *streaming* run, returning as soon as
        the host acknowledges.

        The run then advances asynchronously on the host: it keeps going even if
        every UI window is closed, and is steerable via :meth:`pause_run` /
        :meth:`resume_run` / :meth:`stop_run`.  Progress + completion are
        surfaced through the host's status/done events (drained by
        :meth:`poll_logs`, which also refreshes :attr:`run_phase`).  Live values
        flow over the separate variable stream, unchanged.
        """
        if self._sysp is None:
            raise RuntimeError("no model has been built")
        log(f"initialising with {init_kw} …", "status")
        self._sysp.initialise(**init_kw)
        self._drain(log)
        # Force streaming so run() returns after the host ack instead of
        # blocking to completion; throttle status progress events (the live data
        # arrives over vars_stream, so status is only needed for phase/progress).
        stream_kw = dict(run_kw)
        stream_kw["stream"] = True
        stream_kw.setdefault("every", 20)
        self._stopping = False
        self._run_phase = "running"
        # Reset run-status tracking so step/progress restart for this run.
        self._run_step = 0
        self._run_t = 0.0
        self._run_total = int(run_kw.get("steps") or 0)
        st = run_kw.get("stop_time")
        self._run_stop_time = float(st) if st is not None else None
        self._run_wall_t0 = time.monotonic()
        self._run_wall_elapsed = 0.0
        log(f"starting run with {run_kw} … (streaming; keeps running in the "
            f"background)", "status")
        ack = self._sysp.run(**stream_kw)
        self._drain(log)
        return ack

    def _dispatch_run_cmd(self, name: str):
        """Send a run-control command (``pause`` / ``resume`` / ``stop``) off the
        GUI thread.

        The host honours it at the next step boundary and replies then, so the
        call can briefly block; running it on a throwaway daemon thread keeps the
        UI responsive.  The resulting phase change comes back via events.
        """
        sysp = self._sysp
        if sysp is None:
            return

        def go():
            try:
                getattr(sysp, name)()
            except Exception:
                pass

        threading.Thread(target=go, name=f"hydrogen-run-{name}",
                         daemon=True).start()

    def pause_run(self):
        if self.run_active:
            self._dispatch_run_cmd("pause")

    def resume_run(self):
        if self._run_phase == "paused" and not self._run_stale:
            self._dispatch_run_cmd("resume")

    def stop_and_drain(self, log: LogFn):
        """Cooperatively stop a running/paused stream and wait for confirmation."""
        if self._sysp is None:
            return
        if self._run_phase not in ("running", "paused"):
            return
        self._stopping = True
        try:
            self._sysp.stop()
        except Exception:
            pass
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self._drain(log)
            if self._run_phase in (None, "stopped", "finished", "error"):
                break
            time.sleep(0.05)
        self._stopping = False

    def continue_run(self, *, every: int = 20):
        """Continue integrating from the current model time (no re-init)."""
        sysp = self._sysp
        if sysp is None or not self.can_continue or self._run_stale:
            return
        self._stopping = False
        self._run_phase = "running"
        self._continuable = False
        self._run_wall_t0 = time.monotonic()

        def go():
            try:
                sysp.continue_run(stream=True, every=every)
            except Exception:
                pass

        threading.Thread(target=go, name="hydrogen-run-continue",
                         daemon=True).start()

    def push_run_config(self, config: dict, log: LogFn | None = None):
        """Push live simulate knobs to the host (pause / finished / stopped)."""
        if self._sysp is None or not config:
            return
        try:
            applied = self._sysp.update_run_config(**config)
        except Exception as exc:
            if log is not None:
                log(f"update_run_config failed: {type(exc).__name__}: {exc}",
                    "error")
            return
        if "continuable" in applied:
            self._continuable = bool(applied["continuable"])
        if applied.get("stop_time") is not None:
            self._run_stop_time = float(applied["stop_time"])
        if log is not None:
            bits = []
            if "stop_time" in applied:
                bits.append(f"stop_time={applied['stop_time']}")
            rc = applied.get("run_control") or {}
            if "dt_max" in rc:
                bits.append(f"dt_max={rc['dt_max']}")
            if bits:
                log("Live run config: " + ", ".join(bits), "status")

    def stop_run(self):
        if self.run_active:
            self._stopping = True
            self._dispatch_run_cmd("stop")

    # --- teardown ---------------------------------------------------------- #
    def reset(self):
        """Drop the built model (e.g. on new/open project) but keep the host up
        so the next build doesn't pay the host start-up cost again."""
        with self._lock:
            self._gen += 1            # invalidate any in-flight build's commit
            if self._sysp is not None:
                try:
                    self._sysp.close()
                except Exception:
                    pass
                self._sysp = None
            self._inflight = None
            self._sig = None
            self._params = {}
            self._inst_kw = None
            self._run_phase = None
            self._stopping = False
            self._run_step = 0
            self._run_total = 0
            self._run_t = 0.0
            self._run_stop_time = None
            self._run_wall_t0 = None
            self._run_wall_elapsed = 0.0
            self._continuable = False
            self._run_checkpoint = None
            self._run_stale = False
        self.clear_warnings()

    def shutdown(self):
        """Close the model and stop the host process."""
        self.reset()
        if self._service is not None:
            try:
                self._service.shutdown()
            except Exception:
                pass
            self._service = None

    def abort(self):
        """Forcibly cancel an in-flight host operation.

        ``instantiate`` / ``initialise`` aren't cooperatively interruptible (the
        host is single-threaded and busy compiling), so we tear the host process
        down: any blocked client call returns with a connection error -- the
        caller should treat that as a *cancellation*, not a failure -- and a
        fresh host is started on the next build.
        """
        with self._lock:
            self._gen += 1            # any in-flight build will discard its result
            svc = self._service
            self._service = None
            self._sysp = None
            self._inflight = None
            self._sig = None
            self._params = {}
            self._inst_kw = None
            self._run_phase = None
            self._stopping = False
            self._run_step = 0
            self._run_total = 0
            self._run_t = 0.0
            self._run_stop_time = None
            self._run_wall_t0 = None
            self._run_wall_elapsed = 0.0
            self._continuable = False
            self._run_checkpoint = None
            self._run_stale = False
        self.clear_warnings()
        if svc is not None:
            try:
                svc.terminate()       # kill outside the lock (it joins the process)
            except Exception:
                pass

    # --- events ------------------------------------------------------------ #
    def _forward_events(self, sysp, log: LogFn):
        """Drain a proxy's queued host events to the log sink, tagged by level,
        and fold streaming-run ``status`` / ``done`` events into
        :attr:`run_phase` as a side effect (so the phase stays fresh no matter
        which poller drains the queue)."""
        for ev in sysp.poll_events():
            etype = ev.get("type")
            if etype == "log":
                message = ev.get("message", "")
                level = ev.get("level") or ""
                is_warning = ("warn" in level.lower()
                              or "warning" in message.lower())
                if is_warning:
                    comp = ev.get("component")
                    where = ev.get("where")
                    full = message + (f" ({where})" if where else "")
                    tag = f" [{comp}]" if comp else ""
                    log(f"  [warning{tag}] {full}", "warning")
                    if comp:
                        self._record_warning(comp, full)
                else:
                    log(f"  [host] {message}", "host")
            elif etype == "error":
                log(f"  [host error] {ev.get('kind')}: {ev.get('message')}",
                    "error")
                self._run_phase = "error"
            elif etype == "status":
                phase = ev.get("phase")
                if phase:
                    self._run_phase = phase
                if "continuable" in ev:
                    self._continuable = bool(ev["continuable"])
                if ev.get("stop_time") is not None:
                    self._run_stop_time = float(ev["stop_time"])
                self._capture_run_stats(ev)
            elif etype == "done":
                self._capture_run_stats(ev)
                # Prefer the phase the host reports; fall back to inferring a
                # user-requested stop vs. a natural finish.
                phase = ev.get("phase") or (
                    "stopped" if self._stopping else "finished")
                self._run_phase = phase
                if "continuable" in ev:
                    self._continuable = bool(ev["continuable"])
                if ev.get("stop_time") is not None:
                    self._run_stop_time = float(ev["stop_time"])
                self._stopping = False
                self._log_run_summary(ev, phase, log)

    def _capture_run_stats(self, ev: dict):
        """Fold a host status/done event's step / total / time into the tracked
        run status (drives the table's step + progress rows)."""
        step = ev.get("step")
        if isinstance(step, int):
            self._run_step = step
        total = ev.get("total")
        if isinstance(total, int) and total:
            self._run_total = total
        t = ev.get("t")
        if isinstance(t, (int, float)):
            self._run_t = float(t)
        elapsed = ev.get("elapsed")       # only on `done` -> freeze wall time
        if isinstance(elapsed, (int, float)):
            self._run_wall_elapsed = float(elapsed)
        elif self._run_wall_t0 is not None:
            self._run_wall_elapsed = time.monotonic() - self._run_wall_t0

    def _log_run_summary(self, ev: dict, phase: str, log: LogFn):
        """Emit an end-of-run summary block (steps, sim/wall time, last solve)
        from the host's ``done`` event fields (all optional for old hosts)."""
        steps = ev.get("step")
        detail: list[str] = []
        if steps is not None:
            total = ev.get("total") or 0
            detail.append(f"  steps taken : {steps}"
                          + (f" / {total}" if total else ""))
        t_final = ev.get("t")
        if isinstance(t_final, (int, float)):
            detail.append(f"  sim time    : {t_final:.6g} s")
        elapsed = ev.get("elapsed")
        if isinstance(elapsed, (int, float)):
            rate = (f", {steps / elapsed:.1f} steps/s"
                    if steps and elapsed > 0 else "")
            detail.append(f"  wall time   : {elapsed:.3g} s{rate}")
        iters = ev.get("last_iters")
        resid = ev.get("last_residual")
        bits = []
        if iters:
            bits.append(f"{iters} Newton iters")
        if isinstance(resid, float) and resid == resid:   # not NaN
            bits.append(f"residual {resid:.2e}")
        if bits:
            detail.append("  last solve  : " + ", ".join(bits))

        if detail:
            log(f"\nrun {phase} — summary:", "status")
            for line in detail:
                log(line, "status")
        else:
            log(f"\nrun {phase}.", "status")

    def _drain(self, log: LogFn):
        """Forward queued host events for the committed model (if any)."""
        if self._sysp is None:
            return
        self._forward_events(self._sysp, log)

    def poll_logs(self, log: LogFn):
        """Stream any host events queued so far for the active model -- the one
        being built *or* the committed one.

        Safe (and intended) to call repeatedly from the GUI thread while a
        worker thread is blocked in a host call: the host emits its ``print``
        diagnostics line-by-line over the wire, so polling here surfaces
        ``instantiate`` / ``run`` progress *as it happens* instead of in one
        burst when the blocking call finally returns.
        """
        with self._lock:
            sysp = self._sysp or self._inflight
        if sysp is None:
            return
        try:
            self._forward_events(sysp, log)
        except Exception:
            # A teardown (abort) can close the proxy mid-poll; ignore.
            pass
