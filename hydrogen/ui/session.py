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
        # Guards state transitions so a background build and a GUI-thread abort
        # don't interleave; `_gen` bumps on every abort/reset so a build that
        # finishes *after* it was aborted discards its (now-dead) result.
        self._lock = threading.RLock()
        self._gen = 0

    # --- state ------------------------------------------------------------- #
    @property
    def built(self) -> bool:
        """A model is instantiated and kept alive on the host."""
        return self._sysp is not None

    @property
    def sysp(self):
        return self._sysp

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
    def run(self, init_kw: dict, run_kw: dict, log: LogFn):
        """Initialise to t=0 (with current parameters) and run; returns the host
        summary.  Assumes :meth:`ensure_built` has just been called."""
        if self._sysp is None:
            raise RuntimeError("no model has been built")
        log(f"initialising with {init_kw} …", "status")
        self._sysp.initialise(**init_kw)
        self._drain(log)
        log(f"running with {run_kw} …", "status")
        summary = self._sysp.run(**run_kw)
        self._drain(log)
        return summary

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
        if svc is not None:
            try:
                svc.terminate()       # kill outside the lock (it joins the process)
            except Exception:
                pass

    # --- events ------------------------------------------------------------ #
    @staticmethod
    def _forward_events(sysp, log: LogFn):
        """Drain a proxy's queued host events (logs / warnings / errors) to the
        log sink, tagged by level."""
        for ev in sysp.poll_events():
            if ev.get("type") == "log":
                message = ev.get("message", "")
                level = ev.get("level") or ""
                if "warn" in level.lower() or "warning" in message.lower():
                    log(f"  [host warning] {message}", "warning")
                else:
                    log(f"  [host] {message}", "host")
            elif ev.get("type") == "error":
                log(f"  [host error] {ev.get('kind')}: {ev.get('message')}",
                    "error")

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
