"""Setting compile-time Parameters over the service (`SystemProxy.set_param`).

Spawns a real host (``workers=1``) running a tiny CoolProp-free, fully algebraic
control system -- a `Constant` feeding a `Gain` so that ``gain.y == const.k *
gain.k`` exactly on every step -- and checks:

* ``list_params`` exposes the loaded Parameters by full name;
* ``set_param`` / ``set_params`` resolve by dotted-suffix and take effect on the
  next solve (the algebraic output is a deterministic readout of the live values);
* a ``set_param`` issued *mid-run* is honoured at the next step boundary;
* an unknown name is surfaced as a structured ``HostError``.
"""

from __future__ import annotations

import time

import hydrogen

# Constant("src") -> Gain("amp"):  amp.y == src.k * amp.k  (pure algebra).
_SPEC = {
    "hydrogen_version": "0.1.0",
    "schema_version": 1,
    "components": {
        "src": {"type": "hydrogen.control.Constant", "params": {"k": 2.0}},
        "amp": {"type": "hydrogen.control.Gain", "params": {"k": 3.0}},
    },
    "connections": [{"from": "src.y", "to": "amp.u"}],
}


def _build(service):
    system = service.load_dict(_SPEC)
    system.instantiate(max_remove_trival_passes=2)
    system.poll_events()  # drop compile logs
    system.initialise(n=1)
    return system


def test_list_and_set_params_take_effect():
    service = hydrogen.start_host(workers=1)
    try:
        system = _build(service)

        names = system.list_params()
        for suffix in ("src.k", "amp.k"):
            assert any(n.endswith("." + suffix) or n.endswith(suffix)
                       for n in names), f"{suffix!r} missing from {names}"

        # Baseline: 2 * 3 == 6.
        system.step(dt=0.1)
        assert abs(system.get_state(["amp.y"])["amp.y"] - 6.0) < 1e-9

        # Bulk update both gains: 5 * 4 == 20. The reply echoes full names.
        applied = system.set_params({"src.k": 5.0, "amp.k": 4.0})
        assert any(k.endswith("src.k") for k in applied)
        assert any(k.endswith("amp.k") for k in applied)
        system.step(dt=0.1)
        assert abs(system.get_state(["amp.y"])["amp.y"] - 20.0) < 1e-9

        # Single-name update reflected on the very next step: 5 * 10 == 50.
        system.set_param("amp.k", 10.0)
        system.step(dt=0.1)
        assert abs(system.get_state(["amp.y"])["amp.y"] - 50.0) < 1e-9

        # Unknown name -> structured error, not a silent no-op.
        try:
            system.set_param("does.not.exist", 1.0)
        except hydrogen.service.HostError as exc:  # type: ignore[attr-defined]
            assert exc.kind == "ValueError"
        else:
            raise AssertionError("expected HostError for unknown parameter")
    finally:
        service.shutdown()


def test_set_param_mid_run_is_honoured():
    service = hydrogen.start_host(workers=1)
    try:
        system = _build(service)

        # Slow-motion run so we can nudge a parameter mid-flight. During a run
        # only control/stream commands are accepted (no get_state), so we watch
        # the live output through a variable stream.
        system.run(dt=0.02, steps=400, stream=True, every=5, delay=0.02)
        stream = system.vars_stream()
        amp = stream.series("amp.y")   # live handle, backfilled + streamed
        time.sleep(0.2)

        # src.k * amp.k -> 7 * 3 == 21 once the new value is picked up.
        system.set_param("src.k", 7.0)

        deadline = time.monotonic() + 6.0
        seen = False
        while time.monotonic() < deadline:
            stream.update()
            if any(abs(float(v) - 21.0) < 1e-9 for v in amp.array):
                seen = True
                break
            time.sleep(0.05)
        stream.close()
        system.stop()
        assert seen, "live output never reflected the mid-run set_param"
    finally:
        service.shutdown()
