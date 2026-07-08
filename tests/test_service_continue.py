"""Continue a streaming run after extending stop_time post-finish."""

from __future__ import annotations

import time
from pathlib import Path

import hydrogen

_SPEC = (
    Path(__file__).resolve().parent.parent
    / "tutorials" / "host_client" / "signal_dynamics.json"
).read_text()


def _wait_done(system, timeout=15.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        for ev in system.poll_events():
            last = ev
            if ev.get("type") == "done":
                return ev
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for done (last={last!r})")


def test_extend_stop_time_then_continue():
    service = hydrogen.start_host(workers=1)
    try:
        system = service.load_json(_SPEC)
        system.instantiate(max_remove_trival_passes=2)
        system.poll_events()
        system.initialise(n=1)

        stop0 = 0.4
        system.run(stop_time=stop0, strategy="richardson",
                   adaptive={"dt_max": 0.05, "dt_start": 0.01},
                   stream=True, every=1)
        done = _wait_done(system)
        assert done.get("phase") == "finished"
        assert done.get("continuable") is False
        t_done = float(done["t"])
        assert t_done >= stop0 - 1e-6

        applied = system.update_run_config(stop_time=0.9)
        assert applied["continuable"] is True
        assert applied["stop_time"] == 0.9

        system.continue_run(stream=True, every=1)
        done2 = _wait_done(system, timeout=20.0)
        assert done2.get("phase") == "finished"
        assert float(done2["t"]) >= 0.9 - 1e-5
        assert float(done2["t"]) > t_done

        system.close()
    finally:
        service.shutdown()


def test_set_dt_max_while_paused():
    service = hydrogen.start_host(workers=1)
    try:
        system = service.load_json(_SPEC)
        system.instantiate(max_remove_trival_passes=2)
        system.poll_events()
        system.initialise(n=1)

        system.run(stop_time=2.0, strategy="richardson",
                   adaptive={"dt_max": 0.02, "dt_start": 0.01},
                   stream=True, every=1, delay=0.05)
        time.sleep(0.25)
        system.pause()
        paused = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not paused:
            for ev in system.poll_events():
                if ev.get("type") == "status" and ev.get("phase") == "paused":
                    paused = True
            time.sleep(0.02)
        assert paused

        applied = system.update_run_config(adaptive={"dt_max": 0.2})
        rc = applied.get("run_control") or {}
        assert rc.get("dt_max") == 0.2

        system.resume()
        _wait_done(system, timeout=30.0)
        system.close()
    finally:
        service.shutdown()
