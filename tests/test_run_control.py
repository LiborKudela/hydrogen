"""Unit tests for the live Model run controller."""

from hydrogen.serialization import from_json
from pathlib import Path

_SPEC = (
    Path(__file__).resolve().parent.parent
    / "tutorials" / "host_client" / "signal_dynamics.json"
).read_text()


def test_update_dt_max_clamps_hint():
    m = from_json(_SPEC)
    m.instantiate(max_remove_trival_passes=2)
    m.initialise(n=1)
    gen = m.iter_run(stop_time=10.0, strategy="richardson",
                     dt_max=0.1, dt_start=0.01)
    next(gen)  # seed _run_ctrl
    m._dt_hint = 0.5
    m.update_run_control(dt_max=0.05)
    assert m._dt_hint == 0.05
    m.update_run_control(dt_max=1.0)
    assert m._dt_hint == 0.05  # unchanged until controller grows
    next(gen)
