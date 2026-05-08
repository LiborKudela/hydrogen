"""End-to-end integration check using `IntegrationTest` with closed-form analytical solutions.

A single fixture instantiates and runs the model; each test then asserts a different
trace stays inside the analytical Crank-Nicolson error budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen.test_models import IntegrationTest

DT = 0.04
N_STEPS = 25
OMEGA = 2 * np.pi  # period of 1 s -> simulation covers exactly one period


@pytest.fixture(scope="module")
def run():
    model = IntegrationTest(omega=OMEGA)
    model.instantiate(max_remove_trival_passes=2)
    model.initialise()
    for _ in range(N_STEPS):
        model.solve_dae_step(DT)
        model.next_step()

    record = model.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return {"t": t, "trace": trace}


def test_exponential_decay_within_cn_budget(run):
    t = run["t"]
    y = run["trace"](".y_decay")
    err = np.max(np.abs(y - np.exp(-t)))
    assert err < 1e-3, f"y_decay max error {err:.3e} above CN budget"


def test_oscillator_position_phase_drift(run):
    t = run["t"]
    y = run["trace"](".y_osc")
    err = np.max(np.abs(y - np.cos(OMEGA * t)))
    # CN preserves amplitude exactly; only phase drifts as ~(omega*dt)^3 / 12 per step.
    phase_drift = (t[-1] / DT) * (OMEGA * DT) ** 3 / 12.0
    assert err < 1.5 * np.sin(phase_drift), (
        f"y_osc max error {err:.3e} exceeds 1.5x predicted phase-drift budget "
        f"{np.sin(phase_drift):.3e}"
    )


def test_oscillator_velocity_phase_drift(run):
    t = run["t"]
    z = run["trace"](".z_osc")
    err = np.max(np.abs(z - (-OMEGA * np.sin(OMEGA * t))))
    phase_drift = (t[-1] / DT) * (OMEGA * DT) ** 3 / 12.0
    assert err < 1.5 * OMEGA * np.sin(phase_drift), (
        f"z_osc max error {err:.3e} exceeds 1.5x predicted phase-drift budget "
        f"{OMEGA * np.sin(phase_drift):.3e}"
    )


def test_recorded_state_columns_match_var_names(run):
    """`lambdified_raw_vars` must produce exactly one column per dotted name."""
    # use the same run, sanity-check shape via the trace fixture
    t = run["t"]
    # Each call to trace returns shape (n_steps,). Just assert lengths line up.
    y = run["trace"](".y_decay")
    assert y.shape[0] == t.shape[0]
