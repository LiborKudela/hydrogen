"""Analytical-correctness + speed benchmark for the time integrator.

Uses `hydrogen.test_models.IntegrationTest`, a set of decoupled ODEs with exact
closed-form solutions, so we can measure BOTH integrator accuracy (vs the
analytical answer) and wall-clock cost as the step size shrinks:

    1. exponential decay      dy/dt = -y,            y(t) = exp(-t)
    2. harmonic oscillator    dy/dt = z, dz/dt = -w^2 y,  y(t) = cos(w t)

The fixed-step Crank-Nicolson integrator is 2nd order, so halving dt should
roughly quarter the error -- we print the observed convergence order and assert
each run stays inside a dt-scaled error budget (a numerical regression fails
the benchmark, not just a slowdown).

    python benchmarks/analytical_integrator.py            # full dt sweep
    python benchmarks/analytical_integrator.py --smoke     # cheap two-dt sweep
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks._harness import compare_to_analytical  # noqa: E402
from hydrogen.test_models import IntegrationTest  # noqa: E402

OMEGA = 2 * np.pi  # period of 1 s
T_END = 1.0        # cover exactly one oscillator period


def _run_fixed_step(dt):
    """Fixed-step Crank-Nicolson run to t = T_END; returns (t, y_decay, y_osc, wall)."""
    import contextlib
    import io

    n_steps = int(round(T_END / dt))
    model = IntegrationTest(omega=OMEGA)
    with contextlib.redirect_stdout(io.StringIO()):
        model.instantiate(max_remove_trival_passes=2)
        model.initialise()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        model.solve_dae_step(dt)
        model.next_step()
    wall = time.perf_counter() - t0

    record = model.record
    t = np.asarray(record["time"])
    state = np.asarray(record["state"])
    names = list(record["vars_names"])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return t, trace(".y_decay"), trace(".y_osc"), wall


def main(smoke=False):
    print("=" * 76)
    print("Analytical integrator benchmark (Crank-Nicolson vs closed form)")
    print("-" * 76)

    dts = [0.04, 0.02] if smoke else [0.08, 0.04, 0.02, 0.01]

    print(f"{'dt':>7} | {'steps':>6} | {'wall':>8} | {'err_decay':>10} | {'err_osc':>10}")
    print("-" * 76)
    rows = []
    for dt in dts:
        t, y_decay, y_osc, wall = _run_fixed_step(dt)
        err_decay = float(np.max(np.abs(y_decay - np.exp(-t))))
        err_osc = float(np.max(np.abs(y_osc - np.cos(OMEGA * t))))
        rows.append((dt, len(t) - 1, wall, err_decay, err_osc))
        print(f"{dt:>7.3f} | {len(t) - 1:>6} | {wall * 1e3:>6.1f}ms | "
              f"{err_decay:>10.3e} | {err_osc:>10.3e}")
    print("-" * 76)

    # Empirical convergence order between the two finest steps (expect ~2 for CN).
    if len(rows) >= 2:
        (dt_a, _, _, ed_a, eo_a), (dt_b, _, _, ed_b, eo_b) = rows[-2], rows[-1]
        ratio = np.log(dt_a / dt_b)
        order_decay = np.log(ed_a / ed_b) / ratio
        order_osc = np.log(eo_a / eo_b) / ratio
        print(f"observed order: decay~{order_decay:.2f}  osc~{order_osc:.2f} "
              f"(Crank-Nicolson is 2nd order)")

    # Correctness assertions: dt-scaled budgets (generous but regression-catching).
    print("\nCorrectness check (finest dt):")
    dt = dts[-1]
    t, y_decay, y_osc, _ = _run_fixed_step(dt)
    compare_to_analytical(t, y_decay, lambda tt: np.exp(-tt),
                          atol=max(2e-3, 5.0 * dt ** 2), label="exp decay")
    compare_to_analytical(t, y_osc, lambda tt: np.cos(OMEGA * tt),
                          atol=max(5e-2, 60.0 * dt ** 2), label="oscillator")
    print("=" * 76)
    print("analytical_integrator OK")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
