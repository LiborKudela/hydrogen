"""End-to-end demo: ambient inlet -> two heated pipes, plus a decoupled IntegrationTest.

Run with `python -m examples.run_system` from the project root, or just
`python examples/run_system.py` (the small `sys.path` shim below makes the latter
work without first `pip install -e .`-ing the package).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running this script directly via `python examples/run_system.py` by adding the
# project root (one level up from this file) to sys.path. No-op when the package is
# already installed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402  (after sys.path tweak)

from hydrogen import (  # noqa: E402
    AmbientInlet,
    CoolPropMedium,
    IntegrationTest,
    Model,
    StraightPipe,
    plot_results,
)


N = 100
L = 10
air = CoolPropMedium(
    'air',
    disable_warnings=True,
    backend="BICUBIC&HEOS",
    scalar_cache_maxsize=1000,
)


class System(Model):
    def declare_components(self):
        self.add_component('ambient_inlet', AmbientInlet(air, p_ambient=101325, T_ambient=273.15 + 60, m_flow=0.0745, D=0.0545))
        self.add_component('straight_pipe_1', StraightPipe(air, D=0.0545, L=L, epsilon=0.0001, z_in=0, z_out=0, n_segments=N))
        self.add_component('straight_pipe_2', StraightPipe(air, D=0.0545, L=L, epsilon=0.0001, z_in=0, z_out=0, n_segments=N))
        # decoupled sanity-check ODEs whose values we compare against analytical solutions.
        self.add_component('integration_test', IntegrationTest(omega=2 * np.pi))

    def declare_equations(self):
        res_1 = self['ambient_inlet']['p_out'].symbol - self['straight_pipe_1']['p_in'].symbol
        res_2 = self['ambient_inlet']['h_out'].symbol - self['straight_pipe_1']['h_in'].symbol
        res_3 = self['ambient_inlet']['w_out'].symbol - self['straight_pipe_1']['w_in'].symbol

        res_4 = self['straight_pipe_1']['p_out'].symbol - self['straight_pipe_2']['p_in'].symbol
        res_5 = self['straight_pipe_1']['h_out'].symbol - self['straight_pipe_2']['h_in'].symbol
        res_6 = self['straight_pipe_1']['w_out'].symbol - self['straight_pipe_2']['w_in'].symbol

        return [res_1, res_2, res_3, res_4, res_5, res_6]


def main():
    model_test = System()
    model_test.instantiate(aditional_modules=air.modules, max_remove_trival_passes=5)
    model_test.initialise(n=1)

    # Time-step with the adaptive predictor-corrector controller.
    #
    # NOTE on `tol_local`: PC's metric is the FE-CN gap, which is `dt`-times
    # LARGER than the actual CN local error.  The DEFAULT `tol_local=1e-2`
    # targets a tight scientific accuracy (CN local error ~1e-4 per step at
    # dt=0.01); on this 1 Hz harmonic-oscillator problem that translates to
    # `dt <= 0.02 s` -- twice what the original fixed-dt loop was using.
    # We loosen to `tol_local=1e-1` here, which matches the per-step error
    # that fixed dt=0.04 was already tolerating and lets the controller GROW
    # `dt` up to 0.08 s near the oscillator's inflection points.  Empirically
    # ~15% faster than fixed dt=0.04 at indistinguishable accuracy.
    #
    # The timed loop is kept to the bare minimum (solve + commit + one
    # already-warm attribute read) so wall-clock here measures the solver,
    # not Python bookkeeping.  Step diagnostics are reconstructed AFTER the
    # timer from `record['time']`; `_dt_hint` is seeded once before the loop
    # so the `getattr` / first-step branch doesn't have to live in the hot
    # path.
    T_END = 1.0
    DT_TARGET = 0.04
    DT_MAX = 2 * DT_TARGET
    STRATEGY = {"name": "predictor_corrector", "tol_local": 1e-1, "atol": 1e-6}
    model_test._dt_hint = DT_TARGET                      # seed; controller updates it after each step

    start_time = time.time()
    while model_test.get_t_value() < T_END - 1e-12:
        dt_try = min(DT_MAX, model_test._dt_hint, T_END - model_test.get_t_value())
        model_test.solve_adaptive_step(dt_try, dt_max=DT_MAX, strategy=STRATEGY)
        model_test.next_step()
    elapsed = time.time() - start_time

    # Reconstruct dt history from the recorded time stamps (rejections aren't
    # recorded -- they're rolled back -- so we can only see ACCEPTED dts here).
    dt_history = np.diff(np.asarray(model_test.record['time']))
    print(f"Time taken to solve: {elapsed:.4f} s "
          f"({len(dt_history)} accepted steps, "
          f"dt range {dt_history.min():.4f} .. {dt_history.max():.4f} s)")

    # Validate the time integrator against the analytical solutions of
    # IntegrationTest. With dt_target = 0.04 the simulation covers t in [0, 1] s
    # but dt may vary -- so the CN error bound below uses the LARGEST dt used.
    record = model_test.record
    t_arr = np.array(record['time'])
    state_arr = np.array(record['state'])
    names = list(record['vars_names'])

    def trace(name):
        return state_arr[:, names.index(name)]

    omega_val = 2 * np.pi
    y_decay_num = trace('System.integration_test.y_decay')
    y_osc_num = trace('System.integration_test.y_osc')
    z_osc_num = trace('System.integration_test.z_osc')

    y_decay_exact = np.exp(-t_arr)
    y_osc_exact = np.cos(omega_val * t_arr)
    z_osc_exact = -omega_val * np.sin(omega_val * t_arr)

    dt_max_used = dt_history.max()
    expected_decay = 0.5 * t_arr[-1] * dt_max_used ** 2  # loose bound for y=exp(-t)
    phase_drift = (t_arr[-1] / dt_max_used) * (omega_val * dt_max_used) ** 3 / 12.0
    expected_osc_y = np.sin(phase_drift)              # CN preserves amplitude; only phase drifts
    expected_osc_z = omega_val * np.sin(phase_drift)

    print("--- Time-integration check (IntegrationTest sub-model) ---")
    print(f"Exponential decay  y_decay : max |err| = {np.max(np.abs(y_decay_num - y_decay_exact)):.3e}    (CN expected: < {expected_decay:.1e})")
    print(f"Harmonic oscillator y_osc  : max |err| = {np.max(np.abs(y_osc_num - y_osc_exact)):.3e}    (CN expected: ~ {expected_osc_y:.1e})")
    print(f"Harmonic oscillator z_osc  : max |err| = {np.max(np.abs(z_osc_num - z_osc_exact)):.3e}    (CN expected: ~ {expected_osc_z:.1e})")
    print(f"  (CN preserves amplitude; residual is pure phase drift, bounded by")
    print(f"   ~ (omega*dt_max)^3 / 12 per step at dt_max = {dt_max_used:.4f} s.)")

    plot_results(model_test.record, "model_test.html", show=False)


if __name__ == "__main__":
    main()
