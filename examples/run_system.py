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


N = 3
L = 10
air = CoolPropMedium('air', disable_warnings=True)


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

    start_time = time.time()
    for _ in range(25):
        model_test.solve_dae_step(0.04)
        model_test.next_step()
    print(f"Time taken to solve: {time.time() - start_time} seconds")

    # Validate the time integrator against the analytical solutions of IntegrationTest.
    # With dt = 0.04 and 25 steps the simulation covers t in [0, 1] s.
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

    dt_used = 0.04
    expected_decay = 0.5 * t_arr[-1] * dt_used ** 2  # loose conservative bound for y=exp(-t)
    phase_drift = abs(t_arr[-1] / dt_used) * (omega_val * dt_used) ** 3 / 12.0
    expected_osc_y = np.sin(phase_drift)              # CN preserves amplitude; only phase drifts
    expected_osc_z = omega_val * np.sin(phase_drift)

    print("--- Time-integration check (IntegrationTest sub-model) ---")
    print(f"Exponential decay  y_decay : max |err| = {np.max(np.abs(y_decay_num - y_decay_exact)):.3e}    (CN expected: < {expected_decay:.1e})")
    print(f"Harmonic oscillator y_osc  : max |err| = {np.max(np.abs(y_osc_num - y_osc_exact)):.3e}    (CN expected: ~ {expected_osc_y:.1e})")
    print(f"Harmonic oscillator z_osc  : max |err| = {np.max(np.abs(z_osc_num - z_osc_exact)):.3e}    (CN expected: ~ {expected_osc_z:.1e})")
    print("  (CN preserves oscillator amplitude exactly; the residual is pure phase drift")
    print(f"   ~ (omega*dt)^3 / 12 per step = {phase_drift:.3e} rad over the full window.)")

    plot_results(model_test.record, "model_test.html", show=False)


if __name__ == "__main__":
    main()
