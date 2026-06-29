"""Fill a `PressureVessel` from a higher-pressure source through a `StraightPipe`.

Demonstrates the natural pressure-driven transient: as the vessel pressure rises toward
the source pressure, the driving differential `dp = p_source - p_vessel` shrinks, the
pipe friction equation balances at a lower velocity, and the inflow gradually decays
toward zero.

System layout:

    PressureSource (2 bar, 293 K)  --[ StraightPipe (3 mm x 1 m, adiabatic) ]-->  PressureVessel (1 atm, 1 L)

Run with `python tutorials/fill_vessel.py` from the project root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from hydrogen import CoolPropMedium, Model, plot_results  # noqa: E402
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    PressureSource,
    PressureVessel,
    StraightPipe,
)

# Geometry / boundary conditions --------------------------------------------------------
P_SOURCE = 2.0e5         # Pa  (2 bar stagnation pressure upstream)
T_SOURCE = 293.15        # K
P_VESSEL_INIT = 1.013e5  # Pa  (1 atm)
T_VESSEL_INIT = 293.15   # K
PIPE_D = 0.003           # m   (3 mm bore -> friction-limited flow)
PIPE_L = 1.0             # m
PIPE_EPSILON = 1e-6      # m   (very smooth inner wall)
N_SEGMENTS = 2
VESSEL_V = 1e-3          # m^3 (1 L)
A_PORT = np.pi * PIPE_D ** 2 / 4

# Time-stepping --------------------------------------------------------------------------
DT = 0.025
N_STEPS = 120             # 3 s simulated


class FillSystem(Model):
    """`PressureSource -> StraightPipe -> PressureVessel`."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('source', PressureSource(self.medium, P_SOURCE, T_SOURCE, A=A_PORT))
        self.add_component(
            'pipe',
            StraightPipe(
                self.medium, PIPE_D, PIPE_L, PIPE_EPSILON,
                z_in=0.0, z_out=0.0, n_segments=N_SEGMENTS, adiabatic=True,
            ),
        )
        self.add_component(
            'vessel',
            PressureVessel(self.medium, VESSEL_V, A_PORT, P_VESSEL_INIT, T_VESSEL_INIT),
        )

    def declare_equations(self):
        # Wire `(p, h, m_dot)` through every joint via the typed-port
        # `connect()` API.  Each call expands to one signed `add_connection`
        # per channel, all of which collapse via the union-find pass at
        # instantiate time (cheaper than leaving residuals for the trivial
        # reducer).
        self.connect(self['source'].ports['outlet'], self['pipe'].ports['inlet'])
        self.connect(self['pipe'].ports['outlet'],   self['vessel'].ports['inlet'])
        return []


def main():
    print("Building model...")
    system = FillSystem()

    print("Instantiating (symbolic Jacobian + lambdify; can take ~10-30 s)...")
    t0 = time.time()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    print(f"  instantiate: {time.time() - t0:.2f} s")

    # Warm-start every `m_dot` unknown.  Default initial guess of `m_dot = 0`
    # places the boundary at the source's stagnation enthalpy with no flow,
    # but the 1 bar pressure mismatch at the vessel end has no friction loss
    # to absorb it (friction scales as `m_dot**2`), so Newton tries to push
    # m_dot to extreme values to satisfy the boundary pressure equation,
    # sometimes overshooting into a negative-density regime.  Pinning m_dot to
    # a physically plausible steady-state value side-steps this.
    h_src = float(system.medium.eval_h_pT(P_SOURCE, T_SOURCE))
    rho_src = float(system.medium.eval_rho_ph(P_SOURCE, h_src))
    WARM_M_DOT = 30.0 * rho_src * A_PORT  # ~30 m/s order-of-magnitude steady-state
    for var in system.active_vars_references:
        full = getattr(var, 'full_name', '')
        if full.endswith('.m_dot_in') or full.endswith('.m_dot_out'):
            var.value = WARM_M_DOT

    # Heavier damping (relaxation=0.3) keeps the t=0 Newton solve from
    # overshooting into a negative-pressure state that CoolProp can't flash;
    # the 1 bar source/vessel mismatch makes the first few steps stiff.
    print("Initialising (damped Newton at t = 0)...")
    t0 = time.time()
    system.initialise(relaxation=0.3, max_iter=600)
    print(f"  initialise:  {time.time() - t0:.2f} s")

    print(f"Running {N_STEPS} steps of dt = {DT:g} s ({N_STEPS * DT:g} s total)...")
    t0 = time.time()
    for _ in range(N_STEPS):
        system.solve_dae_step(DT)
        system.next_step()
    print(f"  solve loop:  {time.time() - t0:.2f} s")

    # --- post-process ----------------------------------------------------------------
    record = system.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    p_v = trace('.vessel.p')
    h_v = trace('.vessel.h')
    m_v = trace('.vessel.m')
    m_dot_in = trace('.vessel.m_dot_in')

    # Reconstruct vessel temperature from (p, h) via the medium.
    T_v = np.array([system.medium.eval_T_ph(float(pi), float(hi)) for pi, hi in zip(p_v, h_v)])

    decay_pct = (1.0 - m_dot_in[-1] / m_dot_in[0]) * 100.0 if m_dot_in[0] != 0.0 else 0.0
    pressure_progress = (p_v[-1] - p_v[0]) / (P_SOURCE - p_v[0]) * 100.0

    print()
    print("=== Filling transient summary ===")
    print(f"Source:        p = {P_SOURCE / 1e5:.3f} bar,  T = {T_SOURCE:.2f} K")
    print(f"Vessel start:  p = {p_v[0] / 1e5:.3f} bar,  T = {T_v[0]:.2f} K,  m = {m_v[0] * 1000:.3f} g")
    print(f"Vessel end:    p = {p_v[-1] / 1e5:.3f} bar,  T = {T_v[-1]:.2f} K,  m = {m_v[-1] * 1000:.3f} g")
    print(f"Inlet m_dot:   start = {m_dot_in[0] * 1000:.3f} g/s,  end = {m_dot_in[-1] * 1000:.3f} g/s   ({decay_pct:.1f}% decay)")
    print(f"Vessel pressure has closed {pressure_progress:.1f}% of the gap to source pressure.")

    print()
    print(f"Sample trajectory (every {max(1, N_STEPS // 10)} steps):")
    print(f"{'t [s]':>7}  {'p_v [bar]':>10}  {'T_v [K]':>8}  {'m_dot [g/s]':>11}  {'m [g]':>7}")
    for i in range(0, len(t), max(1, N_STEPS // 10)):
        print(f"{t[i]:7.3f}  {p_v[i] / 1e5:10.4f}  {T_v[i]:8.2f}  {m_dot_in[i] * 1000:11.4f}  {m_v[i] * 1000:7.3f}")

    # Self-validation: a higher-pressure source filling the vessel must raise
    # its pressure and mass monotonically toward the source, while the inflow
    # decays as the driving pressure difference shrinks.
    assert p_v[-1] > p_v[0], "vessel pressure should rise while filling"
    assert m_v[-1] > m_v[0], "vessel mass should grow while filling"
    assert p_v[-1] <= P_SOURCE + 1.0, "vessel pressure should not overshoot the source"
    assert m_dot_in[-1] < m_dot_in[0], "inflow should decay as the vessel pressurises"
    assert np.all(np.diff(m_v) >= -1e-9), "vessel mass should be non-decreasing"

    out_path = plot_results(system.record, "fill_vessel.html",
                            show=False, subdir="tutorials")
    print()
    print(f"Plot written to {out_path}")


if __name__ == "__main__":
    main()
