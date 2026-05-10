"""Fill a `PressureVessel` from a higher-pressure source through a `StraightPipe`.

Demonstrates the natural pressure-driven transient: as the vessel pressure rises toward
the source pressure, the driving differential `dp = p_source - p_vessel` shrinks, the
pipe friction equation balances at a lower velocity, and the inflow gradually decays
toward zero.

System layout:

    PressureSource (2 bar, 293 K)  --[ StraightPipe (3 mm x 1 m, adiabatic) ]-->  PressureVessel (1 atm, 1 L)

Run with `python examples/fill_vessel.py` from the project root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from hydrogen import (  # noqa: E402
    CoolPropMedium,
    Model,
    PressureSource,
    PressureVessel,
    StraightPipe,
    plot_results,
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
        self.add_component('source', PressureSource(self.medium, P_SOURCE, T_SOURCE))
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
        return [
            self['source']['p_out'].symbol - self['pipe']['p_in'].symbol,
            self['source']['h_out'].symbol - self['pipe']['h_in'].symbol,
            self['source']['w_out'].symbol - self['pipe']['w_in'].symbol,
            self['pipe']['p_out'].symbol - self['vessel']['p_in'].symbol,
            self['pipe']['h_out'].symbol - self['vessel']['h_in'].symbol,
            self['pipe']['w_out'].symbol - self['vessel']['w_in'].symbol,
        ]


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

    # Warm-start the velocity unknowns. The default initial guess is `w ≈ 0.1` m/s
    # everywhere, but at the boundaries we have a 1 bar pressure mismatch with no
    # friction loss to support it (since friction scales as `ρ*w²`). The Newton solve
    # then tries to push `w` to supersonic values just to balance the boundary pressure
    # eq, which sends `h_out = h_total - w²/2` negative and crashes CoolProp. Putting
    # all velocities near a physically plausible steady-state value side-steps this.
    WARM_W = 30.0  # m/s, rough order-of-magnitude steady-state velocity
    for var in system.active_vars_references:
        full = getattr(var, 'full_name', '')
        if full.endswith('.w_in') or full.endswith('.w_out'):
            var.value = WARM_W

    print("Initialising (damped Newton at t = 0)...")
    t0 = time.time()
    system.initialise(relaxation=0.5, max_iter=400)
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
    w_in = trace('.vessel.w_in')

    # Reconstruct vessel temperature from (p, h) via the medium.
    T_v = np.array([system.medium.eval_T_ph(float(pi), float(hi)) for pi, hi in zip(p_v, h_v)])

    decay_pct = (1.0 - w_in[-1] / w_in[0]) * 100.0 if w_in[0] != 0.0 else 0.0
    pressure_progress = (p_v[-1] - p_v[0]) / (P_SOURCE - p_v[0]) * 100.0

    print()
    print("=== Filling transient summary ===")
    print(f"Source:        p = {P_SOURCE / 1e5:.3f} bar,  T = {T_SOURCE:.2f} K")
    print(f"Vessel start:  p = {p_v[0] / 1e5:.3f} bar,  T = {T_v[0]:.2f} K,  m = {m_v[0] * 1000:.3f} g")
    print(f"Vessel end:    p = {p_v[-1] / 1e5:.3f} bar,  T = {T_v[-1]:.2f} K,  m = {m_v[-1] * 1000:.3f} g")
    print(f"Inlet w_in:    start = {w_in[0]:.3f} m/s,  end = {w_in[-1]:.3f} m/s   ({decay_pct:.1f}% decay)")
    print(f"Vessel pressure has closed {pressure_progress:.1f}% of the gap to source pressure.")

    print()
    print(f"Sample trajectory (every {max(1, N_STEPS // 10)} steps):")
    print(f"{'t [s]':>7}  {'p_v [bar]':>10}  {'T_v [K]':>8}  {'w_in [m/s]':>11}  {'m [g]':>7}")
    for i in range(0, len(t), max(1, N_STEPS // 10)):
        print(f"{t[i]:7.3f}  {p_v[i] / 1e5:10.4f}  {T_v[i]:8.2f}  {w_in[i]:11.4f}  {m_v[i] * 1000:7.3f}")

    out_path = plot_results(system.record, "fill_vessel.html",
                            show=False, subdir="examples")
    print()
    print(f"Plot written to {out_path}")


if __name__ == "__main__":
    main()
