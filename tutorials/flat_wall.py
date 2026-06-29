"""Transient heat-up of a plane wall driven on one face, cooled on the other.

A 2 cm thick concrete wall (10 cm x 10 cm patch) starts at ambient. At t = 0
a constant heat flux is applied to the inner face while the outer face loses
heat by convection to still air. The two-node `FlatWall` captures the
through-thickness temperature gradient as the wall charges toward a new
steady state.

System layout:

    FixedHeatFlow(Q_in)  --[ port_a | FlatWall | port_b ]--  ConvectiveBoundary(h, T_inf)
                              inner surface   outer surface

Closed-form steady state (reached as t -> infinity):

    T_outer  = T_inf + Q_in / (h * A)              (all heat leaves by convection)
    T_inner  = T_outer + Q_in / (k * A / L)         (Fourier drop across the wall)

Run with `python tutorials/flat_wall.py` from the project root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from hydrogen import Model, plot_results  # noqa: E402
from hydrogen.components.thermofluid.walls import (  # noqa: E402
    ConvectiveBoundary,
    FixedHeatFlow,
    FlatWall,
)

# Material: concrete-ish slab ------------------------------------------------------------
RHO = 2300.0        # kg/m^3
CP = 880.0          # J/kg/K
K = 1.4             # W/m/K
A = 0.01            # m^2   (10 cm x 10 cm patch)
L = 0.02            # m     (2 cm thick)

# Boundary conditions --------------------------------------------------------------------
Q_IN = 20.0         # W     applied to the inner face
H_CONV = 10.0       # W/m^2/K  natural convection to still air on the outer face
T_INF = 293.15      # K     ambient
T_START = 293.15    # K     uniform initial wall temperature

# Time-stepping --------------------------------------------------------------------------
# The convective bulk time constant is R_conv * C_total = C_total/(h*A) ~ 4050 s,
# so march out to ~10 time constants to settle into the analytical steady state.
DT = 240.0
N_STEPS = 170       # ~40800 s simulated (~11 h)


class WallSystem(Model):
    """`FixedHeatFlow -> FlatWall -> ConvectiveBoundary`."""

    def declare_components(self):
        self.add_component('heater', FixedHeatFlow(Q_IN, T_init=T_START))
        self.add_component('wall', FlatWall(RHO, CP, K, A, L, T_init=T_START))
        self.add_component('air', ConvectiveBoundary(H_CONV, A, T_INF))

    def declare_equations(self):
        self.connect(self['heater'].ports['heat'], self['wall'].ports['port_a'])
        self.connect(self['wall'].ports['port_b'], self['air'].ports['heat'])
        return []


def main():
    print("Building model...")
    system = WallSystem()

    print("Instantiating (symbolic Jacobian + lambdify)...")
    t0 = time.time()
    system.instantiate(max_remove_trival_passes=3)
    print(f"  instantiate: {time.time() - t0:.2f} s")

    print("Initialising (Newton at t = 0)...")
    system.initialise()

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

    T_inner = trace('.wall.T_a')
    T_outer = trace('.wall.T_b')
    Q_in = trace('.wall.Q_dot_a')

    # Analytical steady state.
    G_cond = K * A / L
    T_outer_ss = T_INF + Q_IN / (H_CONV * A)
    T_inner_ss = T_outer_ss + Q_IN / G_cond

    print()
    print("=== Flat-wall heat-up summary ===")
    print(f"Material: rho={RHO} kg/m^3, cp={CP} J/kg/K, k={K} W/m/K")
    print(f"Geometry: A={A} m^2, L={L * 1000:.0f} mm  ->  C_total={RHO * CP * A * L:.1f} J/K, "
          f"G_cond={G_cond:.3f} W/K")
    print(f"Drive:    Q_in={Q_IN} W at inner face;  h={H_CONV} W/m^2/K to {T_INF:.2f} K outside")
    print()
    print(f"{'t [s]':>8}  {'T_inner [K]':>12}  {'T_outer [K]':>12}  {'dT_wall [K]':>12}  {'Q_in [W]':>9}")
    for i in range(0, len(t), max(1, N_STEPS // 10)):
        print(f"{t[i]:8.0f}  {T_inner[i]:12.3f}  {T_outer[i]:12.3f}  "
              f"{T_inner[i] - T_outer[i]:12.4f}  {Q_in[i]:9.4f}")

    print()
    print("Steady-state check (simulation end vs analytical):")
    print(f"  T_inner: {T_inner[-1]:8.3f} K  (analytical {T_inner_ss:8.3f} K)")
    print(f"  T_outer: {T_outer[-1]:8.3f} K  (analytical {T_outer_ss:8.3f} K)")
    print(f"  wall dT: {T_inner[-1] - T_outer[-1]:8.4f} K  (analytical {Q_IN / G_cond:8.4f} K)")

    # Self-validation: after ~10 time constants the simulation must sit on the
    # analytical steady state, the wall must carry exactly the injected flux,
    # and the through-thickness drop must obey Fourier's law.
    assert abs(T_outer[-1] - T_outer_ss) < 0.1, "outer-face temperature off steady state"
    assert abs(T_inner[-1] - T_inner_ss) < 0.1, "inner-face temperature off steady state"
    assert abs(Q_in[-1] - Q_IN) < 1e-3, "wall through-flux must equal the injected rate"
    assert abs((T_inner[-1] - T_outer[-1]) - Q_IN / G_cond) < 1e-2, "Fourier drop violated"

    out_path = plot_results(system.record, "flat_wall.html",
                            show=False, subdir="tutorials")
    print()
    print(f"Plot written to {out_path}")


if __name__ == "__main__":
    main()
