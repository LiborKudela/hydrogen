"""Conjugate heat transfer: hot gas cooling in a metal pipe (power domain).

Hot air (200 C) is pushed at a fixed mass flow into a 2 m steel pipe whose
wall has real thermal mass.  Each fluid segment exchanges heat with the metal
through its `wall` port; the metal in turn loses heat to still ambient air by
convection on its outer surface.  Over time the metal warms toward a steady
profile and the gas leaves cooler than it entered.

System layout (`power.ConjugatePipe` wires the inner half automatically):

    AmbientInlet(200 C, m_flow) --> [ ConjugatePipe ] --> (open outlet)

        fluid:  ==[seg_0]==[seg_1]== ... ==[seg_N-1]==
                    |wall     |wall            |wall
        metal:   [wall_0]  [wall_1]   ...  [wall_N-1]
                    |          |               |
        ambient: convective film to 20 C on each outer surface

Validation (printed + asserted at the end):
  * Telescoping energy balance  m_dot*(h_out - h_in) == sum_i q_i  (this is
    what the q/m_dot and per-segment-area corrections buy us).
  * The gas cools (h_out < h_in) and the metal warms above its 20 C start.

Run with `python tutorials/conjugate_pipe.py` from the project root.
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
from hydrogen.components.power.power_components import ConjugatePipe  # noqa: E402
from hydrogen.components.thermofluid.flow import AmbientInlet  # noqa: E402

# Geometry / material -------------------------------------------------------
D = 0.05            # m     inner (flow) diameter
L = 2.0             # m     pipe length
N = 6               # number of axial segments
WALL_T = 0.004      # m     steel wall thickness
RHO_W = 7850.0      # kg/m^3   steel
CP_W = 490.0        # J/kg/K
K_W = 45.0          # W/m/K

# Operating point -----------------------------------------------------------
T_HOT = 273.15 + 200.0   # K   hot gas inlet
M_FLOW = 0.03            # kg/s
H_EXT = 15.0             # W/m^2/K  outer natural convection
T_AMB = 273.15 + 20.0    # K   ambient
T_WALL_START = T_AMB     # metal starts cold

# Time-stepping -------------------------------------------------------------
DT = 2.0
N_STEPS = 150

AIR = CoolPropMedium('air', disable_warnings=True, backend="BICUBIC&HEOS", scalar_cache_maxsize=1000)


class HotGasPipe(Model):
    def declare_components(self):
        self.add_component('inlet', AmbientInlet(AIR, p_ambient=101325, T_ambient=T_HOT, m_flow=M_FLOW, D=D))
        self.add_component('pipe', ConjugatePipe(
            AIR, D=D, L=L, epsilon=1e-4, z_in=0, z_out=0, n_segments=N,
            wall_thickness=WALL_T, rho_wall=RHO_W, cp_wall=CP_W, k_wall=K_W,
            T_wall_init=T_WALL_START, outer='convective', h_ext=H_EXT, T_ext=T_AMB))

    def declare_equations(self):
        # AmbientInlet imposes m_flow + isentropic inlet state (square on its
        # own), so it fixes the pressure level; leave the pipe outlet open.
        self.connect(self['inlet'].ports['outlet'], self['pipe'].ports['inlet'])
        return []


def main():
    print("Building model...")
    system = HotGasPipe()

    print("Instantiating (symbolic Jacobian + lambdify)...")
    t0 = time.time()
    system.instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=5)
    print(f"  instantiate: {time.time() - t0:.2f} s")

    print("Initialising (Newton at t = 0)...")
    system.initialise(n=1)

    print(f"Running {N_STEPS} steps of dt = {DT:g} s ({N_STEPS * DT:g} s total)...")
    t0 = time.time()
    for _ in range(N_STEPS):
        system.solve_dae_step(DT)
        system.next_step()
    print(f"  solve loop:  {time.time() - t0:.2f} s")

    rec = system.record
    t = np.asarray(rec['time'])
    names = list(rec['vars_names'])
    state = np.asarray(rec['state'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    # The pipe is a SegmentedChannel with N cells / N+1 shared faces: face 0 is
    # the inlet, face N the outlet (h_0..h_N, p_0..p_N), with per-cell
    # q_inflow_i and wall nodes wall_i_0.
    h_in = trace('.pipe.pipe.h_0')
    h_out = trace(f'.pipe.pipe.h_{N}')
    m_dot = trace('.pipe.pipe.m_dot_in')
    q_total = np.zeros_like(h_in)
    for i in range(N):
        q_total = q_total + trace(f'.pipe.pipe.q_inflow_{i}')
    fluid_power = m_dot * (h_out - h_in)

    # Convert inlet/outlet enthalpy to temperature for a readable summary.
    p_io = trace('.pipe.pipe.p_0')
    T_in = np.array([AIR.eval_T_ph(p_io[k], h_in[k]) for k in range(len(t))])
    p_out = trace(f'.pipe.pipe.p_{N}')
    T_out = np.array([AIR.eval_T_ph(p_out[k], h_out[k]) for k in range(len(t))])

    wall_first = trace('.wall_0_0.T_a')
    wall_last = trace(f'.wall_{N - 1}_0.T_a')

    print()
    print("=== Conjugate hot-gas pipe summary ===")
    print(f"Gas: air at {T_HOT - 273.15:.0f} C, m_flow={M_FLOW} kg/s through D={D*1000:.0f} mm x {L} m pipe")
    print(f"Wall: steel, {WALL_T*1000:.0f} mm thick; outer h={H_EXT} W/m^2/K to {T_AMB - 273.15:.0f} C ambient")
    print()
    print(f"{'t [s]':>8}  {'T_in [C]':>9}  {'T_out [C]':>9}  {'wall_in [C]':>11}  {'wall_out [C]':>12}  {'Q_fluid [W]':>11}")
    for i in range(0, len(t), max(1, N_STEPS // 10)):
        print(f"{t[i]:8.0f}  {T_in[i]-273.15:9.2f}  {T_out[i]-273.15:9.2f}  "
              f"{wall_first[i]-273.15:11.2f}  {wall_last[i]-273.15:12.2f}  {fluid_power[i]:11.2f}")

    rel_err = abs(fluid_power[-1] - q_total[-1]) / max(1e-9, abs(q_total[-1]))
    print()
    print("Energy-balance check (final step):")
    print(f"  m_dot*(h_out - h_in) = {fluid_power[-1]:10.3f} W")
    print(f"  sum_i q_i            = {q_total[-1]:10.3f} W   (rel. err {rel_err:.2e})")

    # Self-validation.
    assert rel_err < 1e-2, "telescoping energy balance violated (q/m_dot or area bug?)"
    assert h_out[-1] < h_in[-1], "hot gas must cool along the pipe"
    assert wall_first[-1] > T_WALL_START + 1.0, "metal wall must heat up"
    # Outer surface loses heat to ambient: fluid heat in ~= ambient heat out at
    # quasi-steady state (wall storage still charging, so allow slack).
    assert fluid_power[-1] < 0.0, "fluid must be giving up heat"

    out_path = plot_results(system.record, "conjugate_pipe.html", show=False, subdir="tutorials")
    print()
    print(f"Plot written to {out_path}")


if __name__ == "__main__":
    main()
