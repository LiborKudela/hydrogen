"""Pressurise a hydrogen pipe to 20 MPa and watch it leak through the wall.

This is the *authoring* side of the tutorial: we build the model in Python,
save it to JSON, then simulate it in-process and print a small leakage table.

The companion ``run_service.py`` loads the saved ``system.json`` on a hydrogen
host -- it never sees this file -- and prints the same table, so the two runs
can be compared by eye.

    python examples/h2_permeation_pressurize/run_local.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow running the script directly (no `pip install -e .` required).
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

import hydrogen as hd  # noqa: E402

SPEC_PATH = _HERE / "system.json"

# --- operating point & geometry --------------------------------------------
# Tabular backend: faster + more robust than per-call HEOS flashes through the
# big pressurisation transient.
H2 = hd.CoolPropMedium("Hydrogen", backend="BICUBIC&HEOS", disable_warnings=True)

P_FULL = 20.0e6        # Pa     target / source pressure (20 MPa)
T_OP = 423.15          # K      150 C
P_START = 1.0e5        # Pa     initial bore pressure (1 bar) -> pressurises up
P_EXT = 1.0            # Pa     external H2 partial pressure (vents to air ~ 0)

R_IN = 1.5e-3          # m      bore radius   (D_in  = 3 mm)
R_OUT = 3.0e-3         # m      outer radius  (D_out = 6 mm) -> wall 1.5 mm
L_PIPE = 10.0          # m      length
A_BORE = math.pi * R_IN ** 2

N_SEG = 2              # pipe flow segments (each with its own wall)
N_NODES = 8            # wall diffusion shells (transient model)
T_RAMP = 0.06          # s      pressurisation window (ramp 1 bar -> 20 MPa)

# --- run configuration ------------------------------------------------------
DAY = 24 * 3600.0
T_END = T_RAMP + 365.0 * DAY                 # ramp, then ~1 year of leak
STRATEGY = {"name": "richardson", "tol_local": 1e-3, "atol": 1.0}


class PressurizedPipe(hd.Model):
    """A hydrogen flow pipe held at 150 C, pressurised to 20 MPa by a ramp,
    each of its ``N_SEG`` segments wrapped in a permeable AISI-316 wall.

    A `Ramp` drives the source's `p_set` signal from 1 bar to 20 MPa over
    `T_RAMP`, so the whole pressurise-then-leak schedule is just "advance time".
    """

    def declare_components(self):
        # Supply-pressure command: ramp 1 bar -> 20 MPa over T_RAMP, then flat.
        self.add_component("p_cmd", hd.components.control.Ramp(
            height=P_FULL - P_START, duration=T_RAMP, start_time=0.0,
            offset=P_START, unit="Pa"))
        # Pressure source driven by the ramp through its `p_set` signal input.
        self.add_component("source", hd.components.thermofluid.PressureSource(
            H2, p_source=P_START, T_source=T_OP, A=A_BORE, p_control=True))
        # The walled pipe: a flow pipe wrapped per segment in a permeable
        # AISI-316 wall (transient diffusion), held isothermal and venting to a
        # near-zero external H2 partial pressure.
        self.add_component("pipe", hd.components.thermofluid.Pipe(
            H2, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
            n_segments=N_SEG,
            layers=[hd.components.thermofluid.WallLayer(
                hd.components.materials.AISI_316, R_OUT - R_IN,
                permeation=hd.components.thermofluid.TransientDiffusion(
                    hd.components.thermofluid.H2_IN_AISI_304, n_nodes=N_NODES),
                dynamic=False)],
            outer_thermal="fixed", T_outer=T_OP, p_ext=P_EXT,
            T_wall_init=T_OP, p_init=P_START))
        # Far end sealed -> the only outflow is the wall permeation.
        self.add_component("cap", hd.components.thermofluid.ClosedEnd(
            H2, p_init=P_START, T_init=T_OP))

    def declare_equations(self):
        self.connect(self["p_cmd"].ports["y"], self["source"].ports["p_set"])
        self.connect(self["source"].ports["outlet"], self["pipe"].ports["inlet"])
        self.connect(self["pipe"].ports["outlet"], self["cap"].ports["inlet"])
        return []


def report(t, p_bore, uptake, env):
    """Print a sampled leakage table + summary from the result timeseries."""
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (env[1:] + env[:-1]) * np.diff(t))))
    print("\n" + "-" * 64)
    print(f"{'t [days]':>10}  {'p [MPa]':>8}  {'uptake [kg/s]':>13}  "
          f"{'env leak [kg/s]':>15}  {'cum loss [ug]':>13}")
    print("-" * 64)
    for i in np.linspace(0, len(t) - 1, 12).astype(int):
        print(f"{t[i]/DAY:10.1f}  {p_bore[i]/1e6:8.3f}  {abs(uptake[i]):13.3e}  "
              f"{env[i]:15.3e}  {cum[i]*1e9:13.4f}")
    print(f"\n  final bore pressure : {p_bore[-1]/1e6:.3f} MPa")
    print(f"  env leak at end     : {env[-1]:.3e} kg/s")
    print(f"  cumulative H2 lost  : {cum[-1]*1e9:.3f} ug")


def main():
    system = PressurizedPipe()

    # Save the model to JSON (run_service.py loads this exact file).
    SPEC_PATH.write_text(hd.to_json(system))
    print(f"saved model to {SPEC_PATH}")

    # Simulate it in-process.
    system.instantiate(aditional_modules=H2.modules)
    system.initialise(n=1, tol=1e-6, max_iter=200, line_search=True)
    system.run(stop_time=T_END, strategy=STRATEGY, dt_start=1e-4, dt_min=1e-7,
               dt_max=10.0 * DAY, tol=1e-6, max_iter=200)

    # Read the timeseries back through the in-process Model accessors.
    report(
        t=system.record_time(),
        p_bore=system.series("pipe_segment_0.p_in"),
        uptake=system.series_values("m_dot_a_leak").sum(axis=1),
        env=np.abs(system.series_values("m_dot_b_leak").sum(axis=1)),
    )


if __name__ == "__main__":
    main()
