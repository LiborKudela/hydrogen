"""Headless (pure-Python) twin of ``tutorials/saved_system_2.json``.

Same system the UI builds from that project file, wired and run *without* the
editor or the host/service layer -- so you can inspect the simulation output
directly:

    PressureSource ==> [ CompressibleValve ] ==> [ Pipe ] ==> [ Tank ]
                              ^ opening
                              |
                            Ramp (0 -> 1 over 1 s, starting at t = 1 s)

A high-pressure hydrogen source (~10 bar) feeds a compressible valve whose
opening is commanded by a ramp; downstream of a short pipe sits a closed tank
that fills up. The instantiate / initialise / run options mirror the
``sim_options`` saved in the JSON (richardson adaptive stepping to t = 1 s).

Run with ``python tutorials/saved_system_2.py`` from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydrogen import CoolPropMedium, Model, plot_results  # noqa: E402
from hydrogen.components.control.control_components import Ramp  # noqa: E402
from hydrogen.components.materials import AISI_316  # noqa: E402
from hydrogen.components.thermofluid.assemblies import (  # noqa: E402
    Pipe,
    Tank,
    WallLayer,
)
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    CompressibleValve,
    PressureSource,
)

from hydrogen.components.thermofluid.permeation import (  # noqa: E402
    H2_IN_AISI_316,
    TransientDiffusion,   # or SteadyRichardson for an algebraic, cheaper model
)

# The fluid for every wetted component (the JSON's "Hydrogen" medium).
HYDROGEN = CoolPropMedium("Hydrogen", disable_warnings=True, backend="BICUBIC&HEOS", scalar_cache_maxsize=1000)

# Default 2 mm AISI 316 wall layer (matches the catalogue template the UI used
# for the Pipe/Tank, which had `params: null`).
WALL = [WallLayer(
    material=AISI_316, thickness=0.002, dynamic=True,
    permeation=TransientDiffusion(H2_IN_AISI_316, n_nodes=5),
)]


class SavedSystem2(Model):
    """The five-component network from ``saved_system_2.json``.

    The valve opening is commanded by a :class:`Ramp` whose corners are
    declared as explicit time events, so the adaptive solver lands on
    ``start +/- eps`` rather than integrating across the kink.
    """

    def declare_components(self):
        # Non-default params come straight from the JSON; the valve / pipe /
        # tank had `params: null`, i.e. the catalogue defaults.
        self.add_component("pressuresource_5", PressureSource(
            HYDROGEN, p_source=1001325.0, T_source=293.15, A=0.001,
            p_control=False))
        self.add_component("compressiblevalve_4", CompressibleValve(
            HYDROGEN, Kv=1.0, D=0.01))
        self.add_component("pipe_2", Pipe(
            HYDROGEN, D=0.01, L=1.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
            n_segments=100, layers=WALL, p_ext=101325.0))
        self.add_component("tank_3", Tank(
            HYDROGEN, volume=0.05, diameter=0.3, layers=WALL, h_inner=50.0, p_ext=101325.0))
        self.add_component("ramp_6", Ramp(
            height=1.0, duration=1.0, start_time=1.0, offset=0.0, unit="1"))

    def declare_equations(self):
        self.connect(self["pressuresource_5"].ports["outlet"],
                     self["compressiblevalve_4"].ports["inlet"])
        self.connect(self["compressiblevalve_4"].ports["outlet"],
                     self["pipe_2"].ports["inlet"])
        self.connect(self["pipe_2"].ports["outlet"],
                     self["tank_3"].ports["inlet"])
        self.connect(self["ramp_6"].ports["y"],
                     self["compressiblevalve_4"].ports["opening"])
        return []


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def print_host_state():
    """Print CPU governor / clock / power state.

    Instantiation timings (especially the single-threaded ``collect_equations``
    phase) scale ~1/clock, so the same model can take 2x longer on battery with
    the ``powersave`` governor (base clock) than plugged in at boost clock.
    Logging this makes timing comparisons across runs apples-to-apples.
    """
    import glob
    import os

    governors = sorted({
        g for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
        if (g := _read(p))
    })
    cur_khz = [
        int(v) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")
        if (v := _read(p)) and v.isdigit()
    ]
    max_khz = [
        int(v) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq")
        if (v := _read(p)) and v.isdigit()
    ]
    ac = next((v for p in sorted(glob.glob("/sys/class/power_supply/A*/online"))
               if (v := _read(p)) is not None), None)
    power = {"1": "AC", "0": "battery"}.get(ac, "unknown")

    parts = [f"cores={os.cpu_count()}"]
    if governors:
        parts.append("governor=" + ",".join(governors))
    if cur_khz:
        parts.append(f"clock={max(cur_khz) / 1e6:.2f}GHz")
    if max_khz:
        parts.append(f"max={max(max_khz) / 1e6:.2f}GHz")
    parts.append(f"power={power}")
    print("host: " + "  ".join(parts))


def main():
    print("=" * 72)
    print("saved_system_2 — headless run (PressureSource -> Valve -> Pipe -> Tank)")
    print("=" * 72)
    print_host_state()

    m = SavedSystem2()

    # --- instantiate / initialise / run : mirror the JSON's sim_options ----- #
    m.instantiate(
        aditional_modules=HYDROGEN.modules,
        cse=True, enable_blt=True, enable_var_scaling=False,
        max_remove_trival_passes=1, max_remove_duplicate_passes=5,
        max_remove_linear_block_passes=3,
    )
    m.initialise(n=1, relaxation=1.0, tol=1e-6, max_iter=200)

    summary = m.run(
        stop_time=100.0,
        strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 1.0},
        dt_start=1e-4, dt_min=1e-9, dt_max=50.0,
        grow=1.5, shrink=0.5, max_retries=30,
        relaxation=1.0, tol=1e-6, max_iter=200,
        # Return gracefully (stop_reason="error") instead of raising if the
        # adaptive controller gives up, so the per-step trace below still
        # prints whatever steps completed.
        raise_on_no_convergence=False,
    )
    print(f"\nrun summary: {summary}")
    print(f"ramp corner events: {list(getattr(m, '_event_times', []) or [])}\n")

    # --- report a few representative traces over time ----------------------- #
    import numpy as np

    t = np.asarray(m.record["time"])

    def trace(suffix):
        return np.asarray(m.series(suffix))

    opening = trace("compressiblevalve_4.opening")
    m_dot = trace("compressiblevalve_4.m_dot_in")
    p_tank = trace("tank_3.gas.p")
    T_tank = trace("tank_3.gas.T")
    step_wall = np.asarray(m.record["step_wall_time"])
    step_dt_ms = step_wall * 1e3
    step_err = np.asarray(m.record["step_error"])
    # Simulated time advanced per step (s) and the real-time factor:
    # sim seconds advanced per wall second (>1 == faster than real time).
    sim_dt = np.diff(t, prepend=np.nan)
    rel_time = sim_dt / step_wall

    print(f"{'t [s]':>8}  {'opening':>8}  {'m_dot [kg/s]':>13}  "
          f"{'p_tank [bar]':>13}  {'T_tank [K]':>11}  "
          f"{'step [ms]':>10}  {'step_err':>10}  {'rel(dt/wall)':>13}")
    n = len(t)
    for i in range(n):
        print(f"{t[i]:8.4f}  {opening[i]:8.3f}  {m_dot[i]:13.5e}  "
              f"{p_tank[i] / 1e5:13.4f}  {T_tank[i]:11.3f}  "
              f"{step_dt_ms[i]:10.3f}  {step_err[i]:10.3e}  {rel_time[i]:13.3f}")

    print(f"\nsteps={summary['steps']} rejections={summary['rejections']} "
          f"t_end={summary['t_end']:.4f} s  stop={summary['stop_reason']}")
    print(f"total solver step time {np.nansum(step_dt_ms) / 1e3:.3f} s  "
          f"(max step {np.nanmax(step_dt_ms):.3f} ms)  "
          f"max step_err {np.nanmax(step_err):.3e}")
    print(f"tank pressure rose {p_tank[0] / 1e5:.3f} -> {p_tank[-1] / 1e5:.3f} bar")

    out = plot_results(m.record, "saved_system_2.html", show=False,
                       subdir="tutorials")
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
