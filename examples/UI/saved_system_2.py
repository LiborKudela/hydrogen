"""Headless (pure-Python) twin of ``examples/UI/saved_system_2.json``.

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

Run with ``python examples/UI/saved_system_2.py`` from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

# The fluid for every wetted component (the JSON's "Hydrogen" medium).
HYDROGEN = CoolPropMedium("Hydrogen", disable_warnings=True, backend="BICUBIC&HEOS", scalar_cache_maxsize=1000)

# Default 2 mm AISI 316 wall layer (matches the catalogue template the UI used
# for the Pipe/Tank, which had `params: null`).
WALL = [WallLayer(material=AISI_316, thickness=0.002, dynamic=True)]


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
            n_segments=100, layers=WALL))
        self.add_component("tank_3", Tank(
            HYDROGEN, volume=0.05, diameter=0.3, layers=WALL, h_inner=50.0))
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


STOP_TIME = 100.0

# A tight reference trajectory, ~2 orders below the methods under test.  We use
# the L-stable TR-BDF2 here because Crank-Nicolson step-doubling (`richardson`)
# at this tolerance takes a pathological number of tiny steps on this stiff
# valve/pipe transient.  At tol_local=1e-6 the reference is ~converged to the
# true ODE solution, so it is a fair yardstick for every method (CN included).
REFERENCE = {"name": "tr_bdf2", "tol_local": 1e-6, "atol": 1.0}

# The methods under test, all at the SAME local-error budget so the comparison
# isolates the integrator (CN trapezoidal vs L-stable TR-BDF2) and its
# controller, not the tolerance.  Note `predictor_corrector`'s metric is the
# FE-CN mismatch (~dt larger than the CN local error), so its tol is listed
# separately for context.
METHODS = [
    {"name": "predictor_corrector", "tol_local": 1e-3, "atol": 1.0},
    {"name": "richardson",          "tol_local": 1e-4, "atol": 1.0},
    {"name": "tr_bdf2",             "tol_local": 1e-4, "atol": 1.0},
    {"name": "tr_bdf2",             "tol_local": 1e-5, "atol": 1.0},
]


def _build():
    """Instantiate + initialise a fresh network (mirrors the JSON sim_options)."""
    m = SavedSystem2()
    m.instantiate(
        aditional_modules=HYDROGEN.modules,
        cse=True, enable_blt=True, enable_var_scaling=False,
        max_remove_trival_passes=1, max_remove_duplicate_passes=5,
        max_remove_linear_block_passes=3,
    )
    m.initialise(n=1, relaxation=1.0, tol=1e-6, max_iter=200)
    return m


def _run(strategy):
    """Build, time, and run one method to ``STOP_TIME``.  Returns
    ``(model, summary, wall_seconds)``."""
    import time

    m = _build()
    t0 = time.perf_counter()
    summary = m.run(
        stop_time=STOP_TIME,
        strategy=strategy,
        dt_start=1e-4, dt_min=1e-9, dt_max=10.0,
        grow=1.5, shrink=0.5, max_retries=30,
        relaxation=1.0, tol=1e-6, max_iter=200,
        raise_on_no_convergence=False,
    )
    wall = time.perf_counter() - t0
    return m, summary, wall


def _max_trajectory_error(ref, run, tq):
    """Max *relative* deviation of `run` from `ref` across EVERY recorded
    variable, resampled onto the shared query grid `tq`.  Each variable is
    scaled by its own reference magnitude (floored at 1) so a 1e5 Pa pressure
    and a 1e-3 kg/s flow are weighted comparably.  Also returns the absolute
    tank-pressure error [bar] as a concrete headline number."""
    import numpy as np

    ref_state = ref.interp_state(tq)
    run_state = run.interp_state(tq)
    worst_rel, worst_var = 0.0, ""
    for name, ref_vals in ref_state.items():
        run_vals = run_state.get(name)
        if run_vals is None:
            continue
        # Skip the derivative companion variables (der_*): they are internal
        # RHS quantities that legitimately spike through zero at the valve
        # transient, so their *relative* error is meaningless and would
        # dominate a state-trajectory accuracy metric.
        if "der_" in name:
            continue
        # Amplitude-normalised error: divide the worst pointwise deviation by
        # the variable's own peak magnitude over the whole trajectory (floored
        # at 1).  This is a true relative error that doesn't blow up for
        # quantities that merely pass through zero mid-transient.
        scale = max(float(np.nanmax(np.abs(ref_vals))), 1.0)
        rel = float(np.nanmax(np.abs(run_vals - ref_vals))) / scale
        if rel > worst_rel:
            worst_rel, worst_var = rel, name
    p_ref = ref.interp_series("tank_3.gas.p", tq)
    p_run = run.interp_series("tank_3.gas.p", tq)
    p_abs_bar = float(np.nanmax(np.abs(p_run - p_ref))) / 1e5
    return worst_rel, worst_var, p_abs_bar


def main():
    import numpy as np

    print("=" * 78)
    print("saved_system_2 — method comparison (PressureSource -> Valve -> Pipe -> Tank)")
    print(f"stop_time={STOP_TIME:g}s   reference={REFERENCE['name']}"
          f"(tol_local={REFERENCE['tol_local']:g})")
    print("=" * 78)

    # Shared comparison grid: dense early (where the ramp/valve transient lives)
    # then sparse, so the max-error metric isn't dominated by the long tail.
    tq = np.unique(np.concatenate([
        np.linspace(0.0, 5.0, 500),
        np.linspace(5.0, STOP_TIME, 500),
    ]))

    # --- reference trajectory ---------------------------------------------- #
    ref_model, ref_summary, ref_wall = _run(REFERENCE)
    print(f"\nreference: steps={ref_summary['steps']} "
          f"rejections={ref_summary['rejections']} "
          f"t_end={ref_summary['t_end']:.4f}s wall={ref_wall:.2f}s "
          f"stop={ref_summary['stop_reason']}")

    # --- each method under test -------------------------------------------- #
    rows = []
    last = None
    for strat in METHODS:
        model, summary, wall = _run(strat)
        rel, var, p_bar = _max_trajectory_error(ref_model, model, tq)
        rows.append({
            "name": strat["name"], "tol": strat["tol_local"],
            "steps": summary["steps"], "rej": summary["rejections"],
            "wall": wall, "rel": rel, "var": var, "p_bar": p_bar,
            "stop": summary["stop_reason"],
        })
        last = model

    # --- comparison table -------------------------------------------------- #
    print(f"\n{'method':<20} {'tol_local':>10} {'steps':>7} {'rej':>5} "
          f"{'wall [s]':>9} {'speedup':>8} {'max rel err':>12} "
          f"{'max p err [bar]':>16}")
    print("-" * 95)
    slowest = max(r["wall"] for r in rows)
    for r in rows:
        print(f"{r['name']:<20} {r['tol']:>10.0e} {r['steps']:>7d} {r['rej']:>5d} "
              f"{r['wall']:>9.2f} {slowest / r['wall']:>7.2f}x "
              f"{r['rel']:>12.3e} {r['p_bar']:>16.3e}")
    print("-" * 95)
    print("max rel err = worst-variable relative deviation from the reference "
          "trajectory\n(resampled onto a shared grid, scale-floored at 1); "
          "speedup is vs the slowest method.")
    for r in rows:
        print(f"  {r['name']:<20} worst variable: {r['var']}")

    # --- plot the last (TR-BDF2) run --------------------------------------- #
    out = plot_results(last.record, "saved_system_2.html", show=False,
                       subdir="examples")
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
