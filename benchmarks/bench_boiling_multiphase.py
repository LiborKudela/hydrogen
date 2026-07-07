"""Boiling-water (multiphase / two-phase dome) benchmark.

A heated pipe carries subcooled liquid water; the wall heat is ramped until
the outlet crosses the WHOLE two-phase dome -- subcooled liquid -> saturated
mixture -> superheated steam.  The pipe segments are steady (algebraic), so
every ramp step is an exact steady state and carries an ANALYTIC reference:

    h_out = h_in + Q / m_dot           (energy balance, exact)
    T_out = T_sat(p_out)  while  0 < x < 1   (boiling plateau)

The same rig is run on two property paths:

  * ``coolprop`` -- `CoolPropMedium` with the smooth finite-difference HEM
    partials (``multiphase="HEM"``), the reference implementation;
  * ``tab``      -- `TabulatedMedium` over a dome-crossing window (saturation
    splines + mapped single-phase surfaces + analytic HEM mixture rules).

and the benchmark reports, for each: the max energy-balance defect, the
outlet-temperature deviation between the two paths, the boiling-plateau
flatness, and the wall time per ramp step.  This is the reference case for
two-phase property-path work (e.g. the numba dome kernel -- run with
``--numba`` to see which equation templates compile).

    python benchmarks/bench_boiling_multiphase.py            # both paths
    python benchmarks/bench_boiling_multiphase.py --medium tab --numba
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import CoolProp.CoolProp as CP  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hydrogen import CoolPropMedium, Model, TabulatedMedium  # noqa: E402
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    AmbientInlet,
    StraightPipe,
)
from hydrogen.components.thermofluid.walls import FixedHeatFlow  # noqa: E402

# --- operating point ---------------------------------------------------------
FLUID = "Water"
P_OP = 5.0e5         # Pa    (T_sat = 151.8 degC)
T_IN = 300.0         # K     subcooled liquid feed
M_FLOW = 1.0e-3      # kg/s
D_BORE = 0.01        # m
L_PIPE = 1.0         # m
N_SEG = 4            # steady segments
Q_OVERDRIVE = 1.3    # ramp to 1.3x the heat that just dries the outlet
# Ramp steps (each an exact steady state).  >=150 keeps the per-step heat jump
# small enough that the tab path's stiff vapor-side steady Jacobian stays
# solvable all the way to superheat; CoolProp is happy at any count.
N_RAMP = 200


def _coolprop_medium():
    # BICUBIC&HEOS: fast tabular flash for water (fine here -- it is only
    # subcooled LN2 that breaks it), FD-HEM partials inside the dome.
    return CoolPropMedium(FLUID, disable_warnings=True,
                          backend="BICUBIC&HEOS", scalar_cache_maxsize=4000)


def _tab_medium():
    """Dome-crossing water window enclosing the whole ramp trajectory.

    p: around the operating 5 bar (the rig's pressure drop is tiny, but the
    window must keep the saturation edges inside it for the mapped
    surfaces).  h: from below the 300 K feed up past the saturated-vapor
    line everywhere in the p window, plus superheat headroom.
    """
    src = CoolPropMedium(FLUID, disable_warnings=True, backend="HEOS",
                         scalar_cache_maxsize=4000)
    p_lo, p_hi = 1.0e5, 10.0e5
    h_lo = 0.8e5                                     # < h(300 K, 10 bar)
    # Top of the window must ENCLOSE the whole ramp: the outlet reaches
    # x ~ Q_OVERDRIVE (deep superheat) at the operating pressure, i.e.
    # h_out_max = h_in + Q_OVERDRIVE * (h_g - h_in).  Sampling past that keeps
    # the deep-superheat points interpolated (not log-linearly extrapolated,
    # which drifts ~40 K by x~1.4) so the tab T matches CoolProp everywhere.
    hg_op = float(CP.PropsSI("H", "P", P_OP, "Q", 1, FLUID))
    h_in_op = float(src.eval_h_pT(P_OP, T_IN))
    h_hi = h_in_op + Q_OVERDRIVE * (hg_op - h_in_op) + 0.2e6
    t0 = time.time()
    tab = TabulatedMedium(src, p_range=(p_lo, p_hi), h_range=(h_lo, h_hi),
                          n_p=128, n_h=256)
    print(f"  TabulatedMedium: p=[{p_lo / 1e5:.0f}, {p_hi / 1e5:.0f}] bar, "
          f"h=[{h_lo / 1e3:.0f}, {h_hi / 1e3:.0f}] kJ/kg, built in "
          f"{time.time() - t0:.1f} s, max rel err "
          f"{max(tab.validation_max_rel_err.values()):.1e}")
    return tab


class Boiler(Model):
    """AmbientInlet(subcooled water) -> heated steady StraightPipe."""

    def __init__(self, medium, N=N_SEG):
        self._medium = medium
        self.N = N
        super().__init__()

    def declare_components(self):
        self.add_component("inlet", AmbientInlet(
            self._medium, p_ambient=P_OP, T_ambient=T_IN, m_flow=M_FLOW,
            D=D_BORE))
        self.add_component("pipe", StraightPipe(
            self._medium, D=D_BORE, L=L_PIPE, epsilon=1e-5, z_in=0, z_out=0,
            n_segments=self.N, heat_port=True, multiphase="HEM"))
        for i in range(self.N):
            self.add_component(f"heat_{i}",
                               FixedHeatFlow(Q_flow=0.0, T_init=T_IN))

    def declare_equations(self):
        self.connect(self["inlet"].ports["outlet"], self["pipe"].ports["inlet"])
        wall_ports = self["pipe"].segment_wall_ports
        for i in range(self.N):
            self.connect(wall_ports[i], self[f"heat_{i}"].ports["heat"])
        return []


def run_boiler(med, label, n_ramp=N_RAMP, numba=False):
    """Ramp the wall heat and return the per-step outlet trajectory."""
    hf = float(CP.PropsSI("H", "P", P_OP, "Q", 0, FLUID))
    hg = float(CP.PropsSI("H", "P", P_OP, "Q", 1, FLUID))
    h_in = float(med.eval_h_pT(P_OP, T_IN))
    q_to_dry = M_FLOW * (hg - h_in)

    # Newton tolerance for the steady solves.  The single-phase liquid feed is
    # resolved to 1e-6, but INSIDE the dome the HEM mixture derivatives amplify
    # the TabulatedMedium's spline noise (~2.6e-8 relative on the properties
    # themselves) into a scaled-residual floor around ~4e-6 that no Newton step
    # can beat.  1e-5 sits just above that floor -- still a hard steady-state
    # convergence -- and CoolProp meets it with room to spare, so the two paths
    # are compared at the same tolerance.
    tol = 1e-5 if isinstance(med, TabulatedMedium) else 1e-6

    sys_m = Boiler(med)
    t0 = time.time()
    sys_m.instantiate(aditional_modules=med.modules,
                      max_remove_trival_passes=5, numba=numba)
    t_inst = time.time() - t0
    sys_m.initialise(n=1, tol=tol, max_iter=300, line_search=True)

    names = list(sys_m.record["vars_names"])
    i_h = next(i for i, n in enumerate(names)
               if n.endswith(f".pipe_segment_{sys_m.N - 1}.h_out"))
    i_p = next(i for i, n in enumerate(names)
               if n.endswith(f".pipe_segment_{sys_m.N - 1}.p_out"))

    def _solve_at(q, tol):
        for i in range(sys_m.N):
            sys_m[f"heat_{i}"]["Q_flow"].set_value(q / sys_m.N)
        sys_m.solve_dae_step(1.0, tol=tol, max_iter=250, line_search=True,
                             raise_on_no_convergence=True)
        sys_m.next_step()

    Q_full = np.linspace(0.0, Q_OVERDRIVE * q_to_dry, n_ramp + 1)[1:]
    q_span = Q_full[-1]
    h_out = np.empty_like(Q_full)
    p_out = np.empty_like(Q_full)
    # Continuation ramp.  Each ramp point is an independent steady state; with a
    # fine enough heat step the previous state is a good Newton seed and the
    # solve converges directly.  On a failed step we restore the last converged
    # state and BISECT the increment a bounded number of times (a Newton
    # globalisation safety net for coarse ramps).  Deep in the vapor side of the
    # dome the density collapses ~100x across the four cells and the steady
    # Jacobian becomes extremely stiff (raw cond ~1e22; the linear solver's
    # equilibration tames it to ~1e10) -- if even the finest bisection cannot
    # converge there, we stop the ramp gracefully and report how far it reached
    # rather than crashing the benchmark.
    good = sys_m.get_vars_values()
    q_prev = 0.0
    n_solve = 0
    n_done = 0
    t0 = time.time()
    for k, q in enumerate(Q_full):
        pending = [q]                      # targets to reach, nearest last
        stuck = False
        while pending:
            target = pending[-1]
            try:
                _solve_at(target, tol)
                n_solve += 1
                good = sys_m.get_vars_values()
                q_prev = target
                pending.pop()
            except Exception:
                sys_m.set_vars_values(good)          # roll back to last good
                if target - q_prev <= q_span * 2 ** -12:
                    stuck = True                     # ill-conditioned wall
                    break
                pending.append(0.5 * (q_prev + target))
        if stuck:
            print(f"  [{label}] ramp stopped at x_out~"
                  f"{(h_out[k - 1] - hf) / (hg - hf) if k else 0.0:+.2f} "
                  f"(step {k}/{n_ramp}): steady Jacobian too ill-conditioned "
                  f"to continue")
            break
        state = np.asarray(sys_m.record["state"][-1])
        h_out[k] = state[i_h]
        p_out[k] = state[i_p]
        n_done = k + 1
    t_ramp = time.time() - t0
    Q = Q_full[:n_done]
    h_out = h_out[:n_done]
    p_out = p_out[:n_done]

    T_out = np.array([float(med.eval_T_ph(p, h))
                      for p, h in zip(p_out, h_out)])
    x_out = (h_out - hf) / (hg - hf)
    # Exact steady reference: all wall heat ends up in the fluid.
    h_ref = h_in + Q / M_FLOW
    e_bal = np.max(np.abs(h_out - h_ref)) / (hg - hf)

    extra = n_solve - len(Q)
    substep = f" (+{extra} adaptive sub-steps)" if extra > 0 else ""
    print(f"  [{label}] instantiate={t_inst:.1f} s  ramp: {len(Q)} steady "
          f"points{substep} in {t_ramp:.1f} s "
          f"({1e3 * t_ramp / n_solve:.0f} ms/solve)  "
          f"x: {x_out[0]:.2f} -> {x_out[-1]:.2f}  "
          f"energy defect={e_bal:.2e} of h_fg")
    return {"label": label, "Q": Q, "h_out": h_out, "p_out": p_out,
            "T_out": T_out, "x_out": x_out, "h_ref": h_ref,
            "e_bal": e_bal, "t_ramp": t_ramp, "med": med}


def make_chart(runs, out_png):
    hf = float(CP.PropsSI("H", "P", P_OP, "Q", 0, FLUID))
    hg = float(CP.PropsSI("H", "P", P_OP, "Q", 1, FLUID))
    T_sat = float(CP.PropsSI("T", "P", P_OP, "Q", 0, FLUID))

    fig, (axT, axR) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # --- Panel L: outlet temperature vs delivered heat (boiling plateau) ----
    colors = {"coolprop": "tab:red", "tab": "tab:blue"}
    styles = {"coolprop": "-", "tab": "--"}
    for r in runs:
        axT.plot(r["Q"], r["T_out"] - 273.15, styles.get(r["label"], "-"),
                 color=colors.get(r["label"], None), lw=1.8,
                 label=f"model outlet T ({r['label']})")
    axT.axhline(T_sat - 273.15, color="0.6", lw=1.0, ls=":",
                label=f"T_sat({P_OP / 1e5:.0f} bar) = {T_sat - 273.15:.1f} C")
    q_boil = M_FLOW * (hf - runs[0]["h_ref"][0] + runs[0]["Q"][0] / M_FLOW)
    axT.set_xlabel("wall heat Q [W]")
    axT.set_ylabel("outlet temperature [\u00b0C]")
    axT.set_title("Boiling water: subcooled \u2192 dome \u2192 superheat")
    axT.legend(loc="upper left")
    axT.grid(True, alpha=0.3)

    # --- Panel R: property line rho(p=5 bar, h) across the dome ------------
    h_line = np.linspace(0.15e6, 2.95e6, 1200)
    for r in runs:
        med = r["med"]
        rho = np.asarray(med.eval_rho_ph(np.full_like(h_line, P_OP), h_line)) \
            if getattr(med.eval_rho_ph, "_hydrogen_vectorised", False) \
            else np.array([med.eval_rho_ph(P_OP, h) for h in h_line])
        axR.semilogy(h_line / 1e6, rho, styles.get(r["label"], "-"),
                     color=colors.get(r["label"], None), lw=1.6,
                     label=f"rho(5 bar, h) ({r['label']})")
    axR.axvspan(hf / 1e6, hg / 1e6, color="0.9",
                label="two-phase dome")
    axR.set_xlabel("h [MJ/kg]")
    axR.set_ylabel("density [kg/m\u00b3]")
    axR.set_title("Property line across the dome")
    axR.legend(loc="upper right")
    axR.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  chart written to {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", choices=("coolprop", "tab", "both"),
                    default="both")
    ap.add_argument("--steps", type=int, default=N_RAMP)
    ap.add_argument("--numba", action="store_true",
                    help="JIT the vectorised equation templates (tab path "
                         "only; documents which templates compile for a "
                         "dome window)")
    args = ap.parse_args()

    print(f"Boiling benchmark: water {P_OP / 1e5:.0f} bar, feed {T_IN:.0f} K, "
          f"m={M_FLOW * 1e3:.1f} g/s, {N_SEG} steady segments, "
          f"{args.steps} ramp steps to x~{Q_OVERDRIVE:.1f}")

    runs = []
    if args.medium in ("coolprop", "both"):
        runs.append(run_boiler(_coolprop_medium(), "coolprop",
                               n_ramp=args.steps))
    if args.medium in ("tab", "both"):
        runs.append(run_boiler(_tab_medium(), "tab", n_ramp=args.steps,
                               numba=args.numba))

    if len(runs) == 2:
        # Same rig, same heat: compare the two property paths directly over the
        # heat range both paths covered (one path may stop earlier).
        m = min(len(runs[0]["h_out"]), len(runs[1]["h_out"]))
        dT = np.max(np.abs(runs[0]["T_out"][:m] - runs[1]["T_out"][:m]))
        dh = np.max(np.abs(runs[0]["h_out"][:m] - runs[1]["h_out"][:m]))
        speed = runs[0]["t_ramp"] / runs[1]["t_ramp"]
        print(f"  coolprop vs tab (over {m} common points): "
              f"max |dT_out| = {dT:.3f} K, max |dh_out| = {dh:.0f} J/kg, "
              f"tab speedup = {speed:.1f}x")

    out_dir = _ROOT / "benchmarks" / "plots"
    out_dir.mkdir(exist_ok=True)
    make_chart(runs, out_dir / "boiling_multiphase.png")


if __name__ == "__main__":
    main()
