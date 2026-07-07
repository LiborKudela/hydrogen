"""Validate the `acoustic` pipe level against MEASURED fluid-hammer data.

Experiment: DLR PS-LN2-Set1 (open data, CC-BY-4.0)
    S. Klein, T. Traudt, J. Deeken, "Liquid Nitrogen Fluid Hammer:
    PS-LN2-Set1", DLR Lampoldshausen.  doi:10.5281/zenodo.15526459
    (See also Klein et al., Experiments in Fluids 64, 2023.)

Rig: two 80 l LN2 tanks joined by a straight stainless-steel pipe
(L = 9.29 m, bore 19 mm, wall 1.5 mm); a fast-closing valve at the
downstream end stops a steady flow of subcooled liquid nitrogen
(~86 K, 3.5-21 bar, 5-12 m/s).  Pressure sensors S1/S2/S3 sit at
6.46 / 47.3 / 88.2 % of the pipe length; the published traces are 10 kHz.

Model: `Pipe(dynamic="acoustic", wall_elasticity=True)` fed by a
`PressureSource` (the HP tank) and closed by an `IncompressibleValve`
whose opening REPLAYS the measured valve position (`CsvTable`).
The boundary calibration is taken from the measured steady state:

  * inlet pressure     -- linear extrapolation of the S1..S3 friction
                          gradient to x = 0 (+ dynamic head for stagnation);
  * back pressure      -- the MEASURED LP-tank pressure (0.5-0.9 bar).  That
                          is far below the LN2 vapor pressure (~2.4 bar), so
                          the real valve is deeply CHOKED (cavitating /
                          flashing): the model valve runs the ISA liquid
                          choking clamp (`p_vap`, FL=0.9,
                          FF = 0.96-0.28*sqrt(p_vap/p_crit)) with
                          `multiphase="HEM"` so its flashing outlet state is
                          dome-safe;
  * valve Kv           -- from the measured flow and the CHOKED pressure
                          drop FL^2*(p_valve - FF*p_vap);
  * valve trim         -- the DLR shutoff valve is a ball valve, whose flow
                          coefficient is strongly progressive in the ball
                          angle; the replayed position is mapped through
                          Kv*theta^trim_exp (default 2, --trim to sweep).
                          A linear trim smears the flow cutoff over the
                          whole ~20 ms stroke and visibly underpredicts the
                          peaks (worst at S1, whose peak only lives for
                          2*x_S1/a ~ 1.6 ms before the tank reflection
                          cancels it).

CAVITATION: the reflected rarefaction pulls the line to the vapor
pressure (~2.4 bar) -- the experiment cavitates (column separation).
The pipe runs with `cavitation=True`: the discrete-vapor-cavity model
(DVCM, smoothed complementarity clamp at p_vap) lets the run continue
straight through the cavitation phase and the cavity-collapse shocks,
so the benchmark compares:

  * the steady friction profile (S1/S2/S3 pre-closure pressures),
  * the Joukowsky pressure rise at each sensor,
  * the wave-front arrival times (wave speed),
  * the cavitation phase: clamp level, first-cavity duration and the
    collapse-shock timing (the DVCM clamps at p_vap(T0); the sensors
    read somewhat lower during the vapor phase, which is expected --
    see Klein et al. on the sensor response inside the cavity).

CoolProp note: the tabular backends are broken for subcooled LN2 around
85 K (BICUBIC ph-flash hits invalid cells, TTSE transport wants "four
valid corners" and its pT flash picks the vapor branch), so the full
HEOS backend is used; ~15-20 min per 0.15 s replayed (use --stop to
shorten).  Pass ``--tab`` to run on hydrogen's own `TabulatedMedium`
spline surrogate of HEOS (dome-conforming tables, analytic derivatives,
disk-cached): ~7x faster per step with <~1e-4 property error.

    python benchmarks/bench_hammer_validation.py                 # mild case
    python benchmarks/bench_hammer_validation.py --tab           # fast
    python benchmarks/bench_hammer_validation.py 20200803_8      # medium
    python benchmarks/bench_hammer_validation.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys as _sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import CoolProp.CoolProp as CP  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from hydrogen import CoolPropMedium, Model, TabulatedMedium  # noqa: E402
from hydrogen.components.control.control_components import CsvTable  # noqa: E402
from hydrogen.components.thermofluid.assemblies import Pipe  # noqa: E402
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    IncompressibleValve,
    PressureOutlet,
    PressureSource,
)

DATA = _ROOT / "benchmarks" / "hammer_data"
META = json.load(open(DATA / "meta.json"))
GEO = META["geometry"]
L_PIPE, D_BORE, WALL_E = GEO["L"], GEO["D"], GEO["wall_e"]
XS = GEO["x_sensors"]                       # {"s1": x/L, ...}
E_STEEL = 193e9                             # Pa, stainless 1.4541
T_PRE = 0.05                                # s of steady pre-closure replay
N_DEFAULT = 12


def _load_case(case: str):
    mc = META["cases"][case]
    csv = np.genfromtxt(DATA / f"PS-LN2-{case}.csv", delimiter=",",
                        names=True)
    return mc, csv


FL_VALVE = 0.9                              # liquid recovery, full-bore ball


def _boundary_calibration(med, mc):
    """Steady boundary conditions from the measured pre-closure state."""
    p1, p3 = mc["p_s1_bar"] * 1e5, mc["p_s3_bar"] * 1e5
    x1, x3 = XS["s1"], XS["s3"]
    slope = (p3 - p1) / (x3 - x1)           # friction gradient, Pa per x/L
    p_in = p1 - slope * x1                  # extrapolated pipe inlet
    p_valve = p1 + slope * (1.0 - x1)       # extrapolated pipe outlet
    T0, m0 = mc["T"], mc["m_flow"]
    A = math.pi * D_BORE ** 2 / 4.0
    h0 = float(med.eval_h_pT(p_in, T0))
    rho0 = float(med.eval_rho_ph(p_in, h0))
    w0 = m0 / (rho0 * A)
    p_vap = float(CP.PropsSI("P", "T", T0, "Q", 0, "Nitrogen"))
    p_crit = float(CP.PropsSI("PCRIT", "Nitrogen"))
    FF = 0.96 - 0.28 * math.sqrt(p_vap / p_crit)
    p_lp = mc["p_lp_bar"] * 1e5             # real (measured) LP-tank pressure
    # The steady operating point is choked (p_lp << p_vap): size Kv from the
    # ISA-clamped pressure drop, which the model valve reproduces.
    dp_choked = FL_VALVE ** 2 * (p_valve - FF * p_vap)
    dp_steady = min(p_valve - p_lp, dp_choked)
    return {
        "T0": T0, "m0": m0, "A": A, "h0": h0, "rho0": rho0, "w0": w0,
        "slope": slope, "p_in": p_in, "p_valve": p_valve, "p_vap": p_vap,
        "FF": FF, "p_lp": p_lp,
        "p_src": p_in + 0.5 * rho0 * w0 ** 2,
        "Kv": 36000.0 * m0 / math.sqrt(rho0 * dp_steady),
    }


def _tab_medium(src, cal):
    """Spline-surrogate LN2 covering the full transient envelope.

    * p window: from below the LP-tank pressure (the valve's flashing outlet)
      up to just below the critical pressure (33.96 bar).  Collapse shocks
      that overshoot the window top (severe cases) evaluate on the clamped
      linear C^1 extension -- subcooled-LN2 rho(p) is nearly linear there.
    * h window: from well below the coldest operating enthalpy up past the
      saturated-liquid line (the DVCM cavity + valve flashing graze the dome
      by design, never deeper than a few 10 kJ/kg).
    """
    p_crit = float(CP.PropsSI("PCRIT", "Nitrogen"))
    p_lo = min(0.3e5, 0.5 * cal["p_lp"])
    p_hi = 0.985 * p_crit
    h_lo = min(float(src.eval_h_pT(p_hi, max(cal["T0"] - 12.0, 64.5))),
               float(CP.PropsSI("H", "P", p_lo, "Q", 0, "Nitrogen")) - 15e3)
    h_hi = float(CP.PropsSI("H", "P", 4e5, "Q", 0, "Nitrogen")) + 40e3
    t0 = time.time()
    tab = TabulatedMedium(src, p_range=(p_lo, p_hi), h_range=(h_lo, h_hi),
                          n_p=192, n_h=192)
    print(f"  TabulatedMedium: window p=[{p_lo / 1e5:.2f}, {p_hi / 1e5:.2f}] "
          f"bar, h=[{h_lo / 1e3:.0f}, {h_hi / 1e3:.0f}] kJ/kg, built in "
          f"{time.time() - t0:.1f} s, max rel err "
          f"{max(tab.validation_max_rel_err.values()):.1e}")
    return tab


def _wave_speed(med, cal):
    """Model-consistent Korteweg wave speed at the operating point."""
    rho_p = float(med.eval_drho_ph_dp(cal["p_in"], cal["h0"]))
    rho_h = float(med.eval_drho_ph_dh(cal["p_in"], cal["h0"]))
    rho = cal["rho0"]
    rp = rho_p + rho * D_BORE / (WALL_E * E_STEEL)
    return 1.0 / math.sqrt(rp + rho_h / rho)


def build_rig(med, case: str, cal, N: int, trim_exp: float):
    class Rig(Model):
        def declare_components(self):
            self.add_component("theta", CsvTable(
                str(DATA / f"PS-LN2-{case}.csv"),
                value_column="valve_theta", time_column="time_sim"))
            self.add_component("src", PressureSource(
                med, p_source=cal["p_src"], T_source=cal["T0"], A=cal["A"]))
            self.add_component("pipe", Pipe(
                med, D=D_BORE, L=L_PIPE, epsilon=1.5e-5, z_in=0.0, z_out=0.0,
                n_segments=N, layers=[], outer_thermal="adiabatic",
                dynamic="acoustic", T_wall_init=cal["T0"], p_init=cal["p_in"],
                wall_elasticity=True, wall_E=E_STEEL, wall_e=WALL_E,
                cavitation=True, p_vap=cal["p_vap"]))
            # Choked (cavitating) ball valve: ISA liquid-choking clamp with
            # the real vapor pressure, HEM so the flashing outlet state is
            # dome-safe, progressive trim replaying the measured position.
            self.add_component("valve", IncompressibleValve(
                med, Kv=cal["Kv"], D=D_BORE, opening=1.0,
                trim_exp=trim_exp, p_vap=cal["p_vap"], FL=FL_VALVE,
                FF=cal["FF"], p_eps=100.0, multiphase="HEM"))
            self.add_component("out", PressureOutlet(
                med, p_ambient=cal["p_lp"], T_ambient=cal["T0"]))

        def declare_equations(self):
            self.connect(self["theta"].ports["y"],
                         self["valve"].ports["opening"])
            self.connect(self["src"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["valve"].ports["inlet"])
            self.connect(self["valve"].ports["outlet"],
                         self["out"].ports["inlet"])
            return []

    return Rig()


def seed_rig(sys, med, cal, N):
    """Seed pipe, valve internals, and outlet with the measured steady state
    (the library defaults are ambient air-like -- a wrong Newton basin for a
    cryogenic liquid)."""
    ch = sys["pipe"]["pipe"]
    h0, rho0, T0, w0 = cal["h0"], cal["rho0"], cal["T0"], cal["w0"]
    k0 = float(med.eval_k_ph(cal["p_in"], h0))
    for i in range(N):
        x = (i + 0.5) / N
        ch[f"hc_{i}"].value = h0
        ch[f"Tc_{i}"].value = T0
        ch[f"pc_{i}"].value = cal["p_in"] + cal["slope"] * x
        ch[f"rhoc_{i}"].value = rho0
        ch[f"kc_{i}"].value = k0
    for j in range(N + 1):
        x = j / N
        for stem, val in (("h", h0), ("p", cal["p_in"] + cal["slope"] * x),
                          ("T", T0), ("rho", rho0), ("w", w0),
                          ("M", rho0 * cal["A"] * w0)):
            key = f"{stem}_{j}"
            if key in ch.components:
                ch[key].value = val
    vch = sys["valve"]
    p_mid = 0.5 * (cal["p_valve"] + cal["p_lp"])
    T_fl = float(med.eval_T_ph(cal["p_lp"], h0))
    rho_v = float(med.eval_rho_ph(p_mid, h0))
    k_v = float(med.eval_k_ph(p_mid, h0))
    for comp, val in (("hc_0", h0), ("Tc_0", T_fl), ("pc_0", p_mid),
                      ("rhoc_0", rho_v), ("kc_0", k_v)):
        if comp in vch.components:
            vch[comp].value = val
    for j, pj in ((0, cal["p_valve"]), (1, cal["p_lp"])):
        for stem, val in (("h", h0), ("p", pj), ("T", T_fl), ("rho", rho_v),
                          ("w", cal["m0"] / (rho_v * cal["A"])),
                          ("M", cal["m0"])):
            key = f"{stem}_{j}"
            if key in vch.components:
                vch[key].value = val
    sys["out"]["h_in"].value = h0
    sys["out"]["p_in"].value = cal["p_lp"]


def _cavitation_onset(csv, cal):
    """Time at which the measured pressure first drops below the vapor
    pressure at any sensor (plot annotation)."""
    t = csv["time_sim"]
    lo = np.full_like(t, np.inf)
    for col in ("p_s1_bar", "p_s2_bar", "p_s3_bar"):
        lo = np.minimum(lo, csv[col])
    below = np.where((t > T_PRE) & (lo * 1e5 < cal["p_vap"]))[0]
    if below.size:
        return float(t[below[0]])
    return None


def _sensor_series(sys, N):
    """Model pressures interpolated to the sensor x/L positions from the
    face-pressure series."""
    t = np.asarray(sys.record["time"])
    faces = np.stack([np.asarray(sys.series(f"pipe.pipe.p_{j}"))
                      for j in range(N + 1)])
    xf = np.arange(N + 1) / N
    out = {}
    for s, xs in XS.items():
        out[s] = np.array([np.interp(xs, xf, faces[:, k])
                           for k in range(len(t))])
    return t, out


def run_case(case: str, N: int = N_DEFAULT, stop: float | None = None,
             trim_exp: float = 2.0, tab: bool = False):
    print(f"=== PS-LN2-{case} ===")
    med = CoolPropMedium("Nitrogen", disable_warnings=True, backend="HEOS",
                         scalar_cache_maxsize=4000)
    mc, csv = _load_case(case)
    cal = _boundary_calibration(med, mc)
    if tab:
        med = _tab_medium(med, cal)
    a = _wave_speed(med, cal)
    dp_jouk = cal["rho0"] * a * cal["w0"]
    t_stop = float(csv["time_sim"][-1]) if stop is None else T_PRE + stop
    t_cav = _cavitation_onset(csv, cal)
    print(f"  T0={cal['T0']:.1f} K  m0={cal['m0']:.3f} kg/s  "
          f"w0={cal['w0']:.2f} m/s  p(S3)={mc['p_s3_bar']:.2f} bar")
    print(f"  Korteweg wave speed a={a:.0f} m/s -> Joukowsky dp="
          f"{dp_jouk / 1e5:.1f} bar;  4L/a={4 * L_PIPE / a * 1e3:.1f} ms")
    print(f"  valve: choked (p_lp={cal['p_lp'] / 1e5:.2f} bar << p_vap="
          f"{cal['p_vap'] / 1e5:.2f} bar), Kv={cal['Kv']:.2f}, "
          f"trim theta^{trim_exp:g}")
    if t_cav is not None:
        print(f"  measured cavitation onset: t_sim={t_cav - T_PRE:.4f} s "
              f"after trigger (model runs through it with the DVCM cavity)")

    sys = build_rig(med, case, cal, N, trim_exp)
    sys.instantiate(aditional_modules=med.modules, cse=True, enable_blt=True,
                    enable_var_scaling=True, max_remove_trival_passes=1,
                    max_remove_duplicate_passes=5,
                    max_remove_linear_block_passes=3)
    seed_rig(sys, med, cal, N)
    sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300,
                   line_search=True, steady=True)

    dt_wave = (L_PIPE / N) / a / 4.0
    t0 = time.time()
    summary = sys.run(
        stop_time=t_stop,
        strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 0.5},
        dt_start=1e-4, dt_min=1e-10, dt_max=dt_wave, grow=1.5, shrink=0.5,
        max_retries=40, relaxation=1.0, tol=1e-8, max_iter=300,
        raise_on_no_convergence=False)
    print(f"  run: {summary['steps']} steps in {time.time() - t0:.0f} s, "
          f"stop={summary['stop_reason']}")

    t_m, p_m = _sensor_series(sys, N)
    if summary["stop_reason"] != "stop_time":
        print(f"  WARNING: run ended early at t={t_m[-1]:.4f} s "
              f"({summary['error']}); comparing up to there.")
    v_cav = np.stack([np.asarray(sys.series(f"pipe.pipe.V_cav_{i}"))
                      for i in range(N)])
    V_cell = cal["A"] * L_PIPE / N
    print(f"  max vapor-cavity fraction: {v_cav.max() / V_cell:.3f} of a "
          f"cell (cell {int(np.unravel_index(v_cav.argmax(), v_cav.shape)[0])})")

    # --- metrics on the measured time base, over the COMMON window ---------
    # (the model may end slightly before `t_stop` if its inlet cell touches
    # the dome first; compare only where both traces exist).
    tw = csv["time_sim"]
    win = tw <= t_m[-1]
    metrics = {}
    for s in ("s1", "s2", "s3"):
        meas = csv[f"p_{s}_bar"][win] * 1e5
        mod = np.interp(tw[win], t_m, p_m[s])
        rmse = float(np.sqrt(np.mean((mod - meas) ** 2)))
        p_steady = np.interp(XS[s], [0, 1], [cal["p_in"], cal["p_valve"]])
        pk_meas = float(meas.max() - mc[f"p_{s}_bar"] * 1e5)
        pk_mod = float(mod.max() - p_steady)
        metrics[s] = (rmse, pk_meas, pk_mod)
        print(f"  {s.upper()}: RMSE={rmse / 1e5:.2f} bar   peak rise (common "
              f"window) meas={pk_meas / 1e5:.1f} / model={pk_mod / 1e5:.1f}"
              f" bar ({(pk_mod / pk_meas - 1) * 100:+.0f}%)")

    # --- chart --------------------------------------------------------------
    t_end_plot = min(float(tw[-1]), t_m[-1] + 0.02)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for ax, s in zip(axes, ("s1", "s2", "s3")):
        ax.plot(tw - T_PRE, csv[f"p_{s}_bar"], color="tab:red", lw=1.2,
                label=f"measured {s.upper()} (x/L={XS[s]:.3f})")
        ax.plot(t_m - T_PRE, p_m[s] / 1e5, color="tab:blue", lw=1.8, ls="--",
                label="model (acoustic, Korteweg wall, DVCM cavity, "
                      "choked valve)")
        ax.axhline(cal["p_vap"] / 1e5, color="k", lw=0.8, ls=":",
                   label="vapor pressure (DVCM clamp)" if s == "s1" else None)
        if t_cav is not None:
            ax.axvline(t_cav - T_PRE, color="0.5", lw=0.9, ls="-.",
                       label="measured cavitation onset" if s == "s1"
                       else None)
        ax.set_xlim(-0.02, t_end_plot - T_PRE)
        ax.set_ylabel("pressure [bar]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        rmse = metrics[s][0]
        ax.text(0.015, 0.93, f"window RMSE = {rmse / 1e5:.2f} bar",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    axes[0].set_title(
        f"DLR PS-LN2-{case}: LN2 water hammer, measured vs acoustic level\n"
        f"(m\u0307={cal['m0']:.2f} kg/s, T={cal['T0']:.1f} K, "
        f"a_Korteweg={a:.0f} m/s, Joukowsky dp={dp_jouk / 1e5:.1f} bar)")
    axes[-1].set_xlabel("time since valve trigger [s]")
    out_dir = _ROOT / "benchmarks" / "plots"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / f"hammer_valid_{case}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  chart written to {out_png}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="20200806_13",
                    choices=sorted(META["cases"]))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--stop", type=float, default=None,
                    help="simulated seconds after the valve trigger "
                         "(default: the full measured trace)")
    ap.add_argument("--trim", type=float, default=2.0,
                    help="valve trim exponent (Kv_eff = Kv*theta^n; "
                         "1 = linear, 2-3 ~ ball valve)")
    ap.add_argument("--tab", action="store_true",
                    help="run on a TabulatedMedium spline surrogate of the "
                         "HEOS backend (vectorised, ~an order of magnitude "
                         "faster; tables disk-cached after the first build)")
    args = ap.parse_args()
    cases = sorted(META["cases"]) if args.all else [args.case]
    for c in cases:
        run_case(c, N=args.N, stop=args.stop, trim_exp=args.trim,
                 tab=args.tab)
