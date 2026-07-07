"""Validate the advective pipe model: advection + diffusion/dispersion.

Two complementary validations are provided:

1. DISPERSION (``--mode dispersion``) -- a controlled numerical verification of
   the advective level's *axial diffusion* operator.  An adiabatic pipe carries
   water at a steady velocity with a known, imposed constant effective
   diffusivity ``D`` (``Pipe(dispersion="constant", D_axial=D)``); the inlet
   temperature is stepped and the modelled outlet response is overlaid on the
   analytical advection-dispersion (Ogata-Banks) solution.  A grid sweep shows
   convergence to the analytical curve as the cell-Peclet number drops, i.e.
   the model reproduces the prescribed dispersion (not just numerical smearing).

2. EXPERIMENT (``--mode experiment``) -- validation against the University of
   Liege (ULg) single-pipe transient test bench: water at a fixed mass flow
   through a ~39 m insulated steel pipe with a stepped/ramped inlet
   temperature.  The model is driven by the *measured* inlet temperature
   (replayed by `control.CsvTable` into a `TemperatureInlet`) and its predicted
   outlet temperature is overlaid on the measured one.  Here the smearing is set
   by advection + pipe-wall thermal mass (turbulent flow, Re ~ 1e4, so the
   axial diffusion is the conduction-only default).

    python benchmarks/bench_advection_validation.py --mode dispersion
    python benchmarks/bench_advection_validation.py --mode experiment PipeDataULg151202
    python benchmarks/bench_advection_validation.py --mode all

Pass ``--tab`` to serve the water properties from hydrogen's own
`TabulatedMedium` spline surrogate instead of CoolProp directly (checks the
surrogate on a plain single-phase liquid problem and measures the speedup).

Datasets (deg C, kg/s), from Wetter et al. / ThermoCycle ULg test bench:
    PipeDataULg151202     18 -> 52,  m=0.589
    PipeDataULg151204_1   step up then down, m=1.618
    PipeDataULg151204_2   step up then down, m=1.251
    PipeDataULg151204_4   28 -> 60,  m=1.257
    PipeDataULg160104_2   15 -> 35,  m=0.249  (low flow, long)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc, erfcx

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hydrogen import CoolPropMedium, Model, TabulatedMedium  # noqa: E402
from hydrogen.components.control.control_components import (  # noqa: E402
    CsvTable,
    Step,
)
from hydrogen.components.materials import WallMaterial  # noqa: E402
from hydrogen.components.thermofluid.assemblies import Pipe, WallLayer  # noqa: E402
from hydrogen.components.thermofluid.flow import (  # noqa: E402
    PressureOutlet,
    TemperatureInlet,
)

# --- ULg single-steel-pipe test-bench geometry / materials -----------------
# Source: Wetter et al., "Dynamic equation-based thermo-hydraulic pipe model"
# (the dataset's reference) + Modelica Buildings PipeDataULg records.
D_INNER = 0.05248        # m   inner (flow) diameter
WALL_TH = 0.00391        # m   steel wall thickness (OD 60.3 mm)
INS_TH = 0.013           # m   Tubolit insulation thickness
L_PIPE = 39.0            # m   pipe length
H_EXT = 4.0              # W/m^2/K  outer natural convection
T_AMB_C = 18.0           # deg C    ambient near the pipe
P_OP = 2.0e5             # Pa   (liquid water; pressure is thermally inert)

# Wall material properties match the thesis port (validation_IBPSA.jl):
# steel cp=500, rho=7800, k=50; insulation rho=40, cp=1200, k=0.04.
STEEL = WallMaterial(name="ULg carbon steel", rho=7800.0, cp=500.0, k=50.0)
# Tubolit closed-cell foam: a near-massless conduction resistance (k ~ 0.04).
INSULATION = WallMaterial(name="Tubolit insulation", rho=40.0, cp=1200.0,
                          k=0.04)

WATER = CoolPropMedium("Water", disable_warnings=True, backend="BICUBIC&HEOS",
                       scalar_cache_maxsize=4000)


def use_tabulated_water():
    """Swap the module-global WATER for a `TabulatedMedium` spline surrogate
    covering the whole liquid envelope of both validation modes (5-90 degC
    at 0.5-5 bar: single-phase, well clear of the dome)."""
    global WATER
    import time as _time
    src = CoolPropMedium("Water", disable_warnings=True, backend="HEOS",
                         scalar_cache_maxsize=4000)
    # 1 bar keeps 90 degC liquid below the saturated-liquid line everywhere
    # in the window (h_l(1 bar) ~ 417 kJ/kg > h(90 degC) ~ 377 kJ/kg), so the
    # table is purely single-phase.
    p_lo, p_hi = 1.0e5, 5.0e5
    h_lo = float(src.eval_h_pT(p_hi, 273.15 + 5.0))
    h_hi = float(src.eval_h_pT(p_hi, 273.15 + 90.0))
    t0 = _time.time()
    WATER = TabulatedMedium(src, p_range=(p_lo, p_hi), h_range=(h_lo, h_hi),
                            n_p=96, n_h=192)
    print(f"TabulatedMedium(Water): p=[{p_lo / 1e5:.1f}, {p_hi / 1e5:.1f}] bar,"
          f" h=[{h_lo / 1e3:.0f}, {h_hi / 1e3:.0f}] kJ/kg, built in "
          f"{_time.time() - t0:.1f} s, max rel err "
          f"{max(WATER.validation_max_rel_err.values()):.1e}")


def load_dataset(path: Path):
    """Return dict with time[s], m_flow[kg/s] (mean), and the four temperature
    columns in deg C plus the initial temperature."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    arr = lambda key: np.array([float(r[key]) for r in rows])
    t = arr("time")
    return {
        "t": t,
        "m_flow": float(np.mean(arr("m_flow"))),
        "water_inlet": arr("water_inlet"),
        "water_outlet": arr("water_outlet"),
        # The pipe starts full of water that has been resting at the
        # pre-transient resident temperature -- that is the initial *outlet*
        # reading, not the inlet's first sample (the inlet is already being
        # driven at t=0).
        "T0_C": float(arr("water_outlet")[0]),
        "t_end": float(t[-1]),
        "path": path,
    }


def seed_pipe_temperature(pipe, p0, T0):
    """Override the advective channel's fluid-cell initial state to a uniform
    `T0` (the experiment starts with the pipe full of water at `T0`)."""
    ch = pipe["pipe"]               # the inner SegmentedChannel
    N = ch.N
    h0 = float(WATER.eval_h_pT(p0, T0))
    rho0 = float(WATER.eval_rho_ph(p0, h0))
    k0 = float(WATER.eval_k_ph(p0, h0))
    A = math.pi * ch.D ** 2 / 4.0
    V_cell = A * (ch.L / N)         # count == 1
    for i in range(N):
        ch[f"hc_{i}"].value = h0
        ch[f"Tc_{i}"].value = T0
        ch[f"pc_{i}"].value = p0
        ch[f"rhoc_{i}"].value = rho0
        ch[f"kc_{i}"].value = k0
    for j in range(N + 1):
        if f"h_{j}" in ch.components:
            ch[f"h_{j}"].value = h0
        if f"p_{j}" in ch.components:
            ch[f"p_{j}"].value = p0
        if f"T_{j}" in ch.components:
            ch[f"T_{j}"].value = T0


def build_model(ds: dict, N: int, dispersion: str = "conduction"):
    T0 = ds["T0_C"] + 273.15
    m_flow = ds["m_flow"]
    csv_path = str(ds["path"])

    class ULgPipe(Model):
        def declare_components(self):
            self.add_component("tin", CsvTable(
                csv_path, value_column="water_inlet", value_offset=273.15,
                unit="K"))
            self.add_component("inlet", TemperatureInlet(
                WATER, m_flow=m_flow, p_init=P_OP, T_init=T0))
            self.add_component("pipe", Pipe(
                WATER, D=D_INNER, L=L_PIPE, epsilon=1e-5, z_in=0.0, z_out=0.0,
                n_segments=N,
                layers=[
                    WallLayer(material=STEEL, thickness=WALL_TH, dynamic=True,
                              capacity_split="fem_logmean"),
                    WallLayer(material=INSULATION, thickness=INS_TH, dynamic=True,
                              capacity_split="fem_logmean"),
                ],
                outer_thermal="convective", h_ext=H_EXT, T_ext=T_AMB_C + 273.15,
                dynamic="advective", dispersion=dispersion,
                T_wall_init=T0, p_init=P_OP))
            self.add_component("outlet", PressureOutlet(
                WATER, p_ambient=P_OP, T_ambient=T0))

        def declare_equations(self):
            self.connect(self["tin"].ports["y"], self["inlet"].ports["T_set"])
            self.connect(self["inlet"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["outlet"].ports["inlet"])
            return []

    return ULgPipe()


def run_case(name: str, N: int, dispersion: str = "conduction",
             numba: bool = False):
    import time as _time
    path = _ROOT / "benchmarks" / f"{name}.csv"
    ds = load_dataset(path)
    T0 = ds["T0_C"] + 273.15
    print(f"Dataset {name}: m_flow={ds['m_flow']:.4g} kg/s  T0={ds['T0_C']:.1f} C"
          f"  t_end={ds['t_end']:.0f} s  N={N}  dispersion={dispersion}"
          f"  numba={numba}")
    w = ds["m_flow"] / (WATER.eval_rho_ph(P_OP, WATER.eval_h_pT(P_OP, T0))
                        * math.pi * D_INNER ** 2 / 4)
    print(f"  mean velocity ~ {w:.3f} m/s, residence ~ {L_PIPE / w:.0f} s")

    m = build_model(ds, N, dispersion=dispersion)
    t_inst = _time.time()
    m.instantiate(aditional_modules=WATER.modules, cse=True, enable_blt=True,
                  enable_var_scaling=True, max_remove_trival_passes=1,
                  max_remove_duplicate_passes=5, max_remove_linear_block_passes=3,
                  numba=numba)
    t_inst = _time.time() - t_inst
    seed_pipe_temperature(m["pipe"], P_OP, T0)
    t_init = _time.time()
    m.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300)
    t_init = _time.time() - t_init

    t_run = _time.time()
    summary = m.run(
        stop_time=ds["t_end"],
        strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 1.0},
        dt_start=0.05, dt_min=1e-9, dt_max=ds["t_end"] / 40.0, grow=1.5,
        shrink=0.5, max_retries=40, relaxation=1.0, tol=1e-8, max_iter=300,
        raise_on_no_convergence=False)
    t_run = _time.time() - t_run
    print(f"  run: {summary['steps']} steps, stop={summary['stop_reason']}")
    print(f"  TIMING N={N} numba={numba}: instantiate={t_inst:.1f} s  "
          f"initialise={t_init:.1f} s  run={t_run:.1f} s  "
          f"({1e3 * t_run / max(1, summary['steps']):.1f} ms/step)")

    # Model outlet water temperature: convert outlet face enthalpy -> T.
    t_model = np.asarray(m.record["time"])
    h_out = np.asarray(m.series(f"pipe.pipe.h_{N}"))
    p_out = np.asarray(m.series(f"pipe.pipe.p_{N}"))
    T_out_model = np.array([WATER.eval_T_ph(p_out[k], h_out[k]) - 273.15
                            for k in range(len(t_model))])

    # --- chart: measured inlet/outlet vs model outlet ----------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(ds["t"], ds["water_inlet"], color="tab:gray", lw=1.4,
            label="measured inlet (BC)")
    ax.plot(ds["t"], ds["water_outlet"], color="tab:red", lw=2.0,
            label="measured outlet")
    ax.plot(t_model, T_out_model, color="tab:blue", lw=2.0, ls="--",
            label="model outlet (advective)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("temperature [\u00b0C]")
    ax.set_title(f"ULg pipe transient \u2014 {name}  "
                 f"(m\u0307={ds['m_flow']:.3g} kg/s, N={N})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Outlet-only error metric on the measured time grid.
    T_out_interp = np.interp(ds["t"], t_model, T_out_model)
    rmse = float(np.sqrt(np.mean((T_out_interp - ds["water_outlet"]) ** 2)))
    ax.text(0.98, 0.04, f"outlet RMSE = {rmse:.2f} \u00b0C",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))

    out_dir = _ROOT / "benchmarks" / "plots"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / f"adv_valid_{name}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  outlet RMSE = {rmse:.2f} C -> chart written to {out_png}")
    return rmse


# ===========================================================================
# 2. Dispersion validation: imposed constant D vs analytical advection-dispersion
# ===========================================================================
# Controlled numerical test pipe (clean geometry, properties ~constant).
L_DISP = 20.0            # m     test-pipe length
D_DISP = 0.05            # m     bore diameter
W_DISP = 0.20            # m/s   target mean velocity
D_EFF_DISP = 0.05        # m^2/s imposed effective axial diffusivity (>> molecular
                         #       alpha~1.5e-7, so prescribed dispersion dominates)
T0_DISP = 293.15         # K     baseline temperature (= property std state)
DT_DISP = 10.0           # K     inlet step height (small -> ~constant properties)
T_STEP = 20.0            # s     time of the inlet step
P_DISP = 2.0e5           # Pa


def ogata_banks(x, t, w, D, T0, dT):
    """Analytical 1D advection-dispersion (Ogata & Banks 1961) outlet response
    to a step inlet of size `dT` on a semi-infinite domain::

        T = T0 + dT/2 [ erfc((x-w t)/(2 sqrt(D t)))
                        + exp(w x/D) erfc((x+w t)/(2 sqrt(D t))) ]

    The second term is evaluated in the numerically stable form
    ``exp(-a^2) * erfcx(b)`` (since ``w x/D - b^2 == -a^2``), avoiding the
    ``exp(w x/D)`` overflow at high Peclet number."""
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, T0)
    m = t > 0.0
    tt = t[m]
    s = 2.0 * np.sqrt(D * tt)
    a = (x - w * tt) / s
    b = (x + w * tt) / s
    out[m] = T0 + 0.5 * dT * (erfc(a) + np.exp(-a * a) * erfcx(b))
    return out


def build_dispersion_model(N, D_axial, m_flow, scheme="U2D1"):
    class DispPipe(Model):
        def declare_components(self):
            self.add_component("tin", Step(
                offset=T0_DISP, height=DT_DISP, start_time=T_STEP, unit="K"))
            self.add_component("inlet", TemperatureInlet(
                WATER, m_flow=m_flow, p_init=P_DISP, T_init=T0_DISP))
            self.add_component("pipe", Pipe(
                WATER, D=D_DISP, L=L_DISP, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=N, layers=[],            # bare, adiabatic pipe
                outer_thermal="adiabatic",
                dynamic="advective", dispersion="constant", D_axial=D_axial,
                advection_scheme=scheme, T_wall_init=T0_DISP, p_init=P_DISP))
            self.add_component("outlet", PressureOutlet(
                WATER, p_ambient=P_DISP, T_ambient=T0_DISP))

        def declare_equations(self):
            self.connect(self["tin"].ports["y"], self["inlet"].ports["T_set"])
            self.connect(self["inlet"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["outlet"].ports["inlet"])
            return []

    return DispPipe()


def run_dispersion_model(N, D_axial, scheme="U2D1", t_end=180.0):
    """Run the adiabatic constant-D pipe; return (t, T_outlet[C], w_actual)."""
    A = math.pi * D_DISP ** 2 / 4.0
    h0 = WATER.eval_h_pT(P_DISP, T0_DISP)
    rho0 = WATER.eval_rho_ph(P_DISP, h0)
    m_flow = rho0 * A * W_DISP

    m = build_dispersion_model(N, D_axial, m_flow, scheme)
    m.instantiate(aditional_modules=WATER.modules, cse=True, enable_blt=True,
                  enable_var_scaling=True, max_remove_trival_passes=1,
                  max_remove_duplicate_passes=5, max_remove_linear_block_passes=3)
    seed_pipe_temperature(m["pipe"], P_DISP, T0_DISP)
    m.initialise(n=1, relaxation=1.0, tol=1e-9, max_iter=300)
    m.run(stop_time=t_end,
          strategy={"name": "tr_bdf2", "tol_local": 5e-4, "atol": 0.5},
          dt_start=0.02, dt_min=1e-9, dt_max=1.0, grow=1.4, shrink=0.5,
          max_retries=40, relaxation=1.0, tol=1e-9, max_iter=300,
          raise_on_no_convergence=False)

    t = np.asarray(m.record["time"])
    h_out = np.asarray(m.series(f"pipe.pipe.h_{N}"))
    p_out = np.asarray(m.series(f"pipe.pipe.p_{N}"))
    T_out = np.array([WATER.eval_T_ph(p_out[k], h_out[k]) - 273.15
                      for k in range(len(t))])
    return t, T_out, W_DISP


def run_dispersion_validation(N=40):
    print(f"Dispersion validation: L={L_DISP} m, D={D_DISP} m, "
          f"w~{W_DISP} m/s, imposed D_eff={D_EFF_DISP} m^2/s")
    Pe_cell = W_DISP * (L_DISP / N) / D_EFF_DISP
    Pe_tot = W_DISP * L_DISP / D_EFF_DISP
    print(f"  Pe_total = {Pe_tot:.0f}, Pe_cell(N={N}) = {Pe_cell:.2f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Panel L: model (imposed D) vs analytical, plus conduction-only -----
    t_ref = np.linspace(0.0, 180.0, 1200)
    T_ana = ogata_banks(L_DISP, t_ref - T_STEP, W_DISP, D_EFF_DISP,
                        T0_DISP - 273.15, DT_DISP)
    axL.plot(t_ref, T_ana, color="k", lw=2.4, label="analytical ADE (Ogata-Banks)")

    t_m, T_m, _ = run_dispersion_model(N, D_EFF_DISP)
    axL.plot(t_m, T_m, color="tab:blue", lw=1.8, ls="--",
             label=f"model, D imposed (N={N})")

    t_c, T_c, _ = run_dispersion_model(N, 0.0)
    axL.plot(t_c, T_c, color="tab:green", lw=1.4, ls=":",
             label=f"model, D=0 (advection only, N={N})")

    T_m_on_ref = np.interp(t_ref, t_m, T_m)
    rmse = float(np.sqrt(np.mean((T_m_on_ref - T_ana) ** 2)))
    axL.axvline(T_STEP, color="0.8", lw=1, ls="-")
    axL.set_xlabel("time [s]")
    axL.set_ylabel("outlet temperature [\u00b0C]")
    axL.set_title(f"Advection-dispersion vs analytical  (Pe_cell={Pe_cell:.1f})")
    axL.legend(loc="lower right")
    axL.grid(True, alpha=0.3)
    axL.text(0.03, 0.95, f"model vs analytical RMSE = {rmse:.3f} \u00b0C",
             transform=axL.transAxes, ha="left", va="top",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    # --- Panel R: grid convergence to the analytical curve ------------------
    axR.plot(t_ref, T_ana, color="k", lw=2.4, label="analytical ADE")
    rmses = {}
    for Ni, col in zip((10, 20, 40, 80), ("tab:red", "tab:orange",
                                          "tab:blue", "tab:purple")):
        ti, Ti, _ = run_dispersion_model(Ni, D_EFF_DISP)
        axR.plot(ti, Ti, color=col, lw=1.3, ls="--", label=f"model N={Ni}")
        rmses[Ni] = float(np.sqrt(np.mean(
            (np.interp(t_ref, ti, Ti) - T_ana) ** 2)))
    axR.set_xlabel("time [s]")
    axR.set_ylabel("outlet temperature [\u00b0C]")
    axR.set_title("Grid convergence (cell-Peclet -> 0)")
    axR.legend(loc="lower right")
    axR.grid(True, alpha=0.3)
    conv = "  ".join(f"N{n}:{r:.2f}" for n, r in rmses.items())
    axR.text(0.03, 0.95, f"RMSE [\u00b0C]  {conv}", transform=axR.transAxes,
             ha="left", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    out_dir = _ROOT / "benchmarks" / "plots"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "adv_valid_dispersion.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  N={N} RMSE vs analytical = {rmse:.3f} C; "
          f"convergence {rmses} -> chart written to {out_png}")
    return rmse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("experiment", "dispersion", "all"),
                    default="dispersion")
    ap.add_argument("dataset", nargs="?", default="PipeDataULg151202")
    ap.add_argument("--N", type=int, default=25)
    ap.add_argument("--dispersion", default="conduction",
                    choices=("conduction", "taylor_aris", "turbulent"))
    ap.add_argument("--tab", action="store_true",
                    help="serve water properties from a TabulatedMedium "
                         "spline surrogate instead of CoolProp directly")
    ap.add_argument("--numba", action="store_true",
                    help="JIT-compile the vectorised equation templates with "
                         "numba (implies --tab: the CoolProp callbacks can't "
                         "be compiled, the tabulated njit twins can)")
    args = ap.parse_args()

    if args.tab or args.numba:
        use_tabulated_water()
    if args.mode in ("dispersion", "all"):
        run_dispersion_validation(N=40)
    if args.mode in ("experiment", "all"):
        run_case(args.dataset, args.N, dispersion=args.dispersion,
                 numba=args.numba)
