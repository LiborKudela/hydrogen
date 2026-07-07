"""Extract trimmed CSV traces from the DLR PS-LN2-Set1 fluid-hammer dataset.

Source (open data, CC-BY-4.0):
    S. Klein, T. Traudt, J. Deeken, "Liquid Nitrogen Fluid Hammer:
    PS-LN2-Set1", DLR Institute of Space Propulsion, Lampoldshausen.
    Zenodo, 2025.  https://doi.org/10.5281/zenodo.15526459

Rig (see the dataset's overview PDF and Klein et al., Exp. Fluids 64, 2023):
    Two 80 l LN2 tanks (HP upstream, LP downstream) joined by a straight
    stainless-steel (1.4541) pipe with a 1.5-turn spiral:
        length L = 9.29 m, bore d_i = 19 mm, wall e = 1.5 mm.
    A fast-closing valve at the DOWNSTREAM end (in front of the LP tank)
    stops a steady flow; the hammer wave then bounces between the valve and
    the HP tank.  Pressure/temperature sensors sit at S1 = 6.46 %,
    S2 = 47.3 %, S3 = 88.2 % of the pipe length (S3 closest to the valve).
    Sensor accuracy: dP = +/-0.552 bar, dT = +/-3.7 K.

This script reads the per-case Excel files of the original archive and
writes, for each requested case:

  * ``PS-LN2-<case>.csv`` -- a single time-aligned table on the 10 kHz
    pressure clock, trimmed to ``T_PRE .. T_POST`` around the valve trigger
    (t = 0), with columns::

        time        [s]   dataset time (t=0 = closure trigger)
        time_sim    [s]   = time - T_PRE, so a simulation started from the
                          steady pre-closure state at t_sim = 0 lines up
        p_s1_bar    [bar] pressure at S1 (x/L = 0.0646)  -- sensor PS402
        p_s2_bar    [bar] pressure at S2 (x/L = 0.473)   -- sensor PS405
        p_s3_bar    [bar] pressure at S3 (x/L = 0.882)   -- sensor PS408
        valve_theta [-]   normalised valve opening 1 -> 0 (from valve_pos,
                          latched at 0 once fully closed)

  * an entry in ``meta.json`` with the pre-closure steady state (mass flow,
    Coriolis density, temperature, tank pressures) used by the benchmark to
    set up boundary conditions.

Usage (needs the extracted Zenodo archive):
    python benchmarks/hammer_data/extract_ps_ln2.py /path/to/PS-LN2-Set1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

#: Cases to extract: mild / medium / strong hammer (see the dataset table).
CASES = ("20200806_13", "20200803_8", "20200803_15")

T_PRE = -0.05            # s  window start (steady flow, for calibration)
T_POST = 0.50            # s  window end (several wave periods + cavitation)

# Fixed rig geometry from the dataset overview.
GEOMETRY = {
    "L": 9.29,           # m   pipe length
    "D": 0.019,          # m   inner diameter
    "wall_e": 0.0015,    # m   wall thickness
    "x_sensors": {"s1": 0.0646, "s2": 0.473, "s3": 0.882},   # x/L
    "wall_material": "stainless steel 1.4541",
}


def _steady(t, x, t_lo=-3.0, t_hi=-0.2):
    """Mean of signal `x` over the pre-closure window [t_lo, t_hi] (s)."""
    m = (t > t_lo) & (t < t_hi)
    return float(np.mean(x[m]))


def extract_case(root: Path, case: str) -> dict:
    base = root / f"PS-LN2-{case}" / f"PS-LN2-{case}-"
    p = pd.read_excel(f"{base}Pressure.xlsx")
    v = pd.read_excel(f"{base}Valve.xlsx")
    c = pd.read_excel(f"{base}Coriolis.xlsx")
    T = pd.read_excel(f"{base}Temperature.xlsx")

    tp = p["Time_p"].to_numpy() / 1e3          # ms -> s
    tv = v["Time_valve"].to_numpy() / 1e3
    tc = c["Time_cor"].to_numpy() / 1e3
    tt = T["Time_temp"].to_numpy() / 1e3

    # --- normalised valve opening ------------------------------------------
    pos = v["valve_pos"].to_numpy()
    pos_open = _steady(tv, pos)
    pos_closed = float(np.mean(pos[(tv > 0.5) & (tv < 1.5)]))
    # Keep a tiny leakage floor: at theta = 0 exactly, a Kv valve's zero-flow
    # state makes the enthalpy equation degenerate (m * (h_in - h_out) = 0
    # pins nothing) and the model Jacobian goes singular.  theta = 1e-3 is a
    # ~0.1 % leak -- immaterial next to the hammer flows.
    THETA_FLOOR = 1e-3
    theta = np.clip((pos - pos_closed) / (pos_open - pos_closed),
                    THETA_FLOOR, 1.0)
    # Latch fully-closed AFTER the trigger: position noise (~1 %) must not
    # re-open the valve.  (Only look at t >= 0 -- the raw record starts 10 s
    # early, where the valve sat closed before the run was set up.)
    closed = np.where((tv >= 0.0) & (theta < 0.01))[0]
    if closed.size:
        theta[closed[0]:] = THETA_FLOOR

    # --- resample everything on the trimmed pressure clock -----------------
    win = (tp >= T_PRE) & (tp <= T_POST)
    t_out = tp[win]
    table = pd.DataFrame({
        "time": np.round(t_out, 7),
        "time_sim": np.round(t_out - T_PRE, 7),
        "p_s1_bar": p["PS402"].to_numpy()[win],
        "p_s2_bar": p["PS405"].to_numpy()[win],
        "p_s3_bar": p["PS408"].to_numpy()[win],
        "valve_theta": np.interp(t_out, tv, theta),
    })
    out_csv = HERE / f"PS-LN2-{case}.csv"
    table.to_csv(out_csv, index=False, float_format="%.6g")

    meta = {
        "m_flow": _steady(tc, c["mflow_cor"].to_numpy()),        # kg/s
        "rho_cor": _steady(tc, c["rho_cor"].to_numpy()),          # kg/m3
        "T": _steady(tt, T["Ti407"].to_numpy()),                  # K
        "p_hp_bar": _steady(tp, p["PSHP"].to_numpy()),            # bar
        "p_lp_bar": _steady(tp, p["PSLP"].to_numpy()),            # bar
        "p_s1_bar": _steady(tp, p["PS402"].to_numpy()),
        "p_s2_bar": _steady(tp, p["PS405"].to_numpy()),
        "p_s3_bar": _steady(tp, p["PS408"].to_numpy()),
    }
    print(f"{case}: m={meta['m_flow']:.3f} kg/s  T={meta['T']:.1f} K  "
          f"p(S1..S3)={meta['p_s1_bar']:.2f}/{meta['p_s2_bar']:.2f}/"
          f"{meta['p_s3_bar']:.2f} bar -> {out_csv.name}")
    return meta


def main(root: Path):
    meta = {
        "source": "DLR PS-LN2-Set1, doi:10.5281/zenodo.15526459 (CC-BY-4.0)",
        "geometry": GEOMETRY,
        "cases": {case: extract_case(root, case) for case in CASES},
    }
    with open(HERE / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {HERE / 'meta.json'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
