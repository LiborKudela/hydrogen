"""Boil water with the fluid library: single-phase closures vs the smooth HEM.

Takes subcooled liquid water, heats a pipe from the outside, and ramps the wall
heat until the fluid boils into steam -- solved twice, once with the default
single-phase property closures and once with `multiphase="HEM"`, so the two can
be compared head to head.

It has three parts:

  PART 1 -- Medium probe (the "why").
      Sweep specific enthalpy along an isobar from subcooled liquid, through the
      two-phase dome, into superheated steam, printing the CoolProp `(p, h)`
      properties AND the partial derivatives the symbolic Jacobian is built
      from -- for BOTH the single-phase partials and the smooth-HEM partials.
      This exposes what breaks Newton in the dome and what the HEM fixes:
        * T(p, h) goes FLAT  (dT/dh -> 0): temperature loses all sensitivity
          to enthalpy at constant pressure (correct -- T = Tsat(p) in the dome).
        * rho(p, h) falls off a CLIFF at the saturation line (x = 0): a ~900 ->
          ~15 kg/m^3 drop over a sliver of enthalpy.
        * the SINGLE-PHASE analytic drho/dh COLLAPSES to ~0 *inside* the dome
          (CoolProp reports a near-zero slope) and is DISCONTINUOUS at the
          saturation boundary -- so the Jacobian no longer matches how the
          residual actually moves.  The HEM column instead shows a consistent,
          continuous drho/dh (a central difference of the same values).

  PART 2 -- Heated boiler pipe, multiphase="single".
      AmbientInlet(liquid water) -> StraightPipe(heat_port=True) -> open outlet,
      with a FixedHeatFlow on every segment wall.  Ramp the wall heat and watch
      the single-phase Newton solve go singular the moment the outlet touches
      the saturation line -- exactly the failure Part 1 predicts.

  PART 3 -- The same pipe, multiphase="HEM".
      Identical geometry/boundary conditions, but the segments use the smooth
      homogeneous-equilibrium property variants (CoolProp values + smoothed,
      consistent finite-difference partials), solved with a backtracking line
      search (`line_search=True`).  At FULL Newton steps the very same boiler
      now marches cleanly from subcooled liquid, through the two-phase dome,
      into superheated steam -- no events, no regime switching, no hand-tuned
      damping (the line search backs off only at the density cliff).

Run with `python tutorials/two_phase_boiler.py` from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import CoolProp.CoolProp as CP  # noqa: E402
import numpy as np  # noqa: E402

from hydrogen import CoolPropMedium, Model, NewtonConvergenceFailure  # noqa: E402
from hydrogen.components.thermofluid.flow import AmbientInlet, StraightPipe  # noqa: E402
from hydrogen.components.thermofluid.walls import FixedHeatFlow  # noqa: E402

# --- operating point -------------------------------------------------------
FLUID = "Water"
P_OP = 5.0e5             # Pa     operating pressure (set by the inlet)
T_IN = 300.0            # K      subcooled liquid inlet (~27 C, well below Tsat)
M_FLOW = 1.0e-3         # kg/s   mass flow (1 g/s)
D = 0.01               # m      pipe inner diameter
L = 1.0                # m      pipe length
N = 6                   # axial segments

# Tabular backend: ~50-60x faster per property call than HEOS and noticeably
# smoother through the saturation lines (it interpolates a pre-built (p, h)
# table), which keeps both the value and the finite-difference HEM partials
# well-behaved.  Engineering-grade accuracy for this kind of study.
WATER = CoolPropMedium(FLUID, disable_warnings=True, backend="BICUBIC&HEOS")


# ---------------------------------------------------------------------------
# PART 1 -- why the (p, h) property layer breaks in the dome
# ---------------------------------------------------------------------------


def probe_medium():
    hf = CP.PropsSI("H", "P", P_OP, "Q", 0, FLUID)   # saturated liquid enthalpy
    hg = CP.PropsSI("H", "P", P_OP, "Q", 1, FLUID)   # saturated vapour enthalpy
    tsat = CP.PropsSI("T", "P", P_OP, "Q", 0, FLUID)
    h_in = WATER.eval_h_pT(P_OP, T_IN)

    print("=" * 78)
    print(f"PART 1 -- {FLUID} property probe along the p = {P_OP/1e5:.2f} bar isobar")
    print("=" * 78)
    print(f"  inlet liquid:   h(p,{T_IN:.0f}K) = {h_in/1e3:8.1f} kJ/kg")
    print(f"  saturation:     Tsat = {tsat:.2f} K ({tsat-273.15:.2f} C),  "
          f"h_f = {hf/1e3:.1f} kJ/kg,  h_g = {hg/1e3:.1f} kJ/kg,  "
          f"latent = {(hg-hf)/1e3:.1f} kJ/kg")
    print()
    print("  Sweep h from subcooled liquid -> two-phase dome -> superheated steam.")
    print("  Watch T go flat, rho fall off a cliff, and the Jacobian slopes")
    print("  (drho/dh, dT/dh) collapse to ~0 and jump at the saturation line.\n")

    # Sample enthalpies that bracket the dome (x in [0,1]) plus liquid/vapour.
    def at_quality(x):
        return hf + x * (hg - hf)

    samples = [
        ("liquid",  h_in),
        ("liquid",  0.5 * (h_in + hf)),
        ("x=-0.01", hf - 0.01 * (hg - hf)),
        ("x=0.00",  at_quality(0.0)),
        ("x=0.01",  at_quality(0.01)),
        ("x=0.10",  at_quality(0.10)),
        ("x=0.50",  at_quality(0.50)),
        ("x=0.90",  at_quality(0.90)),
        ("x=1.00",  at_quality(1.00)),
        ("x=1.01",  at_quality(1.01)),
        ("steam",   hg + 0.30 * (hg - hf)),
    ]

    # Two drho/dh columns side by side: the single-phase analytic partial
    # (what the default closures feed the Jacobian) vs the smooth-HEM central
    # difference (what multiphase="HEM" feeds it).
    hdr = (f"{'region':>9}  {'h[kJ/kg]':>9}  {'T[K]':>8}  {'rho[kg/m3]':>10}  "
           f"{'mu[uPa.s]':>9}  {'drho/dh(1ph)':>12}  {'drho/dh(HEM)':>12}  {'dT/dh':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    def _safe(fn):
        # The single-phase analytic partials are simply not defined in the
        # dome: HEOS returns an inconsistent ~0, the tabular backend raises
        # outright ("...are two-phase; cannot use single-phase derivatives").
        # Either way it is a non-result -- report it as such.
        try:
            return fn(P_OP, h)
        except Exception:
            return float("nan")

    def _fmt(v, w=12):
        return f"{'n/a':>{w}}" if v != v else f"{v:{w}.2e}"  # v!=v -> NaN

    for label, h in samples:
        T = WATER.eval_T_ph(P_OP, h)
        rho = WATER.eval_rho_ph(P_OP, h)
        mu = WATER.eval_mu_ph(P_OP, h)
        drho = _safe(WATER.eval_drho_ph_dh)
        drho_hem = WATER.eval_drho_ph_hem_dh(P_OP, h)
        dT = _safe(WATER.eval_dT_ph_dh)
        print(f"{label:>9}  {h/1e3:9.1f}  {T:8.2f}  {rho:10.3f}  "
              f"{mu*1e6:9.3f}  {_fmt(drho)}  {_fmt(drho_hem)}  {_fmt(dT, 10)}")

    print()
    print("  Read-out:")
    print("   * Across the dome T is pinned at Tsat -> dT/dh = 0 (degenerate row).")
    print("   * rho collapses from ~liquid to ~vapour right at x=0 (a near-cliff).")
    print("   * drho/dh(1ph): CoolProp's single-phase analytic partial -> ~0 INSIDE")
    print("     the dome while rho keeps moving, and it jumps at x=0.  That mismatch")
    print("     is what makes the single-phase Jacobian go singular when boiling.")
    print("   * drho/dh(HEM): a central difference of the SAME values -> consistent")
    print("     and continuous across the saturation line. Same physics, smooth")
    print("     Jacobian -- which is exactly what lets multiphase='HEM' boil.\n")
    return hf, hg


# ---------------------------------------------------------------------------
# PART 2 -- drive a heated pipe from liquid toward steam and watch it cope
# ---------------------------------------------------------------------------


class Boiler(Model):
    """Liquid water in, externally heated pipe, open outlet.

    Every segment of a `heat_port=True` `StraightPipe` is strapped to its own
    `FixedHeatFlow` boundary so we can dial the wall heat up uniformly and
    march the fluid through the saturation dome.  `multiphase` ("single" or
    "HEM") is forwarded to the pipe so the same geometry can be solved with
    either property model.
    """

    def __init__(self, multiphase="single"):
        self._multiphase = multiphase
        super().__init__()

    def declare_components(self):
        self.add_component("inlet", AmbientInlet(
            WATER, p_ambient=P_OP, T_ambient=T_IN, m_flow=M_FLOW, D=D))
        self.add_component("pipe", StraightPipe(
            WATER, D=D, L=L, epsilon=1e-5, z_in=0, z_out=0,
            n_segments=N, heat_port=True, multiphase=self._multiphase))
        for i in range(N):
            self.add_component(f"heat_{i}", FixedHeatFlow(Q_flow=0.0, T_init=T_IN))

    def declare_equations(self):
        self.connect(self["inlet"].ports["outlet"], self["pipe"].ports["inlet"])
        wall_ports = self["pipe"].segment_wall_ports
        for i in range(N):
            self.connect(wall_ports[i], self[f"heat_{i}"].ports["heat"])
        return []


def run_boiler(hf, hg, multiphase="single", relaxation=1.0, n_ramp=26,
               line_search=False):
    print("=" * 78)
    print(f"heated water pipe (multiphase={multiphase!r}, relaxation={relaxation}, "
          f"line_search={line_search}): ramp wall heat and watch the solve")
    print("=" * 78)

    # Liquid water is stiff/ill-conditioned (near-incompressible: tiny density
    # changes against large pressures), so the Newton STEP-norm floors out around
    # ~1e-4 even for a perfectly healthy single-phase solve. Use an engineering
    # tolerance here so the liquid baseline counts as converged and we can ramp
    # heat until the *dome* (not conditioning) is what actually breaks Newton.
    #
    # Boiling density falls off a cliff (vg/vf ~ 300), so a full Newton step
    # from the liquid side overshoots straight to negative density.  Two ways
    # to tame that: a small fixed `relaxation` (needs a fine heat ramp so each
    # step stays in the smooth neighbourhood), OR `line_search=True`, which
    # takes the full step where it is feasible and automatically backs off only
    # at the cliff -- so the HEM run below crosses the whole dome at
    # relaxation=1.0 with a much coarser ramp.
    TOL = 1e-3
    MAX_ITER = 400

    system = Boiler(multiphase=multiphase)
    print("Instantiating (symbolic Jacobian + lambdify)...")
    system.instantiate(aditional_modules=WATER.modules, max_remove_trival_passes=5)
    print("Initialising at zero heat (subcooled liquid, single-phase)...")
    system.initialise(n=1, tol=TOL, max_iter=MAX_ITER, line_search=line_search)

    names = list(system.record["vars_names"])

    def idx(suffix):
        return next(i for i, n in enumerate(names) if n.endswith(suffix))

    i_h_out = idx(f".pipe_segment_{N-1}.h_out")
    i_p_out = idx(f".pipe_segment_{N-1}.p_out")
    i_h_in = idx(".pipe_segment_0.h_in")

    # Heat needed to just saturate (reach x=0) and to fully vaporise (x=1),
    # so we can size the ramp to actually cross the dome.
    h_in = WATER.eval_h_pT(P_OP, T_IN)
    q_to_sat = M_FLOW * (hf - h_in)
    q_to_dry = M_FLOW * (hg - h_in)
    print(f"\nHeat budget @ m_flow={M_FLOW*1e3:.1f} g/s:  "
          f"to saturate ~{q_to_sat:.0f} W,  to fully vaporise ~{q_to_dry:.0f} W\n")

    q_totals = np.linspace(0.0, 1.30 * q_to_dry, n_ramp)
    # Print at most ~24 rows so a fine (HEM) ramp stays readable; always print
    # the first/last and any row whose phase label changes.
    print_stride = max(1, n_ramp // 24)

    hdr = (f"{'Q_tot[W]':>9}  {'h_out[kJ/kg]':>12}  {'T_out[K]':>9}  "
           f"{'rho_out':>9}  {'phase':>10}  {'iters':>6}  {'step|d|':>10}  {'solve':>7}")
    print(hdr)
    print("-" * len(hdr))

    last_ok_q = 0.0
    crossed = False
    reached_steam = False
    h_out_last = None
    broke_down = False
    prev_phase = None
    for step_i, q_tot in enumerate(q_totals):
        q_each = q_tot / N
        for i in range(N):
            # set_value() writes straight into the live solver buffer slot, so
            # the next solve sees it. (A time-varying drive would use Input.)
            system[f"heat_{i}"]["Q_flow"].set_value(q_each)

        status = "ok"
        breakdown = None
        try:
            system.solve_dae_step(1.0, tol=TOL, max_iter=MAX_ITER,
                                   relaxation=relaxation, line_search=line_search,
                                   raise_on_no_convergence=True)
            system.next_step()
            last_ok_q = q_tot
        except NewtonConvergenceFailure:
            status = "DIVERGED"
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            # A singular Jacobian factorisation is the degenerate dT/dh=0 row
            # from Part 1 made concrete -- treat it as a solve breakdown, not a
            # crash, so the table stays readable.
            status = "SINGULAR"
            breakdown = f"{type(exc).__name__}: {exc}"
        except ValueError as exc:
            # The single-phase property derivatives raise outright once a face
            # state lands in the dome (tabular backend: "...are two-phase;
            # cannot use single-phase derivatives").  That IS the single-phase
            # limitation -- record it as a breakdown rather than crashing.
            status = "PROP-ERR"
            breakdown = f"{type(exc).__name__}: {exc}"

        # custom_solve sets these on the model on BOTH success and failure.
        iters = int(getattr(system, "_last_solve_iters", 0))
        resid = float(getattr(system, "_last_solve_error_norm", float("nan")))

        if status == "ok":
            row = np.asarray(system.record["state"][-1])
            h_out = row[i_h_out]
            p_out = row[i_p_out]
            T_out = WATER.eval_T_ph(p_out, h_out)
            rho_out = WATER.eval_rho_ph(p_out, h_out)
            h_out_last = h_out
            if h_out < hf:
                phase = "liquid"
            elif h_out <= hg:
                phase = "TWO-PHASE"
                crossed = True
            else:
                phase = "steam"
                crossed = True
                reached_steam = True
            # Keep the table short on a fine ramp: print on a stride, but always
            # show the first/last step and any phase-label transition.
            show = (step_i % print_stride == 0
                    or step_i == len(q_totals) - 1
                    or phase != prev_phase)
            prev_phase = phase
            if show:
                print(f"{q_tot:9.1f}  {h_out/1e3:12.1f}  {T_out:9.2f}  {rho_out:9.3f}  "
                      f"{phase:>10}  {iters:6d}  {resid:10.2e}  {'ok':>7}")
        else:
            broke_down = True
            print(f"{q_tot:9.1f}  {'--':>12}  {'--':>9}  {'--':>9}  "
                  f"{'--':>10}  {iters:6d}  {resid:10.2e}  {status:>7}")
            print(f"\nNewton broke down at Q_tot = {q_tot:.0f} W "
                  f"(last good solve at {last_ok_q:.0f} W).")
            if breakdown:
                print(f"  -> linear solve failed: {breakdown}")
            break

    print()
    print("Summary:")
    if multiphase == "single":
        if broke_down:
            print("  The single-phase closures broke down at/just past the saturation")
            print("  line: their (p, h) property partials are either inconsistent")
            print("  (drho/dh ~ 0 while rho moves, HEOS) or undefined outright in the")
            print("  dome (tabular backend raises) -- exactly the Part 1 failure.")
        else:
            print("  The single-phase build limped through, but watch the residual /")
            print("  iteration count spike at the dome -- it is fighting the property")
            print("  cliff + the collapsed Jacobian slopes shown in Part 1.")
    else:  # HEM
        if reached_steam:
            print("  The smooth-HEM build carried the solve ALL THE WAY from subcooled")
            print("  liquid, through the two-phase dome, to superheated steam -- at FULL")
            print("  Newton steps (relaxation=1.0): the consistent, continuous Jacobian")
            print("  (Part 1's HEM column) plus the backtracking line search (which only")
            print("  damps at the density cliff) carry it across cleanly. No events.")
        elif crossed:
            print("  The smooth-HEM build pushed the outlet into the dome and kept")
            print("  converging where the single-phase model could not.")
        else:
            print("  The smooth-HEM build stayed subcooled over this heat ramp.")

    return {
        "multiphase": multiphase,
        "crossed": crossed,
        "reached_steam": reached_steam,
        "broke_down": broke_down,
        "last_ok_q": last_ok_q,
        "h_out_last": h_out_last,
    }


def main():
    hf, hg = probe_medium()

    # Single-phase: a coarse ramp with a healthy step -- it dies at the dome
    # lip anyway.  HEM: FULL Newton steps (relaxation=1.0) with line_search=True
    # -- the backtracking picks the step length itself, so the same boiler
    # crosses the whole dome on a much coarser ramp with no hand-tuned damping.
    run_cfg = {
        "single": dict(relaxation=0.5, n_ramp=30),
        "HEM": dict(relaxation=1.0, n_ramp=60, line_search=True),
    }
    results = {}
    for mode in ("single", "HEM"):
        print()
        try:
            results[mode] = run_boiler(hf, hg, multiphase=mode, **run_cfg[mode])
        except Exception as exc:  # noqa: BLE001 - experiment: report, don't crash
            print(f"\n{mode!r} run aborted with {type(exc).__name__}: {exc}")
            results[mode] = {"multiphase": mode, "crossed": False,
                             "reached_steam": False, "broke_down": True}

    print()
    print("=" * 78)
    print("VERDICT -- single-phase vs smooth HEM through the saturation dome")
    print("=" * 78)
    for mode in ("single", "HEM"):
        r = results.get(mode, {})
        if r.get("reached_steam"):
            outcome = "reached superheated steam"
        elif r.get("crossed") and not r.get("broke_down"):
            outcome = "entered the dome and kept solving"
        elif r.get("crossed"):
            outcome = "entered the dome, then broke down"
        elif r.get("broke_down"):
            outcome = "broke down before/at the dome"
        else:
            outcome = "stayed subcooled"
        print(f"  multiphase={mode!r:8}: {outcome}")
    print()
    print("  The single-phase closures are correct and fast in one phase but cannot")
    print("  cross the saturation lines; multiphase='HEM' swaps in smooth, consistent")
    print("  property partials (CoolProp values, finite-difference Jacobian) so the")
    print("  very same pipe boils cleanly with no event handling.")


if __name__ == "__main__":
    main()
