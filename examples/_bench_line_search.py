"""Ad-hoc benchmark: damped Newton vs. backtracking line search.

Compares the solver with `line_search=False` (fixed `relaxation` damping) against
`line_search=True` (feasibility-guarded backtracking) on the two_phase_boiler in
two regimes:

  * pure single-phase liquid (no dome)  -> measures the *overhead* of the line
    search when the full step is always feasible (it should be ~free in iters,
    paying only for extra residual evals).
  * crossing the boiling dome to steam  -> measures *robustness* (does it cross
    at all?) and the cost to do so.

Not committed / not a pytest: run directly.
"""
import time
import numpy as np

import examples.two_phase_boiler as B
from hydrogen import CoolPropMedium
import CoolProp.CoolProp as CP

B.WATER = CoolPropMedium(B.FLUID, disable_warnings=True, backend="BICUBIC&HEOS")
B.N = 4
WATER, P_OP, T_IN, N, M_FLOW = B.WATER, B.P_OP, B.T_IN, B.N, B.M_FLOW

HF = CP.PropsSI("H", "P", P_OP, "Q", 0, "Water")
HG = CP.PropsSI("H", "P", P_OP, "Q", 1, "Water")
H_IN = WATER.eval_h_pT(P_OP, T_IN)
Q_DRY = M_FLOW * (HG - H_IN)  # wall heat needed to reach saturated vapour


def _instrument(system):
    """Wrap eval_residuals to count residual evaluations."""
    counter = {"resid": 0}
    orig = system.eval_residuals

    def counting(vars):
        counter["resid"] += 1
        return orig(vars)

    system.eval_residuals = counting
    return counter


def run(name, *, multiphase, relaxation, n_ramp, line_search, q_factor,
        ls_grow=2.0):
    system = B.Boiler(multiphase=multiphase)
    # Silence the instantiation chatter for a clean table.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        system.instantiate(aditional_modules=WATER.modules,
                           max_remove_trival_passes=5)
        system.initialise(n=1, tol=1e-3, max_iter=400, line_search=line_search)

    counter = _instrument(system)
    names = list(system.record["vars_names"])
    i_h = next(i for i, n in enumerate(names)
               if n.endswith(f".pipe_segment_{N - 1}.h_out"))

    q_max = q_factor * Q_DRY
    total_iters = 0
    x_max = -1.0
    ok = 0
    fail = None
    t0 = time.perf_counter()
    for k in range(1, n_ramp + 1):
        q = q_max * k / n_ramp
        for i in range(N):
            system[f"heat_{i}"]["Q_flow"].set_value(q / N)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # Drive custom_solve directly so we can sweep `ls_grow`
                # (solve_dae_step keeps the shipped default).
                system.set_dt(1.0)
                system.set_t(system.get_t_value() + 1.0)
                system.custom_solve(tol=1e-3, max_iter=400,
                                    relaxation=relaxation,
                                    line_search=line_search, ls_grow=ls_grow,
                                    raise_on_no_convergence=True)
                system.next_step()
            ok = k
            total_iters += int(system._last_solve_iters)
            h = float(np.asarray(system.record["state"][-1])[i_h])
            x_max = max(x_max, (h - HF) / (HG - HF))
        except Exception as e:  # noqa: BLE001
            fail = type(e).__name__
            break
    wall = time.perf_counter() - t0

    status = "STEAM" if x_max > 1.0 else ("dome" if x_max > 0 else "liquid")
    if fail:
        status = f"BROKE@x={x_max:.2f}"
    grow = "inf" if ls_grow == np.inf else f"{ls_grow:g}"
    print(f"{name:34} | LS={str(line_search):5} relax={relaxation:<4} "
          f"grow={grow:<3} ramp={n_ramp:<4} | steps {ok:>3}/{n_ramp:<3} | "
          f"iters {total_iters:>5} | resid {counter['resid']:>6} | "
          f"{wall:6.2f}s | x_max {x_max:6.3f} | {status}")
    return dict(ok=ok, iters=total_iters, resid=counter["resid"], wall=wall,
                x_max=x_max, fail=fail)


if __name__ == "__main__":
    print("=" * 132)
    print("REGIME 1 -- strictly subcooled liquid (q ramps to 0.18*q_dry, never reaches x=0)")
    print("           => isolates OVERHEAD when every full step is feasible. grow=inf is a pure feasibility guard.")
    print("-" * 132)
    run("liquid: damped Newton",          multiphase="HEM", relaxation=1.0, n_ramp=20, line_search=False, q_factor=0.18)
    run("liquid: line search grow=2",     multiphase="HEM", relaxation=1.0, n_ramp=20, line_search=True,  q_factor=0.18, ls_grow=2.0)
    run("liquid: line search grow=inf",   multiphase="HEM", relaxation=1.0, n_ramp=20, line_search=True,  q_factor=0.18, ls_grow=np.inf)

    print("=" * 132)
    print("REGIME 2 -- cross the boiling dome to superheated steam (q ramps to 1.3*q_dry)")
    print("           => robustness + cost. Full Newton steps overshoot the density cliff without globalization.")
    print("-" * 132)
    run("dome: full step, NO globalization", multiphase="HEM", relaxation=1.0,  n_ramp=60,  line_search=False, q_factor=1.3)
    run("dome: manual damping (coarse ramp)", multiphase="HEM", relaxation=0.2, n_ramp=60,  line_search=False, q_factor=1.3)
    run("dome: manual damping (fine ramp)",   multiphase="HEM", relaxation=0.2, n_ramp=150, line_search=False, q_factor=1.3)
    run("dome: line search grow=2  (coarse)", multiphase="HEM", relaxation=1.0, n_ramp=60,  line_search=True,  q_factor=1.3, ls_grow=2.0)
    run("dome: line search grow=inf(coarse)", multiphase="HEM", relaxation=1.0, n_ramp=60,  line_search=True,  q_factor=1.3, ls_grow=np.inf)
    run("dome: line search grow=2  (fine)",   multiphase="HEM", relaxation=1.0, n_ramp=150, line_search=True,  q_factor=1.3, ls_grow=2.0)
    print("=" * 132)
