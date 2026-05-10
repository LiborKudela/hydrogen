"""Benchmark adaptive time-stepping vs fixed dt on the `fill_vessel` transient.

The fill transient is the canonical "stiff at the start, easy at the end"
problem -- vessel pressure rises rapidly while the source-vessel gradient is
large, then slows dramatically as the gradient closes.  Fixed dt is forced to
size for the WORST case (the initial transient); adaptive can grow dt by 10-50x
once the dynamics quiet down.

What we report per-strategy:
  - end-of-run accuracy:         max abs deviation in vessel pressure vs the
                                 finest-resolution reference run
  - total Newton iterations:     direct cost in the inner solve loop
  - total wall-clock time:       end-to-end including overhead
  - number of steps:             accepted steps only (not retries)
  - retry count (adaptive only): rejected steps that had to be re-tried

The reference run uses a TIGHT fixed dt (10x smaller than the coarse fixed
run) so we can compute realistic accuracy numbers.

Run with `python examples/bench_adaptive.py` from the project root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

# Reuse the FillSystem definition from the existing example
from examples.fill_vessel import (  # noqa: E402
    DT,
    FillSystem,
    N_STEPS,
    P_SOURCE,
    P_VESSEL_INIT,
)


T_END = N_STEPS * DT                        # 3 s simulated time


def _build_and_warm():
    """Fresh FillSystem instance ready for stepping (vars warm-started, init solved)."""
    system = FillSystem()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    for var in system.active_vars_references:
        full = getattr(var, "full_name", "")
        if full.endswith(".w_in") or full.endswith(".w_out"):
            var.value = 30.0
    system.initialise(relaxation=0.5, max_iter=400)
    return system


def _trace_p_vessel(system):
    record = system.record
    t = np.asarray(record["time"])
    state = np.asarray(record["state"])
    names = list(record["vars_names"])
    idx = next(i for i, n in enumerate(names) if n.endswith(".vessel.p"))
    return t, state[:, idx]


def _peak_p_dot(t, p_v):
    """Peak |dp/dt| across the trajectory -- a single scalar that captures
    'how aggressively the vessel was filling'.  We compare this between runs
    as a proxy for resolution of the early transient."""
    if len(t) < 2:
        return 0.0
    dp = np.diff(p_v) / np.maximum(np.diff(t), 1e-12)
    return float(np.max(np.abs(dp)))


def run_fixed(dt, label, t_end=T_END, relaxation=1.0):
    """Fixed-dt loop until t = t_end.  Returns dict with timing + accuracy info."""
    system = _build_and_warm()
    n_iters = 0
    n_steps = 0
    t0 = time.perf_counter()
    while system.get_t_value() < t_end - 1e-12:
        dt_use = min(dt, t_end - system.get_t_value())
        system.solve_dae_step(dt_use, relaxation=relaxation)
        n_iters += system._last_solve_iters
        system.next_step()
        n_steps += 1
    wall = time.perf_counter() - t0
    t, p_v = _trace_p_vessel(system)
    return {
        "label": label,
        "wall_s": wall,
        "n_iters": n_iters,
        "n_steps": n_steps,
        "n_rejections": 0,
        "t": t,
        "p_v": p_v,
        "p_v_end": p_v[-1],
        "p_dot_max": _peak_p_dot(t, p_v),
    }


def run_adaptive(dt_target, strategy, label, dt_min=1e-4, dt_max=None,
                 relaxation=1.0, t_end=T_END):
    """Adaptive loop until t = t_end.  Returns dict with timing + accuracy info."""
    system = _build_and_warm()
    n_iters = 0
    n_steps = 0
    n_rejections = 0
    dt_history = []
    t0 = time.perf_counter()
    while system.get_t_value() < t_end - 1e-12:
        # Cap dt_try at the dt_max ceiling, NOT the dt_target hint -- otherwise
        # the controller can never grow beyond the initial guess.
        dt_try = min(dt_max if dt_max is not None else dt_target,
                     t_end - system.get_t_value())
        if hasattr(system, "_dt_hint"):
            dt_try = min(dt_try, system._dt_hint)
        else:
            # First call: respect dt_target as the initial guess.
            dt_try = min(dt_try, dt_target)
        dt_used, info = system.solve_adaptive_step(
            dt_try, strategy=strategy,
            dt_min=dt_min, dt_max=dt_max,
            relaxation=relaxation, max_iter=200,
        )
        n_iters += info["n_iters"]
        n_rejections += info["rejections"]
        n_steps += 1
        dt_history.append(dt_used)
        system.next_step()
    wall = time.perf_counter() - t0
    t, p_v = _trace_p_vessel(system)
    return {
        "label": label,
        "wall_s": wall,
        "n_iters": n_iters,
        "n_steps": n_steps,
        "n_rejections": n_rejections,
        "dt_min_used": min(dt_history),
        "dt_max_used": max(dt_history),
        "dt_avg_used": sum(dt_history) / len(dt_history),
        "t": t,
        "p_v": p_v,
        "p_v_end": p_v[-1],
        "p_dot_max": _peak_p_dot(t, p_v),
    }


def _interp_p_at(t, p_v, t_query):
    """Linear interpolation of p_v at t_query (for accuracy comparison
    between runs whose recording grids don't line up)."""
    return float(np.interp(t_query, t, p_v))


def main():
    print("=== Adaptive vs fixed dt benchmark on fill_vessel ===")
    print(f"Simulating t = 0 .. {T_END:g} s "
          f"(p_source = {P_SOURCE/1e5:.2f} bar, p_vessel_0 = {P_VESSEL_INIT/1e5:.2f} bar)")
    print()

    # --- Reference run: fine fixed dt -----------------------------------------
    # 50x finer than the coarse fixed run -- effectively "exact" for our
    # accuracy comparison.
    dt_ref = DT / 50
    print(f"Reference run (fine fixed dt = {dt_ref:g} s, "
          f"~{int(T_END / dt_ref)} steps)...")
    ref = run_fixed(dt_ref, label=f"fixed dt={dt_ref:g} (REF)")
    print(f"  wall {ref['wall_s']:.3f}s, {ref['n_iters']} Newton iters, "
          f"{ref['n_steps']} steps")
    print()

    # Compare accuracy at a probe point in the EARLY transient (where things
    # are happening), not at T_END (where everything has equilibrated).
    t_probe = 0.5  # 0.5 s -- middle of the steep rise
    p_ref_at_probe = _interp_p_at(ref["t"], ref["p_v"], t_probe)
    print(f"Accuracy probe: p_vessel at t={t_probe:g}s = {p_ref_at_probe:.3f} Pa")
    print()

    runs = []

    # --- Coarse fixed run ------------------------------------------------------
    print(f"Coarse fixed dt = {DT:g} s (the example's setting)...")
    runs.append(run_fixed(DT, label=f"fixed dt={DT:g}"))

    # --- Adaptive runs ---------------------------------------------------------
    print(f"Adaptive runs (initial dt_target = {DT/4:g}s, dt_max = {4*DT:g}s)...")
    for strat_dict in [
        {"name": "predictor_corrector", "tol_local": 1e-2, "atol": 1e-3},
        {"name": "predictor_corrector", "tol_local": 1e-3, "atol": 1e-3},
        {"name": "derivative_limit",    "rel_tol": 1e-1,   "atol": 1e-3},
        {"name": "richardson",          "tol_local": 1e-2, "atol": 1e-3},
    ]:
        label = (f"{strat_dict['name'][:8]} "
                 f"tol={strat_dict.get('tol_local', strat_dict.get('rel_tol')):g}")
        runs.append(run_adaptive(
            DT / 4, strategy=strat_dict, label=label,
            dt_min=1e-5, dt_max=4 * DT,
        ))

    # --- Print results table ---------------------------------------------------
    print()
    print(f"{'strategy':<25s} {'wall[s]':>8s} {'iters':>7s} {'steps':>6s} "
          f"{'rej':>4s} {'p@0.5s err [Pa]':>16s} {'dt range [s]':>22s}")
    print("-" * 95)
    # Reference first (just for context)
    err = abs(_interp_p_at(ref["t"], ref["p_v"], t_probe) - p_ref_at_probe)
    print(f"{ref['label']:<25s} {ref['wall_s']:8.3f} {ref['n_iters']:7d} "
          f"{ref['n_steps']:6d} {ref['n_rejections']:4d} {err:16.3f} {dt_ref:22.5f}")
    for r in runs:
        err = abs(_interp_p_at(r["t"], r["p_v"], t_probe) - p_ref_at_probe)
        if "dt_min_used" in r:
            dt_str = f"{r['dt_min_used']:.4f} .. {r['dt_max_used']:.4f}"
        else:
            dt_str = "fixed"
        print(f"{r['label']:<25s} {r['wall_s']:8.3f} {r['n_iters']:7d} "
              f"{r['n_steps']:6d} {r['n_rejections']:4d} {err:16.3f} {dt_str:>22s}")

    # --- Headline summary ------------------------------------------------------
    print()
    fixed_coarse = runs[0]
    # Pick the first PC result to summarise
    pc = next(r for r in runs if r["label"].startswith("predicto"))
    err_pc = abs(_interp_p_at(pc["t"], pc["p_v"], t_probe) - p_ref_at_probe)
    err_fixed = abs(_interp_p_at(fixed_coarse["t"], fixed_coarse["p_v"], t_probe)
                    - p_ref_at_probe)
    speedup_iters = fixed_coarse["n_iters"] / pc["n_iters"] if pc["n_iters"] else float("nan")
    speedup_wall = fixed_coarse["wall_s"] / pc["wall_s"] if pc["wall_s"] else float("nan")
    print(f"Adaptive predictor_corrector (tol_local=1e-2) vs coarse fixed dt={DT:g}:")
    print(f"  Newton iters:  {speedup_iters:5.2f}x  "
          f"({pc['n_iters']} adaptive vs {fixed_coarse['n_iters']} fixed)")
    print(f"  Wall time:     {speedup_wall:5.2f}x  "
          f"({pc['wall_s']:.3f}s adaptive vs {fixed_coarse['wall_s']:.3f}s fixed)")
    print(f"  Accuracy:      adaptive {err_pc:6.2f} Pa  vs  fixed {err_fixed:6.2f} Pa  "
          f"(absolute err in p_vessel @ t={t_probe:g}s)")
    print(f"  PC steps:      {pc['n_steps']} accepted, {pc['n_rejections']} rejected, "
          f"dt grew from {pc['dt_min_used']:.4f}s to {pc['dt_max_used']:.4f}s")


if __name__ == "__main__":
    main()
