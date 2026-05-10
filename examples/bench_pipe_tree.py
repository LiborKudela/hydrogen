"""Benchmark + correctness harness for pipe_tree, used to drive optimisation work.

Measures, for several (N, K, M) sizes:
  * `Model.instantiate` wall time (split per sub-phase via timestamps printed by `Model`)
  * `Model.initialise` wall time
  * 5-step solve-loop wall time
  * peak RSS (via `resource.getrusage`) and tracemalloc peak Python heap
  * a stable correctness fingerprint (steady-state mass-flow at the source +
    every leaf-pipe `w_out`, hashed) so we can verify each refactor preserves
    the numerical answer.

Run as:  python3 examples/bench_pipe_tree.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from examples.pipe_tree import TreeSystem, _bernoulli_warm_start  # noqa: E402
from hydrogen import CoolPropMedium  # noqa: E402

# Sizes to benchmark.  Tuned so the smallest finishes in ~seconds on the
# baseline and the biggest is large enough that O(N^2) effects are visible.
SIZES = [
    dict(N=2, K=2, M=2),  # 7 pipes
    dict(N=3, K=2, M=3),  # 15 pipes
    dict(N=4, K=2, M=3),  # 31 pipes
]


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # ru_maxrss is KiB on linux


def _state_fingerprint(system) -> dict:
    """A small set of robust steady-state numbers + a hash of every leaf w_out."""
    record = system.record
    state = np.asarray(record["state"])
    names = list(record["vars_names"])

    def trace_last(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return float(state[-1, idx])

    leaf_w = sorted(
        float(state[-1, i])
        for i, n in enumerate(names)
        if n.endswith(".pipe.w_out") and ".child_" in n and (n.count(".child_") == system.N)
    )
    rho_src = float(system.medium.eval_rho_ph(
        trace_last(".source.p_out"),
        trace_last(".source.h_out"),
    ))
    m_dot_src = trace_last(".source.w_out") * system.A_pipe * rho_src
    h = hashlib.md5(json.dumps([round(w, 6) for w in leaf_w]).encode()).hexdigest()[:12]
    return {
        "n_vars_active": int(system.n_v),
        "m_dot_source_g_s": round(m_dot_src * 1000, 6),
        "leaf_w_min": round(min(leaf_w), 6),
        "leaf_w_max": round(max(leaf_w), 6),
        "leaf_w_hash": h,
        "n_leaves": len(leaf_w),
    }


def bench_one(label: str, size: dict) -> dict:
    print(f"\n--- {label} :: N={size['N']} K={size['K']} M={size['M']} ---", flush=True)
    medium = CoolPropMedium("Air", disable_warnings=True)
    system = TreeSystem(medium, **size)
    n_pipes, n_splitters, n_outlets = system.topology()

    # Force a clean baseline before measuring.
    gc.collect()
    tracemalloc.start()
    rss_before = _peak_rss_mb()

    t0 = time.perf_counter()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=4,
    )
    t_inst = time.perf_counter() - t0

    gc.collect()
    cur_py_after_inst, peak_py_after_inst = tracemalloc.get_traced_memory()
    rss_after_inst = _peak_rss_mb()

    warm = _bernoulli_warm_start(system)
    for var in system.active_vars_references:
        full = getattr(var, "full_name", "")
        if full.endswith(".w_in") or full.endswith(".w_out") or ".w_out_" in full:
            var.value = warm

    t0 = time.perf_counter()
    system.initialise(relaxation=0.5, max_iter=400)
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(5):
        system.solve_dae_step(0.05)
        system.next_step()
    t_solve = time.perf_counter() - t0

    rss_after_solve = _peak_rss_mb()
    cur_py_total, peak_py_total = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fp = _state_fingerprint(system)

    result = {
        "label": label,
        "N": size["N"], "K": size["K"], "M": size["M"],
        "n_pipes": n_pipes, "n_splitters": n_splitters, "n_outlets": n_outlets,
        "t_instantiate_s": round(t_inst, 3),
        "t_initialise_s": round(t_init, 3),
        "t_solve5_s": round(t_solve, 3),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_inst_mb": round(rss_after_inst, 1),
        "rss_after_solve_mb": round(rss_after_solve, 1),
        "py_peak_after_inst_mb": round(peak_py_after_inst / (1024 * 1024), 1),
        "py_cur_after_inst_mb": round(cur_py_after_inst / (1024 * 1024), 1),
        "py_peak_total_mb": round(peak_py_total / (1024 * 1024), 1),
        "py_cur_total_mb": round(cur_py_total / (1024 * 1024), 1),
        **fp,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    label = os.environ.get("BENCH_LABEL", "baseline")
    all_results = []
    for size in SIZES:
        # Skip the largest unless the user asks for it (it's the slowest).
        if size["N"] >= 4 and os.environ.get("BENCH_SKIP_LARGE"):
            continue
        all_results.append(bench_one(label, size))

    from hydrogen import local_results_path
    out_path = Path(local_results_path("examples", "bench_pipe_tree.jsonl"))
    with out_path.open("a") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(all_results)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
