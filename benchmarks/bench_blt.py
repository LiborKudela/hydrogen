"""Compare solve-time with vs without BLT on representative workloads."""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from tutorials.pipe_tree import TreeSystem, _bernoulli_warm_start  # noqa: E402
from tutorials.run_system import System as RunSystem  # noqa: E402
from hydrogen import CoolPropMedium  # noqa: E402


def _hash(system):
    record = system.record
    state = np.asarray(record["state"])
    names = list(record["vars_names"])
    # First diff variable
    h_signature = []
    for i in range(min(10, len(names))):
        h_signature.append(round(float(state[-1, i]), 6))
    import hashlib
    import json
    return hashlib.md5(json.dumps(h_signature).encode()).hexdigest()[:12]


def bench_run_system(enable_blt, label):
    print(f"\n=== run_system :: {label} (enable_blt={enable_blt}) ===", flush=True)
    air = CoolPropMedium('air', disable_warnings=True, backend="BICUBIC&HEOS",
                        scalar_cache_maxsize=1000)
    model = RunSystem()
    model.medium = air  # for parity with other usage
    gc.collect()

    t0 = time.perf_counter()
    model.instantiate(aditional_modules=air.modules, max_remove_trival_passes=5,
                      enable_blt=enable_blt)
    t_inst = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.initialise(n=1)
    t_init = time.perf_counter() - t0

    # Fixed-dt loop for stable timing
    n_iters_total = 0
    n_steps = 30
    t0 = time.perf_counter()
    for _ in range(n_steps):
        model.solve_dae_step(0.04)
        n_iters_total += model._last_solve_iters
        model.next_step()
    t_solve = time.perf_counter() - t0

    print(f"  inst={t_inst:.2f}s init={t_init:.2f}s solve{n_steps}={t_solve:.3f}s "
          f"(per-step={t_solve/n_steps*1000:.2f}ms, n_iters_total={n_iters_total}) "
          f"n_v={model.n_v} fp={_hash(model)}")
    return t_solve


def bench_pipe_tree(enable_blt, label, N=3, K=2, M=3):
    print(f"\n=== pipe_tree N={N} K={K} M={M} :: {label} (enable_blt={enable_blt}) ===", flush=True)
    medium = CoolPropMedium("Air", disable_warnings=True)
    system = TreeSystem(medium, N=N, K=K, M=M)
    gc.collect()

    t0 = time.perf_counter()
    system.instantiate(aditional_modules=system.medium.modules,
                       max_remove_trival_passes=4, enable_blt=enable_blt)
    t_inst = time.perf_counter() - t0

    warm_m_dot = _bernoulli_warm_start(system)
    for var in system.active_vars_references:
        full = getattr(var, "full_name", "")
        if (full.endswith(".m_dot_in") or full.endswith(".m_dot_out")
                or ".m_dot_out_" in full):
            depth = full.count(".child_")
            var.value = warm_m_dot / (system.K ** depth)

    t0 = time.perf_counter()
    system.initialise(relaxation=0.5, max_iter=400)
    t_init = time.perf_counter() - t0

    n_iters_total = 0
    n_steps = 20
    t0 = time.perf_counter()
    for _ in range(n_steps):
        system.solve_dae_step(0.05)
        n_iters_total += system._last_solve_iters
        system.next_step()
    t_solve = time.perf_counter() - t0

    print(f"  inst={t_inst:.2f}s init={t_init:.2f}s solve{n_steps}={t_solve:.3f}s "
          f"(per-step={t_solve/n_steps*1000:.2f}ms, n_iters_total={n_iters_total}) "
          f"n_v={system.n_v} fp={_hash(system)}")
    return t_solve


def main():
    results = {}

    print("\n========== run_system ==========")
    t_off = bench_run_system(enable_blt=False, label="off")
    t_on = bench_run_system(enable_blt=True, label="on")
    results['run_system'] = (t_off, t_on)

    print("\n========== pipe_tree N=3 ==========")
    t_off = bench_pipe_tree(enable_blt=False, label="off", N=3)
    t_on = bench_pipe_tree(enable_blt=True, label="on", N=3)
    results['pipe_tree_N3'] = (t_off, t_on)

    print("\n\n========== SUMMARY (solve time only) ==========")
    print(f"{'workload':<20}  {'BLT off (s)':>12}  {'BLT on (s)':>12}  {'speedup':>10}")
    for name, (t_off, t_on) in results.items():
        print(f"{name:<20}  {t_off:>12.3f}  {t_on:>12.3f}  {t_off/t_on:>9.2f}x")


if __name__ == "__main__":
    main()
