"""Scaling + correctness benchmark for the SegmentedChannel pipe engine.

Two things in one script:

  1. SCALING -- sweep the pipe segment count N and report how
     `collect_equations` / instantiate / initialise / per-step solve scale with
     problem size (`n_v`).  This directly exercises the per-cell template
     optimisation in `SegmentedChannel.declare_equations` (the symbolic build
     that used to dominate `collect_equations`).

  2. CORRECTNESS -- at a small N, run the SAME network with the legacy
     `straight` engine and the `segmented` engine and assert the tank
     trajectory matches to ~machine precision.  A divergence here means the
     segmented engine changed the physics, not just the speed.

    python benchmarks/scaling_segmented_pipe.py            # default sweep
    python benchmarks/scaling_segmented_pipe.py --smoke     # tiny sweep (CI)
    python benchmarks/scaling_segmented_pipe.py --leaky     # permeation ON (heavier)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks import bench_segmented as bs  # noqa: E402
from benchmarks._harness import run_scaling  # noqa: E402

_INSTANTIATE_KWARGS = dict(
    aditional_modules=bs.HYDROGEN.modules, cse=True, enable_blt=True,
    enable_var_scaling=False, max_remove_trival_passes=1,
    max_remove_duplicate_passes=5, max_remove_linear_block_passes=3,
)
_INITIALISE_KWARGS = dict(n=1, relaxation=1.0, tol=1e-6, max_iter=200)
_RUN_KWARGS = dict(
    strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 1.0},
    dt_start=1e-4, dt_min=1e-9, dt_max=50.0, grow=1.5, shrink=0.5,
    max_retries=30, relaxation=1.0, tol=1e-6, max_iter=200,
    raise_on_no_convergence=False,
)


def _engine_endstate(engine, n_segments, stop_time):
    """Instantiate + initialise + run one engine; return (p_tank_end, T_tank_end)."""
    m = bs.build_model(n_segments, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        m.instantiate(**_INSTANTIATE_KWARGS)
        m.initialise(**_INITIALISE_KWARGS)
        m.run(stop_time=stop_time, **_RUN_KWARGS)
    p = np.asarray(m.series("tank_3.gas.p"))
    T = np.asarray(m.series("tank_3.gas.T"))
    return float(p[-1]), float(T[-1])


def check_engine_match(n_segments=5, stop_time=3.0, tol=1e-6):
    """Assert the segmented engine reproduces the straight engine's tank state."""
    print("=" * 76)
    print(f"Engine-match correctness  (N={n_segments}, stop_time={stop_time}s)")
    print("-" * 76)
    p_s, T_s = _engine_endstate("straight", n_segments, stop_time)
    p_g, T_g = _engine_endstate("segmented", n_segments, stop_time)
    rel_p = abs(p_s - p_g) / abs(p_s)
    rel_T = abs(T_s - T_g) / abs(T_s)
    print(f"  straight : p_tank={p_s / 1e5:.6f} bar  T_tank={T_s:.4f} K")
    print(f"  segmented: p_tank={p_g / 1e5:.6f} bar  T_tank={T_g:.4f} K")
    print(f"  rel dp={rel_p:.3e}  rel dT={rel_T:.3e}  (tol {tol:.0e}) -> "
          f"{'OK' if max(rel_p, rel_T) <= tol else 'FAIL'}")
    assert rel_p <= tol and rel_T <= tol, (
        f"segmented vs straight mismatch: rel dp={rel_p:.3e} rel dT={rel_T:.3e}")
    print("=" * 76)


def main(smoke=False, leaky=False):
    if leaky:
        bs.PERMEATION = True

    sizes = [5, 10] if smoke else [10, 50, 100, 300, 1000]
    label = (f"Segmented pipe scaling  (engine=segmented, "
             f"permeation={'on' if leaky else 'off'})")
    run_scaling(
        label,
        build=lambda n: bs.build_model(n, "segmented"),
        sizes=sizes,
        instantiate_kwargs=_INSTANTIATE_KWARGS,
        initialise_kwargs=_INITIALISE_KWARGS,
        solve_dt=0.05,
        solve_steps=5,
    )

    # Correctness is engine-independent of N; check at a small size to keep it cheap.
    check_engine_match(n_segments=5, stop_time=3.0)
    print("scaling_segmented_pipe OK")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(smoke="--smoke" in args, leaky="--leaky" in args)
