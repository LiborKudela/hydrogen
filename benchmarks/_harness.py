"""Shared helpers for the benchmark scripts.

Underscore-prefixed so the test harness (`tests/test_benchmarks.py`) does not
pick it up as a runnable benchmark.

Two building blocks:

  * `run_scaling(...)` -- time instantiate / initialise / solve across a sweep
    of problem sizes and print a compact scaling table.  `collect_equations`
    is reported separately (parsed from `Model.instantiate`'s own stdout line)
    because it is the single-threaded symbolic-build phase that dominates large
    models.

  * `compare_to_analytical(...)` -- max-abs + L2 error of a simulated trace
    against a closed-form solution, with an assertion against an error budget,
    so a benchmark that changes the *numbers* (not just the speed) fails loudly.
"""

from __future__ import annotations

import contextlib
import io
import re
import time
from dataclasses import dataclass, field

import numpy as np

_COLLECT_RE = re.compile(r"collect_equations=([\d.]+)s")


@dataclass
class SizeResult:
    size: object
    n_v: int
    t_collect: float
    t_instantiate: float
    t_initialise: float
    t_solve: float
    extra: dict = field(default_factory=dict)


def _quiet(fn, *args, **kwargs):
    """Run `fn`, capturing stdout; return (result, captured_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def run_scaling(
    label,
    build,
    sizes,
    *,
    instantiate_kwargs=None,
    initialise_kwargs=None,
    solve_dt=0.05,
    solve_steps=5,
    warmstart=None,
    measure=None,
):
    """Time the instantiate -> initialise -> solve pipeline across `sizes`.

    Parameters
    ----------
    label : str
        Heading printed above the table.
    build : callable(size) -> Model
        Returns a fresh, uninstantiated model for the given size token.
    sizes : iterable
        Size tokens passed to `build` (e.g. segment counts, or dicts).
    instantiate_kwargs, initialise_kwargs : dict, optional
        Forwarded to `Model.instantiate` / `Model.initialise`.
    solve_dt, solve_steps : float, int
        Fixed-step solve used purely to time the inner loop.
    warmstart : callable(model) -> None, optional
        Called after instantiate, before initialise (e.g. Bernoulli seeding).
    measure : callable(model, size) -> dict, optional
        Extra per-size metrics merged into the result (e.g. a correctness
        fingerprint).

    Returns a list of `SizeResult`.
    """
    instantiate_kwargs = dict(instantiate_kwargs or {})
    initialise_kwargs = dict(initialise_kwargs or {})

    print("=" * 76)
    print(label)
    print("-" * 76)
    header = (f"{'size':>14} | {'n_v':>7} | {'collect':>8} | {'instantiate':>11} | "
              f"{'initialise':>10} | {'solve/step':>10}")
    print(header)
    print("-" * 76)

    results = []
    for size in sizes:
        model = build(size)

        t0 = time.perf_counter()
        _, out = _quiet(model.instantiate, **instantiate_kwargs)
        t_inst = time.perf_counter() - t0
        m = _COLLECT_RE.search(out)
        t_collect = float(m.group(1)) if m else float("nan")

        if warmstart is not None:
            warmstart(model)

        t0 = time.perf_counter()
        _quiet(model.initialise, **initialise_kwargs)
        t_init = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(solve_steps):
            model.solve_dae_step(solve_dt)
            model.next_step()
        t_solve = (time.perf_counter() - t0) / max(1, solve_steps)

        extra = measure(model, size) if measure is not None else {}
        res = SizeResult(
            size=size, n_v=int(getattr(model, "n_v", -1)),
            t_collect=t_collect, t_instantiate=t_inst,
            t_initialise=t_init, t_solve=t_solve, extra=extra,
        )
        results.append(res)
        print(f"{str(size):>14} | {res.n_v:>7} | {t_collect:>7.2f}s | "
              f"{t_inst:>10.2f}s | {t_init:>9.2f}s | {t_solve * 1e3:>8.1f}ms")

    print("=" * 76)
    return results


def compare_to_analytical(t, y, f_exact, *, atol, rtol=0.0, label=""):
    """Compare a simulated trace `y(t)` against closed-form `f_exact(t)`.

    Returns `(max_abs_err, l2_err)` and asserts the max error is within
    `atol + rtol * max|y_exact|`, so a numerical regression fails the benchmark.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y_exact = np.asarray(f_exact(t), dtype=float)

    abs_err = np.abs(y - y_exact)
    max_abs = float(np.max(abs_err))
    l2 = float(np.sqrt(np.mean(abs_err ** 2)))
    budget = atol + rtol * float(np.max(np.abs(y_exact)))

    tag = f"{label}: " if label else ""
    print(f"  {tag}max|err|={max_abs:.3e}  L2={l2:.3e}  "
          f"(budget {budget:.3e})  -> {'OK' if max_abs <= budget else 'FAIL'}")
    assert max_abs <= budget, (
        f"{tag}max error {max_abs:.3e} exceeds budget {budget:.3e}")
    return max_abs, l2
