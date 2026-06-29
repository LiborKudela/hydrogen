"""End-to-end smoke tests for the scripts in `benchmarks/`.

These are **deselected by default** (every test here carries both the
`benchmarks` and `slow` markers, and the default `addopts` in `pyproject.toml`
runs `-m 'not tutorials and not benchmarks'`).  Run them explicitly:

    pytest -m benchmarks                      # all benchmarks
    pytest -m benchmarks -k scaling           # one benchmark
    pytest -m "benchmarks and not slow"       # (everything here is slow today)

Benchmarks are perf / scaling harnesses, but the analytical ones also assert
correctness against closed-form solutions, so a regression that changes the
numbers (not just the speed) fails here too.  Each runs in its own subprocess
with a default size/sweep small enough to finish in CI.

Shared helpers live in `benchmarks/_harness.py` (underscore-prefixed, so it is
not discovered as a runnable benchmark).

Artifacts are redirected into a per-run temp directory via
`HYDROGEN_LOCAL_RESULTS`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"

# Discover every runnable benchmark (skip package/dunder and the `_harness`).
BENCHMARK_SCRIPTS = sorted(
    p.name for p in _BENCHMARKS_DIR.glob("*.py") if not p.name.startswith("_")
)

# Optional per-script CLI args to keep the CI sweep cheap (benchmarks default
# to heavier sizes when run by hand).
_SCRIPT_ARGS = {
    "bench_segmented.py": ["smoke"],          # cheap N=2 both-engine build
    "bench_feos_vs_coolprop.py": ["5"],       # short stop_time
    "scaling_segmented_pipe.py": ["--smoke"],  # tiny size sweep
}

_TIMEOUT_S = 1200


@pytest.mark.parametrize(
    "script", [pytest.param(n, marks=[pytest.mark.benchmarks, pytest.mark.slow],
                            id=n)
               for n in BENCHMARK_SCRIPTS]
)
def test_benchmark_runs(script, tmp_path):
    """Run `benchmarks/<script>` to completion with a zero exit code."""
    env = dict(os.environ)
    env["HYDROGEN_LOCAL_RESULTS"] = str(tmp_path)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, str(_BENCHMARKS_DIR / script)]
    cmd += _SCRIPT_ARGS.get(script, [])

    proc = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        pytest.fail(
            f"benchmarks/{script} exited with code {proc.returncode}.\n"
            f"--- last 40 lines of output ---\n{tail}"
        )
