"""End-to-end smoke tests for the scripts in `examples/`.

These are **deselected by default** (every test here carries the `examples`
marker, and the default `addopts` in `pyproject.toml` runs `-m 'not
examples'`).  Run them explicitly when a change might affect the worked
examples:

    pytest -m examples                 # all examples
    pytest -m examples -k flat_wall    # one example
    pytest -m examples -s              # stream each script's stdout

Each example is executed in its own subprocess (exactly how a user runs
`python examples/foo.py`), so module-level state in one example can't leak
into another and a crash surfaces as a non-zero exit code.  "Correctness"
is enforced by the examples themselves: the physics scripts assert their
analytical/ conservation invariants internally, so a wrong result makes
the script exit non-zero and fails the test here.

Plot artifacts are redirected into a per-run temp directory via
`HYDROGEN_LOCAL_RESULTS` so running the suite never pollutes the repo's
`local_results/`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _PROJECT_ROOT / "examples"

# Discover every runnable example script (skip package/dunder files).
EXAMPLE_SCRIPTS = sorted(
    p.name for p in _EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_")
)

# Benchmarks are perf comparisons rather than correctness checks; they still
# need to *run*, but they're the slowest scripts, so flag them `slow` too in
# case someone wants `-m "examples and not slow"`.
_BENCHMARKS = {"bench_adaptive.py", "bench_blt.py", "bench_pipe_tree.py"}

# Generous per-script wall-clock ceiling (instantiate + lambdify of the
# larger systems dominates; benchmarks run several systems back to back).
_TIMEOUT_S = 900


def _params():
    for name in EXAMPLE_SCRIPTS:
        marks = [pytest.mark.examples]
        if name in _BENCHMARKS:
            marks.append(pytest.mark.slow)
        yield pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize("script", _params())
def test_example_runs(script, tmp_path):
    """Run `examples/<script>` to completion with a zero exit code."""
    env = dict(os.environ)
    env["HYDROGEN_LOCAL_RESULTS"] = str(tmp_path)
    # Unbuffered so partial output is captured if the script times out.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(
        [sys.executable, str(_EXAMPLES_DIR / script)],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        pytest.fail(
            f"examples/{script} exited with code {proc.returncode}.\n"
            f"--- last 40 lines of output ---\n{tail}"
        )
