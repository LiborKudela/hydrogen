"""End-to-end smoke tests for the scripts in `tutorials/`.

These are **deselected by default** (every test here carries the `tutorials`
marker, and the default `addopts` in `pyproject.toml` runs
`-m 'not tutorials and not benchmarks'`).  Run them explicitly when a change
might affect the worked tutorials:

    pytest -m tutorials                 # all tutorials
    pytest -m tutorials -k flat_wall    # one tutorial
    pytest -m tutorials -s              # stream each script's stdout

Each tutorial is executed in its own subprocess (exactly how a user runs
`python tutorials/foo.py`), so module-level state in one tutorial can't leak
into another and a crash surfaces as a non-zero exit code.  "Correctness" is
enforced by the tutorials themselves: the physics scripts assert their
analytical / conservation invariants internally, so a wrong result makes the
script exit non-zero and fails the test here.

Only top-level scripts are discovered (matching the historical behaviour);
the subdirectory demos under `tutorials/host_client/` and
`tutorials/h2_permeation_pressurize/` need a live host / display and are run
by hand.

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
_TUTORIALS_DIR = _PROJECT_ROOT / "tutorials"

# Discover every runnable top-level tutorial (skip package/dunder files).
TUTORIAL_SCRIPTS = sorted(
    p.name for p in _TUTORIALS_DIR.glob("*.py") if not p.name.startswith("_")
)

# Generous per-script wall-clock ceiling (instantiate + lambdify of the larger
# systems dominates; saved_system_2 runs a full transient).
_TIMEOUT_S = 900


@pytest.mark.parametrize(
    "script", [pytest.param(n, marks=pytest.mark.tutorials, id=n)
               for n in TUTORIAL_SCRIPTS]
)
def test_tutorial_runs(script, tmp_path):
    """Run `tutorials/<script>` to completion with a zero exit code."""
    env = dict(os.environ)
    env["HYDROGEN_LOCAL_RESULTS"] = str(tmp_path)
    # Unbuffered so partial output is captured if the script times out.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(
        [sys.executable, str(_TUTORIALS_DIR / script)],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )

    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
        pytest.fail(
            f"tutorials/{script} exited with code {proc.returncode}.\n"
            f"--- last 40 lines of output ---\n{tail}"
        )
