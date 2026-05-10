"""Make `hydrogen` importable when the package isn't installed.

If you `pip install -e .` first, this is a no-op. Otherwise we just prepend the project
root to `sys.path` so `import hydrogen` resolves directly from the source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def local_results_path():
    """Drop test artifacts under `<project_root>/local_results/tests/`.

    Usage:
        def test_something(local_results_path):
            out = local_results_path("my_plot.html")
            plot_results(record, out)

    The directory is git-ignored, so artifacts never pollute the repo.
    Override the root with the `HYDROGEN_LOCAL_RESULTS` env var.
    """
    from hydrogen import local_results_path as _resolve

    def _make(filename: str) -> str:
        return _resolve("tests", filename)

    return _make
