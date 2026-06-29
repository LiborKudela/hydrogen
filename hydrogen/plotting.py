"""Plotly-based time-series plotting for `Model.record`."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# `<project_root>/local_results/` -- a single git-ignored sandbox where
# examples + tests dump every artifact (HTML plots, JSONL benchmark logs,
# CSV exports, ...).  Importers can override the location via the env var
# `HYDROGEN_LOCAL_RESULTS` (useful in CI to redirect into a workspace-
# scoped tmpdir).  The directory is materialised lazily by the helpers
# below so just importing `plotting` doesn't create empty folders.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _local_results_root() -> Path:
    override = os.environ.get("HYDROGEN_LOCAL_RESULTS")
    if override:
        return Path(override).expanduser().resolve()
    return _PROJECT_ROOT / "local_results"


def local_results_path(subdir: str, filename: str) -> str:
    """Return `<local_results>/<subdir>/<filename>`, creating the directory.

    `<local_results>` defaults to `<project_root>/local_results/` (git-
    ignored) and can be overridden with the `HYDROGEN_LOCAL_RESULTS` env
    var.  Use this for any artifact a script wants on disk without
    polluting the repo:

        out = local_results_path("tutorials", "fill_vessel.html")
        plot_results(record, out)                      # explicit path

        # or pass the subdir to plot_results directly:
        plot_results(record, "fill_vessel.html", subdir="tutorials")
    """
    out_dir = _local_results_root() / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / filename)


def plot_results(record, filename, show=False, max_vars=None, subdir=None):
    """Plot every column of `record['state']` against `record['time']`.

    `record['state']` is produced by `Model.lambdified_raw_vars`, so it already covers
    every original variable (including those eliminated by trivial-equation removal,
    reconstructed from their substitutions). Columns line up with `record['vars_names']`.

    Pass `max_vars` to clamp the number of traces (handy for very wide systems).

    `subdir`: if given, the plot is written to `<local_results>/<subdir>/<filename>`
    instead of treating `filename` as a cwd-relative path.  Tutorials pass
    `subdir="tutorials"`, benchmarks `subdir="benchmarks"`, tests
    `subdir="tests"`, so artifacts collect under the git-ignored
    `local_results/` sandbox.  Pass an absolute `filename` to bypass this
    entirely.
    """
    import plotly.graph_objects as go

    t = record['time']
    y = np.array(record['state']).T
    vars_names = list(record['vars_names'])

    if y.shape[0] != len(vars_names):
        raise ValueError(
            f"plot_results: state has {y.shape[0]} columns but {len(vars_names)} variable "
            f"names were recorded. Did instantiate() finish before the first record_state()?"
        )

    if max_vars is not None:
        vars_names = vars_names[:max_vars]
        y = y[:len(vars_names)]

    if subdir is not None and not os.path.isabs(filename):
        filename = local_results_path(subdir, filename)

    fig = go.Figure()
    for i, name in enumerate(vars_names):
        fig.add_trace(go.Scatter(
            x=t,
            y=y[i],
            mode='lines',
            name=name,
            line=dict(width=2),
        ))

    fig.update_layout(
        title='Simulation Results',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
    )

    if show:
        fig.show()
    fig.write_html(filename)
    return filename
