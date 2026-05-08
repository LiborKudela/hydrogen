"""Plotly-based time-series plotting for `Model.record`."""

from __future__ import annotations

import numpy as np


def plot_results(record, filename, show=False, max_vars=None):
    """Plot every column of `record['state']` against `record['time']`.

    `record['state']` is produced by `Model.lambdified_raw_vars`, so it already covers
    every original variable (including those eliminated by trivial-equation removal,
    reconstructed from their substitutions). Columns line up with `record['vars_names']`.

    Pass `max_vars` to clamp the number of traces (handy for very wide systems).
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
