"""Trivial-equation removal end-to-end behavior, including the prev-symbol mirror.

The negated-RHS test (`x + y = 0` -> `x = -y`) is a regression for the bug where
`var.symbol == sol[0]` lookup silently failed for non-Symbol RHSs and re-used a
stale `sol_0_prev_symbol` from an earlier equation, corrupting the prev-step
substitution chain. Without the fix in `Model.remove_trivial_equations` the
`x` trace would be coupled to an unrelated previous-symbol and fail this test.
"""

from __future__ import annotations

import numpy as np

from hydrogen.model import Model, Variable


class _IdentityTrivial(Model):
    """`x - y = 0`, `2y - 10 = 0` -> trivial drops `x`, surviving system pins y = 5.

    The `2.0` coefficient is intentional: it forces the lambdified Jacobian to be a
    float matrix, which is required by the numba `np.linalg.solve` overload used in
    `fast_linear_solve`.
    """

    def declare_components(self):
        self.add_component('x', Variable(1.0))
        self.add_component('y', Variable(2.0))

    def declare_equations(self):
        return [
            self['x'].symbol - self['y'].symbol,
            2.0 * self['y'].symbol - 10.0,
        ]


class _NegatedTrivial(Model):
    """`x + y = 0`, `2y - 10 = 0` -> trivial drops `x` with RHS = -y; analytical x = -5, y = 5."""

    def declare_components(self):
        self.add_component('x', Variable(1.0))
        self.add_component('y', Variable(2.0))

    def declare_equations(self):
        return [
            self['x'].symbol + self['y'].symbol,
            2.0 * self['y'].symbol - 10.0,
        ]


def _trace(record, suffix):
    names = list(record['vars_names'])
    state = np.asarray(record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


def test_identity_trivial_eliminates_one_variable():
    model = _IdentityTrivial()
    model.instantiate(max_remove_trival_passes=2)
    model.initialise()

    # `x` got eliminated; `y` survived.
    surviving = [s.name for s in model.improved_vars]
    assert any(s.endswith("y") for s in surviving)
    assert not any(s.endswith("x") for s in surviving)
    assert len(model.improve_subs) >= 1

    # Reconstructed full state still surfaces both variables and they take the right values.
    assert abs(_trace(model.record, ".y")[-1] - 5.0) < 1e-9
    assert abs(_trace(model.record, ".x")[-1] - 5.0) < 1e-9  # x = y = 5


def test_negated_trivial_handles_expression_rhs():
    """Regression: solving `x + y = 0` for `x` yields `-y`, an *expression*, not a Symbol.

    The prev-symbol mirror must derive `x_prev = -y_prev` via xreplace; otherwise the
    integrator silently aliases `x_prev` to a leftover sibling and the value of `x`
    drifts away from `-y` over time. We assert the tighter algebraic identity and the
    final reconstructed value here.
    """
    model = _NegatedTrivial()
    model.instantiate(max_remove_trival_passes=2)
    model.initialise()

    # Take a couple of additional steps to amplify any drift in the prev-symbol layer.
    for _ in range(3):
        model.solve_dae_step(0.1)
        model.next_step()

    x_trace = _trace(model.record, ".x")
    y_trace = _trace(model.record, ".y")

    assert np.allclose(y_trace, 5.0, atol=1e-9)
    assert np.allclose(x_trace, -5.0, atol=1e-9)
    # Algebraic identity x = -y must hold at every recorded step (this is what the
    # buggy code violated).
    assert np.allclose(x_trace, -y_trace, atol=1e-9)
