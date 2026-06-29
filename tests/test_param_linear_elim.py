"""Parameter-coefficient linear elimination (`instantiate(eliminate_param_linear=...)`).

The trivial / linear-block reducers normally only treat *Numbers* as constant
coefficients, so a wiring like

    alpha + k * beta == 0        (k a Parameter)

is opaque to them: `k * beta` is a product of two symbols, which the
numeric-only classifier rejects as nonlinear.  With parameter-coefficient
elimination enabled, parameters are folded into the constant coefficient field,
so the equation is recognised as linear in `{alpha, beta}` and `alpha` is
substituted out as `-k * beta`.

These tests verify:
  1. With the flag OFF, `alpha` is NOT eliminated (legacy behaviour).
  2. With the flag ON, `alpha` IS eliminated -> a strictly smaller system.
  3. The physical solution is identical either way (the substitution is exact).
  4. Guard: a wiring whose coefficient is a time-varying `Input` is left alone
     even with the flag on (the prev-step mirror can't carry a dynamic param).
"""

from __future__ import annotations

import numpy as np

from hydrogen.model import Input, Model, Parameter, Variable


class _ParamCoeffWiring(Model):
    """Two variables wired by a PARAMETER coefficient:

        alpha + k * beta == 0      (k = 3.0, a Parameter)
        2 * beta - 10    == 0      (pins beta = 5)

    => beta = 5, alpha = -k * beta = -15.
    """

    def declare_components(self):
        self.add_component('alpha', Variable(1.0))
        self.add_component('beta', Variable(2.0))
        self.add_component('k', Parameter(3.0))

    def declare_equations(self):
        alpha = self['alpha'].symbol
        beta = self['beta'].symbol
        k = self['k'].symbol
        return [alpha + k * beta, 2.0 * beta - 10.0]


class _InputCoeffWiring(Model):
    """Same shape, but the coefficient is a time-varying `Input` `u(t)`:

        alpha + u * beta == 0
        2 * beta - 10    == 0

    The flag must NOT eliminate `alpha` here: its substitution `-u * beta`
    references a dynamic parameter that the prev-step mirror cannot carry to the
    previous time level.
    """

    def declare_components(self):
        self.add_component('alpha', Variable(1.0))
        self.add_component('beta', Variable(2.0))
        self.add_component('u', Input(lambda t: 3.0, "1"))

    def declare_equations(self):
        alpha = self['alpha'].symbol
        beta = self['beta'].symbol
        u = self['u'].symbol
        return [alpha + u * beta, 2.0 * beta - 10.0]


def _trace(record, suffix):
    names = list(record['vars_names'])
    state = np.asarray(record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


def test_param_coeff_not_eliminated_by_default():
    """Flag OFF: legacy reducer can't touch `alpha + k*beta`, so both variables
    survive -- but the Newton solve still finds the right answer."""
    model = _ParamCoeffWiring()
    model.instantiate(max_remove_trival_passes=2, eliminate_param_linear=False)
    model.initialise()

    assert model.n_v == 2  # alpha and beta both survive
    assert abs(_trace(model.record, ".beta")[-1] - 5.0) < 1e-9
    assert abs(_trace(model.record, ".alpha")[-1] - (-15.0)) < 1e-9


def test_param_coeff_eliminated_with_flag():
    """Flag ON: `alpha` is substituted out as `-k*beta`, leaving a strictly
    smaller (1-variable) system, and the reconstructed answer is unchanged."""
    model = _ParamCoeffWiring()
    model.instantiate(max_remove_trival_passes=2, eliminate_param_linear=True)
    model.initialise()

    # Strictly fewer surviving variables than the default run above.
    assert model.n_v == 1
    surviving = [s.name for s in model.improved_vars]
    assert any(s.endswith("beta") for s in surviving)
    assert not any(s.endswith("alpha") for s in surviving)
    assert len(model.improve_subs) >= 1

    # The eliminated variable is reconstructed exactly from `-k*beta`.
    assert abs(_trace(model.record, ".beta")[-1] - 5.0) < 1e-9
    assert abs(_trace(model.record, ".alpha")[-1] - (-15.0)) < 1e-9


def test_param_coeff_flag_matches_default_solution():
    """The two configurations must agree on the physical solution to machine
    precision (the elimination is an exact algebraic substitution)."""
    m_off = _ParamCoeffWiring()
    m_off.instantiate(max_remove_trival_passes=2, eliminate_param_linear=False)
    m_off.initialise()

    m_on = _ParamCoeffWiring()
    m_on.instantiate(max_remove_trival_passes=2, eliminate_param_linear=True)
    m_on.initialise()

    assert m_on.n_v < m_off.n_v
    for suffix in (".alpha", ".beta"):
        assert np.isclose(_trace(m_off.record, suffix)[-1],
                          _trace(m_on.record, suffix)[-1], atol=1e-9)


def test_dynamic_input_coefficient_is_not_eliminated():
    """Guard: even with the flag on, a coefficient that is a time-varying Input
    must NOT be eliminated -- `alpha` stays in the system."""
    model = _InputCoeffWiring()
    model.instantiate(max_remove_trival_passes=2, eliminate_param_linear=True)
    model.initialise()

    assert model.n_v == 2  # alpha was NOT eliminated (guard fired)
    surviving = [s.name for s in model.improved_vars]
    assert any(s.endswith("alpha") for s in surviving)

    # Still solves correctly: alpha = -u*beta = -3*5 = -15.
    assert abs(_trace(model.record, ".beta")[-1] - 5.0) < 1e-9
    assert abs(_trace(model.record, ".alpha")[-1] - (-15.0)) < 1e-9
