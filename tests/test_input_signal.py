"""Tests for the time-dependent ``Input`` signal type.

An ``Input`` is a *known* function of time ``u(t)`` that drives the system
but is never solved for (it is not a ``Variable`` and never enters the
Jacobian).  Unlike a ``Parameter`` it carries two values at once -- ``u(t_k)``
via ``prev_symbol`` and ``u(t_{k+1})`` via ``symbol`` -- so a Crank-Nicolson
balance integrates the driving term at full second-order accuracy.

The reference problem is Newton cooling toward a moving ambient:

    C * dT/dt = G * (u(t) - T),      tau = C / G

For a linear ramp ``u(t) = a + b*t`` the closed-form solution is

    T(t) = a + b*t - b*tau + (T0 - a + b*tau) * exp(-t/tau)

which CN tracks to O(dt^2).  A constant ``u`` reduces the same model to the
classic exponential relaxation toward ``u`` -- i.e. it behaves exactly like a
``Parameter`` source, which we also check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydrogen.model import DifferentialVariable, Input, Model, Parameter


class DrivenCooling(Model):
    """Single lumped capacity relaxing toward a time-dependent ambient `u(t)`.

    C * der_T = G * (u - T)
    """

    def __init__(self, C, G, func, T0):
        self._C = C
        self._G = G
        self._func = func
        self._T0 = T0
        super().__init__()

    def declare_components(self):
        self.add_component('T', DifferentialVariable(self._T0, "K"))
        self.add_component('C', Parameter(self._C, "J/K"))
        self.add_component('G', Parameter(self._G, "W/K"))
        self.add_component('u', Input(self._func, "K"))

    def declare_equations(self):
        C = self['C'].symbol
        G = self['G'].symbol
        T = self['T'].symbol
        der_T = self['der_T'].symbol
        u = self['u'].symbol
        return [C * der_T - G * (u - T)]


def _run(model, dt, n_steps):
    model.instantiate(max_remove_trival_passes=3)
    model.initialise()
    for _ in range(n_steps):
        model.solve_dae_step(dt)
        model.next_step()
    t = np.asarray(model.record['time'])
    names = list(model.record['vars_names'])
    state = np.asarray(model.record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith('.T'))
    return t, state[:, idx], model


# ---------------------------------------------------------------------------
# Structural: an Input is NOT a solved variable
# ---------------------------------------------------------------------------

def test_input_is_not_a_solved_variable():
    """Only `T` and its `der_T` companion are solved; `u` lives in the param
    block, so it never adds an unknown to the Newton system."""
    model = DrivenCooling(C=500.0, G=5.0, func=lambda t: 300.0, T0=300.0)
    model.instantiate(max_remove_trival_passes=3)
    model.initialise()

    solved = {v.full_name.split('.')[-1] for v in model.active_vars_references}
    assert solved == {"T", "der_T"}

    # exactly one registered Input, with two parameter slots wired up
    assert len(model._input_refs) == 1
    inp, i_cur, i_prev = model._input_refs[0]
    assert i_cur != i_prev


# ---------------------------------------------------------------------------
# Constant input behaves exactly like a Parameter source
# ---------------------------------------------------------------------------

def test_constant_input_relaxes_like_parameter():
    C, G, U, T0 = 500.0, 5.0, 350.0, 300.0
    tau = C / G
    dt = 1.0
    n = 1000          # 10 * tau -> essentially fully relaxed
    t, T, _ = _run(DrivenCooling(C, G, lambda _t: U, T0), dt, n)

    analytic = U + (T0 - U) * np.exp(-t / tau)
    assert np.allclose(T, analytic, rtol=0, atol=5e-3)
    # and it actually reaches the constant target after ~10 time constants
    assert T[-1] == pytest.approx(U, abs=2e-2)


# ---------------------------------------------------------------------------
# Ramp input: requires correct use of BOTH time levels (u(t_k), u(t_{k+1}))
# ---------------------------------------------------------------------------

def test_ramp_input_tracks_closed_form():
    C, G, T0 = 500.0, 5.0, 300.0
    a, b = 300.0, 0.5          # u(t) = 300 + 0.5 t  [K]
    tau = C / G
    dt = 0.5
    n = 400                    # to t = 200 s = 2*tau

    t, T, model = _run(DrivenCooling(C, G, lambda tt: a + b * tt, T0), dt, n)

    analytic = a + b * t - b * tau + (T0 - a + b * tau) * np.exp(-t / tau)
    assert np.allclose(T, analytic, rtol=0, atol=5e-3)

    # the input itself reports the right value at the final time level
    assert model['u'].value == pytest.approx(a + b * t[-1], rel=1e-12)


def test_ramp_input_second_order_convergence():
    """Halving dt should cut the max error by ~4x (CN is O(dt^2)); the
    driving term being integrated at both levels is what makes this hold.
    A naive zero-order-hold on the input would only give O(dt)."""
    C, G, T0 = 500.0, 5.0, 300.0
    a, b = 300.0, 0.5
    tau = C / G
    T_end = 100.0

    def max_err(dt):
        n = int(round(T_end / dt))
        t, T, _ = _run(DrivenCooling(C, G, lambda tt: a + b * tt, T0), dt, n)
        analytic = a + b * t - b * tau + (T0 - a + b * tau) * np.exp(-t / tau)
        return np.max(np.abs(T - analytic))

    e_coarse = max_err(2.0)
    e_fine = max_err(1.0)
    # second order => ratio ~4; allow generous slack but reject first order (~2)
    assert e_coarse / e_fine > 3.0


# ---------------------------------------------------------------------------
# Sinusoidal input: smooth periodic driver, check sampled values + tracking
# ---------------------------------------------------------------------------

def test_sinusoidal_input_is_sampled_each_step():
    C, G, T0 = 200.0, 4.0, 300.0
    amp, omega, mean = 20.0, 2.0 * math.pi / 50.0, 300.0
    func = lambda tt: mean + amp * math.sin(omega * tt)
    dt = 0.5
    n = 200

    t, T, model = _run(DrivenCooling(C, G, func, T0), dt, n)

    # The model stays bounded around the mean (no runaway), and the final
    # reported input equals the analytic sample.
    assert np.all(np.abs(T - mean) < amp + 5.0)
    assert model['u'].value == pytest.approx(func(t[-1]), rel=1e-12)
