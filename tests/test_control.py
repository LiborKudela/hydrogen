"""Tests for the control / signal-block domain (`hydrogen.components.control`).

Covers the `RealSignal` connector (value-equality wiring + one-to-many
fan-out), the source / maths / continuous blocks, the unconnected-input
warning, and serialization auto-registration of signal blocks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hydrogen import (
    Add,
    Constant,
    Feedback,
    FirstOrder,
    Gain,
    Integrator,
    Limiter,
    Model,
    PID,
    Product,
    Ramp,
    Sine,
    Step,
    Sum,
    from_dict,
    to_dict,
)
from hydrogen.ports import PortAlreadyConnectedError, PortNotConnectedWarning


def _solve(model, *, steps=1, dt=1.0):
    model.instantiate(max_remove_trival_passes=5)
    model.initialise(n=1)
    for _ in range(steps):
        model.solve_dae_step(dt)
        model.next_step()
    names = list(model.record['vars_names'])
    state = np.asarray(model.record['state'])
    t = np.asarray(model.record['time'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return t, trace


# --- algebraic blocks -------------------------------------------------------


def test_constant_gain_chain():
    class S(Model):
        def declare_components(self):
            self.add_component('c', Constant(k=2.0))
            self.add_component('g', Gain(k=3.0))

        def declare_equations(self):
            self.connect(self['c'].ports['y'], self['g'].ports['u'])
            return []

    _, trace = _solve(S())
    assert trace('.g.y')[-1] == pytest.approx(6.0)


def test_signal_output_fans_out_to_many_inputs():
    class S(Model):
        def declare_components(self):
            self.add_component('c', Constant(k=2.0))
            self.add_component('ga', Gain(k=3.0))
            self.add_component('gb', Gain(k=5.0))

        def declare_equations(self):
            # One output -> two inputs (allowed for signal outputs only).
            self.connect(self['c'].ports['y'], self['ga'].ports['u'])
            self.connect(self['c'].ports['y'], self['gb'].ports['u'])
            return []

    _, trace = _solve(S())
    assert trace('.ga.y')[-1] == pytest.approx(6.0)
    assert trace('.gb.y')[-1] == pytest.approx(10.0)


def test_feedback_and_add():
    class S(Model):
        def declare_components(self):
            self.add_component('a', Constant(k=7.0))
            self.add_component('b', Constant(k=4.0))
            self.add_component('fb', Feedback())
            self.add_component('add', Add(k1=1.0, k2=2.0))

        def declare_equations(self):
            self.connect(self['a'].ports['y'], self['fb'].ports['u1'])
            self.connect(self['b'].ports['y'], self['fb'].ports['u2'])
            self.connect(self['a'].ports['y'], self['add'].ports['u1'])
            self.connect(self['b'].ports['y'], self['add'].ports['u2'])
            return []

    _, trace = _solve(S())
    assert trace('.fb.y')[-1] == pytest.approx(3.0)        # 7 - 4
    assert trace('.add.y')[-1] == pytest.approx(7.0 + 2 * 4.0)


def test_sum_and_product():
    class S(Model):
        def declare_components(self):
            self.add_component('a', Constant(k=2.0))
            self.add_component('b', Constant(k=3.0))
            self.add_component('s', Sum(n=2, weights=[1.0, 4.0]))
            self.add_component('p', Product())

        def declare_equations(self):
            self.connect(self['a'].ports['y'], self['s'].ports['u_0'])
            self.connect(self['b'].ports['y'], self['s'].ports['u_1'])
            self.connect(self['a'].ports['y'], self['p'].ports['u1'])
            self.connect(self['b'].ports['y'], self['p'].ports['u2'])
            return []

    _, trace = _solve(S())
    assert trace('.s.y')[-1] == pytest.approx(2.0 + 4 * 3.0)
    assert trace('.p.y')[-1] == pytest.approx(6.0)


def test_limiter_saturates_and_passes():
    class S(Model):
        def declare_components(self):
            self.add_component('hi', Constant(k=5.0))     # above the cap
            self.add_component('mid', Constant(k=0.4))    # inside range
            self.add_component('lim_hi', Limiter(lo=0.0, hi=1.0, eps=1e-4))
            self.add_component('lim_mid', Limiter(lo=0.0, hi=1.0, eps=1e-4))

        def declare_equations(self):
            self.connect(self['hi'].ports['y'], self['lim_hi'].ports['u'])
            self.connect(self['mid'].ports['y'], self['lim_mid'].ports['u'])
            return []

    _, trace = _solve(S())
    assert trace('.lim_hi.y')[-1] == pytest.approx(1.0, abs=1e-3)
    assert trace('.lim_mid.y')[-1] == pytest.approx(0.4, abs=1e-3)


# --- continuous blocks ------------------------------------------------------


def test_integrator_of_constant_is_linear_in_time():
    class S(Model):
        def declare_components(self):
            self.add_component('c', Constant(k=1.0))
            self.add_component('i', Integrator(k=1.0))

        def declare_equations(self):
            self.connect(self['c'].ports['y'], self['i'].ports['u'])
            return []

    t, trace = _solve(S(), steps=20, dt=0.1)
    y = trace('.i.y')
    assert y[-1] == pytest.approx(t[-1], abs=1e-6)         # integral of 1 = t


def test_first_order_step_response_reaches_steady_state():
    class S(Model):
        def declare_components(self):
            self.add_component('u', Constant(k=1.0))
            self.add_component('lag', FirstOrder(T=0.5, k=2.0))

        def declare_equations(self):
            self.connect(self['u'].ports['y'], self['lag'].ports['u'])
            return []

    _, trace = _solve(S(), steps=80, dt=0.1)               # >> 5*T
    assert trace('.lag.y')[-1] == pytest.approx(2.0, abs=1e-2)   # k*u


def test_pid_closed_loop_tracks_setpoint():
    class Loop(Model):
        def declare_components(self):
            self.add_component('sp', Constant(k=1.0))
            self.add_component('fb', Feedback())
            self.add_component('pid', PID(kp=2.0, ki=1.5, kd=0.0, Tf=0.1))
            self.add_component('plant', FirstOrder(T=0.5, k=1.0))

        def declare_equations(self):
            self.connect(self['sp'].ports['y'], self['fb'].ports['u1'])
            self.connect(self['plant'].ports['y'], self['fb'].ports['u2'])
            self.connect(self['fb'].ports['y'], self['pid'].ports['u'])
            self.connect(self['pid'].ports['y'], self['plant'].ports['u'])
            return []

    _, trace = _solve(Loop(), steps=300, dt=0.1)
    # Integral action removes steady-state offset -> output tracks setpoint.
    assert trace('.plant.y')[-1] == pytest.approx(1.0, abs=2e-2)


# --- wiring rules -----------------------------------------------------------


def test_two_outputs_into_one_input_is_rejected():
    class S(Model):
        def declare_components(self):
            self.add_component('a', Constant(k=1.0))
            self.add_component('b', Constant(k=2.0))
            self.add_component('g', Gain(k=1.0))

        def declare_equations(self):
            self.connect(self['a'].ports['y'], self['g'].ports['u'])
            self.connect(self['b'].ports['y'], self['g'].ports['u'])  # input is single-use
            return []

    with pytest.raises(PortAlreadyConnectedError):
        S().instantiate(max_remove_trival_passes=5)


def test_unconnected_input_warns():
    class S(Model):
        def declare_components(self):
            self.add_component('g', Gain(k=2.0))   # 'u' left open

        def declare_equations(self):
            return []

    m = S()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            m.instantiate(max_remove_trival_passes=5)
        except Exception:
            pass  # singular system may fail later; we only assert the warning
    assert any(issubclass(w.category, PortNotConnectedWarning) for w in caught)


# --- serialization ----------------------------------------------------------


def test_signal_blocks_serialize_round_trip():
    class S(Model):
        def declare_components(self):
            self.add_component('step', Step(height=2.0, start_time=0.5, offset=0.1))
            self.add_component('g', Gain(k=3.0))

        def declare_equations(self):
            self.connect(self['step'].ports['y'], self['g'].ports['u'])
            return []

    s = S()
    s.declare_equations()  # wire so the reflective dump captures the connection
    d = to_dict(s)
    assert d['components']['step']['type'] == 'hydrogen.control.Step'
    assert d['components']['step']['params'] == {
        'height': 2.0, 'start_time': 0.5, 'offset': 0.1, 'unit': None}
    rebuilt = from_dict(d)
    assert sorted(rebuilt.components) == ['g', 'step']
