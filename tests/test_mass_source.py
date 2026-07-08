"""Tests for the mass-flow-imposed inlet boundary (`MassSource`).

`MassSource` is the flow-imposed dual of `PressureSource`: it pins the delivered
mass flow (and the reservoir temperature) while letting the downstream network
set the pressure.  Covered here:
  * the imposed mass flow propagates to the outlet (continuity),
  * the injected enthalpy matches the reservoir state at the local pressure,
  * `m_control=True` lets a control block drive the flow through `m_set`,
  * an unconnected `m_set` port warns,
  * serialization round-trip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hydrogen import CoolPropMedium, Model, from_dict, to_dict
from hydrogen.components.control.control_components import Constant
from hydrogen.components.thermofluid.flow import (
    MassSource,
    PressureOutlet,
    StraightPipe,
)
from hydrogen.ports import PortNotConnectedWarning

_WATER = CoolPropMedium('water', disable_warnings=True)


def _solve(model, medium):
    model.instantiate(aditional_modules=medium.modules, max_remove_trival_passes=5)
    model.initialise(n=1)
    model.solve_dae_step(1.0)
    names = list(model.record['vars_names'])
    last = np.asarray(model.record['state'])[-1]

    def val(suffix):
        return last[next(i for i, n in enumerate(names) if n.endswith(suffix))]

    return val


class _SourceRig(Model):
    """`MassSource -> StraightPipe -> PressureOutlet`.

    A fixed-pressure sink terminates the line so the floating source pressure
    has a well-defined level.
    """

    def __init__(self, m_flow=0.05, T_source=300.0, p_sink=2e5, m_control=False):
        self._m_flow = m_flow
        self._T_source = T_source
        self._p_sink = p_sink
        self._m_control = m_control
        super().__init__()

    def declare_components(self):
        self.add_component('src', MassSource(
            _WATER, m_flow=self._m_flow, T_source=self._T_source,
            p_init=self._p_sink, m_control=self._m_control))
        self.add_component('pipe', StraightPipe(
            _WATER, D=0.02, L=1.0, epsilon=1e-5,
            z_in=0.0, z_out=0.0, n_segments=2, adiabatic=True))
        self.add_component('out', PressureOutlet(
            _WATER, p_ambient=self._p_sink, T_ambient=self._T_source))
        if self._m_control:
            self.add_component('cmd', Constant(k=self._m_flow))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['pipe'].ports['inlet'])
        self.connect(self['pipe'].ports['outlet'], self['out'].ports['inlet'])
        if self._m_control:
            self.connect(self['cmd'].ports['y'], self['src'].ports['m_set'])
        return []


def test_imposed_mass_flow_reaches_outlet():
    """Continuity: the source delivers exactly `m_flow` downstream, so under
    the 'flow into me' convention its own `m_dot_out` is `-m_flow`."""
    m_flow = 0.05
    val = _solve(_SourceRig(m_flow=m_flow), _WATER)
    assert val('.src.m_dot_out') == pytest.approx(-m_flow, rel=1e-8)
    # The pipe inlet sees the positive delivered flow.
    assert val('.pipe.m_dot_in') == pytest.approx(m_flow, rel=1e-6)


def test_injected_enthalpy_matches_reservoir_state():
    """The outlet enthalpy equals the reservoir enthalpy evaluated at the
    local (downstream-set) boundary pressure."""
    T_source = 310.0
    val = _solve(_SourceRig(m_flow=0.02, T_source=T_source, p_sink=3e5), _WATER)
    p_out = val('.src.p_out')
    h_expected = _WATER.eval_h_pT(float(p_out), T_source)
    assert val('.src.h_out') == pytest.approx(h_expected, rel=1e-6)


def test_zero_flow_gives_no_delivery():
    val = _solve(_SourceRig(m_flow=0.0), _WATER)
    assert abs(val('.src.m_dot_out')) < 1e-9
    assert abs(val('.pipe.m_dot_in')) < 1e-6


def test_signal_controlled_mass_flow_drives_source():
    """With `m_control=True` the delivered flow comes purely from the control
    signal wired into `m_set`."""
    m_flow = 0.03
    val = _solve(_SourceRig(m_flow=m_flow, m_control=True), _WATER)
    assert val('.src.m_flow') == pytest.approx(m_flow, rel=1e-8)
    assert val('.src.m_dot_out') == pytest.approx(-m_flow, rel=1e-6)


def test_unconnected_m_set_warns():
    class Rig(Model):
        def declare_components(self):
            self.add_component('src', MassSource(
                _WATER, m_flow=0.05, T_source=300.0, m_control=True))
            self.add_component('out', PressureOutlet(
                _WATER, p_ambient=2e5, T_ambient=300.0))

        def declare_equations(self):
            self.connect(self['src'].ports['outlet'], self['out'].ports['inlet'])
            return []  # m_set left unconnected

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            Rig().instantiate(aditional_modules=_WATER.modules,
                              max_remove_trival_passes=5)
        except Exception:
            pass
    assert any(issubclass(w.category, PortNotConnectedWarning) for w in caught)


def test_mass_source_serialization_round_trip():
    rig = _SourceRig(m_flow=0.075, T_source=305.0)
    rig.declare_equations()  # wire so reflective dump captures connections
    d = to_dict(rig)
    sspec = d['components']['src']
    assert sspec['type'] == 'hydrogen.thermofluid.MassSource'
    assert sspec['params']['m_flow'] == 0.075
    assert sspec['params']['T_source'] == 305.0
    rebuilt = from_dict(d)
    assert 'src' in rebuilt.components
