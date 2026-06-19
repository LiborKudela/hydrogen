"""Tests for the fluid control valves (`IncompressibleValve`, `CompressibleValve`).

Covers the Kv constitutive law (liquid), linear opening trim, signal-driven
opening through the control domain, IEC choked gas flow, the unconnected
opening-port warning, and serialization round-trip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hydrogen import (
    CompressibleValve,
    Constant,
    CoolPropMedium,
    IncompressibleValve,
    Model,
    PressureOutlet,
    PressureSource,
    from_dict,
    to_dict,
)
from hydrogen.ports import PortNotConnectedWarning

_WATER = CoolPropMedium('water', disable_warnings=True)
_AIR = CoolPropMedium('air', disable_warnings=True)


def _solve(model, medium):
    model.instantiate(aditional_modules=medium.modules, max_remove_trival_passes=5)
    model.initialise(n=1)
    model.solve_dae_step(1.0)
    names = list(model.record['vars_names'])
    last = np.asarray(model.record['state'])[-1]

    def val(suffix):
        return last[next(i for i, n in enumerate(names) if n.endswith(suffix))]

    return val


class _LiquidRig(Model):
    def __init__(self, Kv=10.0, opening=1.0, p_in=3e5, p_out=2e5):
        self._Kv = Kv
        self._opening = opening
        self._p_in = p_in
        self._p_out = p_out
        super().__init__()

    def declare_components(self):
        # Large boundary area -> negligible KE correction, near-fixed dp.
        self.add_component('src', PressureSource(_WATER, p_source=self._p_in, T_source=300.0, A=1e-2))
        self.add_component('v', IncompressibleValve(_WATER, Kv=self._Kv, D=0.02))
        self.add_component('out', PressureOutlet(_WATER, p_ambient=self._p_out, T_ambient=300.0))
        self.add_component('cmd', Constant(k=self._opening))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['v'].ports['opening'])
        return []


def test_incompressible_kv_law_holds_at_solution():
    val = _solve(_LiquidRig(Kv=10.0, opening=1.0), _WATER)
    dp = val('.v.p_in') - val('.v.p_out')
    rho = 0.5 * (val('.v.rho_in') + val('.v.rho_out'))
    m_dot = val('.v.m_dot_in')
    # Same regularised law the model implements (dp >> dp_eps so ~ sqrt(dp)).
    g = dp / (dp ** 2 + 1.0 ** 2) ** 0.25
    expected = (10.0 / 36000.0) * np.sqrt(rho) * g
    assert m_dot == pytest.approx(expected, rel=1e-6)
    assert m_dot > 0.0 and dp > 0.0


def test_opening_linearly_scales_flow():
    full = _solve(_LiquidRig(opening=1.0), _WATER)
    half = _solve(_LiquidRig(opening=0.5), _WATER)
    # Near-fixed dp (large boundary area), so flow tracks the opening linearly.
    ratio = half('.v.m_dot_in') / full('.v.m_dot_in')
    assert ratio == pytest.approx(0.5, abs=2e-2)


def test_shut_valve_blocks_flow():
    val = _solve(_LiquidRig(opening=0.0), _WATER)
    assert abs(val('.v.m_dot_in')) < 1e-6


def test_signal_controlled_opening_drives_valve():
    # The opening is set purely through the RealSignal port from a control
    # block -- exercise that the signal domain actuates the fluid valve.
    val = _solve(_LiquidRig(opening=0.3), _WATER)
    assert val('.v.opening') == pytest.approx(0.3)
    assert val('.v.m_dot_in') > 0.0


class _GasRig(Model):
    def __init__(self, p_out, xT=0.7):
        self._p_out = p_out
        self._xT = xT
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(_AIR, p_source=8e5, T_source=300.0, A=1e-2))
        self.add_component('v', CompressibleValve(_AIR, Kv=5.0, D=0.02, xT=self._xT, gamma=1.4, p_eps=200.0))
        self.add_component('out', PressureOutlet(_AIR, p_ambient=self._p_out, T_ambient=300.0))
        self.add_component('cmd', Constant(k=1.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['v'].ports['opening'])
        return []


def test_compressible_flow_chokes():
    # Below the critical pressure ratio the flow grows with dp; past xT it
    # saturates (becomes independent of downstream pressure).
    m_low = _solve(_GasRig(p_out=7.0e5), _AIR)('.v.m_dot_in')   # small dp
    m_mid = _solve(_GasRig(p_out=4.0e5), _AIR)('.v.m_dot_in')   # x ~ 0.5
    m_choke1 = _solve(_GasRig(p_out=2.0e5), _AIR)('.v.m_dot_in')  # x > xT
    m_choke2 = _solve(_GasRig(p_out=1.1e5), _AIR)('.v.m_dot_in')  # deeper choke
    assert m_low < m_mid < m_choke1                  # monotone before choke
    # Choked: further lowering p_out barely changes flow.
    assert m_choke2 == pytest.approx(m_choke1, rel=1e-2)


def test_compressible_expansion_factor_below_incompressible():
    # At an equal moderate dp the gas (Y < 1) passes less than the liquid law
    # would predict (Y = 1).  Compare the dimensionless Y implied by the flow.
    val = _solve(_GasRig(p_out=5.0e5), _AIR)
    dp = val('.v.p_in') - val('.v.p_out')
    rho = 0.5 * (val('.v.rho_in') + val('.v.rho_out'))
    m_dot = val('.v.m_dot_in')
    g = dp / (dp ** 2 + 1.0 ** 2) ** 0.25
    incompressible = (5.0 / 36000.0) * np.sqrt(rho) * g
    Y_implied = m_dot / incompressible
    assert 0.66 < Y_implied < 1.0          # expansion factor in (2/3, 1)


def test_unconnected_opening_warns():
    class Rig(Model):
        def declare_components(self):
            self.add_component('src', PressureSource(_WATER, p_source=3e5, T_source=300.0, A=1e-2))
            self.add_component('v', IncompressibleValve(_WATER, Kv=5.0, D=0.02))
            self.add_component('out', PressureOutlet(_WATER, p_ambient=2e5, T_ambient=300.0))

        def declare_equations(self):
            self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
            self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
            return []  # opening left unconnected

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            Rig().instantiate(aditional_modules=_WATER.modules, max_remove_trival_passes=5)
        except Exception:
            pass
    assert any(issubclass(w.category, PortNotConnectedWarning) for w in caught)


def test_valve_serialization_round_trip():
    rig = _LiquidRig(Kv=12.5, opening=0.7)
    rig.declare_equations()  # wire so reflective dump captures connections
    d = to_dict(rig)
    vspec = d['components']['v']
    assert vspec['type'] == 'IncompressibleValve'
    assert vspec['params']['Kv'] == 12.5
    assert vspec['params']['D'] == 0.02
    rebuilt = from_dict(d)
    assert 'v' in rebuilt.components
