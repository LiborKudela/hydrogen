"""Tests for the fluid control valves (`IncompressibleValve`, `CompressibleValve`).

Covers the Kv constitutive law (liquid), linear opening trim, signal-driven
opening through the control domain, IEC choked gas flow, the unconnected
opening-port warning, and serialization round-trip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hydrogen import CoolPropMedium, Model, from_dict, to_dict
from hydrogen.components.control.control_components import Constant
from hydrogen.components.materials import WallMaterial
from hydrogen.components.thermofluid.assemblies import Valve, WallLayer
from hydrogen.components.thermofluid.flow import (
    CompressibleValve,
    IncompressibleValve,
    PressureOutlet,
    PressureSource,
)
from hydrogen.components.thermofluid.permeation import (
    H2,
    H2_IN_AUSTENITIC,
    SpecifiedFlux,
    SteadyRichardson,
)
from hydrogen.ports import PortNotConnectedWarning

_WATER = CoolPropMedium('water', disable_warnings=True)
_AIR = CoolPropMedium('air', disable_warnings=True)
_H2 = CoolPropMedium('hydrogen', disable_warnings=True)
_STEEL = WallMaterial(name="steel", rho=7850.0, cp=500.0, k=15.0)


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
    # The valve is a single-cell SegmentedChannel: face 0 = inlet, face 1 = outlet.
    dp = val('.v.p_0') - val('.v.p_1')
    rho = 0.5 * (val('.v.rho_0') + val('.v.rho_1'))
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


class _ChokedLiquidRig(Model):
    """Water valve near-vacuum discharge: with `p_vap` set, the ISA liquid
    choking clamps dp at FL^2*(p_up - FF*p_vap)."""

    def __init__(self, p_out, p_vap=None, trim_exp=1.0, opening=1.0):
        self._p_out = p_out
        self._p_vap = p_vap
        self._trim_exp = trim_exp
        self._opening = opening
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(
            _WATER, p_source=3e5, T_source=300.0, A=1e-2))
        self.add_component('v', IncompressibleValve(
            _WATER, Kv=10.0, D=0.02, p_vap=self._p_vap,
            trim_exp=self._trim_exp, FL=0.9, FF=0.96, p_eps=100.0,
            multiphase="HEM"))
        self.add_component('out', PressureOutlet(
            _WATER, p_ambient=self._p_out, T_ambient=300.0))
        self.add_component('cmd', Constant(k=self._opening))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['v'].ports['opening'])
        return []


def test_liquid_valve_chokes_at_vapor_pressure():
    """With p_vap set, the liquid flow saturates once the downstream pressure
    falls below p_up - FL^2*(p_up - FF*p_vap): valve cavitation (flashing)
    makes the flow independent of the back pressure."""
    pv = 3536.0                                    # ~ p_sat(water, 300 K)
    m_mild = _solve(_ChokedLiquidRig(p_out=2.0e5, p_vap=pv),
                    _WATER)('.v.m_dot_in')         # dp=1 bar, not choked
    m_choke1 = _solve(_ChokedLiquidRig(p_out=0.5e5, p_vap=pv),
                      _WATER)('.v.m_dot_in')       # dp > dp_choked
    m_choke2 = _solve(_ChokedLiquidRig(p_out=0.1e5, p_vap=pv),
                      _WATER)('.v.m_dot_in')       # deeper vacuum
    assert m_mild < m_choke1
    # Choked: further lowering p_out barely changes the flow.
    assert m_choke2 == pytest.approx(m_choke1, rel=1e-2)
    # And the choked flow matches the ISA clamp value.
    dp_choked = 0.9 ** 2 * (3.0e5 - 0.96 * pv)
    expected = (10.0 / 36000.0) * np.sqrt(996.5) * np.sqrt(dp_choked)
    assert m_choke2 == pytest.approx(expected, rel=2e-2)


def test_unchoked_flow_matches_plain_kv_law():
    """Away from the choke point the p_vap-enabled law reduces to the plain
    Kv law (same rig with and without p_vap agree)."""
    m_plain = _solve(_ChokedLiquidRig(p_out=2.5e5), _WATER)('.v.m_dot_in')
    m_choked = _solve(_ChokedLiquidRig(p_out=2.5e5, p_vap=3536.0),
                      _WATER)('.v.m_dot_in')
    assert m_choked == pytest.approx(m_plain, rel=1e-3)


def test_trim_exponent_shapes_opening_curve():
    """trim_exp=2 gives quadratic (ball-valve-like) opening behaviour: at
    half opening the flow is ~ 25 % of full (linear trim gives ~ 50 %)."""
    full = _solve(_ChokedLiquidRig(p_out=2.0e5, trim_exp=2.0),
                  _WATER)('.v.m_dot_in')
    half = _solve(_ChokedLiquidRig(p_out=2.0e5, trim_exp=2.0, opening=0.5),
                  _WATER)('.v.m_dot_in')
    assert half / full == pytest.approx(0.25, abs=2e-2)


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
    # The valve is a single-cell SegmentedChannel: face 0 = inlet, face 1 = outlet.
    dp = val('.v.p_0') - val('.v.p_1')
    rho = 0.5 * (val('.v.rho_0') + val('.v.rho_1'))
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
    assert vspec['type'] == 'hydrogen.thermofluid.IncompressibleValve'
    assert vspec['params']['Kv'] == 12.5
    assert vspec['params']['D'] == 0.02
    rebuilt = from_dict(d)
    assert 'v' in rebuilt.components


# --- pluggable dynamic-level momentum (single-volume valve body) -------------

def _solve_n(model, medium, dt, n):
    model.instantiate(aditional_modules=medium.modules, max_remove_trival_passes=5)
    model.initialise(n=1)
    for _ in range(n):
        model.solve_dae_step(dt)
    names = list(model.record['vars_names'])
    last = np.asarray(model.record['state'])[-1]

    def val(suffix):
        return last[next(i for i, nm in enumerate(names) if nm.endswith(suffix))]

    return val


class _DynValveRig(Model):
    """Bare (wall-less) incompressible valve at a chosen dynamic level, seeded
    at the mid pressure so the body cell starts near its steady state."""

    def __init__(self, dynamic, p_in=3e5, p_out=2e5):
        self._dynamic = dynamic
        self._p_in = p_in
        self._p_out = p_out
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(_AIR, p_source=self._p_in, T_source=300.0, A=1e-2))
        self.add_component('v', IncompressibleValve(
            _AIR, Kv=10.0, D=0.02, opening=1.0, dynamic=self._dynamic,
            L_body=0.05, p_init=0.5 * (self._p_in + self._p_out)))
        self.add_component('out', PressureOutlet(_AIR, p_ambient=self._p_out, T_ambient=300.0))
        self.add_component('cmd', Constant(k=1.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['v'].ports['opening'])
        return []


@pytest.mark.parametrize("dynamic,dt,n", [("advective", 1e-2, 60),
                                          ("compressible", 1e-3, 300)])
def test_dynamic_valve_body_matches_static_steady_flow(dynamic, dt, n):
    """The dynamic single-volume valve (two staggered throttle faces around a
    storing body cell) reproduces the static single-throttle steady flow: the
    per-face law is scaled so the series stack composes to the lumped drop."""
    stat = _solve_n(_DynValveRig("static"), _AIR, 1.0, 1)('.v.m_dot_in')
    dyn = _solve_n(_DynValveRig(dynamic), _AIR, dt, n)('.v.m_dot_in')
    assert dyn == pytest.approx(stat, rel=0.12)


def test_dynamic_valve_body_pressure_equilibrates():
    """At steady state the body-cell pressure sits between the two ports (equal
    split across the two throttle faces)."""
    val = _solve_n(_DynValveRig("compressible"), _AIR, 1e-3, 300)
    pc = val('.v.pc_0')
    assert val('.v.p_1') < pc < val('.v.p_0')


def test_acoustic_valve_rejected():
    with pytest.raises(NotImplementedError, match="acoustic"):
        IncompressibleValve(_WATER, Kv=5.0, D=0.02, dynamic="acoustic")


def test_choked_valve_rejects_dynamic():
    with pytest.raises(NotImplementedError):
        IncompressibleValve(_WATER, Kv=5.0, D=0.02, p_vap=3000.0,
                            dynamic="compressible")


# --- Valve assembly: equivalent cylindrical wall (thermal + permeation) ------

class _AssemblyRig(Model):
    def __init__(self, layers, medium=_WATER, flow_law="incompressible",
                 outer_thermal="convective", p_in=3e5, p_out=2e5, dynamic="static"):
        self._layers = layers
        self._medium = medium
        self._flow_law = flow_law
        self._outer = outer_thermal
        self._p_in = p_in
        self._p_out = p_out
        self._dynamic = dynamic
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(self._medium, p_source=self._p_in, T_source=320.0, A=1e-2))
        self.add_component('v', Valve(
            self._medium, D=0.02, Kv=10.0, flow_law=self._flow_law,
            L_body=0.08, layers=self._layers, outer_thermal=self._outer,
            h_ext=25.0, T_ext=290.0, dynamic=self._dynamic, T_wall_init=320.0,
            p_init=self._p_in,
            multiphase=("HEM" if self._medium is _WATER else "single")))
        self.add_component('out', PressureOutlet(self._medium, p_ambient=self._p_out, T_ambient=320.0))
        self.add_component('cmd', Constant(k=1.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['v'].ports['inlet'])
        self.connect(self['v'].ports['outlet'], self['out'].ports['inlet'])
        self.connect(self['cmd'].ports['y'], self['v'].ports['opening'])
        return []


def test_valve_assembly_is_the_catalog_valve():
    from hydrogen.serialization.registry import builtin_registry
    assert builtin_registry()['hydrogen.thermofluid.Valve'] is Valve


def test_valve_assembly_thermal_wall_passes_kv_flow():
    layers = [WallLayer(material=_STEEL, thickness=0.003, dynamic=True)]
    val = _solve_n(_AssemblyRig(layers), _WATER, 1.0, 1)
    dp = val('.v.valve.p_0') - val('.v.valve.p_1')
    m = val('.v.valve.m_dot_in')
    rho = 0.5 * (val('.v.valve.rho_0') + val('.v.valve.rho_1'))
    g = dp / (dp ** 2 + 1.0 ** 2) ** 0.25
    assert m == pytest.approx((10.0 / 36000.0) * np.sqrt(rho) * g, rel=1e-5)
    assert m > 0 and dp > 0


def test_valve_assembly_specified_permeation_leaks():
    perm = SpecifiedFlux(H2, leak_rate=5.0, scaling="linear")
    layers = [WallLayer(material=_STEEL, thickness=0.003, permeation=perm)]
    val = _solve_n(_AssemblyRig(layers, medium=_H2, p_in=5e5, p_out=3e5), _H2, 1.0, 1)
    assert abs(val('.v.m_dot_leak_env')) > 0.0


def test_valve_assembly_physics_permeation_builds():
    layers = [WallLayer(material=_STEEL, thickness=0.003,
                        permeation=SteadyRichardson(H2_IN_AUSTENITIC))]
    val = _solve_n(_AssemblyRig(layers, medium=_H2, p_in=5e5, p_out=3e5), _H2, 1.0, 1)
    # Leak is finite (austenitic steel at moderate T => tiny but well-defined).
    assert np.isfinite(val('.v.m_dot_leak_env'))


def test_valve_assembly_serialization_round_trip():
    perm = SpecifiedFlux(H2, leak_rate=5.0, scaling="linear")
    layers = [WallLayer(material=_STEEL, thickness=0.003, permeation=perm)]
    rig = _AssemblyRig(layers, medium=_H2)
    rig.declare_equations()
    d = to_dict(rig)
    assert d['components']['v']['type'] == 'hydrogen.thermofluid.Valve'
    rebuilt = from_dict(d)
    assert 'v' in rebuilt.components


def test_valve_assembly_compressible_flow_law():
    layers = [WallLayer(material=_STEEL, thickness=0.003)]
    val = _solve_n(_AssemblyRig(layers, medium=_AIR, flow_law="compressible",
                                p_in=8e5, p_out=5e5), _AIR, 1.0, 1)
    assert val('.v.valve.m_dot_in') > 0.0
