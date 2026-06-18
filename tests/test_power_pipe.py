"""Tests for the power-engineering domain (`ConjugatePipe`) and the fluid
`heat_port` mechanism it is built on.

Covered:
  * `heat_port=False` (default) is adiabatic and exposes no thermal ports.
  * `heat_port=True` exposes a `wall` `ThermalPort_TQ` per segment and warns
    (`PortNotConnectedWarning`) if those ports are left unconnected.
  * the deprecated `adiabatic=` flag still works / warns appropriately.
  * the corrected energy balance conserves energy: the total fluid enthalpy
    flux change equals the sum of the per-segment wall heats (this is what the
    q/m_dot and per-segment-area fixes buy us).
  * a `FixedTemperature` on each segment wall reproduces the old
    "convect to a fixed wall temperature" behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hydrogen import (
    AmbientInlet,
    ConjugatePipe,
    CoolPropMedium,
    FixedTemperature,
    Model,
    StraightPipe,
)
from hydrogen.components.thermal import ThermalPort_TQ
from hydrogen.ports import PortNotConnectedWarning

# One shared medium for the whole module (CoolProp table build is expensive).
AIR = CoolPropMedium('air', disable_warnings=True, backend="BICUBIC&HEOS", scalar_cache_maxsize=1000)

D = 0.05
L = 2.0
EPS = 1e-4


def _trace(record, suffix):
    names = list(record['vars_names'])
    state = np.asarray(record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


# ---------------------------------------------------------------------------
# Structural behaviour of the heat_port flag (no solve needed)
# ---------------------------------------------------------------------------


def test_heat_port_false_has_no_wall_ports():
    pipe = StraightPipe(AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=3)
    assert pipe.heat_port is False
    for i in range(3):
        assert 'wall' not in pipe[f'pipe_segment_{i}'].ports
    with pytest.raises(AttributeError):
        _ = pipe.segment_wall_ports


def test_heat_port_true_exposes_wall_ports():
    pipe = StraightPipe(AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=3, heat_port=True)
    assert pipe.heat_port is True
    ports = pipe.segment_wall_ports
    assert len(ports) == 3
    for p in ports:
        assert isinstance(p, ThermalPort_TQ)
        assert p.require_connection is True


def test_deprecated_adiabatic_flag():
    # adiabatic=True == default (no heat port), no warning required.
    p_true = StraightPipe(AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=2, adiabatic=True)
    assert p_true.heat_port is False
    # adiabatic=False warns (legacy fixed-wall heating removed).
    with pytest.warns(DeprecationWarning):
        p_false = StraightPipe(AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=2, adiabatic=False)
    assert p_false.heat_port is False
    # Conflicting flags are rejected.
    with pytest.raises(ValueError):
        StraightPipe(AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=2, adiabatic=True, heat_port=True)


# ---------------------------------------------------------------------------
# Unconnected required port warning (instantiate, no solve)
# ---------------------------------------------------------------------------


def test_unconnected_wall_ports_warn():
    class Bare(Model):
        def declare_components(self):
            self.add_component('p', StraightPipe(
                AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=2, heat_port=True))

        def declare_equations(self):
            return []

    with pytest.warns(PortNotConnectedWarning):
        Bare().instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=2)


def test_conjugate_pipe_wall_ports_get_connected_no_warning():
    # A fully wired ConjugatePipe must NOT emit any PortNotConnectedWarning:
    # every segment wall port is connected to a CylindricalWall inside.
    class Sys(Model):
        def declare_components(self):
            self.add_component('inlet', AmbientInlet(AIR, p_ambient=101325, T_ambient=400.0, m_flow=0.02, D=D))
            self.add_component('pipe', ConjugatePipe(
                AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=3,
                wall_thickness=0.004, rho_wall=7850.0, cp_wall=490.0, k_wall=45.0,
                outer='adiabatic'))

        def declare_equations(self):
            self.connect(self['inlet'].ports['outlet'], self['pipe'].ports['inlet'])
            return []

    with warnings.catch_warnings():
        warnings.simplefilter("error", PortNotConnectedWarning)
        Sys().instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=5)


# ---------------------------------------------------------------------------
# Physics: corrected energy balance is conservative
# ---------------------------------------------------------------------------


def _build_conjugate_system(outer, T_hot=273.15 + 200.0, m_flow=0.03, n=4, wall_dynamic=True):
    class Sys(Model):
        def declare_components(self):
            self.add_component('inlet', AmbientInlet(AIR, p_ambient=101325, T_ambient=T_hot, m_flow=m_flow, D=D))
            self.add_component('pipe', ConjugatePipe(
                AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=n,
                wall_thickness=0.004, rho_wall=7850.0, cp_wall=490.0, k_wall=45.0,
                T_wall_init=293.15, outer=outer, h_ext=15.0, T_ext=293.15,
                wall_dynamic=wall_dynamic))

        def declare_equations(self):
            # AmbientInlet is square (imposes m_flow + isentropic state), so it
            # fixes the inlet pressure; leave the outlet open.
            self.connect(self['inlet'].ports['outlet'], self['pipe'].ports['inlet'])
            return []

    return Sys()


def test_conjugate_pipe_energy_conservation_and_cooling():
    n = 4
    sys = _build_conjugate_system(outer='convective', n=n)
    sys.instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=5)
    sys.initialise(n=1)
    for _ in range(80):
        sys.solve_dae_step(2.0)
        sys.next_step()

    rec = sys.record
    h_in = _trace(rec, '.pipe.pipe.h_in')[-1]
    h_out = _trace(rec, '.pipe.pipe.h_out')[-1]
    m_dot = _trace(rec, '.pipe.pipe.m_dot_in')[-1]

    q_total = sum(_trace(rec, f'.pipe_segment_{i}.q_inflow')[-1] for i in range(n))

    # Telescoping energy balance: total fluid enthalpy-flux change == sum of
    # per-segment wall heats.  This is the headline check for the q/m_dot and
    # per-segment-area corrections; the OLD (buggy) energy term, which added
    # the raw power [W] to a specific enthalpy [J/kg], fails it by orders of
    # magnitude.
    fluid_power = m_dot * (h_out - h_in)
    assert q_total != 0.0
    rel_err = abs(fluid_power - q_total) / abs(q_total)
    assert rel_err < 1e-2, f"energy balance off by {rel_err:.2%}"

    # Hot fluid gives up heat -> it cools (outlet enthalpy below inlet) and the
    # metal wall warms above its 293.15 K start.
    assert h_out < h_in
    assert fluid_power < 0.0
    for i in range(n):
        T_a = _trace(rec, f'.wall_{i}.T_a')[-1]
        assert T_a > 293.15 + 1.0, f"wall_{i} did not heat up"


def test_conjugate_pipe_quasi_static_walls():
    # wall_dynamic=False: the metal walls are massless (no thermal mass / no
    # ODEs), so heat conducts straight through to the convective outer boundary.
    # The telescoping fluid energy balance must still hold, and there must be no
    # wall der_ states.
    n = 4
    sys = _build_conjugate_system(outer='convective', n=n, wall_dynamic=False)
    sys.instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=5)
    sys.initialise(n=1)
    for _ in range(10):
        sys.solve_dae_step(2.0)
        sys.next_step()

    rec = sys.record
    # No wall capacities -> no der_ states attached to the walls.
    assert not any('.wall_' in n_ and 'der_' in n_ for n_ in rec['vars_names'])

    h_in = _trace(rec, '.pipe.pipe.h_in')[-1]
    h_out = _trace(rec, '.pipe.pipe.h_out')[-1]
    m_dot = _trace(rec, '.pipe.pipe.m_dot_in')[-1]
    q_total = sum(_trace(rec, f'.pipe_segment_{i}.q_inflow')[-1] for i in range(n))

    fluid_power = m_dot * (h_out - h_in)
    rel_err = abs(fluid_power - q_total) / abs(q_total)
    assert rel_err < 1e-2, f"energy balance off by {rel_err:.2%}"
    assert h_out < h_in  # hot gas still cools through the massless wall


def test_fixed_temperature_recovers_legacy_wall_heating():
    # The removed "convect to a fixed wall temperature" behaviour is recovered
    # by connecting a FixedTemperature to each segment wall port.  A wall HOTTER
    # than the inlet must heat the fluid.
    n = 3
    T_wall = 500.0
    T_in = 300.0

    class Sys(Model):
        def declare_components(self):
            self.add_component('inlet', AmbientInlet(AIR, p_ambient=101325, T_ambient=T_in, m_flow=0.02, D=D))
            self.add_component('pipe', StraightPipe(
                AIR, D=D, L=L, epsilon=EPS, z_in=0, z_out=0, n_segments=n, heat_port=True))
            for i in range(n):
                self.add_component(f'tw_{i}', FixedTemperature(T_wall))

        def declare_equations(self):
            self.connect(self['inlet'].ports['outlet'], self['pipe'].ports['inlet'])
            for i in range(n):
                self.connect(self['pipe'].segment_wall_ports[i], self[f'tw_{i}'].ports['heat'])
            return []

    sys = Sys()
    sys.instantiate(aditional_modules=AIR.modules, max_remove_trival_passes=5)
    sys.initialise(n=1)
    for _ in range(5):
        sys.solve_dae_step(1.0)
        sys.next_step()

    rec = sys.record
    h_in = _trace(rec, '.pipe.h_in')[-1]
    h_out = _trace(rec, '.pipe.h_out')[-1]
    m_dot = _trace(rec, '.pipe.m_dot_in')[-1]
    q_total = sum(_trace(rec, f'.pipe_segment_{i}.q_inflow')[-1] for i in range(n))

    # Wall hotter than fluid -> fluid heated -> outlet enthalpy above inlet.
    assert h_out > h_in
    # Energy balance still telescopes.
    fluid_power = m_dot * (h_out - h_in)
    rel_err = abs(fluid_power - q_total) / abs(q_total)
    assert rel_err < 1e-2, f"energy balance off by {rel_err:.2%}"
