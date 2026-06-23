"""Tests for the two-node `CylindricalWall` radial-conduction component.

The circular counterpart of `FlatWall`, validated with the same closed-form
references but using cylindrical geometry:

  1. Steady radial conduction, two reservoirs through conductors:
         Q = (T_hot - T_cold) / (1/G_a + ln(r_out/r_in)/(2*pi*k*length) + 1/G_b)

  2. One face heated at a constant rate, the other insulated:
         d/dt (T_a + T_b)/2 = Q_in / (rho*cp*V)             (mean temp, CN-exact)
         steady (T_a - T_b)  -> Q_in / (2*G)                (radial gradient)
         stored energy        = Q_in * t                    (first law)

with V = pi*(r_out**2 - r_in**2)*length and
G = 2*pi*k*length / ln(r_out/r_in).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydrogen.components.thermofluid.walls import (
    CylindricalWall,
    FixedHeatFlow,
    FixedTemperature,
    ThermalConductor,
)
from hydrogen.model import Model

# Steel-ish tube: 50 mm bore, 60 mm outer, 1 m long.
RHO = 8000.0     # kg/m^3
CP = 500.0       # J/kg/K
K = 15.0         # W/m/K
R_IN = 0.05      # m
R_OUT = 0.06     # m
LENGTH = 1.0     # m

V_WALL = math.pi * (R_OUT ** 2 - R_IN ** 2) * LENGTH   # annular volume [m^3]
C_NODE = RHO * CP * V_WALL / 2.0                        # per-surface heat capacity [J/K]
C_TOTAL = RHO * CP * V_WALL                             # whole-wall heat capacity  [J/K]
G_COND = 2.0 * math.pi * K * LENGTH / math.log(R_OUT / R_IN)   # radial conductance [W/K]


def _trace_factory(model):
    record = model.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return t, trace


def test_constructor_rejects_bad_radii():
    """r_out must exceed r_in (both positive)."""
    with pytest.raises(ValueError):
        CylindricalWall(RHO, CP, K, r_in=0.06, r_out=0.05, length=LENGTH)


# ---------------------------------------------------------------------------
# 1. Steady radial conduction between two reservoirs (via conductors)
# ---------------------------------------------------------------------------

T_HOT = 400.0
T_COLD = 300.0
G_CONTACT = 5000.0   # W/K contact conductance on each surface


class _ConductionSystem(Model):
    """`FixedTemperature -> ThermalConductor -> CylindricalWall ->
    ThermalConductor -> FixedTemperature`."""

    def declare_components(self):
        self.add_component('hot', FixedTemperature(T_HOT))
        self.add_component('contact_a', ThermalConductor(G_CONTACT, T_init=T_HOT))
        self.add_component('wall', CylindricalWall(
            RHO, CP, K, R_IN, R_OUT, LENGTH, T_init=0.5 * (T_HOT + T_COLD)))
        self.add_component('contact_b', ThermalConductor(G_CONTACT, T_init=T_COLD))
        self.add_component('cold', FixedTemperature(T_COLD))

    def declare_equations(self):
        self.connect(self['hot'].ports['heat'], self['contact_a'].ports['heat_a'])
        self.connect(self['contact_a'].ports['heat_b'], self['wall'].ports['port_a'])
        self.connect(self['wall'].ports['port_b'], self['contact_b'].ports['heat_a'])
        self.connect(self['contact_b'].ports['heat_b'], self['cold'].ports['heat'])
        return []


@pytest.fixture(scope="module")
def conduction():
    model = _ConductionSystem()
    model.instantiate(max_remove_trival_passes=3)
    model.initialise()
    for _ in range(60):
        model.solve_dae_step(0.5)
        model.next_step()
    t, trace = _trace_factory(model)
    return {"t": t, "trace": trace}


def test_conduction_heat_flow_is_series_resistance(conduction):
    R_total = 1.0 / G_CONTACT + 1.0 / G_COND + 1.0 / G_CONTACT
    Q_expected = (T_HOT - T_COLD) / R_total
    Q_dot_a = conduction["trace"](".wall.Q_dot_a")[-1]
    assert Q_dot_a == pytest.approx(Q_expected, rel=1e-5)


def test_conduction_obeys_radial_fourier(conduction):
    """Across the wall the flux equals G*(T_a - T_b) with the cylindrical G."""
    T_a = conduction["trace"](".wall.T_a")[-1]
    T_b = conduction["trace"](".wall.T_b")[-1]
    Q_dot_a = conduction["trace"](".wall.Q_dot_a")[-1]
    assert Q_dot_a == pytest.approx(G_COND * (T_a - T_b), rel=1e-5)


def test_conduction_no_storage_at_steady_state(conduction):
    Q_dot_a = conduction["trace"](".wall.Q_dot_a")[-1]
    Q_dot_b = conduction["trace"](".wall.Q_dot_b")[-1]
    assert Q_dot_a + Q_dot_b == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. One surface heated at a constant rate, the other insulated
# ---------------------------------------------------------------------------

Q_IN = 500.0       # W injected at the inner surface
T_START = 300.0
DT = 1.0
N_STEPS = 120      # 120 s >> radial gradient time constant (~6.7 s)


class _HeatedInsulatedSystem(Model):
    """`FixedHeatFlow(Q_in) -> CylindricalWall -> FixedHeatFlow(0)`."""

    def declare_components(self):
        self.add_component('heater', FixedHeatFlow(Q_IN, T_init=T_START))
        self.add_component('wall', CylindricalWall(
            RHO, CP, K, R_IN, R_OUT, LENGTH, T_init=T_START))
        self.add_component('insulation', FixedHeatFlow(0.0, T_init=T_START))

    def declare_equations(self):
        self.connect(self['heater'].ports['heat'], self['wall'].ports['port_a'])
        self.connect(self['insulation'].ports['heat'], self['wall'].ports['port_b'])
        return []


@pytest.fixture(scope="module")
def heated():
    model = _HeatedInsulatedSystem()
    model.instantiate(max_remove_trival_passes=3)
    model.initialise()
    for _ in range(N_STEPS):
        model.solve_dae_step(DT)
        model.next_step()
    t, trace = _trace_factory(model)
    return {"t": t, "trace": trace}


def test_mean_temperature_rises_linearly(heated):
    t = heated["t"]
    T_a = heated["trace"](".wall.T_a")
    T_b = heated["trace"](".wall.T_b")
    T_mean = 0.5 * (T_a + T_b)
    slope = Q_IN / C_TOTAL
    expected = T_START + slope * t
    assert np.allclose(T_mean, expected, rtol=0, atol=1e-3)


def test_steady_state_temperature_difference(heated):
    T_a = heated["trace"](".wall.T_a")[-1]
    T_b = heated["trace"](".wall.T_b")[-1]
    assert (T_a - T_b) == pytest.approx(Q_IN / (2.0 * G_COND), rel=1e-3)


def test_energy_balance_closes(heated):
    t = heated["t"]
    T_a = heated["trace"](".wall.T_a")
    T_b = heated["trace"](".wall.T_b")
    stored = C_NODE * (T_a - T_a[0]) + C_NODE * (T_b - T_b[0])
    assert np.allclose(stored, Q_IN * t, rtol=0, atol=1e-2)
