"""Tests for the inline `thermofluid.sensors` flow instruments.

A `MassSource` imposes a constant mass flow into a series of sensors terminated
by a fixed-pressure `PressureOutlet`.  Because every sensor is a lossless
pass-through (no pressure drop, no enthalpy change), the whole line sits at the
sink pressure and the reservoir enthalpy, so each reading has a closed-form
expected value we can check against the medium's property functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen.components.thermofluid.flow import MassSource, PressureOutlet
from hydrogen.components.thermofluid.sensors import (
    MassFlowSensor,
    MassTotalizer,
    PressureSensor,
    TemperatureSensor,
    VolumeFlowSensor,
)
from hydrogen.medium import CoolPropMedium
from hydrogen.model import Model

M_FLOW = 0.05        # kg/s   - imposed forward mass flow
P_SINK = 2.0e5       # Pa     - fixed outlet pressure
T_SRC = 320.0        # K      - reservoir temperature
D_SENS = 0.008       # m      - temperature-sensor bore
DT = 0.5
N_STEPS = 6


class _SensorRig(Model):
    """`MassSource -> [4 sensors in series] -> PressureOutlet`."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('src', MassSource(self.medium, m_flow=M_FLOW,
                                             T_source=T_SRC, p_init=P_SINK))
        self.add_component('mflow', MassFlowSensor(self.medium))
        self.add_component('total', MassTotalizer(self.medium))
        self.add_component('temp', TemperatureSensor(self.medium, D=D_SENS))
        self.add_component('temp0', TemperatureSensor(self.medium, D=D_SENS,
                                                      total=True))
        self.add_component('pres', PressureSensor(self.medium, D=D_SENS))
        self.add_component('pres0', PressureSensor(self.medium, D=D_SENS,
                                                   total=True))
        self.add_component('vflow', VolumeFlowSensor(self.medium))
        self.add_component('sink', PressureOutlet(self.medium,
                                                  p_ambient=P_SINK,
                                                  T_ambient=T_SRC))

    def declare_equations(self):
        # Source + every sensor expose 'outlet'; every sensor + the sink expose
        # 'inlet', so the chain wires outlet -> inlet down the line.
        chain = ['src', 'mflow', 'total', 'temp', 'temp0', 'pres', 'pres0',
                 'vflow', 'sink']
        for up, dn in zip(chain[:-1], chain[1:]):
            self.connect(self[up].ports['outlet'], self[dn].ports['inlet'])
        return []


@pytest.fixture(scope="module")
def run():
    model = _SensorRig()
    model.instantiate(max_remove_trival_passes=3,
                      aditional_modules=model.medium.modules)
    model.initialise()
    for _ in range(N_STEPS):
        model.solve_dae_step(DT)
        model.next_step()

    record = model.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return {"t": t, "trace": trace, "medium": model.medium}


def test_mass_flow_reads_axial_flow(run):
    y = run["trace"](".mflow.y")
    assert np.allclose(y, M_FLOW, rtol=1e-6, atol=1e-9)


def test_totalizer_integrates_mass(run):
    """der(y) = m_dot with a constant flow -> y = y0 + m_flow * t (CN exact)."""
    t = run["t"]
    y = run["trace"](".total.y")
    assert np.allclose(y, y[0] + M_FLOW * (t - t[0]), rtol=0, atol=1e-8)
    assert y[-1] > y[0]


def test_static_temperature_matches_reservoir(run):
    """Zero pressure drop + reservoir enthalpy => static T equals T_source."""
    medium = run["medium"]
    h = float(medium.eval_h_pT(P_SINK, T_SRC))
    expected = float(medium.eval_T_ph(P_SINK, h))
    y = run["trace"](".temp.y")
    assert np.allclose(y, expected, rtol=0, atol=1e-4)
    assert np.allclose(y, T_SRC, rtol=0, atol=1e-3)


def test_total_temperature_adds_kinetic_energy(run):
    """Total temperature = T(p, h + w^2/2) and exceeds the static reading."""
    medium = run["medium"]
    h = float(medium.eval_h_pT(P_SINK, T_SRC))
    rho = float(medium.eval_rho_ph(P_SINK, h))
    area = np.pi * D_SENS ** 2 / 4.0
    w = M_FLOW / (rho * area)
    expected = float(medium.eval_T_ph(P_SINK, h + w ** 2 / 2.0))
    y0 = run["trace"](".temp0.y")
    y_static = run["trace"](".temp.y")
    assert np.allclose(y0, expected, rtol=0, atol=1e-3)
    assert np.all(y0 >= y_static - 1e-9)


def test_static_pressure_matches_sink(run):
    """Zero pressure drop => static pressure equals the sink pressure."""
    y = run["trace"](".pres.y")
    assert np.allclose(y, P_SINK, rtol=0, atol=1e-3)


def test_total_pressure_adds_dynamic_head(run):
    """Total pressure = p + rho*w^2/2 and exceeds the static reading."""
    medium = run["medium"]
    h = float(medium.eval_h_pT(P_SINK, T_SRC))
    rho = float(medium.eval_rho_ph(P_SINK, h))
    area = np.pi * D_SENS ** 2 / 4.0
    w = M_FLOW / (rho * area)
    expected = P_SINK + rho * w ** 2 / 2.0
    y0 = run["trace"](".pres0.y")
    y_static = run["trace"](".pres.y")
    assert np.allclose(y0, expected, rtol=1e-6, atol=1e-3)
    assert np.all(y0 >= y_static - 1e-9)


def test_volume_flow_is_mass_over_density(run):
    medium = run["medium"]
    h = float(medium.eval_h_pT(P_SINK, T_SRC))
    rho = float(medium.eval_rho_ph(P_SINK, h))
    expected = M_FLOW / rho
    y = run["trace"](".vflow.y")
    assert np.allclose(y, expected, rtol=1e-5, atol=1e-9)
