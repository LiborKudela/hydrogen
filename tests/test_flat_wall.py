"""Tests for the two-node `FlatWall` heat-conduction component and the
thermal boundary conditions / passive elements that drive it
(`FixedTemperature`, `FixedHeatFlow`, `ConvectiveBoundary`,
`ThermalConductor`).

Three scenarios, each with a closed-form reference:

  1. Steady-state conduction, two temperature reservoirs through conductors:
         Q = (T_hot - T_cold) / (1/G_a + L/(k*A) + 1/G_b)   (series R)

  2. One face heated at a constant rate, the other insulated:
         d/dt (T_a + T_b)/2 = Q_in / (rho*cp*A*L)           (mean temp, CN-exact)
         steady (T_a - T_b)  -> Q_in * L / (2*k*A)          (gradient)
         stored energy        = Q_in * t                    (first law)

  3. One face heated, the far face convecting to ambient:
         T_b -> T_inf + Q_in / (h*A)                        (convective drop)
         T_a - T_b -> Q_in / (k*A/L)                        (conduction drop)
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen.components import (
    ConvectiveBoundary,
    FixedHeatFlow,
    FixedTemperature,
    FlatWall,
    ThermalConductor,
)
from hydrogen.model import Model

# Aluminium-ish slab: 10 cm^2 face, 2 cm thick.
RHO = 2700.0     # kg/m^3
CP = 900.0       # J/kg/K
K = 200.0        # W/m/K
A = 0.01         # m^2
L = 0.02         # m

C_NODE = RHO * CP * A * L / 2.0   # per-surface heat capacity [J/K]
C_TOTAL = RHO * CP * A * L        # whole-slab heat capacity  [J/K]
G_COND = K * A / L                # node-to-node conductance  [W/K]


def _trace_factory(model):
    record = model.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return t, trace


# ---------------------------------------------------------------------------
# 1. Steady-state conduction between two clamped temperatures (via conductors)
# ---------------------------------------------------------------------------

T_HOT = 350.0
T_COLD = 300.0
G_CONTACT = 500.0   # W/K contact conductance on each face


class _ConductionSystem(Model):
    """`FixedTemperature(hot) -> ThermalConductor -> FlatWall ->
    ThermalConductor -> FixedTemperature(cold)`.

    A prescribed temperature cannot be wired straight onto a capacitive
    wall surface (high-index constraint), so each reservoir drives the
    surface through a `ThermalConductor`.
    """

    def declare_components(self):
        self.add_component('hot', FixedTemperature(T_HOT))
        self.add_component('contact_a', ThermalConductor(G_CONTACT, T_init=T_HOT))
        self.add_component('wall', FlatWall(RHO, CP, K, A, L, T_init=0.5 * (T_HOT + T_COLD)))
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
    for _ in range(40):
        model.solve_dae_step(0.5)
        model.next_step()
    t, trace = _trace_factory(model)
    return {"t": t, "trace": trace}


def test_conduction_heat_flow_is_series_resistance(conduction):
    """Steady heat flow is set by the two contacts and the wall in series."""
    R_total = 1.0 / G_CONTACT + L / (K * A) + 1.0 / G_CONTACT
    Q_expected = (T_HOT - T_COLD) / R_total
    Q_dot_a = conduction["trace"](".wall.Q_dot_a")[-1]
    assert Q_dot_a == pytest.approx(Q_expected, rel=1e-5)


def test_conduction_obeys_fourier_across_wall(conduction):
    """Across the wall itself the flux equals k*A/L*(T_a - T_b)."""
    T_a = conduction["trace"](".wall.T_a")[-1]
    T_b = conduction["trace"](".wall.T_b")[-1]
    Q_dot_a = conduction["trace"](".wall.Q_dot_a")[-1]
    assert Q_dot_a == pytest.approx(G_COND * (T_a - T_b), rel=1e-5)


def test_conduction_no_storage_at_steady_state(conduction):
    """No accumulation at steady state: heat in at A leaves at B."""
    Q_dot_a = conduction["trace"](".wall.Q_dot_a")[-1]
    Q_dot_b = conduction["trace"](".wall.Q_dot_b")[-1]
    assert Q_dot_a + Q_dot_b == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. One face heated at a constant rate, the other insulated
# ---------------------------------------------------------------------------

Q_IN = 1000.0      # W injected at face A
T_START = 300.0
DT = 0.25
N_STEPS = 40       # 10 s >> conduction time constant (~1.2 s)


class _HeatedInsulatedSystem(Model):
    """`FixedHeatFlow(Q_in) -> FlatWall -> FixedHeatFlow(0)` (insulated far face)."""

    def declare_components(self):
        self.add_component('heater', FixedHeatFlow(Q_IN, T_init=T_START))
        self.add_component('wall', FlatWall(RHO, CP, K, A, L, T_init=T_START))
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
    """The conduction terms cancel in the node-sum balance, so the mean
    temperature has an exactly-constant rate that CN integrates exactly."""
    t = heated["t"]
    T_a = heated["trace"](".wall.T_a")
    T_b = heated["trace"](".wall.T_b")
    T_mean = 0.5 * (T_a + T_b)
    slope = Q_IN / C_TOTAL          # K/s
    expected = T_START + slope * t
    assert np.allclose(T_mean, expected, rtol=0, atol=1e-3)


def test_steady_state_temperature_difference(heated):
    """The surface-to-surface gradient relaxes toward Q_in/(2*G), i.e.
    Q_in * L / (2 * k * A): subtracting the two node balances gives
    d/dt(T_a - T_b) = (Q_in - 2*G*(T_a - T_b)) / C_node."""
    T_a = heated["trace"](".wall.T_a")[-1]
    T_b = heated["trace"](".wall.T_b")[-1]
    delta_expected = Q_IN / (2.0 * G_COND)
    assert (T_a - T_b) == pytest.approx(delta_expected, rel=1e-3)


def test_energy_balance_closes(heated):
    """Total stored energy change equals the net heat injected (Q_in * t)."""
    t = heated["t"]
    T_a = heated["trace"](".wall.T_a")
    T_b = heated["trace"](".wall.T_b")
    stored = C_NODE * (T_a - T_a[0]) + C_NODE * (T_b - T_b[0])
    assert np.allclose(stored, Q_IN * t, rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. One face heated, the far face convecting to ambient
# ---------------------------------------------------------------------------

Q_CONV = 50.0       # W injected at face A
H_CONV = 200.0      # W/m^2/K film coefficient
T_INF = 300.0


class _ConvectiveSystem(Model):
    """`FixedHeatFlow(Q_in) -> FlatWall -> ConvectiveBoundary(ambient)`."""

    def declare_components(self):
        self.add_component('heater', FixedHeatFlow(Q_CONV, T_init=T_INF))
        self.add_component('wall', FlatWall(RHO, CP, K, A, L, T_init=T_INF))
        self.add_component('film', ConvectiveBoundary(H_CONV, A, T_INF))

    def declare_equations(self):
        self.connect(self['heater'].ports['heat'], self['wall'].ports['port_a'])
        self.connect(self['film'].ports['heat'], self['wall'].ports['port_b'])
        return []


@pytest.fixture(scope="module")
def convective():
    model = _ConvectiveSystem()
    model.instantiate(max_remove_trival_passes=3)
    model.initialise()
    # Bulk time constant is R_conv * C_total = C_total/(h*A) ~ 240 s, so step
    # well past it (dt ~ tau keeps Crank-Nicolson's decay factor comfortably
    # positive while still reaching the steady fixed point).
    for _ in range(80):
        model.solve_dae_step(50.0)
        model.next_step()
    t, trace = _trace_factory(model)
    return {"t": t, "trace": trace}


def test_convective_face_temperature(convective):
    """All injected heat leaves by convection: T_b = T_inf + Q_in/(h*A)."""
    T_b = convective["trace"](".wall.T_b")[-1]
    assert T_b == pytest.approx(T_INF + Q_CONV / (H_CONV * A), rel=1e-4)


def test_convective_conduction_drop(convective):
    """Across the wall the conduction temperature drop is Q_in / (k*A/L)."""
    T_a = convective["trace"](".wall.T_a")[-1]
    T_b = convective["trace"](".wall.T_b")[-1]
    # The conduction-gradient mode is fast (tau ~ 1.2 s) relative to the
    # dt ~ 50 s steps used to march the slow convective bulk to steady
    # state, so Crank-Nicolson leaves a ~1e-4 residual on this mode.
    assert (T_a - T_b) == pytest.approx(Q_CONV / G_COND, rel=1e-3)


def test_convective_through_flow_matches_injection(convective):
    """At steady state the heat carried through equals the injected rate."""
    Q_dot_a = convective["trace"](".wall.Q_dot_a")[-1]
    assert Q_dot_a == pytest.approx(Q_CONV, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. Quasi-static (dynamic=False): massless wall, pure conduction
# ---------------------------------------------------------------------------

T_QS_HOT = 500.0
H_QS = 25.0
T_QS_INF = 300.0


class _QuasiStaticSystem(Model):
    """`FixedTemperature(hot) -> FlatWall(dynamic=False) -> ConvectiveBoundary`.

    With no node capacity a prescribed temperature may be wired straight onto
    the face (it is not a high-index constraint), so no `ThermalConductor` is
    needed -- unlike the capacitive case.
    """

    def declare_components(self):
        self.add_component('hot', FixedTemperature(T_QS_HOT))
        self.add_component('wall', FlatWall(RHO, CP, K, A, L, T_init=T_QS_HOT, dynamic=False))
        self.add_component('air', ConvectiveBoundary(H_QS, A, T_QS_INF))

    def declare_equations(self):
        self.connect(self['hot'].ports['heat'], self['wall'].ports['port_a'])
        self.connect(self['wall'].ports['port_b'], self['air'].ports['heat'])
        return []


@pytest.fixture(scope="module")
def quasi_static():
    model = _QuasiStaticSystem()
    model.instantiate(max_remove_trival_passes=3)
    model.initialise()
    # Quasi-static: no states, so the solution is the steady answer in one
    # step; take a couple of steps anyway to exercise the time loop.
    for _ in range(3):
        model.solve_dae_step(1.0)
        model.next_step()
    t, trace = _trace_factory(model)
    return {"t": t, "trace": trace, "model": model}


def test_quasi_static_has_no_differential_states(quasi_static):
    """dynamic=False removes the node ODEs and capacities -> no der_ vars."""
    names = list(quasi_static["model"].record['vars_names'])
    assert not any('der_' in n for n in names), "quasi-static wall must have no der_ variables"


def test_quasi_static_series_resistance(quasi_static):
    """Massless wall = series of conduction (k*A/L) and film (h*A):
    T_b = (G*T_hot + h*A*T_inf) / (G + h*A), heat through = G*(T_hot - T_b)."""
    T_a = quasi_static["trace"](".wall.T_a")[-1]
    T_b = quasi_static["trace"](".wall.T_b")[-1]
    Q_a = quasi_static["trace"](".wall.Q_dot_a")[-1]

    hA = H_QS * A
    T_b_exact = (G_COND * T_QS_HOT + hA * T_QS_INF) / (G_COND + hA)
    Q_exact = G_COND * (T_QS_HOT - T_b_exact)

    assert T_a == pytest.approx(T_QS_HOT, abs=1e-9)
    assert T_b == pytest.approx(T_b_exact, rel=1e-6)
    assert Q_a == pytest.approx(Q_exact, rel=1e-6)


def test_quasi_static_no_storage(quasi_static):
    """A massless wall transmits all heat: Q_dot_a + Q_dot_b == 0 always."""
    Q_a = quasi_static["trace"](".wall.Q_dot_a")[-1]
    Q_b = quasi_static["trace"](".wall.Q_dot_b")[-1]
    assert Q_a + Q_b == pytest.approx(0.0, abs=1e-6)
