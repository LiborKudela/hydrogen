"""Tests for the lumped-volume `PressureVessel` component.

A `_ForcedInflow` source feeds a constant mass flow into the vessel at constant
enthalpy. With both right-hand sides constant, Crank-Nicolson integrates the
mass and internal-energy ODEs exactly, so we can assert tight linear growth
and an exact algebraic closure `U = m*h - p*V` at every recorded step.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen.components import PressureVessel
from hydrogen.medium import CoolPropMedium
from hydrogen.model import Model, Parameter, Variable

M_DOT = 0.001       # kg/s   - imposed mass flow into the vessel
A_PORT = 1e-3       # m^2    - inlet area shared by source + vessel
V_VESSEL = 0.01     # m^3
P_INIT = 101325.0   # Pa     - vessel & source initial pressure (1 atm)
T_INIT = 293.15     # K      - vessel initial temperature
T_INLET = 343.15    # K      - source supplies warmer gas (50 K hotter)
DT = 0.5
N_STEPS = 8


class _ForcedInflow(Model):
    """Imposes a constant `(m_dot, h_set)` boundary at its `(p_out, h_out, m_dot_out)` port.

    `p_out` is set by the downstream connection (vessel back-pressure); the
    source's `m_dot_out` is pinned to the requested mass flow via a trivial
    equation that the reducer collapses at instantiate time.

    Sign convention: `m_dot` is the user-facing "physical outflow rate"
    (positive forward, leaving this boundary into the downstream).  Under
    "flow into me", the boundary's own `m_dot_out` measures fluid
    *entering* through its out-face, so the pin reads `m_dot_out = -m_dot`.
    """

    def __init__(self, medium, m_dot, h_set):
        self.medium = medium
        self.m_dot_value = m_dot
        self.h_set_value = h_set
        super().__init__()

    def declare_components(self):
        self.add_component('m_dot', Parameter(self.m_dot_value, "kg/s"))
        self.add_component('h_set', Parameter(self.h_set_value, "J/kg"))
        self.add_component('p_out', Variable(P_INIT, "Pa"))
        self.add_component('h_out', Variable(self.h_set_value, "J/kg"))
        self.add_component('m_dot_out', Variable(-self.m_dot_value, "kg/s"))

    def declare_equations(self):
        eq_m = self['m_dot'].symbol + self['m_dot_out'].symbol
        eq_h = self['h_out'].symbol - self['h_set'].symbol
        return [eq_m, eq_h]


class _VesselSystem(Model):
    """`_ForcedInflow -> PressureVessel`: minimal closed system for testing."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        # Inlet enthalpy at the inlet pressure: h(p, T_INLET). Held fixed by the source.
        self._h_inlet = float(self.medium.eval_h_pT(P_INIT, T_INLET))
        self.add_component('source', _ForcedInflow(self.medium, M_DOT, self._h_inlet))
        self.add_component('vessel', PressureVessel(self.medium, V_VESSEL, A_PORT, P_INIT, T_INIT))

    def declare_equations(self):
        # m_dot is a flow variable: both ends measure "into me" at the
        # shared interface so they sum to zero (sign=-1 connection).
        # p and h are across variables and stay direct equalities.
        return [
            self['source']['p_out'].symbol - self['vessel']['p_in'].symbol,
            self['source']['h_out'].symbol - self['vessel']['h_in'].symbol,
            self['source']['m_dot_out'].symbol + self['vessel']['m_dot_in'].symbol,
        ]


@pytest.fixture(scope="module")
def run():
    model = _VesselSystem()
    model.instantiate(max_remove_trival_passes=3, aditional_modules=model.medium.modules)
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

    return {"t": t, "trace": trace, "h_inlet": model._h_inlet}


def test_mass_grows_linearly(run):
    """CN integrates a constant `dm/dt = m_dot` exactly (to Newton tolerance)."""
    t = run["t"]
    m = run["trace"](".vessel.m")
    expected = m[0] + M_DOT * t
    assert np.allclose(m, expected, rtol=0, atol=1e-6)


def test_internal_energy_grows_linearly(run):
    """CN integrates a constant `dU/dt = m_dot * h_in` exactly (to Newton tolerance)."""
    t = run["t"]
    U = run["trace"](".vessel.U")
    expected = U[0] + M_DOT * run["h_inlet"] * t
    # U grows from ~3 kJ to ~5 kJ over the run, so a 0.1 J abs tolerance is ~3e-5 rel.
    assert np.allclose(U, expected, rtol=1e-5, atol=0.1)


def test_pressure_increases_monotonically(run):
    """Vessel pressure rises strictly as warmer gas accumulates."""
    p = run["trace"](".vessel.p")
    diffs = np.diff(p)
    assert np.all(diffs > 0), f"pressure not strictly increasing; diffs={diffs}"
    # Sanity floor: with ~M_DOT*N_STEPS*DT = 4 g of air added to a 10 L vessel at ~1 atm,
    # ideal-gas back-of-envelope predicts ~30+ kPa rise. Require at least 10 kPa to catch
    # any sign/units regression, but stay well below the analytical estimate.
    assert p[-1] - p[0] > 1e4, f"pressure barely rose: {p[-1] - p[0]:.0f} Pa"


def test_energy_state_closure_holds(run):
    """The algebraic closure `U = m*h - p*V` must hold at every recorded step."""
    m = run["trace"](".vessel.m")
    U = run["trace"](".vessel.U")
    p = run["trace"](".vessel.p")
    h = run["trace"](".vessel.h")
    residual = U - (m * h - p * V_VESSEL)
    # `U` magnitude ~ a few kJ, so a 1e-3 absolute tolerance is well below Newton tol*scale.
    assert np.max(np.abs(residual)) < 1e-3, (
        f"energy-state closure violated; max |U - (m*h - p*V)| = "
        f"{np.max(np.abs(residual)):.3e}"
    )


def test_density_closure_holds(run):
    """`m = rho(p, h) * V` must hold at every recorded step (Newton residual scale)."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    m = run["trace"](".vessel.m")
    p = run["trace"](".vessel.p")
    h = run["trace"](".vessel.h")
    rho = np.array([medium.eval_rho_ph(float(pi), float(hi)) for pi, hi in zip(p, h)])
    residual = m - rho * V_VESSEL
    assert np.max(np.abs(residual)) < 1e-6, (
        f"density closure violated; max |m - rho*V| = {np.max(np.abs(residual)):.3e}"
    )
