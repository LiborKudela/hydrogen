"""Tests for the `MixingJunction` component (both modes) and dynamic flow reversal.

The file is split in two halves:

  * DYNAMIC test (4-port junction with `dynamic=True`):
      Driven by four `_PrescribedStreamBoundary`s.  Each boundary pins its
      `m_dot_out` and `h_set_out` to a Parameter so the test can flip flow
      directions deterministically between two phases.  Mass conservation is
      satisfied trivially because the prescribed m_dots sum to zero.  The
      junction's `m, U` storage states absorb any transient imbalance.

  * QUASI-STATIC test (4-port junction with `dynamic=False`):
      The algebraic `sum_k m_dot_k = 0` would over-determine the system if
      every port were `m_dot`-pinned (no degree of freedom for the balance to
      close).  We instead use one `_PressurePinBoundary` (which pins `p` and
      `h_set`, leaving `m_dot` free) plus three `_PrescribedStreamBoundary`s.
      The pressure-pinned port's `m_dot` floats to whatever value the
      junction's mass balance requires; flipping the three prescribed m_dots
      between phases is what reverses flow on the free port.

The carrier port-enthalpy `h_k` is checked in both flavours: it should track
the upstream's `h_set` while sourcing and the junction's well-mixed `h` while
sinking, blended smoothly across the zero-crossing.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen.components.thermofluid.flow import MixingJunction
from hydrogen.medium import CoolPropMedium
from hydrogen.model import Model, Parameter, Variable

# --- Shared test parameters ---------------------------------------------------
M_DOT_BASE      = 1e-3      # kg/s
P_INIT          = 101325.0  # Pa
T_JUNCTION_INIT = 293.15    # K
T_WARM          = 343.15    # K  (50 K hotter than junction at t = 0)
T_COOL          = 273.15    # K  (20 K cooler than junction at t = 0)
V_JUNCTION      = 1e-3      # m^3 (dynamic mode only)
DT              = 0.05      # s
N_STEPS_PHASE   = 8         # fixed-dt steps per phase
MDOT_EPS        = 1e-6      # kg/s, smooth-switch width (shared with junction)

# --- Shared boundary components -----------------------------------------------

class _PrescribedStreamBoundary(Model):
    """Prescribes `m_dot_out` and the stream-in enthalpy `h_set_out`; both via
    trivial Parameter pins that the reducer collapses at instantiate time.

    The junction's per-port smooth-blend closure consumes `h_set_out` (wired
    to `junction.h_set_k`) and dictates the carrier `h_k`.

    Sign convention: `m_dot_target` is the user-facing "physical outflow
    rate" (positive forward, leaving this boundary into the junction).
    Under "flow into me", the boundary's own `m_dot_out` measures fluid
    *entering* through its out-face, so the pin reads
    `m_dot_out = -m_dot_target`.
    """

    def __init__(self, medium: CoolPropMedium, m_dot_init: float, h_set: float):
        self.medium = medium
        self._m_dot_init = m_dot_init
        self._h_set = h_set
        super().__init__()

    def declare_components(self):
        self.add_component('m_dot_target', Parameter(self._m_dot_init, "kg/s"))
        self.add_component('h_set',        Parameter(self._h_set, "J/kg"))
        self.add_component('p_out',        Variable(P_INIT, "Pa"))
        self.add_component('h_set_out',    Variable(self._h_set, "J/kg"))
        self.add_component('m_dot_out',    Variable(-self._m_dot_init, "kg/s"))

    def declare_equations(self):
        return [
            self['m_dot_target'].symbol + self['m_dot_out'].symbol,
            self['h_set'].symbol        - self['h_set_out'].symbol,
        ]


class _PressurePinBoundary(Model):
    """Pins port pressure and stream-in enthalpy; `m_dot_out` is free.

    Wired into the quasi-static junction it provides exactly the degree of
    freedom the algebraic mass balance needs to close: the junction's
    `sum_k m_dot_k = 0` constraint determines this port's `m_dot_out` from
    the other ports' prescribed flows.
    """

    def __init__(self, medium: CoolPropMedium, p_set: float, h_set: float):
        self.medium = medium
        self._p_set = p_set
        self._h_set = h_set
        super().__init__()

    def declare_components(self):
        self.add_component('p_target',  Parameter(self._p_set, "Pa"))
        self.add_component('h_set',     Parameter(self._h_set, "J/kg"))
        self.add_component('p_out',     Variable(self._p_set, "Pa"))
        self.add_component('h_set_out', Variable(self._h_set, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))

    def declare_equations(self):
        return [
            self['p_target'].symbol - self['p_out'].symbol,
            self['h_set'].symbol    - self['h_set_out'].symbol,
        ]


def _wire_port(parent: Model, bnd_path: str, junc_path: str, k: int) -> None:
    """Connect a boundary's `_out` triple (+ stream-in) to junction port k.

    Under "flow into me", both `bnd.m_dot_out` and `junction.m_dot_k`
    measure fluid entering their respective component at the shared
    interface -- equal in magnitude with opposite sign, so the flow
    channel uses a `sign=-1` (sum-to-zero) connection.  Pressure and
    stream-in enthalpy are across variables (single-valued at the
    interface) and stay direct equalities.
    """
    parent.add_connection(parent[bnd_path]['p_out'],     parent[junc_path][f'p_{k}'])
    parent.add_connection(parent[bnd_path]['h_set_out'], parent[junc_path][f'h_set_{k}'])
    parent.add_connection(parent[bnd_path]['m_dot_out'], parent[junc_path][f'm_dot_{k}'],
                          sign=-1)


# =============================================================================
# DYNAMIC MIXING JUNCTION
# =============================================================================

class FourPortDynamicSystem(Model):
    """4 prescribed-flow boundaries wired to a dynamic 4-port `MixingJunction`."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self._h_warm = float(self.medium.eval_h_pT(P_INIT, T_WARM))
        self._h_cool = float(self.medium.eval_h_pT(P_INIT, T_COOL))
        self._h_init = float(self.medium.eval_h_pT(P_INIT, T_JUNCTION_INIT))

        # Phase-1 initial m_dot's (sum to 0).
        self.add_component('bnd_0', _PrescribedStreamBoundary(self.medium, +M_DOT_BASE,     self._h_warm))
        self.add_component('bnd_1', _PrescribedStreamBoundary(self.medium, +M_DOT_BASE,     self._h_cool))
        self.add_component('bnd_2', _PrescribedStreamBoundary(self.medium, +M_DOT_BASE,     self._h_cool))
        self.add_component('bnd_3', _PrescribedStreamBoundary(self.medium, -3 * M_DOT_BASE, self._h_cool))

        self.add_component('junction',
            MixingJunction(self.medium, N=4, V=V_JUNCTION,
                           p_init=P_INIT, T_init=T_JUNCTION_INIT,
                           m_dot_eps=MDOT_EPS, dynamic=True))

    def declare_equations(self):
        for k in range(4):
            _wire_port(self, f'bnd_{k}', 'junction', k)
        return []


@pytest.fixture(scope="module")
def run_dynamic():
    system = FourPortDynamicSystem()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    system.initialise(relaxation=0.5, max_iter=200)

    # Phase 1: warm source on port 0
    for _ in range(N_STEPS_PHASE):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()

    # Reverse port 0; rebalance port 3 so the prescribed flows still sum to 0.
    system['bnd_0']['m_dot_target'].set_value(-M_DOT_BASE)
    system['bnd_3']['m_dot_target'].set_value(-M_DOT_BASE)

    # Phase 2: port 0 now receives the (warmer) mixed outflow
    for _ in range(N_STEPS_PHASE):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()

    record = system.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return {
        "t":        t,
        "trace":    trace,
        "h_warm":   system._h_warm,
        "h_cool":   system._h_cool,
        "h_init":   system._h_init,
        "n_phase1": N_STEPS_PHASE,
    }


def test_dynamic_port_0_flow_reverses_between_phases(run_dynamic):
    """`m_dot_0` flips from + to - across the phase boundary."""
    m_dot_0 = run_dynamic["trace"](".junction.m_dot_0")
    n1 = run_dynamic["n_phase1"]
    assert np.all(m_dot_0[:n1 + 1] > 0), (
        f"port 0 should source during phase 1; got m_dot_0[:{n1+1}]={m_dot_0[:n1+1]}"
    )
    assert np.all(m_dot_0[n1 + 1:] < 0), (
        f"port 0 should sink during phase 2; got m_dot_0[{n1+1}:]={m_dot_0[n1+1:]}"
    )


def test_dynamic_other_ports_keep_their_directions(run_dynamic):
    """Ports 1, 2 always source; port 3 always sinks (their `m_dot` targets
    never flip during the run)."""
    assert np.all(run_dynamic["trace"](".junction.m_dot_1") > 0)
    assert np.all(run_dynamic["trace"](".junction.m_dot_2") > 0)
    assert np.all(run_dynamic["trace"](".junction.m_dot_3") < 0)


def test_dynamic_mass_balance_within_each_phase(run_dynamic):
    """Prescribed flows sum to zero in both phases -> stored mass stays
    constant within each phase (to Newton tolerance)."""
    n1 = run_dynamic["n_phase1"]
    m = run_dynamic["trace"](".junction.m")
    drift_phase1 = abs(m[n1] - m[0])
    drift_phase2 = abs(m[-1] - m[n1 + 1])
    bound = M_DOT_BASE * DT * 0.01
    assert drift_phase1 < bound, f"phase-1 mass drifted by {drift_phase1:.3e} kg"
    assert drift_phase2 < bound, f"phase-2 mass drifted by {drift_phase2:.3e} kg"


def test_dynamic_energy_state_closure_holds(run_dynamic):
    """`U = m*h - p*V` must hold at every recorded step."""
    m = run_dynamic["trace"](".junction.m")
    U = run_dynamic["trace"](".junction.U")
    p = run_dynamic["trace"](".junction.p")
    h = run_dynamic["trace"](".junction.h")
    residual = U - (m * h - p * V_JUNCTION)
    assert np.max(np.abs(residual)) < 1e-3, (
        f"energy-state closure violated; max |U - (m*h - p*V)| = "
        f"{np.max(np.abs(residual)):.3e}"
    )


def test_dynamic_density_closure_holds(run_dynamic):
    """`m = rho(p, h) * V` must hold at every recorded step."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    m = run_dynamic["trace"](".junction.m")
    p = run_dynamic["trace"](".junction.p")
    h = run_dynamic["trace"](".junction.h")
    rho = np.array([medium.eval_rho_ph(float(pi), float(hi)) for pi, hi in zip(p, h)])
    residual = m - rho * V_JUNCTION
    assert np.max(np.abs(residual)) < 1e-6, (
        f"density closure violated; max |m - rho*V| = {np.max(np.abs(residual)):.3e}"
    )


def test_dynamic_junction_h_responds_to_phase_change(run_dynamic):
    """Junction `h` rises in phase 1 (warm inflow) and falls in phase 2
    (warm source replaced by sink)."""
    n1 = run_dynamic["n_phase1"]
    h = run_dynamic["trace"](".junction.h")
    assert h[n1] > h[0] + 1.0, (
        f"junction h should rise during phase 1; got h[0]={h[0]:.1f}, h[end_phase1]={h[n1]:.1f}"
    )
    assert h[-1] < h[n1] - 1.0, (
        f"junction h should fall during phase 2; got h[end_phase1]={h[n1]:.1f}, h[end]={h[-1]:.1f}"
    )


def test_dynamic_port_0_h_tracks_source_then_mixed(run_dynamic):
    """Port 0's carrier `h_0` follows the warm source while sourcing and the
    mixed `h` after reversing."""
    n1 = run_dynamic["n_phase1"]
    h_0 = run_dynamic["trace"](".junction.h_0")
    h_junction = run_dynamic["trace"](".junction.h")
    h_warm = run_dynamic["h_warm"]
    err1 = abs(h_0[n1] - h_warm) / abs(h_warm)
    assert err1 < 1e-3, (
        f"port 0 should carry warm source h during phase 1; "
        f"got h_0={h_0[n1]:.1f} vs h_warm={h_warm:.1f} (rel. err {err1:.3e})"
    )
    err2 = abs(h_0[-1] - h_junction[-1]) / abs(h_junction[-1])
    assert err2 < 1e-3, (
        f"port 0 should carry junction's mixed h during phase 2; "
        f"got h_0={h_0[-1]:.1f} vs h_mixed={h_junction[-1]:.1f} (rel. err {err2:.3e})"
    )


# =============================================================================
# QUASI-STATIC MIXING JUNCTION
# =============================================================================

class FourPortQuasiStaticSystem(Model):
    """1 pressure-pin + 3 flow-pin boundaries -> quasi-static 4-port `MixingJunction`.

    The pressure-pin's `m_dot_out` is the degree of freedom that closes the
    junction's algebraic mass balance.  Flipping the three flow pins between
    phases reverses flow on the pressure-pinned port (port 0).
    """

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self._h_warm = float(self.medium.eval_h_pT(P_INIT, T_WARM))
        self._h_cool = float(self.medium.eval_h_pT(P_INIT, T_COOL))
        self._h_init = float(self.medium.eval_h_pT(P_INIT, T_JUNCTION_INIT))

        # Port 0: pressure pin with WARM stream-in.  Its m_dot floats.
        self.add_component('bnd_0',
            _PressurePinBoundary(self.medium, p_set=P_INIT, h_set=self._h_warm))

        # Ports 1, 2, 3: flow pins.  Phase-1 initial state has them all
        # sinking (m_dot < 0) so that mass conservation forces port 0 to
        # source (m_dot_0 > 0).
        self.add_component('bnd_1', _PrescribedStreamBoundary(self.medium, -M_DOT_BASE, self._h_cool))
        self.add_component('bnd_2', _PrescribedStreamBoundary(self.medium, -M_DOT_BASE, self._h_cool))
        self.add_component('bnd_3', _PrescribedStreamBoundary(self.medium, -M_DOT_BASE, self._h_cool))

        self.add_component('junction',
            MixingJunction(self.medium, N=4,
                           p_init=P_INIT, T_init=T_JUNCTION_INIT,
                           m_dot_eps=MDOT_EPS, dynamic=False))

    def declare_equations(self):
        for k in range(4):
            _wire_port(self, f'bnd_{k}', 'junction', k)
        return []


@pytest.fixture(scope="module")
def run_quasi():
    system = FourPortQuasiStaticSystem()
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    system.initialise(relaxation=0.5, max_iter=200)

    # Phase 1: ports 1, 2, 3 sink at -M; mass balance -> port 0 sources at +3M.
    for _ in range(N_STEPS_PHASE):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()

    # Flip all three flow pins to source -> mass balance reverses port 0.
    system['bnd_1']['m_dot_target'].set_value(+M_DOT_BASE)
    system['bnd_2']['m_dot_target'].set_value(+M_DOT_BASE)
    system['bnd_3']['m_dot_target'].set_value(+M_DOT_BASE)

    # Phase 2: ports 1, 2, 3 source cool fluid at +M; mass balance -> port 0
    # sinks at -3M, now receiving the well-mixed (cool) outflow.
    for _ in range(N_STEPS_PHASE):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()

    record = system.record
    t = np.asarray(record['time'])
    state = np.asarray(record['state'])
    names = list(record['vars_names'])

    def trace(suffix):
        idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
        return state[:, idx]

    return {
        "t":        t,
        "trace":    trace,
        "h_warm":   system._h_warm,
        "h_cool":   system._h_cool,
        "h_init":   system._h_init,
        "n_phase1": N_STEPS_PHASE,
    }


def test_quasi_port_0_flow_reverses_between_phases(run_quasi):
    """Mass balance forces the pressure-pinned port to reverse when the three
    flow pins flip sign."""
    m_dot_0 = run_quasi["trace"](".junction.m_dot_0")
    n1 = run_quasi["n_phase1"]
    assert np.all(m_dot_0[:n1 + 1] > 0), (
        f"port 0 should source (compensate sinks) during phase 1; "
        f"got m_dot_0[:{n1+1}]={m_dot_0[:n1+1]}"
    )
    assert np.all(m_dot_0[n1 + 1:] < 0), (
        f"port 0 should sink (compensate sources) during phase 2; "
        f"got m_dot_0[{n1+1}:]={m_dot_0[n1+1:]}"
    )


def test_quasi_mass_balance_is_algebraic_zero(run_quasi):
    """In quasi-static mode `sum_k m_dot_k = 0` holds exactly at every step,
    not just on average -- there is no storage to absorb a transient
    imbalance."""
    m_dot_0 = run_quasi["trace"](".junction.m_dot_0")
    m_dot_1 = run_quasi["trace"](".junction.m_dot_1")
    m_dot_2 = run_quasi["trace"](".junction.m_dot_2")
    m_dot_3 = run_quasi["trace"](".junction.m_dot_3")
    total = m_dot_0 + m_dot_1 + m_dot_2 + m_dot_3
    # Newton tolerance (1e-6) is on the scaled residual; the actual m_dot
    # residual scales with the equation scale.  1e-6 kg/s leaves ample margin.
    assert np.max(np.abs(total)) < 1e-6, (
        f"algebraic mass balance violated; max |sum m_dot| = "
        f"{np.max(np.abs(total)):.3e} kg/s"
    )


def test_quasi_junction_h_matches_inflow_weighted_average(run_quasi):
    """Quasi-static energy balance + h-anchor regularization -> the junction's
    mixed `h` is the inflow-mass-flow-weighted average of the inflow ports'
    `h_set` values.

    Phase 1: only port 0 inflows (h_set_0 = h_warm)  ->  h_junction ~ h_warm.
    Phase 2: ports 1, 2, 3 inflow (h_set = h_cool)   ->  h_junction ~ h_cool.
    """
    n1 = run_quasi["n_phase1"]
    h = run_quasi["trace"](".junction.h")
    h_warm = run_quasi["h_warm"]
    h_cool = run_quasi["h_cool"]
    # End of phase 1: junction's h must match the single warm inflow.
    err1 = abs(h[n1] - h_warm) / abs(h_warm)
    assert err1 < 1e-3, (
        f"phase-1 junction h should equal h_warm; "
        f"got h_junction={h[n1]:.1f} vs h_warm={h_warm:.1f} (rel. err {err1:.3e})"
    )
    # End of phase 2: three equal cool inflows -> mixed h equals h_cool.
    err2 = abs(h[-1] - h_cool) / abs(h_cool)
    assert err2 < 1e-3, (
        f"phase-2 junction h should equal h_cool; "
        f"got h_junction={h[-1]:.1f} vs h_cool={h_cool:.1f} (rel. err {err2:.3e})"
    )


def test_quasi_port_0_h_tracks_warm_then_mixed(run_quasi):
    """Port-0 carrier `h_0` follows the smooth-blend rule:
        phase 1 (alpha~1): h_0 = h_set_0 = h_warm
        phase 2 (alpha~0): h_0 = h        = h_cool
    """
    n1 = run_quasi["n_phase1"]
    h_0 = run_quasi["trace"](".junction.h_0")
    h_junc = run_quasi["trace"](".junction.h")
    h_warm = run_quasi["h_warm"]
    err1 = abs(h_0[n1] - h_warm) / abs(h_warm)
    assert err1 < 1e-3, (
        f"phase-1 port 0 h should equal h_warm; got h_0={h_0[n1]:.1f} (rel. err {err1:.3e})"
    )
    err2 = abs(h_0[-1] - h_junc[-1]) / abs(h_junc[-1])
    assert err2 < 1e-3, (
        f"phase-2 port 0 h should equal junction's mixed h; "
        f"got h_0={h_0[-1]:.1f} vs h_mixed={h_junc[-1]:.1f} (rel. err {err2:.3e})"
    )


def test_quasi_no_storage_states_present():
    """Sanity: a quasi-static junction must not introduce `m`, `U`, or `V`
    components -- otherwise it would still be carrying the dynamic API."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    junc = MixingJunction(medium, N=3, p_init=P_INIT, T_init=T_JUNCTION_INIT,
                          dynamic=False)
    assert 'm' not in junc.components, "quasi-static junction should not own `m`"
    assert 'U' not in junc.components, "quasi-static junction should not own `U`"
    assert 'V' not in junc.components, "quasi-static junction should not own `V`"


def test_dynamic_requires_volume():
    """`dynamic=True` without a `V` is a configuration error, not a silent
    crash later in CoolProp."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    with pytest.raises(ValueError, match="V"):
        MixingJunction(medium, N=2, dynamic=True)  # missing V
