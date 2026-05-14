"""Tests for the `LoopBuffer` subclass of `MixingJunction`.

Covers two scenarios:

  * Forward flow:  the legacy `dU/dt = m_dot_in*h_in - m_dot_out*h` and
    `dm/dt = m_dot_in - m_dot_out` behavior must be reproduced exactly,
    so that existing examples like `examples/loop_pump_pipe.py` keep
    behaving the same.

  * Reverse flow:  with `m_dot_in < 0` and `m_dot_out < 0` (fluid moving
    through the buffer backwards), the smooth donor-cell upwind on the
    inlet kicks in.  The inlet's `h_in` contribution must fade out (no
    inflow), the outlet must take over by carrying the buffer's own
    bulk `h` back through itself (`h_set_1` is pinned to bulk `h`, so
    this contribution cancels), and Newton must converge cleanly across
    the zero-crossing.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen.components import LoopBuffer
from hydrogen.medium import CoolPropMedium
from hydrogen.model import Model, Parameter, Variable

# --- Shared parameters --------------------------------------------------------
P_INIT       = 101325.0
T_INIT       = 293.15
T_WARM       = 343.15
V_BUFFER     = 1e-3        # m^3
M_DOT_BASE   = 1e-3        # kg/s
DT           = 0.05        # s
N_STEPS      = 6


def _push_params(model: Model) -> None:
    model.set_param_values(
        np.array([p.value for p in model.raw_param_references])
    )


class _InletPin(Model):
    """Inlet that pins both `m_dot_out` and the enthalpy it pushes into
    the buffer (`h_out`, unioned to `buffer.h_in`).

    Sign convention: `m_dot_target` is the user-facing "physical outflow
    rate" (positive forward, i.e. flow leaving this pin into the
    downstream buffer).  Under "flow into me", the boundary's own
    `m_dot_out` measures fluid *entering* through its out-face, so we
    pin `m_dot_out = -m_dot_target`.
    """

    def __init__(self, medium: CoolPropMedium, m_dot: float, h_set: float):
        self.medium = medium
        self._m_dot = m_dot
        self._h_set = h_set
        super().__init__()

    def declare_components(self):
        self.add_component('m_dot_target', Parameter(self._m_dot, "kg/s"))
        self.add_component('h_target',     Parameter(self._h_set, "J/kg"))
        self.add_component('p_out',        Variable(P_INIT, "Pa"))
        self.add_component('h_out',        Variable(self._h_set, "J/kg"))
        self.add_component('m_dot_out',    Variable(-self._m_dot, "kg/s"))

    def declare_equations(self):
        return [
            self['m_dot_target'].symbol + self['m_dot_out'].symbol,
            self['h_target'].symbol     - self['h_out'].symbol,
        ]


class _OutletPin(Model):
    """Outlet that pins only `m_dot_in`.  Its `p_in` is unioned with the
    buffer's `p_out` (== buffer.p) but the buffer's algebraic states fix
    everything else; we deliberately do NOT pin enthalpy here because the
    buffer's `h_out` collapses to the buffer's bulk `h` internally."""

    def __init__(self, m_dot: float):
        self._m_dot = m_dot
        super().__init__()

    def declare_components(self):
        self.add_component('m_dot_target', Parameter(self._m_dot, "kg/s"))
        self.add_component('p_in',         Variable(P_INIT, "Pa"))
        self.add_component('m_dot_in',     Variable(self._m_dot, "kg/s"))

    def declare_equations(self):
        return [self['m_dot_target'].symbol - self['m_dot_in'].symbol]


class _BufferTestSystem(Model):
    """Inlet pin -> LoopBuffer -> outlet pin (linear chain, no actual loop).

    This is enough to exercise the buffer's mass + energy balance and its
    new smooth-blend inlet under both flow directions; the loop-breaking
    rank-deficiency aspect is covered by `examples/loop_pump_pipe.py`.
    """

    def __init__(self, h_inlet: float):
        self._h_inlet = h_inlet
        super().__init__()

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('inlet',  _InletPin(self.medium, +M_DOT_BASE, self._h_inlet))
        self.add_component('buffer', LoopBuffer(self.medium, V=V_BUFFER,
                                                 p_init=P_INIT, T_init=T_INIT))
        self.add_component('outlet', _OutletPin(+M_DOT_BASE))

    def declare_equations(self):
        # Standard (p, h, m_dot) wiring -- exactly the same shape as
        # `examples/loop_pump_pipe.py`.  Under "flow into me", both faces
        # of every m_dot wire describe fluid entering their own component
        # at the shared interface, so the two values are equal in
        # magnitude with opposite sign -> sum-to-zero connection on the
        # flow channel.  `p` and `h` are across variables (single-valued
        # at the interface) and stay direct equalities.
        for var in ('p', 'h'):
            self.add_connection(self['inlet'][f'{var}_out'],
                                self['buffer'][f'{var}_in'])
        self.add_connection(self['inlet']['m_dot_out'],
                            self['buffer']['m_dot_in'],
                            sign=-1)
        self.add_connection(self['buffer']['p_out'],
                            self['outlet']['p_in'])
        self.add_connection(self['buffer']['m_dot_out'],
                            self['outlet']['m_dot_in'],
                            sign=-1)
        return []


def _trace(record, suffix: str) -> np.ndarray:
    names = list(record['vars_names'])
    state = np.asarray(record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


# --- Tests --------------------------------------------------------------------

def test_loop_buffer_subclasses_mixing_junction():
    """`LoopBuffer` must inherit from `MixingJunction` (so any code that
    isinstance-checks `MixingJunction` also accepts `LoopBuffer`)."""
    from hydrogen.components import MixingJunction
    assert issubclass(LoopBuffer, MixingJunction)


def test_loop_buffer_exposes_directional_ports():
    """The directional `_in`/`_out` API must stay intact -- the whole point
    of the subclass is to preserve it."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    buf = LoopBuffer(medium, V=V_BUFFER, p_init=P_INIT, T_init=T_INIT)
    for name in ('p_in', 'h_in', 'm_dot_in', 'p_out', 'h_out', 'm_dot_out'):
        assert name in buf.components, f"LoopBuffer missing port `{name}`"


@pytest.fixture(scope="module")
def forward_run():
    medium = CoolPropMedium("Air", disable_warnings=True)
    h_warm = float(medium.eval_h_pT(P_INIT, T_WARM))
    system = _BufferTestSystem(h_inlet=h_warm)
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    system.initialise(relaxation=0.5, max_iter=200)
    for _ in range(N_STEPS):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()
    return {"record": system.record, "h_warm": h_warm,
            "h_init": float(medium.eval_h_pT(P_INIT, T_INIT))}


def test_forward_energy_balance_warms_buffer(forward_run):
    """Steady forward inflow at h_warm > h_init must monotonically warm
    the buffer toward h_warm."""
    h = _trace(forward_run["record"], ".buffer.h")
    assert h[0] < forward_run["h_warm"]
    assert h[-1] > h[0] + 100.0, (
        f"buffer h should rise under warm inflow; got h[0]={h[0]:.1f}, h[-1]={h[-1]:.1f}"
    )
    assert h[-1] <= forward_run["h_warm"] + 1.0, (
        f"buffer h shouldn't overshoot inlet h; got h[-1]={h[-1]:.1f} vs h_warm={forward_run['h_warm']:.1f}"
    )
    assert np.all(np.diff(h) > 0), "h should be strictly increasing under steady warm inflow"


def test_forward_steady_mass_balance(forward_run):
    """`m_dot_in == m_dot_out` (both pinned to M_DOT_BASE) -> dm/dt == 0,
    so the buffer's mass is constant within numerical tolerance."""
    m = _trace(forward_run["record"], ".buffer.m")
    drift = abs(m.max() - m.min())
    bound = M_DOT_BASE * DT * N_STEPS * 1e-6
    assert drift < bound, f"buffer mass drifted by {drift:.3e} kg (expected < {bound:.3e})"


def test_forward_h_out_equals_bulk_h(forward_run):
    """The legacy `h_out == h` invariant must still hold (the port-1 blend
    is collapsed by the h_set_1 pin to bulk h)."""
    h = _trace(forward_run["record"], ".buffer.h")
    h_out = _trace(forward_run["record"], ".buffer.h_out")
    assert np.max(np.abs(h_out - h)) < 1e-9, (
        f"h_out should equal bulk h; max |h_out - h| = {np.max(np.abs(h_out - h)):.3e}"
    )


@pytest.fixture(scope="module")
def reverse_run():
    """Run forward first, then flip BOTH `m_dot`s to negative and verify
    the buffer keeps converging.  In steady reverse flow at the same
    magnitude, the inlet's smooth donor-cell carries the buffer's `h`
    out through the inlet while the outlet pin carries it back in, so
    `dU/dt = 0` and `dm/dt = 0` -- the buffer should freeze at whatever
    state forward flow drove it to."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    h_warm = float(medium.eval_h_pT(P_INIT, T_WARM))
    system = _BufferTestSystem(h_inlet=h_warm)
    system.instantiate(
        aditional_modules=system.medium.modules,
        max_remove_trival_passes=3,
    )
    system.initialise(relaxation=0.5, max_iter=200)

    # Phase 1: forward, warm up
    for _ in range(N_STEPS):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()

    # Capture state at end of phase 1 (before reversal).
    rec_idx_phase1_end = len(system.record['time']) - 1

    # Flip the sign of BOTH pinned mass flows.
    system['inlet']['m_dot_target'].value = -M_DOT_BASE
    system['outlet']['m_dot_target'].value = -M_DOT_BASE
    _push_params(system)

    # Phase 2: reverse, expect freeze at the warmed-up state.
    for _ in range(N_STEPS):
        system.solve_dae_step(DT, max_iter=200)
        system.next_step()

    return {"record": system.record, "n_phase1": rec_idx_phase1_end}


def test_reverse_flow_newton_converges(reverse_run):
    """Just running through reversal without a RuntimeError is the
    primary regression target: the legacy LoopBuffer would have happily
    integrated a wrong `m_dot_in * h_in` term with a negative m_dot_in
    and pushed the buffer into nonphysical territory before crashing
    in CoolProp.

    Under the "flow into me" convention, forward axial flow through the
    buffer reports m_dot_in > 0 (fluid entering at the inlet face) and
    m_dot_out < 0 (fluid leaving at the outlet face = negative "into
    me" rate); reverse flow flips both signs.
    """
    m_dot_in = _trace(reverse_run["record"], ".buffer.m_dot_in")
    m_dot_out = _trace(reverse_run["record"], ".buffer.m_dot_out")
    n1 = reverse_run["n_phase1"]
    # Forward phase: m_dot_in > 0 (inflow at inlet), m_dot_out < 0
    # (outflow at outlet under "into me" reporting).
    assert np.all(m_dot_in[:n1 + 1] > 0)
    assert np.all(m_dot_out[:n1 + 1] < 0)
    # Reverse phase: both signs flipped.
    assert np.all(m_dot_in[n1 + 1:] < 0)
    assert np.all(m_dot_out[n1 + 1:] > 0)


def test_reverse_flow_h_stays_constant(reverse_run):
    """In steady reverse flow at `m_dot_in == m_dot_out`, the smooth-blend
    inlet contribution is `-|m_dot_in| * h` (outflow of bulk h) while the
    outlet's pinned `h_set_1 == h` contributes `+|m_dot_out| * h`.  Net
    dU/dt == 0, so h must remain at whatever value forward flow left it
    -- to Newton tolerance."""
    h = _trace(reverse_run["record"], ".buffer.h")
    n1 = reverse_run["n_phase1"]
    h_after_reversal = h[n1 + 1:]
    drift = h_after_reversal.max() - h_after_reversal.min()
    assert drift < 1.0, (
        f"buffer h should stay frozen during steady reverse flow; "
        f"got drift = {drift:.3e} J/kg over {len(h_after_reversal)} steps"
    )


def test_reverse_flow_mass_stays_constant(reverse_run):
    """Mass balance is `dm/dt = m_dot_in - m_dot_out`; both negative and
    equal -> dm/dt == 0."""
    m = _trace(reverse_run["record"], ".buffer.m")
    n1 = reverse_run["n_phase1"]
    m_after_reversal = m[n1 + 1:]
    drift = abs(m_after_reversal.max() - m_after_reversal.min())
    bound = M_DOT_BASE * DT * N_STEPS * 1e-6
    assert drift < bound, (
        f"buffer mass should stay flat under steady reverse flow; "
        f"got drift = {drift:.3e} kg (expected < {bound:.3e})"
    )


def test_reverse_flow_eos_closure_holds(reverse_run):
    """`U = m*h - p*V` and `m = rho(p,h)*V` must still hold during reverse
    flow -- if Newton diverged into nonphysical territory these would
    fail."""
    medium = CoolPropMedium("Air", disable_warnings=True)
    rec = reverse_run["record"]
    m = _trace(rec, ".buffer.m")
    U = _trace(rec, ".buffer.U")
    p = _trace(rec, ".buffer.p")
    h = _trace(rec, ".buffer.h")
    eos_residual_U = U - (m * h - p * V_BUFFER)
    rho = np.array([medium.eval_rho_ph(float(pi), float(hi))
                    for pi, hi in zip(p, h)])
    eos_residual_m = m - rho * V_BUFFER
    assert np.max(np.abs(eos_residual_U)) < 1e-3
    assert np.max(np.abs(eos_residual_m)) < 1e-6
