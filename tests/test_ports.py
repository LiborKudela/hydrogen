"""Tests for the typed-port `connect()` API and signed `add_connection`.

Covers:
  * Standard `out -> in` wires: equality (sign = +1) per channel; identical to
    the legacy `add_connection` per-channel loop.
  * Same-orientation wires (`in -> in` and `out -> out`): sum-to-zero on the
    flow channel via signed UF; equality on the across channels.
  * Multiplicity: a port can be wired exactly once (`PortAlreadyConnectedError`).
  * Kind mismatch: connecting fluid to thermal raises `PortKindMismatchError`.
  * Medium mismatch: two fluid ports with different `medium` instances raise.
  * Signed UF cycle detection: an inconsistent loop is reported at instantiate.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen import (
    FluidPort_phm,
    Model,
    PortAlreadyConnectedError,
    PortKindMismatchError,
    PortMediumMismatchError,
    ThermalPort_TQ,
    Variable,
)
from hydrogen.components import (
    AmbientInlet,
    PressureOutlet,
    PressureSource,
    StraightPipe,
)
from hydrogen.medium import CoolPropMedium


# ---------------------------------------------------------------------------
# Standard out -> in wiring through `connect()` matches the legacy behavior
# ---------------------------------------------------------------------------

class _OutInSystemPorts(Model):
    """`PressureSource -> StraightPipe -> PressureOutlet` wired through the
    new typed-port `connect()` API."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('source', PressureSource(
            self.medium, p_source=1.2e5, T_source=293.15, A=np.pi * 0.02**2 / 4,
        ))
        self.add_component('pipe', StraightPipe(
            self.medium, D=0.02, L=1.0, epsilon=1e-5,
            z_in=0.0, z_out=0.0, n_segments=2, adiabatic=True,
        ))
        self.add_component('outlet', PressureOutlet(
            self.medium, p_ambient=1.0e5, T_ambient=293.15,
        ))

    def declare_equations(self):
        self.connect(self['source'].ports['outlet'],
                     self['pipe'].ports['inlet'])
        self.connect(self['pipe'].ports['outlet'],
                     self['outlet'].ports['inlet'])
        return []


class _OutInSystemLegacy(Model):
    """The SAME topology wired through the old per-channel `add_connection`
    loop; used as a head-to-head structural comparison fixture for
    `_OutInSystemPorts` above."""

    def declare_components(self):
        self.medium = CoolPropMedium("Air", disable_warnings=True)
        self.add_component('source', PressureSource(
            self.medium, p_source=1.2e5, T_source=293.15, A=np.pi * 0.02**2 / 4,
        ))
        self.add_component('pipe', StraightPipe(
            self.medium, D=0.02, L=1.0, epsilon=1e-5,
            z_in=0.0, z_out=0.0, n_segments=2, adiabatic=True,
        ))
        self.add_component('outlet', PressureOutlet(
            self.medium, p_ambient=1.0e5, T_ambient=293.15,
        ))

    def declare_equations(self):
        for io in ('p', 'h', 'm_dot'):
            self.add_connection(self['source'][f'{io}_out'],
                                self['pipe'][f'{io}_in'])
            self.add_connection(self['pipe'][f'{io}_out'],
                                self['outlet'][f'{io}_in'])
        return []


def test_connect_matches_legacy_add_connection_structurally():
    """The new typed-port `connect()` API must produce the SAME reduced
    symbolic system as the legacy per-channel `add_connection` loop on a
    canonical `source -> pipe -> outlet` topology.

    "Same" here is structural: identical number of surviving variables AND
    identical number of surviving equations after the full instantiate
    pipeline (signed UF + trivial reducer + dedup pass).  Identity of the
    Newton path itself is the strongest guarantee we can give without
    solving (which is fragile to initial-guess details independent of
    wiring)."""
    p = _OutInSystemPorts()
    p.instantiate(aditional_modules=p.medium.modules, max_remove_trival_passes=3)

    L = _OutInSystemLegacy()
    L.instantiate(aditional_modules=L.medium.modules, max_remove_trival_passes=3)

    # `n_v` and the active variable names are the post-reduction surface that
    # the runtime Newton solve sees.  If wiring is structurally equivalent
    # these must agree exactly.  We compare names with the top-level system
    # class prefix stripped, so the deliberately distinct class names
    # (`_OutInSystemPorts` vs `_OutInSystemLegacy`) don't fail the diff.
    assert p.n_v == L.n_v, (
        f"port-based system has n_v={p.n_v} active vars, legacy n_v={L.n_v}"
    )

    def _strip_root(name: str) -> str:
        return name.split(".", 1)[1] if "." in name else name

    p_names = sorted(_strip_root(v.full_name) for v in p.active_vars_references)
    L_names = sorted(_strip_root(v.full_name) for v in L.active_vars_references)
    assert p_names == L_names, (
        "active variable sets differ between port-based and legacy wiring:\n"
        f"  only in port-based: {set(p_names) - set(L_names)}\n"
        f"  only in legacy:     {set(L_names) - set(p_names)}"
    )


# ---------------------------------------------------------------------------
# Same-orientation wire emits sum-to-zero on flow channels (signed UF)
# ---------------------------------------------------------------------------

class _PinnedInletNode(Model):
    """`flow_orientation='in'` port whose THREE channels are all pinned by
    its own residuals.  Used as the "active" side of a same-orientation
    wire test: this node fully constrains its own port, and a sibling
    `_FloatingInletNode` (no equations, also `in`) connects to it.
    """

    def __init__(self, p, h, m):
        self._p, self._h, self._m = p, h, m
        super().__init__()

    def declare_components(self):
        self.add_component('p_in', Variable(self._p))
        self.add_component('h_in', Variable(self._h))
        self.add_component('m_dot_in', Variable(self._m))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        return [
            self['p_in'].symbol - self._p,
            self['h_in'].symbol - self._h,
            self['m_dot_in'].symbol - self._m,
        ]


class _FloatingInletNode(Model):
    """Mirror of `_PinnedInletNode` but contributes NO equations.  Its three
    port channels are determined entirely by the wire from the pinned side
    (via union-find), so the global system stays square (3 surviving vars
    after UF unifies the six raw vars, 3 residuals from the pinned side)."""

    def declare_components(self):
        self.add_component('p_in', Variable(0.0))
        self.add_component('h_in', Variable(0.0))
        self.add_component('m_dot_in', Variable(0.0))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        return []


class _SameOrientSystem(Model):
    """Two `_in`-orientation ports wired together.  Under the Kirchhoff /
    Modelica rule (and our `connect()` implementation), the FLOW channel
    obeys `a.m_dot + b.m_dot == 0` while the across channels (`p`, `h`)
    still satisfy `a == b`.  This is the topology you get when a junction's
    "into me" port is wired to a pipe's "into me" inlet (flow going
    junction -> pipe; physically one's inflow IS the other's outflow)."""

    def declare_components(self):
        self.add_component('a', _PinnedInletNode(p=1.0e5, h=3.0e5, m=+0.4))
        self.add_component('b', _FloatingInletNode())

    def declare_equations(self):
        self.connect(self['a'].ports['inlet'], self['b'].ports['inlet'])
        return []


def test_same_orientation_wire_sums_flow_to_zero():
    sys = _SameOrientSystem()
    sys.instantiate(max_remove_trival_passes=3)
    sys.initialise()

    state = np.asarray(sys.record['state'])[-1]
    names = list(sys.record['vars_names'])

    def get(suffix):
        return state[next(i for i, n in enumerate(names) if n.endswith(suffix))]

    # ACROSS channels: union-find sets a.p == b.p and a.h == b.h.
    assert abs(float(get('.a.p_in')) - float(get('.b.p_in'))) < 1e-9
    assert abs(float(get('.a.h_in')) - float(get('.b.h_in'))) < 1e-9
    # FLOW channel: signed UF (`sign=-1`) means `a.m_dot + b.m_dot == 0`.
    m_a = float(get('.a.m_dot_in'))
    m_b = float(get('.b.m_dot_in'))
    assert abs(m_a + m_b) < 1e-9, (
        f"same-orientation wire should give a.m_dot + b.m_dot == 0; "
        f"got {m_a} + {m_b} = {m_a + m_b}"
    )
    # Pinned side fixed at +0.4 -> floating side must take -0.4 via the
    # signed connection.
    assert abs(m_a - 0.4) < 1e-9
    assert abs(m_b - (-0.4)) < 1e-9


# ---------------------------------------------------------------------------
# Multiplicity / kind / medium guards fire at connect() time
# ---------------------------------------------------------------------------

def _make_pair_models():
    medium = CoolPropMedium("Air", disable_warnings=True)
    a = AmbientInlet(medium, p_ambient=101325, T_ambient=293.15, m_flow=0.05, D=0.02)
    b = StraightPipe(medium, D=0.02, L=1.0, epsilon=1e-5,
                      z_in=0.0, z_out=0.0, n_segments=2, adiabatic=True)
    return medium, a, b


def test_double_connect_raises():
    """A port can be wired exactly once.  The second connect raises with a
    pointer to the existing wire."""

    class _Bad(Model):
        def declare_components(self):
            self.medium, self._inlet_model, self._pipe_model = _make_pair_models()
            self.add_component('inlet', self._inlet_model)
            self.add_component('pipe1', self._pipe_model)
            # A second pipe that we'll try to wire to the same inlet outlet.
            self.add_component('pipe2', StraightPipe(
                self.medium, D=0.02, L=1.0, epsilon=1e-5,
                z_in=0.0, z_out=0.0, n_segments=2, adiabatic=True,
            ))

        def declare_equations(self):
            self.connect(self['inlet'].ports['outlet'],
                         self['pipe1'].ports['inlet'])
            # Attempted second connection to the same already-wired port.
            self.connect(self['inlet'].ports['outlet'],
                         self['pipe2'].ports['inlet'])
            return []

    with pytest.raises(PortAlreadyConnectedError):
        # The error fires inside declare_equations during instantiate().
        _Bad().instantiate(max_remove_trival_passes=0)


def test_kind_mismatch_raises():
    medium = CoolPropMedium("Air", disable_warnings=True)

    class _Bad(Model):
        def declare_components(self):
            self.add_component('inlet', AmbientInlet(
                medium, p_ambient=101325, T_ambient=293.15, m_flow=0.05, D=0.02,
            ))
            # A throwaway sub-model that exposes a ThermalPort_TQ with the
            # same channel names but a different `kind`.
            class _ThermalStub(Model):
                def declare_components(_self):
                    _self.add_component('T', Variable(300.0))
                    _self.add_component('Q_dot', Variable(0.0))
                    _self.add_port('port', ThermalPort_TQ(
                        _self,
                        channels={'T': _self['T'], 'Q_dot': _self['Q_dot']},
                        flow_orientation='in',
                    ))
                def declare_equations(_self):
                    return [_self['T'].symbol - 300.0, _self['Q_dot'].symbol - 0.0]
            self.add_component('therm', _ThermalStub())

        def declare_equations(self):
            self.connect(self['inlet'].ports['outlet'],
                         self['therm'].ports['port'])
            return []

    with pytest.raises(PortKindMismatchError):
        _Bad().instantiate(max_remove_trival_passes=0)


def test_medium_mismatch_raises():
    """Two FluidPort_phm with different `medium` instances cannot be wired."""
    air = CoolPropMedium("Air", disable_warnings=True)
    h2 = CoolPropMedium("Hydrogen", disable_warnings=True)

    class _Bad(Model):
        def declare_components(self):
            self.add_component('inlet', AmbientInlet(
                air, p_ambient=101325, T_ambient=293.15, m_flow=0.05, D=0.02,
            ))
            self.add_component('pipe', StraightPipe(
                h2, D=0.02, L=1.0, epsilon=1e-5,
                z_in=0.0, z_out=0.0, n_segments=2, adiabatic=True,
            ))

        def declare_equations(self):
            self.connect(self['inlet'].ports['outlet'],
                         self['pipe'].ports['inlet'])
            return []

    with pytest.raises(PortMediumMismatchError):
        _Bad().instantiate(max_remove_trival_passes=0)


# ---------------------------------------------------------------------------
# Signed UF: inconsistent cycle is surfaced at instantiate()
# ---------------------------------------------------------------------------

class _InconsistentCycle(Model):
    """`x == y`, `x == -y`  =>  forces 2x = 0, i.e. all vars in the class go
    to zero.  The signed-UF path raises this as a topology error rather
    than letting the solver silently converge to a degenerate state.

    Two separate residuals so that the system would otherwise be square.
    """

    def declare_components(self):
        self.add_component('x', Variable(1.0))
        self.add_component('y', Variable(1.0))

    def declare_equations(self):
        self.add_connection(self['x'], self['y'], sign=+1)
        self.add_connection(self['x'], self['y'], sign=-1)
        return [self['x'].symbol - 1.0]


def test_signed_uf_detects_inconsistent_cycle():
    with pytest.raises(ValueError, match="Inconsistent signed-connection cycle"):
        _InconsistentCycle().instantiate(max_remove_trival_passes=0)
