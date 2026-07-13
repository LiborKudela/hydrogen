"""Tests for the variable per-face flow area of `SegmentedChannel`.

A `SegmentedChannel` given an `A_faces` profile (one flow area per shared face)
becomes a variable-area duct: the mass-flow closure, cell volumes, wetted areas
and hydraulic diameters follow the taper, and the STATIC momentum balance turns
into the reversible (Bernoulli) area-change relation.  A uniform bore (``D``
only, or an ``A_faces`` whose values are all equal) must reproduce the previous
single-area element exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen import CoolPropMedium, Model
from hydrogen.components.thermofluid.flow import (
    PressureOutlet,
    PressureSource,
    SegmentedChannel,
)

_WATER = CoolPropMedium('water', disable_warnings=True)
_AIR = CoolPropMedium('air', disable_warnings=True)


def _solve_n(model, medium, dt=1.0, n=1):
    model.instantiate(aditional_modules=medium.modules, max_remove_trival_passes=5)
    model.initialise(n=1)
    for _ in range(n):
        model.solve_dae_step(dt)
    names = list(model.record['vars_names'])
    last = np.asarray(model.record['state'])[-1]

    def val(suffix):
        return last[next(i for i, nm in enumerate(names) if nm.endswith(suffix))]

    return val


def _taper(A_in, A_out, N):
    """Linear area profile with N+1 faces."""
    return [A_in + (A_out - A_in) * j / N for j in range(N + 1)]


class _ChannelRig(Model):
    """PressureSource -> SegmentedChannel -> PressureOutlet."""

    def __init__(self, *, medium=_WATER, D=0.05, L=0.02, N=1, A_faces=None,
                 epsilon=0.0, p_in=2.2e5, p_out=2.0e5, dynamic="static",
                 T=300.0):
        self._medium = medium
        self._D = D
        self._L = L
        self._N = N
        self._A_faces = A_faces
        self._epsilon = epsilon
        self._p_in = p_in
        self._p_out = p_out
        self._dynamic = dynamic
        self._T = T
        super().__init__()

    def declare_components(self):
        A0 = self._A_faces[0] if self._A_faces else np.pi * self._D ** 2 / 4
        self.add_component('src', PressureSource(
            self._medium, p_source=self._p_in, T_source=self._T, A=A0))
        self.add_component('ch', SegmentedChannel(
            self._medium, D=self._D, L=self._L, epsilon=self._epsilon,
            z_in=0.0, z_out=0.0, N=self._N, A_faces=self._A_faces,
            dynamic=self._dynamic, p_init=self._p_out))
        self.add_component('out', PressureOutlet(
            self._medium, p_ambient=self._p_out, T_ambient=self._T))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['ch'].ports['inlet'])
        self.connect(self['ch'].ports['outlet'], self['out'].ports['inlet'])
        return []


# --- backward compatibility: uniform A_faces == single D --------------------

def test_uniform_A_faces_matches_single_D():
    """An `A_faces` profile whose values are all `pi*D**2/4` gives the same
    steady solution as the plain uniform-bore channel (the variable-area path
    reduces to the historical single-area element)."""
    D, N = 0.03, 3
    A = np.pi * D ** 2 / 4
    ref = _solve_n(_ChannelRig(D=D, N=N), _WATER)
    var = _solve_n(_ChannelRig(D=D, N=N, A_faces=[A] * (N + 1)), _WATER)
    for suffix in ('.ch.m_dot_in', '.ch.p_0', f'.ch.p_{N}',
                   '.ch.w_0', f'.ch.w_{N}'):
        assert var(suffix) == pytest.approx(ref(suffix), rel=1e-6, abs=1e-9)


def test_variable_area_validation():
    """`A_faces` must have length N+1 and be strictly positive."""
    with pytest.raises(ValueError, match="N.1"):
        SegmentedChannel(_WATER, D=0.02, L=0.1, epsilon=0.0, z_in=0.0,
                         z_out=0.0, N=3, A_faces=[1e-3, 1e-3])
    with pytest.raises(ValueError, match="> 0"):
        SegmentedChannel(_WATER, D=0.02, L=0.1, epsilon=0.0, z_in=0.0,
                         z_out=0.0, N=1, A_faces=[1e-3, 0.0])


# --- mass conservation through a taper --------------------------------------

def test_converging_nozzle_conserves_mass_and_accelerates():
    """A converging channel conserves axial mass flow and accelerates the flow
    inversely with the area (rho*A*w continuity), so the outlet velocity is
    ~A_in/A_out times the inlet velocity."""
    A_in = np.pi * 0.05 ** 2 / 4
    A_out = A_in / 2.0
    val = _solve_n(_ChannelRig(A_faces=[A_in, A_out], N=1), _WATER)
    m_in = val('.ch.M_0')
    m_out = val('.ch.M_1')
    assert m_in == pytest.approx(m_out, rel=1e-6)          # axial mass flow
    w0, w1 = val('.ch.w_0'), val('.ch.w_1')
    rho0, rho1 = val('.ch.rho_0'), val('.ch.rho_1')
    # rho*A*w conserved => w1/w0 ~ (rho0*A_in)/(rho1*A_out) ~ 2 for a liquid.
    assert (rho1 * A_out * w1) == pytest.approx(rho0 * A_in * w0, rel=1e-6)
    assert w1 > w0
    assert val('.ch.p_1') < val('.ch.p_0')                 # accelerate -> drop


# --- reversible Bernoulli behaviour (static) --------------------------------

def test_static_nozzle_conserves_total_pressure():
    """With negligible friction the STATIC variable-area balance is reversible:
    the total pressure ``p + rho*w**2/2`` is (nearly) conserved across a smooth
    converging nozzle -- the hallmark of the Bernoulli area-change relation
    (as opposed to a lossy sudden-contraction form)."""
    A_in = np.pi * 0.05 ** 2 / 4
    A_faces = _taper(A_in, A_in / 2.0, N=12)
    val = _solve_n(_ChannelRig(A_faces=A_faces, N=12,
                               p_in=2.4e5, p_out=2.0e5), _WATER)
    p0, w0, rho0 = val('.ch.p_0'), val('.ch.w_0'), val('.ch.rho_0')
    pN, wN, rhoN = val('.ch.p_12'), val('.ch.w_12'), val('.ch.rho_12')
    tot0 = p0 + 0.5 * rho0 * w0 ** 2
    totN = pN + 0.5 * rhoN * wN ** 2
    assert wN > w0 and pN < p0
    # Total pressure conserved to within the (small) wall-friction loss.
    assert totN == pytest.approx(tot0, rel=3e-2)
    assert totN <= tot0 + 1.0    # friction can only remove total pressure


def test_diffuser_recovers_static_pressure():
    """A diverging channel (diffuser) decelerates the flow and RECOVERS static
    pressure -- reversible behaviour only the Bernoulli (wall-pressure) balance
    can produce; a bare `p_in*A_in - p_out*A_out` form could not."""
    A_in = np.pi * 0.035 ** 2 / 4
    A_faces = _taper(A_in, 2.0 * A_in, N=12)
    val = _solve_n(_ChannelRig(A_faces=A_faces, N=12,
                               p_in=2.4e5, p_out=2.0e5), _WATER)
    w0, wN = val('.ch.w_0'), val('.ch.w_12')
    p0, pN = val('.ch.p_0'), val('.ch.p_12')
    assert wN < w0            # decelerates
    assert pN > p0            # static pressure recovers downstream


# --- transient level builds with a taper ------------------------------------

def test_advective_taper_builds_and_conserves_mass():
    """The advective (transient-energy) level accepts a variable-area profile:
    it builds, solves, and at steady state carries a uniform axial mass flow
    through the taper (quasi-steady continuity).  Uses (near-incompressible)
    water with real wall friction -- the regime the advective level targets."""
    A_in = np.pi * 0.04 ** 2 / 4
    A_faces = _taper(A_in, A_in / 1.5, N=4)
    val = _solve_n(_ChannelRig(A_faces=A_faces, N=4, L=0.5, epsilon=1e-4,
                               p_in=3e5, p_out=2e5, dynamic="advective"),
                   _WATER, dt=0.5, n=4)
    flows = [val(f'.ch.M_{j}') for j in range(5)]
    assert all(f == pytest.approx(flows[0], rel=1e-3) for f in flows)
    assert flows[0] > 0
