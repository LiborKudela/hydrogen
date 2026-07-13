"""Tests for the local- (minor-) pressure-loss component and its pluggable
correlations (`FixedK`, `SuddenExpansion`, `SuddenContraction`).

Covers the ``Δp = K * rho * v**2 / 2`` constitutive law, the correlation
coefficients, serialization round-trip of the value objects, the `LocalLoss`
assembly (equivalent cylindrical wall: conjugate heat + gas permeation), the
single-volume dynamic body levels, and the acoustic guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen import CoolPropMedium, Model, from_dict, to_dict
from hydrogen.components.materials import WallMaterial
from hydrogen.components.thermofluid.assemblies import LocalLoss, WallLayer
from hydrogen.components.thermofluid.flow import (
    LocalResistance,
    PressureOutlet,
    PressureSource,
)
from hydrogen.components.thermofluid.local_loss import (
    FixedK,
    LaminarTransitionK,
    SuddenContraction,
    SuddenExpansion,
    local_loss_model_from_spec,
)
from hydrogen.components.thermofluid.permeation import H2, SpecifiedFlux

_WATER = CoolPropMedium('water', disable_warnings=True)
_AIR = CoolPropMedium('air', disable_warnings=True)
_H2 = CoolPropMedium('hydrogen', disable_warnings=True)
_STEEL = WallMaterial(name="steel", rho=7850.0, cp=500.0, k=15.0)


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


# --- bare LocalResistance (flow primitive) ----------------------------------

class _LossRig(Model):
    def __init__(self, loss_model, medium=_WATER, D=0.02, p_in=3e5, p_out=2e5):
        self._loss = loss_model
        self._medium = medium
        self._D = D
        self._p_in = p_in
        self._p_out = p_out
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(
            self._medium, p_source=self._p_in, T_source=300.0, A=1e-2))
        self.add_component('r', LocalResistance(
            self._medium, D=self._D, loss_model=self._loss))
        self.add_component('out', PressureOutlet(
            self._medium, p_ambient=self._p_out, T_ambient=300.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['r'].ports['inlet'])
        self.connect(self['r'].ports['outlet'], self['out'].ports['inlet'])
        return []


def test_fixed_k_loss_law_holds_at_solution():
    """The solved flow satisfies both the inverted sqrt law and, equivalently,
    the velocity-head definition ``Δp = K * rho * v**2 / 2``."""
    K, D = 2.0, 0.02
    val = _solve_n(_LossRig(FixedK(K), D=D), _WATER)
    dp = val('.r.p_0') - val('.r.p_1')
    rho = 0.5 * (val('.r.rho_0') + val('.r.rho_1'))
    m_dot = val('.r.m_dot_in')
    A = np.pi * D ** 2 / 4
    g = dp / (dp ** 2 + 1.0 ** 2) ** 0.25
    assert m_dot == pytest.approx(A * np.sqrt(2 * rho / K) * g, rel=1e-6)
    # Velocity-head form.
    v = m_dot / (rho * A)
    assert dp == pytest.approx(K * rho * v ** 2 / 2, rel=1e-4)
    assert m_dot > 0 and dp > 0


def test_higher_k_reduces_flow():
    """A larger loss coefficient throttles the flow (monotone)."""
    m_low = _solve_n(_LossRig(FixedK(1.0)), _WATER)('.r.m_dot_in')
    m_high = _solve_n(_LossRig(FixedK(8.0)), _WATER)('.r.m_dot_in')
    assert m_high < m_low
    # sqrt(K) scaling at (near-)fixed dp: m ~ 1/sqrt(K).
    assert m_high / m_low == pytest.approx(np.sqrt(1.0 / 8.0), rel=5e-2)


def test_sudden_expansion_matches_borda_carnot():
    """SuddenExpansion(beta) reproduces the flow of an equal FixedK with
    K = (1 - beta)**2 (Borda-Carnot, bore-referenced)."""
    beta = 0.4
    m_corr = _solve_n(_LossRig(SuddenExpansion(beta)), _WATER)('.r.m_dot_in')
    m_ref = _solve_n(_LossRig(FixedK((1 - beta) ** 2)), _WATER)('.r.m_dot_in')
    assert m_corr == pytest.approx(m_ref, rel=1e-6)


def test_sudden_contraction_matches_idelchik():
    """SuddenContraction(beta) reproduces the flow of an equal FixedK with
    K = 0.5*(1 - beta)**0.75 (Idelchik, bore-referenced)."""
    beta = 0.3
    m_corr = _solve_n(_LossRig(SuddenContraction(beta)), _WATER)('.r.m_dot_in')
    m_ref = _solve_n(_LossRig(FixedK(0.5 * (1 - beta) ** 0.75)),
                     _WATER)('.r.m_dot_in')
    assert m_corr == pytest.approx(m_ref, rel=1e-6)


# --- Reynolds-dependent K (dynamical K) -------------------------------------

def test_reynolds_dependent_k_closes():
    """A velocity-dependent coefficient ``K(Re) = K_turb + K_lam/Re`` closes: at
    the solution the loss law ``Δp = K(Re) * rho * v**2 / 2`` holds with ``K``
    evaluated at the solved bore Reynolds number."""
    K_turb, K_lam, D = 2.0, 900.0, 0.02
    val = _solve_n(_LossRig(LaminarTransitionK(K_turb, K_lam), D=D), _WATER)
    rho = 0.5 * (val('.r.rho_0') + val('.r.rho_1'))
    mu = 0.5 * (val('.r.mu_0') + val('.r.mu_1'))
    w = 0.5 * (val('.r.w_0') + val('.r.w_1'))
    dp = val('.r.p_0') - val('.r.p_1')
    m = val('.r.m_dot_in')
    A = np.pi * D ** 2 / 4
    v = m / (rho * A)
    Re = rho * abs(w) * D / mu + 1.0
    K = K_turb + K_lam / Re
    assert dp == pytest.approx(K * rho * v ** 2 / 2, rel=2e-3)
    assert m > 0 and dp > 0


def test_laminar_term_reduces_to_fixed_k_at_zero_k_lam():
    """With ``K_lam = 0`` the Reynolds-dependent model is exactly a `FixedK`."""
    m_re = _solve_n(_LossRig(LaminarTransitionK(3.0, 0.0)), _WATER)('.r.m_dot_in')
    m_fx = _solve_n(_LossRig(FixedK(3.0)), _WATER)('.r.m_dot_in')
    assert m_re == pytest.approx(m_fx, rel=1e-6)


# --- changing cross-section (D_in != D_out) + reference section -------------

class _AreaRig(Model):
    def __init__(self, loss_model, D_in, D_out, reference="inlet",
                 medium=_WATER, p_in=3e5, p_out=2e5):
        self._loss = loss_model
        self._D_in = D_in
        self._D_out = D_out
        self._reference = reference
        self._medium = medium
        self._p_in = p_in
        self._p_out = p_out
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(
            self._medium, p_source=self._p_in, T_source=300.0, A=1e-2))
        self.add_component('r', LocalResistance(
            self._medium, D=self._D_in, loss_model=self._loss,
            D_in=self._D_in, D_out=self._D_out, reference=self._reference))
        self.add_component('out', PressureOutlet(
            self._medium, p_ambient=self._p_out, T_ambient=300.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['r'].ports['inlet'])
        self.connect(self['r'].ports['outlet'], self['out'].ports['inlet'])
        return []


def test_area_change_momentum_is_bernoulli_plus_loss():
    """A contraction (D_in > D_out) solves the reversible (Bernoulli) area-change
    balance PLUS the irreversible inlet-referenced loss: reconstructing that
    exact residual from the solved state vanishes."""
    K, D_in, D_out = 2.5, 0.03, 0.02
    val = _solve_n(_AreaRig(FixedK(K), D_in, D_out, reference="inlet"), _WATER)
    A_in = np.pi * D_in ** 2 / 4
    A_out = np.pi * D_out ** 2 / 4
    A_avg = 0.5 * (A_in + A_out)
    p0, p1 = val('.r.p_0'), val('.r.p_1')
    w0, w1 = val('.r.w_0'), val('.r.w_1')
    rho = 0.5 * (val('.r.rho_0') + val('.r.rho_1'))
    m = 0.5 * (val('.r.M_0') + val('.r.M_1'))
    reversible = (p0 * A_in - p1 * A_out + 0.5 * (p0 + p1) * (A_out - A_in)
                  - m * (w1 - w0))
    dp_loss = K * rho * w0 * abs(w0) / 2          # inlet-referenced
    residual = reversible - dp_loss * A_avg
    # Normalise by the pressure-force scale.
    assert residual / (p0 * A_avg) == pytest.approx(0.0, abs=1e-6)
    assert val('.r.M_0') == pytest.approx(val('.r.M_1'), rel=1e-6)  # mass cons.
    assert w1 > w0                                 # contraction accelerates


def test_reference_section_changes_flow():
    """For a contraction the outlet velocity exceeds the inlet velocity, so an
    outlet-referenced K imposes a larger velocity-head loss and throttles the
    flow more than the same K referenced to the inlet."""
    m_in = _solve_n(_AreaRig(FixedK(3.0), 0.03, 0.02, "inlet"),
                    _WATER)('.r.m_dot_in')
    m_out = _solve_n(_AreaRig(FixedK(3.0), 0.03, 0.02, "outlet"),
                     _WATER)('.r.m_dot_in')
    assert m_out < m_in
    assert m_in > 0 and m_out > 0


def test_equal_area_default_unchanged_by_area_api():
    """D_in == D_out (== D) is exactly the equal-area element (no variable-area
    path); flows match the plain single-D resistance."""
    m_area = _solve_n(_AreaRig(FixedK(4.0), 0.02, 0.02), _WATER)('.r.m_dot_in')
    m_plain = _solve_n(_LossRig(FixedK(4.0), D=0.02), _WATER)('.r.m_dot_in')
    assert m_area == pytest.approx(m_plain, rel=1e-6)


def test_area_change_rejected_on_dynamic_levels():
    with pytest.raises(NotImplementedError, match="area-changing"):
        LocalResistance(_WATER, D=0.03, loss_model=FixedK(2.0),
                        D_in=0.03, D_out=0.02, dynamic="advective")


@pytest.mark.parametrize("model", [
    FixedK(3.5),
    SuddenExpansion(0.35),
    SuddenContraction(0.6),
    LaminarTransitionK(1.5, 300.0),
])
def test_loss_model_serialization_round_trip(model):
    d = model.to_spec()
    back = local_loss_model_from_spec(d)
    assert type(back) is type(model)
    assert back.to_spec() == d


def test_loss_model_validation():
    with pytest.raises(ValueError):
        FixedK(0.0)
    with pytest.raises(ValueError):
        SuddenExpansion(1.5)
    with pytest.raises(ValueError):
        SuddenContraction(0.0)


# --- single-volume dynamic body ---------------------------------------------

class _DynLossRig(Model):
    def __init__(self, dynamic, p_in=3e5, p_out=2e5):
        self._dynamic = dynamic
        self._p_in = p_in
        self._p_out = p_out
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(_AIR, p_source=self._p_in, T_source=300.0, A=1e-2))
        self.add_component('r', LocalResistance(
            _AIR, D=0.02, loss_model=FixedK(3.0), dynamic=self._dynamic,
            L_body=0.05, p_init=0.5 * (self._p_in + self._p_out)))
        self.add_component('out', PressureOutlet(_AIR, p_ambient=self._p_out, T_ambient=300.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['r'].ports['inlet'])
        self.connect(self['r'].ports['outlet'], self['out'].ports['inlet'])
        return []


def test_dynamic_local_loss_matches_static_steady_flow():
    """The dynamic single-volume body (two staggered loss faces around a
    storing cell) reproduces the static single-loss steady flow: the per-face
    law is scaled by sqrt(n_faces) so the series stack composes to the lumped
    drop."""
    stat = _solve_n(_DynLossRig("static"), _AIR, 1.0, 1)('.r.m_dot_in')
    dyn = _solve_n(_DynLossRig("advective"), _AIR, 1e-2, 120)('.r.m_dot_in')
    assert dyn == pytest.approx(stat, rel=0.12)


def test_dynamic_local_loss_body_pressure_equilibrates():
    """At steady state the storing body-cell pressure sits between the two
    ports (the two throttle faces share the drop)."""
    val = _solve_n(_DynLossRig("advective"), _AIR, 1e-2, 120)
    pc = val('.r.pc_0')
    assert val('.r.p_1') < pc < val('.r.p_0')


def test_acoustic_local_loss_rejected():
    with pytest.raises(NotImplementedError, match="acoustic"):
        LocalResistance(_WATER, D=0.02, loss_model=FixedK(2.0),
                        dynamic="acoustic")


# --- LocalLoss assembly (equivalent cylindrical wall) -----------------------

class _AssemblyRig(Model):
    def __init__(self, layers, loss_model=None, medium=_WATER,
                 outer_thermal="convective", p_in=3e5, p_out=2e5,
                 dynamic="static"):
        self._layers = layers
        self._loss = loss_model if loss_model is not None else FixedK(4.0)
        self._medium = medium
        self._outer = outer_thermal
        self._p_in = p_in
        self._p_out = p_out
        self._dynamic = dynamic
        super().__init__()

    def declare_components(self):
        self.add_component('src', PressureSource(self._medium, p_source=self._p_in, T_source=320.0, A=1e-2))
        self.add_component('r', LocalLoss(
            self._medium, D=0.02, loss_model=self._loss, L_body=0.08,
            layers=self._layers, outer_thermal=self._outer, h_ext=25.0,
            T_ext=290.0, dynamic=self._dynamic, T_wall_init=320.0,
            p_init=self._p_in,
            multiphase=("HEM" if self._medium is _WATER else "single")))
        self.add_component('out', PressureOutlet(self._medium, p_ambient=self._p_out, T_ambient=320.0))

    def declare_equations(self):
        self.connect(self['src'].ports['outlet'], self['r'].ports['inlet'])
        self.connect(self['r'].ports['outlet'], self['out'].ports['inlet'])
        return []


def test_local_loss_assembly_is_catalog_component():
    from hydrogen.serialization.registry import builtin_registry
    assert builtin_registry()['hydrogen.thermofluid.LocalLoss'] is LocalLoss


def test_local_loss_assembly_thermal_wall_passes_flow():
    layers = [WallLayer(material=_STEEL, thickness=0.003, dynamic=True)]
    val = _solve_n(_AssemblyRig(layers, loss_model=FixedK(4.0)), _WATER, 1.0, 1)
    dp = val('.r.loss.p_0') - val('.r.loss.p_1')
    m = val('.r.loss.m_dot_in')
    rho = 0.5 * (val('.r.loss.rho_0') + val('.r.loss.rho_1'))
    A = np.pi * 0.02 ** 2 / 4
    g = dp / (dp ** 2 + 1.0 ** 2) ** 0.25
    assert m == pytest.approx(A * np.sqrt(2 * rho / 4.0) * g, rel=1e-5)
    assert m > 0 and dp > 0


def test_local_loss_assembly_specified_permeation_leaks():
    perm = SpecifiedFlux(H2, leak_rate=5.0, scaling="linear")
    layers = [WallLayer(material=_STEEL, thickness=0.003, permeation=perm)]
    val = _solve_n(_AssemblyRig(layers, medium=_H2, p_in=5e5, p_out=3e5), _H2, 1.0, 1)
    assert abs(val('.r.m_dot_leak_env')) > 0.0


def test_local_loss_assembly_bare_body_builds():
    """No layers + adiabatic outer = a plain (isenthalpic) local loss."""
    val = _solve_n(_AssemblyRig([], outer_thermal="adiabatic"), _WATER, 1.0, 1)
    assert val('.r.loss.m_dot_in') > 0.0


def test_local_loss_must_be_sole_leaky_layer():
    perm = SpecifiedFlux(H2, leak_rate=5.0)
    other = WallLayer(material=_STEEL, thickness=0.003,
                      permeation=SpecifiedFlux(H2, leak_rate=1.0))
    with pytest.raises(ValueError, match="ONLY leaky layer"):
        LocalLoss(_H2, D=0.02,
                  layers=[WallLayer(material=_STEEL, thickness=0.003,
                                    permeation=perm), other])


def test_local_loss_assembly_serialization_round_trip():
    layers = [WallLayer(material=_STEEL, thickness=0.003)]
    rig = _AssemblyRig(layers, loss_model=SuddenExpansion(0.4))
    rig.declare_equations()
    d = to_dict(rig)
    rspec = d['components']['r']
    assert rspec['type'] == 'hydrogen.thermofluid.LocalLoss'
    assert rspec['params']['loss_model']['__type__'] == 'SuddenExpansion'
    assert rspec['params']['loss_model']['area_ratio'] == 0.4
    rebuilt = from_dict(d)
    assert 'r' in rebuilt.components
