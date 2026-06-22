"""Tests for the gas-permeation physics of the thermofluid domain.

Covers:

  1. `TransportFit` Arrhenius properties and the `Phi = D * S` identity, plus
     the austenitic-SS hydrogen fit.
  2. Flag validation and per-class cache keying on a leaky `CylindricalWall`.
  3. Steady leak == the closed-form radial Richardson flux.
  4. The transient n-shell wall relaxes to the SAME steady leak, independent of
     `n_nodes` (the conductance chain telescopes to the exact Richardson rate).
  5. End to end: a leaky `StraightPipe` flow pipe pressurised to 20 MPa drives the
     wall, and pipe continuity holds (inlet make-up == wall leak).

The wall-only tests drive the wall's `leak` port from a fixed partial-pressure
boundary, so they isolate the permeation physics from any fluid-flow solve and
stay fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydrogen import (
    H2,
    H2_IN_AUSTENITIC,
    HELIUM,
    AISI_304,
    AISI_316,
    ClosedEnd,
    CoolPropMedium,
    CylindricalWall,
    FixedPartialPressure,
    FixedTemperature,
    Model,
    Pipe,
    PressureSource,
    SteadyRichardson,
    StraightPipe,
    TransientDiffusion,
    TransportFit,
    WallLayer,
    WallMaterial,
)
from hydrogen.components.materials import R_GAS

#: Thermal wall material + matching hydrogen transport fit used by the rigs.
MAT = AISI_316
FIT = H2_IN_AUSTENITIC


def _flux(fit, leak_model, n_nodes=6):
    """Map a `leak_model` string onto a permeation flux model."""
    if leak_model == "steady":
        return SteadyRichardson(fit)
    return TransientDiffusion(fit, n_nodes=n_nodes)


# --- geometry / operating point (matches the example) -----------------------
R_IN = 1.5e-3
R_OUT = 3.0e-3
L_PIPE = 0.100
T_OP = 423.15        # 150 C
P_OP = 20.0e6        # 20 MPa
P_EXT = 1.0          # external partial pressure ~ 0


def _leaky_wall(flux, length=L_PIPE, T=T_OP, p_in=P_OP, p_out=P_EXT):
    """A leaky `CylindricalWall` built from the thermal material + a flux model."""
    return CylindricalWall(
        MAT.rho, MAT.cp, MAT.k, r_in=R_IN, r_out=R_OUT, length=length,
        T_init=T, dynamic=False, leaky=True, permeation_flux=flux,
        p_in_init=p_in, p_out_init=p_out)


def analytic_steady_leak(fit, p_in, T=T_OP, p_out=P_EXT):
    """Closed-form radial Richardson leak [kg/s]."""
    Phi = float(fit.Phi(T))
    N_dot = (2 * math.pi * Phi * L_PIPE / math.log(R_OUT / R_IN)
             * (math.sqrt(p_in) - math.sqrt(p_out)))
    return fit.permeant.M * N_dot


class _WallRig(Model):
    """`FixedPartialPressure(p) -> leaky CylindricalWall -> FixedPartialPressure(P_EXT)`,
    both thermal faces held at T."""

    def __init__(self, fit, leak_model, n_nodes=6, p_in=P_OP, T=T_OP):
        self._fit = fit
        self._leak_model = leak_model
        self._n_nodes = n_nodes
        self._p = p_in
        self._T = T
        super().__init__()

    def declare_components(self):
        self.add_component("res", FixedPartialPressure(p_partial=self._p))
        self.add_component("wall", _leaky_wall(
            _flux(self._fit, self._leak_model, self._n_nodes),
            T=self._T, p_in=self._p, p_out=P_EXT))
        self.add_component("env", FixedPartialPressure(p_partial=P_EXT))
        self.add_component("T_in", FixedTemperature(T_set=self._T))
        self.add_component("T_out", FixedTemperature(T_set=self._T))

    def declare_equations(self):
        # Inner surface driven by the reservoir; outer surface vents to P_EXT.
        self.connect(self["res"].ports["leak"], self["wall"].ports["leak_a"])
        self.connect(self["env"].ports["leak"], self["wall"].ports["leak_b"])
        self.connect(self["wall"].ports["port_a"], self["T_in"].ports["heat"])
        self.connect(self["wall"].ports["port_b"], self["T_out"].ports["heat"])
        return []


def _val(system, suffix):
    names = list(system.record["vars_names"])
    i = next(i for i, n in enumerate(names)
             if n == suffix or n.endswith("." + suffix))
    return float(np.asarray(system.record["state"][-1])[i])


# -----------------------------------------------------------------------------
# 1. Transport-fit Arrhenius properties
# -----------------------------------------------------------------------------


def test_transport_fit_arrhenius_and_identity():
    fit = FIT
    # Phi = D * S exactly (S is defined as Phi/D).
    for T in (300.0, 423.15, 600.0):
        assert float(fit.Phi(T)) == pytest.approx(
            float(fit.D(T)) * float(fit.S(T)), rel=1e-12)
    # Arrhenius slope matches the stored activation energy.
    T1, T2 = 400.0, 500.0
    slope = (math.log(float(fit.Phi(T2))) - math.log(float(fit.Phi(T1)))) \
        / (1.0 / T2 - 1.0 / T1)
    assert slope == pytest.approx(-fit.E_Phi / R_GAS, rel=1e-9)
    # Known order of magnitude at 150 C (austenitic SS).
    assert float(fit.Phi(T_OP)) == pytest.approx(5.0e-15, rel=0.15)
    # The fit is for hydrogen (Sieverts, n = 2).
    assert fit.permeant is H2
    assert fit.permeant.solubility_exponent == 2.0


def test_aisi_presets_differ_thermally_share_transport_fit():
    # 304 and 316 differ in thermal conductivity ...
    assert AISI_304.k != AISI_316.k
    assert isinstance(AISI_316, WallMaterial)
    # ... and a single austenitic transport fit covers both (composition-
    # independent permeability), so there is one shared TransportFit, not one
    # per steel grade.
    assert isinstance(FIT, TransportFit)


# -----------------------------------------------------------------------------
# 2. Flag validation / cache key
# -----------------------------------------------------------------------------


def test_leaky_flag_and_flux_validated_and_cache_keyed():
    # The leaky toggle and the injected flux model's structural identity both
    # key the per-class equation cache (alongside the thermal `dynamic` flag).
    # `dynamic`/`leaky` are derived from their `structural=True` ParamSpecs and
    # `_perm_key` from the class's explicit (computed) `_cache_key_flags`.
    from hydrogen.paramspec import cache_key_flag_names
    derived = cache_key_flag_names(CylindricalWall)
    for flag in ("dynamic", "leaky", "_perm_key"):
        assert flag in derived
    # Distinct flux models must produce distinct cache keys.
    assert SteadyRichardson(FIT).cache_key != TransientDiffusion(FIT).cache_key
    assert (TransientDiffusion(FIT, n_nodes=3).cache_key
            != TransientDiffusion(FIT, n_nodes=4).cache_key)
    # A leaky wall requires a flux model.
    with pytest.raises(ValueError):
        CylindricalWall(8000.0, 500.0, 15.0, R_IN, R_OUT, L_PIPE, leaky=True)
    # The transient flux validates its node count.
    with pytest.raises(ValueError):
        TransientDiffusion(FIT, n_nodes=0)


def test_surface_law_exponent_keys_distinct_fluxes():
    # A Henry permeant (n = 1) produces a structurally different residual than a
    # Sieverts one (n = 2), so their cache keys must differ.
    he_fit = TransportFit(permeant=HELIUM, Phi0=1e-12, E_Phi=40e3,
                          D0=1e-7, E_D=30e3, name="He test")
    assert SteadyRichardson(FIT).cache_key != SteadyRichardson(he_fit).cache_key


# -----------------------------------------------------------------------------
# 3. Steady leak == analytic Richardson flux
# -----------------------------------------------------------------------------


def test_steady_leak_matches_analytic_richardson():
    sys = _WallRig(FIT, leak_model="steady")
    sys.instantiate()
    sys.initialise(n=1, tol=1e-10, max_iter=100)

    leak = _val(sys, "wall.m_dot_a_leak")   # inner uptake, into the wall (> 0)
    env = -_val(sys, "wall.m_dot_b_leak")   # outer venting to environment (> 0)
    expected = analytic_steady_leak(FIT, P_OP)
    assert leak == pytest.approx(expected, rel=1e-6)
    # In steady mode the inner uptake and the environment flux are identical.
    assert env == pytest.approx(expected, rel=1e-6)


# -----------------------------------------------------------------------------
# 4. Transient relaxes to the steady leak, independent of n_nodes
# -----------------------------------------------------------------------------


def _march_to_steady(n_nodes, days=400, dt0=0.25 * 86400.0):
    sys = _WallRig(FIT, leak_model="transient", n_nodes=n_nodes)
    sys.instantiate()
    sys.initialise(n=1, tol=1e-9, max_iter=200)
    day = 86400.0
    t, dt = 0.0, dt0
    while t < days * day:
        sys.solve_dae_step(dt, tol=1e-9, max_iter=200,
                           raise_on_no_convergence=True)
        sys.next_step()
        t += dt
        dt = min(dt * 1.3, 20 * day)
    return -_val(sys, "wall.m_dot_b_leak"), _val(sys, "wall.m_dot_a_leak")


def test_transient_relaxes_to_analytic_steady():
    env, uptake = _march_to_steady(n_nodes=6)
    expected = analytic_steady_leak(FIT, P_OP)
    # After many diffusion time-constants both the environment flux and the
    # inner uptake collapse onto the steady Richardson rate.
    assert env == pytest.approx(expected, rel=1e-3)
    assert uptake == pytest.approx(expected, rel=1e-3)


def test_transient_steady_limit_is_node_count_independent():
    expected = analytic_steady_leak(FIT, P_OP)
    env3, _ = _march_to_steady(n_nodes=3)
    env10, _ = _march_to_steady(n_nodes=10)
    # The (n+1) equal-ln conductances telescope to the exact Richardson
    # resistance for ANY n, so the steady leak does not depend on the mesh.
    assert env3 == pytest.approx(expected, rel=1e-3)
    assert env10 == pytest.approx(expected, rel=1e-3)
    assert env3 == pytest.approx(env10, rel=1e-3)


def test_transient_environment_leak_starts_below_steady():
    # Charge-up: with a degassed wall the OUTER (environment) flux starts near
    # zero and only the inner surface takes up hydrogen, so very early on the
    # environment leak is well below the steady value.
    sys = _WallRig(FIT, leak_model="transient", n_nodes=6)
    sys.instantiate()
    sys.initialise(n=1, tol=1e-9, max_iter=200)
    sys.solve_dae_step(60.0, tol=1e-9, max_iter=200,
                       raise_on_no_convergence=True)
    sys.next_step()
    env_early = -_val(sys, "wall.m_dot_b_leak")
    expected = analytic_steady_leak(FIT, P_OP)
    # The environment flux has barely begun (the diffusion front has not crossed
    # the wall yet), so its magnitude is a tiny fraction of the steady leak.
    # (It can even be slightly negative: with an empty interior the outer
    # surface-law boundary concentration marginally exceeds the first interior
    # node, so a negligible amount briefly diffuses inward.)
    assert abs(env_early) < 0.1 * expected


# -----------------------------------------------------------------------------
# 5. End to end: flow pipe pressurises and feeds the wall
# -----------------------------------------------------------------------------


class _PipeRig(Model):
    """PressureSource -> StraightPipe(leaky, 1 seg) -> ClosedEnd; pipe.leak -> wall."""

    def __init__(self, medium):
        self._medium = medium
        super().__init__()

    def declare_components(self):
        A = math.pi * R_IN ** 2
        self.add_component("source", PressureSource(
            self._medium, p_source=1e5, T_source=T_OP, A=A))
        self.add_component("pipe", StraightPipe(
            self._medium, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
            n_segments=1, leaky=True))
        self.add_component("cap", ClosedEnd(self._medium, p_init=1e5, T_init=T_OP))
        self.add_component("wall", _leaky_wall(
            SteadyRichardson(FIT), T=T_OP, p_in=1e5, p_out=P_EXT))
        self.add_component("env", FixedPartialPressure(p_partial=P_EXT))
        self.add_component("T_in", FixedTemperature(T_set=T_OP))
        self.add_component("T_out", FixedTemperature(T_set=T_OP))

    def declare_equations(self):
        self.connect(self["source"].ports["outlet"], self["pipe"].ports["inlet"])
        self.connect(self["pipe"].ports["outlet"], self["cap"].ports["inlet"])
        self.connect(self["pipe"].segment_leak_ports[0], self["wall"].ports["leak_a"])
        self.connect(self["env"].ports["leak"], self["wall"].ports["leak_b"])
        self.connect(self["wall"].ports["port_a"], self["T_in"].ports["heat"])
        self.connect(self["wall"].ports["port_b"], self["T_out"].ports["heat"])
        return []


def test_flow_pipe_pressurises_and_conserves_mass():
    med = CoolPropMedium("Hydrogen", backend="BICUBIC&HEOS", disable_warnings=True)
    sys = _PipeRig(med)
    sys.instantiate(aditional_modules=med.modules)
    sys.initialise(n=1, tol=1e-6, max_iter=200, line_search=True)

    # Quasi-static pressurisation: ramp the supply pressure 1 bar -> 20 MPa.
    # The source computes its own stagnation h/s from (p_source, T_source).
    for p_set in np.geomspace(1e5, P_OP, 40):
        sys["source"]["p_source"].set_value(float(p_set))
        sys.solve_dae_step(1e-3, tol=1e-6, max_iter=200, line_search=True,
                           raise_on_no_convergence=True)
        sys.next_step()

    p_pipe = _val(sys, "pipe.p_in")
    m_in = _val(sys, "pipe.m_dot_in")
    leak = _val(sys, "wall.m_dot_a_leak")      # inner uptake into the wall (> 0)
    expected = analytic_steady_leak(FIT, P_OP)

    # Pressurised (pressure went UP to ~20 MPa).
    assert p_pipe == pytest.approx(P_OP, rel=1e-3)
    # The wall leaks the analytic steady rate at 20 MPa ...
    assert leak == pytest.approx(expected, rel=1e-3)
    # ... and the capped pipe conserves mass: inlet make-up == wall leak.
    assert m_in == pytest.approx(leak, rel=1e-3)
    assert m_in > 0.0


# -----------------------------------------------------------------------------
# 6. Multi-segment leaky StraightPipe
# -----------------------------------------------------------------------------


class _MultiPipeRig(Model):
    """PressureSource -> StraightPipe(leaky, n_seg) -> ClosedEnd, one wall per segment."""

    def __init__(self, medium, n_seg):
        self._medium = medium
        self._n_seg = n_seg
        super().__init__()

    def declare_components(self):
        A = math.pi * R_IN ** 2
        L_seg = L_PIPE / self._n_seg
        self.add_component("source", PressureSource(
            self._medium, p_source=1e5, T_source=T_OP, A=A))
        self.add_component("pipe", StraightPipe(
            self._medium, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
            n_segments=self._n_seg, leaky=True))
        self.add_component("cap", ClosedEnd(self._medium, p_init=1e5, T_init=T_OP))
        for i in range(self._n_seg):
            self.add_component(f"wall_{i}", _leaky_wall(
                SteadyRichardson(FIT), length=L_seg, T=T_OP, p_in=1e5, p_out=P_EXT))
            self.add_component(f"env_{i}", FixedPartialPressure(p_partial=P_EXT))
            self.add_component(f"T_in_{i}", FixedTemperature(T_set=T_OP))
            self.add_component(f"T_out_{i}", FixedTemperature(T_set=T_OP))

    def declare_equations(self):
        self.connect(self["source"].ports["outlet"], self["pipe"].ports["inlet"])
        self.connect(self["pipe"].ports["outlet"], self["cap"].ports["inlet"])
        leak_ports = self["pipe"].segment_leak_ports
        for i in range(self._n_seg):
            self.connect(leak_ports[i], self[f"wall_{i}"].ports["leak_a"])
            self.connect(self[f"env_{i}"].ports["leak"],
                         self[f"wall_{i}"].ports["leak_b"])
            self.connect(self[f"wall_{i}"].ports["port_a"],
                         self[f"T_in_{i}"].ports["heat"])
            self.connect(self[f"wall_{i}"].ports["port_b"],
                         self[f"T_out_{i}"].ports["heat"])
        return []


def _pressurise_multi(n_seg):
    med = CoolPropMedium("Hydrogen", backend="BICUBIC&HEOS", disable_warnings=True)
    sys = _MultiPipeRig(med, n_seg)
    sys.instantiate(aditional_modules=med.modules)
    sys.initialise(n=1, tol=1e-6, max_iter=200, line_search=True)
    for p_set in np.geomspace(1e5, P_OP, 40):
        sys["source"]["p_source"].set_value(float(p_set))
        sys.solve_dae_step(1e-3, tol=1e-6, max_iter=200, line_search=True,
                           raise_on_no_convergence=True)
        sys.next_step()
    names = list(sys.record["vars_names"])
    st = np.asarray(sys.record["state"][-1])
    total_leak = sum(st[i] for i, n in enumerate(names)
                     if "wall_" in n and n.endswith(".m_dot_a_leak"))
    return total_leak


def test_leaky_pipe_validation():
    med = CoolPropMedium("Hydrogen", disable_warnings=True)
    with pytest.raises(ValueError):
        StraightPipe(med, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
                     n_segments=0, leaky=True)
    with pytest.raises(ValueError):
        StraightPipe(med, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
                     multiphase="bogus", leaky=True)


def test_leaky_pipe_exposes_one_leak_port_per_segment():
    med = CoolPropMedium("Hydrogen", disable_warnings=True)
    pipe = StraightPipe(med, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
                        n_segments=5, leaky=True)
    ports = pipe.segment_leak_ports
    assert len(ports) == 5
    assert all(p.kind == "permeation_pN" for p in ports)


def test_leaky_pipe_total_leak_is_segment_count_independent():
    # The whole-pipe leak (sum over the per-segment walls) equals the analytic
    # single-tube Richardson rate regardless of how many segments the pipe is
    # split into.
    expected = analytic_steady_leak(FIT, P_OP)   # full L
    leak1 = _pressurise_multi(n_seg=1)
    leak4 = _pressurise_multi(n_seg=4)
    assert leak1 == pytest.approx(expected, rel=1e-3)
    assert leak4 == pytest.approx(expected, rel=1e-3)
    assert leak4 == pytest.approx(leak1, rel=1e-3)


# -----------------------------------------------------------------------------
# 7. Convenience `Pipe` assembly (boundary - pipe - boundary)
# -----------------------------------------------------------------------------


class _WalledPipeRig(Model):
    """PressureSource -> Pipe(leaky AISI-316 layer, vented) -> ClosedEnd.

    The walled pipe is one component: the per-segment walls, outer thermal
    boundary, and partial-pressure vent are all internal.
    """

    def __init__(self, medium, n_seg):
        self._medium = medium
        self._n_seg = n_seg
        super().__init__()

    def declare_components(self):
        A = math.pi * R_IN ** 2
        self.add_component("source", PressureSource(
            self._medium, p_source=1e5, T_source=T_OP, A=A))
        self.add_component("pipe", Pipe(
            self._medium, D=2 * R_IN, L=L_PIPE, epsilon=1e-6, z_in=0.0, z_out=0.0,
            n_segments=self._n_seg,
            layers=[WallLayer(MAT, R_OUT - R_IN, permeation=SteadyRichardson(FIT),
                              dynamic=False)],
            outer_thermal="fixed", T_outer=T_OP, p_ext=P_EXT,
            T_wall_init=T_OP, p_init=1e5))
        self.add_component("cap", ClosedEnd(self._medium, p_init=1e5, T_init=T_OP))

    def declare_equations(self):
        self.connect(self["source"].ports["outlet"], self["pipe"].ports["inlet"])
        self.connect(self["pipe"].ports["outlet"], self["cap"].ports["inlet"])
        return []


def test_walled_pipe_leaks_analytic_rate_and_conserves_mass():
    med = CoolPropMedium("Hydrogen", backend="BICUBIC&HEOS", disable_warnings=True)
    sys = _WalledPipeRig(med, n_seg=2)
    sys.instantiate(aditional_modules=med.modules)
    sys.initialise(n=1, tol=1e-6, max_iter=200, line_search=True)
    for p_set in np.geomspace(1e5, P_OP, 40):
        sys["source"]["p_source"].set_value(float(p_set))
        sys.solve_dae_step(1e-3, tol=1e-6, max_iter=200, line_search=True,
                           raise_on_no_convergence=True)
        sys.next_step()

    names = list(sys.record["vars_names"])
    st = np.asarray(sys.record["state"][-1])
    total_leak = sum(st[i] for i, n in enumerate(names)
                     if "wall_" in n and n.endswith(".m_dot_a_leak"))
    expected = analytic_steady_leak(FIT, P_OP)
    assert total_leak == pytest.approx(expected, rel=1e-3)
