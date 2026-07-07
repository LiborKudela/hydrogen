"""Tests for the advective pipe level (`SegmentedChannel(dynamic='advective')`
behind the `Pipe` assembly), exercising it for a *gas* and for the *multiphase*
(HEM) property mode -- not just the (near-incompressible) liquid-water transients
covered by the ULg advection benchmark.

What is checked:

  * GAS (Air): a genuinely compressible gas thermal transient runs on the
    ``compressible`` level (per-cell mass storage + free cell pressure).  The
    inlet-temperature step is transported to the outlet at the flow speed, the
    outlet mass flow transiently exceeds the (fixed) inlet flow as the hot gas
    expands (mass storage -- the ``advective`` quasi-steady-mass level cannot
    represent this and diverges for a gas), and a constant-inlet gas pipe holds
    its steady state.  A narrow bore / decent velocity is used so the friction
    pressure drop is non-negligible (the quasi-steady compressible momentum is
    only well-conditioned there).

  * MULTIPHASE (HEM) advection: the advective level built with
    ``multiphase='HEM'`` marches a subcooled-liquid temperature transient with
    no breakdown and transports the front -- i.e. the cell-centred energy DOF,
    the reconstructed face enthalpies and the general axial-dispersion closure
    all work through the HEM mixture property functions.

  * The general axial-dispersion closure is the advective default and is
    media-agnostic (finite, positive `D_eff` for gas and HEM alike), and a
    too-coarse grid raises the cell-Peclet warning.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hydrogen import CoolPropMedium, Model
from hydrogen.components.control.control_components import Step
from hydrogen.components.thermofluid.assemblies import Pipe
from hydrogen.components.thermofluid.flow import (
    PressureOutlet,
    SegmentedChannel,
    TemperatureInlet,
    _NumLib,
)


# --- helpers ----------------------------------------------------------------
def _gas_medium():
    return CoolPropMedium("Air", disable_warnings=True, backend="BICUBIC&HEOS",
                          scalar_cache_maxsize=2000)


def _water_medium():
    return CoolPropMedium("Water", disable_warnings=True, backend="BICUBIC&HEOS",
                          scalar_cache_maxsize=2000)


def _build_step_pipe(medium, *, p, T0, dT, t_step, m_flow, D, L, N,
                     dispersion="general", multiphase="single",
                     dynamic="advective"):
    """`Step(temperature) -> TemperatureInlet -> adiabatic pipe ->
    PressureOutlet`.  A bare pipe (no wall stack) so the response is pure
    advection + axial dispersion."""

    class StepPipe(Model):
        def declare_components(self):
            self.add_component("tin", Step(
                offset=T0, height=dT, start_time=t_step, unit="K"))
            self.add_component("inlet", TemperatureInlet(
                medium, m_flow=m_flow, p_init=p, T_init=T0))
            self.add_component("pipe", Pipe(
                medium, D=D, L=L, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=N, layers=[], outer_thermal="adiabatic",
                dynamic=dynamic, dispersion=dispersion,
                multiphase=multiphase, T_wall_init=T0, p_init=p))
            self.add_component("outlet", PressureOutlet(
                medium, p_ambient=p, T_ambient=T0))

        def declare_equations(self):
            self.connect(self["tin"].ports["y"], self["inlet"].ports["T_set"])
            self.connect(self["inlet"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["outlet"].ports["inlet"])
            return []

    return StepPipe()


def _seed_uniform(sys, medium, p0, T0, m_flow=None):
    """Seed the primitive (p, h) channel to a uniform (p0, T0) initial state
    (and, when `m_flow` is given, a consistent uniform flow field -- important
    for the stiff-liquid acoustic level, where a velocity seed far from the
    running point makes the cold-start Newton wander through huge pressures)."""
    ch = sys["pipe"]["pipe"]
    N = ch.N
    h0 = float(medium.eval_h_pT(p0, T0))
    rho0 = float(medium.eval_rho_ph(p0, h0))
    k0 = float(medium.eval_k_ph(p0, h0))
    w0 = None if m_flow is None else m_flow / (rho0 * float(ch.D) ** 2
                                               * math.pi / 4.0)
    for i in range(N):
        ch[f"hc_{i}"].value = h0
        ch[f"Tc_{i}"].value = T0
        ch[f"pc_{i}"].value = p0
        ch[f"rhoc_{i}"].value = rho0
        ch[f"kc_{i}"].value = k0
    for j in range(N + 1):
        seeds = [("h", h0), ("p", p0), ("T", T0), ("rho", rho0)]
        if m_flow is not None:
            seeds += [("w", w0), ("M", m_flow)]
        for stem, val in seeds:
            key = f"{stem}_{j}"
            if key in ch.components:
                ch[key].value = val


def _run(sys, medium, p0, T0, *, stop, dt_max, m_flow=None):
    sys.instantiate(aditional_modules=medium.modules, cse=True, enable_blt=True,
                    enable_var_scaling=True, max_remove_trival_passes=1,
                    max_remove_duplicate_passes=5,
                    max_remove_linear_block_passes=3)
    _seed_uniform(sys, medium, p0, T0, m_flow=m_flow)
    # These are step-response cases (steady flow until t_step > 0), so relax
    # to the true steady state: leftover seeding imbalance in the derivative
    # companions would otherwise poison the first adaptive steps (critical for
    # the near-incompressible water cases, where 1/rho_p amplifies it).
    sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300,
                   line_search=(sys["pipe"].dynamic == "acoustic"),
                   steady=True)
    return sys.run(
        stop_time=stop,
        strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 0.5},
        dt_start=1e-3, dt_min=1e-9, dt_max=dt_max, grow=1.5, shrink=0.5,
        max_retries=40, relaxation=1.0, tol=1e-8, max_iter=300,
        raise_on_no_convergence=False)


def _assert_transported_front(t, T, T0, dT, t_step, residence):
    """Assert a temperature step is *transported* to the outlet: starts at T0,
    ends at T0+dT, rises monotonically (up to numerical wiggle), and its
    half-rise arrives with a genuine transport delay -- not instantly (which
    would mean pure diffusion) and within a few residence times.  This is robust
    to the dispersive smearing that a coarse grid puts on the front's toe."""
    t = np.asarray(t)
    T = np.asarray(T)
    assert T[0] == pytest.approx(T0, abs=0.5)
    assert T[-1] == pytest.approx(T0 + dT, abs=max(0.05 * abs(dT), 1.0))
    # Bounded over/undershoot around the [T0, T0+dT] band.  A central advection
    # scheme on a coarse (high cell-Peclet) grid shows a modest Gibbs-like
    # overshoot at the sharp front; allow ~15% of the step.
    over = max(0.15 * abs(dT), 1.5)
    lo, hi = min(T0, T0 + dT), max(T0, T0 + dT)
    assert T.min() > lo - over
    assert T.max() < hi + over
    # Half-rise crossing time: interpolate where T first reaches the midpoint.
    mid = T0 + 0.5 * dT
    above = np.where(T >= mid)[0] if dT > 0 else np.where(T <= mid)[0]
    assert above.size > 0, "front never reached the half-rise level"
    k = above[0]
    assert k > 0, "front reached the outlet instantaneously (no transport delay)"
    t_half = t[k]
    assert t_half >= t_step + 0.25 * residence, (
        f"front arrived too early (t_half={t_half:.2f}, "
        f"expected >= {t_step + 0.25 * residence:.2f})")
    assert t_half <= t_step + 3.0 * residence, (
        f"front arrived too late (t_half={t_half:.2f})")


def _outlet_T(sys, medium, N):
    t = np.asarray(sys.record["time"])
    h = np.asarray(sys.series(f"pipe.pipe.h_{N}"))
    p = np.asarray(sys.series(f"pipe.pipe.p_{N}"))
    T = np.array([medium.eval_T_ph(float(p[k]), float(h[k]))
                  for k in range(len(t))])
    return t, T


# --- gas advection (compressible level) -------------------------------------
# A genuinely compressible gas transient needs the `compressible` level (mass
# storage): the `advective` level's quasi-steady mass is inconsistent when the
# density changes in time.  The `compressible` momentum is quasi-steady (no
# acoustics), so it is well-conditioned only when the friction pressure drop is
# non-negligible -- Air at ~2 bar through a narrow bore at a decent velocity.
GAS_P = 2.0e5
GAS_T0 = 300.0
GAS_DT = 15.0
GAS_TSTEP = 1.0
GAS_D = 0.004
GAS_L = 5.0
GAS_N = 6
GAS_W = 10.0


def _gas_m_flow(med):
    h0 = med.eval_h_pT(GAS_P, GAS_T0)
    rho0 = med.eval_rho_ph(GAS_P, h0)
    return rho0 * (math.pi * GAS_D ** 2 / 4.0) * GAS_W


@pytest.fixture(scope="module")
def gas_compressible_run():
    med = _gas_medium()
    residence = GAS_L / GAS_W
    sys = _build_step_pipe(med, p=GAS_P, T0=GAS_T0, dT=GAS_DT, t_step=GAS_TSTEP,
                           m_flow=_gas_m_flow(med), D=GAS_D, L=GAS_L, N=GAS_N,
                           dynamic="compressible")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = _run(sys, med, GAS_P, GAS_T0,
                       stop=GAS_TSTEP + 4.0 * residence, dt_max=residence / 10.0,
                       m_flow=_gas_m_flow(med))
    t, T = _outlet_T(sys, med, GAS_N)
    return {"sys": sys, "med": med, "t": t, "T": T,
            "residence": residence, "summary": summary}


def test_gas_compressible_completes(gas_compressible_run):
    """The compressible gas transient integrates cleanly to the stop time --
    the quasi-steady-mass `advective` level cannot (its Newton solve diverges at
    the front for a strongly compressible medium)."""
    assert gas_compressible_run["summary"]["stop_reason"] == "stop_time", (
        f"gas compressible solve broke down: "
        f"{gas_compressible_run['summary']['stop_reason']}")
    assert gas_compressible_run["summary"]["steps"] > 10


def test_gas_compressible_transports_front(gas_compressible_run):
    """The inlet-temperature step is transported to the outlet at the flow
    speed: delayed ~one residence time, then reaching the new inlet temp."""
    _assert_transported_front(
        gas_compressible_run["t"], gas_compressible_run["T"],
        GAS_T0, GAS_DT, GAS_TSTEP, gas_compressible_run["residence"])


def test_gas_compressible_mass_flow_goes_nonuniform(gas_compressible_run):
    """The signature of true compressibility: as the (lighter) hot gas sweeps
    through, mass is stored/released so the outlet mass flow transiently EXCEEDS
    the (fixed) inlet mass flow -- the flow field is non-uniform.  At both steady
    ends the pipe conserves mass exactly (M uniform)."""
    sys = gas_compressible_run["sys"]
    N = GAS_N
    M0 = np.asarray(sys.series("pipe.pipe.M_0"))
    MN = np.asarray(sys.series(f"pipe.pipe.M_{N}"))

    # Inlet flow is the imposed constant.
    assert M0.std() / abs(M0.mean()) < 1e-6

    # During the transient the outlet flow rises above the inlet flow (gas
    # expands as it heats) -- something a uniform-M model cannot show.
    assert (MN / M0).max() > 1.02, (
        f"outlet mass flow never exceeded inlet (max ratio "
        f"{(MN / M0).max():.4f}); expected transient gas expansion")

    # ... and mass balance is restored once the front has passed (T uniform
    # again at the new level -> uniform M).
    assert MN[-1] == pytest.approx(M0[-1], rel=1e-3)


def test_gas_compressible_steady_hold_is_stable():
    """With a constant inlet the compressible gas pipe holds its steady state
    for many residence times.  (The `advective` level, lacking mass storage, is
    inconsistent for a gas and diverges even from this steady point.)"""
    med = _gas_medium()
    residence = GAS_L / GAS_W
    sys = _build_step_pipe(med, p=GAS_P, T0=GAS_T0, dT=0.0, t_step=1e9,
                           m_flow=_gas_m_flow(med), D=GAS_D, L=GAS_L, N=GAS_N,
                           dynamic="compressible")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = _run(sys, med, GAS_P, GAS_T0,
                       stop=20.0 * residence, dt_max=residence,
                       m_flow=_gas_m_flow(med))
    assert summary["stop_reason"] == "stop_time"
    t, T = _outlet_T(sys, med, GAS_N)
    # Past the brief start-up transient (flow establishing from the seed) the
    # outlet holds the inlet temperature steadily.
    settled = T[t >= 2.0 * residence]
    assert settled.size > 0
    assert np.max(np.abs(settled - GAS_T0)) < 0.5


# --- gas acoustic (momentum inertia) ----------------------------------------
# The `acoustic` level = compressible mass/energy storage + a transient momentum
# balance on the interior faces (adds the d(rho*w)/dt inertia the other levels
# drop).  It is the AC-free ("exact") level; here it transports a gas thermal
# front just like the compressible level, additionally resolving the momentum
# dynamics.
@pytest.fixture(scope="module")
def gas_acoustic_run():
    med = _gas_medium()
    residence = GAS_L / GAS_W
    sys = _build_step_pipe(med, p=GAS_P, T0=GAS_T0, dT=GAS_DT, t_step=GAS_TSTEP,
                           m_flow=_gas_m_flow(med), D=GAS_D, L=GAS_L, N=GAS_N,
                           dynamic="acoustic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = _run(sys, med, GAS_P, GAS_T0,
                       stop=GAS_TSTEP + 4.0 * residence, dt_max=residence / 10.0,
                       m_flow=_gas_m_flow(med))
    t, T = _outlet_T(sys, med, GAS_N)
    return {"sys": sys, "med": med, "t": t, "T": T,
            "residence": residence, "summary": summary}


def test_acoustic_gas_completes(gas_acoustic_run):
    """The acoustic gas transient (mass storage + interior-face momentum
    inertia) integrates cleanly to the stop time."""
    assert gas_acoustic_run["summary"]["stop_reason"] == "stop_time", (
        f"acoustic gas solve broke down: "
        f"{gas_acoustic_run['summary']['stop_reason']}")
    assert gas_acoustic_run["summary"]["steps"] > 10


def test_acoustic_gas_transports_front(gas_acoustic_run):
    """The inlet-temperature step is transported to the outlet at the flow
    speed on the acoustic level too."""
    _assert_transported_front(
        gas_acoustic_run["t"], gas_acoustic_run["T"],
        GAS_T0, GAS_DT, GAS_TSTEP, gas_acoustic_run["residence"])


def test_acoustic_level_structure():
    """The acoustic level carries per-cell (pc, hc) storage states and a
    momentum (velocity) state on EVERY face, end faces included -- that is what
    lets pressure waves reflect correctly off the port boundary conditions."""
    med = _gas_medium()
    ch = SegmentedChannel(med, D=GAS_D, L=GAS_L, epsilon=1e-6, z_in=0.0,
                          z_out=0.0, N=4, dynamic="acoustic")
    assert ch._momentum_inertia and ch._mass_storage and ch._cell_centered
    # Per-cell primitive storage states; the old conserved (U, m) states and
    # the old interior-face-only `pi_j` momentum states are gone.
    assert all(f"der_hc_{i}" in ch.components for i in range(4))
    assert all(f"der_pc_{i}" in ch.components for i in range(4))
    assert all(f"der_w_{j}" in ch.components for j in range(5))
    assert not any(k.startswith("pi_") or k.startswith("U_")
                   or (k.startswith("m_") and k[2:].isdigit())
                   for k in ch.components)


def test_compressible_level_structure():
    """The compressible level is the low-Mach split: ONE pipe-level pressure
    state `p_pipe` carries the mass-storage dynamics, the per-cell pressures
    stay algebraic (quasi-steady momentum profile), and no face has a momentum
    state."""
    med = _gas_medium()
    ch = SegmentedChannel(med, D=GAS_D, L=GAS_L, epsilon=1e-6, z_in=0.0,
                          z_out=0.0, N=4, dynamic="compressible")
    assert ch._pressure_split and ch._mass_storage
    assert not ch._momentum_inertia
    assert "p_pipe" in ch.components and "der_p_pipe" in ch.components
    assert all(f"pc_{i}" in ch.components for i in range(4))
    assert not any(f"der_pc_{i}" in ch.components for i in range(4))
    assert not any(f"der_w_{j}" in ch.components for j in range(5))


# --- near-incompressible subcooled liquid on the mass-storage levels --------
# The compressible level's low-Mach split (single anchored p_pipe state) and
# the acoustic level's scaled per-cell (p, h) + all-face momentum scheme both
# stay well-conditioned for a *near-incompressible subcooled liquid* -- the
# regime where the old conserved-mass (m = V*rho) formulation was singular
# (dp/dm ~ 1/(V*drho/dp) blows up) -- with no artificial compressibility.
LIQ_P = 5.0e5
LIQ_T0 = 320.0                     # Tsat(5 bar) ~ 425 K: comfortably subcooled
LIQ_DT = 20.0
LIQ_TSTEP = 1.0
LIQ_D = 0.01
LIQ_L = 1.0
LIQ_N = 4
LIQ_W = 0.2


def _liq_m_flow(med):
    h0 = med.eval_h_pT(LIQ_P, LIQ_T0)
    rho0 = med.eval_rho_ph(LIQ_P, h0)
    return rho0 * (math.pi * LIQ_D ** 2 / 4.0) * LIQ_W


@pytest.mark.parametrize("dynamic", ["compressible", "acoustic"])
def test_subcooled_liquid_mass_storage_level_converges(dynamic):
    """A subcooled-liquid temperature step marches cleanly on both mass-storage
    levels (compressible + acoustic) with the primitive (p, h) formulation --
    the near-incompressible case that the old conserved-mass scheme could not
    integrate -- and the front is transported to the outlet."""
    med = _water_medium()
    residence = LIQ_L / LIQ_W
    sys = _build_step_pipe(med, p=LIQ_P, T0=LIQ_T0, dT=LIQ_DT, t_step=LIQ_TSTEP,
                           m_flow=_liq_m_flow(med), D=LIQ_D, L=LIQ_L, N=LIQ_N,
                           multiphase="HEM", dynamic=dynamic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = _run(sys, med, LIQ_P, LIQ_T0,
                       stop=LIQ_TSTEP + 5.0 * residence, dt_max=residence / 8.0,
                       m_flow=_liq_m_flow(med))
    assert summary["stop_reason"] == "stop_time", (
        f"{dynamic} subcooled-liquid solve broke down: "
        f"{summary['stop_reason']} ({summary.get('error')})")
    assert summary["steps"] > 10
    t, T = _outlet_T(sys, med, LIQ_N)
    _assert_transported_front(t, T, LIQ_T0, LIQ_DT, LIQ_TSTEP, residence)


# --- multiphase (HEM) advection --------------------------------------------
def test_hem_advective_pipe_forwards_mode():
    """`Pipe(multiphase='HEM', dynamic='advective')` forwards the mode to its
    SegmentedChannel."""
    med = _water_medium()
    pipe = Pipe(med, D=0.01, L=1.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=4, layers=[], outer_thermal="adiabatic",
                dynamic="advective", multiphase="HEM")
    assert pipe.multiphase == "HEM"
    assert pipe["pipe"].multiphase == "HEM"


def test_hem_advection_transports_subcooled_front():
    """The advective level built with the HEM mixture property functions marches
    a subcooled-liquid temperature step and transports the front to the outlet
    with no breakdown -- proving the cell-centred energy DOF, face-enthalpy
    reconstruction and general dispersion are all media-agnostic."""
    med = _water_medium()
    p, T0, dT, t_step = 5.0e5, 300.0, 30.0, 1.0   # Tsat(5 bar) ~ 425 K: subcooled
    D, L, N, w = 0.01, 1.0, 5, 0.2
    h0 = med.eval_h_pT(p, T0)
    rho0 = med.eval_rho_ph(p, h0)
    m_flow = rho0 * (math.pi * D ** 2 / 4.0) * w
    residence = L / w

    sys = _build_step_pipe(med, p=p, T0=T0, dT=dT, t_step=t_step, m_flow=m_flow,
                           D=D, L=L, N=N, multiphase="HEM")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = _run(sys, med, p, T0, stop=t_step + 5.0 * residence,
                       dt_max=residence / 8.0, m_flow=m_flow)

    assert summary["stop_reason"] == "stop_time", (
        f"HEM advective solve broke down: {summary['stop_reason']} "
        f"({summary.get('error')})")

    t, T = _outlet_T(sys, med, N)
    _assert_transported_front(t, T, T0, dT, t_step, residence)


# --- general dispersion: default + media-agnostic + Peclet warning ----------
def test_general_dispersion_is_advective_default():
    """The advective level defaults to the general (regime-blended) dispersion
    closure for both `Pipe` and the raw `SegmentedChannel`."""
    med = _water_medium()
    pipe = Pipe(med, D=0.01, L=1.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=3, layers=[], outer_thermal="adiabatic",
                dynamic="advective")
    assert pipe.dispersion == "general"
    ch = pipe["pipe"]
    assert ch.dispersion_func == ch.get_general_dispersion


@pytest.mark.parametrize("medium_factory", [_gas_medium, _water_medium])
def test_general_dispersion_is_finite_and_positive(medium_factory):
    """`get_general_dispersion` returns a finite, positive effective diffusivity
    across laminar / transitional / turbulent velocities for any medium (built
    purely from the local alpha = k/(rho*cp) and nu = mu/rho)."""
    med = medium_factory()
    ch = SegmentedChannel(med, D=0.02, L=1.0, epsilon=1e-5, z_in=0.0, z_out=0.0,
                          N=4, dynamic="advective", heat_port=True)
    cp = ch._cp_std
    alpha = ch._k_std / (ch._rho_std * cp)
    nu = ch._mu_std / ch._rho_std
    prev_Re = -1.0
    for w in (1e-3, 1e-2, 0.1, 1.0, 10.0):
        D_eff = ch._general_dispersion(w, ch.D, alpha, nu, _NumLib)
        assert np.isfinite(D_eff) and D_eff > 0.0
        # Never below the molecular floor (both branches include `alpha`).
        assert D_eff >= alpha - 1e-12
        Re = abs(w) * ch.D / nu
        assert Re > prev_Re
        prev_Re = Re


def test_coarse_grid_raises_cell_peclet_warning():
    """A deliberately coarse advective grid trips the cell-Peclet warning so the
    user learns the discretisation is too coarse for non-oscillatory transport.
    (The build-time estimate fires while the channel is constructed.)"""
    med = _water_medium()
    with pytest.warns(UserWarning, match="[Pp]eclet"):
        _build_step_pipe(med, p=2.0e5, T0=300.0, dT=20.0, t_step=1.0,
                         m_flow=0.02, D=0.01, L=2.0, N=3)
