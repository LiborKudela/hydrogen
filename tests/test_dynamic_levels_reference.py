"""Reference-solution tests for the three dynamic pipe levels.

Each dynamic level of `SegmentedChannel` is validated against an analytic (or
exactly-known) reference:

  * ``advective``    -- outlet response of a constant-`D` advection-dispersion
    pipe vs the closed-form Ogata & Banks (1961) solution.
  * ``compressible`` -- adiabatic charging of a dead-ended pipe vs the exact
    ideal-gas pressurisation rate ``dP/dt = gamma*R*T_in*m_dot/V`` (and exact
    mass inventory ``m(t) = m0 + m_dot*t``).
  * ``acoustic``     -- sudden valve closure (water hammer) vs the Joukowsky
    amplitude ``dp = rho*a*w0`` and the ``4L/a`` oscillation period; with
    `wall_elasticity` the wave speed drops to the classic Korteweg
    elastic-line value and the peak scales down with it.
  * ``acoustic`` + ``cavitation`` -- column separation on a low-pressure
    line: the DVCM cavity clamps the rarefaction at the vapor pressure, the
    first cavity phase matches the rigid-column lifetime estimate
    ``2*rho*L*w0/(p_res - p_vap)`` and the collapse re-emits a
    Joukowsky-order shock.

Plus structural smoke checks for the testing-grade wall flags
(`unsteady_friction`, `viscoelastic_wall`) and the `fsi` placeholder.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc, erfcx

from hydrogen import CoolPropMedium, Model
from hydrogen.components.control.control_components import SmoothRamp, Step
from hydrogen.components.thermofluid.assemblies import Pipe
from hydrogen.components.thermofluid.flow import (
    ClosedEnd,
    IncompressibleValve,
    PressureOutlet,
    PressureSource,
    SegmentedChannel,
    TemperatureInlet,
)


def _water():
    return CoolPropMedium("Water", disable_warnings=True,
                          backend="BICUBIC&HEOS", scalar_cache_maxsize=2000)


def _air():
    return CoolPropMedium("Air", disable_warnings=True,
                          backend="BICUBIC&HEOS", scalar_cache_maxsize=2000)


def _seed_channel(ch, medium, p0, T0, w0=0.0):
    """Uniform (p0, T0, w0) initial state on a SegmentedChannel."""
    N = ch.N
    h0 = float(medium.eval_h_pT(p0, T0))
    rho0 = float(medium.eval_rho_ph(p0, h0))
    k0 = float(medium.eval_k_ph(p0, h0))
    A = math.pi * float(ch.D) ** 2 / 4.0
    for i in range(N):
        ch[f"hc_{i}"].value = h0
        ch[f"Tc_{i}"].value = T0
        ch[f"pc_{i}"].value = p0
        ch[f"rhoc_{i}"].value = rho0
        ch[f"kc_{i}"].value = k0
    for j in range(N + 1):
        for stem, val in (("h", h0), ("p", p0), ("T", T0), ("rho", rho0),
                          ("w", w0), ("M", rho0 * A * w0)):
            key = f"{stem}_{j}"
            if key in ch.components:
                ch[key].value = val


def _instantiate(sys, medium):
    sys.instantiate(aditional_modules=medium.modules, cse=True,
                    enable_blt=True, enable_var_scaling=True,
                    max_remove_trival_passes=1, max_remove_duplicate_passes=5,
                    max_remove_linear_block_passes=3)


def _run(sys, *, stop, dt_max, dt_start=1e-3, tol_local=1e-3):
    return sys.run(
        stop_time=stop,
        strategy={"name": "tr_bdf2", "tol_local": tol_local, "atol": 0.5},
        dt_start=dt_start, dt_min=1e-9, dt_max=dt_max, grow=1.5, shrink=0.5,
        max_retries=40, relaxation=1.0, tol=1e-8, max_iter=300,
        raise_on_no_convergence=False)


# ===========================================================================
# 1. advective vs Ogata-Banks (analytic advection-dispersion)
# ===========================================================================
def _ogata_banks(x, t, w, D, T0, dT):
    """Closed-form 1-D advection-dispersion step response (Ogata & Banks
    1961) on a semi-infinite domain, in the erfcx-stabilised form."""
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, T0)
    m = t > 0.0
    tt = t[m]
    s = 2.0 * np.sqrt(D * tt)
    a = (x - w * tt) / s
    b = (x + w * tt) / s
    out[m] = T0 + 0.5 * dT * (erfc(a) + np.exp(-a * a) * erfcx(b))
    return out


def test_advective_matches_ogata_banks():
    """The advective level with an imposed constant axial diffusivity
    reproduces the analytic Ogata-Banks outlet temperature response."""
    med = _water()
    P0, T0, dT, t_step = 2.0e5, 293.15, 10.0, 5.0
    D_bore, L, N, w, D_ax = 0.05, 5.0, 20, 0.2, 0.05   # Pe_cell = 1
    h0 = med.eval_h_pT(P0, T0)
    rho0 = med.eval_rho_ph(P0, h0)
    m_flow = rho0 * (math.pi * D_bore ** 2 / 4.0) * w

    class DispPipe(Model):
        def declare_components(self):
            self.add_component("tin", Step(offset=T0, height=dT,
                                           start_time=t_step, unit="K"))
            self.add_component("inlet", TemperatureInlet(
                med, m_flow=m_flow, p_init=P0, T_init=T0))
            self.add_component("pipe", Pipe(
                med, D=D_bore, L=L, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=N, layers=[], outer_thermal="adiabatic",
                dynamic="advective", dispersion="constant", D_axial=D_ax,
                T_wall_init=T0, p_init=P0))
            self.add_component("outlet", PressureOutlet(
                med, p_ambient=P0, T_ambient=T0))

        def declare_equations(self):
            self.connect(self["tin"].ports["y"], self["inlet"].ports["T_set"])
            self.connect(self["inlet"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["outlet"].ports["inlet"])
            return []

    sys = DispPipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _instantiate(sys, med)
        _seed_channel(sys["pipe"]["pipe"], med, P0, T0, w0=w)
        sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300)
        summary = _run(sys, stop=70.0, dt_max=1.0, dt_start=0.02,
                       tol_local=5e-4)
    assert summary["stop_reason"] == "stop_time"

    t = np.asarray(sys.record["time"])
    hN = np.asarray(sys.series(f"pipe.pipe.h_{N}"))
    pN = np.asarray(sys.series(f"pipe.pipe.p_{N}"))
    T_model = np.array([med.eval_T_ph(float(pN[k]), float(hN[k]))
                        for k in range(len(t))])
    T_ana = _ogata_banks(L, t - t_step, w, D_ax, T0, dT)
    rmse = float(np.sqrt(np.mean((T_model - T_ana) ** 2)))
    assert rmse < 0.35, f"advective vs Ogata-Banks RMSE {rmse:.3f} K"
    # The front actually arrived (response spans most of the step).
    assert T_model[-1] > T0 + 0.8 * dT


# ===========================================================================
# 2. compressible: adiabatic charging of a dead-ended pipe (exact ideal gas)
# ===========================================================================
def test_compressible_closed_pipe_charging_rate():
    """Charging a dead-ended air line: the pipe-level pressure rises at the
    exact adiabatic ideal-gas rate ``dP/dt = gamma*R*T_in*m_dot/V`` and the
    mass inventory tracks ``m0 + m_dot*t`` exactly."""
    med = _air()
    P0, T0 = 2.0e5, 300.0
    D_bore, L, N = 0.05, 2.0, 4
    V = math.pi * D_bore ** 2 / 4.0 * L
    m_dot = 2.0e-4
    R_air, gamma = 287.05, 1.4
    slope_exact = gamma * R_air * T0 * m_dot / V     # ~6.2 kPa/s

    class Charge(Model):
        def declare_components(self):
            self.add_component("tin", Step(offset=T0, height=0.0,
                                           start_time=1e9, unit="K"))
            self.add_component("inlet", TemperatureInlet(
                med, m_flow=m_dot, p_init=P0, T_init=T0))
            self.add_component("pipe", Pipe(
                med, D=D_bore, L=L, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=N, layers=[], outer_thermal="adiabatic",
                dynamic="compressible", T_wall_init=T0, p_init=P0))
            self.add_component("cap", ClosedEnd(med, p_init=P0, T_init=T0))

        def declare_equations(self):
            self.connect(self["tin"].ports["y"], self["inlet"].ports["T_set"])
            self.connect(self["inlet"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["cap"].ports["inlet"])
            return []

    sys = Charge()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _instantiate(sys, med)
        _seed_channel(sys["pipe"]["pipe"], med, P0, T0, w0=0.0)
        sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300)
        summary = _run(sys, stop=2.0, dt_max=0.05)
    assert summary["stop_reason"] == "stop_time"

    t = np.asarray(sys.record["time"])
    P = np.asarray(sys.series("pipe.pipe.p_pipe"))
    # Fit the pressurisation slope on an early window (T drifts up slowly, so
    # the exact-rate comparison is an initial-slope statement).
    m = (t >= 0.2) & (t <= 1.2)
    slope = float(np.polyfit(t[m], P[m], 1)[0])
    assert slope == pytest.approx(slope_exact, rel=0.10), (
        f"dP/dt = {slope:.0f} Pa/s vs gamma*R*T*m_dot/V = {slope_exact:.0f}")

    # Exact mass inventory: V/N * sum(rhoc_i) grows by m_dot * t.
    rho_cells = np.stack([np.asarray(sys.series(f"pipe.pipe.rhoc_{i}"))
                          for i in range(N)])
    m_tot = rho_cells.mean(axis=0) * V
    m_expected = m_tot[0] + m_dot * t
    err = np.max(np.abs(m_tot - m_expected)) / (m_dot * t[-1])
    assert err < 0.02, f"mass inventory error {err:.3%} of the charged mass"


# ===========================================================================
# 3. acoustic: water hammer (Joukowsky amplitude, 4L/a period, Korteweg wall)
# ===========================================================================
P_SRC, P_OUT, T_HAM = 10.5e5, 10.0e5, 293.15
D_HAM, L_HAM, N_HAM = 0.05, 50.0, 10
T_CLOSE, DUR_CLOSE = 0.2, 0.02
E_WALL, e_WALL = 3.0e9, 0.005


def _hammer_system(med, *, elastic):
    class Hammer(Model):
        def declare_components(self):
            self.add_component("src", PressureSource(
                med, p_source=P_SRC, T_source=T_HAM,
                A=math.pi * D_HAM ** 2 / 4.0))
            self.add_component("pipe", Pipe(
                med, D=D_HAM, L=L_HAM, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=N_HAM, layers=[], outer_thermal="adiabatic",
                dynamic="acoustic", T_wall_init=T_HAM, p_init=P_OUT,
                wall_elasticity=elastic, wall_E=E_WALL, wall_e=e_WALL))
            self.add_component("valve", IncompressibleValve(
                med, Kv=5.0, D=D_HAM, opening=1.0))
            self.add_component("cmd", SmoothRamp(
                offset=1.0, height=-1.0, duration=DUR_CLOSE,
                start_time=T_CLOSE, corner=0.25))
            self.add_component("out", PressureOutlet(
                med, p_ambient=P_OUT, T_ambient=T_HAM))

        def declare_equations(self):
            self.connect(self["src"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["valve"].ports["inlet"])
            self.connect(self["valve"].ports["outlet"],
                         self["out"].ports["inlet"])
            self.connect(self["cmd"].ports["y"],
                         self["valve"].ports["opening"])
            return []

    return Hammer()


def _wave_speed(med, elastic):
    """Model-consistent wave speed 1/sqrt(rho_p_eff + rho_h/rho) at the
    hammer operating point (Korteweg-corrected when elastic)."""
    h0 = med.eval_h_pT(P_OUT, T_HAM)
    rho0 = med.eval_rho_ph(P_OUT, h0)
    rho_p = med.eval_drho_ph_dp(P_OUT, h0)
    rho_h = med.eval_drho_ph_dh(P_OUT, h0)
    rp = rho_p + (rho0 * D_HAM / (e_WALL * E_WALL) if elastic else 0.0)
    return rho0, 1.0 / math.sqrt(rp + rho_h / rho0)


def _run_hammer(elastic, stop_after_close):
    med = _water()
    rho0, a = _wave_speed(med, elastic)
    sys = _hammer_system(med, elastic=elastic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _instantiate(sys, med)
        _seed_channel(sys["pipe"]["pipe"], med, P_OUT, T_HAM, w0=0.5)
        sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300,
                       line_search=True)
        dt_wave = (L_HAM / N_HAM) / a / 4.0
        summary = _run(sys, stop=T_CLOSE + stop_after_close, dt_max=dt_wave)
    assert summary["stop_reason"] == "stop_time"
    t = np.asarray(sys.record["time"])
    p_valve = np.asarray(sys.series(f"pipe.pipe.p_{N_HAM}"))
    p_inlet = np.asarray(sys.series("pipe.pipe.p_0"))
    w_mid = np.asarray(sys.series(f"pipe.pipe.w_{N_HAM // 2}"))
    pre = t < T_CLOSE
    return {"t": t, "p": p_valve, "p_in": p_inlet, "rho": rho0, "a": a,
            "w0": float(w_mid[pre][-1]), "p0": float(p_valve[pre][-1])}


_PLOT_DIR = Path(__file__).resolve().parent / "plots"


def _analytic_hammer_wave(r):
    """Closed-form valve pressure for the IDEAL water hammer: instantaneous
    closure at t0 on a frictionless line fed by a constant-pressure reservoir
    (Allievi/Joukowsky).  The reservoir reflection flips the wave sign every
    half period, so the valve sees a square wave

        p(t) = p0 + rho*a*w0 * s(t),  s = +1 on [t0, t0 + 2L/a),
                                          -1 on [t0 + 2L/a, t0 + 4L/a), ...

    The model deviates from it for the *physical* reasons: finite closure time
    (DUR_CLOSE ramp smears the fronts), wall friction (amplitude decay +
    line-packing overshoot), and numerical wave-front smearing at finite N.
    """
    t = np.asarray(r["t"], dtype=float)
    dp = r["rho"] * r["a"] * r["w0"]
    t0 = T_CLOSE + DUR_CLOSE / 2.0            # effective closure instant
    half = 2.0 * L_HAM / r["a"]               # one-way + return transit time
    tt = np.linspace(t[0], t[-1], 4000)
    sign = np.where(tt < t0, 0.0,
                    1.0 - 2.0 * (np.floor((tt - t0) / half) % 2))
    return tt, r["p0"] + dp * sign


def _plot_hammer(runs, filename, title):
    """Write a model-vs-analytic pressure chart for visual inspection.

    `runs` is a list of `(label, result-dict)` pairs from `_run_hammer`.  One
    panel per run: the simulated valve pressure (solid, coloured) overlaid on
    the full analytic frictionless square-wave solution
    (`_analytic_hammer_wave`, black dashed) so amplitude AND period are
    directly comparable by eye; the inlet trace (light grey) shows the
    reservoir boundary staying fixed.  Purely diagnostic: the tests never
    assert on the plot, and a missing matplotlib is a no-op.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                                   # pragma: no cover
        return
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n), sharex=(n > 1))
    axes = np.atleast_1d(axes)
    colors = ("tab:blue", "tab:red", "tab:green")
    for ax, (label, r), color in zip(axes, runs, colors):
        ta, pa = _analytic_hammer_wave(r)
        ax.plot(ta, pa / 1e5, color="k", lw=1.6, ls="--",
                label=f"analytic square wave (a={r['a']:.0f} m/s, "
                      f"dp={r['rho'] * r['a'] * r['w0'] / 1e5:.2f} bar)")
        ax.plot(r["t"], r["p"] / 1e5, color=color, lw=1.8,
                label=f"model: valve pressure p_{N_HAM}")
        ax.plot(r["t"], r["p_in"] / 1e5, color="grey", lw=1.0, alpha=0.7,
                label="model: inlet pressure p_0 (reservoir)")
        ax.axvline(T_CLOSE, color="k", lw=0.9, alpha=0.6)
        ax.text(T_CLOSE, ax.get_ylim()[1], " valve closes", va="top",
                fontsize=8)
        ax.set_ylabel("pressure [bar]")
        ax.set_title(f"{title} -- {label} wall")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = _PLOT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"hammer chart written to {out}")


@pytest.fixture(scope="module")
def hammer_rigid():
    return _run_hammer(elastic=False, stop_after_close=0.35)


def test_water_hammer_joukowsky_amplitude(hammer_rigid):
    """Sudden valve closure on subcooled water: the first pressure peak at the
    valve matches the Joukowsky rise rho*a*w0 (some line-packing excess above
    it is physical; a large deviation either way is not)."""
    r = hammer_rigid
    _plot_hammer([("rigid", r)], "hammer_rigid.png",
                 "Water hammer, rigid wall: inlet vs valve pressure")
    dp_jouk = r["rho"] * r["a"] * r["w0"]
    peak = float(r["p"].max()) - r["p0"]
    assert 0.85 * dp_jouk < peak < 1.4 * dp_jouk, (
        f"peak {peak:.0f} Pa vs Joukowsky {dp_jouk:.0f} Pa")
    # It is a genuine wave: the pressure also swings BELOW the pre-closure
    # level after the reservoir reflection returns.
    post = r["t"] > T_CLOSE
    assert float(r["p"][post].min()) < r["p0"] - 0.5 * dp_jouk


def test_water_hammer_period_is_4L_over_a(hammer_rigid):
    """The hammer oscillation period matches the classic 4L/a."""
    r = hammer_rigid
    period_exact = 4.0 * L_HAM / r["a"]
    post = r["t"] > T_CLOSE + DUR_CLOSE
    sig = r["p"][post] - r["p0"]
    tt = r["t"][post]
    crossings = [tt[k] for k in range(1, len(sig))
                 if sig[k - 1] < 0.0 <= sig[k]]
    assert len(crossings) >= 2, "did not capture a full oscillation"
    period = float(np.mean(np.diff(crossings)))
    assert period == pytest.approx(period_exact, rel=0.10), (
        f"period {period:.4f} s vs 4L/a = {period_exact:.4f} s")


def test_water_hammer_korteweg_wall_reduces_peak(hammer_rigid):
    """`wall_elasticity=True` lowers the wave speed to the Korteweg
    elastic-line value; the hammer peak follows rho*a_eff*w0."""
    r_el = _run_hammer(elastic=True, stop_after_close=0.15)
    _plot_hammer([("rigid", hammer_rigid), ("elastic", r_el)],
                 "hammer_korteweg.png",
                 "Water hammer: rigid vs Korteweg elastic wall")
    med_probe = _water()
    _, a_rigid = _wave_speed(med_probe, elastic=False)
    _, a_el = _wave_speed(med_probe, elastic=True)
    assert a_el < 0.5 * a_rigid          # PVC-like wall: big reduction

    dp_jouk_el = r_el["rho"] * r_el["a"] * r_el["w0"]
    peak_el = float(r_el["p"].max()) - r_el["p0"]
    assert 0.85 * dp_jouk_el < peak_el < 1.4 * dp_jouk_el, (
        f"elastic peak {peak_el:.0f} Pa vs Joukowsky {dp_jouk_el:.0f} Pa")

    # And the elastic peak is far below the rigid one (the whole point of
    # modelling the wall).
    r = hammer_rigid
    peak_rigid = float(r["p"].max()) - r["p0"]
    assert peak_el < 0.55 * peak_rigid


# ===========================================================================
# 4. testing-grade wall flags: structure + placeholder
# ===========================================================================
def test_unsteady_friction_and_viscoelastic_flags_build():
    """The Brunone unsteady-friction and Kelvin-Voigt viscoelastic-wall flags
    produce the expected extra states/parameters and a well-formed equation
    set (structural smoke check)."""
    med = _water()
    ch = SegmentedChannel(med, D=0.05, L=10.0, epsilon=1e-6, z_in=0.0,
                          z_out=0.0, N=3, dynamic="acoustic",
                          unsteady_friction=True, k_uf=0.03,
                          viscoelastic_wall=True, J_ve=1e-11, tau_ve=0.05,
                          wall_elasticity=True, wall_E=3e9, wall_e=0.004)
    assert "k_uf" in ch.components
    assert all(f"eps_ve_{i}" in ch.components for i in range(3))
    assert all(f"der_eps_ve_{i}" in ch.components for i in range(3))
    assert "J_ve" in ch.components and "tau_ve" in ch.components
    assert "wall_E" in ch.components and "wall_e" in ch.components
    ch.assign_symbols(top_level=True)
    eqs = ch.collect_equations()
    assert len(eqs) > 0


def test_fsi_flag_raises_not_implemented():
    """Full fluid-structure interaction is a documented placeholder."""
    med = _water()
    with pytest.raises(NotImplementedError, match="[Ff]luid-structure"):
        SegmentedChannel(med, D=0.05, L=10.0, epsilon=1e-6, z_in=0.0,
                         z_out=0.0, N=3, dynamic="acoustic", fsi=True)


# ===========================================================================
# 5. acoustic + cavitation: column separation (DVCM clamp, rigid-column
#    cavity lifetime, collapse shock)
# ===========================================================================
P_SRC_CAV, P_OUT_CAV = 2.5e5, 2.0e5
T_CLOSE_CAV, DUR_CAV = 0.05, 0.02


def test_cavitation_flag_validation_and_structure():
    """`cavitation` needs the acoustic level and an explicit (or Pipe-derived)
    `p_vap`; when valid it adds one cavity state per cell and the p_vap
    parameter."""
    med = _water()
    common = dict(D=0.05, L=10.0, epsilon=1e-6, z_in=0.0, z_out=0.0, N=3)
    with pytest.raises(ValueError, match="acoustic"):
        SegmentedChannel(med, dynamic="advective", cavitation=True,
                         p_vap=2.3e3, **common)
    with pytest.raises(ValueError, match="p_vap"):
        SegmentedChannel(med, dynamic="acoustic", cavitation=True, **common)
    ch = SegmentedChannel(med, dynamic="acoustic", cavitation=True,
                          p_vap=2.3e3, **common)
    assert all(f"V_cav_{i}" in ch.components for i in range(3))
    assert all(f"der_V_cav_{i}" in ch.components for i in range(3))
    assert "p_vap" in ch.components
    ch.assign_symbols(top_level=True)
    assert len(ch.collect_equations()) > 0


def test_pipe_defaults_p_vap_from_saturation():
    """Pipe(cavitation=True) fills p_vap with the saturation pressure at
    T_wall_init automatically."""
    import CoolProp.CoolProp as CP
    med = _water()
    pipe = Pipe(med, D=0.05, L=10.0, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=3, layers=[], outer_thermal="adiabatic",
                dynamic="acoustic", T_wall_init=293.15, p_init=2.0e5,
                cavitation=True)
    p_sat = float(CP.PropsSI("P", "T", 293.15, "Q", 0, "Water"))
    assert pipe.p_vap == pytest.approx(p_sat, rel=1e-6)
    assert pipe["pipe"].p_vap == pytest.approx(p_sat, rel=1e-6)


def _cavitation_system(med):
    """Low-line-pressure hammer rig: the reflected rarefaction pulls the
    whole line far below the vapor pressure (Joukowsky downswing ~ -5 bar
    on a 2 bar line), i.e. hard distributed column separation."""

    class CavHammer(Model):
        def declare_components(self):
            self.add_component("src", PressureSource(
                med, p_source=P_SRC_CAV, T_source=T_HAM,
                A=math.pi * D_HAM ** 2 / 4.0))
            self.add_component("pipe", Pipe(
                med, D=D_HAM, L=L_HAM, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=N_HAM, layers=[], outer_thermal="adiabatic",
                dynamic="acoustic", T_wall_init=T_HAM, p_init=P_OUT_CAV,
                cavitation=True))
            self.add_component("valve", IncompressibleValve(
                med, Kv=5.0, D=D_HAM, opening=1.0))
            self.add_component("cmd", SmoothRamp(
                offset=1.0, height=-1.0, duration=DUR_CAV,
                start_time=T_CLOSE_CAV, corner=0.25))
            self.add_component("out", PressureOutlet(
                med, p_ambient=P_OUT_CAV, T_ambient=T_HAM))

        def declare_equations(self):
            self.connect(self["src"].ports["outlet"],
                         self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"],
                         self["valve"].ports["inlet"])
            self.connect(self["valve"].ports["outlet"],
                         self["out"].ports["inlet"])
            self.connect(self["cmd"].ports["y"],
                         self["valve"].ports["opening"])
            return []

    return CavHammer()


@pytest.fixture(scope="module")
def cavitating_hammer():
    med = _water()
    h0 = med.eval_h_pT(P_OUT_CAV, T_HAM)
    rho0 = float(med.eval_rho_ph(P_OUT_CAV, h0))
    rho_p = float(med.eval_drho_ph_dp(P_OUT_CAV, h0))
    rho_h = float(med.eval_drho_ph_dh(P_OUT_CAV, h0))
    a = 1.0 / math.sqrt(rho_p + rho_h / rho0)
    sys = _cavitation_system(med)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _instantiate(sys, med)
        _seed_channel(sys["pipe"]["pipe"], med, P_OUT_CAV, T_HAM, w0=0.5)
        sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300,
                       line_search=True)
        dt_wave = (L_HAM / N_HAM) / a / 4.0
        summary = _run(sys, stop=T_CLOSE_CAV + 0.5, dt_max=dt_wave)
    t = np.asarray(sys.record["time"])
    w_mid = np.asarray(sys.series(f"pipe.pipe.w_{N_HAM // 2}"))
    pre = t < T_CLOSE_CAV
    return {
        "summary": summary,
        "t": t,
        "p_valve": np.asarray(sys.series(f"pipe.pipe.p_{N_HAM}")),
        "p_cells": np.stack([np.asarray(sys.series(f"pipe.pipe.pc_{i}"))
                             for i in range(N_HAM)]),
        "V_cav": np.stack([np.asarray(sys.series(f"pipe.pipe.V_cav_{i}"))
                           for i in range(N_HAM)]),
        "p_vap": float(sys["pipe"].p_vap),
        "rho": rho0, "a": a, "w0": float(w_mid[pre][-1]),
        "V_cell": math.pi * D_HAM ** 2 / 4.0 * L_HAM / N_HAM,
    }


def test_column_separation_runs_through_and_clamps(cavitating_hammer):
    """With `cavitation=True` the run survives the dome crossing that kills
    the plain acoustic level, and every cell pressure is clamped near p_vap
    instead of following the (unphysical, negative) Joukowsky downswing."""
    r = cavitating_hammer
    assert r["summary"]["stop_reason"] == "stop_time", r["summary"]
    dp_jouk = r["rho"] * r["a"] * r["w0"]
    # Without a cavity model the reflected wave would demand ~ -5 bar.
    assert P_OUT_CAV - dp_jouk < -3.0e5
    p_min = float(r["p_cells"].min())
    assert p_min > 0.0
    # Clamped within a few kPa above the vapor pressure (the smoothed
    # complementarity keeps a small positive margin).
    assert p_min < r["p_vap"] + 0.05 * (P_OUT_CAV - r["p_vap"]), (
        f"min cell pressure {p_min:.0f} Pa never came down to the "
        f"p_vap clamp {r['p_vap']:.0f} Pa")


def test_column_separation_cavity_lifetime(cavitating_hammer):
    """The first vapor-cavity phase at the valve lasts ~ the rigid-column
    reference 2*rho*L*w0/(p_res - p_vap) (the classic column-separation
    estimate: the liquid column decelerates under the reservoir head and
    returns to collapse the cavity)."""
    r = cavitating_hammer
    thresh = r["p_vap"] + 0.1 * (P_OUT_CAV - r["p_vap"])
    clamped = r["p_valve"] < thresh
    assert clamped.any(), "valve pressure never reached the cavity clamp"
    idx = np.where(clamped)[0]
    brk = np.where(np.diff(idx) > 5)[0]
    i_end = idx[brk[0]] if brk.size else idx[-1]
    t_cav = float(r["t"][i_end] - r["t"][idx[0]])
    t_ref = 2.0 * r["rho"] * L_HAM * r["w0"] / (P_SRC_CAV - r["p_vap"])
    assert t_cav == pytest.approx(t_ref, rel=0.35), (
        f"first cavity phase {t_cav:.3f} s vs rigid-column {t_ref:.3f} s")
    # A real (macroscopic) cavity formed and then collapsed again.
    v_max = float(r["V_cav"].max())
    assert v_max > 1e-4 * r["V_cell"]
    assert float(r["V_cav"][:, -1].max()) < 0.5 * v_max


def test_column_separation_collapse_shock(cavitating_hammer):
    """Cavity collapse re-emits a water-hammer shock of the Joukowsky
    order -- the signature (and the danger) of column separation."""
    r = cavitating_hammer
    thresh = r["p_vap"] + 0.1 * (P_OUT_CAV - r["p_vap"])
    idx = np.where(r["p_valve"] < thresh)[0]
    brk = np.where(np.diff(idx) > 5)[0]
    i_end = idx[brk[0]] if brk.size else idx[-1]
    post = r["t"] > r["t"][i_end]
    assert post.any()
    dp_jouk = r["rho"] * r["a"] * r["w0"]
    rebound = float(r["p_valve"][post].max()) - P_OUT_CAV
    assert rebound > 0.5 * dp_jouk, (
        f"no collapse shock: rebound {rebound:.0f} Pa vs Joukowsky "
        f"{dp_jouk:.0f} Pa")
