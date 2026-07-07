"""Tests for `TabulatedMedium` (spline-surrogate property layer).

Three groups:

1. *Surrogate accuracy* -- tabulated values/partials vs the source medium on
   single-phase points, plus exactness of the piecewise value join across the
   saturation line.
2. *Internal consistency* -- the analytic partials must be the derivatives of
   the interpolant itself (finite-difference cross-checks, including second
   derivatives of rho and continuity through the dome-edge blend band), and
   evaluation outside the window must extend linearly (finite, C^1).
3. *Integration* -- an acoustic water-hammer rig instantiated with the
   tabulated modules reproduces the Joukowsky amplitude, proving the module
   names / symbolic wiring are drop-in compatible with `CoolPropMedium`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import CoolProp.CoolProp as CP

from hydrogen import CoolPropMedium, Model, TabulatedMedium
from hydrogen.components.control.control_components import SmoothRamp
from hydrogen.components.thermofluid.flow import (
    IncompressibleValve,
    PressureOutlet,
    PressureSource,
    SegmentedChannel,
)

# ---------------------------------------------------------------------------
# fixtures: a liquid-only window and a dome-crossing window (both Nitrogen,
# small grids so the one-time sampling stays test-friendly)
# ---------------------------------------------------------------------------

P_RANGE = (0.8e5, 30e5)


@pytest.fixture(scope="module")
def src():
    return CoolPropMedium("Nitrogen", disable_warnings=True, backend="HEOS",
                          scalar_cache_maxsize=4000)


@pytest.fixture(scope="module")
def tab(src):
    """Dome-crossing window: subcooled LN2 up to ~30 kJ/kg inside the dome
    (the DVCM-grazing envelope of the hammer benchmark)."""
    h_lo = float(src.eval_h_pT(P_RANGE[1], 64.0))
    h_hi = float(CP.PropsSI("H", "P", P_RANGE[0], "Q", 0, "Nitrogen")) + 3e4
    return TabulatedMedium(src, p_range=P_RANGE, h_range=(h_lo, h_hi),
                           n_p=64, n_h=64, cache=False, validate=True)


def _liquid_points(src, tab, n=60, seed=0):
    """Random single-phase liquid points INSIDE the table window, clear of
    the blend band (the window is a (p, h) rectangle: at high pressure the
    liquid region extends beyond h_max, where evaluation is by-design a
    clamped linear extension -- excluded here)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(*P_RANGE, n)
    h_l = np.array([float(CP.PropsSI("H", "P", pi, "Q", 0, "Nitrogen"))
                    for pi in p])
    h_top = np.minimum(h_l - 2 * tab.blend_width, tab.h_max)
    h = tab.h_min + rng.uniform(0.05, 0.95, n) * (h_top - tab.h_min)
    return p, h


# ===========================================================================
# 1. surrogate accuracy vs the source medium
# ===========================================================================

def test_values_match_source_single_phase(src, tab):
    p, h = _liquid_points(src, tab)
    for prop, tol in (("rho", 5e-5), ("T", 5e-6), ("mu", 5e-4),
                      ("k", 5e-4), ("s", 5e-4)):
        got = np.array([getattr(tab, f"eval_{prop}_ph")(pi, hi)
                        for pi, hi in zip(p, h)])
        ref = np.array([float(getattr(src, f"eval_{prop}_ph")(pi, hi))
                        for pi, hi in zip(p, h)])
        rel = np.abs(got - ref) / np.abs(ref)
        assert rel.max() < tol, f"{prop}: max rel err {rel.max():.2e}"


def test_first_partials_match_source_single_phase(src, tab):
    p, h = _liquid_points(src, tab, seed=1)
    for name, tol in (("drho_ph_dp", 2e-3), ("drho_ph_dh", 2e-3),
                      ("dT_ph_dh", 2e-3), ("ds_ph_dh", 2e-3)):
        got = np.array([getattr(tab, f"eval_{name}")(pi, hi)
                        for pi, hi in zip(p, h)])
        ref = np.array([float(getattr(src, f"eval_{name}")(pi, hi))
                        for pi, hi in zip(p, h)])
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12)
        assert np.median(rel) < tol, f"{name}: median rel err {np.median(rel):.2e}"


def test_values_exact_across_dome(src, tab):
    """The piecewise value join keeps rho/T EXACT (to table resolution) right
    through the saturation line, including inside the blend band."""
    p0 = 2.0e5
    hl = float(CP.PropsSI("H", "P", p0, "Q", 0, "Nitrogen"))
    # dh capped so h stays inside the table window (h_max = h_l(0.8 bar)+30
    # kJ/kg; outside it the clamp+linear-extension is intentional).
    for dh in (-5e3, -500.0, -5.0, 5.0, 500.0, 5e3, 1.2e4):
        h = hl + dh
        r_tab = tab.eval_rho_ph(p0, h)
        r_src = float(src.eval_rho_ph(p0, h))
        assert r_tab == pytest.approx(r_src, rel=2e-3), f"dh={dh}"
        assert tab.eval_T_ph(p0, h) == pytest.approx(
            float(src.eval_T_ph(p0, h)), rel=1e-4), f"dh={dh}"


def test_batch_equals_scalar(tab):
    p = np.linspace(2e5, 25e5, 40)
    h = np.linspace(tab.h_min + 1e3, tab.h_max - 1e3, 40)
    batch = tab.eval_rho_ph(p, h)
    scal = np.array([tab.eval_rho_ph(float(pi), float(hi))
                     for pi, hi in zip(p, h)])
    assert np.allclose(batch, scal, rtol=0, atol=0)
    assert getattr(tab.eval_rho_ph, "_hydrogen_vectorised", False)


# ===========================================================================
# 2. internal consistency of the interpolant
# ===========================================================================

def _fd(f, p, h, wrt, step):
    if wrt == "p":
        return (f(p + step, h) - f(p - step, h)) / (2 * step)
    return (f(p, h + step) - f(p, h - step)) / (2 * step)


def test_partials_are_derivatives_of_the_value(tab):
    """Analytic partials == FD of the tabulated value itself (single phase)."""
    pts = [(8e5, tab.h_min + 2e4), (20e5, tab.h_min + 5e4)]
    for p0, h0 in pts:
        assert tab.eval_drho_ph_dp(p0, h0) == pytest.approx(
            _fd(tab.eval_rho_ph, p0, h0, "p", 20.0), rel=1e-5)
        assert tab.eval_drho_ph_dh(p0, h0) == pytest.approx(
            _fd(tab.eval_rho_ph, p0, h0, "h", 5.0), rel=1e-5)
        assert tab.eval_d2rho_ph_dp2(p0, h0) == pytest.approx(
            _fd(tab.eval_drho_ph_dp, p0, h0, "p", 50.0), rel=1e-4, abs=1e-18)
        assert tab.eval_d2rho_ph_dpdh(p0, h0) == pytest.approx(
            _fd(tab.eval_drho_ph_dp, p0, h0, "h", 10.0), rel=1e-4, abs=1e-14)
        assert tab.eval_d2rho_ph_dh2(p0, h0) == pytest.approx(
            _fd(tab.eval_drho_ph_dh, p0, h0, "h", 10.0), rel=1e-4, abs=1e-12)


def test_second_partials_consistent_inside_blend_band(tab):
    """The dome-edge blend keeps d2rho/dh2 the exact derivative of drho/dh --
    what the acoustic Newton Jacobian differentiates."""
    p0 = 2.0e5
    hl = float(CP.PropsSI("H", "P", p0, "Q", 0, "Nitrogen"))
    for dh in (-0.4 * tab.blend_width, 0.3 * tab.blend_width,
               0.9 * tab.blend_width):
        h0 = hl + dh
        assert tab.eval_d2rho_ph_dh2(p0, h0) == pytest.approx(
            _fd(tab.eval_drho_ph_dh, p0, h0, "h", 2.0), rel=1e-3)


def test_first_derivative_continuous_through_dome_edge(tab):
    """drho/dh has no jumps through the saturation line (the HEM smoothing
    role): scan with a fine step and bound the relative increments."""
    p0 = 2.0e5
    hl = float(CP.PropsSI("H", "P", p0, "Q", 0, "Nitrogen"))
    hs = np.linspace(hl - 2 * tab.blend_width, hl + 2 * tab.blend_width, 801)
    d = tab.eval_drho_ph_dh(np.full_like(hs, p0), hs)
    assert np.all(np.isfinite(d))
    steps = np.abs(np.diff(d))
    scale = np.abs(d).max()
    # A hard kink would concentrate the whole slope change (~scale) into one
    # step; the quintic blend spreads it over the band (>= 100 scan steps).
    assert steps.max() < 0.05 * scale


def test_out_of_window_extension_is_finite_positive_c1(tab):
    """Outside the window, positive-definite properties (rho, T, mu, k)
    extend EXPONENTIALLY: C^1 at the edge, finite and positive everywhere --
    even absurdly far out (an unseeded ambient-gas boundary node against a
    cryogenic-liquid table must not produce negative rho/mu)."""
    p_hi = tab.p_max + 5e5
    h0 = tab.h_min + 3e4
    f_edge = tab.eval_rho_ph(tab.p_max, h0)
    fp_edge = tab.eval_drho_ph_dp(tab.p_max, h0)
    got = tab.eval_rho_ph(p_hi, h0)
    assert np.isfinite(got) and got > 0
    assert got == pytest.approx(
        f_edge * math.exp(fp_edge * (p_hi - tab.p_max) / f_edge), rel=1e-12)
    # C^1: the analytic slope at the edge matches an FD straddling it.
    step = 5.0
    fd = (tab.eval_rho_ph(tab.p_max + step, h0)
          - tab.eval_rho_ph(tab.p_max - step, h0)) / (2 * step)
    assert fd == pytest.approx(fp_edge, rel=1e-3)
    # Extremely far out (ambient gas state vs LN2 table): finite + positive.
    for prop in ("rho", "T", "mu", "k"):
        v = getattr(tab, f"eval_{prop}_ph")(101325.0, 3.0e5)
        assert np.isfinite(v) and v > 0, f"{prop}: {v}"


def test_disk_cache_roundtrip(src, tmp_path, monkeypatch):
    monkeypatch.setenv("HYDROGEN_LAMBDA_CACHE", str(tmp_path))
    h_lo, h_hi = -140e3, -110e3            # small liquid-only window
    kw = dict(p_range=(5e5, 20e5), h_range=(h_lo, h_hi), n_p=24, n_h=24,
              validate=False)
    t1 = TabulatedMedium(src, cache=True, **kw)
    assert list(tmp_path.glob("proptab_*.npz")), "table file not written"
    # Second construction loads the npz (poison the sampler to prove it).
    monkeypatch.setattr(TabulatedMedium, "_sample_tables",
                        lambda self: (_ for _ in ()).throw(
                            AssertionError("cache not used")))
    t2 = TabulatedMedium(src, cache=True, **kw)
    p0, h0 = 12e5, -125e3
    assert t1.eval_rho_ph(p0, h0) == t2.eval_rho_ph(p0, h0)


def test_dome_window_requires_saturation_protocol(src):
    class NoSat:
        medium = src.medium
        eval_h_pT = src.eval_h_pT

    with pytest.raises(ValueError, match="sample_saturation"):
        TabulatedMedium(NoSat(), p_range=(1e5, 5e5), h_range=(-1e5, 2e5),
                        n_p=8, n_h=8, two_phase=True, cache=False,
                        validate=False)


# ===========================================================================
# 3. integration: acoustic water hammer on tabulated water
# ===========================================================================

P_SRC, P_OUT, T_HAM = 10.5e5, 10.0e5, 293.15
D_HAM, L_HAM, N_HAM = 0.05, 50.0, 6
T_CLOSE, DUR_CLOSE = 0.05, 0.01


def _seed_channel(ch, medium, p0, T0, w0=0.0):
    N = ch.N
    h0 = float(medium.eval_h_pT(p0, T0))
    rho0 = float(medium.eval_rho_ph(p0, h0))
    k0 = float(medium.eval_k_ph(p0, h0))
    A = math.pi * float(ch.D) ** 2 / 4.0
    for i in range(N):
        for stem, val in (("hc", h0), ("Tc", T0), ("pc", p0),
                          ("rhoc", rho0), ("kc", k0)):
            key = f"{stem}_{i}"
            if key in ch.components:
                ch[key].value = val
    for j in range(N + 1):
        for stem, val in (("h", h0), ("p", p0), ("T", T0), ("rho", rho0),
                          ("w", w0), ("M", rho0 * A * w0)):
            key = f"{stem}_{j}"
            if key in ch.components:
                ch[key].value = val


@pytest.mark.slow
def test_water_hammer_runs_on_tabulated_medium():
    """Drop-in check: the acoustic level instantiated with `tab.modules`
    (same lambdify names as CoolProp) reproduces the Joukowsky rise."""
    src_w = CoolPropMedium("Water", disable_warnings=True,
                           backend="BICUBIC&HEOS", scalar_cache_maxsize=2000)
    h0 = float(src_w.eval_h_pT(P_OUT, T_HAM))
    tab_w = TabulatedMedium(src_w, p_range=(0.5e5, 40e5),
                            h_range=(h0 - 60e3, h0 + 120e3),
                            n_p=64, n_h=64, cache=False)

    class Hammer(Model):
        def declare_components(self):
            self.add_component("src", PressureSource(
                tab_w, p_source=P_SRC, T_source=T_HAM,
                A=math.pi * D_HAM ** 2 / 4.0))
            self.add_component("pipe", SegmentedChannel(
                tab_w, D=D_HAM, L=L_HAM, epsilon=1e-6, z_in=0.0, z_out=0.0,
                N=N_HAM, dynamic="acoustic", p_init=P_OUT))
            self.add_component("valve", IncompressibleValve(
                tab_w, Kv=5.0, D=D_HAM, opening=1.0))
            self.add_component("cmd", SmoothRamp(
                offset=1.0, height=-1.0, duration=DUR_CLOSE,
                start_time=T_CLOSE, corner=0.25))
            self.add_component("out", PressureOutlet(
                tab_w, p_ambient=P_OUT, T_ambient=T_HAM))

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

    rho0 = float(tab_w.eval_rho_ph(P_OUT, h0))
    rho_p = float(tab_w.eval_drho_ph_dp(P_OUT, h0))
    rho_h = float(tab_w.eval_drho_ph_dh(P_OUT, h0))
    a = 1.0 / math.sqrt(rho_p + rho_h / rho0)

    sys = Hammer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.instantiate(aditional_modules=tab_w.modules, cse=True,
                        enable_blt=True, enable_var_scaling=True,
                        max_remove_trival_passes=1,
                        max_remove_duplicate_passes=5,
                        max_remove_linear_block_passes=3)
        _seed_channel(sys["pipe"], tab_w, P_OUT, T_HAM, w0=0.5)
        sys.initialise(n=1, relaxation=1.0, tol=1e-8, max_iter=300,
                       line_search=True)
        dt_wave = (L_HAM / N_HAM) / a / 4.0
        summary = sys.run(
            stop_time=T_CLOSE + 1.5 * 4 * L_HAM / a,
            strategy={"name": "tr_bdf2", "tol_local": 1e-3, "atol": 0.5},
            dt_start=1e-3, dt_min=1e-9, dt_max=dt_wave, grow=1.5, shrink=0.5,
            max_retries=40, relaxation=1.0, tol=1e-8, max_iter=300,
            raise_on_no_convergence=False)
    assert summary["stop_reason"] == "stop_time"

    t = np.asarray(sys.record["time"])
    p_valve = np.asarray(sys.series(f"pipe.p_{N_HAM}"))
    w_mid = np.asarray(sys.series(f"pipe.w_{N_HAM // 2}"))
    pre = t < T_CLOSE
    w0 = float(w_mid[pre][-1])
    p0 = float(p_valve[pre][-1])
    dp_jouk = rho0 * a * w0
    peak = float(p_valve.max()) - p0
    assert 0.85 * dp_jouk < peak < 1.4 * dp_jouk, (
        f"peak {peak:.0f} Pa vs Joukowsky {dp_jouk:.0f} Pa "
        f"(a={a:.0f} m/s, w0={w0:.3f} m/s)")
    post = t > T_CLOSE
    assert float(p_valve[post].min()) < p0 - 0.5 * dp_jouk
