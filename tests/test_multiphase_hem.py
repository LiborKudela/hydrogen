"""Tests for the smooth homogeneous-equilibrium (HEM) two-phase property mode.

Three things are checked:

  1. The medium's `*_ph_hem` partials are SMOOTH and CONSISTENT inside the
     two-phase dome -- a finite, non-zero `drho/dh` (a central difference of
     the same CoolProp values) where the single-phase analytic partial is
     either ~0 or undefined.

  2. The `multiphase` flag is validated and is part of the per-class equation
     cache key, and `StraightPipe` forwards it to its segments.

  3. End to end: the SAME heated boiler that the single-phase closures cannot
     push past the saturation line marches well into the two-phase dome when
     built with `multiphase="HEM"`.

The tabular ``BICUBIC&HEOS`` backend is used throughout: it is ~50x faster than
HEOS (keeps the boiler test snappy) and raises outright on single-phase
derivatives in the dome, which makes the single-phase failure unambiguous.
"""

from __future__ import annotations

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from hydrogen import (
    AmbientInlet,
    CoolPropMedium,
    FixedHeatFlow,
    Model,
    NewtonConvergenceFailure,
    StraightPipe,
)
from hydrogen.components.thermofluid.flow import TwoPortSegment

# --- shared operating point ---------------------------------------------------
FLUID = "Water"
P_OP = 5.0e5        # Pa
T_IN = 300.0        # K (subcooled liquid)
M_FLOW = 1.0e-3     # kg/s
D = 0.01            # m
L = 1.0             # m


def _medium():
    return CoolPropMedium(FLUID, disable_warnings=True, backend="BICUBIC&HEOS")


def _sat_enthalpies():
    hf = CP.PropsSI("H", "P", P_OP, "Q", 0, FLUID)
    hg = CP.PropsSI("H", "P", P_OP, "Q", 1, FLUID)
    return hf, hg


# -----------------------------------------------------------------------------
# 1. Medium-level HEM partials
# -----------------------------------------------------------------------------


def test_hem_partials_are_smooth_and_consistent_in_the_dome():
    med = _medium()
    hf, hg = _sat_enthalpies()
    h_mid = hf + 0.3 * (hg - hf)  # squarely inside the two-phase dome

    # The smooth-HEM symbolic property functions must be exposed.
    for name in ("rho_ph_hem", "T_ph_hem", "mu_ph_hem", "k_ph_hem"):
        assert hasattr(med, name), f"medium missing {name}"

    # HEM value == single-phase value (CoolProp's (p,h) flash already gives the
    # HEM mixture inside the dome); only the partials differ.
    assert med.eval_rho_ph(P_OP, h_mid) == pytest.approx(
        CP.PropsSI("D", "P", P_OP, "H", h_mid, FLUID), rel=1e-9)

    # drho/dh(HEM) is finite, non-zero, and negative (density falls as the
    # mixture boils) -- a usable Jacobian entry.
    drho_hem = med.eval_drho_ph_hem_dh(P_OP, h_mid)
    assert np.isfinite(drho_hem)
    assert drho_hem < -1e-7

    # And it equals an independent central difference of the SAME values.
    e = med.hem_fd_dh
    cd = (med.eval_rho_ph(P_OP, h_mid + e) - med.eval_rho_ph(P_OP, h_mid - e)) / (2 * e)
    assert drho_hem == pytest.approx(cd, rel=1e-9)

    # The single-phase analytic partial, by contrast, is NOT a usable Jacobian
    # entry in the dome (tabular backend raises; HEOS would return ~0).
    with pytest.raises(Exception):
        med.eval_drho_ph_dh(P_OP, h_mid)

    # In a single phase (subcooled liquid) the two agree to FD accuracy.
    h_liq = 0.5 * (med.eval_h_pT(P_OP, T_IN) + hf)
    assert med.eval_drho_ph_hem_dh(P_OP, h_liq) == pytest.approx(
        med.eval_drho_ph_dh(P_OP, h_liq), rel=5e-2)


# -----------------------------------------------------------------------------
# 2. Flag validation / cache key / propagation
# -----------------------------------------------------------------------------


def test_multiphase_flag_is_validated_and_cache_keyed():
    med = _medium()

    from hydrogen.paramspec import cache_key_flag_names
    assert "multiphase" in cache_key_flag_names(TwoPortSegment)

    # Bad values are rejected eagerly on both the segment and the pipe.
    with pytest.raises(ValueError):
        TwoPortSegment(med, D, D, D, D, 0, 0, L, 0.0,
                       lambda *a: 0.0, lambda *a: 0.0, multiphase="bogus")
    with pytest.raises(ValueError):
        StraightPipe(med, D=D, L=L, epsilon=0.0, z_in=0, z_out=0,
                     n_segments=2, multiphase="bogus")

    # The pipe forwards the mode verbatim to every segment.
    pipe = StraightPipe(med, D=D, L=L, epsilon=1e-5, z_in=0, z_out=0,
                        n_segments=3, heat_port=True, multiphase="HEM")
    assert pipe.multiphase == "HEM"
    for i in range(3):
        assert pipe[f"pipe_segment_{i}"].multiphase == "HEM"


# -----------------------------------------------------------------------------
# 3. End-to-end: HEM boils where single-phase cannot
# -----------------------------------------------------------------------------


class _Boiler(Model):
    """AmbientInlet(liquid) -> heated StraightPipe -> open outlet."""

    N = 2

    def __init__(self, medium, multiphase):
        self._medium = medium
        self._multiphase = multiphase
        super().__init__()

    def declare_components(self):
        self.add_component("inlet", AmbientInlet(
            self._medium, p_ambient=P_OP, T_ambient=T_IN, m_flow=M_FLOW, D=D))
        self.add_component("pipe", StraightPipe(
            self._medium, D=D, L=L, epsilon=1e-5, z_in=0, z_out=0,
            n_segments=self.N, heat_port=True, multiphase=self._multiphase))
        for i in range(self.N):
            self.add_component(f"heat_{i}", FixedHeatFlow(Q_flow=0.0, T_init=T_IN))

    def declare_equations(self):
        self.connect(self["inlet"].ports["outlet"], self["pipe"].ports["inlet"])
        wall_ports = self["pipe"].segment_wall_ports
        for i in range(self.N):
            self.connect(wall_ports[i], self[f"heat_{i}"].ports["heat"])
        return []


def _ramp_quality(multiphase, relaxation, n_ramp, line_search=False):
    """Ramp wall heat on a `_Boiler(multiphase)` and return the max outlet
    quality reached before the solve (if ever) breaks down."""
    med = _medium()
    hf, hg = _sat_enthalpies()
    h_in = med.eval_h_pT(P_OP, T_IN)
    q_to_dry = M_FLOW * (hg - h_in)

    sys = _Boiler(med, multiphase)
    sys.instantiate(aditional_modules=med.modules, max_remove_trival_passes=5)
    sys.initialise(n=1, tol=1e-3, max_iter=400, line_search=line_search)

    names = list(sys.record["vars_names"])
    i_h = next(i for i, n in enumerate(names)
               if n.endswith(f".pipe_segment_{_Boiler.N - 1}.h_out"))

    x_max = -1.0
    broke = False
    for k in range(1, n_ramp + 1):
        q = 1.05 * q_to_dry * k / n_ramp
        for i in range(_Boiler.N):
            sys[f"heat_{i}"]["Q_flow"].set_value(q / _Boiler.N)
        try:
            sys.solve_dae_step(1.0, tol=1e-3, max_iter=250,
                               relaxation=relaxation, line_search=line_search,
                               raise_on_no_convergence=True)
            sys.next_step()
        except (NewtonConvergenceFailure, RuntimeError, ValueError,
                np.linalg.LinAlgError):
            broke = True
            break
        h_out = float(np.asarray(sys.record["state"][-1])[i_h])
        x_max = max(x_max, (h_out - hf) / (hg - hf))
    return x_max, broke


def test_hem_boils_through_the_dome_where_single_phase_breaks():
    # Single phase: a healthy step, but it cannot cross the saturation line --
    # it breaks down while still essentially liquid (x ~ 0).
    x_single, broke_single = _ramp_quality("single", relaxation=0.5, n_ramp=24)
    assert broke_single, "single-phase solve unexpectedly survived the dome"
    assert x_single < 0.05, f"single-phase unexpectedly entered the dome (x={x_single:.3f})"

    # HEM: a damped step walks the SAME boiler well into the two-phase dome
    # with no breakdown.  (A coarser ramp than the example -- enough to prove
    # it crosses the saturation line -- to keep the test snappy.)
    x_hem, broke_hem = _ramp_quality("HEM", relaxation=0.25, n_ramp=60)
    assert not broke_hem, f"HEM solve broke down inside the dome (x={x_hem:.3f})"
    assert x_hem > 0.25, f"HEM did not get far into the dome (x={x_hem:.3f})"


def test_hem_with_line_search_crosses_to_steam_at_full_steps():
    # The backtracking line search lets HEM take FULL Newton steps
    # (relaxation=1.0) and still cross the whole dome to superheated steam (x>1)
    # -- no hand-tuned damping, and on a coarse heat ramp.
    x_hem, broke_hem = _ramp_quality("HEM", relaxation=1.0, n_ramp=40,
                                     line_search=True)
    assert not broke_hem, f"line-search HEM broke down (x={x_hem:.3f})"
    assert x_hem > 1.0, f"line-search HEM did not reach superheated steam (x={x_hem:.3f})"
