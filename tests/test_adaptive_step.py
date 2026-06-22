"""Tests for `Model.solve_adaptive_step` and the four adaptation strategies.

Strategies under test:
  - "fixed"               (no adaptation, calls solve_dae_step once)
  - "derivative_limit"    (B: per-variable change limiter, 0 extra solves)
  - "predictor_corrector" (P: explicit-Euler predictor vs CN corrector mismatch)
  - "richardson"          (R: full step vs two half-steps step-doubling)

Validation strategy:
  1. API surface: shorthand string, dict overrides, defaults, errors.
  2. Accuracy on `IntegrationTest` (closed-form decay + harmonic oscillator):
     each strategy must integrate to within tolerance when given LOOSE per-
     strategy tols (the accuracy comes from CN itself; this test verifies the
     adaptive machinery doesn't break it).
  3. Rejection logic: each strategy must reject + retry on a deliberately
     too-large dt with tight tols.
  4. Richardson commits the half-step result, so it matches a fixed-dt/2 run.
  5. Performance: `predictor_corrector` should reach a target end-time
     accuracy with comparable Newton iteration count to fixed dt.
  6. Per-variable `Variable(..., atol=...)` overrides the global atol.
  7. Newton non-convergence is caught (adaptive) or propagated (when asked).
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen import (
    DifferentialVariable,
    Model,
    NewtonConvergenceFailure,
    Parameter,
)
from hydrogen.test_models import IntegrationTest


OMEGA = 2 * np.pi  # oscillator period of 1 s


def _build_test_model():
    m = IntegrationTest(omega=OMEGA)
    m.instantiate(max_remove_trival_passes=2)
    m.initialise()
    return m


class _DecayOnly(Model):
    """Monotonic single-variable decay used for `derivative_limit` tests.

    `derivative_limit` measures `|x_new - x_pre| / max(|x_pre|, |x_new|, atol)`,
    which is ill-defined for variables that pass through zero (the relative
    change is always ~100% at a zero crossing, so the strategy cannot satisfy
    a tight `rel_tol`).  A monotonically decaying variable avoids this.
    """

    def declare_components(self):
        self.add_component("y", DifferentialVariable(1.0, None))

    def declare_equations(self):
        return [self["der_y"].symbol + self["y"].symbol]


def _build_decay_model():
    m = _DecayOnly()
    m.instantiate(max_remove_trival_passes=2)
    m.initialise()
    return m


def _trace(model, suffix):
    state = np.asarray(model.record["state"])
    names = list(model.record["vars_names"])
    idx = next(i for i, n in enumerate(names) if n.endswith(suffix))
    return state[:, idx]


# Loose per-strategy params for the accuracy/runs-to-completion tests --
# "loose" means "let any reasonable step be accepted, exercise the loop".
# Each value is well above the respective metric on the test problem.
_LOOSE_PARAMS = {
    "fixed":               {},
    "derivative_limit":    {"name": "derivative_limit",    "rel_tol": 5.0},
    "predictor_corrector": {"name": "predictor_corrector", "tol_local": 5.0},
    "richardson":          {"name": "richardson",          "tol_local": 1e-2},
}


def _run_adaptive(model, t_end, dt_target, strategy, **adaptive_kwargs):
    """Step until simulated time >= `t_end`. Returns the list of `info` dicts."""
    info_log = []
    while model.get_t_value() < t_end - 1e-12:
        dt_try = min(dt_target, t_end - model.get_t_value())
        if hasattr(model, "_dt_hint"):
            dt_try = min(dt_try, max(adaptive_kwargs.get("dt_min", 1e-9),
                                     model._dt_hint))
        dt_used, info = model.solve_adaptive_step(
            dt_try, strategy=strategy, **adaptive_kwargs)
        info_log.append(info)
        model.next_step()
    return info_log


# --- 1. Strategy dispatch / API surface ----------------------------------------

def test_strategy_string_shorthand_uses_defaults():
    """Plain `strategy="derivative_limit"` should work with built-in defaults
    on a problem the strategy is suited for (monotonic decay)."""
    model = _build_decay_model()
    dt_used, info = model.solve_adaptive_step(0.001, strategy="derivative_limit")
    assert info["strategy"] == "derivative_limit"
    assert dt_used > 0


def test_strategy_dict_overrides_defaults():
    """User-provided dict params merge over the defaults."""
    model = _build_test_model()
    # tighten tol_local to force at least one rejection on a too-large dt
    dt_used, info = model.solve_adaptive_step(
        0.5,
        strategy={"name": "predictor_corrector", "tol_local": 1e-8, "atol": 1e-10},
        dt_min=1e-6,
    )
    assert info["strategy"] == "predictor_corrector"
    assert info["rejections"] >= 1, (
        "expected at least one rejection on dt=0.5 with tol_local=1e-8")
    assert dt_used < 0.5


def test_strategy_none_defaults_to_predictor_corrector():
    model = _build_test_model()
    _, info = model.solve_adaptive_step(0.001, strategy=None)
    assert info["strategy"] == "predictor_corrector"


def test_unknown_strategy_raises():
    model = _build_test_model()
    with pytest.raises(ValueError, match="unknown adaptive strategy"):
        model.solve_adaptive_step(0.01, strategy="bogus")


def test_strategy_dict_without_name_raises():
    model = _build_test_model()
    with pytest.raises(ValueError, match="must include a 'name' key"):
        model.solve_adaptive_step(0.01, strategy={"tol_local": 1e-4})


# --- 2. Accuracy: each strategy integrates within tolerance --------------------

@pytest.mark.parametrize("strategy_name", [
    "fixed", "derivative_limit", "predictor_corrector", "richardson",
])
def test_accuracy_exponential_decay(strategy_name):
    """With loose per-strategy params, every strategy must integrate exp(-t)
    accurately (the accuracy comes from CN; this test verifies the adaptive
    machinery doesn't regress it)."""
    model = _build_test_model()
    strategy = (strategy_name if strategy_name == "fixed"
                else _LOOSE_PARAMS[strategy_name])
    _run_adaptive(model, t_end=1.0, dt_target=0.04, strategy=strategy,
                  dt_min=1e-6)
    t = np.asarray(model.record["time"])
    y = _trace(model, ".y_decay")
    err = np.max(np.abs(y - np.exp(-t)))
    assert err < 1e-3, f"{strategy_name}: y_decay max error {err:.3e}"


@pytest.mark.parametrize("strategy_name", [
    "fixed", "derivative_limit", "predictor_corrector", "richardson",
])
def test_accuracy_oscillator(strategy_name):
    """Same as decay test but on the harmonic oscillator (whose `z_osc`
    passes through zero -- the case that motivated the symmetric scale)."""
    model = _build_test_model()
    strategy = (strategy_name if strategy_name == "fixed"
                else _LOOSE_PARAMS[strategy_name])
    _run_adaptive(model, t_end=1.0, dt_target=0.04, strategy=strategy,
                  dt_min=1e-6)
    t = np.asarray(model.record["time"])
    y = _trace(model, ".y_osc")
    err = np.max(np.abs(y - np.cos(OMEGA * t)))
    # CN's phase drift over one period at dt=0.04 is ~3%; allow 5%.
    assert err < 5e-2, f"{strategy_name}: y_osc max error {err:.3e}"


# --- 3. Rejection logic: each strategy rejects on too-large dt + tight tols -----

@pytest.mark.parametrize("strategy_dict,model_factory,dt_target", [
    # `derivative_limit` is tested on the monotonic decay model (see comment
    # on `_DecayOnly` -- it can't satisfy a tight `rel_tol` on through-zero
    # variables, which is by design).
    ({"name": "derivative_limit",    "rel_tol": 1e-8,    "atol": 1e-12},
     _build_decay_model, 0.5),
    ({"name": "predictor_corrector", "tol_local": 1e-8,  "atol": 1e-12},
     _build_test_model, 0.1),
    ({"name": "richardson",          "tol_local": 1e-10, "atol": 1e-14},
     _build_test_model, 0.1),
])
def test_strategy_rejects_on_tight_tol(strategy_dict, model_factory, dt_target):
    """Pathologically tight tol must trigger at least one rejection."""
    model = model_factory()
    dt_used, info = model.solve_adaptive_step(
        dt_target, strategy=strategy_dict, dt_min=1e-9, max_retries=40)
    assert info["rejections"] >= 1, info
    assert dt_used < dt_target


# --- 4. Fixed strategy is a thin wrapper around solve_dae_step -----------------

def test_fixed_strategy_matches_solve_dae_step():
    """`strategy='fixed'` MUST produce bit-identical state to `solve_dae_step`."""
    m1 = _build_test_model()
    m2 = _build_test_model()
    for _ in range(5):
        m1.solve_dae_step(0.04)
        m1.next_step()
        m2.solve_adaptive_step(0.04, strategy="fixed")
        m2.next_step()
    np.testing.assert_array_equal(
        np.asarray(m1.record["state"]),
        np.asarray(m2.record["state"]),
    )


# --- 5. Rejected steps must NOT pollute `record` -------------------------------

def test_rejected_step_does_not_record():
    model = _build_test_model()
    n_rec_before = len(model.record["state"])
    _, info = model.solve_adaptive_step(
        0.2,
        strategy={"name": "predictor_corrector", "tol_local": 1e-12, "atol": 1e-14},
        dt_min=1e-9,
        max_retries=40,
    )
    n_rec_mid = len(model.record["state"])
    assert n_rec_mid == n_rec_before, "solve_adaptive_step should not record"
    assert info["rejections"] >= 1
    model.next_step()
    n_rec_after = len(model.record["state"])
    assert n_rec_after == n_rec_before + 1, "next_step should record exactly once"


# --- 6. dt-hint propagates between calls and grows on easy steps ---------------

def test_dt_hint_grows_when_metric_low():
    model = _build_test_model()
    dts = []
    for _ in range(8):
        dt_try = 0.001
        if hasattr(model, "_dt_hint"):
            dt_try = min(0.5, model._dt_hint)
        dt_used, _ = model.solve_adaptive_step(
            dt_try,
            strategy={"name": "derivative_limit", "rel_tol": 1.0, "atol": 1e-3},
            dt_max=1.0,
        )
        dts.append(dt_used)
        model.next_step()
    assert max(dts) > min(dts), f"dt should have grown across calls, got {dts}"


# --- 7. Richardson commits half-step result ------------------------------------

def test_richardson_commits_half_step_result():
    """One Richardson step at dt commits the result of two half-steps at dt/2."""
    m_richardson = _build_test_model()
    m_fixed = _build_test_model()
    dt = 0.04
    # Loose tol so we never reject (we want to compare COMMITTED state)
    m_richardson.solve_adaptive_step(
        dt,
        strategy={"name": "richardson", "tol_local": 10.0, "atol": 1.0},
    )
    m_fixed.solve_dae_step(dt / 2)
    m_fixed.next_step()
    m_fixed.solve_dae_step(dt / 2)
    np.testing.assert_allclose(
        m_richardson.get_vars_values(),
        m_fixed.get_vars_values(),
        rtol=1e-12, atol=1e-14,
    )


# --- 8. Performance: adaptive PC is competitive with fixed dt ------------------

def test_predictor_corrector_competitive_with_fixed_dt():
    """For the same end-of-run accuracy on `y_osc`, the adaptive strategy
    should use a comparable number of total Newton iterations to fixed dt."""
    t_end = 1.0
    target_err = 1e-2

    # Fixed dt: hand-tuned to land near `target_err`
    m_fixed = _build_test_model()
    n_fixed_iters = 0
    dt_fixed = 0.025
    while m_fixed.get_t_value() < t_end - 1e-12:
        m_fixed.solve_dae_step(min(dt_fixed, t_end - m_fixed.get_t_value()))
        n_fixed_iters += m_fixed._last_solve_iters
        m_fixed.next_step()
    err_fixed = np.max(np.abs(
        _trace(m_fixed, ".y_osc")
        - np.cos(OMEGA * np.asarray(m_fixed.record["time"]))))

    # Adaptive (PC) tuned to similar end-of-run accuracy.  PC's metric is
    # the FE-CN mismatch, which is ~dt times larger than the CN local error;
    # for a 1Hz oscillator with peak |y'''| = omega^3, tol_local=2e-2 puts
    # dt around the same order as the fixed dt above.
    m_ad = _build_test_model()
    n_ad_iters = 0
    n_ad_steps = 0
    n_ad_rejections = 0
    while m_ad.get_t_value() < t_end - 1e-12:
        dt_try = min(0.04, t_end - m_ad.get_t_value())
        if hasattr(m_ad, "_dt_hint"):
            dt_try = min(dt_try, m_ad._dt_hint)
        _, info = m_ad.solve_adaptive_step(
            dt_try,
            strategy={"name": "predictor_corrector", "tol_local": 1e-2, "atol": 1e-6},
        )
        n_ad_iters += info["n_iters"]
        n_ad_rejections += info["rejections"]
        n_ad_steps += 1
        m_ad.next_step()
    err_ad = np.max(np.abs(
        _trace(m_ad, ".y_osc")
        - np.cos(OMEGA * np.asarray(m_ad.record["time"]))))

    assert err_fixed < target_err, f"fixed err={err_fixed:.3e}"
    assert err_ad < target_err,    f"adaptive err={err_ad:.3e}"

    # Loose bound: adaptive shouldn't be more than 4x the fixed iter count.
    # (In practice a linear oscillator is the WORST case for adaptive --
    # nothing varies in stiffness so there's no win to extract.)
    assert n_ad_iters <= 4 * n_fixed_iters, (
        f"adaptive Newton iters {n_ad_iters} vs fixed {n_fixed_iters} "
        f"(adaptive: {n_ad_steps} steps, {n_ad_rejections} rejections)")


# --- 9. Per-variable atol on `Variable(..., atol=...)` overrides global --------

def test_variable_atol_overrides_global():
    class TightModel(Model):
        def declare_components(self):
            self.add_component("p", Parameter(1.0, None))
            # Tight per-variable atol that would never come from a global default.
            self.add_component("y", DifferentialVariable(1.0, None, atol=1e-12))

        def declare_equations(self):
            return [self["der_y"].symbol + self["p"].symbol * self["y"].symbol]

    m = TightModel()
    m.instantiate(max_remove_trival_passes=2)
    m.initialise()

    refs = m.active_vars_references
    y_idx = next(i for i, v in enumerate(refs) if v is m["y"])
    assert refs[y_idx].atol == 1e-12, "atol kwarg should reach active_vars"

    atols = m._get_var_atols(fallback_atol=1.0)
    assert atols[y_idx] == 1e-12, "per-variable atol should override fallback"
    assert any(a == 1.0 for a in atols), "fallback atol must apply to non-overridden vars"


# --- 10. Newton non-convergence handling ---------------------------------------

def test_newton_failure_is_caught_and_retried():
    """If Newton can't converge, the strategy must catch the
    `NewtonConvergenceFailure` and rejection-loop -- never let the bare
    exception bubble up.  Here we provoke non-convergence with `relaxation`
    so small that NO dt can converge in `max_iter`, so the strategy will
    eventually give up at `dt_min` -- but it must give up via its own
    `RuntimeError` ("hit dt_min"), NOT by leaking the underlying
    `NewtonConvergenceFailure`."""
    model = _build_test_model()
    with pytest.raises(RuntimeError, match="hit dt_min"):
        model.solve_adaptive_step(
            0.5,
            strategy={"name": "derivative_limit", "rel_tol": 1.0, "atol": 1.0},
            dt_min=1e-4,
            relaxation=0.001,    # too damped to converge in `max_iter` at any dt
            max_iter=5,
            tol=1e-12,
        )


def test_newton_failure_propagates_when_directly_requested():
    """`solve_dae_step(..., raise_on_no_convergence=True)` SHOULD raise
    when Newton actually fails to reduce residual below `tol` in `max_iter`."""
    model = _build_test_model()
    with pytest.raises(NewtonConvergenceFailure):
        # Tiny relaxation + tight tol + low max_iter -> can't possibly converge.
        model.solve_dae_step(
            5.0, max_iter=3, tol=1e-12, relaxation=1e-4,
            raise_on_no_convergence=True,
        )


# --- 11. Strategy returns useful `info` dict -----------------------------------

def test_info_dict_contents():
    model = _build_test_model()
    _, info = model.solve_adaptive_step(0.01, strategy="predictor_corrector")
    for key in ("strategy", "rejections", "metric", "n_iters"):
        assert key in info, f"missing {key} in info dict: {info}"
    assert info["rejections"] >= 0
    assert info["metric"] >= 0.0
    assert info["n_iters"] >= 1


# --- 12. Model.run: the high-level loop driver ---------------------------------

def test_run_to_stop_time_integrates_accurately():
    """`run(stop_time=...)` owns the loop, lands exactly on stop_time, and
    integrates the decay model to the analytic value."""
    model = _build_decay_model()
    summary = model.run(
        stop_time=1.0,
        strategy={"name": "richardson", "tol_local": 1e-3, "atol": 1.0},
        dt_start=0.01, dt_min=1e-6, dt_max=0.2)
    assert summary["stop_reason"] == "stop_time"
    assert summary["steps"] > 0
    assert abs(summary["t_end"] - 1.0) < 1e-9
    y_end = _trace(model, ".y")[-1]
    assert abs(y_end - np.exp(-1.0)) < 1e-3


def test_run_fixed_steps_counts_and_advances():
    """`run(strategy='fixed', dt, steps)` takes exactly `steps` steps."""
    model = _build_test_model()
    summary = model.run(strategy="fixed", dt=0.02, steps=10)
    assert summary["stop_reason"] == "steps"
    assert summary["steps"] == 10
    assert summary["rejections"] == 0
    assert abs(summary["t_end"] - 0.2) < 1e-12


def test_run_fixed_matches_manual_loop():
    """The fixed path must reproduce a hand-rolled solve_dae_step loop exactly."""
    m_run = _build_test_model()
    m_manual = _build_test_model()
    m_run.run(strategy="fixed", dt=0.03, steps=7)
    for _ in range(7):
        m_manual.solve_dae_step(0.03)
        m_manual.next_step()
    np.testing.assert_array_equal(
        np.asarray(m_run.record["state"]),
        np.asarray(m_manual.record["state"]),
    )


def test_run_requires_a_stop_condition():
    model = _build_test_model()
    with pytest.raises(ValueError, match="stop_time, steps"):
        model.run(strategy="fixed", dt=0.01)


def test_run_fixed_requires_dt():
    model = _build_test_model()
    with pytest.raises(ValueError, match="requires a dt"):
        model.run(steps=5, strategy="fixed")


def test_run_on_step_callback_can_stop_early():
    """Returning False from `on_step` requests a cooperative stop."""
    model = _build_test_model()
    seen = []

    def _cb(_m, info):
        seen.append(info["step"])
        return info["step"] < 3            # stop right after the 3rd step

    summary = model.run(steps=100, strategy="fixed", dt=0.01, on_step=_cb)
    assert summary["stop_reason"] == "callback"
    assert summary["steps"] == 3
    assert seen == [1, 2, 3]


def test_run_resumes_from_current_time():
    """Two back-to-back `run` calls continue from where the first left off."""
    model = _build_test_model()
    model.run(stop_time=0.5, strategy="fixed", dt=0.05)
    t_mid = model.get_t_value()
    summary = model.run(stop_time=1.0, strategy="fixed", dt=0.05)
    assert abs(t_mid - 0.5) < 1e-12
    assert abs(summary["t_end"] - 1.0) < 1e-12
    assert summary["t_start"] == pytest.approx(t_mid)
