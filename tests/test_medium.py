"""CoolPropMedium tests covering the batch evaluators and `backend` kwarg.

The batch evaluators (`eval_*_batch`) MUST agree bit-for-bit with their
scalar counterparts on the HEOS backend, since they are exposed as
`hydrogen.batch_modules` for users who want to opt into vectorised
property evaluation.  The `backend` kwarg lets callers swap the slow
Newton-loop HEOS solver for the much faster BICUBIC&HEOS tabular
backend (~50x faster per state update, ~1e-4 max relative error on
typical pipe-flow operating envelopes).
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrogen import CoolPropMedium


@pytest.fixture(scope="module")
def heos_air():
    return CoolPropMedium("Air", disable_warnings=True)


@pytest.fixture(scope="module")
def bicubic_air():
    return CoolPropMedium("Air", disable_warnings=True, backend="BICUBIC&HEOS")


# Properties exposed via `self.modules` -> their `_batch` counterparts.
_PH_PROPS = (
    "rho_ph", "drho_ph_dp", "drho_ph_dh",
    "mu_ph",  "dmu_ph_dp",  "dmu_ph_dh",
    "T_ph",   "dT_ph_dp",   "dT_ph_dh",
    "s_ph",   "ds_ph_dp",   "ds_ph_dh",
    "k_ph",   "dk_ph_dp",   "dk_ph_dh",
)


@pytest.mark.parametrize("prop", _PH_PROPS)
def test_batch_agrees_with_scalar_heos(heos_air, prop):
    """Bit-exact agreement: batch evaluators must reproduce scalar `eval_*`."""
    scalar = getattr(heos_air, f"eval_{prop}")
    batch = getattr(heos_air, f"eval_{prop}_batch")
    ps = np.linspace(0.5e5, 5e5, 16)
    hs = np.linspace(2.5e5, 5.0e5, 16)
    expected = np.array([scalar(float(p), float(h)) for p, h in zip(ps, hs)])
    got = batch(ps, hs)
    np.testing.assert_array_equal(got, expected)


def test_batch_scalar_fast_path(heos_air):
    """Calling a `_batch` evaluator with scalar args returns a scalar (not array)."""
    rho = heos_air.eval_rho_ph_batch(2e5, 4e5)
    assert isinstance(rho, float)
    assert rho == heos_air.eval_rho_ph(2e5, 4e5)


def test_batch_marker(heos_air):
    """`_hydrogen_vectorised = True` is set so `_vectorise_callable` skips
    re-wrapping batch evaluators (see hydrogen.model._vectorise_callable)."""
    for prop in _PH_PROPS:
        fn = getattr(heos_air, f"eval_{prop}_batch")
        assert getattr(fn, "_hydrogen_vectorised", False), (
            f"eval_{prop}_batch missing _hydrogen_vectorised marker")


def test_modules_default_uses_scalar(heos_air):
    """`self.modules` (the default lambdify namespace) MUST point at the scalar
    evaluators -- the lru_cache hits between templates that share boundary
    nodes (a splitter junction's (p, h) reused by every connected pipe) are
    the dominant CoolProp speedup in HEOS systems and a 3x measured
    regression came from accidentally exposing the batch variants here."""
    # Each module dict has exactly one key -> callable mapping; the callable
    # should be the lru_cache-decorated scalar `eval_*` (i.e. carry
    # `cache_info`), not the un-cached batch wrapper.
    for module in heos_air.modules:
        ((_name, fn),) = module.items()
        assert hasattr(fn, "cache_info"), (
            f"hydrogen.modules entry {_name} should be the scalar lru_cache "
            "wrapper, got something else (batch_modules?)")


def test_backend_kwarg_bicubic_accuracy(heos_air, bicubic_air):
    """BICUBIC tabular backend agrees with HEOS to engineering accuracy
    over typical air operating envelopes."""
    ps = np.linspace(0.5e5, 5e5, 11)
    hs = np.linspace(2.5e5, 5.0e5, 11)
    for p in ps:
        for h in hs:
            rho_h = heos_air.eval_rho_ph(float(p), float(h))
            rho_b = bicubic_air.eval_rho_ph(float(p), float(h))
            T_h = heos_air.eval_T_ph(float(p), float(h))
            T_b = bicubic_air.eval_T_ph(float(p), float(h))
            assert abs(rho_b - rho_h) / rho_h < 1e-3, f"rho mismatch at ({p}, {h})"
            assert abs(T_b - T_h) / T_h < 1e-3, f"T mismatch at ({p}, {h})"


def test_batch_state_pool_reuses_states(heos_air):
    """Repeated batch calls at the same `(p_arr, h_arr)` should hit the LRU
    cache and skip the EOS update.  Easy to verify by counting `set_state_ph`
    misses (the scalar fall-back doesn't run, so its `lru_cache` is untouched)."""
    heos_air._batch_state_cache_ph.clear()
    ps = np.linspace(1e5, 3e5, 8)
    hs = np.linspace(3e5, 4e5, 8)
    rho = heos_air.eval_rho_ph_batch(ps, hs)
    n_after_first = len(heos_air._batch_state_cache_ph)
    mu = heos_air.eval_mu_ph_batch(ps, hs)
    n_after_second = len(heos_air._batch_state_cache_ph)
    # Same `(p, h)` arrays -> single cache entry containing pre-updated states
    assert n_after_first == 1, f"expected 1 cached pool, got {n_after_first}"
    assert n_after_second == 1, "second property call must REUSE states"
    # Sanity: the property values are correct
    expected_rho = np.array([heos_air.eval_rho_ph(float(p), float(h)) for p, h in zip(ps, hs)])
    expected_mu = np.array([heos_air.eval_mu_ph(float(p), float(h)) for p, h in zip(ps, hs)])
    np.testing.assert_array_equal(rho, expected_rho)
    np.testing.assert_array_equal(mu, expected_mu)


def test_batch_modules_have_same_keys_as_scalar(heos_air):
    """`batch_modules` is a drop-in replacement for `modules` (same dict keys)
    so callers can opt in by switching the `aditional_modules=` argument."""
    scalar_keys = {next(iter(d)) for d in heos_air.modules}
    batch_keys = {next(iter(d)) for d in heos_air.batch_modules}
    assert scalar_keys == batch_keys
