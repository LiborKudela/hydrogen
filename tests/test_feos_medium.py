"""FeosMedium tests: drop-in compatibility with CoolPropMedium.

`FeosMedium` is an alternative thermophysical backend (feos for the equation of
state, CoolProp for transport) that must expose the IDENTICAL public surface as
`CoolPropMedium` so the rest of hydrogen can use either interchangeably.  These
tests assert that contract -- the sympy-able property functions, the `modules`
lambdify namespace, `default_vars`, the finite-difference partials, the caches,
and round-tripping through the media (de)serialisation registry.

The whole module is skipped when the optional `feos` / `si-units` packages are
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

feos = pytest.importorskip("feos")
pytest.importorskip("si_units")

from hydrogen import CoolPropMedium, FeosMedium  # noqa: E402
from hydrogen.serialization.registry import make_medium, serialize_medium  # noqa: E402


# The public property functions + their HEM variants every medium must expose.
_PROP_FUNCS = (
    "h_pT", "rho_ph", "T_ph", "s_ph", "k_ph", "mu_ph",
    "rho_ph_hem", "T_ph_hem", "mu_ph_hem", "k_ph_hem",
)


@pytest.fixture(scope="module")
def feos_h2():
    return FeosMedium("Hydrogen", disable_warnings=True)


@pytest.fixture(scope="module")
def cp_h2():
    return CoolPropMedium("Hydrogen", disable_warnings=True)


def test_exposes_coolprop_surface(feos_h2):
    """Every attribute the rest of hydrogen reads off a medium must exist."""
    for name in _PROP_FUNCS:
        assert callable(getattr(feos_h2, name)), name
    for name in ("modules", "batch_modules", "default_vars"):
        assert hasattr(feos_h2, name), name
    assert set(feos_h2.default_vars) == {"p", "T", "h"}
    # Same `eval_*` evaluator set (names shared with CoolPropMedium).
    for name in FeosMedium._SCALAR_EVAL_NAMES + FeosMedium._HEM_EVAL_NAMES:
        assert callable(getattr(feos_h2, name)), name


def test_module_keys_match_coolprop(feos_h2, cp_h2):
    """`modules` (the lambdify namespace) must carry exactly the same keys as
    CoolPropMedium so component equations resolve regardless of backend."""
    feos_keys = {next(iter(d)) for d in feos_h2.modules}
    cp_keys = {next(iter(d)) for d in cp_h2.modules}
    assert feos_keys == cp_keys
    batch_keys = {next(iter(d)) for d in feos_h2.batch_modules}
    assert batch_keys == feos_keys


def test_modules_default_uses_cached_scalars(feos_h2):
    """`modules` must point at the lru_cache-wrapped scalar evaluators (carry
    `cache_info`), matching CoolPropMedium's caching contract."""
    for module in feos_h2.modules:
        ((name, fn),) = module.items()
        assert hasattr(fn, "cache_info"), name


def test_values_are_physically_sane(feos_h2):
    """Gas-phase hydrogen properties should land in physically reasonable
    ranges (feos Peng-Robinson + CoolProp transport)."""
    p, h = 2.0e6, 4.0e6
    rho = feos_h2.eval_rho_ph(p, h)
    T = feos_h2.eval_T_ph(p, h)
    mu = feos_h2.eval_mu_ph(p, h)
    k = feos_h2.eval_k_ph(p, h)
    assert 0.5 < rho < 5.0, rho            # ~1-2 kg/m3 for H2 at 2 MPa, ~280 K
    assert 200.0 < T < 400.0, T
    assert 5e-6 < mu < 2e-5, mu            # H2 viscosity ~9 uPa.s
    assert 0.1 < k < 0.3, k               # H2 conductivity ~0.18 W/m/K


def test_ph_flash_round_trips(feos_h2):
    """h_pT then T_ph must recover the temperature: the feos (p, h) flash is the
    inverse of the (p, T) state."""
    p, T = 3.0e6, 320.0
    h = feos_h2.eval_h_pT(p, T)
    T_back = feos_h2.eval_T_ph(p, h)
    assert abs(T_back - T) < 1e-3, (T, T_back)


def test_fd_partials_match_numeric_gradient(feos_h2):
    """The exposed partials must equal a (coarser) finite-difference gradient of
    the value evaluators -- i.e. the Jacobian entries are self-consistent."""
    p, h = 2.5e6, 3.5e6
    dp, dh = 200.0, 500.0
    drho_dp = (feos_h2.eval_rho_ph(p + dp, h) - feos_h2.eval_rho_ph(p - dp, h)) / (2 * dp)
    drho_dh = (feos_h2.eval_rho_ph(p, h + dh) - feos_h2.eval_rho_ph(p, h - dh)) / (2 * dh)
    assert feos_h2.eval_drho_ph_dp(p, h) == pytest.approx(drho_dp, rel=1e-2)
    assert feos_h2.eval_drho_ph_dh(p, h) == pytest.approx(drho_dh, rel=1e-2)


def test_batch_agrees_with_scalar(feos_h2):
    """Batch (array-aware) module callables must reproduce the scalar values."""
    by_name = {next(iter(d)): list(d.values())[0] for d in feos_h2.modules}
    batch_by_name = {next(iter(d)): list(d.values())[0] for d in feos_h2.batch_modules}
    ps = np.linspace(1e6, 4e6, 6)
    hs = np.linspace(3e6, 5e6, 6)
    name = "Hydrogen_rho_ph"
    expected = np.array([by_name[name](float(p), float(h)) for p, h in zip(ps, hs)])
    got = batch_by_name[name](ps, hs)
    np.testing.assert_allclose(got, expected)


def test_clear_cache(feos_h2):
    feos_h2.clear_cache()
    feos_h2.eval_rho_ph(2e6, 4e6)
    feos_h2.eval_T_ph(2e6, 4e6)
    assert feos_h2.eval_rho_ph.cache_info().currsize == 1
    feos_h2.clear_cache()
    assert feos_h2.eval_rho_ph.cache_info().currsize == 0


def test_transport_none_raises():
    m = FeosMedium("Nitrogen", disable_warnings=True, transport="none")
    with pytest.raises(NotImplementedError):
        m.eval_mu_ph(2e6, 4.5e5)


def test_scalar_cache_maxsize_kwarg():
    m = FeosMedium("Nitrogen", disable_warnings=True, scalar_cache_maxsize=512)
    assert m.eval_rho_ph.cache_info().maxsize == 512
    for name in FeosMedium._SCALAR_EVAL_NAMES:
        assert getattr(m, name).cache_info().maxsize == 512, name


def test_registry_round_trip():
    """`serialize_medium` -> `make_medium` must rebuild a working FeosMedium and
    tag it with provider='feos' (CoolProp specs stay provider-less)."""
    m = FeosMedium("Nitrogen", disable_warnings=True, scalar_cache_maxsize=256)
    spec = serialize_medium(m)
    assert spec["provider"] == "feos"
    assert spec["fluid"] == "Nitrogen"
    rebuilt = make_medium(spec)
    assert isinstance(rebuilt, FeosMedium)
    assert rebuilt.medium == "Nitrogen"
    assert rebuilt.scalar_cache_maxsize == 256
    # Same value after a round trip.
    assert rebuilt.eval_rho_ph(2e6, 4.5e5) == pytest.approx(m.eval_rho_ph(2e6, 4.5e5))


def test_make_medium_defaults_to_coolprop():
    """A spec without a `provider` field must still build a CoolPropMedium."""
    m = make_medium({"fluid": "Air", "disable_warnings": True})
    assert isinstance(m, CoolPropMedium)
