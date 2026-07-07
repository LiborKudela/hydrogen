"""CoolProp-backed thermophysical medium with sympy-friendly property functions."""

from __future__ import annotations

import functools
from collections import OrderedDict, namedtuple

import CoolProp.CoolProp as CP
import numpy as np
import sympy as sp


def get_symbolic_property_function(eval_func, deriv_funcs, args_names, medium_name, function_name=None):
    """Build a `sympy.Function` subclass whose numerical evaluation defers to `eval_func`.

    `deriv_funcs` is a `{argindex (1-based) -> callable}` mapping; sympy uses these to
    compute symbolic derivatives via `fdiff`. The returned class is uniquely named
    `{medium_name}_{function_name}` so it can be registered as a `lambdify` module entry.
    """

    class Symbolic_property(sp.Function):

        @classmethod
        def eval(cls, *args):
            if all(arg.is_number for arg in args):
                return eval_func(*args)

        def fdiff(self, argindex=1):
            if argindex not in deriv_funcs:
                raise NotImplementedError(f"Derivative w.r.t. argument {argindex} not defined")
            wrt = args_names[argindex - 1]
            deriv_class = get_symbolic_property_function(
                deriv_funcs[argindex], {}, args_names, medium_name, f"d{function_name}_d{wrt}"
            )
            return deriv_class(self.args[0], self.args[1])

        def _inv(self, *args):
            print("calling _eval_inverse")
            return super()._inv(*args)

    NewName = type(f'{medium_name}_{function_name}', (Symbolic_property,), {})
    return NewName


class CoolPropMedium:
    """Caches CoolProp `AbstractState` lookups and exposes sympy-able property functions.

    `backend` selects the CoolProp backend used to build every internal
    `AbstractState`.  Defaults to ``"HEOS"`` (full equation-of-state solver,
    bit-exact reference quality) for backwards compatibility.  Set
    ``"BICUBIC&HEOS"`` for the tabular backend: 50-60x faster per property
    evaluation, with relative errors of ~1e-4 (rho), ~1e-5 (T), ~5e-4 (partial
    derivatives) over typical pipe-flow operating envelopes -- engineering
    grade for fluid system simulation.

    `batch_state_pool_size` controls the per-property "states-already-updated-
    at-this-(p,h)-array" cache used by the batch evaluators (`eval_*_ph_batch`).
    A pool size of 8 covers ~all pipe templates that ever appear in
    `pipe_tree`-style systems (each template typically references 2-4 distinct
    `(p, h)` boundary states).

    `scalar_cache_maxsize` controls the per-property `lru_cache` size on each
    scalar `eval_*` evaluator.  Default ``100`` is fine for systems with up to
    ~50 pipe segments (~50 unique `(p, h)` states); bump to ``1000+`` for
    pipe-tree / long-pipe systems where the working set exceeds 100 unique
    states (working set is roughly the number of active variables for HEOS,
    or ~4x that for media whose `eval_dmu_ph_*` / `eval_dk_ph_*` partials
    fall back to finite differences -- those add 4 extra eval points per
    requested partial).  Below the threshold the cache thrashes and HEOS
    scales super-linearly because the same `(p, h)` keeps getting re-computed
    every Newton iteration.
    """

    # Class-level defaults; override per-instance via the `__init__` kwargs.
    scalar_cache_maxsize = 100
    max_array_size = 10
    batch_state_pool_size = 8

    # Names of the scalar `eval_*` evaluators that get wrapped per-instance
    # with a configurable `lru_cache` in `__init__`.  Listed once so the
    # wrapping loop, `clear_cache`, and any future cache-aware diagnostic
    # tooling stay in sync.  `set_state_ph` / `set_state_pT` / `set_state_ps`
    # have their own size-1 caches and are NOT in this list (they are an
    # adjacent micro-optimisation: amortise the EOS update across the
    # several property reads at the SAME `(p, h)` within one expression).
    _SCALAR_EVAL_NAMES = (
        "eval_h_pT",   "eval_dh_pT_dp",  "eval_dh_pT_dT",
        "eval_rho_ph", "eval_drho_ph_dp", "eval_drho_ph_dh",
        "eval_mu_ph",  "eval_dmu_ph_dp",  "eval_dmu_ph_dh",
        "eval_T_ph",   "eval_dT_ph_dp",   "eval_dT_ph_dh",
        "eval_s_ph",   "eval_ds_ph_dp",   "eval_ds_ph_dh",
        "eval_k_ph",   "eval_dk_ph_dp",   "eval_dk_ph_dh",
        "eval_d2rho_ph_dp2", "eval_d2rho_ph_dpdh", "eval_d2rho_ph_dh2",
    )

    # Smooth-HEM property variants (`*_ph_hem`) reuse the single-phase CoolProp
    # VALUE -- which already returns the homogeneous-equilibrium mixture inside
    # the saturation dome and the true single-phase value outside -- but replace
    # the analytic partials with central finite differences.  CoolProp's own
    # two-phase analytic derivatives are inconsistent (they report ~0 while the
    # value moves) and discontinuous at the saturation lines, which is exactly
    # what makes a single-phase Newton solve stall at the dome.  An FD step a bit
    # WIDER than the saturation cliff turns that discontinuous slope jump into a
    # smooth transition, giving a consistent, continuous Jacobian.  The default
    # steps below are tuned for water-scale enthalpies/pressures; override per
    # instance if a medium needs a different smoothing band.
    _HEM_EVAL_NAMES = (
        "eval_drho_ph_hem_dp", "eval_drho_ph_hem_dh",
        "eval_dT_ph_hem_dp",   "eval_dT_ph_hem_dh",
        "eval_dmu_ph_hem_dp",  "eval_dmu_ph_hem_dh",
        "eval_dk_ph_hem_dp",   "eval_dk_ph_hem_dh",
        "eval_d2rho_ph_hem_dp2", "eval_d2rho_ph_hem_dpdh", "eval_d2rho_ph_hem_dh2",
    )

    def __init__(self, medium, p=101325, T=293.15, disable_warnings=False,
                 backend="HEOS", scalar_cache_maxsize=None,
                 hem_fd_dh=5000.0, hem_fd_dp=5000.0):
        self.medium = medium
        self.backend = backend
        # Finite-difference smoothing bands for the HEM property partials
        # (J/kg and Pa respectively).  See `_HEM_EVAL_NAMES` note above.
        self.hem_fd_dh = float(hem_fd_dh)
        self.hem_fd_dp = float(hem_fd_dp)
        self.abstarct_state_ph = CP.AbstractState(backend, self.medium)
        self.abstarct_state_pT = CP.AbstractState(backend, self.medium)
        self.abstarct_state_ps = CP.AbstractState(backend, self.medium)
        self.disable_warnings = disable_warnings

        # Per-instance cache size override -- falls back to the class
        # attribute (default 100), so users can either set the kwarg here
        # or set `CoolPropMedium.scalar_cache_maxsize = N` once at import
        # time to change the default for ALL future instances.  Wrap each
        # scalar `eval_*` bound method with its own `lru_cache`; subsequent
        # `self.eval_X_ph(p, h)` calls dispatch to the wrapper because
        # instance attributes shadow class-level methods.
        if scalar_cache_maxsize is not None:
            self.scalar_cache_maxsize = scalar_cache_maxsize
        for _name in self._SCALAR_EVAL_NAMES + self._HEM_EVAL_NAMES:
            _bound = getattr(self, _name)
            setattr(self, _name,
                    functools.lru_cache(maxsize=self.scalar_cache_maxsize)(_bound))

        # --- Batch state pool: a small LRU of `(p_arr.tobytes(), h_arr.tobytes())`
        # -> list-of-AbstractStates that have ALREADY been `update()`-d to the
        # corresponding `(p[i], h[i])` pair.  Reused across consecutive batch
        # property calls so the EOS update is paid ONCE per distinct (p, h)
        # array, not once per (property, instance).  Without this cache the
        # vectorised template path would do 6 (properties) x 6 (derivatives) =
        # 12x as much EOS work as the per-instance scalar path which has the
        # `set_state_ph(p, h)` size-1 cache.
        self._batch_state_cache_ph = OrderedDict()
        # Free pool of AbstractState objects to reuse (avoid the ~50-100 us
        # construction cost per state).  Grows on demand and is never shrunk.
        self._batch_state_free_ph = []

        self.h, self.p, self.T = sp.symbols('h p T', real=True)
        self.h_pT = get_symbolic_property_function(self.eval_h_pT,    {1: self.eval_dh_pT_dp,  2: self.eval_dh_pT_dT},  ["p", "T"], medium, "h_pT")
        self.rho_ph = get_symbolic_property_function(self.eval_rho_ph, {1: self.eval_drho_ph_dp, 2: self.eval_drho_ph_dh}, ["p", "h"], medium, "rho_ph")
        self.mu_ph = get_symbolic_property_function(self.eval_mu_ph,  {1: self.eval_dmu_ph_dp,  2: self.eval_dmu_ph_dh},  ["p", "h"], medium, "mu_ph")
        self.T_ph = get_symbolic_property_function(self.eval_T_ph,    {1: self.eval_dT_ph_dp,   2: self.eval_dT_ph_dh},   ["p", "h"], medium, "T_ph")
        self.s_ph = get_symbolic_property_function(self.eval_s_ph,    {1: self.eval_ds_ph_dp,   2: self.eval_ds_ph_dh},   ["p", "h"], medium, "s_ph")
        self.k_ph = get_symbolic_property_function(self.eval_k_ph,    {1: self.eval_dk_ph_dp,   2: self.eval_dk_ph_dh},   ["p", "h"], medium, "k_ph")

        # Smooth-HEM variants: SAME value evaluator as single-phase (CoolProp's
        # (p, h) flash already gives the HEM mixture inside the dome), but with
        # the smoothed finite-difference partials so the Jacobian stays
        # consistent and continuous through the saturation lines.  Used by
        # fluid components built with `multiphase="HEM"`.
        self.rho_ph_hem = get_symbolic_property_function(self.eval_rho_ph, {1: self.eval_drho_ph_hem_dp, 2: self.eval_drho_ph_hem_dh}, ["p", "h"], medium, "rho_ph_hem")
        self.T_ph_hem   = get_symbolic_property_function(self.eval_T_ph,   {1: self.eval_dT_ph_hem_dp,   2: self.eval_dT_ph_hem_dh},   ["p", "h"], medium, "T_ph_hem")
        self.mu_ph_hem  = get_symbolic_property_function(self.eval_mu_ph,  {1: self.eval_dmu_ph_hem_dp,  2: self.eval_dmu_ph_hem_dh},  ["p", "h"], medium, "mu_ph_hem")
        self.k_ph_hem   = get_symbolic_property_function(self.eval_k_ph,   {1: self.eval_dk_ph_hem_dp,   2: self.eval_dk_ph_hem_dh},   ["p", "h"], medium, "k_ph_hem")

        # `drho/dp` as first-class symbolic functions (single-phase + HEM) so a
        # component can put the true isothermal-ish compressibility into a
        # residual and still get a consistent Jacobian.  Consumed by the
        # SegmentedChannel `compressible`-level stabilisation gate.
        self.drho_ph_dp = get_symbolic_property_function(self.eval_drho_ph_dp, {1: self.eval_d2rho_ph_dp2, 2: self.eval_d2rho_ph_dpdh}, ["p", "h"], medium, "drho_ph_dp")
        self.drho_ph_hem_dp = get_symbolic_property_function(self.eval_drho_ph_hem_dp, {1: self.eval_d2rho_ph_hem_dp2, 2: self.eval_d2rho_ph_hem_dpdh}, ["p", "h"], medium, "drho_ph_hem_dp")
        # `drho/dh` as a first-class symbolic function too (single-phase + HEM),
        # with its own consistent second derivatives.  The primitive `(p, h)`
        # dynamic levels need BOTH `rho_p` and `rho_h` in the mass / energy cell
        # balances, and lambdifying the Newton Jacobian differentiates them once
        # more -- so `drho/dh` must expose `d2rho/dhdp` and `d2rho/dh2`.
        self.drho_ph_dh = get_symbolic_property_function(self.eval_drho_ph_dh, {1: self.eval_d2rho_ph_dpdh, 2: self.eval_d2rho_ph_dh2}, ["p", "h"], medium, "drho_ph_dh")
        self.drho_ph_hem_dh = get_symbolic_property_function(self.eval_drho_ph_hem_dh, {1: self.eval_d2rho_ph_hem_dpdh, 2: self.eval_d2rho_ph_hem_dh2}, ["p", "h"], medium, "drho_ph_hem_dh")

        self.default_vars = {'p': p, 'T': T, 'h': self.h_pT(p, T)}
        # `self.modules` exposes the SCALAR `eval_*_ph` functions to
        # `sympy.lambdify`.  Each scalar evaluator carries an
        # `lru_cache(maxsize=100)`, which catches cross-template (p, h)
        # sharing (e.g. a splitter junction's (p, h) is reused by every
        # pipe edge meeting at it -- typical for tree-/network-shaped
        # systems).  In the vectorised template path (`model.py`'s
        # `_eval_per_template`), `_vectorise_callable` wraps these in a
        # Python loop that preserves the scalar lru_cache benefit.
        # Profiling pipe_tree(N=4) showed batch evaluators were ~3x
        # slower with HEOS because they DEFEAT cross-template caching:
        # each template's batch (p_arr, h_arr) is independent, so
        # boundary nodes shared between templates get re-computed once
        # per template instead of once globally.
        self.modules = [
            {f"{medium}_h_pT":   self.eval_h_pT},   {f"{medium}_dh_pT_dp":  self.eval_dh_pT_dp},  {f"{medium}_dh_pT_dT":  self.eval_dh_pT_dT},
            {f"{medium}_rho_ph": self.eval_rho_ph}, {f"{medium}_drho_ph_dp": self.eval_drho_ph_dp}, {f"{medium}_drho_ph_dh": self.eval_drho_ph_dh},
            {f"{medium}_mu_ph":  self.eval_mu_ph},  {f"{medium}_dmu_ph_dp":  self.eval_dmu_ph_dp},  {f"{medium}_dmu_ph_dh":  self.eval_dmu_ph_dh},
            {f"{medium}_T_ph":   self.eval_T_ph},   {f"{medium}_dT_ph_dp":   self.eval_dT_ph_dp},   {f"{medium}_dT_ph_dh":   self.eval_dT_ph_dh},
            {f"{medium}_s_ph":   self.eval_s_ph},   {f"{medium}_ds_ph_dp":   self.eval_ds_ph_dp},   {f"{medium}_ds_ph_dh":   self.eval_ds_ph_dh},
            {f"{medium}_k_ph":   self.eval_k_ph},   {f"{medium}_dk_ph_dp":   self.eval_dk_ph_dp},   {f"{medium}_dk_ph_dh":   self.eval_dk_ph_dh},
            # Second derivatives of rho (single-phase): the `_dp_*` pair
            # linearises `drho/dp`, the `_dh_*` pair linearises `drho/dh` -- both
            # needed for the Jacobian of the primitive `(p, h)` cell balances.
            {f"{medium}_ddrho_ph_dp_dp": self.eval_d2rho_ph_dp2}, {f"{medium}_ddrho_ph_dp_dh": self.eval_d2rho_ph_dpdh},
            {f"{medium}_ddrho_ph_dh_dp": self.eval_d2rho_ph_dpdh}, {f"{medium}_ddrho_ph_dh_dh": self.eval_d2rho_ph_dh2},
            # HEM variants: value reuses the single-phase evaluator; partials are
            # the smoothed finite-difference ones registered below.
            {f"{medium}_rho_ph_hem": self.eval_rho_ph}, {f"{medium}_drho_ph_hem_dp": self.eval_drho_ph_hem_dp}, {f"{medium}_drho_ph_hem_dh": self.eval_drho_ph_hem_dh},
            {f"{medium}_T_ph_hem":   self.eval_T_ph},   {f"{medium}_dT_ph_hem_dp":   self.eval_dT_ph_hem_dp},   {f"{medium}_dT_ph_hem_dh":   self.eval_dT_ph_hem_dh},
            {f"{medium}_mu_ph_hem":  self.eval_mu_ph},  {f"{medium}_dmu_ph_hem_dp":  self.eval_dmu_ph_hem_dp},  {f"{medium}_dmu_ph_hem_dh":  self.eval_dmu_ph_hem_dh},
            {f"{medium}_k_ph_hem":   self.eval_k_ph},   {f"{medium}_dk_ph_hem_dp":   self.eval_dk_ph_hem_dp},   {f"{medium}_dk_ph_hem_dh":   self.eval_dk_ph_hem_dh},
            {f"{medium}_ddrho_ph_hem_dp_dp": self.eval_d2rho_ph_hem_dp2}, {f"{medium}_ddrho_ph_hem_dp_dh": self.eval_d2rho_ph_hem_dpdh},
            {f"{medium}_ddrho_ph_hem_dh_dp": self.eval_d2rho_ph_hem_dpdh}, {f"{medium}_ddrho_ph_hem_dh_dh": self.eval_d2rho_ph_hem_dh2},
        ]
        # `self.batch_modules` exposes the batch-aware (`numpy-array`-friendly)
        # variants for users who manually build models that benefit from
        # batched property evaluation -- e.g. a single template with many
        # instances and zero cross-template aliasing.  Not used by default
        # because the lru_cache cross-template savings dominate for HEOS.
        self.batch_modules = [
            {f"{medium}_h_pT":   self.eval_h_pT_batch},   {f"{medium}_dh_pT_dp":  self.eval_dh_pT_dp_batch},  {f"{medium}_dh_pT_dT":  self.eval_dh_pT_dT_batch},
            {f"{medium}_rho_ph": self.eval_rho_ph_batch}, {f"{medium}_drho_ph_dp": self.eval_drho_ph_dp_batch}, {f"{medium}_drho_ph_dh": self.eval_drho_ph_dh_batch},
            {f"{medium}_mu_ph":  self.eval_mu_ph_batch},  {f"{medium}_dmu_ph_dp":  self.eval_dmu_ph_dp_batch},  {f"{medium}_dmu_ph_dh":  self.eval_dmu_ph_dh_batch},
            {f"{medium}_T_ph":   self.eval_T_ph_batch},   {f"{medium}_dT_ph_dp":   self.eval_dT_ph_dp_batch},   {f"{medium}_dT_ph_dh":   self.eval_dT_ph_dh_batch},
            {f"{medium}_s_ph":   self.eval_s_ph_batch},   {f"{medium}_ds_ph_dp":   self.eval_ds_ph_dp_batch},   {f"{medium}_ds_ph_dh":   self.eval_ds_ph_dh_batch},
            {f"{medium}_k_ph":   self.eval_k_ph_batch},   {f"{medium}_dk_ph_dp":   self.eval_dk_ph_dp_batch},   {f"{medium}_dk_ph_dh":   self.eval_dk_ph_dh_batch},
            {f"{medium}_ddrho_ph_dp_dp": self.eval_d2rho_ph_dp2_batch}, {f"{medium}_ddrho_ph_dp_dh": self.eval_d2rho_ph_dpdh_batch},
            {f"{medium}_ddrho_ph_dh_dp": self.eval_d2rho_ph_dpdh_batch}, {f"{medium}_ddrho_ph_dh_dh": self.eval_d2rho_ph_dh2_batch},
            # HEM variants (batch): value reuses the single-phase batch evaluator,
            # partials are the smoothed central differences driven through it.
            {f"{medium}_rho_ph_hem": self.eval_rho_ph_batch}, {f"{medium}_drho_ph_hem_dp": self.eval_drho_ph_hem_dp_batch}, {f"{medium}_drho_ph_hem_dh": self.eval_drho_ph_hem_dh_batch},
            {f"{medium}_T_ph_hem":   self.eval_T_ph_batch},   {f"{medium}_dT_ph_hem_dp":   self.eval_dT_ph_hem_dp_batch},   {f"{medium}_dT_ph_hem_dh":   self.eval_dT_ph_hem_dh_batch},
            {f"{medium}_mu_ph_hem":  self.eval_mu_ph_batch},  {f"{medium}_dmu_ph_hem_dp":  self.eval_dmu_ph_hem_dp_batch},  {f"{medium}_dmu_ph_hem_dh":  self.eval_dmu_ph_hem_dh_batch},
            {f"{medium}_k_ph_hem":   self.eval_k_ph_batch},   {f"{medium}_dk_ph_hem_dp":   self.eval_dk_ph_hem_dp_batch},   {f"{medium}_dk_ph_hem_dh":   self.eval_dk_ph_hem_dh_batch},
            {f"{medium}_ddrho_ph_hem_dp_dp": self.eval_d2rho_ph_hem_dp2_batch}, {f"{medium}_ddrho_ph_hem_dp_dh": self.eval_d2rho_ph_hem_dpdh_batch},
            {f"{medium}_ddrho_ph_hem_dh_dp": self.eval_d2rho_ph_hem_dpdh_batch}, {f"{medium}_ddrho_ph_hem_dh_dh": self.eval_d2rho_ph_hem_dh2_batch},
        ]

    @functools.lru_cache(maxsize=1)
    def set_state_ph(self, p, h):
        self.abstarct_state_ph.update(CP.HmassP_INPUTS, h, p)

    @functools.lru_cache(maxsize=1)
    def set_state_pT(self, p, T):
        self.abstarct_state_pT.update(CP.PT_INPUTS, p, T)

    @functools.lru_cache(maxsize=1)
    def set_state_ps(self, p, s):
        self.abstarct_state_ps.update(CP.PSmass_INPUTS, p, s)

    # --- batch infrastructure ---------------------------------------------------------
    #
    # The (p, h) batch evaluators below all share a small LRU pool of
    # `AbstractState` objects.  When a batch property function is called with
    # `(p_arr, h_arr)`, `_ensure_states_ph` returns a list of `n = p_arr.size`
    # `AbstractState`s, each pre-`update()`-d to `(p[i], h[i])`.  Subsequent
    # batch calls at the *same* `(p_arr, h_arr)` reuse those states for free
    # -- crucial because a single CSE'd template lambda typically evaluates
    # rho/mu/T/k all at the same boundary `(p, h)` pair, and we want the
    # expensive EOS update paid ONCE per pair, not once per (property, instance).
    # The size-1 `set_state_ph` lru_cache provides the same amortisation in the
    # SCALAR per-instance path; this is its array-aware counterpart.

    @staticmethod
    def _is_array(x):
        return isinstance(x, np.ndarray) and x.ndim > 0

    def _ensure_states_ph(self, p_arr, h_arr):
        """Return a tuple of `len(p_arr)` AbstractStates updated to `(p[i], h[i])`.

        Cache hit: we already saw this exact `(p, h)` byte pattern -> return
        the cached states without re-running CoolProp.
        Cache miss: pull `n` states from the free pool (growing it on demand),
        run `update()` on each, store the tuple in the LRU cache.  When the
        cache evicts an entry, its states are returned to the free pool.
        """
        n = p_arr.size
        key = (p_arr.tobytes(), h_arr.tobytes())
        cached = self._batch_state_cache_ph.get(key)
        if cached is not None and len(cached) == n:
            self._batch_state_cache_ph.move_to_end(key)
            return cached
        # Allocate `n` states from the free pool, growing as needed.
        free = self._batch_state_free_ph
        while len(free) < n:
            free.append(CP.AbstractState(self.backend, self.medium))
        states = tuple(free[i] for i in range(n))
        del free[:n]
        p_flat = p_arr.ravel()
        h_flat = h_arr.ravel()
        for i in range(n):
            states[i].update(CP.HmassP_INPUTS, float(h_flat[i]), float(p_flat[i]))
        self._batch_state_cache_ph[key] = states
        if len(self._batch_state_cache_ph) > self.batch_state_pool_size:
            _, evicted = self._batch_state_cache_ph.popitem(last=False)
            free.extend(evicted)
        return states

    def _ensure_states_pT(self, p_arr, T_arr):
        """`_ensure_states_ph` analogue for `(p, T)` inputs.

        Used by `eval_h_pT_batch` and friends -- these are typically only
        called during initialisation (warm-start), so we don't even bother
        with a separate cache: we just blow through `set_state_pT` per
        element.  If a future hot path uses `h_pT` heavily, mirror the
        `_ensure_states_ph` design here.
        """
        return None  # see comment; per-element scalar fallback is fine

    def _batch_eval_ph(self, p, h, scalar_method, getter):
        """Apply `getter(state)` to every element of a `(p, h)` batch input.

        Falls back to `scalar_method(p, h)` when args are scalars (the
        per-instance Newton path) OR when an individual `getter(state)`
        raises (the same finite-difference fallback the scalar `eval_*`
        functions use for unsupported partial derivatives).
        """
        if not (self._is_array(p) or self._is_array(h)):
            return scalar_method(p, h)
        shape = np.broadcast_shapes(np.shape(p), np.shape(h))
        p_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(p, dtype=float), shape))
        h_arr = np.ascontiguousarray(np.broadcast_to(np.asarray(h, dtype=float), shape))
        states = self._ensure_states_ph(p_arr, h_arr)
        out = np.empty(p_arr.size, dtype=float)
        try:
            for i in range(p_arr.size):
                out[i] = getter(states[i])
        except Exception:
            # Per-element fallback (typically only reached for unsupported
            # mu/k partial derivatives -- mu's `first_partial_deriv` doesn't
            # always work in CoolProp, and the scalar `eval_dmu_*` already
            # handles the finite-difference fallback).
            for i in range(p_arr.size):
                try:
                    out[i] = getter(states[i])
                except Exception:
                    out[i] = scalar_method(float(p_arr.flat[i]), float(h_arr.flat[i]))
        return out.reshape(p_arr.shape)

    # --- enthalpy h(p, T) -------------------------------------------------------------
    #
    # Note: the scalar `eval_*` evaluators below are NOT decorated with
    # `@functools.lru_cache` -- the per-instance wrapping in `__init__`
    # handles that, picking up `self.scalar_cache_maxsize` at instance
    # creation time so the cache size is configurable.

    def eval_h_pT(self, p, T):
        if p == 0 or T == 0:
            return None
        self.set_state_pT(p, T)
        return self.abstarct_state_pT.hmass()

    def eval_dh_pT_dp(self, p, T):
        self.set_state_pT(p, T)
        return self.abstarct_state_pT.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)

    def eval_dh_pT_dT(self, p, T):
        self.set_state_pT(p, T)
        return self.abstarct_state_pT.first_partial_deriv(CP.iHmass, CP.iT, CP.iP)

    # --- density rho(p, h) ------------------------------------------------------------

    def eval_rho_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.rhomass()

    def eval_drho_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)

    def eval_drho_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)

    # Second derivatives of rho(p, h), used *only* to give the compressible
    # channel's adaptive artificial-compressibility gate a consistent Jacobian
    # (the gate keys on `drho/dp`, so its linearisation needs `d2rho/dp2` and
    # `d2rho/dpdh`).  Analytic where CoolProp supports it; central-difference of
    # the first partial otherwise.
    def eval_d2rho_ph_dp2(self, p, h):
        self.set_state_ph(p, h)
        try:
            return self.abstarct_state_ph.second_partial_deriv(
                CP.iDmass, CP.iP, CP.iHmass, CP.iP, CP.iHmass)
        except Exception:
            e = self.hem_fd_dp
            p_hi = p + e
            p_lo = np.maximum(p - e, e)
            return (self.eval_drho_ph_dp(p_hi, h)
                    - self.eval_drho_ph_dp(p_lo, h)) / (p_hi - p_lo)

    def eval_d2rho_ph_dpdh(self, p, h):
        self.set_state_ph(p, h)
        try:
            return self.abstarct_state_ph.second_partial_deriv(
                CP.iDmass, CP.iP, CP.iHmass, CP.iHmass, CP.iP)
        except Exception:
            e = self.hem_fd_dh
            return (self.eval_drho_ph_dp(p, h + e)
                    - self.eval_drho_ph_dp(p, h - e)) / (2.0 * e)

    def eval_d2rho_ph_dh2(self, p, h):
        self.set_state_ph(p, h)
        try:
            return self.abstarct_state_ph.second_partial_deriv(
                CP.iDmass, CP.iHmass, CP.iP, CP.iHmass, CP.iP)
        except Exception:
            e = self.hem_fd_dh
            return (self.eval_drho_ph_dh(p, h + e)
                    - self.eval_drho_ph_dh(p, h - e)) / (2.0 * e)

    # --- viscosity mu(p, h) -----------------------------------------------------------

    def eval_mu_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.viscosity()

    def eval_dmu_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        try:
            dmu_dp = self.abstarct_state_ph.first_partial_deriv(CP.iviscosity, CP.iP, CP.iHmass)
        except Exception:
            if not self.disable_warnings:
                print("Warning: partial derivative of mu_ph w.r.t. p failed, using finite difference instead")
            eps = 1e-3
            dmu_dp = (self.eval_mu_ph(p + eps, h) - self.eval_mu_ph(p - eps, h)) / (2 * eps)
        return dmu_dp

    def eval_dmu_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        try:
            dmu_dh = self.abstarct_state_ph.first_partial_deriv(CP.iviscosity, CP.iHmass, CP.iP)
        except Exception:
            if not self.disable_warnings:
                print("Warning: partial derivative of mu_ph w.r.t. h failed, using finite difference instead")
            eps = 1e-3
            dmu_dh = (self.eval_mu_ph(p, h + eps) - self.eval_mu_ph(p, h - eps)) / (2 * eps)
        return dmu_dh

    # --- temperature T(p, h) ----------------------------------------------------------

    def eval_T_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.T()

    def eval_dT_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iT, CP.iP, CP.iHmass)

    def eval_dT_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iT, CP.iHmass, CP.iP)

    # --- entropy s(p, h) --------------------------------------------------------------

    def eval_s_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.smass()

    def eval_ds_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iSmass, CP.iP, CP.iHmass)

    def eval_ds_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iSmass, CP.iHmass, CP.iP)

    # --- thermal conductivity k(p, h) -------------------------------------------------

    def eval_k_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.conductivity()

    def eval_dk_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        try:
            dk_dp = self.abstarct_state_ph.first_partial_deriv(CP.iconductivity, CP.iP, CP.iHmass)
        except Exception:
            if not self.disable_warnings:
                print("Warning: partial derivative of k_ph w.r.t. p failed, using finite difference instead")
            eps = 1e-3
            dk_dp = (self.eval_k_ph(p + eps, h) - self.eval_k_ph(p - eps, h)) / (2 * eps)
        return dk_dp

    def eval_dk_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        try:
            dk_dh = self.abstarct_state_ph.first_partial_deriv(CP.iconductivity, CP.iHmass, CP.iP)
        except Exception:
            if not self.disable_warnings:
                print("Warning: partial derivative of k_ph w.r.t. h failed, using finite difference instead")
            eps = 1e-3
            dk_dh = (self.eval_k_ph(p, h + eps) - self.eval_k_ph(p, h - eps)) / (2 * eps)
        return dk_dh

    # --- smooth-HEM partials (central finite differences) -----------------------------
    #
    # These back the `*_ph_hem` symbolic property functions.  Each is a central
    # difference of the SINGLE-PHASE value evaluator with a step (`hem_fd_dh` /
    # `hem_fd_dp`) deliberately a little wider than the saturation cliff, so the
    # otherwise-discontinuous slope at the phase boundary is smeared into a
    # smooth, consistent transition.  Inside the dome the central difference
    # recovers the genuine HEM slope (e.g. drho/dh = -rho^2 (vg-vf)/(hg-hf));
    # outside it recovers the single-phase slope.  Pressures are floored to keep
    # `p - dp` inside the EOS's valid domain.

    def _fd_dh(self, eval_func, p, h):
        e = self.hem_fd_dh
        return (eval_func(p, h + e) - eval_func(p, h - e)) / (2.0 * e)

    def _fd_dp(self, eval_func, p, h):
        e = self.hem_fd_dp
        p_hi = p + e
        # Keep the low sample strictly positive; `np.maximum` so the same
        # helper serves both the scalar and the array (batch) callers.
        p_lo = np.maximum(p - e, e)
        return (eval_func(p_hi, h) - eval_func(p_lo, h)) / (p_hi - p_lo)

    def eval_drho_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_rho_ph, p, h)

    def eval_drho_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_rho_ph, p, h)

    def eval_dT_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_T_ph, p, h)

    def eval_dT_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_T_ph, p, h)

    def eval_dmu_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_mu_ph, p, h)

    def eval_dmu_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_mu_ph, p, h)

    def eval_dk_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_k_ph, p, h)

    def eval_dk_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_k_ph, p, h)

    # Second derivatives of the smooth-HEM `drho/dp` (central difference of the
    # already-smoothed first partial).  Only the compressible channel's
    # stabilisation gate consumes these; the FD-of-FD noise is harmless there
    # because the gate is a smooth, bounded conditioning term.
    def eval_d2rho_ph_hem_dp2(self, p, h):
        return self._fd_dp(self.eval_drho_ph_hem_dp, p, h)

    def eval_d2rho_ph_hem_dpdh(self, p, h):
        return self._fd_dh(self.eval_drho_ph_hem_dp, p, h)

    def eval_d2rho_ph_hem_dh2(self, p, h):
        return self._fd_dh(self.eval_drho_ph_hem_dh, p, h)

    # Batch (array-aware) counterparts: the value reuses `eval_*_ph_batch`
    # (already vectorised), and the partial is the same central difference but
    # driven through the batch value evaluator so `(p, h)` arrays work.
    def eval_drho_ph_hem_dp_batch(self, p, h):
        return self._fd_dp(self.eval_rho_ph_batch, p, h)

    def eval_drho_ph_hem_dh_batch(self, p, h):
        return self._fd_dh(self.eval_rho_ph_batch, p, h)

    def eval_dT_ph_hem_dp_batch(self, p, h):
        return self._fd_dp(self.eval_T_ph_batch, p, h)

    def eval_dT_ph_hem_dh_batch(self, p, h):
        return self._fd_dh(self.eval_T_ph_batch, p, h)

    def eval_dmu_ph_hem_dp_batch(self, p, h):
        return self._fd_dp(self.eval_mu_ph_batch, p, h)

    def eval_dmu_ph_hem_dh_batch(self, p, h):
        return self._fd_dh(self.eval_mu_ph_batch, p, h)

    def eval_dk_ph_hem_dp_batch(self, p, h):
        return self._fd_dp(self.eval_k_ph_batch, p, h)

    def eval_dk_ph_hem_dh_batch(self, p, h):
        return self._fd_dh(self.eval_k_ph_batch, p, h)

    # Batch second derivatives of rho(p, h) for the stabilisation gate.
    def eval_d2rho_ph_dp2_batch(self, p, h):
        return self._fd_dp(self.eval_drho_ph_dp_batch, p, h)

    def eval_d2rho_ph_dpdh_batch(self, p, h):
        return self._fd_dh(self.eval_drho_ph_dp_batch, p, h)

    def eval_d2rho_ph_dh2_batch(self, p, h):
        return self._fd_dh(self.eval_drho_ph_dh_batch, p, h)

    def eval_d2rho_ph_hem_dp2_batch(self, p, h):
        return self._fd_dp(self.eval_drho_ph_hem_dp_batch, p, h)

    def eval_d2rho_ph_hem_dpdh_batch(self, p, h):
        return self._fd_dh(self.eval_drho_ph_hem_dp_batch, p, h)

    def eval_d2rho_ph_hem_dh2_batch(self, p, h):
        return self._fd_dh(self.eval_drho_ph_hem_dh_batch, p, h)

    # --- batch (array-aware) variants -------------------------------------------------
    #
    # Each `eval_*_batch` accepts EITHER scalar `(p, h)` (delegates to the
    # scalar `eval_*` with its own `lru_cache`) OR numpy arrays (uses the
    # pooled `_ensure_states_ph`).  Marked `_hydrogen_vectorised = True` so
    # `model.py`'s post-hoc `_vectorise_callable` patch leaves them alone --
    # they already handle arrays natively, no Python-loop wrapper needed.

    def eval_h_pT_batch(self, p, T):
        # `(p, T)` callbacks aren't on the hot path (only used at warm-start),
        # so we skip the batch state pool and just loop the scalar version.
        if not (self._is_array(p) or self._is_array(T)):
            return self.eval_h_pT(p, T)
        shape = np.broadcast_shapes(np.shape(p), np.shape(T))
        p_arr = np.broadcast_to(np.asarray(p, dtype=float), shape).ravel()
        T_arr = np.broadcast_to(np.asarray(T, dtype=float), shape).ravel()
        return np.fromiter(
            (self.eval_h_pT(float(p_arr[i]), float(T_arr[i])) for i in range(p_arr.size)),
            dtype=float, count=p_arr.size,
        ).reshape(shape)

    def eval_dh_pT_dp_batch(self, p, T):
        if not (self._is_array(p) or self._is_array(T)):
            return self.eval_dh_pT_dp(p, T)
        shape = np.broadcast_shapes(np.shape(p), np.shape(T))
        p_arr = np.broadcast_to(np.asarray(p, dtype=float), shape).ravel()
        T_arr = np.broadcast_to(np.asarray(T, dtype=float), shape).ravel()
        return np.fromiter(
            (self.eval_dh_pT_dp(float(p_arr[i]), float(T_arr[i])) for i in range(p_arr.size)),
            dtype=float, count=p_arr.size,
        ).reshape(shape)

    def eval_dh_pT_dT_batch(self, p, T):
        if not (self._is_array(p) or self._is_array(T)):
            return self.eval_dh_pT_dT(p, T)
        shape = np.broadcast_shapes(np.shape(p), np.shape(T))
        p_arr = np.broadcast_to(np.asarray(p, dtype=float), shape).ravel()
        T_arr = np.broadcast_to(np.asarray(T, dtype=float), shape).ravel()
        return np.fromiter(
            (self.eval_dh_pT_dT(float(p_arr[i]), float(T_arr[i])) for i in range(p_arr.size)),
            dtype=float, count=p_arr.size,
        ).reshape(shape)

    def eval_rho_ph_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_rho_ph,
                                   lambda s: s.rhomass())

    def eval_drho_ph_dp_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_drho_ph_dp,
                                   lambda s: s.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass))

    def eval_drho_ph_dh_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_drho_ph_dh,
                                   lambda s: s.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP))

    def eval_mu_ph_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_mu_ph,
                                   lambda s: s.viscosity())

    def eval_dmu_ph_dp_batch(self, p, h):
        # Note: `first_partial_deriv(iviscosity, ...)` is unsupported for some
        # CoolProp media; `_batch_eval_ph` falls back to the scalar
        # `eval_dmu_ph_dp` (which uses finite differences) on a per-element
        # basis when the batch getter raises.
        return self._batch_eval_ph(p, h, self.eval_dmu_ph_dp,
                                   lambda s: s.first_partial_deriv(CP.iviscosity, CP.iP, CP.iHmass))

    def eval_dmu_ph_dh_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_dmu_ph_dh,
                                   lambda s: s.first_partial_deriv(CP.iviscosity, CP.iHmass, CP.iP))

    def eval_T_ph_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_T_ph,
                                   lambda s: s.T())

    def eval_dT_ph_dp_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_dT_ph_dp,
                                   lambda s: s.first_partial_deriv(CP.iT, CP.iP, CP.iHmass))

    def eval_dT_ph_dh_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_dT_ph_dh,
                                   lambda s: s.first_partial_deriv(CP.iT, CP.iHmass, CP.iP))

    def eval_s_ph_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_s_ph,
                                   lambda s: s.smass())

    def eval_ds_ph_dp_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_ds_ph_dp,
                                   lambda s: s.first_partial_deriv(CP.iSmass, CP.iP, CP.iHmass))

    def eval_ds_ph_dh_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_ds_ph_dh,
                                   lambda s: s.first_partial_deriv(CP.iSmass, CP.iHmass, CP.iP))

    def eval_k_ph_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_k_ph,
                                   lambda s: s.conductivity())

    def eval_dk_ph_dp_batch(self, p, h):
        # Same `iconductivity` partial-deriv caveat as `dmu_ph_*`.
        return self._batch_eval_ph(p, h, self.eval_dk_ph_dp,
                                   lambda s: s.first_partial_deriv(CP.iconductivity, CP.iP, CP.iHmass))

    def eval_dk_ph_dh_batch(self, p, h):
        return self._batch_eval_ph(p, h, self.eval_dk_ph_dh,
                                   lambda s: s.first_partial_deriv(CP.iconductivity, CP.iHmass, CP.iP))

    # --- saturation-line sampling (consumed by `TabulatedMedium`) ----------------------

    def sample_saturation(self, p_array):
        """Saturation-line quantities at each pressure of `p_array` (Q=0/Q=1
        flashes).  Returns a dict of arrays: ``h_l, h_v, T_sat, rho_l, rho_v,
        mu_l, mu_v, k_l, k_v, s_l, s_v``.

        This is the sampling protocol hook consumed by `TabulatedMedium` when
        its window intersects the two-phase dome.  Pressures at or above the
        critical pressure yield NaN (there is no saturation line there);
        transport properties that CoolProp cannot evaluate on the saturation
        boundary also yield NaN (the table builder interpolates over them).
        """
        p_array = np.asarray(p_array, dtype=float)
        try:
            st = CP.AbstractState(self.backend, self.medium)
            st.p_critical()
        except Exception:
            # Tabular backends can lack PQ flashes / critical-state queries.
            st = CP.AbstractState("HEOS", self.medium)
        p_crit = st.p_critical()
        keys = ("h_l", "h_v", "T_sat", "rho_l", "rho_v",
                "mu_l", "mu_v", "k_l", "k_v", "s_l", "s_v")
        out = {k: np.full(p_array.shape, np.nan) for k in keys}
        for i, p in enumerate(p_array.ravel()):
            if not (0.0 < p < p_crit):
                continue
            for q, tag in ((0.0, "l"), (1.0, "v")):
                st.update(CP.PQ_INPUTS, float(p), q)
                out[f"h_{tag}"].flat[i] = st.hmass()
                out[f"rho_{tag}"].flat[i] = st.rhomass()
                out[f"s_{tag}"].flat[i] = st.smass()
                if tag == "l":
                    out["T_sat"].flat[i] = st.T()
                try:
                    out[f"mu_{tag}"].flat[i] = st.viscosity()
                    out[f"k_{tag}"].flat[i] = st.conductivity()
                except Exception:
                    pass  # left NaN; interpolated by the table builder
        return out

    # --- introspection helpers --------------------------------------------------------

    def get_default_vars(self):
        return self.default_vars

    def get_lru_chache_info_str(self, func, indent=0):
        name = func.__name__
        hits = func.cache_info().hits
        misses = func.cache_info().misses
        calls = hits + misses
        if calls > 0:
            return (
                f"{' ' * indent}{name}: ({calls} calls, {hits} hits, {misses} misses - "
                f"{hits / (hits + misses) * 100 if (hits + misses) > 0 else 0:.1f}% cache efficiency)"
            )
        return f"{' ' * indent}{name}: (0 calls)"

    def print_cache_info(self):
        print("Medium:", self.medium, "- cache info:")
        total_hits = 0
        total_misses = 0
        for m in self.modules:
            func = list(m.values())[0]
            if not hasattr(func, "cache_info"):
                continue
            total_hits += func.cache_info().hits
            total_misses += func.cache_info().misses
            print(self.get_lru_chache_info_str(func, indent=2))
        total_efficiency = total_hits / (total_hits + total_misses) * 100 if (total_hits + total_misses) > 0 else 0
        print(f"Total cache efficiency: {total_efficiency:.1f}%")
        if self._batch_state_cache_ph:
            print(f"  batch (p,h) state pool: {len(self._batch_state_cache_ph)} cached batches, "
                  f"{len(self._batch_state_free_ph)} free states")

    def clear_cache(self):
        # Size-1 EOS-update caches.
        self.set_state_ph.cache_clear()
        self.set_state_pT.cache_clear()
        self.set_state_ps.cache_clear()
        # Per-property `lru_cache`s (set up per-instance in `__init__`).
        for _name in self._SCALAR_EVAL_NAMES + self._HEM_EVAL_NAMES:
            getattr(self, _name).cache_clear()
        # Batch-evaluator state pool.
        self._batch_state_cache_ph.clear()


# Mark every batch evaluator as already-vectorised so `model.py`'s
# `_vectorise_callable` (which would otherwise re-wrap them in a Python loop
# at instantiate time) returns them unchanged.  Done at class scope so the
# `_hydrogen_vectorised` attribute lives on the underlying function -- bound
# methods inherit attribute lookups from their `__func__`.
for _name in (
    "eval_h_pT_batch", "eval_dh_pT_dp_batch", "eval_dh_pT_dT_batch",
    "eval_rho_ph_batch", "eval_drho_ph_dp_batch", "eval_drho_ph_dh_batch",
    "eval_mu_ph_batch", "eval_dmu_ph_dp_batch", "eval_dmu_ph_dh_batch",
    "eval_T_ph_batch", "eval_dT_ph_dp_batch", "eval_dT_ph_dh_batch",
    "eval_s_ph_batch", "eval_ds_ph_dp_batch", "eval_ds_ph_dh_batch",
    "eval_k_ph_batch", "eval_dk_ph_dp_batch", "eval_dk_ph_dh_batch",
    "eval_drho_ph_hem_dp_batch", "eval_drho_ph_hem_dh_batch",
    "eval_dT_ph_hem_dp_batch", "eval_dT_ph_hem_dh_batch",
    "eval_dmu_ph_hem_dp_batch", "eval_dmu_ph_hem_dh_batch",
    "eval_dk_ph_hem_dp_batch", "eval_dk_ph_hem_dh_batch",
):
    getattr(CoolPropMedium, _name)._hydrogen_vectorised = True
del _name


# ===========================================================================
# feos-backed medium (optional dependency)
# ===========================================================================
#
# `FeosMedium` is a drop-in alternative to `CoolPropMedium`: it exposes the
# IDENTICAL public surface (the sympy-able `h_pT`, `rho_ph`, `T_ph`, `s_ph`,
# `k_ph`, `mu_ph` property functions and their `*_hem` variants, a `modules`
# list for `lambdify`, `default_vars`, the scalar `eval_*` evaluators, and the
# `clear_cache` / introspection helpers), so the rest of hydrogen treats the
# two backends interchangeably.
#
# Division of labour
# ------------------
# * Thermodynamics (rho, T, s, h and the (p, h) flash) come from a feos
#   `EquationOfState`.  feos solves the state natively from `(p, molar_enthalpy)`
#   -- no hand-rolled flash needed.
# * Transport (mu, k) delegate to CoolProp by default: feos only computes
#   viscosity/thermal conductivity when the residual model carries
#   entropy-scaling parameters (Peng-Robinson and bare PC-SAFT do NOT, and they
#   *panic* if asked), so CoolProp is the robust default.  Pass
#   ``transport="feos"`` to force feos transport when you know the EOS supports
#   it.
# * Partial derivatives of the THERMODYNAMIC properties (rho, T, s) are
#   ANALYTIC: they are assembled from feos's exact high-level derivative
#   quantities (`joule_thomson`, `specific_isobaric_heat_capacity`,
#   `thermal_expansivity`, `isothermal_compressibility`) evaluated at the SINGLE
#   value flash, via standard thermodynamic identities (see `_deriv_inputs_ph`).
#   This avoids the ~12 extra (p, h) flashes per state that a finite-difference
#   Jacobian column would need -- the dominant cost in a feos-backed solve.
#   Transport (mu, k) partials use the chain rule through T(p, h) with cheap
#   CoolProp (T, p) finite differences (no extra feos flash).  If any feos
#   derivative quantity is unavailable for a state, the affected partial falls
#   back to a central finite difference automatically.
#
# EOS construction
# ----------------
# feos ships no parameter database, so an EOS must be parameterised.  In order
# of precedence `FeosMedium` uses:
#   1. an explicit, fully-built ``eos=`` (advanced users -- must already include
#      an ideal-gas contribution),
#   2. ``parameters=`` (a feos PC-SAFT ``Parameters`` object) -> PC-SAFT + a
#      constant-cp ideal gas,
#   3. by fluid name -> a Peng-Robinson EOS built from CoolProp's critical
#      constants (Tc, Pc, acentric factor, molar mass) + a constant-cp ideal
#      gas whose cp is taken from CoolProp's ideal-gas heat capacity.
#
# The by-name Peng-Robinson path makes ``FeosMedium("Hydrogen")`` work out of
# the box exactly like ``CoolPropMedium("Hydrogen")``.  Peng-Robinson is a cubic
# EOS: gas-phase densities are good (~0.1-0.5%), liquid densities are only
# qualitative; supply a PC-SAFT/multiparameter ``eos=`` when you need reference
# accuracy.


_FEOS_CACHE = None

#: Analytic-derivative inputs read off a single (p, h) flash.  `T` [K], `rho`
#: [kg/m^3], `s` [J/kg/K], `cp` (specific isobaric heat capacity) [J/kg/K],
#: `alpha` (thermal expansivity) [1/K], `kT` (isothermal compressibility) [1/Pa],
#: `muJT` (Joule-Thomson coefficient) [K/Pa].
_PhDeriv = namedtuple("_PhDeriv", "T rho s cp alpha kT muJT")

#: Analytic-derivative inputs read off a single (p, T) flash (for `h_pT`).
_PTDeriv = namedtuple("_PTDeriv", "rho cp alpha")


def _load_feos():
    """Import feos + si_units lazily so feos stays an OPTIONAL dependency.

    `CoolPropMedium` (the default backend) must keep working when feos is not
    installed, so the import only happens when a `FeosMedium` is actually
    constructed.
    """
    global _FEOS_CACHE
    if _FEOS_CACHE is None:
        import importlib
        try:
            feos = importlib.import_module("feos")
            si = importlib.import_module("si_units")
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "FeosMedium requires the optional 'feos' and 'si-units' "
                "packages (pip install feos si-units)."
            ) from exc
        _FEOS_CACHE = (feos, si)
    return _FEOS_CACHE


class _ConstantCpIdealGas:
    """Minimal feos Python ideal-gas model with a constant molar heat capacity.

    feos splits the Helmholtz energy into an ideal-gas and a residual part.  A
    residual EOS (Peng-Robinson, PC-SAFT, ...) on its own *cannot* evaluate any
    caloric property (enthalpy, entropy, temperature-from-enthalpy) -- it panics
    with "No ideal gas model initialized".  This class supplies the missing
    ideal-gas contribution through the temperature dependence of the thermal de
    Broglie wavelength ``Lambda(T)`` (the only thing feos needs from an ideal-gas
    model).

    feos differentiates ``ln(Lambda^3)`` w.r.t. temperature to get the caloric
    properties:

        u_ig(T) (per mole) = -R T^2 d/dT ln(Lambda^3)
        c_v_ig             =  d u_ig / dT

    so ``ln(Lambda^3) = -(c_v/R) ln(T)`` yields a perfect gas with constant molar
    ``c_v`` (hence constant ``c_p = c_v + R``).  The arbitrary additive constant
    only shifts the (internal-to-one-medium) enthalpy/entropy reference and is
    irrelevant to the dynamics.
    """

    def __init__(self, cv_molar, R):
        self._cv_over_R = float(cv_molar) / float(R)

    def components(self):
        return 1

    def subset(self, _indices):
        # Pure-component model: any subset is the same single component.
        return self

    def ln_lambda3(self, temperature):
        # `temperature` arrives as a feos dual number (unit-stripped Kelvin);
        # numpy ufuncs dispatch to the dual type's own ops so autodiff works.
        return -self._cv_over_R * np.log(temperature)


class FeosMedium:
    """feos-backed thermophysical medium, interface-compatible with `CoolPropMedium`.

    See the module-level section comment for the design (thermo via feos,
    transport via CoolProp, finite-difference partials, EOS construction order).

    Parameters
    ----------
    medium : str
        Fluid name.  Used as the symbolic-function prefix, for the default
        Peng-Robinson parameterisation (via CoolProp critical constants), and as
        the CoolProp fluid for transport delegation.
    p, T : float
        Reference state for `default_vars` and the (p, h) flash warm-start.
    eos : feos.EquationOfState, optional
        A fully-built feos EOS (must already include an ideal-gas contribution).
        Overrides the by-name / `parameters` construction.
    parameters : feos.Parameters, optional
        PC-SAFT parameters; an EOS is built as ``pcsaft(parameters)`` plus a
        constant-cp ideal gas.
    cp_ideal : float, optional
        Ideal-gas molar heat capacity [J/mol/K] for the auto-built ideal gas.
        Defaults to CoolProp's ``CP0MOLAR`` at ``T_ref``.
    T_ref : float
        Temperature [K] at which the default ideal-gas cp is sampled.
    transport : {"coolprop", "feos", "none"}
        Source of viscosity/thermal conductivity.  ``"coolprop"`` (default)
        evaluates them at the feos-flashed ``(p, T)``; ``"feos"`` uses feos's
        entropy-scaling transport (only valid for EOSs that support it).
    transport_fluid : str, optional
        CoolProp fluid name for transport delegation (defaults to ``medium``).
    scalar_cache_maxsize : int, optional
        Per-property `lru_cache` size (see `CoolPropMedium`).
    hem_fd_dh, hem_fd_dp : float
        Finite-difference smoothing bands for the `*_hem` partials.
    fd_rel : float
        Relative step for the (non-HEM) finite-difference partials.
    """

    # Class-level defaults; override per instance via the `__init__` kwargs.
    scalar_cache_maxsize = 100
    max_array_size = 10

    # Reuse CoolPropMedium's evaluator-name lists so the cache-wrapping loop,
    # `clear_cache`, and any cache-aware tooling stay in lock-step across both
    # backends.
    _SCALAR_EVAL_NAMES = CoolPropMedium._SCALAR_EVAL_NAMES
    _HEM_EVAL_NAMES = CoolPropMedium._HEM_EVAL_NAMES

    # Internal cached helpers backing the analytic partials (wrapped with the
    # same per-instance lru_cache as the scalar evaluators).
    _AUX_CACHE_NAMES = ("_deriv_inputs_ph", "_deriv_inputs_pT",
                        "_mu_partials", "_k_partials")

    _R = 8.314462618  # universal gas constant [J/mol/K]

    def __init__(self, medium, p=101325, T=293.15, disable_warnings=False,
                 eos=None, parameters=None, cp_ideal=None, T_ref=298.15,
                 transport="coolprop", transport_fluid=None,
                 transport_backend="BICUBIC&HEOS",
                 scalar_cache_maxsize=None,
                 hem_fd_dh=5000.0, hem_fd_dp=5000.0, fd_rel=1e-6):
        feos, si = _load_feos()
        self._feos = feos
        self._si = si
        self.medium = medium
        self.disable_warnings = disable_warnings
        self.transport = transport
        self.transport_fluid = transport_fluid or medium
        # CoolProp transport is, by far, the most expensive part of a feos solve
        # when routed through `PropsSI` (HEOS): a tabular `AbstractState`
        # (BICUBIC&HEOS) gives identical mu/k ~500x faster.  Build it once and
        # reuse via `update(PT_INPUTS, ...)`; fall back to HEOS, then PropsSI.
        self.transport_backend = transport_backend
        self._tr_state = None
        self._tr_key = None  # size-1 (T, p) memo so value+partials share update
        if self.transport == "coolprop":
            self._tr_state = self._make_transport_state()
        self.hem_fd_dh = float(hem_fd_dh)
        self.hem_fd_dp = float(hem_fd_dp)
        self.fd_rel = float(fd_rel)
        # `backend` mirrors CoolPropMedium's attribute so `serialize_medium`
        # and any backend-agnostic diagnostics work uniformly.
        self.backend = "feos"

        # si_units shorthands (built once; multiply floats by these to get SI
        # quantities, divide quantities by these to recover plain floats).
        self._PA = si.PASCAL
        self._K = si.KELVIN
        self._RHOU = si.KILOGRAM / si.METER**3
        self._HU = si.JOULE / si.KILOGRAM
        self._SU = si.JOULE / si.KILOGRAM / si.KELVIN
        self._HMU = si.JOULE / si.MOL
        self._MMU = si.KILOGRAM / si.MOL
        self._VISCU = si.PASCAL * si.SECOND
        self._CONDU = si.WATT / si.METER / si.KELVIN
        # Units for the analytic-derivative high-level quantities.
        self._INVK = 1 / si.KELVIN              # thermal expansivity
        self._INVPA = 1 / si.PASCAL             # isothermal compressibility
        self._JTU = si.KELVIN / si.PASCAL       # Joule-Thomson coefficient

        self._eos, self._mm = self._build_eos(eos, parameters, cp_ideal, T_ref)
        self._T_init = float(T)

        # Size-1 flash memos: rho/T/s requested at the SAME (p, h) within one
        # CSE'd template lambda reuse a single feos solve (the analogue of
        # CoolPropMedium's `set_state_ph` size-1 cache).
        self._ph_key = None
        self._ph_state = None
        self._pT_key = None
        self._pT_state = None

        if scalar_cache_maxsize is not None:
            self.scalar_cache_maxsize = scalar_cache_maxsize
        # Wrap the public scalar evaluators AND the internal analytic-derivative
        # helpers with a per-instance lru_cache so a state's whole Jacobian
        # column is assembled from a single flash (`_deriv_inputs_ph` caches the
        # high-level derivative quantities; `_mu_partials` / `_k_partials` cache
        # the transport chain-rule (dp, dh) pair).
        for _name in (self._SCALAR_EVAL_NAMES + self._HEM_EVAL_NAMES
                      + self._AUX_CACHE_NAMES):
            _bound = getattr(self, _name)
            setattr(self, _name,
                    functools.lru_cache(maxsize=self.scalar_cache_maxsize)(_bound))

        self.h, self.p, self.T = sp.symbols('h p T', real=True)
        self.h_pT = get_symbolic_property_function(self.eval_h_pT,    {1: self.eval_dh_pT_dp,  2: self.eval_dh_pT_dT},  ["p", "T"], medium, "h_pT")
        self.rho_ph = get_symbolic_property_function(self.eval_rho_ph, {1: self.eval_drho_ph_dp, 2: self.eval_drho_ph_dh}, ["p", "h"], medium, "rho_ph")
        self.mu_ph = get_symbolic_property_function(self.eval_mu_ph,  {1: self.eval_dmu_ph_dp,  2: self.eval_dmu_ph_dh},  ["p", "h"], medium, "mu_ph")
        self.T_ph = get_symbolic_property_function(self.eval_T_ph,    {1: self.eval_dT_ph_dp,   2: self.eval_dT_ph_dh},   ["p", "h"], medium, "T_ph")
        self.s_ph = get_symbolic_property_function(self.eval_s_ph,    {1: self.eval_ds_ph_dp,   2: self.eval_ds_ph_dh},   ["p", "h"], medium, "s_ph")
        self.k_ph = get_symbolic_property_function(self.eval_k_ph,    {1: self.eval_dk_ph_dp,   2: self.eval_dk_ph_dh},   ["p", "h"], medium, "k_ph")

        # Smooth-HEM variants: SAME value evaluator, wider FD partials so the
        # Jacobian stays continuous through the saturation lines (parity with
        # CoolPropMedium's HEM machinery).
        self.rho_ph_hem = get_symbolic_property_function(self.eval_rho_ph, {1: self.eval_drho_ph_hem_dp, 2: self.eval_drho_ph_hem_dh}, ["p", "h"], medium, "rho_ph_hem")
        self.T_ph_hem   = get_symbolic_property_function(self.eval_T_ph,   {1: self.eval_dT_ph_hem_dp,   2: self.eval_dT_ph_hem_dh},   ["p", "h"], medium, "T_ph_hem")
        self.mu_ph_hem  = get_symbolic_property_function(self.eval_mu_ph,  {1: self.eval_dmu_ph_hem_dp,  2: self.eval_dmu_ph_hem_dh},  ["p", "h"], medium, "mu_ph_hem")
        self.k_ph_hem   = get_symbolic_property_function(self.eval_k_ph,   {1: self.eval_dk_ph_hem_dp,   2: self.eval_dk_ph_hem_dh},   ["p", "h"], medium, "k_ph_hem")

        # `drho/dp` symbolic functions for the compressible-level stabilisation
        # gate (parity with CoolPropMedium).
        self.drho_ph_dp = get_symbolic_property_function(self.eval_drho_ph_dp, {1: self.eval_d2rho_ph_dp2, 2: self.eval_d2rho_ph_dpdh}, ["p", "h"], medium, "drho_ph_dp")
        self.drho_ph_hem_dp = get_symbolic_property_function(self.eval_drho_ph_hem_dp, {1: self.eval_d2rho_ph_hem_dp2, 2: self.eval_d2rho_ph_hem_dpdh}, ["p", "h"], medium, "drho_ph_hem_dp")
        self.drho_ph_dh = get_symbolic_property_function(self.eval_drho_ph_dh, {1: self.eval_d2rho_ph_dpdh, 2: self.eval_d2rho_ph_dh2}, ["p", "h"], medium, "drho_ph_dh")
        self.drho_ph_hem_dh = get_symbolic_property_function(self.eval_drho_ph_hem_dh, {1: self.eval_d2rho_ph_hem_dpdh, 2: self.eval_d2rho_ph_hem_dh2}, ["p", "h"], medium, "drho_ph_hem_dh")

        self.default_vars = {'p': p, 'T': T, 'h': self.h_pT(p, T)}
        self.modules = [
            {f"{medium}_h_pT":   self.eval_h_pT},   {f"{medium}_dh_pT_dp":  self.eval_dh_pT_dp},  {f"{medium}_dh_pT_dT":  self.eval_dh_pT_dT},
            {f"{medium}_rho_ph": self.eval_rho_ph}, {f"{medium}_drho_ph_dp": self.eval_drho_ph_dp}, {f"{medium}_drho_ph_dh": self.eval_drho_ph_dh},
            {f"{medium}_mu_ph":  self.eval_mu_ph},  {f"{medium}_dmu_ph_dp":  self.eval_dmu_ph_dp},  {f"{medium}_dmu_ph_dh":  self.eval_dmu_ph_dh},
            {f"{medium}_T_ph":   self.eval_T_ph},   {f"{medium}_dT_ph_dp":   self.eval_dT_ph_dp},   {f"{medium}_dT_ph_dh":   self.eval_dT_ph_dh},
            {f"{medium}_s_ph":   self.eval_s_ph},   {f"{medium}_ds_ph_dp":   self.eval_ds_ph_dp},   {f"{medium}_ds_ph_dh":   self.eval_ds_ph_dh},
            {f"{medium}_k_ph":   self.eval_k_ph},   {f"{medium}_dk_ph_dp":   self.eval_dk_ph_dp},   {f"{medium}_dk_ph_dh":   self.eval_dk_ph_dh},
            {f"{medium}_ddrho_ph_dp_dp": self.eval_d2rho_ph_dp2}, {f"{medium}_ddrho_ph_dp_dh": self.eval_d2rho_ph_dpdh},
            {f"{medium}_ddrho_ph_dh_dp": self.eval_d2rho_ph_dpdh}, {f"{medium}_ddrho_ph_dh_dh": self.eval_d2rho_ph_dh2},
            {f"{medium}_rho_ph_hem": self.eval_rho_ph}, {f"{medium}_drho_ph_hem_dp": self.eval_drho_ph_hem_dp}, {f"{medium}_drho_ph_hem_dh": self.eval_drho_ph_hem_dh},
            {f"{medium}_T_ph_hem":   self.eval_T_ph},   {f"{medium}_dT_ph_hem_dp":   self.eval_dT_ph_hem_dp},   {f"{medium}_dT_ph_hem_dh":   self.eval_dT_ph_hem_dh},
            {f"{medium}_mu_ph_hem":  self.eval_mu_ph},  {f"{medium}_dmu_ph_hem_dp":  self.eval_dmu_ph_hem_dp},  {f"{medium}_dmu_ph_hem_dh":  self.eval_dmu_ph_hem_dh},
            {f"{medium}_k_ph_hem":   self.eval_k_ph},   {f"{medium}_dk_ph_hem_dp":   self.eval_dk_ph_hem_dp},   {f"{medium}_dk_ph_hem_dh":   self.eval_dk_ph_hem_dh},
            {f"{medium}_ddrho_ph_hem_dp_dp": self.eval_d2rho_ph_hem_dp2}, {f"{medium}_ddrho_ph_hem_dp_dh": self.eval_d2rho_ph_hem_dpdh},
            {f"{medium}_ddrho_ph_hem_dh_dp": self.eval_d2rho_ph_hem_dpdh}, {f"{medium}_ddrho_ph_hem_dh_dh": self.eval_d2rho_ph_hem_dh2},
        ]
        # `batch_modules`: same keys as `modules`, values are numpy-array-aware
        # closures (each just loops the scalar evaluator) so callers can opt into
        # the batch lambdify namespace exactly like CoolPropMedium.
        self.batch_modules = [
            {name: self._vectorised(next(iter(d.values())), pT=name.endswith("_h_pT") or "_pT_" in name)}
            for d in self.modules for name in d
        ]

    # --- EOS construction -------------------------------------------------------------

    def _build_eos(self, eos, parameters, cp_ideal, T_ref):
        """Return ``(equation_of_state, molar_mass_kg_per_mol)``."""
        feos = self._feos
        if eos is not None:
            # Caller-supplied EOS is assumed complete (ideal gas included).
            return eos, self._molar_mass_from_state(eos)

        R = self._R
        if cp_ideal is None:
            import CoolProp.CoolProp as CP
            cp_ideal = CP.PropsSI("CP0MOLAR", "T", float(T_ref), "P", 101325.0, self.medium)
        ideal_gas = _ConstantCpIdealGas(float(cp_ideal) - R, R)

        if parameters is not None:
            residual = feos.EquationOfState.pcsaft(parameters)
            full = residual.python_ideal_gas([ideal_gas])
            return full, self._molar_mass_from_state(full)

        # Default: Peng-Robinson from CoolProp critical constants.
        import CoolProp.CoolProp as CP
        Tc = CP.PropsSI("Tcrit", self.medium)
        Pc = CP.PropsSI("pcrit", self.medium)
        omega = CP.PropsSI("acentric", self.medium)
        M = CP.PropsSI("M", self.medium)  # kg/mol
        record = feos.PureRecord(
            feos.Identifier(name=self.medium),
            M * 1000.0,  # feos PureRecord molar weight is in g/mol
            tc=Tc, pc=Pc, acentric_factor=omega,
        )
        residual = feos.EquationOfState.peng_robinson(feos.Parameters.new_pure(record))
        full = residual.python_ideal_gas([ideal_gas])
        return full, M

    def _molar_mass_from_state(self, eos):
        si = self._si
        st = self._feos.State(eos, temperature=300 * si.KELVIN, pressure=101325 * si.PASCAL)
        return float(self._value(st.total_molar_weight) / self._MMU)

    # --- feos state helpers -----------------------------------------------------------

    @staticmethod
    def _value(member):
        """feos exposes most state properties as methods but a few (e.g.
        ``temperature``) as plain attributes; normalise both to a value."""
        return member() if callable(member) else member

    def _flash_pT(self, p, T):
        p = float(p)
        T = float(T)
        key = (p, T)
        if self._pT_key == key:
            return self._pT_state
        st = self._feos.State(self._eos, temperature=T * self._K, pressure=p * self._PA)
        self._pT_key, self._pT_state = key, st
        return st

    def _flash_ph(self, p, h):
        p = float(p)
        h = float(h)
        key = (p, h)
        if self._ph_key == key:
            return self._ph_state
        st = self._feos.State(
            self._eos,
            pressure=p * self._PA,
            molar_enthalpy=(h * self._mm) * self._HMU,
            initial_temperature=self._T_init * self._K,
        )
        self._ph_key, self._ph_state = key, st
        return st

    # --- analytic-derivative inputs ---------------------------------------------------
    #
    # Each helper does NO extra flash: it reuses the size-1 `_flash_*` memo (a hit
    # when the value evaluator was just called at the same point) and reads feos's
    # exact high-level derivative quantities off that one state.  The partials are
    # then assembled via standard thermodynamic identities (validated against
    # finite differences to ~7 significant figures).  Returns ``None`` if any
    # quantity is unavailable for the state, so the caller can fall back to FD.

    def _deriv_inputs_ph(self, p, h):
        try:
            st = self._flash_ph(p, h)
            v = self._value
            return _PhDeriv(
                T=float(v(st.temperature) / self._K),
                rho=float(v(st.mass_density) / self._RHOU),
                s=float(v(st.specific_entropy) / self._SU),
                cp=float(v(st.specific_isobaric_heat_capacity) / self._SU),
                alpha=float(v(st.thermal_expansivity) / self._INVK),
                kT=float(v(st.isothermal_compressibility) / self._INVPA),
                muJT=float(v(st.joule_thomson) / self._JTU),
            )
        except Exception:
            return None

    def _deriv_inputs_pT(self, p, T):
        try:
            st = self._flash_pT(p, T)
            v = self._value
            return _PTDeriv(
                rho=float(v(st.mass_density) / self._RHOU),
                cp=float(v(st.specific_isobaric_heat_capacity) / self._SU),
                alpha=float(v(st.thermal_expansivity) / self._INVK),
            )
        except Exception:
            return None

    def _mu_partials(self, p, h):
        return self._transport_partials("V", p, h)

    def _k_partials(self, p, h):
        return self._transport_partials("L", p, h)

    def _transport_partials(self, cp_key, p, h):
        """``(dprop/dp|h, dprop/dh|p)`` for a CoolProp transport property.

        Chain rule through ``T(p, h)``:
            dX/dp|h = dX/dT|p * (dT/dp|h) + dX/dp|T
            dX/dh|p = dX/dT|p * (dT/dh|p)
        with ``dT/dp|h`` = Joule-Thomson coefficient and ``dT/dh|p`` = 1/cp from
        the feos flash, and the CoolProp (T, p) gradients by finite difference --
        which needs NO extra feos flash (the costly part)."""
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            eval_fn = self.eval_mu_ph if cp_key == "V" else self.eval_k_ph
            return (self._d_dp(eval_fn, p, h), self._d_dh(eval_fn, p, h))
        T = d.T
        p = float(p)
        eT = max(1e-4 * T, 1e-3)
        ep = max(1e-3 * p, 1.0)

        def g(TT, pp):
            return self._cp_transport(cp_key, TT, pp)

        dX_dT = (g(T + eT, p) - g(T - eT, p)) / (2.0 * eT)
        dX_dp_T = (g(T, p + ep) - g(T, p - ep)) / (2.0 * ep)
        return (dX_dT * d.muJT + dX_dp_T, dX_dT / d.cp)

    # --- finite-difference helpers ----------------------------------------------------

    def _d_dp(self, eval_func, p, h):
        e = max(self.fd_rel * max(abs(float(p)), 1e3), 1.0)
        return (eval_func(p + e, h) - eval_func(p - e, h)) / (2.0 * e)

    def _d_dh(self, eval_func, p, h):
        e = max(self.fd_rel * max(abs(float(h)), 1e3), 1.0)
        return (eval_func(p, h + e) - eval_func(p, h - e)) / (2.0 * e)

    def _d_dT(self, eval_func, p, T):
        e = max(self.fd_rel * max(abs(float(T)), 1.0), 1e-4)
        return (eval_func(p, T + e) - eval_func(p, T - e)) / (2.0 * e)

    def _fd_dh(self, eval_func, p, h):
        e = self.hem_fd_dh
        return (eval_func(p, h + e) - eval_func(p, h - e)) / (2.0 * e)

    def _fd_dp(self, eval_func, p, h):
        e = self.hem_fd_dp
        p_hi = p + e
        p_lo = max(p - e, e)
        return (eval_func(p_hi, h) - eval_func(p_lo, h)) / (p_hi - p_lo)

    # --- value evaluators -------------------------------------------------------------

    def eval_h_pT(self, p, T):
        if p == 0 or T == 0:
            return None
        return float(self._value(self._flash_pT(p, T).specific_enthalpy) / self._HU)

    def eval_rho_ph(self, p, h):
        return float(self._value(self._flash_ph(p, h).mass_density) / self._RHOU)

    def eval_T_ph(self, p, h):
        return float(self._value(self._flash_ph(p, h).temperature) / self._K)

    def eval_s_ph(self, p, h):
        return float(self._value(self._flash_ph(p, h).specific_entropy) / self._SU)

    def eval_mu_ph(self, p, h):
        return self._transport("V", self._VISCU, lambda s: s.viscosity, p, h)

    def eval_k_ph(self, p, h):
        return self._transport("L", self._CONDU, lambda s: s.thermal_conductivity, p, h)

    def _make_transport_state(self):
        """Build a reusable CoolProp transport state, fastest backend first."""
        backends = [self.transport_backend, "HEOS"]
        for backend in backends:
            try:
                st = CP.AbstractState(backend, self.transport_fluid)
                st.update(CP.PT_INPUTS, 101325.0, 300.0)
                st.viscosity()
                st.conductivity()
                return st
            except Exception:
                continue
        return None  # caller falls back to PropsSI

    def _cp_transport(self, cp_key, T, p):
        """CoolProp viscosity ('V') / conductivity ('L') at (T, p).

        Uses the cached tabular `AbstractState` (with a size-1 (T, p) memo so the
        mu and k value reads at one state share a single ``update``); falls back
        to ``PropsSI`` if no reusable state could be built."""
        st = self._tr_state
        if st is None:
            return CP.PropsSI(cp_key, "T", float(T), "P", float(p),
                              self.transport_fluid)
        key = (T, p)
        if self._tr_key != key:
            st.update(CP.PT_INPUTS, float(p), float(T))
            self._tr_key = key
        return st.viscosity() if cp_key == "V" else st.conductivity()

    def _transport(self, cp_key, feos_unit, feos_getter, p, h):
        if self.transport == "feos":
            st = self._flash_ph(p, h)
            return float(self._value(feos_getter(st)) / feos_unit)
        if self.transport == "none":
            raise NotImplementedError(
                f"FeosMedium('{self.medium}') was built with transport='none'")
        # Default: CoolProp transport at the feos-flashed (p, T).  T is a true
        # physical temperature (reference-independent), so mixing feos
        # thermodynamics with CoolProp transport is consistent.
        T = self.eval_T_ph(p, h)
        return self._cp_transport(cp_key, T, float(p))

    # --- (p, T) partials (analytic, with FD fallback) ---------------------------------
    #
    #   dh/dT|p = cp                       dh/dp|T = (1/rho)(1 - T*alpha)

    def eval_dh_pT_dp(self, p, T):
        if p == 0 or T == 0:
            return None
        d = self._deriv_inputs_pT(p, T)
        if d is None:
            return self._d_dp(self.eval_h_pT, p, T)
        return (1.0 / d.rho) * (1.0 - T * d.alpha)

    def eval_dh_pT_dT(self, p, T):
        if p == 0 or T == 0:
            return None
        d = self._deriv_inputs_pT(p, T)
        if d is None:
            return self._d_dT(self.eval_h_pT, p, T)
        return d.cp

    # --- (p, h) partials (analytic, with FD fallback) ---------------------------------
    #
    # From the single (p, h) flash via standard identities:
    #   dT/dp|h   = muJT                       dT/dh|p   = 1/cp
    #   drho/dp|h = rho*(kT - alpha*muJT)      drho/dh|p = -rho*alpha/cp
    #   ds/dp|h   = -1/(rho*T)                 ds/dh|p   = 1/T

    def eval_drho_ph_dp(self, p, h):
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            return self._d_dp(self.eval_rho_ph, p, h)
        return d.rho * (d.kT - d.alpha * d.muJT)

    def eval_drho_ph_dh(self, p, h):
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            return self._d_dh(self.eval_rho_ph, p, h)
        return -d.rho * d.alpha / d.cp

    # Second derivatives of rho(p, h) (finite difference of the analytic first
    # partial), for the compressible channel's stabilisation gate Jacobian.
    def eval_d2rho_ph_dp2(self, p, h):
        return self._d_dp(self.eval_drho_ph_dp, p, h)

    def eval_d2rho_ph_dpdh(self, p, h):
        return self._d_dh(self.eval_drho_ph_dp, p, h)

    def eval_d2rho_ph_dh2(self, p, h):
        return self._d_dh(self.eval_drho_ph_dh, p, h)

    def eval_dT_ph_dp(self, p, h):
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            return self._d_dp(self.eval_T_ph, p, h)
        return d.muJT

    def eval_dT_ph_dh(self, p, h):
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            return self._d_dh(self.eval_T_ph, p, h)
        return 1.0 / d.cp

    def eval_ds_ph_dp(self, p, h):
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            return self._d_dp(self.eval_s_ph, p, h)
        return -1.0 / (d.rho * d.T)

    def eval_ds_ph_dh(self, p, h):
        d = self._deriv_inputs_ph(p, h)
        if d is None:
            return self._d_dh(self.eval_s_ph, p, h)
        return 1.0 / d.T

    def eval_dmu_ph_dp(self, p, h):
        if self.transport == "coolprop":
            return self._mu_partials(p, h)[0]
        return self._d_dp(self.eval_mu_ph, p, h)

    def eval_dmu_ph_dh(self, p, h):
        if self.transport == "coolprop":
            return self._mu_partials(p, h)[1]
        return self._d_dh(self.eval_mu_ph, p, h)

    def eval_dk_ph_dp(self, p, h):
        if self.transport == "coolprop":
            return self._k_partials(p, h)[0]
        return self._d_dp(self.eval_k_ph, p, h)

    def eval_dk_ph_dh(self, p, h):
        if self.transport == "coolprop":
            return self._k_partials(p, h)[1]
        return self._d_dh(self.eval_k_ph, p, h)

    # --- smooth-HEM partials (wider central differences) ------------------------------

    def eval_drho_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_rho_ph, p, h)

    def eval_drho_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_rho_ph, p, h)

    def eval_dT_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_T_ph, p, h)

    def eval_dT_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_T_ph, p, h)

    def eval_dmu_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_mu_ph, p, h)

    def eval_dmu_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_mu_ph, p, h)

    def eval_dk_ph_hem_dp(self, p, h):
        return self._fd_dp(self.eval_k_ph, p, h)

    def eval_dk_ph_hem_dh(self, p, h):
        return self._fd_dh(self.eval_k_ph, p, h)

    def eval_d2rho_ph_hem_dp2(self, p, h):
        return self._fd_dp(self.eval_drho_ph_hem_dp, p, h)

    def eval_d2rho_ph_hem_dpdh(self, p, h):
        return self._fd_dh(self.eval_drho_ph_hem_dp, p, h)

    def eval_d2rho_ph_hem_dh2(self, p, h):
        return self._fd_dh(self.eval_drho_ph_hem_dh, p, h)

    # --- batch (array-aware) variants -------------------------------------------------

    @staticmethod
    def _is_array(x):
        return isinstance(x, np.ndarray) and x.ndim > 0

    def _vectorised(self, scalar, pT=False):
        """Wrap a scalar `(p, x)` evaluator so it also accepts numpy arrays.

        Marked `_hydrogen_vectorised` so `model.py`'s `_vectorise_callable`
        leaves it alone.  feos has no batch state pool, so this just loops the
        (lru-cached) scalar evaluator -- correctness with array inputs, the
        cross-call caching still does the heavy lifting.
        """
        def wrapper(p, x):
            if not (self._is_array(p) or self._is_array(x)):
                return scalar(p, x)
            shape = np.broadcast_shapes(np.shape(p), np.shape(x))
            p_arr = np.broadcast_to(np.asarray(p, dtype=float), shape).ravel()
            x_arr = np.broadcast_to(np.asarray(x, dtype=float), shape).ravel()
            return np.fromiter(
                (scalar(float(p_arr[i]), float(x_arr[i])) for i in range(p_arr.size)),
                dtype=float, count=p_arr.size,
            ).reshape(shape)
        wrapper._hydrogen_vectorised = True
        return wrapper

    # --- introspection helpers --------------------------------------------------------

    def get_default_vars(self):
        return self.default_vars

    def get_lru_chache_info_str(self, func, indent=0):
        name = func.__name__
        info = func.cache_info()
        calls = info.hits + info.misses
        if calls > 0:
            return (
                f"{' ' * indent}{name}: ({calls} calls, {info.hits} hits, "
                f"{info.misses} misses - {info.hits / calls * 100:.1f}% cache efficiency)"
            )
        return f"{' ' * indent}{name}: (0 calls)"

    def print_cache_info(self):
        print("Medium:", self.medium, "(feos) - cache info:")
        total_hits = total_misses = 0
        for m in self.modules:
            func = list(m.values())[0]
            if not hasattr(func, "cache_info"):
                continue
            total_hits += func.cache_info().hits
            total_misses += func.cache_info().misses
            print(self.get_lru_chache_info_str(func, indent=2))
        total = total_hits + total_misses
        print(f"Total cache efficiency: {total_hits / total * 100 if total else 0:.1f}%")

    def clear_cache(self):
        self._ph_key = self._ph_state = None
        self._pT_key = self._pT_state = None
        self._tr_key = None
        for _name in (self._SCALAR_EVAL_NAMES + self._HEM_EVAL_NAMES
                      + self._AUX_CACHE_NAMES):
            getattr(self, _name).cache_clear()
