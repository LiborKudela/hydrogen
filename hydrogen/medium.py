"""CoolProp-backed thermophysical medium with sympy-friendly property functions."""

from __future__ import annotations

import functools
from collections import OrderedDict

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
            # HEM variants: value reuses the single-phase evaluator; partials are
            # the smoothed finite-difference ones registered below.
            {f"{medium}_rho_ph_hem": self.eval_rho_ph}, {f"{medium}_drho_ph_hem_dp": self.eval_drho_ph_hem_dp}, {f"{medium}_drho_ph_hem_dh": self.eval_drho_ph_hem_dh},
            {f"{medium}_T_ph_hem":   self.eval_T_ph},   {f"{medium}_dT_ph_hem_dp":   self.eval_dT_ph_hem_dp},   {f"{medium}_dT_ph_hem_dh":   self.eval_dT_ph_hem_dh},
            {f"{medium}_mu_ph_hem":  self.eval_mu_ph},  {f"{medium}_dmu_ph_hem_dp":  self.eval_dmu_ph_hem_dp},  {f"{medium}_dmu_ph_hem_dh":  self.eval_dmu_ph_hem_dh},
            {f"{medium}_k_ph_hem":   self.eval_k_ph},   {f"{medium}_dk_ph_hem_dp":   self.eval_dk_ph_hem_dp},   {f"{medium}_dk_ph_hem_dh":   self.eval_dk_ph_hem_dh},
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
            # HEM variants (batch): value reuses the single-phase batch evaluator,
            # partials are the smoothed central differences driven through it.
            {f"{medium}_rho_ph_hem": self.eval_rho_ph_batch}, {f"{medium}_drho_ph_hem_dp": self.eval_drho_ph_hem_dp_batch}, {f"{medium}_drho_ph_hem_dh": self.eval_drho_ph_hem_dh_batch},
            {f"{medium}_T_ph_hem":   self.eval_T_ph_batch},   {f"{medium}_dT_ph_hem_dp":   self.eval_dT_ph_hem_dp_batch},   {f"{medium}_dT_ph_hem_dh":   self.eval_dT_ph_hem_dh_batch},
            {f"{medium}_mu_ph_hem":  self.eval_mu_ph_batch},  {f"{medium}_dmu_ph_hem_dp":  self.eval_dmu_ph_hem_dp_batch},  {f"{medium}_dmu_ph_hem_dh":  self.eval_dmu_ph_hem_dh_batch},
            {f"{medium}_k_ph_hem":   self.eval_k_ph_batch},   {f"{medium}_dk_ph_hem_dp":   self.eval_dk_ph_hem_dp_batch},   {f"{medium}_dk_ph_hem_dh":   self.eval_dk_ph_hem_dh_batch},
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
