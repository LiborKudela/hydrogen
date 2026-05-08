"""CoolProp-backed thermophysical medium with sympy-friendly property functions."""

from __future__ import annotations

import functools

import CoolProp.CoolProp as CP
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
    """Caches CoolProp `AbstractState` lookups and exposes sympy-able property functions."""

    scalar_cache_maxsize = 100
    max_array_size = 10

    def __init__(self, medium, p=101325, T=293.15, disable_warnings=False):
        self.medium = medium
        self.abstarct_state_ph = CP.AbstractState("HEOS", self.medium)
        self.abstarct_state_pT = CP.AbstractState("HEOS", self.medium)
        self.abstarct_state_ps = CP.AbstractState("HEOS", self.medium)
        self.disable_warnings = disable_warnings

        self.h, self.p, self.T = sp.symbols('h p T', real=True)
        self.h_pT = get_symbolic_property_function(self.eval_h_pT,    {1: self.eval_dh_pT_dp,  2: self.eval_dh_pT_dT},  ["p", "T"], medium, "h_pT")
        self.rho_ph = get_symbolic_property_function(self.eval_rho_ph, {1: self.eval_drho_ph_dp, 2: self.eval_drho_ph_dh}, ["p", "h"], medium, "rho_ph")
        self.mu_ph = get_symbolic_property_function(self.eval_mu_ph,  {1: self.eval_dmu_ph_dp,  2: self.eval_dmu_ph_dh},  ["p", "h"], medium, "mu_ph")
        self.T_ph = get_symbolic_property_function(self.eval_T_ph,    {1: self.eval_dT_ph_dp,   2: self.eval_dT_ph_dh},   ["p", "h"], medium, "T_ph")
        self.s_ph = get_symbolic_property_function(self.eval_s_ph,    {1: self.eval_ds_ph_dp,   2: self.eval_ds_ph_dh},   ["p", "h"], medium, "s_ph")
        self.k_ph = get_symbolic_property_function(self.eval_k_ph,    {1: self.eval_dk_ph_dp,   2: self.eval_dk_ph_dh},   ["p", "h"], medium, "k_ph")

        self.default_vars = {'p': p, 'T': T, 'h': self.h_pT(p, T)}
        self.modules = [
            {f"{medium}_h_pT":   self.eval_h_pT},   {f"{medium}_dh_pT_dp":  self.eval_dh_pT_dp},  {f"{medium}_dh_pT_dT":  self.eval_dh_pT_dT},
            {f"{medium}_rho_ph": self.eval_rho_ph}, {f"{medium}_drho_ph_dp": self.eval_drho_ph_dp}, {f"{medium}_drho_ph_dh": self.eval_drho_ph_dh},
            {f"{medium}_mu_ph":  self.eval_mu_ph},  {f"{medium}_dmu_ph_dp":  self.eval_dmu_ph_dp},  {f"{medium}_dmu_ph_dh":  self.eval_dmu_ph_dh},
            {f"{medium}_T_ph":   self.eval_T_ph},   {f"{medium}_dT_ph_dp":   self.eval_dT_ph_dp},   {f"{medium}_dT_ph_dh":   self.eval_dT_ph_dh},
            {f"{medium}_s_ph":   self.eval_s_ph},   {f"{medium}_ds_ph_dp":   self.eval_ds_ph_dp},   {f"{medium}_ds_ph_dh":   self.eval_ds_ph_dh},
            {f"{medium}_k_ph":   self.eval_k_ph},   {f"{medium}_dk_ph_dp":   self.eval_dk_ph_dp},   {f"{medium}_dk_ph_dh":   self.eval_dk_ph_dh},
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

    # --- enthalpy h(p, T) -------------------------------------------------------------

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_h_pT(self, p, T):
        if p == 0 or T == 0:
            return None
        self.set_state_pT(p, T)
        return self.abstarct_state_pT.hmass()

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dh_pT_dp(self, p, T):
        self.set_state_pT(p, T)
        return self.abstarct_state_pT.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dh_pT_dT(self, p, T):
        self.set_state_pT(p, T)
        return self.abstarct_state_pT.first_partial_deriv(CP.iHmass, CP.iT, CP.iP)

    # --- density rho(p, h) ------------------------------------------------------------

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_rho_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.rhomass()

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_drho_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_drho_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)

    # --- viscosity mu(p, h) -----------------------------------------------------------

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_mu_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.viscosity()

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
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

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
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

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_T_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.T()

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dT_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iT, CP.iP, CP.iHmass)

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dT_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iT, CP.iHmass, CP.iP)

    # --- entropy s(p, h) --------------------------------------------------------------

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_s_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.smass()

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_ds_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iSmass, CP.iP, CP.iHmass)

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_ds_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.first_partial_deriv(CP.iSmass, CP.iHmass, CP.iP)

    # --- thermal conductivity k(p, h) -------------------------------------------------

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_k_ph(self, p, h):
        self.set_state_ph(p, h)
        return self.abstarct_state_ph.conductivity()

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
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

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
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
            total_hits += func.cache_info().hits
            total_misses += func.cache_info().misses
            print(self.get_lru_chache_info_str(func, indent=2))
        total_efficiency = total_hits / (total_hits + total_misses) * 100 if (total_hits + total_misses) > 0 else 0
        print(f"Total cache efficiency: {total_efficiency:.1f}%")

    def clear_cache(self):
        self.set_state_ph.cache_clear()
        self.set_state_pT.cache_clear()
        self.set_state_ps.cache_clear()
