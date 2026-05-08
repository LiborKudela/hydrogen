import numpy as np
import CoolProp.CoolProp as CP
import time
import numba
import functools
import inspect
import hashlib


from collections import OrderedDict, namedtuple
import line_profiler
import sympy as sp
import multiprocessing

print("sympy version:", sp.__version__)


G_const = 9.81
total_set_vars_time = 0

def lambdify_compat(args, expr, modules=None, cse=True, docstring_limit=-1):
    supported_kwargs = inspect.signature(sp.lambdify).parameters
    kwargs = {"modules": modules}
    if "cse" in supported_kwargs:
        kwargs["cse"] = cse
    if "docstring_limit" in supported_kwargs:
        kwargs["docstring_limit"] = docstring_limit
    return sp.lambdify(args, expr, **kwargs)

def hash_array(arr):
    # Convert array to a byte representation and hash it
    return hash(arr.tobytes()) if isinstance(arr, np.ndarray) else hash(arr)

def hash_args(*args):
    m = hashlib.md5()
    for arg in args:
        if isinstance(arg, np.ndarray):
            m.update(arg.tobytes())
        else:
            m.update(str(arg).encode())
    return m.hexdigest()

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

def numpy_cache(maxsize=128, include_data=None):
    def decorator(func):
        # Use a descriptor to handle method binding
        class CacheDescriptor:
            def __init__(self):
                self.caches = {}  # Instance-specific caches
                self.include_data = include_data

            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                # Get or create cache for this instance
                if id(obj) not in self.caches:
                    self.caches[id(obj)] = {
                        'cache': {},
                        'hits': 0,
                        'misses': 0
                    }
                cache_state = self.caches[id(obj)]

                @functools.wraps(func)
                def wrapper(*args):
                    #arr = args[0]  # Assuming first arg is the array
                    key = hash_args(*args)
                    cache = cache_state['cache']
                    if key not in cache:
                        cache[key] = func(obj, *args)  # Pass self (obj) to func
                        cache_state['misses'] += 1
                        # Handle maxsize (simple FIFO eviction)
                        if len(cache) > maxsize:
                            cache.pop(next(iter(cache)))
                    else:
                        cache_state['hits'] += 1
                    return cache[key]

                def cache_info():
                    return _CacheInfo(
                        cache_state['hits'],
                        cache_state['misses'],
                        maxsize,
                        len(cache_state['cache'])
                    )

                wrapper.cache_info = cache_info
                return wrapper

        return CacheDescriptor()
    return decorator

@numba.jit(nopython=True)
def fast_error_norm(vars):
    return np.linalg.norm(vars)

@numba.jit(nopython=True)
def fast_linear_solve(A, b):
    return np.linalg.solve(A, b)

class CoolPropMedium:
    scalar_cache_maxsize = 100
    max_array_size = 10

    def __init__(self, medium, p=101325, T=293.15, disable_warnings=False):
        self.medium = medium
        self.abstarct_state_ph = CP.AbstractState("HEOS", self.medium)
        self.abstarct_state_pT = CP.AbstractState("HEOS", self.medium)
        self.abstarct_state_ps = CP.AbstractState("HEOS", self.medium)
        self.disable_warnings = disable_warnings
        
        self.h, self.p, self.T = sp.symbols('h p T', real=True)
        self.h_pT = get_symbolic_property_function(self.eval_h_pT, {1: self.eval_dh_pT_dp, 2: self.eval_dh_pT_dT}, ["p", "T"], medium, "h_pT")
        self.rho_ph = get_symbolic_property_function(self.eval_rho_ph, {1: self.eval_drho_ph_dp, 2: self.eval_drho_ph_dh}, ["p", "h"], medium, "rho_ph")
        self.mu_ph = get_symbolic_property_function(self.eval_mu_ph, {1: self.eval_dmu_ph_dp, 2: self.eval_dmu_ph_dh}, ["p", "h"], medium, "mu_ph")
        self.T_ph = get_symbolic_property_function(self.eval_T_ph, {1: self.eval_dT_ph_dp, 2: self.eval_dT_ph_dh}, ["p", "h"], medium, "T_ph")
        self.s_ph = get_symbolic_property_function(self.eval_s_ph, {1: self.eval_ds_ph_dp, 2: self.eval_ds_ph_dh}, ["p", "h"], medium, "s_ph")
        self.k_ph = get_symbolic_property_function(self.eval_k_ph, {1: self.eval_dk_ph_dp, 2: self.eval_dk_ph_dh}, ["p", "h"], medium, "k_ph")

        self.default_vars = {'p':p, 'T':T, 'h':self.h_pT(p, T)}
        self.modules = [
            {f"{medium}_h_pT": self.eval_h_pT}, {f"{medium}_dh_pT_dp": self.eval_dh_pT_dp}, {f"{medium}_dh_pT_dT": self.eval_dh_pT_dT},
            {f"{medium}_rho_ph": self.eval_rho_ph}, {f"{medium}_drho_ph_dp": self.eval_drho_ph_dp}, {f"{medium}_drho_ph_dh": self.eval_drho_ph_dh},
            {f"{medium}_mu_ph": self.eval_mu_ph}, {f"{medium}_dmu_ph_dp": self.eval_dmu_ph_dp}, {f"{medium}_dmu_ph_dh": self.eval_dmu_ph_dh},
            {f"{medium}_T_ph": self.eval_T_ph}, {f"{medium}_dT_ph_dp": self.eval_dT_ph_dp}, {f"{medium}_dT_ph_dh": self.eval_dT_ph_dh},
            {f"{medium}_s_ph": self.eval_s_ph}, {f"{medium}_ds_ph_dp": self.eval_ds_ph_dp}, {f"{medium}_ds_ph_dh": self.eval_ds_ph_dh},
            {f"{medium}_k_ph": self.eval_k_ph}, {f"{medium}_dk_ph_dp": self.eval_dk_ph_dp}, {f"{medium}_dk_ph_dh": self.eval_dk_ph_dh},
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

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_h_pT(self, p, T):
        if p == 0 or T == 0:
            return None
        self.set_state_pT(p, T)
        h = self.abstarct_state_pT.hmass()
        return h
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dh_pT_dp(self, p, T):
        self.set_state_pT(p, T)
        dh_dp = self.abstarct_state_pT.first_partial_deriv(CP.iHmass, CP.iP, CP.iT)
        return dh_dp
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dh_pT_dT(self, p, T):
        self.set_state_pT(p, T)
        dh_dT = self.abstarct_state_pT.first_partial_deriv(CP.iHmass, CP.iT, CP.iP)
        return dh_dT

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_rho_ph(self, p, h):
        self.set_state_ph(p, h)
        rho = self.abstarct_state_ph.rhomass()
        return rho 
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_drho_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        drho_dp = self.abstarct_state_ph.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        return drho_dp
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_drho_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        drho_dh = self.abstarct_state_ph.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        return drho_dh

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_mu_ph(self, p, h):
        self.set_state_ph(p, h)
        mu = self.abstarct_state_ph.viscosity()
        return mu
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dmu_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        try:
            dmu_dp = self.abstarct_state_ph.first_partial_deriv(CP.iviscosity, CP.iP, CP.iHmass)
        except:
            if not self.disable_warnings:
                print("Warning: partial derivative of mu_ph w.r.t. p failed, using finite difference instead")
            eps = 1e-3
            dmu_dp = (self.eval_mu_ph(p+eps, h) - self.eval_mu_ph(p-eps, h)) / (2*eps)
        return dmu_dp
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dmu_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        try:
            dmu_dh = self.abstarct_state_ph.first_partial_deriv(CP.iviscosity, CP.iHmass, CP.iP)
        except:
            if not self.disable_warnings:
                print("Warning: partial derivative of mu_ph w.r.t. h failed, using finite difference instead")
            eps = 1e-3
            dmu_dh = (self.eval_mu_ph(p, h+eps) - self.eval_mu_ph(p, h-eps)) / (2*eps)
        return dmu_dh

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_T_ph(self, p, h):
        self.set_state_ph(p, h)
        T = self.abstarct_state_ph.T()
        return T
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dT_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        dT_dp = self.abstarct_state_ph.first_partial_deriv(CP.iT, CP.iP, CP.iHmass)
        return dT_dp
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dT_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        dT_dh = self.abstarct_state_ph.first_partial_deriv(CP.iT, CP.iHmass, CP.iP)
        return dT_dh

    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_s_ph(self, p, h):
        self.set_state_ph(p, h)
        s = self.abstarct_state_ph.smass()
        return s
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_ds_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        ds_dp = self.abstarct_state_ph.first_partial_deriv(CP.iSmass, CP.iP, CP.iHmass)
        return ds_dp
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_ds_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        ds_dh = self.abstarct_state_ph.first_partial_deriv(CP.iSmass, CP.iHmass, CP.iP)
        return ds_dh
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_k_ph(self, p, h):
        self.set_state_ph(p, h)
        k = self.abstarct_state_ph.conductivity()
        return k
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dk_ph_dp(self, p, h):
        self.set_state_ph(p, h)
        try:
            dk_dp = self.abstarct_state_ph.first_partial_deriv(CP.iconductivity, CP.iP, CP.iHmass)
        except:
            if not self.disable_warnings:
                print("Warning: partial derivative of k_ph w.r.t. p failed, using finite difference instead")
            eps = 1e-3
            dk_dp = (self.eval_k_ph(p+eps, h) - self.eval_k_ph(p-eps, h)) / (2*eps)
        return dk_dp
    
    @functools.lru_cache(maxsize=scalar_cache_maxsize)
    def eval_dk_ph_dh(self, p, h):
        self.set_state_ph(p, h)
        try:
            dk_dh = self.abstarct_state_ph.first_partial_deriv(CP.iconductivity, CP.iHmass, CP.iP)
        except:
            if not self.disable_warnings:
                print("Warning: partial derivative of k_ph w.r.t. h failed, using finite difference instead")
            eps = 1e-3
            dk_dh = (self.eval_k_ph(p, h+eps) - self.eval_k_ph(p, h-eps)) / (2*eps)
        return dk_dh

    def get_default_vars(self):
        return self.default_vars

    def get_lru_chache_info_str(self, func, indent=0):
        name = func.__name__
        hits = func.cache_info().hits
        misses = func.cache_info().misses
        calls = hits + misses
        if calls > 0:
            return f"{' '*indent}{name}: ({calls} calls, {hits} hits, {misses} misses - {hits / (hits + misses) * 100 if (hits + misses) > 0 else 0:.1f}% cache efficiency)"
        else:
            return f"{' '*indent}{name}: (0 calls)"

    def print_cache_info(self):
        print("Medium:", self.medium, "- cache info:")
        total_calls = 0
        total_hits = 0
        total_misses = 0
        for m in self.modules:
            func = list(m.values())[0]
            total_hits += func.cache_info().hits
            total_misses += func.cache_info().misses
            total_calls += func.cache_info().hits + func.cache_info().misses
            print(self.get_lru_chache_info_str(func, indent=2))

        #total cache efficiency
        total_efficiency = total_hits / (total_hits + total_misses) * 100 if (total_hits + total_misses) > 0 else 0
        print(f"Total cache efficiency: {total_efficiency:.1f}%")
    
    def clear_cache(self):
        self.set_state_ph.cache_clear()
        self.set_state_pT.cache_clear()
        self.set_state_ps.cache_clear() 

def get_symbolic_property_function(eval_func, deriv_funcs, args_names, medium_name, function_name=None):

    class Symbolic_property(sp.Function):

        @classmethod
        def eval(cls, *args):
            if all(arg.is_number for arg in args):
                return eval_func(*args)

        def fdiff(self, argindex=1):
            if argindex not in deriv_funcs:
                raise NotImplementedError(f"Derivative w.r.t. argument {argindex} not defined")
            wrt = args_names[argindex - 1]
            deriv_class = get_symbolic_property_function(deriv_funcs[argindex], {}, args_names, medium_name, f"d{function_name}_d{wrt}")
            return deriv_class(self.args[0], self.args[1])
        
        def _inv(self, *args):
            print("calling _eval_inverse")
            return super()._inv(*args)

    
    NewName = type(f'{medium_name}_{function_name}', (Symbolic_property,), {}) # change name of class
    return NewName

# Example: Using MyCFunc with a CoolProp h_pT function
# symbolic_medium_test = CoolPropMedium("CO2")

# def test_medium_function(function, symbol_1, symbol_2):
#     print(f"Testing {function.__name__} with {symbol_1} and {symbol_2}")
#     base_f = function(symbol_1, symbol_2)
#     print(f"base_f: {base_f}")
#     d_base_f_d_symbol_1 = base_f.diff(symbol_1)
#     print(f"d_base_f_d_symbol_1: {d_base_f_d_symbol_1}")
#     d_base_f_d_symbol_2 = base_f.diff(symbol_2)
#     print(f"d_base_f_d_symbol_2: {d_base_f_d_symbol_2}")

#     base_f.inverse()

#     eqs = [base_f, d_base_f_d_symbol_1, d_base_f_d_symbol_2, base_f_inv]
#     eqs_lambdify = sp.lambdify([symbol_1, symbol_2], eqs, modules=symbolic_medium_test.modules, cse=True)
#     start_time = time.time()
#     p_val, T_val, h_val = [symbolic_medium_test.default_vars[var] for var in ['p', 'T', 'h']]
#     print(f"Default vars: p: {p_val}, T: {T_val}, h: {h_val}")
#     arg_1 = symbolic_medium_test.default_vars[str(symbol_1)]
#     arg_2 = symbolic_medium_test.default_vars[str(symbol_2)]
#     print(f"arg_1: {arg_1}, arg_2: {arg_2}")
#     eqs_val = eqs_lambdify(arg_1, arg_2)
#     print(eqs_val)
#     end_time = time.time()
#     print(f"Time taken for array: {end_time - start_time} seconds")
#     print()


# # # Evaluate numerically at p=1e5 Pa, T=300 K
# test_medium_function(symbolic_medium_test.h_pT, *sp.symbols('p T'))
# test_medium_function(symbolic_medium_test.rho_ph, *sp.symbols('p h'))
# test_medium_function(symbolic_medium_test.mu_ph, *sp.symbols('p h'))
# test_medium_function(symbolic_medium_test.T_ph, *sp.symbols('p h'))
# test_medium_function(symbolic_medium_test.s_ph, *sp.symbols('p h'))
# test_medium_function(symbolic_medium_test.k_ph, *sp.symbols('p h'))
# symbolic_medium_test.print_cache_info()
# exit()

class ModelCache:
    def __init__(self, name):
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.cache = OrderedDict()

    def add_hit(self):
        self.hits += 1
        self.calls += 1

    def add_miss(self):
        self.misses += 1
        self.calls += 1

    def load_from_cache(self, key):
        if key in self.cache:
            self.add_hit()
            return self.cache[key]
        self.add_miss()
        return None
    
    def save_to_cache(self, key, value):
        self.cache[key] = value

    @property
    def cache_efficiency(self):
        return self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0

    def cache_info(self):
        return _CacheInfo(
            self.hits,
            self.misses,
            self.maxsize,
            self.calls
        )
    
    def __repr__(self, title=None):
        return f"{title}: ({self.calls} calls, {self.hits} hits, {self.misses} misses - {self.cache_efficiency:.1f}% cache efficiency)"

class Model:
    def __init__(self):
        self.can_evaluate = True
        self.components = {}
        self.t_symbols = [sp.symbols('t', real=True), sp.symbols('t_prev', real=True), sp.symbols('dt', real=True)]
        self.t_values = [0.0, 0.0, 0.0]
        self.declare_components()
        self.raw_vars_references, self.raw_param_references = self.get_vars_references()
        self.record = {'time': [], 'state': [], 'vars_names': [v.name for v in self.raw_vars_references], 'subs': []}

    def set_name(self, name):
        self.name = name

    def __getitem__(self, name):
        c = self.components[name]
        if isinstance(c, Parameter) or isinstance(c, Variable):
            return c
        else:
            return c
    
    def add_component(self, name, component):
        if isinstance(component, DifferentialVariable):
            self.components[f"der_{name}"] = component.get_derivative_variable()
            self.components[f"der_{name}"].set_name(f"der_{name}")
            #print(self.components[f"der({name})"])

        #elif isinstance(component, Model):
            #component.t_symbols = self.t_symbols
            #component.t_values = self.t_values
        self.components[name] = component
        self.components[name].set_name(name)

    def declare_components(self):
        pass
    
    def declare_equations(self):
        return []

    def is_composite(self):
        for c in self.components.values():
            if isinstance(c, Model):
                return True
        return False
    
    def get_vars_references(self):
        vars_references = []
        param_references = []
        for name, c in self.components.items():
            if isinstance(c, Parameter):
                param_references.append(c)
            if isinstance(c, Variable) and not c.is_connected:
                vars_references.append(c)
            else:
                v, p = c.get_vars_references()
                vars_references.extend(v)
                param_references.extend(p)
        return vars_references, param_references

    @line_profiler.profile
    def assign_symbols(self, prefix="", dotted_prefix="", top_level=False, vars=[], prev_vars=[], params=[], vars_map={}, params_map={}, subs={}):
        vars_idx = len(vars_map)
        params_idx = len(params_map)
        if top_level:
            print("Assigning symbols")
            if not dotted_prefix:
                # Root the dotted path at the top-level model so leaves get names like
                # "System.ambient_inlet.p_out". Use the explicitly assigned name if there
                # is one, otherwise fall back to the class name.
                dotted_prefix = getattr(self, 'name', None) or self.__class__.__name__
        for c in self.components.values():
            
            if prefix:
              compound_name = f"{prefix}_{c.name}"
            else:
              compound_name = c.name
            full_name = f"{dotted_prefix}.{c.name}" if dotted_prefix else c.name
            if c.is_composite():
                c.assign_symbols(prefix=compound_name, dotted_prefix=full_name, vars=vars, prev_vars=prev_vars, params=params, vars_map=vars_map, params_map=params_map)
            else:
                if isinstance(c, Variable):
                    c.set_symbol(sp.Symbol(f"{compound_name}", real=True))
                    c.set_prev_symbol(sp.Symbol(f"{compound_name}_prev", real=True))
                    c.full_name = full_name
                    vars.append(c.symbol)
                    prev_vars.append(c.prev_symbol)
                    vars_map[c] = vars_idx
                    vars_idx += 1
                elif isinstance(c, Parameter):
                    c.set_symbol(sp.Symbol(f"{compound_name}", real=True))
                    c.full_name = full_name
                    params.append(c.symbol)
                    params_map[c] = params_idx
                    params_idx += 1
                else:
                    raise ValueError(f"Unknown component type: {type(c)}")
        if top_level:
            print(f"Symbols assigned")
            return vars, prev_vars, params, vars_map, params_map
    
    @line_profiler.profile
    def collect_equations(self):
        eqs = []
        eqs.extend(self.declare_equations())

        for c in self.components.values():
            if c.is_composite():
                eqs.extend(c.collect_equations())
            elif isinstance(c, DifferentialVariable):
                eqs.extend(c.declare_equations())
        return eqs

    @line_profiler.profile
    def get_current_system(self):
        vars, prev_vars, params, vars_map, params_map = self.assign_symbols(top_level=True)
        equations = self.collect_equations()
        return equations, vars, prev_vars, params

    @line_profiler.profile
    def remove_trivial_equations(self, equations, var_symbols):
        """
        Process equations to remove trivial ones (e.g., x - y = 0 or x - 5 = 0) and apply substitutions.
        Returns reduced equations, updated variable symbols, and substitutions.
        """
        substitutions = {}
        new_eqs = []
        removed_eqs = []
        new_eqs_idx = []
        
        removed_vars = set()
        
        # Convert to list to ensure consistent ordering
        var_symbols_list = list(var_symbols)

        # Map each current-variable symbol to its previous-step counterpart so we can
        # mirror any substitution we make on the "current" side onto the "previous" side
        # (e.g. der_y = -y  =>  der_y_prev = -y_prev). Built once, reused per equation.
        current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}
        
        # Step 1: Identify and solve trivial equations, applying substitutions immediately
        keep_eq = False
        print(f"Identifying trivial equations")
        for eq in equations:
            # Apply any existing substitutions to this equation first
            #eq_substituted = eq.subs(substitutions)
            eq_substituted = eq
            
            # Check if the equation is linear and involves two variables (e.g., x - y = 0)
            vars_in_eq = eq_substituted.free_symbols
            if len(vars_in_eq) == 2:
                if eq_substituted.is_polynomial() and eq_substituted.as_poly().degree() == 1:
                    # Equations like x - y = 0
                    # Sort variables to ensure consistent ordering
                    eq_substituted = eq_substituted.xreplace(substitutions)
                    vars_in_eq_list = sorted(list(vars_in_eq), key=str)
                    var1, var2 = vars_in_eq_list
                    try:
                        sol = sp.solve(eq_substituted, var1)
                        if sol:  # If solvable for var1
                            substitutions[var1] = sol[0]
                            removed_vars.add(var1)

                            # Mirror the substitution onto the prev_symbol layer. For ANY
                            # solved RHS (a bare symbol like `y`, a negated symbol `-y`,
                            # or a more general linear expression `2*y - z + 5`), the
                            # equivalent constraint at the previous time step is obtained
                            # by replacing every current-variable symbol with its
                            # prev_symbol. xreplace handles all of those cases uniformly,
                            # unlike the previous `var.symbol == sol[0]` lookup which
                            # silently failed for non-Symbol RHSs and reused a stale
                            # `sol_0_prev_symbol` from an earlier equation.
                            var1_prev_symbol = current_to_prev.get(var1)
                            if var1_prev_symbol is not None:
                                sol_0_prev = sol[0].xreplace(current_to_prev)
                                substitutions[var1_prev_symbol] = sol_0_prev
                                removed_vars.add(var1_prev_symbol)
                            # Don't add this equation to new_eqs since it's been solved
                            continue
                        else:
                            keep_eq = True
                    except:
                        keep_eq = True
                else:
                    keep_eq = True
            else:
                keep_eq = True
            if keep_eq:
                new_eqs.append(eq_substituted)

        # Step 2: Apply final substitutions to all remaining equations
        simplified_eqs = []
    
        # for eq in new_eqs:
        #     eq_final = eq.subs(substitutions)
        #     simplified_eqs.append(eq_final)

        print(f"Applying substitutions")
        for k, sub in enumerate(substitutions.items()): # a little faster than going all sub into one eq by one
            for i in range(len(new_eqs)):
                new_eqs[i] = new_eqs[i].subs(*sub)
                # report progres in one line
            print(f"Applied {k+1}/{len(substitutions)} substitutions ({ (k+1)/len(substitutions)*100:.2f}%)", end="\r")
        if len(substitutions) == 0:
            print("No substitutions applied")
        print()
        simplified_eqs = new_eqs

        # Step 3: Resolve chains in substitutions so each RHS only references surviving
        # symbols. Without this, an entry like {a: b} can become stale once b itself is
        # eliminated by a later trivial equation. The downstream consumer (plot
        # reconstruction) relies on every RHS being expressed in surviving symbols only.
        for _ in range(len(substitutions) + 1):
            changed = False
            for k in list(substitutions.keys()):
                new_v = substitutions[k].xreplace(substitutions)
                if new_v != substitutions[k]:
                    substitutions[k] = new_v
                    changed = True
            if not changed:
                break

        # Step 4: Update variable symbols (remove substituted variables)
        updated_var_symbols = [v for v in var_symbols_list if v not in removed_vars]

        return simplified_eqs, updated_var_symbols, substitutions

    @line_profiler.profile
    def instantiate(self, cse=True, aditional_modules=[], max_remove_trival_passes=1):
        all_modules = ["numpy"] + aditional_modules

        print("Instantiating model")

        #collect raw initial system
        start_time = time.time()
        self.all_raw_equations, self.raw_var_symbols, self.raw_prev_var_symbols, self.raw_param_symbols = self.get_current_system()
        self.all_raw_symbols = self.raw_var_symbols + self.raw_prev_var_symbols + self.raw_param_symbols + self.t_symbols
        # `assign_symbols` (called by get_current_system) just stamped each leaf with a
        # dotted hierarchical name like "System.ambient_inlet.p_out". Use those for
        # plotting so legends are unambiguous; fall back to the leaf name if a Variable
        # was somehow added without going through assign_symbols.
        self.record['vars_names'] = [getattr(v, 'full_name', v.name) for v in self.raw_vars_references]
        print(len(self.all_raw_symbols))
        print(f"Current system collected in {time.time() - start_time} s")

        # remove most trivial equations
        start_time = time.time()

        # initial improved system of equations and variables
        self.improved_vars = self.raw_var_symbols
        self.improved_equations = self.all_raw_equations
        self.all_improved_symbols = self.all_raw_symbols
        self.improve_subs = {}

        # remove trivial equations of form x - y = 0
        if max_remove_trival_passes > 0:
            print("Removing trivial equations")
            curent_size = len(self.improved_vars)
            print(f"Original variables: {curent_size}")
            start_time = time.time()
        
            for i in range(max_remove_trival_passes):
                print(f"Removing trivial pass {i+1}")
                self.improved_equations, self.all_improved_symbols, pass_subs = self.remove_trivial_equations(self.improved_equations, self.all_improved_symbols)
                # Accumulate substitutions across passes: push the new pass's subs through
                # the RHS of previously stored ones so every entry stays expressed in terms
                # of the latest-surviving symbols, then merge the new entries in.
                for prev_key in list(self.improve_subs.keys()):
                    self.improve_subs[prev_key] = self.improve_subs[prev_key].xreplace(pass_subs)
                self.improve_subs.update(pass_subs)
                new_size = len(self.improved_equations)
                if new_size == curent_size:
                    break
                curent_size = len(self.improved_equations)
            self.improved_vars = [s for s in self.raw_var_symbols if s in self.all_improved_symbols]
            stop_time = time.time()
            print(f"Removed equations and variables: {len(self.raw_var_symbols) - len(self.improved_vars)} in {stop_time - start_time} s")

        # define arrays for improved system
        self.n_v = len(self.improved_vars)
        self.n_p = len(self.raw_param_symbols)
        self.n_t = len(self.t_symbols)
        print(f"Remaining variables and equations: {self.n_v}")
        self.values = np.zeros(2*self.n_v + self.n_p + self.n_t)
        self.delta_values = np.zeros(self.n_v)
        self.improved_equations = sp.Matrix(self.improved_equations)
        self.all_improved_symbols_matrix = sp.Matrix(self.all_improved_symbols)
            
        print("Lambdifying improved equations")
        start_time = time.time()
        self.lambdified_eqs = lambdify_compat(self.all_improved_symbols_matrix, self.improved_equations, modules=all_modules, cse=cse, docstring_limit=-1)
        print(f"Equations lambdified in {time.time() - start_time} s")
        start_time = time.time()
        self.jacobian = self.improved_equations.jacobian(self.improved_vars)
        print(f"Jacobian generated in {time.time() - start_time} s")
        start_time = time.time()
        self.lambdified_jacobian = lambdify_compat(self.all_improved_symbols_matrix, self.jacobian, modules=all_modules, cse=cse, docstring_limit=-1)
        print(f"Jacobian lambdified in {time.time() - start_time} s")
        self.active_vars_references = [var for var in self.raw_vars_references if var.symbol in self.improved_vars]

        # Build a function that, given the current improved-state vector, returns the FULL
        # set of original variables in `raw_vars_references` order. Surviving variables map
        # to themselves; eliminated ones are rebuilt from their stored substitution
        # expressions, which by now reference only surviving symbols.
        raw_var_exprs = [self.improve_subs.get(var.symbol, var.symbol) for var in self.raw_vars_references]
        self.raw_vars_matrix = sp.Matrix(raw_var_exprs) if raw_var_exprs else sp.Matrix([0])
        self.lambdified_raw_vars = lambdify_compat(self.all_improved_symbols_matrix, self.raw_vars_matrix, modules=all_modules, cse=cse, docstring_limit=-1)
        self.record['subs'] = self.improve_subs
        stop_time = time.time()

        # else:
        #     self.n_v = len(self.raw_var_symbols)
        #     self.n_p = len(self.raw_param_symbols)
        #     self.n_t = len(self.t_symbols)
        #     print(f"Active variables: {self.n_v}")
        #     self.values = np.zeros(2*self.n_v + self.n_p + self.n_t)
        #     self.delta_values = np.zeros(self.n_v)
        #     self.all_raw_symbols_matrix = sp.Matrix(self.all_raw_symbols)
        #     self.all_raw_equations = sp.Matrix(self.all_raw_equations)
        #     print("Lambdifying equations")
        #     start_time = time.time()
        #     self.lambdified_eqs = sp.lambdify(self.all_raw_symbols_matrix, self.all_raw_equations, modules=all_modules, cse=cse, docstring_limit=-1)
        #     self.jacobian = self.all_raw_equations.jacobian(self.raw_var_symbols)
        #     self.lambdified_jacobian = sp.lambdify(self.all_raw_symbols_matrix, self.jacobian, modules=all_modules, cse=cse, docstring_limit=-1)
        #     self.active_vars_references = self.raw_vars_references
        #     stop_time = time.time()
        #     print(f"Time taken to lambdify equations: {stop_time - start_time} seconds")
        # #self.inv_jacobian = self.jacobian.inv()
        # #self.delta = sp.simplify(sp.transpose(self.inv_jacobian @ self.equations))
        # #self.lambdified_delta = sp.lambdify(all_symbols_matrix, self.delta, cse=cse, docstring_limit=-1)
        # #self.lambdified_delta = numba.jit(self.lambdified_delta, nopython=True)
        # #self.lambdified_jacobian = sp.lambdify(all_symbols_matrix, self.jacobian, modules=["numpy"], cse=cse)
        # #self.lambdified_inv_jacobian = sp.lambdify(all_symbols_matrix, self.inv_jacobian, modules=["numpy"], cse=cse)
        # #self.error_norm = sp.sqrt(self.equations.dot(self.equations))
        # #self.lambdified_error_norm = sp.lambdify(all_symbols_matrix, self.error_norm, modules=["numpy"], cse=cse)

    def get_vars_values(self):
        return self.values[:self.n_v]
    
    def get_prev_vars_values(self):
        return self.values[self.n_v:2*self.n_v]

    def get_param_values(self):
        return self.values[2*self.n_v:2*self.n_v+self.n_p]
    
    def get_t_values(self):
        return self.values[2*self.n_v+self.n_p:2*self.n_v+self.n_p+3]
    
    def set_vars_values(self, vars):
        self.values[:self.n_v] = vars

    def set_param_values(self, params):
        self.values[2*self.n_v:2*self.n_v+self.n_p] = params

    def set_prev_vars_values(self, prev_vars):
        self.values[self.n_v:2*self.n_v] = prev_vars

    def set_t_values(self, t_values):
        self.values[2*self.n_v+self.n_p:2*self.n_v+self.n_p+3] = t_values

    @property
    def time(self):
        return self.t_symbols[0]

    @property
    def t_prev(self):
        return self.t_symbols[1]

    @property
    def dt(self):
        return self.t_symbols[2]

    def get_dt_value(self):
        return self.values[2*self.n_v+self.n_p+2]

    def get_t_value(self):
        return self.values[2*self.n_v+self.n_p]
    
    def get_t_prev_value(self):
        return self.values[2*self.n_v+self.n_p+1]

    def set_dt(self, dt):
        self.values[2*self.n_v+self.n_p+2] = dt

    def set_t(self, t):
        self.values[2*self.n_v+self.n_p] = t
    
    def set_t_prev(self, t_prev):
        self.values[2*self.n_v+self.n_p+1] = t_prev

    def eval_residuals(self, vars):
        self.set_vars_values(vars)
        return self.lambdified_eqs(*self.values)

    def eval_jacobian(self, vars):
        self.set_vars_values(vars)
        return self.lambdified_jacobian(*self.values)

    # def eval_inv_jacobian(self, vars):
    #     self.set_vars_values(vars)
    #     return self.reduced_lambdified_inv_jacobian(*self.all_reduced_values)
    
    @line_profiler.profile
    def eval_delta(self):
        self.delta_values[:] = self.lambdified_delta(*self.values)
    
    # def eval_error_norm(self, vars):
    #     self.set_vars_values(vars)
    #     return self.reduced_lambdified_error_norm(*self.all_reduced_values)
    
    def record_state(self):
        self.record['time'].append(self.get_t_value())
        full_state = np.asarray(self.lambdified_raw_vars(*self.values)).reshape(-1)
        self.record['state'].append(full_state)
    
    def next_step(self):
        self.set_prev_vars_values(self.get_vars_values())
        self.set_t_prev(self.get_t_value())
        #self.set_t(self.get_t_value() + self.get_dt_value())
        self.record_state()

    @line_profiler.profile
    def initialise(self, n=1):
        self.set_t_values([0.0, 0.0, 0.0])
        init_values = np.array([var.value for var in self.active_vars_references])
        init_params = np.array([param.value for param in self.raw_param_references])
        self.set_vars_values(init_values)
        self.set_prev_vars_values(init_values)
        self.set_param_values(init_params)
        self.custom_solve()
        self.next_step()

    def update_delta(self):
        j = self.eval_jacobian(self.get_vars_values())
        r = self.eval_residuals(self.get_vars_values())
        self.delta_values[:] = fast_linear_solve(j, r).T[0]

    @line_profiler.profile
    def custom_solve(self, tol=1e-6, max_iter = 100, relaxation=1.0):
        guess = self.get_vars_values()
        error_norm = np.inf
        i = 0
        while error_norm > tol and i < max_iter:
            self.update_delta()
            self.delta_values *= relaxation
            error_norm = fast_error_norm(self.delta_values)
            guess -= self.delta_values
            i += 1
        return guess
    
    @line_profiler.profile
    def solve_dae_step(self, dt):
        self.set_dt(dt)
        self.set_t(self.get_t_value() + dt)
        self.custom_solve()
    
    def set_initial_time(self, t):
        for c in self.components.values():
            if isinstance(c, Model):
                c.set_initial_time(t)
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"

    def get_lru_chache_info_str(self, func, indent=0):
        name = func.__name__
        hits = func.cache_info().hits
        misses = func.cache_info().misses
        calls = hits + misses
        return f"{' '*indent}{name}: ({calls} calls, {hits} hits, {misses} misses - {hits / (hits + misses) * 100 if (hits + misses) > 0 else 0:.1f}% cache efficiency)"
    
    def print_info(self, indent=0):
        print(f"{' '*indent}{self.__class__.__name__}:")
        print(f"{' '*indent} -{self.cache_evaluate.__repr__('Evaluate')}")
        print(f"{' '*indent} -{self.cache_evaluate_wrapper.__repr__('Evaluate wrapper')}")
        print(f"{' '*indent} -{self.cache_jacobian.__repr__('Jacobian')}")
        for name, c in self.components.items():
            if not isinstance(c, (Parameter, Variable)):
                c.print_info(indent=indent+2)
    
class Parameter(Model):
    def __init__(self, value, unit=None):
        self.components = {}
        self.symbol = None
        self.value = value
        self.unit = unit
        self.can_evaluate = False

    def free(self):
        return Variable(self.value, self.unit)
    
    def set_symbol(self, symbol):
        self.symbol = symbol

    def get_vars_len(self):
        return 0
    
    def check(self):
        return True
    
    def get_vars(self):
        return []
    
    def get_equations(self):
        return []
    
    def setup_t_values_referece(self, t_values):
        pass
    
    def __repr__(self):
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{super().__repr__()} with value: {self.value}{unit_str}"
        
class Variable(Model):
    def __init__(self, value, unit=None):
        self.components = {}
        self.value = value
        self._symbold_index = None
        self.symbol = None
        self.prev_value = value
        self.prev_symbol = None
        self.unit = unit
        self.can_evaluate = False
        self.is_connected = False
        self.connected_to = []
        self.t_symbols = [sp.symbols('t', real=True), sp.symbols('t_prev', real=True), sp.symbols('dt', real=True)]


    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, value):
        self._symbol = value

    def fix(self):
        self = Parameter(self.value, self.unit)

    def check(self):
        return True

    def get_vars_len(self):
        return 1
    
    def set_all_connected(self, attr, value, used_list=None):
        if used_list is None:
            used_list = []
        used_list.append(self)
        setattr(self, attr, value)
        for connection in self.connected_to:
            if connection not in used_list:
                connection.set_all_connected(attr, value, used_list)
    
    def set_value(self, value, used_list=None):
        self.set_all_connected('value', value, used_list)

    def set_prev_value(self, value):
        self.set_all_connected('prev_value', value)

    def set_symbol(self, symbol):
        self.set_all_connected('symbol', symbol)

    def set_prev_symbol(self, symbol):
        self.set_all_connected('prev_symbol', symbol)

    def set_initial_time(self, t):
        pass

    def next_step(self):
        pass

    def setup_t_values_referece(self, t_values):
        self.t_values = t_values
    
    def __repr__(self):
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{super().__repr__()} with value: {self.value}{unit_str}"
 
class DifferentialVariable(Variable):
    def __init__(self, value, unit=None):
        super().__init__(value, unit)
        self.der_variable = Variable(0.0, None)

    def declare_equations(self):
        #crank nicolson scheme 
        #TODO: add more multistep schemes
        eq1 = self.symbol - (self.prev_symbol + 0.5 * self.dt * (self.der_variable.symbol + self.der_variable.prev_symbol))
        return [eq1]
        
    def get_derivative_variable(self):
        return self.der_variable

def plot_results(record, filename, show=False, max_vars=None):
    import plotly.graph_objects as go

    t = record['time']
    # `state` is produced by Model.lambdified_raw_vars, so it already covers every
    # original variable (including those eliminated by trivial-equation removal,
    # reconstructed from their substitutions). Columns line up with `vars_names`.
    y = np.array(record['state']).T
    vars_names = list(record['vars_names'])

    if y.shape[0] != len(vars_names):
        raise ValueError(
            f"plot_results: state has {y.shape[0]} columns but {len(vars_names)} variable "
            f"names were recorded. Did instantiate() finish before the first record_state()?"
        )

    if max_vars is not None:
        vars_names = vars_names[:max_vars]
        y = y[:len(vars_names)]

    fig = go.Figure()
    for i, name in enumerate(vars_names):
        fig.add_trace(go.Scatter(
            x=t,
            y=y[i],
            mode='lines',
            name=name,
            line=dict(width=2),
        ))

    fig.update_layout(
        title='Simulation Results',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
    )

    if show:
        fig.show()
    fig.write_html(filename)

class InnerODE_1(Model):
    def __init__(self):
        super().__init__()

    def declare_components(self):
        self.add_component('p', Parameter(1, "m/s"))
        self.add_component('variable', DifferentialVariable(0.1, "m/s"))
        self.add_component('dummy', Variable(0.0, "m/s"))
        
    def declare_equations(self):
        eq2 = self['der_variable'].symbol - self['p'].symbol * self['dummy'].symbol
        return [eq2]

class InnerODE_2(Model):
    def __init__(self):
        super().__init__()

    def declare_components(self):
        self.add_component('variable', DifferentialVariable(0.1, "m/s"))
        
    def declare_equations(self):
        eq2 = self['der_variable'].symbol - self['variable'].symbol
        return [eq2]

class SimpleODE(Model):
    def declare_components(self):
        self.add_component('inner_ode1', InnerODE_1())
        self.add_component('inner_ode2', InnerODE_2())
        #self.add_connection(self['inner_ode1']['dummy'], self['inner_ode2']['variable'])

    def declare_equations(self):
        eq1 = self['inner_ode1']['dummy'].symbol - self['inner_ode2']['variable'].symbol
        return [eq1]

# ode = SimpleODE()
# ode.instantiate(max_remove_trival_passes=1)
# ode.initialise(n=1)


# start_time = time.time()
# for i in range(25):
#     print("step", i)
#     ode.solve_dae_step(0.04)  # Reduced time step for stability
#     ode.next_step()
# print(f"Time taken: {time.time() - start_time} seconds")
# plot_results(ode.record, "ode.html", show=True)
# exit()

class AmbientInlet(Model):
    def __init__(self, medium: CoolPropMedium, p_ambient=101325, T_ambient=293.15, m_flow=0.1, D=0.07):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self.m_flow = m_flow
        self.D = D
        super().__init__()

    def declare_components(self):
        self.add_component('p_ambient', Parameter(self.p_ambient, "Pa"))
        self.add_component('T_ambient', Variable(self.T_ambient, "K"))
        self.add_component('h_ambient', Parameter(self.medium.h_pT(self['p_ambient'].value, self['T_ambient'].value), "J/kg"))
        self.add_component('s_ambient', Parameter(self.medium.s_ph(self['p_ambient'].value, self['h_ambient'].value), "J/kg/K"))
        self.add_component('m_flow', Parameter(self.m_flow, "kg/s"))
        self.add_component('D', Parameter(self.D, "m"))
        self.add_component('p_out', Variable(self.p_ambient*0.99, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(self.p_ambient, self.T_ambient)*0.99, "J/kg"))
        self.add_component('w_out', Variable(0.2, "m/s"))

    def declare_equations(self):
        A = np.pi * self['D'].symbol**2 / 4
        eq1 = self['m_flow'].symbol - self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol) * self['w_out'].symbol * A

        h_in = self['h_ambient'].symbol
        s_in = self['s_ambient'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq2 = s_in - s_out
        eq3 = h_in - (self['h_out'].symbol + self['w_out'].symbol**2 / 2)

        eq4 = self['T_ambient'].symbol - self.T_ambient

        res = [eq1, eq2, eq3, eq4]
        return res
    
medium = CoolPropMedium('water')

# test medium
#print(medium.h_pT(101325, 293.15))

#test ambient inlet
# ambient_inlet = AmbientInlet(medium, p_ambient=101325, T_ambient=293.15, m_flow=0.01, D=0.07)
# ambient_inlet.instantiate(aditional_modules=medium.modules)
# ambient_inlet.initialise(n=1)
# print(ambient_inlet['h_out'].value)
# exit()
# print(ambient_inlet.record['state'][-1])

# exit()

        
    
class AmbientOutlet(Model):
    def __init__(self, medium: CoolPropMedium, p_ambient=101325, T_ambient=293.15, m_flow=0.1, D=0.07):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self.m_flow = m_flow
        self.D = D
        super().__init__()

    def declare_components(self):
        self.add_component('p_ambient', Parameter(self.p_ambient, "Pa"))
        self.add_component('T_ambient', Parameter(self.T_ambient, "K"))
        self.add_component('h_ambient', Parameter(self.medium.h_pT(self['p_ambient'].value, self['T_ambient'].value), "J/kg"))
        self.add_component('s_ambient', Parameter(self.medium.s_ph(self['p_ambient'].value, self['h_ambient'].value), "J/kg/K"))
        self.add_component('m_flow', Parameter(self.m_flow, "kg/s"))
        self.add_component('D', Parameter(self.D, "m"))
        self.add_component('p_out', Variable(self.p_ambient*0.99, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(self.p_ambient, self.T_ambient)*0.99, "J/kg"))
        self.add_component('w_out', Variable(0.2, "m/s"))

    def declare_equations(self):
        A = np.pi * self['D'].symbol**2 / 4
        eq1 = self['m_flow'].symbol - self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol) * self['w_out'].symbol * A

        h_in = self['h_ambient'].symbol
        s_in = self['s_ambient'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq2 = s_in - s_out
        eq3 = h_in - (self['h_out'].symbol + self['w_out'].symbol**2 / 2)

        res = [eq1, eq2, eq3]
        return res

class TwoPortSegment(Model):
    def __init__(self, medium: CoolPropMedium, A_in, A_out, P_in, P_out, z_in, z_out, L, epsilon, f_factor_func, q_inflow_func):
        self.medium = medium
        self.A_in = A_in
        self.A_out = A_out
        self.P_in = P_in
        self.P_out = P_out
        self.z_in = z_in
        self.z_out = z_out
        self.L = L
        self.epsilon = epsilon
        self.f_factor_func = f_factor_func
        self.q_inflow_func = q_inflow_func
        super().__init__()
        
    def declare_components(self):
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('w_in', Variable(0.1, "m/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('w_out', Variable(0.1, "m/s"))
        self.add_component('A_in', Parameter(self.A_in, "m"))
        self.add_component('A_out', Parameter(self.A_out, "m"))
        self.add_component('P_in', Parameter(self.P_in, "m"))
        self.add_component('P_out', Parameter(self.P_out, "m"))
        self.add_component('z_in', Parameter(self.z_in, "m"))
        self.add_component('z_out', Parameter(self.z_out, "m"))
        self.add_component('L', Parameter(self.L, "m"))
        self.add_component('T_wall', Parameter(293.15, "K"))
        self.add_component('q_inflow', Variable(0.0, "W"))

    def declare_equations(self):

       
        # volume averages
        T_in = self.medium.T_ph(self['p_in'].symbol, self['h_in'].symbol)
        rho_in = self.medium.rho_ph(self['p_in'].symbol, self['h_in'].symbol)
        mu_in = self.medium.mu_ph(self['p_in'].symbol, self['h_in'].symbol)
        k_in = self.medium.k_ph(self['p_in'].symbol, self['h_in'].symbol)

        T_out = self.medium.T_ph(self['p_out'].symbol, self['h_out'].symbol)
        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        mu_out = self.medium.mu_ph(self['p_out'].symbol, self['h_out'].symbol)
        k_out = self.medium.k_ph(self['p_out'].symbol, self['h_out'].symbol)

        p_avg = (self['p_in'].symbol + self['p_out'].symbol) / 2
        h_avg = (self['h_in'].symbol + self['h_out'].symbol) / 2
        T_avg = (T_in + T_out) / 2
        mu_avg = (mu_in + mu_out) / 2
        rho_avg = (rho_in + rho_out) / 2
        k_avg = (k_in + k_out) / 2
        w_avg = (self['w_in'].symbol + self['w_out'].symbol) / 2
        A_avg = (self['A_in'].symbol + self['A_out'].symbol) / 2
        P_avg = (self['P_in'].symbol + self['P_out'].symbol) / 2

        Dh_in = 4 * self['A_in'].symbol / self['P_in'].symbol
        Dh_out = 4 * self['A_out'].symbol / self['P_out'].symbol
        Dh_avg = (Dh_in + Dh_out) / 2
        
        mdot = rho_avg * w_avg * A_avg
        Re_avg = rho_avg * abs(w_avg) * Dh_avg / mu_avg

        # Continuity
        eq1 = (rho_in * self['A_in'].symbol * self['w_in'].symbol -
            rho_out * self['A_out'].symbol * self['w_out'].symbol)
        
        # Momentum
        f_avg = self.f_factor_func(Re_avg, self.epsilon, Dh_avg)
        delta_P_friction = f_avg * (self['L'].symbol / Dh_avg) * (rho_avg * abs(w_avg)*w_avg / 2)
        momentum_flux = mdot * (self['w_out'].symbol - self['w_in'].symbol)
        buoyancy_force = -G_const * (self['z_out'].symbol - self['z_in'].symbol) * A_avg * rho_avg
        eq2 = self['p_in'].symbol * self['A_in'].symbol - self['p_out'].symbol * self['A_out'].symbol - delta_P_friction * A_avg - momentum_flux + buoyancy_force

        # Energy
        q = self.q_inflow_func(w_avg, p_avg, h_avg, rho_avg, T_avg, mu_avg, k_avg, f_avg, self['T_wall'].symbol)
        eq3 = self['h_in'].symbol + self['w_in'].symbol**2 / 2 + q - (self['h_out'].symbol + self['w_out'].symbol**2 / 2)

        eq4 = self['q_inflow'].symbol - q

        return [eq1, eq2, eq3, eq4]
        
class AdiabaticPump(TwoPortSegment):
    def __init__(self, medium: CoolPropMedium, A_in, A_out, P_in, P_out, z_in, z_out):
        super().__init__(medium, A_in, A_out, P_in, P_out, z_in, z_out, 1, 0.0, self.get_f_from_a_iz, self.get_q_inflow)
        self.adiabatic = True

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall):
        return 0.0
    
    def get_f_from_a_iz(self, Re_avg, epsilon, Dh_avg):
        return self['a_iz'].symbol / (Re_avg * Dh_avg)
        
    def declare_components(self):
        super().declare_components()
        self.add_component('a_iz', Variable(0.0, "W"))
        
class StraightPipe(Model):
    def __init__(self, medium: CoolPropMedium, D, L, epsilon, z_in, z_out, n_segments=3, adiabatic=False):
        self.medium = medium
        self.D = D
        self.L = L
        self.epsilon = epsilon
        self.z_in = z_in
        self.z_out = z_out
        self.n_segments = n_segments
        self.adiabatic = adiabatic
        super().__init__()

    def get_churchill_f_factor(self, Re, epsilon, D):
        term1 = (8.0 / (Re + 1)) ** 12
        A = (-2.457 * sp.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / (Re + 1)) ** 16
        term2 = 1.0 / (A + B) ** 1.5
        f = (term1 + term2) ** (1.0 / 12.0)
        return f * 8
    
    def calculate_nu(self, Re, Pr, fr):
        U = Re / 1000
        if Re <= 2300:
            nu = (fr / 8) * (Re - 1000) * Pr / (1 + 12.7 * (fr / 8)**0.5 * (Pr**(2/3) - 1))
        elif 2300 < Re <= 3100:
            nu = 3.52 * U**4 - 45.148 * U**3 + 212.13 * U**2 - 427.45 * U + 316.08
        else:
            nu = 3.66
        return nu

    def calculate_nu_smooth(self, Re, Pr, fr):
        #the same as calculate_nu but if statement replaced with smooth transition
        U = Re / 1000
        nu_1 = (fr / 8) * (Re - 1000) * Pr / (1 + 12.7 * (fr / 8)**0.5 * (Pr**(2/3) - 1))
        nu_2 = 3.52 * U**4 - 45.148 * U**3 + 212.13 * U**2 - 427.45 * U + 316.08
        nu_3 = 3.66
        TRANSITION_WIDTH = 100
        TRANSITION_CENTER_1 = 2250
        TRANSITION_CENTER_2 = 3050
        w1 = sp.Max(0.0, sp.Min(1.0, 1.0 - (Re - TRANSITION_CENTER_1) / TRANSITION_WIDTH))
        w3 = sp.Max(0.0, sp.Min(1.0, (Re - TRANSITION_CENTER_2) / TRANSITION_WIDTH))
        w2 = 1.0 - w1 - w3
        nu = w1 * nu_1 + w2 * nu_2 + w3 * nu_3
        return nu
    
    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall):
        Re = w * self.D * rho / mu
        Pr = mu * rho / k
        nu = self.calculate_nu_smooth(Re, Pr, fr)
        alpha = nu * k / self.D
        area = np.pi * self.D * self.L
        return alpha * area * (T_wall - T)
    
    def get_q_inflow_adiabatic(self, w, p, h, rho, T, mu, k, fr, T_wall):
        return 0.0

    def declare_components(self):
        L_segments = self.L / self.n_segments
        A = np.pi * self.D**2 / 4
        P = np.pi * self.D
        dz = (self.z_out - self.z_in) / self.n_segments
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('w_in', Variable(0.1, "m/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('w_out', Variable(0.1, "m/s"))
        
        for i in range(self.n_segments):
            z_in = self.z_in + i * dz
            z_out = self.z_in + (i + 1) * dz
            fr_f = self.get_churchill_f_factor
            q_f = self.get_q_inflow if not self.adiabatic else self.get_q_inflow_adiabatic
            self.add_component(f'pipe_segment_{i}', TwoPortSegment(self.medium, A, A, P, P, z_in, z_out, L_segments, self.epsilon, fr_f, q_f))

    def declare_equations(self):
        res = []
        res.append(self['p_in'].symbol - self['pipe_segment_0']['p_in'].symbol)
        res.append(self['h_in'].symbol - self['pipe_segment_0']['h_out'].symbol)
        res.append(self['w_in'].symbol - self['pipe_segment_0']['w_out'].symbol)

        for i in range(self.n_segments-1):
            res.append(self[f'pipe_segment_{i}']['p_out'].symbol - self[f'pipe_segment_{i+1}']['p_in'].symbol)
            res.append(self[f'pipe_segment_{i}']['h_out'].symbol - self[f'pipe_segment_{i+1}']['h_in'].symbol)
            res.append(self[f'pipe_segment_{i}']['w_out'].symbol - self[f'pipe_segment_{i+1}']['w_in'].symbol)

        res.append(self['p_out'].symbol - self[f'pipe_segment_{self.n_segments-1}']['p_out'].symbol)
        res.append(self['h_out'].symbol - self[f'pipe_segment_{self.n_segments-1}']['h_out'].symbol)
        res.append(self['w_out'].symbol - self[f'pipe_segment_{self.n_segments-1}']['w_out'].symbol)

        return res

N = 3
L = 10
air = CoolPropMedium('air', disable_warnings=True)


class IntegrationTest(Model):
    """
    Decoupled ODEs added purely to validate the time integrator. Each of the
    differential variables below has a closed-form solution that we compare
    against after the simulation loop.

      1. Exponential decay:
             dy/dt = -y,                 y(0) = 1
             analytical:  y(t) = exp(-t)

      2. Harmonic oscillator (omega defaults to 2*pi -> period of 1 s):
             dy/dt = z
             dz/dt = -omega**2 * y,      y(0) = 1, z(0) = 0
             analytical:  y(t) =  cos(omega*t)
                          z(t) = -omega * sin(omega*t)
    """

    def __init__(self, omega=2 * np.pi):
        self.omega_value = omega
        super().__init__()

    def declare_components(self):
        # exponential decay
        self.add_component('y_decay', DifferentialVariable(1.0, None))

        # harmonic oscillator
        self.add_component('omega', Parameter(self.omega_value, "1/s"))
        self.add_component('y_osc', DifferentialVariable(1.0, None))
        self.add_component('z_osc', DifferentialVariable(0.0, None))

    def declare_equations(self):
        eq_decay = self['der_y_decay'].symbol + self['y_decay'].symbol
        eq_osc_y = self['der_y_osc'].symbol - self['z_osc'].symbol
        eq_osc_z = self['der_z_osc'].symbol + self['omega'].symbol ** 2 * self['y_osc'].symbol
        return [eq_decay, eq_osc_y, eq_osc_z]


class System(Model):
    def __init__(self):
        super().__init__()

    def declare_components(self):
        self.add_component('ambient_inlet', AmbientInlet(air, p_ambient=101325, T_ambient=273.15+60, m_flow=0.0745, D=0.0545))
        self.add_component('straight_pipe_1', StraightPipe(air, D=0.0545, L=L, epsilon=0.0001, z_in=0, z_out=0, n_segments=N))
        self.add_component('straight_pipe_2', StraightPipe(air, D=0.0545, L=L, epsilon=0.0001, z_in=0, z_out=0, n_segments=N))
        # decoupled sanity-check ODEs whose values we compare against analytical solutions.
        self.add_component('integration_test', IntegrationTest(omega=2 * np.pi))


    def declare_equations(self):
        res_1 = self['ambient_inlet']['p_out'].symbol - self['straight_pipe_1']['p_in'].symbol
        res_2 = self['ambient_inlet']['h_out'].symbol - self['straight_pipe_1']['h_in'].symbol
        res_3 = self['ambient_inlet']['w_out'].symbol - self['straight_pipe_1']['w_in'].symbol
        print(self['ambient_inlet'].time == self['straight_pipe_1'].time)

        res_4 = self['straight_pipe_1']['p_out'].symbol - self['straight_pipe_2']['p_in'].symbol
        res_5 = self['straight_pipe_1']['h_out'].symbol - self['straight_pipe_2']['h_in'].symbol
        res_6 = self['straight_pipe_1']['w_out'].symbol - self['straight_pipe_2']['w_in'].symbol

        return [res_1, res_2, res_3, res_4, res_5, res_6]
  
model_test = System()
model_test.instantiate(aditional_modules=air.modules, max_remove_trival_passes=5)
model_test.initialise(n=1)

start_time = time.time()
for i in range(25):
    model_test.solve_dae_step(0.04)  # Reduced time step for stability
    model_test.next_step()
print(f"Time taken to solve: {time.time() - start_time} seconds")

# Validate the time integrator against the analytical solutions of IntegrationTest.
# With dt = 0.04 and 25 steps the simulation covers t in [0, 1] s.
record = model_test.record
t_arr = np.array(record['time'])
state_arr = np.array(record['state'])
names = list(record['vars_names'])


def _trace(name):
    return state_arr[:, names.index(name)]


omega_val = 2 * np.pi
y_decay_num = _trace('System.integration_test.y_decay')
y_osc_num = _trace('System.integration_test.y_osc')
z_osc_num = _trace('System.integration_test.z_osc')

y_decay_exact = np.exp(-t_arr)
y_osc_exact = np.cos(omega_val * t_arr)
z_osc_exact = -omega_val * np.sin(omega_val * t_arr)

dt_used = 0.04
expected_decay = 0.5 * t_arr[-1] * dt_used ** 2  # ~|y''(t)|*dt^2/12 summed; for y=exp(-t) this is loose but conservative
phase_drift = abs(t_arr[-1] / dt_used) * (omega_val * dt_used) ** 3 / 12.0
expected_osc_y = np.sin(phase_drift)            # CN preserves amplitude; only phase drifts
expected_osc_z = omega_val * np.sin(phase_drift)

print("--- Time-integration check (IntegrationTest sub-model) ---")
print(f"Exponential decay  y_decay : max |err| = {np.max(np.abs(y_decay_num - y_decay_exact)):.3e}    (CN expected: < {expected_decay:.1e})")
print(f"Harmonic oscillator y_osc  : max |err| = {np.max(np.abs(y_osc_num - y_osc_exact)):.3e}    (CN expected: ~ {expected_osc_y:.1e})")
print(f"Harmonic oscillator z_osc  : max |err| = {np.max(np.abs(z_osc_num - z_osc_exact)):.3e}    (CN expected: ~ {expected_osc_z:.1e})")
print("  (CN preserves oscillator amplitude exactly; the residual is pure phase drift")
print(f"   ~ (omega*dt)^3 / 12 per step = {phase_drift:.3e} rad over the full window.)")

plot_results(model_test.record, "model_test.html", show=False)


