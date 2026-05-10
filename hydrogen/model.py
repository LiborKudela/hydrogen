"""Core symbolic-DAE framework: `Model`, `Parameter`, `Variable`, `DifferentialVariable`.

The framework lets you compose physical components in a tree, declare equations
symbolically (via `sympy`), then call `instantiate()` to:

  1. Walk the tree, assigning each leaf a unique sympy symbol and a dotted hierarchical
     name (e.g. "System.ambient_inlet.p_out") used for plotting.
  2. Collect every component's equations.
  3. Optionally remove trivial equations of the form `x - y = 0` (multiple passes).
     Eliminated variables are reconstructed at record time from the substitution dict
     so plotting still shows them.
  4. Lambdify residuals, jacobian, and a "raw vars" reconstructor function.

Time stepping uses Crank-Nicolson on every `DifferentialVariable`.
"""

from __future__ import annotations

import contextvars
import gc
import os
import time

import line_profiler
import multiprocessing as _mp
import numpy as np
import sympy as sp

from .caching import (
    lambda_cache_default_dir,
    lambda_cache_key,
    load_lambdified_source,
    save_lambdified_source,
)
from .numerics import fast_error_norm, fast_linear_solve, fast_sparse_solve, lambdify_compat


# Module-globals seeded by `_init_lambdify_worker` (only used in worker
# processes spawned by the `multiprocessing.Pool` in `instantiate`).  We pass
# state via globals (fork-inherited from the parent) instead of pickled task
# arguments because:
#
#   - `Symbolic_property` (dynamically-built sympy `Function` subclasses for
#     CoolProp callbacks) cannot be pickled (`attribute lookup … failed`).
#     Templates and modules embed these, so they can never round-trip through
#     `multiprocessing.Pool.map`'s queue.
#   - With the `fork` start method the parent's globals (including these
#     templates and `aditional_modules`) are already present in every worker
#     via copy-on-write -- no pickling required.
_WORKER_MODULES = None
_WORKER_MODULES_SIG = None
_WORKER_CACHE_DIR = None
_WORKER_PAYLOADS = None  # list of (label, key, args_mat, block, cse), indexed by task


# Step B's `declare_equations()` template cache (see `Model.collect_equations`).
#
# Scoped to ONE `Model.instantiate()` call via a `ContextVar` rather than a
# class attribute, because the SAME component class (e.g. `StraightPipe`) can
# legitimately produce DIFFERENT symbolic equations across instantiate calls
# in the same process -- the medium-bound `Symbolic_property` Function classes
# (`Air_rho_ph`, `Hydrogen_rho_ph`, ...) are baked into the cached eqs and
# survive `xreplace` (which only renames Symbol atoms, not outer Function
# nodes).  A class-level cache caused cross-medium contamination when
# `pipe_tree.py` ran Air -> Hydrogen in the same process: Hydrogen's lambdified
# source would emit `Air_rho_ph(...)` calls that didn't exist in Hydrogen's
# module namespace, raising `NameError` at evaluation time.
#
# `instantiate()` sets a fresh dict on entry and tears it down on exit (in a
# `try/finally`) so that any state -- known today or added later -- that
# influences `declare_equations()` is naturally partitioned per call.
# When the var is `None` (e.g. for direct `collect_equations()` calls outside
# of `instantiate()`) the cache is silently disabled.
_eq_cache_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "hydrogen_eq_cache", default=None
)


# Numpy doesn't ship a single-arg `Heaviside` (`np.heaviside` is
# 2-arg: `heaviside(x, h0)`), so `sp.lambdify(..., modules=['numpy', ...])`
# emits the bare name `Heaviside(x)` (sympy <1.10) or `Heaviside(x, 1/2)`
# (sympy >=1.11) which then NameErrors at call time.  `Heaviside` shows up
# in the Jacobian of every `sp.Max`/`sp.Min` -- the `StraightPipe`
# heat-transfer correlation uses both, so any non-adiabatic pipe hits this.
# Accept both arities: when sympy supplies its own `H(0)` value as the
# second positional arg, defer to that; otherwise default to `0.5`.
def _heaviside_compat(x, h0=0.5):
    return np.heaviside(x, h0)


_NUMPY_LAMBDIFY_COMPAT = {
    "Heaviside": _heaviside_compat,
}


def _vectorise_callable(fn):
    """Wrap a scalar `fn(*scalars) -> scalar` so it also works on numpy arrays.

    The medium's CoolProp-backed property functions (e.g. `Air_rho_ph`) are
    Python callables that internally call CoolProp scalar entry points.
    `sp.lambdify` calls them once per (state, instance) when the lambdified
    function is invoked with scalar args.  In the per-template VECTORISED
    Newton path (see `_eval_per_template_vec`), we instead pack
    `n_inst_for_this_template` scalar inputs into numpy arrays of shape
    `(n_inst,)` and call the lambdified function ONCE per template.  That
    requires the medium's callbacks to broadcast across arrays.

    We keep a fast scalar shortcut so that any code path still calling the
    lambdified function with scalars (cached source, get_current_system test
    helpers, ...) doesn't pay the broadcasting overhead.
    """
    # `_orig` lets `_eval_per_template_vec` notice whether a callable has
    # already been wrapped (for idempotence when modules are reused).
    if getattr(fn, "_hydrogen_vectorised", False):
        return fn

    # Hot path: this wrapper sits between the lambdified residual/Jacobian
    # functions and the CoolProp-backed medium callbacks.  In a
    # representative `run_system` Newton solve we measured ~3 MILLION
    # calls into this wrapper -- so EVERY nanosecond of overhead matters.
    #
    # The previous implementation used `any(isinstance(a, np.ndarray) for
    # a in args)` which built a generator + tuple per call (~500 ns).
    # We swap that for a specialised dispatch keyed on `args[0]` only:
    # the per-template eval below always passes either ALL scalars or ALL
    # arrays (every placeholder is gathered from the same `vals_arr`), so
    # checking the first arg is enough.  `type(...) is np.ndarray` skips
    # the MRO walk in `isinstance` (~50 ns total scalar fast-path).
    _ndarray = np.ndarray

    def wrapper(*args):
        if args and type(args[0]) is _ndarray:
            # Array path: hand-rolled tight loop.  np.vectorize was ~5x
            # slower on the 4-50 elements we typically batch -- it
            # negotiates output dtypes, broadcasts shapes, and stamps
            # trip counters on every call.
            bcast = np.broadcast_arrays(*args)
            flat = [np.ascontiguousarray(b).ravel() for b in bcast]
            n = flat[0].size
            out = np.empty(n)
            for i in range(n):
                out[i] = fn(*(f[i] for f in flat))
            return out.reshape(bcast[0].shape)
        return fn(*args)

    wrapper._hydrogen_vectorised = True
    wrapper._hydrogen_orig = fn
    wrapper.__name__ = getattr(fn, "__name__", "vectorised_callable")
    return wrapper


def _wrap_modules_for_vectorisation(aditional_modules):
    """Return a copy of `aditional_modules` whose dict entries have been
    swapped for vectorised wrappers (other entries pass through unchanged).
    Idempotent.
    """
    out = []
    for m in aditional_modules:
        if isinstance(m, dict):
            out.append({k: _vectorise_callable(v) if callable(v) else v
                        for k, v in m.items()})
        else:
            out.append(m)
    return out


def _init_lambdify_worker(modules, modules_sig, cache_dir, payloads):
    global _WORKER_MODULES, _WORKER_MODULES_SIG, _WORKER_CACHE_DIR, _WORKER_PAYLOADS
    _WORKER_MODULES = modules
    _WORKER_MODULES_SIG = modules_sig
    _WORKER_CACHE_DIR = cache_dir
    _WORKER_PAYLOADS = payloads


def _lambdify_worker_task(task_idx):
    """Worker entry: lambdify a single template, write the source to disk, return its key.

    We deliberately do NOT return the lambdified function itself -- function
    objects pickle poorly, and the parent re-loads from the on-disk cache anyway
    (which is a one-time `exec` of stored Python source).  The args/expr live
    in `_WORKER_PAYLOADS`, populated by the Pool initializer; `task_idx` indexes
    into that list.
    """
    label, key, args, expr, cse = _WORKER_PAYLOADS[task_idx]
    func = lambdify_compat(args, expr, modules=_WORKER_MODULES, cse=cse, docstring_limit=-1)
    if _WORKER_CACHE_DIR is not None:
        save_lambdified_source(_WORKER_CACHE_DIR, key, func, _WORKER_MODULES_SIG)
    return task_idx, label, key


class Model:
    # Shared symbols used by every Model/Variable instance.  Previously each
    # `Variable.__init__` re-allocated a fresh triple, which for a system with
    # thousands of variables was thousands of unused sympy `Symbol` objects on
    # the heap.  Sympy's `Symbol` cache makes these structurally identical, but
    # holding distinct Python objects per variable still wastes memory.
    t_symbols = [
        sp.symbols('t', real=True),
        sp.symbols('t_prev', real=True),
        sp.symbols('dt', real=True),
    ]

    # Step B: per-Model-subclass cache of the SYMBOLIC OUTPUT of
    # `declare_equations()`.  After the first instance of a given subclass has
    # run its `declare_equations`, we capture (template_eqs, sym_path_map),
    # validate against the second instance to make sure the structure is
    # genuinely shared, and from the third instance onward we skip the costly
    # CoolProp-laden `Symbolic_property` builds entirely -- replaying the
    # template via `xreplace(symbol_remap)`.
    #
    # The cache lives in `_eq_cache_var` (module-level `ContextVar`) and is
    # scoped to ONE `instantiate()` call -- see the var's docstring for why
    # a class-level cache caused cross-medium contamination.
    #
    # State machine per subclass: 'first' -> 'cached' (after one match) or
    # 'no-cache' (any mismatch / has connections / has side effects).

    def __init__(self):
        self.can_evaluate = True
        self.components = {}
        # Variable-pair connections registered by `declare_equations` via
        # `add_connection`.  These are short-circuited at instantiate time
        # (union-find) instead of being threaded through the symbolic
        # trivial-equation reducer, which is much faster for large systems
        # where the bulk of the trivial equations are connection equalities.
        self.connections = []
        self.t_values = [0.0, 0.0, 0.0]
        self.declare_components()
        # Lazy: only `instantiate()` actually needs the flattened
        # vars/params reference lists, so we build them once at the top
        # level instead of having every nested `Model.__init__` walk its
        # own subtree (which made the old code O(depth^2) over the tree).
        self._raw_refs_cache = None
        self.record = {
            'time': [],
            'state': [],
            'vars_names': [],
            'subs': [],
        }

    @property
    def raw_vars_references(self):
        cache = getattr(self, "_raw_refs_cache", None)
        if cache is None:
            cache = self.get_vars_references()
            self._raw_refs_cache = cache
        return cache[0]

    @property
    def raw_param_references(self):
        cache = getattr(self, "_raw_refs_cache", None)
        if cache is None:
            cache = self.get_vars_references()
            self._raw_refs_cache = cache
        return cache[1]

    # --- composition / declaration ----------------------------------------------------

    def set_name(self, name):
        self.name = name

    def __getitem__(self, name):
        return self.components[name]

    def add_component(self, name, component):
        if isinstance(component, DifferentialVariable):
            self.components[f"der_{name}"] = component.get_derivative_variable()
            self.components[f"der_{name}"].set_name(f"der_{name}")
        self.components[name] = component
        self.components[name].set_name(name)

    def declare_components(self):
        """Override to register sub-components with `add_component`."""
        pass

    def declare_equations(self):
        """Override to return a list of sympy expressions, each implicitly == 0."""
        return []

    def add_connection(self, var_a, var_b):
        """Declare that two `Variable`s must always hold the same value.

        This is a hint to `instantiate()` that lets the framework collapse the
        two variables into one via union-find, BEFORE the symbolic-equation
        machinery runs.  Functionally equivalent to returning `var_a.symbol -
        var_b.symbol` from `declare_equations()`, but much cheaper at scale:
          * no sympy `Add` is built per connection,
          * the connection never enters the equation list,
          * the trivial-equation reducer doesn't have to discover it.

        Use it for any "same physical port wired together" relationships
        (pipe-to-pipe segment continuity, splitter outlet -> child inlet,
        etc.).  Use the regular `declare_equations` return value for any
        constraint that is genuinely non-trivial.
        """
        self.connections.append((var_a, var_b))

    def collect_connections(self):
        """Recursively gather every `(var_a, var_b)` connection registered in
        the tree.  Mirror of `collect_equations` but for connections only.

        IMPORTANT: connections are declared *inside* `declare_equations`
        (because that's where the user has access to sub-components), so
        callers MUST call `collect_equations` first to flush them onto each
        component's `connections` list.
        """
        conns = list(self.connections)
        for c in self.components.values():
            if isinstance(c, Model) and c.is_composite():
                conns.extend(c.collect_connections())
        return conns

    def is_composite(self):
        for c in self.components.values():
            if isinstance(c, Model):
                return True
        return False

    def get_vars_references(self):
        vars_references = []
        param_references = []
        for _, c in self.components.items():
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
    def assign_symbols(self, prefix="", dotted_prefix="", top_level=False, vars=None, prev_vars=None, params=None, vars_map=None, params_map=None):
        # Avoid mutable default arguments: each top-level invocation must start with
        # fresh containers, otherwise repeated `instantiate()` calls in the same
        # process (e.g. across tests) keep accumulating into the same lists/dicts
        # and produce duplicate sympy symbols downstream.
        if vars is None:
            vars = []
        if prev_vars is None:
            prev_vars = []
        if params is None:
            params = []
        if vars_map is None:
            vars_map = {}
        if params_map is None:
            params_map = {}
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
            print("Symbols assigned")
            return vars, prev_vars, params, vars_map, params_map

    def _build_sym_paths(self):
        """Recursive `(path_tuple) -> Symbol` map for every Variable/Parameter
        leaf reachable from this Model.  Used by the Step-B per-class equation
        cache to remap a cached template's symbols onto a fresh instance's
        symbols by structural path (e.g., `('pipe_segment_0', 'p_in')`).
        """
        out = {}
        for name, c in self.components.items():
            if isinstance(c, (Variable, Parameter)):
                sym = getattr(c, 'symbol', None)
                if sym is not None:
                    out[(name,)] = sym
                prev_sym = getattr(c, 'prev_symbol', None)
                if prev_sym is not None:
                    out[(name, '__prev__')] = prev_sym
            elif isinstance(c, Model) and c.is_composite():
                for sub_path, sym in c._build_sym_paths().items():
                    out[(name,) + sub_path] = sym
        return out

    @line_profiler.profile
    def collect_equations(self):
        eqs = []
        cls = self.__class__
        eq_cache = _eq_cache_var.get()
        # `eq_cache is None` -> running outside `instantiate()` (e.g. via
        # `get_current_system()` or a unit test).  Just rebuild equations
        # every time; the cache exists purely as an `instantiate()` speed-up.
        cache = eq_cache.get(cls) if eq_cache is not None else None

        # Path A: third+ instance of a class whose template has been validated.
        # Replay equations from cached template via a cheap path-based symbol
        # remap + a single `xreplace` per equation -- much faster than calling
        # `declare_equations()` again because it skips all the CoolProp-laden
        # `Symbolic_property` instantiations and sympy `Add`/`Mul` building.
        if cache is not None and cache.get('state') == 'cached':
            sym_paths = self._build_sym_paths()
            mapping = {}
            ok = True
            for first_sym, path in cache['sym_to_path'].items():
                target = sym_paths.get(path)
                if target is None:
                    ok = False
                    break
                mapping[first_sym] = target
            if ok:
                eqs.extend(eq.xreplace(mapping) if eq.free_symbols else eq
                           for eq in cache['eqs'])
            else:
                eqs.extend(self.declare_equations())
        # Path B: first or second instance of this class -- run normally and
        # either capture (first) or validate (second).  Classes that emit
        # `add_connection` side effects are conservatively marked 'no-cache'
        # because their `declare_equations` is already light (it just appends
        # to `self.connections`); replaying them via path lookup turned out
        # measurably slower than just re-calling `declare_equations`.
        else:
            n_conns_before = len(self.connections)
            run_eqs = self.declare_equations()
            has_new_conns = len(self.connections) > n_conns_before
            if eq_cache is not None:
                if cache is None:
                    if has_new_conns:
                        eq_cache[cls] = {'state': 'no-cache'}
                    else:
                        sym_paths = self._build_sym_paths()
                        sym_to_path = {s: p for p, s in sym_paths.items()}
                        used_syms = set()
                        for eq in run_eqs:
                            used_syms.update(eq.free_symbols)
                        eq_cache[cls] = {
                            'state': 'first',
                            'eqs': run_eqs,
                            'sym_to_path': {s: sym_to_path[s] for s in used_syms
                                            if s in sym_to_path},
                        }
                elif cache['state'] == 'first':
                    if has_new_conns:
                        cache['state'] = 'no-cache'
                    else:
                        sym_paths = self._build_sym_paths()
                        mapping = {}
                        can_replay = True
                        for first_sym, path in cache['sym_to_path'].items():
                            target = sym_paths.get(path)
                            if target is None:
                                can_replay = False
                                break
                            mapping[first_sym] = target
                        if can_replay:
                            replayed = [eq.xreplace(mapping) if eq.free_symbols else eq
                                        for eq in cache['eqs']]
                            if (len(replayed) == len(run_eqs)
                                    and all(a == b for a, b in zip(replayed, run_eqs))):
                                cache['state'] = 'cached'
                            else:
                                cache['state'] = 'no-cache'
                        else:
                            cache['state'] = 'no-cache'
            eqs.extend(run_eqs)

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

    # --- trivial-equation reduction ---------------------------------------------------

    @staticmethod
    def _classify_linear(eq):
        """Try to express `eq` as `c0 + sum(ci * vi) == 0` where every `vi` is a
        sympy Symbol and every coefficient is a Number.

        Returns `(const_term, {symbol: coeff})` if successful, otherwise `None`.

        Implemented as a recursive structural walk over `Add`/`Mul`/`Symbol`/`Number`
        nodes -- crucially this never builds a `Poly` (which is the slow operation
        the previous `is_polynomial()`/`as_poly().degree()` path was triggering on
        every equation, including the many CoolProp-laden ones that are obviously
        non-linear).
        """
        if isinstance(eq, sp.Symbol):
            return sp.S.Zero, {eq: sp.S.One}
        if eq.is_Number:
            return eq, {}
        if eq.is_Mul:
            coeff = sp.S.One
            sym = None
            for arg in eq.args:
                if arg.is_Number:
                    coeff = coeff * arg
                elif isinstance(arg, sp.Symbol):
                    if sym is not None:
                        return None  # x*y -> not linear
                    sym = arg
                else:
                    return None
            if sym is None:
                return coeff, {}
            return sp.S.Zero, {sym: coeff}
        if eq.is_Add:
            const = sp.S.Zero
            coeffs = {}
            for term in eq.args:
                sub = Model._classify_linear(term)
                if sub is None:
                    return None
                c, syms = sub
                const = const + c
                for s, sc in syms.items():
                    if s in coeffs:
                        coeffs[s] = coeffs[s] + sc
                    else:
                        coeffs[s] = sc
            return const, coeffs
        return None

    @staticmethod
    def _close_substitutions(substitutions):
        """Resolve chains in `substitutions` so each value only references symbols
        that are NOT themselves keys. O(|subs| * average expression size) using a
        memoised DFS, instead of the previous O(|subs|^2) fixed-point iteration.

        Cycles (which would indicate an inconsistent system) raise `ValueError`.
        """
        keys = set(substitutions.keys())
        cache = {}
        visiting = set()

        def resolve(k):
            if k in cache:
                return cache[k]
            if k in visiting:
                raise ValueError(f"Cycle in trivial-equation substitutions involving {k}")
            visiting.add(k)
            v = substitutions[k]
            deps = v.free_symbols & keys
            if deps:
                v = v.xreplace({d: resolve(d) for d in deps})
            visiting.discard(k)
            cache[k] = v
            return v

        for k in list(substitutions.keys()):
            substitutions[k] = resolve(k)
        return substitutions

    @line_profiler.profile
    def remove_trivial_equations(self, equations, var_symbols):
        """Eliminate trivially-linear equations (`a*x + b*y + c == 0` with `a,b,c`
        constants and `x,y` symbols) without invoking `sp.solve`/`sp.Poly`.

        Strategy:
          1. Walk every equation once. Use `_classify_linear` to detect those that
             are linear in their free symbols. For each, pick one surviving free
             Variable to eliminate and store the substitution (current side and the
             mirrored prev-step side).
          2. Close the substitution dict via memoised DFS so each value only
             references surviving symbols.
          3. Apply the closed substitutions to every kept equation once.

        This avoids the previous code's two big costs:
          * `sp.Poly` / `sp.solve` per equation (replaced by a structural walk),
          * an O(|subs|^2) fixed-point xreplace loop to flatten substitution chains
            (replaced by an O(|subs| * expr_size) DFS).
        """
        substitutions = {}
        new_eqs = []
        removed_vars = set()
        kept_indices = []  # for progress logging only

        var_symbols_list = list(var_symbols)
        var_set = set(var_symbols_list)
        raw_var_set = set(self.raw_var_symbols)
        current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}

        print("Identifying trivial equations (structural)")
        for idx, eq in enumerate(equations):
            classified = Model._classify_linear(eq)
            if classified is None:
                new_eqs.append(eq)
                continue

            c0, coeffs = classified
            # drop zero coefficients (e.g. `x - x` style cancellations)
            coeffs = {s: c for s, c in coeffs.items() if c != 0}
            # Match the previous reducer's semantics: only act on 2-symbol equations
            # (`a*x + b*y + c == 0`).  Single-symbol equations (`a*x + c == 0`, which
            # would fully fix `x` to a constant) are deliberately left in the system
            # so downstream code that expects at least one surviving variable / row
            # keeps working.
            if len(coeffs) != 2:
                new_eqs.append(eq)
                continue

            # Only true Variables that are still surviving and not already eliminated
            # this pass are eligible.
            candidates = [
                s for s in coeffs
                if s in raw_var_set and s in var_set and s not in removed_vars
            ]
            if not candidates:
                new_eqs.append(eq)
                continue

            var1 = min(candidates, key=lambda s: s.name)  # deterministic
            coeff1 = coeffs[var1]
            # var1 = -(c0 + sum_{s != var1} c_s * s) / coeff1
            rest = -c0
            for s, c in coeffs.items():
                if s is var1:
                    continue
                rest = rest - c * s
            sol = rest / coeff1

            substitutions[var1] = sol
            removed_vars.add(var1)
            kept_indices.append(idx)

            var1_prev = current_to_prev.get(var1)
            if var1_prev is not None:
                sol_prev = sol.xreplace(current_to_prev) if sol.free_symbols else sol
                substitutions[var1_prev] = sol_prev
                removed_vars.add(var1_prev)

        if substitutions:
            print(f"Closing {len(substitutions)} substitutions")
            Model._close_substitutions(substitutions)
            sub_keys = set(substitutions.keys())

            # Step A: cache the xreplace step by template.  A naive
            # `[e.xreplace(substitutions) for e in new_eqs]` makes sympy walk
            # the FULL tree of every survivor (~389 CoolProp-laden equations
            # at N=4) just to discover that ~9 out of 10 of them never touch
            # any symbol in `substitutions`.  Two stacked optimisations:
            #
            #   1. **Pre-filter:** equations whose `free_symbols` are disjoint
            #      from `sub_keys` keep their identity -- no tree allocated.
            #
            #   2. **Per-template cache:** equations that DO touch a sub_key
            #      are normalised to a `(structural_template, mask)` key
            #      where `mask` is the tuple of placeholder positions whose
            #      backing Symbol appears in `substitutions`.  Two equations
            #      sharing the same `(template, mask)` produce structurally
            #      identical xreplace results -- we compute the substituted
            #      *template* once and re-bind placeholders -> actual symbols
            #      per instance.  Combined with `_close_substitutions` having
            #      already inlined chains, this turns the per-equation tree
            #      walk into a per-template tree walk + a cheap re-bind.
            _ph_pool_a = []

            def _ph(k):
                while len(_ph_pool_a) <= k:
                    _ph_pool_a.append(sp.Symbol(f"_red_ph_{len(_ph_pool_a)}", real=True))
                return _ph_pool_a[k]

            template_subbed_cache = {}
            n_skipped = 0
            n_applied = 0
            n_template_hit = 0
            new_eqs_out = []
            for e in new_eqs:
                if not (e.free_symbols & sub_keys):
                    new_eqs_out.append(e)
                    n_skipped += 1
                    continue
                # Normalise this equation's symbols to placeholders, then
                # mark which placeholder indices are being substituted AND
                # the structural template of the substitution RHS.  Two
                # instances share a cache slot iff they have the same
                # equation template AND the same per-placeholder substitution
                # RHS template (with all RHS-external symbols themselves
                # placeholder-normalised in a per-equation-extension pool).
                sym_order = []
                sym_to_ph = {}
                for atom in sp.preorder_traversal(e):
                    if isinstance(atom, sp.Symbol) and atom not in sym_to_ph:
                        sym_to_ph[atom] = _ph(len(sym_to_ph))
                        sym_order.append(atom)
                template_e = e.xreplace(sym_to_ph)

                # Build the placeholder-space substitution dict for this
                # instance, extending the placeholder pool as needed for any
                # symbols present only on RHS values.
                full_sym_to_ph = dict(sym_to_ph)
                rhs_sym_orders = []
                ph_subs_template = {}
                rhs_templates_for_key = []
                for k, sym in enumerate(sym_order):
                    if sym not in substitutions:
                        continue
                    rhs = substitutions[sym]
                    rhs_extra_order = []
                    for rhs_atom in sp.preorder_traversal(rhs):
                        if isinstance(rhs_atom, sp.Symbol) and rhs_atom not in full_sym_to_ph:
                            full_sym_to_ph[rhs_atom] = _ph(len(full_sym_to_ph))
                            rhs_extra_order.append(rhs_atom)
                    rhs_template = rhs.xreplace(full_sym_to_ph) if rhs.free_symbols else rhs
                    ph_subs_template[_ph(k)] = rhs_template
                    rhs_sym_orders.append(rhs_extra_order)
                    rhs_templates_for_key.append((k, rhs_template))

                cache_key = (template_e, tuple(rhs_templates_for_key))
                cached = template_subbed_cache.get(cache_key)
                if cached is None:
                    cached = template_e.xreplace(ph_subs_template)
                    template_subbed_cache[cache_key] = cached
                else:
                    n_template_hit += 1

                # Re-bind: the cached substituted template has placeholders
                # `_red_ph_*` whose intended actual symbols are recorded in
                # `full_sym_to_ph`.  Reverse the map and xreplace.
                ph_to_sym = {ph: sym for sym, ph in full_sym_to_ph.items()}
                new_eqs_out.append(cached.xreplace(ph_to_sym) if ph_to_sym else cached)
                n_applied += 1

            print(
                f"Applying substitutions to {n_applied}/{len(new_eqs)} equations "
                f"(skipped {n_skipped} disjoint, "
                f"{n_template_hit} template-cache hits / "
                f"{len(template_subbed_cache)} unique templates)"
            )
            new_eqs = new_eqs_out
        else:
            print("No substitutions applied")

        updated_var_symbols = [v for v in var_symbols_list if v not in removed_vars]
        return new_eqs, updated_var_symbols, substitutions

    # --- compilation ------------------------------------------------------------------

    @line_profiler.profile
    def instantiate(self, cse=True, aditional_modules=None, max_remove_trival_passes=1,
                    lambda_cache_dir=None):
        # Step B's `declare_equations()` template cache is scoped to this single
        # `instantiate()` call to avoid cross-call contamination (e.g. Air's
        # `Air_rho_ph` Function nodes leaking into a subsequent Hydrogen
        # instantiation).  Set a fresh dict on entry and reset on exit
        # regardless of whether `instantiate()` returns or raises.
        _eq_cache_token = _eq_cache_var.set({})
        try:
            return self._instantiate_impl(
                cse=cse,
                aditional_modules=aditional_modules,
                max_remove_trival_passes=max_remove_trival_passes,
                lambda_cache_dir=lambda_cache_dir,
            )
        finally:
            _eq_cache_var.reset(_eq_cache_token)

    @line_profiler.profile
    def _instantiate_impl(self, cse=True, aditional_modules=None,
                          max_remove_trival_passes=1, lambda_cache_dir=None):
        if aditional_modules is None:
            aditional_modules = []
        # `_NUMPY_LAMBDIFY_COMPAT` patches sympy callables that `lambdify`
        # doesn't know how to translate to numpy by default (e.g. `Heaviside`,
        # which appears in the Jacobian of any `sp.Max`/`sp.Min`).  Inserted
        # AFTER `'numpy'` so numpy's own names win, but BEFORE the medium
        # modules so the compat dict is part of the runtime namespace used
        # by both fresh-lambdify code-gen AND cached-source re-exec.  Kept
        # out of `_lambda_modules_sig` (we only signature `aditional_modules`)
        # so the cache key stays stable across versions of the compat dict.
        #
        # NOTE on medium-callback wrapping: per-template lambdas that get
        # called with array placeholders need their medium callbacks
        # (`Air_rho_ph` etc.) to broadcast across instances.  However
        # wrapping unconditionally adds ~500 ns per CoolProp call -- a
        # ~3-million-call hot loop in `run_system`'s Newton solve, where
        # ZERO templates qualify for vectorisation under the default
        # cutoff.  We therefore lambdify with the RAW (un-wrapped) modules
        # here and patch only the vectorised templates' `__globals__`
        # post-hoc, after `_vec_template_use` is decided below.  Scalar
        # templates pay no wrapper overhead at all.
        all_modules = ["numpy", _NUMPY_LAMBDIFY_COMPAT] + aditional_modules

        # Disk cache for the lambdified residual + Jacobian source.  When the
        # same (geometry, medium, sympy version) has been compiled before, this
        # turns the multi-second `lambdify` calls into a sub-second source-load.
        # Pass `lambda_cache_dir=False` to disable; default uses ~/.cache/hydrogen.
        if lambda_cache_dir is False:
            self._lambda_cache_dir = None
        elif lambda_cache_dir is None:
            self._lambda_cache_dir = lambda_cache_default_dir()
        else:
            from pathlib import Path as _P
            self._lambda_cache_dir = _P(lambda_cache_dir)
        # Module signature for cache keying: function names brought in via the
        # `modules` arg.  Each medium contributes a unique prefix so the key
        # never collides across media.
        self._lambda_modules_sig = []
        for m in aditional_modules:
            if isinstance(m, dict):
                self._lambda_modules_sig.extend(m.keys())

        print("Instantiating model")

        start_time = time.time()
        _t0 = time.time()
        vars, prev_vars, params, vars_map, params_map = self.assign_symbols(top_level=True)
        _t_assign = time.time() - _t0
        _t0 = time.time()
        # `collect_equations` consults `_eq_cache_var` (set by `instantiate`)
        # so that only the first 1-2 instances of each `Model` subclass build
        # their CoolProp-laden symbolic equations from scratch; subsequent
        # siblings replay via a path-based symbol remap + a single `xreplace`
        # per eq.
        self.all_raw_equations = self.collect_equations()
        _t_collect = time.time() - _t0
        self.raw_var_symbols, self.raw_prev_var_symbols, self.raw_param_symbols = vars, prev_vars, params
        self.all_raw_symbols = self.raw_var_symbols + self.raw_prev_var_symbols + self.raw_param_symbols + self.t_symbols
        self.record['vars_names'] = [getattr(v, 'full_name', v.name) for v in self.raw_vars_references]
        print(len(self.all_raw_symbols))
        print(f"Current system collected in {time.time() - start_time:.2f} s "
              f"(assign_symbols={_t_assign:.2f}s, collect_equations={_t_collect:.2f}s)")

        start_time = time.time()

        self.improved_vars = self.raw_var_symbols
        self.improved_equations = self.all_raw_equations
        self.all_improved_symbols = self.all_raw_symbols
        self.improve_subs = {}

        # Step 10: short-circuit explicit `add_connection` pairs via union-find
        # BEFORE the symbolic trivial-equation reducer runs.  Components that
        # use `add_connection` (in-tree: StraightPipe, Splitter; example tree
        # nodes: BranchNode, TreeSystem) thereby skip building the Add(symA,
        # -symB) sympy expression entirely, and the reducer doesn't have to
        # rediscover them.
        connections = self.collect_connections()
        if connections:
            uf_start = time.time()
            uf_parent = {}

            def find(s):
                parent = uf_parent.get(s, s)
                if parent is s:
                    return s
                root = find(parent)
                uf_parent[s] = root
                return root

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra is rb:
                    return
                # deterministic: smaller symbol-name wins as representative
                if rb.name < ra.name:
                    ra, rb = rb, ra
                uf_parent[rb] = ra

            raw_var_set = set(self.raw_var_symbols)
            deferred_eqs = []
            for var_a, var_b in connections:
                sa, sb = var_a.symbol, var_b.symbol
                if sa is None or sb is None:
                    continue
                if sa not in raw_var_set or sb not in raw_var_set:
                    # One side is a Parameter / t -- can't union with a non-Variable;
                    # defer to the symbolic trivial reducer.
                    deferred_eqs.append(sa - sb)
                    continue
                union(sa, sb)

            current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}
            connection_subs = {}
            for s in list(uf_parent.keys()):
                rep = find(s)
                if rep is s:
                    continue
                connection_subs[s] = rep
                ps = current_to_prev.get(s)
                pr = current_to_prev.get(rep)
                if ps is not None and pr is not None:
                    connection_subs[ps] = pr

            if connection_subs:
                self.improved_equations = [
                    eq.xreplace(connection_subs) for eq in self.improved_equations
                ]
                self.improve_subs.update(connection_subs)
                removed = set(connection_subs.keys())
                self.all_improved_symbols = [
                    s for s in self.all_improved_symbols if s not in removed
                ]
            if deferred_eqs:
                self.improved_equations = list(self.improved_equations) + deferred_eqs
            print(f"add_connection short-circuited {len(connection_subs)} symbols "
                  f"from {len(connections)} pairs ({len(deferred_eqs)} deferred) "
                  f"in {time.time() - uf_start:.2f} s")

        if max_remove_trival_passes > 0:
            print("Removing trivial equations")
            curent_size = len(self.improved_vars)
            print(f"Original variables: {curent_size}")
            start_time = time.time()

            for i in range(max_remove_trival_passes):
                print(f"Removing trivial pass {i+1}")
                self.improved_equations, self.all_improved_symbols, pass_subs = self.remove_trivial_equations(self.improved_equations, self.all_improved_symbols)
                # Accumulate substitutions across passes ONLY when this pass produced
                # new substitutions: the inner xreplace below would otherwise be 1k+
                # no-op `xreplace({})` calls (each one allocates / boxes / returns).
                if pass_subs:
                    # Push the new pass's subs through the RHS of previously stored
                    # ones so every entry stays expressed in terms of the latest-
                    # surviving symbols, then merge the new entries in.
                    for prev_key in list(self.improve_subs.keys()):
                        self.improve_subs[prev_key] = self.improve_subs[prev_key].xreplace(pass_subs)
                    self.improve_subs.update(pass_subs)
                # Step C early-exit BEFORE the next pass walks every equation again.
                #
                # The reducer is deterministic per equation, so the only way the
                # next pass can find a NEW trivial equation is if this pass's
                # `xreplace` mutated the surviving equations in a way that made
                # an equation newly fall into the "linear, exactly 2 symbols"
                # bucket.  That requires the substitution's RHS to be a `Number`
                # (a constant) -- which collapses one of the symbols in any
                # `c0 + c1*var + c2*x + c3*y` shape down so it becomes `c0' +
                # c2*x + c3*y` (newly linear in two symbols).  RHSs that are
                # `Symbol`s or compound expressions never collapse the symbol
                # count of any equation -- they only rename / inflate it.
                #
                # So: if no substitution from this pass has a `Number` RHS,
                # subsequent passes can't find anything new, and we can stop
                # without paying their `_classify_linear` walk.  This is the
                # common case for the pipe-tree topology where every reducer
                # substitution is a `Mul`/`Add` of CoolProp / time-step terms.
                new_size = len(self.improved_equations)
                if new_size == curent_size:
                    break
                if not pass_subs or not any(
                    rhs.is_Number for rhs in pass_subs.values()
                ):
                    break
                curent_size = new_size
            # `s in self.all_improved_symbols` was an O(N) list-scan per probe;
            # for N=4 this turned into ~1M sympy `Symbol.__eq__` calls and
            # dominated the entire trivial-reducer block (~13 s out of ~13 s).
            # Hoist to a hash-set membership test once.
            improved_symbols_set = set(self.all_improved_symbols)
            self.improved_vars = [s for s in self.raw_var_symbols if s in improved_symbols_set]
            stop_time = time.time()
            print(f"Removed equations and variables: {len(self.raw_var_symbols) - len(self.improved_vars)} in {stop_time - start_time} s")

        self.n_v = len(self.improved_vars)
        self.n_p = len(self.raw_param_symbols)
        self.n_t = len(self.t_symbols)
        print(f"Remaining variables and equations: {self.n_v}")
        self.values = np.zeros(2 * self.n_v + self.n_p + self.n_t)
        self.delta_values = np.zeros(self.n_v)
        improved_equations_list = list(self.improved_equations)
        self.all_improved_symbols_matrix = sp.Matrix(self.all_improved_symbols)
        # `improved_equations` Matrix is no longer needed for lambdify; the
        # per-template path below builds tiny per-template Matrices instead.
        self.improved_equations = improved_equations_list

        # ---- Per-template residual + Jacobian (step 12) ---------------------
        # Group equations by structural template (placeholder-normalised form),
        # lambdify ONE block per template returning `[residual, d/dph_0, ...,
        # d/dph_K-1]`, and at runtime loop over instances calling the right
        # template lambda with the appropriate state values.  This collapses
        # what used to be (a) one giant `lambdify` of all 197 residual eqs and
        # (b) one giant `lambdify` of all 788 Jacobian nonzeros into ~8 small
        # `lambdify` calls -- each of which trivially CSE-shares between an
        # equation and its derivatives.
        start_time = time.time()
        var_to_col = {v: j for j, v in enumerate(self.improved_vars)}
        var_set = set(self.improved_vars)
        all_sym_to_state_idx = {s: i for i, s in enumerate(self.all_improved_symbols)}

        _ph_pool = []

        def _placeholder(k):
            while len(_ph_pool) <= k:
                _ph_pool.append(sp.Symbol(f"_jac_ph_{len(_ph_pool)}", real=True))
            return _ph_pool[k]

        # Pass 1: classify each equation by structural template.  Equations
        # that vanished to a constant (`0`) after the trivial / connection
        # reducer are dropped here -- they're tautologies that contribute
        # nothing to either the residual or the Jacobian.
        template_to_instances = {}
        n_constant_dropped = 0
        for i, eq in enumerate(improved_equations_list):
            sym_order = []
            sym_to_ph = {}
            for atom in sp.preorder_traversal(eq):
                if isinstance(atom, sp.Symbol) and atom not in sym_to_ph:
                    ph = _placeholder(len(sym_to_ph))
                    sym_to_ph[atom] = ph
                    sym_order.append(atom)
            if not sym_order:
                # Constant equation -- assert it actually evaluates to 0; if
                # not, the reducer would have produced an infeasibility row
                # which we keep so the Newton solver still surfaces it.
                if eq == 0:
                    n_constant_dropped += 1
                    continue
            template = eq.xreplace(sym_to_ph) if sym_order else eq
            template_to_instances.setdefault(template, []).append((i, sym_order))

        n_eq = len(improved_equations_list)
        n_templates = len(template_to_instances)
        print(f"Found {n_templates} unique equation templates "
              f"covering {n_eq - n_constant_dropped}/{n_eq} equations "
              f"({n_constant_dropped} dropped as constants)")

        # Pass 2: lambdify per template and build the runtime plan.
        # `template_lambdas[tid]` returns shape `(n_ph + 1, 1)`: row 0 is the
        # residual value; row k+1 is d_residual/d_ph_k.
        template_lambdas = []
        template_n_ph = []
        template_jac_outputs = []  # for each tid: list of placeholder indices that are vars in some instance (used for Jac)
        template_keys = []

        plan_inst_template_id = []     # int per instance
        plan_inst_state_indices = []   # int array per instance
        plan_inst_eq_idx = []          # int per instance

        plan_jac_inst = []   # for each jac entry: which instance index
        plan_jac_out = []    # for each jac entry: which row of the lambda output (k+1)
        plan_jac_rows = []   # equation row in the global residual
        plan_jac_cols = []   # variable column in the global jacobian

        # ---- Step D: parallelise per-template lambdify across processes -----
        # For COLD-CACHE runs the dominant cost in this section is the 8x
        # `lambdify_compat` calls (~1 s each at N=4 -> ~9 s sequential).  They
        # are completely independent (different templates, no shared sympy
        # state -- sympy's caches are per-process and CoolProp callables are
        # pure functions).  Run them in a fork-based `multiprocessing.Pool`
        # so cold-cache instantiate scales with `cpu_count() / n_templates`.
        #
        # Pass 1 (sequential): cheap symbolic prep + cache lookup.
        prep_per_template = []
        cache_misses = []
        for template, instances in template_to_instances.items():
            tid = len(prep_per_template)
            n_ph = len(instances[0][1])
            placeholders = [_placeholder(k) for k in range(n_ph)]
            # Lambdify a Python LIST (not `sp.Matrix`).  The list-output form
            # is what makes the per-template VECTORISED Newton path work:
            # when the arguments are numpy arrays the generated source is
            # `return [res_expr, dres/dph_0, ...]`, which produces a Python
            # list of arrays/scalars per call.  The Matrix form would have
            # called `numpy.array([[res], [dres/dph_0], ...])`, which fails
            # the moment any row reduces to a constant (e.g. a partial that
            # CSE-folds to `1`) because numpy can't stack a scalar `1` with
            # an array of length `n_inst`.  Sympy's pickle of a list yields
            # a different cache key from the Matrix form, so on-disk cache
            # entries from prior versions are naturally ignored (left
            # in-place; harmless).
            exprs_list = [template] + [sp.diff(template, ph) for ph in placeholders]
            args_mat = sp.Matrix(placeholders)
            label = f"template_{tid:02d}"
            cached_func = None
            key = None
            if self._lambda_cache_dir is not None:
                key = lambda_cache_key(args_mat, exprs_list, self._lambda_modules_sig, cse)
                namespace = self._build_lambdify_namespace(all_modules)
                cached_func = load_lambdified_source(self._lambda_cache_dir, key, namespace)
                if cached_func is not None:
                    print(f"  [lambda-cache HIT  for {label}: {key[:8]}]")
            prep_per_template.append({
                "label": label, "tid": tid, "n_ph": n_ph,
                "instances": instances, "template": template,
                "args_mat": args_mat, "block": exprs_list, "key": key,
                "cached_func": cached_func,
            })
            if cached_func is None:
                cache_misses.append(prep_per_template[-1])

        # Pass 2 (parallel where it pays off): lambdify + write-to-disk for
        # each cache miss.  Falls back to a sequential loop when there's <2
        # misses (pool startup wouldn't pay off) or when the user opted out
        # via `HYDROGEN_PARALLEL_LAMBDIFY=0`.
        worker_procs = int(os.environ.get(
            "HYDROGEN_PARALLEL_LAMBDIFY",
            str(min(len(cache_misses), max(1, (os.cpu_count() or 1))))
        ))
        if cache_misses and worker_procs > 1 and self._lambda_cache_dir is not None:
            payloads = [
                (p["label"], p["key"], p["args_mat"], p["block"], cse)
                for p in cache_misses
            ]
            t_par = time.time()
            ctx = _mp.get_context("fork")
            pool = ctx.Pool(
                processes=min(worker_procs, len(cache_misses)),
                initializer=_init_lambdify_worker,
                initargs=(all_modules, self._lambda_modules_sig,
                          self._lambda_cache_dir, payloads),
            )
            try:
                # Pass only integer task indices through the queue -- the
                # actual templates are inherited via fork in `_WORKER_PAYLOADS`.
                completed = pool.map(_lambdify_worker_task, range(len(payloads)))
            finally:
                pool.close()
                pool.join()
            print(f"  [parallel lambdify of {len(cache_misses)} templates "
                  f"on {min(worker_procs, len(cache_misses))} procs: "
                  f"{time.time() - t_par:.2f} s]")
            # Each worker wrote its lambda's source to the on-disk cache; load
            # those back in the parent.  This trades a (free) cache-load for
            # the impossibility of pickling the lambda function across procs.
            # IMPORTANT: each lambda gets its OWN namespace dict so the
            # post-hoc `__globals__` patch for vectorised templates (below)
            # doesn't bleed into scalar templates' medium-callback bindings.
            completed_keys = {label: key for _, label, key in completed}
            for prep in cache_misses:
                key = completed_keys.get(prep["label"], prep["key"])
                namespace = self._build_lambdify_namespace(all_modules)
                fn = load_lambdified_source(self._lambda_cache_dir, key, namespace)
                if fn is None:
                    fn = lambdify_compat(
                        prep["args_mat"], prep["block"],
                        modules=all_modules, cse=cse, docstring_limit=-1,
                    )
                prep["cached_func"] = fn
                print(f"  [lambda-cache MISS for {prep['label']}: {key[:8] if key else '????????'} (saved)]")
        elif cache_misses:
            # Sequential fallback.
            for prep in cache_misses:
                fn = lambdify_compat(
                    prep["args_mat"], prep["block"],
                    modules=all_modules, cse=cse, docstring_limit=-1,
                )
                if self._lambda_cache_dir is not None and prep["key"] is not None:
                    save_lambdified_source(
                        self._lambda_cache_dir, prep["key"], fn,
                        self._lambda_modules_sig,
                    )
                    print(f"  [lambda-cache MISS for {prep['label']}: {prep['key'][:8]} (saved)]")
                prep["cached_func"] = fn

        # Pass 3 (sequential): build the runtime plan from the (now fully
        # populated) `cached_func`s, in deterministic insertion order.
        for prep in prep_per_template:
            tid = prep["tid"]
            n_ph = prep["n_ph"]
            template = prep["template"]
            f = prep["cached_func"]
            template_lambdas.append(f)
            template_n_ph.append(n_ph)
            template_keys.append(template)

            for eq_idx, sym_order in prep["instances"]:
                inst_idx = len(plan_inst_template_id)
                plan_inst_template_id.append(tid)
                plan_inst_state_indices.append(
                    np.asarray([all_sym_to_state_idx[s] for s in sym_order],
                               dtype=np.int64)
                )
                plan_inst_eq_idx.append(eq_idx)

                # Jacobian: for each placeholder that maps to a Variable, the
                # (k+1)-th lambda output is the partial derivative; we emit one
                # sparse-Jacobian entry per such placeholder.  Sorted by name
                # to match the pre-step-12 ordering -> identical residual
                # fingerprints and a deterministic cache key.
                var_phs = sorted(
                    [(k, sym) for k, sym in enumerate(sym_order) if sym in var_set],
                    key=lambda p: p[1].name,
                )
                for k, sym in var_phs:
                    plan_jac_inst.append(inst_idx)
                    plan_jac_out.append(k + 1)
                    plan_jac_rows.append(eq_idx)
                    plan_jac_cols.append(var_to_col[sym])

        # Pack the plan into numpy arrays for fast scatter at runtime.
        self._n_eq = n_eq
        self._n_instances = len(plan_inst_template_id)
        self._inst_template = np.asarray(plan_inst_template_id, dtype=np.int64)
        self._inst_state_indices = plan_inst_state_indices  # list of arrays
        self._inst_eq_idx = np.asarray(plan_inst_eq_idx, dtype=np.int64)

        self._jac_sparse_rows = np.asarray(plan_jac_rows, dtype=np.int64)
        self._jac_sparse_cols = np.asarray(plan_jac_cols, dtype=np.int64)
        self._jac_inst = np.asarray(plan_jac_inst, dtype=np.int64)
        self._jac_out = np.asarray(plan_jac_out, dtype=np.int64)
        self._jac_nnz = len(plan_jac_rows)
        self._template_lambdas = template_lambdas

        # ---- Vectorised plan: pre-compute per-template gather + scatter ----
        # The per-instance Newton loop (`_eval_per_template`, kept below as a
        # fallback) calls each template's lambda once per instance.  At
        # 32 instances / 13 templates that's ~32 Python dispatches per Newton
        # iteration; for larger systems (e.g. pipe_tree N=4 with hundreds of
        # instances) the dispatch overhead dominates.
        #
        # The vectorised path collapses this to ONE lambda call per template:
        # for each template `t` with `n_inst_t` instances and `n_ph_t`
        # placeholders, we gather a `(n_ph_t, n_inst_t)` argument array out
        # of `vals_arr` in a single fancy-index op, call the lambda once
        # (its output is a Python list of broadcasted arrays of shape
        # `(n_inst_t,)` -- see `_wrap_modules_for_vectorisation`), and
        # scatter the residual + Jacobian values out via fancy-index assigns.
        #
        # All gather/scatter index arrays are pre-computed here so the
        # runtime path is pure-numpy with no Python-level per-instance work.
        per_t_inst_cols = {tid: [] for tid in range(n_templates)}
        for inst_idx, tid in enumerate(plan_inst_template_id):
            per_t_inst_cols[tid].append(inst_idx)

        self._vec_template_input_idx = []  # list[ndarray (n_ph_t, n_inst_t)]
        self._vec_template_eq_rows = []    # list[ndarray (n_inst_t,)]
        # `inst_to_template_col[g_inst_idx] -> col within its template's
        # gather-array`.  Used below to scatter the per-instance Jacobian
        # rows back into the global jvals buffer.
        inst_to_template_col = np.empty(self._n_instances, dtype=np.int64)
        for tid, inst_list in per_t_inst_cols.items():
            n_inst_t = len(inst_list)
            n_ph_t = template_n_ph[tid]
            ipx = np.empty((n_ph_t, n_inst_t), dtype=np.int64)
            eqr = np.empty(n_inst_t, dtype=np.int64)
            for col, inst_idx in enumerate(inst_list):
                ipx[:, col] = self._inst_state_indices[inst_idx]
                eqr[col] = self._inst_eq_idx[inst_idx]
                inst_to_template_col[inst_idx] = col
            self._vec_template_input_idx.append(ipx)
            self._vec_template_eq_rows.append(eqr)

        # Per-template Jacobian scatter plan: which global `g` indices map
        # to which `(out_row_in_lambda, instance_col_in_template)` pairs.
        per_t_jac = {tid: ([], [], []) for tid in range(n_templates)}
        for g in range(self._jac_nnz):
            inst_idx = int(self._jac_inst[g])
            tid = int(self._inst_template[inst_idx])
            g_list, out_list, col_list = per_t_jac[tid]
            g_list.append(g)
            out_list.append(int(self._jac_out[g]))
            col_list.append(int(inst_to_template_col[inst_idx]))
        self._vec_template_jac_g = []      # list[ndarray] global jvals indices
        self._vec_template_jac_out = []    # list[ndarray] which lambda output row (k+1)
        self._vec_template_jac_col = []    # list[ndarray] which inst column in template
        for tid in range(n_templates):
            g_list, out_list, col_list = per_t_jac[tid]
            self._vec_template_jac_g.append(np.asarray(g_list, dtype=np.int64))
            self._vec_template_jac_out.append(np.asarray(out_list, dtype=np.int64))
            self._vec_template_jac_col.append(np.asarray(col_list, dtype=np.int64))

        elapsed = time.time() - start_time
        print(f"Per-template lambdify done in {elapsed:.2f} s "
              f"({n_templates} templates, {self._n_instances} instances, "
              f"{self._jac_nnz} nonzeros, "
              f"{100.0 * self._jac_nnz / max(1, self.n_v * n_eq):.2f}% dense)")

        # Eval routines that mimic the previous lambda interface.
        n_eq_local = n_eq
        # Vectorisation breakeven: a template with N instances costs
        #   ~  N * (per-scalar-call overhead)        per-instance loop
        #   ~  1 * (per-array-call overhead + scatter)   vectorised
        # Empirically (sympy 1.14 / numpy 2.x, on the run_system +
        # pipe_tree cases) the per-array path starts to win from N >= 8
        # onwards.  Below that the array packing + Python-loop in the
        # vectorised medium-callback wrapper dominates the saving.
        # `HYDROGEN_VECTORISE_MIN=N` lets users override; `=999` disables
        # vectorisation entirely (handy as an escape hatch if a custom
        # medium ships scalar callables that don't tolerate broadcasting).
        try:
            min_vec = int(os.environ.get("HYDROGEN_VECTORISE_MIN", "8"))
        except ValueError:
            min_vec = 8
        # Per-template strategy: True -> vectorised, False -> per-instance loop.
        self._vec_template_use = [int(eqr.size) >= min_vec
                                  for eqr in self._vec_template_eq_rows]
        # Pre-compute per-template `n_outputs = n_ph + 1` for the per-instance
        # branch -- avoids a `len(out_list)` per call.
        self._vec_template_n_out = [int(template_n_ph[tid]) + 1
                                    for tid in range(n_templates)]
        n_vec = sum(self._vec_template_use)
        n_scalar = len(self._vec_template_use) - n_vec
        print(f"  [eval strategy: {n_vec} vectorised templates, "
              f"{n_scalar} per-instance (cutoff >= {min_vec})]")

        # Post-hoc wrap medium callbacks ONLY for the templates that will
        # be invoked vectorised.  We mutate `f.__globals__` in-place to
        # swap the raw scalar callbacks (`Air_rho_ph`, ...) for vectorised
        # wrappers; lambda functions resolve free names through their
        # captured globals on every call, so the patch takes effect from
        # the next invocation.  Scalar-template lambdas are left alone --
        # avoiding the per-call `type(args[0]) is np.ndarray` check that
        # would otherwise be paid ~3 million times in a representative
        # `run_system` Newton solve.
        if n_vec:
            medium_names = set()
            for m in aditional_modules:
                if isinstance(m, dict):
                    medium_names.update(k for k, v in m.items() if callable(v))
            for tid, use_vec in enumerate(self._vec_template_use):
                if not use_vec:
                    continue
                g = template_lambdas[tid].__globals__
                for name in medium_names:
                    raw = g.get(name)
                    if raw is None or getattr(raw, "_hydrogen_vectorised", False):
                        continue
                    g[name] = _vectorise_callable(raw)

        def _eval_per_template(*args):
            """Hybrid per-template residual + Jacobian evaluator.

            Each template is evaluated either vectorised across its
            instances (one lambda call returning length-`n_inst` arrays) or
            instance-by-instance (the legacy path), based on the per-template
            decision in `_vec_template_use`.  The `(template, instance)` ->
            (residual_row, jvals_g) scatter is identical in both branches,
            so we share the scatter code via a pre-allocated
            `(n_outputs, n_inst_t)` work buffer.
            """
            vals_arr = np.asarray(args, dtype=float)
            r = np.zeros((n_eq_local, 1))
            jvals = np.zeros((self._jac_nnz, 1))
            templates = self._template_lambdas
            for tid in range(len(templates)):
                eqr = self._vec_template_eq_rows[tid]
                n_inst_t = eqr.size
                if n_inst_t == 0:
                    continue
                f = templates[tid]
                idx_mat = self._vec_template_input_idx[tid]
                n_outputs = self._vec_template_n_out[tid]
                if self._vec_template_use[tid]:
                    inputs = vals_arr[idx_mat]
                    out_list = f(*inputs)
                    rows = np.empty((n_outputs, n_inst_t))
                    if isinstance(out_list, list):
                        # `out_list` items can be either arrays of shape
                        # `(n_inst_t,)` or scalars (entries that CSE-folded
                        # to a constant).  Broadcast either case via a
                        # single assignment per output row.
                        for k, item in enumerate(out_list):
                            if np.isscalar(item) or (
                                isinstance(item, np.ndarray) and item.ndim == 0
                            ):
                                rows[k, :] = float(item)
                            else:
                                rows[k, :] = item
                    else:
                        # Defensive: a future SymPy could conceivably
                        # return an ndarray directly.
                        rows[:] = np.asarray(out_list).reshape(n_outputs, n_inst_t)
                else:
                    # Per-instance path: cheaper for templates with only a
                    # handful of instances (Python scalar dispatch beats
                    # array-packing overhead).  `rows[:, col] = out`
                    # accepts a list/tuple of scalars and copies them via a
                    # single C-level assignment -- noticeably faster than
                    # an inner Python `for k in range(n_outputs)` loop, and
                    # within a few percent of a hand-tuned per-instance
                    # scatter.
                    rows = np.empty((n_outputs, n_inst_t))
                    for col in range(n_inst_t):
                        scalar_args = vals_arr[idx_mat[:, col]]
                        rows[:, col] = f(*scalar_args)
                r[eqr, 0] = rows[0]
                jac_g = self._vec_template_jac_g[tid]
                if jac_g.size:
                    jvals[jac_g, 0] = rows[
                        self._vec_template_jac_out[tid],
                        self._vec_template_jac_col[tid],
                    ]
            return r, jvals

        # Cache the last-computed (residual, jac_values) keyed by `vals_arr`'s
        # bytes so a Newton iteration that asks for residual then Jacobian only
        # pays the per-instance loop once.
        self._eval_cache = {"key": None, "r": None, "j": None}

        def _eval_with_cache(*args):
            key = bytes(np.asarray(args, dtype=float).tobytes())
            if self._eval_cache["key"] == key:
                return self._eval_cache["r"], self._eval_cache["j"]
            r, j = _eval_per_template(*args)
            self._eval_cache.update(key=key, r=r, j=j)
            return r, j

        def _residual_callable(*args):
            r, _ = _eval_with_cache(*args)
            return r

        def _jac_values_callable(*args):
            _, j = _eval_with_cache(*args)
            return j

        self.lambdified_eqs = _residual_callable
        self._lambdified_jac_values = _jac_values_callable

        # Backward-compat dense Jacobian assembler (rarely used now that the
        # sparse Newton path is the default).
        def _eval_jacobian_dense(*args):
            J = np.zeros((n_eq_local, self.n_v))
            if self._jac_nnz == 0:
                return J
            vals = np.asarray(self._lambdified_jac_values(*args)).reshape(-1)
            J[self._jac_sparse_rows, self._jac_sparse_cols] = vals
            return J

        self.lambdified_jacobian = _eval_jacobian_dense
        # Drop the now-unused symbolic Jacobian Matrix.
        self.jacobian = None

        self.active_vars_references = [var for var in self.raw_vars_references if var.symbol in self.improved_vars]

        # The "raw vars reconstructor" maps the improved state vector to the FULL
        # set of original variables (in `raw_vars_references` order).  For most
        # variables this is just an index permutation; only the ones that were
        # eliminated during trivial-equation removal need a substitution expr.
        # We split into a cheap numpy gather (`_raw_passthrough_*`) plus a
        # lambdified tail (`_lambdified_raw_subs`) that's built lazily on first
        # use -- for runs that never call `record_state`/plot, this saves both a
        # full lambdify and the closure RAM that comes with it.
        improved_index = {s: i for i, s in enumerate(self.improved_vars)}
        passthrough_dst = []
        passthrough_src = []
        sub_dst = []
        sub_exprs = []
        for dst, var in enumerate(self.raw_vars_references):
            sub = self.improve_subs.get(var.symbol)
            if sub is None and var.symbol in improved_index:
                passthrough_dst.append(dst)
                passthrough_src.append(improved_index[var.symbol])
            else:
                sub_dst.append(dst)
                sub_exprs.append(sub if sub is not None else var.symbol)
        self._raw_passthrough_dst = np.asarray(passthrough_dst, dtype=np.int64)
        self._raw_passthrough_src = np.asarray(passthrough_src, dtype=np.int64)
        self._raw_sub_dst = np.asarray(sub_dst, dtype=np.int64)
        self._raw_sub_exprs = sub_exprs  # lambdified lazily
        self._raw_total = len(self.raw_vars_references)
        self._raw_modules = all_modules
        self._raw_cse = cse
        self._lambdified_raw_subs = None  # set by `_get_lambdified_raw_subs`
        # Keep just the symbol matrix needed for the lazy raw-vars compile.
        # The huge `improved_equations` Matrix can still be released below.
        self._raw_symbols_matrix = self.all_improved_symbols_matrix
        self.record['subs'] = self.improve_subs

        # Once everything that downstream code reads is captured in lambdified
        # closures, the raw sympy AST can be released.  These objects (the
        # equation list, the M*N improved-equations Matrix, the improved-vars
        # list, the substitution dict) are by far the biggest chunks of Python
        # heap left over from instantiation.  We deliberately keep references
        # that are still consulted at runtime (`improved_vars` -> n_v ordering,
        # `all_improved_symbols` -> nothing reads it post-instantiate but we
        # zero it explicitly).
        gc_targets = [
            "all_raw_equations", "improved_equations",
            "all_raw_symbols", "all_improved_symbols",
            "all_improved_symbols_matrix",
            "raw_var_symbols", "raw_prev_var_symbols", "raw_param_symbols",
        ]
        for attr in gc_targets:
            if hasattr(self, attr):
                setattr(self, attr, None)
        # Keep `improve_subs` only for cases that actually need plotting/lazy
        # raw-vars compilation.  When `_raw_sub_exprs` is empty there is
        # nothing left to look up, so the dict can be dropped entirely.
        if not self._raw_sub_exprs:
            self.improve_subs = {}
            self.record['subs'] = {}
        gc.collect()

    def _lambdify_with_cache(self, label, args, expr, modules, cse):
        """Disk-cached `lambdify_compat`.

        On a cache hit the expensive sympy code-gen + Python parse is skipped
        and we just `exec` the previously-saved source string into a namespace
        seeded with `numpy` + the medium's `Symbolic_property` callables (whose
        names are baked into the saved source).
        """
        cache_dir = self._lambda_cache_dir
        if cache_dir is not None:
            key = lambda_cache_key(args, expr, self._lambda_modules_sig, cse)
            namespace = self._build_lambdify_namespace(modules)
            cached = load_lambdified_source(cache_dir, key, namespace)
            if cached is not None:
                print(f"  [lambda-cache HIT  for {label}: {key[:8]}]")
                return cached
        func = lambdify_compat(args, expr, modules=modules, cse=cse, docstring_limit=-1)
        if cache_dir is not None:
            save_lambdified_source(cache_dir, key, func, self._lambda_modules_sig)
            print(f"  [lambda-cache MISS for {label}: {key[:8]} (saved)]")
        return func

    @staticmethod
    def _build_lambdify_namespace(modules):
        """Approximation of the namespace `sp.lambdify` builds internally, used
        when re-exec'ing a cached source string.  We import numpy + each entry
        from the user-supplied `modules` list so the source's bare function
        names (e.g. `Air_rho_ph`) resolve at exec time.

        Also pulls in `functools.reduce`: sympy 1.11+ prints `sp.Max(a, b, c)`
        as `reduce(maximum, [a, b, c])`, which appears in any source that has
        a `Max`/`Min` somewhere (e.g. the StraightPipe heat-transfer
        correlation).  Without this, cached-source re-exec NameErrors at
        first call.  `sympy.lambdify` injects `reduce` into the freshly-built
        function's globals on its own, but our on-disk cache only stores the
        function body so we have to seed it explicitly.
        """
        import builtins as _b
        import functools as _ft
        ns = {"__builtins__": _b.__dict__, "reduce": _ft.reduce}
        try:
            import numpy as _np
            ns.update({k: getattr(_np, k) for k in dir(_np) if not k.startswith("_")})
        except ImportError:
            pass
        for m in modules:
            if isinstance(m, dict):
                ns.update(m)
        return ns

    # --- state vector accessors -------------------------------------------------------

    def get_vars_values(self):
        return self.values[:self.n_v]

    def get_prev_vars_values(self):
        return self.values[self.n_v:2 * self.n_v]

    def get_param_values(self):
        return self.values[2 * self.n_v:2 * self.n_v + self.n_p]

    def get_t_values(self):
        return self.values[2 * self.n_v + self.n_p:2 * self.n_v + self.n_p + 3]

    def set_vars_values(self, vars):
        self.values[:self.n_v] = vars

    def set_param_values(self, params):
        self.values[2 * self.n_v:2 * self.n_v + self.n_p] = params

    def set_prev_vars_values(self, prev_vars):
        self.values[self.n_v:2 * self.n_v] = prev_vars

    def set_t_values(self, t_values):
        self.values[2 * self.n_v + self.n_p:2 * self.n_v + self.n_p + 3] = t_values

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
        return self.values[2 * self.n_v + self.n_p + 2]

    def get_t_value(self):
        return self.values[2 * self.n_v + self.n_p]

    def get_t_prev_value(self):
        return self.values[2 * self.n_v + self.n_p + 1]

    def set_dt(self, dt):
        self.values[2 * self.n_v + self.n_p + 2] = dt

    def set_t(self, t):
        self.values[2 * self.n_v + self.n_p] = t

    def set_t_prev(self, t_prev):
        self.values[2 * self.n_v + self.n_p + 1] = t_prev

    # --- evaluation / Newton solve / time stepping -----------------------------------

    def eval_residuals(self, vars):
        self.set_vars_values(vars)
        return self.lambdified_eqs(*self.values)

    def eval_jacobian(self, vars):
        self.set_vars_values(vars)
        return self.lambdified_jacobian(*self.values)

    @line_profiler.profile
    def eval_delta(self):
        self.delta_values[:] = self.lambdified_delta(*self.values)

    def _get_lambdified_raw_subs(self):
        """Lambdify (and cache) the substituted-variable tail of the raw-vars
        reconstructor on first use.  Variables that survived trivial-equation
        removal are scattered with a numpy gather instead -- no lambdify
        needed for them.
        """
        if self._lambdified_raw_subs is not None or len(self._raw_sub_exprs) == 0:
            return self._lambdified_raw_subs
        sub_matrix = sp.Matrix(self._raw_sub_exprs)
        self._lambdified_raw_subs = lambdify_compat(
            self._raw_symbols_matrix, sub_matrix,
            modules=self._raw_modules, cse=self._raw_cse, docstring_limit=-1,
        )
        return self._lambdified_raw_subs

    def lambdified_raw_vars(self, *values):
        """Materialise the full original-variable vector from the improved-state values.

        Kept callable-shaped so external code that previously did
        `model.lambdified_raw_vars(*model.values)` keeps working.
        """
        full_state = np.empty(self._raw_total)
        if self._raw_passthrough_dst.size:
            improved_vec = np.asarray(values[: self.n_v])
            full_state[self._raw_passthrough_dst] = improved_vec[self._raw_passthrough_src]
        if self._raw_sub_dst.size:
            f = self._get_lambdified_raw_subs()
            sub_vals = np.asarray(f(*values)).reshape(-1)
            full_state[self._raw_sub_dst] = sub_vals
        return full_state

    def record_state(self):
        self.record['time'].append(self.get_t_value())
        full_state = self.lambdified_raw_vars(*self.values)
        self.record['state'].append(full_state)

    def next_step(self):
        self.set_prev_vars_values(self.get_vars_values())
        self.set_t_prev(self.get_t_value())
        # Note: `solve_dae_step` already advanced `t` by `dt`. Advancing here too would
        # double-count, so we deliberately do not call `set_t` again.
        self.record_state()

    @line_profiler.profile
    def initialise(self, n=1, relaxation=1.0, tol=1e-6, max_iter=100):
        """Set the system to a Newton-consistent state at t = 0.

        For weakly-coupled or smoothly-conditioned problems the default
        `relaxation=1.0` (full Newton step) is fine. For systems with stiff
        startup transients (e.g. a pressure vessel charging from a much higher
        upstream pressure where the pipe's default initial guesses are far
        from the boundary-driven values), pass a smaller `relaxation` to
        damp the first few iterations and avoid overshooting into infeasible
        thermodynamic states.
        """
        self.set_t_values([0.0, 0.0, 0.0])
        init_values = np.array([var.value for var in self.active_vars_references])
        init_params = np.array([param.value for param in self.raw_param_references])
        self.set_vars_values(init_values)
        self.set_prev_vars_values(init_values)
        self.set_param_values(init_params)
        self.custom_solve(tol=tol, max_iter=max_iter, relaxation=relaxation)
        self.next_step()

    def update_delta(self):
        # Sparse path: when we built a sparse-Jacobian evaluator at instantiate
        # time, evaluate just the nonzero values and solve via scipy's SuperLU.
        # This avoids both materialising a dense (n_eq x n_v) matrix and the
        # cubic-time dense LU.  For the pipe-tree case the Jacobian is < 7%
        # dense, so SuperLU wins both in time and memory.
        if getattr(self, "_lambdified_jac_values", None) is not None:
            self.set_vars_values(self.get_vars_values())  # no-op write to refresh `values`
            jac_vals = np.asarray(self._lambdified_jac_values(*self.values)).reshape(-1)
            r = np.asarray(self.eval_residuals(self.get_vars_values())).reshape(-1)
            n_eq = self.delta_values.size  # square system: n_eq == n_v
            self.delta_values[:] = fast_sparse_solve(
                jac_vals,
                self._jac_sparse_rows,
                self._jac_sparse_cols,
                (n_eq, self.n_v),
                r,
            )
            return
        # Dense fallback for legacy / non-sparse codepaths.
        j = self.eval_jacobian(self.get_vars_values())
        r = self.eval_residuals(self.get_vars_values())
        self.delta_values[:] = fast_linear_solve(j, r).T[0]

    @line_profiler.profile
    def custom_solve(self, tol=1e-6, max_iter=100, relaxation=1.0):
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
    def solve_dae_step(self, dt, relaxation=1.0, tol=1e-6, max_iter=100):
        self.set_dt(dt)
        self.set_t(self.get_t_value() + dt)
        self.custom_solve(tol=tol, max_iter=max_iter, relaxation=relaxation)

    def set_initial_time(self, t):
        for c in self.components.values():
            if isinstance(c, Model):
                c.set_initial_time(t)

    def __repr__(self):
        return f"{self.__class__.__name__} {getattr(self, 'name', '')}"

    def get_lru_chache_info_str(self, func, indent=0):
        name = func.__name__
        hits = func.cache_info().hits
        misses = func.cache_info().misses
        calls = hits + misses
        return (
            f"{' ' * indent}{name}: ({calls} calls, {hits} hits, {misses} misses - "
            f"{hits / (hits + misses) * 100 if (hits + misses) > 0 else 0:.1f}% cache efficiency)"
        )

    def print_info(self, indent=0):
        print(f"{' ' * indent}{self.__class__.__name__}:")
        for _, c in self.components.items():
            if not isinstance(c, (Parameter, Variable)):
                c.print_info(indent=indent + 2)


class Parameter(Model):
    """Compile-time-known scalar that is passed to the lambdified residual but never solved for."""

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
    """Algebraic unknown; a Newton solve adjusts it to satisfy the system's residuals."""

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
    """Variable whose time evolution is constrained by the Crank-Nicolson rule.

    Adding a `DifferentialVariable` named `x` to a `Model` automatically also adds a
    companion `Variable` named `der_x`, and emits the constraint
        x - (x_prev + 0.5 * dt * (der_x + der_x_prev)) == 0
    so the user only needs to provide an algebraic equation defining `der_x` (the RHS
    of the ODE).
    """

    def __init__(self, value, unit=None):
        super().__init__(value, unit)
        self.der_variable = Variable(0.0, None)

    def declare_equations(self):
        # TODO: support more multistep schemes; for now Crank-Nicolson only.
        eq1 = self.symbol - (self.prev_symbol + 0.5 * self.dt * (self.der_variable.symbol + self.der_variable.prev_symbol))
        return [eq1]

    def get_derivative_variable(self):
        return self.der_variable
