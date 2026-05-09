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

import gc
import time

import line_profiler
import numpy as np
import sympy as sp

from .caching import (
    lambda_cache_default_dir,
    lambda_cache_key,
    load_lambdified_source,
    save_lambdified_source,
)
from .numerics import fast_error_norm, fast_linear_solve, fast_sparse_solve, lambdify_compat


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
            print(f"Applying substitutions to {len(new_eqs)} equations")
            new_eqs = [e.xreplace(substitutions) for e in new_eqs]
        else:
            print("No substitutions applied")

        updated_var_symbols = [v for v in var_symbols_list if v not in removed_vars]
        return new_eqs, updated_var_symbols, substitutions

    # --- compilation ------------------------------------------------------------------

    @line_profiler.profile
    def instantiate(self, cse=True, aditional_modules=None, max_remove_trival_passes=1,
                    lambda_cache_dir=None):
        if aditional_modules is None:
            aditional_modules = []
        all_modules = ["numpy"] + aditional_modules

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
        self.all_raw_equations, self.raw_var_symbols, self.raw_prev_var_symbols, self.raw_param_symbols = self.get_current_system()
        self.all_raw_symbols = self.raw_var_symbols + self.raw_prev_var_symbols + self.raw_param_symbols + self.t_symbols
        # `assign_symbols` (called by get_current_system) just stamped each leaf with a
        # dotted hierarchical name like "System.ambient_inlet.p_out". Use those for
        # plotting so legends are unambiguous; fall back to the leaf name if a Variable
        # was somehow added without going through assign_symbols.
        self.record['vars_names'] = [getattr(v, 'full_name', v.name) for v in self.raw_vars_references]
        print(len(self.all_raw_symbols))
        print(f"Current system collected in {time.time() - start_time} s")

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

        for template, instances in template_to_instances.items():
            tid = len(template_lambdas)
            n_ph = len(instances[0][1])
            placeholders = [_placeholder(k) for k in range(n_ph)]
            # Block lambda: residual (row 0) + every placeholder derivative
            # (rows 1..n_ph).  CSE across these outputs collapses shared
            # subexpressions like `rho_ph(p_in, h_in)` -- which appears in
            # both the residual and several of its derivatives.
            block = sp.Matrix(
                [template] + [sp.diff(template, ph) for ph in placeholders]
            )
            f = self._lambdify_with_cache(
                f"template_{tid:02d}",
                sp.Matrix(placeholders), block,
                all_modules, cse,
            )
            template_lambdas.append(f)
            template_n_ph.append(n_ph)
            template_keys.append(template)

            for eq_idx, sym_order in instances:
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

        elapsed = time.time() - start_time
        print(f"Per-template lambdify done in {elapsed:.2f} s "
              f"({n_templates} templates, {self._n_instances} instances, "
              f"{self._jac_nnz} nonzeros, "
              f"{100.0 * self._jac_nnz / max(1, self.n_v * n_eq):.2f}% dense)")

        # Eval routines that mimic the previous lambda interface.
        n_eq_local = n_eq

        def _eval_per_template(*args):
            """Returns `(residual_col_vector, jac_values_column_vector)`.

            Looped per-instance because the CoolProp-backed `Symbolic_property`
            functions are scalar (cannot be vectorised across instances).
            """
            vals_arr = np.asarray(args, dtype=float)
            r = np.zeros((n_eq_local, 1))
            per_inst_results = [None] * self._n_instances
            inst_state_indices = self._inst_state_indices
            inst_template = self._inst_template
            inst_eq_idx = self._inst_eq_idx
            templates = self._template_lambdas
            for inst_idx in range(self._n_instances):
                f = templates[inst_template[inst_idx]]
                inst_args = vals_arr[inst_state_indices[inst_idx]]
                result = np.asarray(f(*inst_args)).reshape(-1)
                per_inst_results[inst_idx] = result
                r[inst_eq_idx[inst_idx], 0] = result[0]
            jvals = np.zeros((self._jac_nnz, 1))
            jac_inst = self._jac_inst
            jac_out = self._jac_out
            for g in range(self._jac_nnz):
                jvals[g, 0] = per_inst_results[jac_inst[g]][jac_out[g]]
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
        """
        import builtins as _b
        ns = {"__builtins__": _b.__dict__}
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
