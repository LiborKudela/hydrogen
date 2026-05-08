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

import time

import line_profiler
import numpy as np
import sympy as sp

from .numerics import fast_error_norm, fast_linear_solve, lambdify_compat


class Model:
    def __init__(self):
        self.can_evaluate = True
        self.components = {}
        self.t_symbols = [
            sp.symbols('t', real=True),
            sp.symbols('t_prev', real=True),
            sp.symbols('dt', real=True),
        ]
        self.t_values = [0.0, 0.0, 0.0]
        self.declare_components()
        self.raw_vars_references, self.raw_param_references = self.get_vars_references()
        self.record = {
            'time': [],
            'state': [],
            'vars_names': [v.name for v in self.raw_vars_references],
            'subs': [],
        }

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

    @line_profiler.profile
    def remove_trivial_equations(self, equations, var_symbols):
        """
        Process equations to remove trivial ones (e.g., x - y = 0 or x - 5 = 0) and apply substitutions.
        Returns reduced equations, updated variable symbols, and substitutions.
        """
        substitutions = {}
        new_eqs = []
        removed_vars = set()

        var_symbols_list = list(var_symbols)
        # Surviving-symbol set (incl. params + t) and the strictly-Variable set. We only
        # ever eliminate a symbol that (a) currently survives and (b) is a true Variable.
        # Eliminating a Parameter or `t`/`dt` would shrink `all_improved_symbols` while
        # `values` keeps its original size, causing a positional-arg mismatch downstream.
        var_set = set(var_symbols_list)
        raw_var_set = set(self.raw_var_symbols)

        # Map each current-variable symbol to its previous-step counterpart so we can
        # mirror any substitution we make on the "current" side onto the "previous" side
        # (e.g. der_y = -y  =>  der_y_prev = -y_prev). Built once, reused per equation.
        current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}

        # Step 1: Identify and solve trivial equations, applying substitutions immediately
        keep_eq = False
        print("Identifying trivial equations")
        for eq in equations:
            eq_substituted = eq

            vars_in_eq = eq_substituted.free_symbols
            if len(vars_in_eq) == 2:
                if eq_substituted.is_polynomial() and eq_substituted.as_poly().degree() == 1:
                    eq_substituted = eq_substituted.xreplace(substitutions)
                    # Only eliminate a free Variable: if e.g. one side is a Parameter,
                    # we must solve for the Variable. If neither symbol is a current
                    # surviving Variable (both are params, or one was already removed),
                    # we can't safely eliminate anything; keep the equation.
                    candidates = sorted(
                        [
                            s for s in vars_in_eq
                            if s in raw_var_set and s in var_set and s not in removed_vars
                        ],
                        key=str,
                    )
                    if not candidates:
                        keep_eq = True
                        if keep_eq:
                            new_eqs.append(eq_substituted)
                        continue
                    var1 = candidates[0]
                    try:
                        sol = sp.solve(eq_substituted, var1)
                        if sol:
                            substitutions[var1] = sol[0]
                            removed_vars.add(var1)

                            # Mirror the substitution onto the prev_symbol layer. For ANY
                            # solved RHS (a bare symbol like `y`, a negated symbol `-y`,
                            # or a more general linear expression `2*y - z + 5`), the
                            # equivalent constraint at the previous time step is obtained
                            # by replacing every current-variable symbol with its
                            # prev_symbol. xreplace handles all of those cases uniformly,
                            # unlike a `var.symbol == sol[0]` lookup which silently
                            # fails for non-Symbol RHSs.
                            var1_prev_symbol = current_to_prev.get(var1)
                            if var1_prev_symbol is not None:
                                sol_0_prev = sol[0].xreplace(current_to_prev)
                                substitutions[var1_prev_symbol] = sol_0_prev
                                removed_vars.add(var1_prev_symbol)
                            continue
                        else:
                            keep_eq = True
                    except Exception:
                        keep_eq = True
                else:
                    keep_eq = True
            else:
                keep_eq = True
            if keep_eq:
                new_eqs.append(eq_substituted)

        # Step 2: Apply final substitutions to all remaining equations
        print("Applying substitutions")
        for k, sub in enumerate(substitutions.items()):
            for i in range(len(new_eqs)):
                new_eqs[i] = new_eqs[i].subs(*sub)
            print(f"Applied {k+1}/{len(substitutions)} substitutions ({(k+1)/len(substitutions)*100:.2f}%)", end="\r")
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

    # --- compilation ------------------------------------------------------------------

    @line_profiler.profile
    def instantiate(self, cse=True, aditional_modules=[], max_remove_trival_passes=1):
        all_modules = ["numpy"] + aditional_modules

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

    def record_state(self):
        self.record['time'].append(self.get_t_value())
        full_state = np.asarray(self.lambdified_raw_vars(*self.values)).reshape(-1)
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
        self.t_symbols = [
            sp.symbols('t', real=True),
            sp.symbols('t_prev', real=True),
            sp.symbols('dt', real=True),
        ]

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
