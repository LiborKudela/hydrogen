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

import contextlib
import contextvars
import gc
import itertools
import math
import os
import time

import line_profiler
import multiprocessing as _mp
import numpy as np
import sympy as sp
from sympy.functions.elementary.miscellaneous import MinMaxBase

from .caching import (
    lambda_cache_default_dir,
    lambda_cache_key,
    load_lambdified_source,
    save_lambdified_source,
)
from .numerics import (fast_error_norm, fast_linear_solve, fast_sparse_solve,
                        fast_sparse_solve_cached, lambdify_compat)
from .paramspec import cache_key_flag_names


def _cheap_minmax_is_connected(cls, x, y):
    """Drop-in for `MinMaxBase._is_connected` that omits the expensive
    `factor_terms(x - y)` symbolic-domination retry.

    `Min`/`Max` canonicalisation calls `_is_connected` pairwise on every
    argument to discard provably-dominated ones.  Its best-effort second pass
    runs `factor_terms` (which calls `gcd_terms`) on the difference of two
    arguments -- and for the large choked-flow `Min`/`Max` closures in a
    discretised pipe that single call dominates the *entire* `instantiate()`
    (every `xreplace` that rebuilds such a node re-triggers it).

    We keep the cheap, exact checks -- argument equality and the direct
    relational comparison (which still collapses numeric and obviously-ordered
    args) -- and only skip the costly `factor_terms` simplification retry.
    Skipping it is always safe: sympy itself notes the retry is a conservative
    best-effort, so omitting it can only leave a `Min`/`Max` *less* simplified,
    never wrong.  Applied uniformly across a whole `instantiate()` so every
    equation is canonicalised consistently.
    """
    if x == y:
        return True
    t, f = Max, Min
    for op in "><":
        for _ in range(2):
            try:
                v = x >= y if op == ">" else x <= y
            except TypeError:
                return False  # non-real arg
            if not v.is_Relational:
                return t if v else f
            t, f = f, t
            x, y = y, x
        x, y = y, x
    return False


# Bound names used by the cheap `_is_connected` above.
Min = sp.Min
Max = sp.Max


@contextlib.contextmanager
def _cheap_minmax_simplification():
    """Temporarily patch `MinMaxBase._is_connected` to skip its expensive
    `factor_terms` retry for the duration of an `instantiate()`."""
    orig = MinMaxBase.__dict__["_is_connected"]
    MinMaxBase._is_connected = classmethod(_cheap_minmax_is_connected)
    try:
        yield
    finally:
        MinMaxBase._is_connected = orig


class NewtonConvergenceFailure(RuntimeError):
    """Raised by `custom_solve(..., raise_on_no_convergence=True)` when Newton
    hits `max_iter` without reducing the residual norm below `tol`.

    `solve_adaptive_step` catches this to trigger a step rejection + dt cut,
    so any of the three adaptive strategies can use Newton non-convergence
    as one of their rejection signals (a too-large `dt` typically destabilises
    the Newton solve before it shows up as a violated error metric).
    """

    def __init__(self, error_norm, n_iters, max_iter, tol):
        super().__init__(
            f"Newton failed to converge in {n_iters}/{max_iter} iterations "
            f"(final error norm {error_norm:.3e}, target tol {tol:.3e})"
        )
        self.error_norm = error_norm
        self.n_iters = n_iters
        self.max_iter = max_iter
        self.tol = tol


# Exceptions raised when a Newton iterate lands in a thermodynamically
# invalid state and the property back-end / lambdified residual cannot be
# evaluated there.  CoolProp surfaces out-of-domain inputs (e.g. an enthalpy
# below the medium's minimum after a step overshoots) as plain `ValueError`;
# `ArithmeticError` / `FloatingPointError` cover overflow / divide-by-zero in
# the numeric residual itself.  `custom_solve` catches these during the
# Jacobian/residual evaluation and converts them into a
# `NewtonConvergenceFailure`, so the adaptive stepper rejects the step and
# retries at a smaller `dt` instead of aborting the whole run.
PropertyEvaluationError = (ValueError, ArithmeticError, FloatingPointError)


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


def _mp_fork_available() -> bool:
    """Return whether the multiprocessing ``fork`` start method exists.

    Parallel template lambdify relies on fork so worker processes inherit the
    parent's module globals (CoolProp ``Symbolic_property`` classes are not
    pickle-safe).  Windows has no ``fork`` context, so callers must fall back
    to the sequential path.
    """
    try:
        _mp.get_context("fork")
    except ValueError:
        return False
    return True


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


class EquationCacheValidationError(RuntimeError):
    """A replayed (cached) `declare_equations()` template diverged from a
    freshly built one for the SAME instance.

    Raised only when the paranoid cache-validation guard is enabled (see
    `set_equation_cache_validation` / the `HYDROGEN_VALIDATE_EQUATION_CACHE`
    env var).  A divergence means the per-class template is NOT actually
    instance-invariant: some value that changes the equation set is being
    baked into the equations as a Python literal instead of being represented
    as a `Parameter` (whose numeric value is applied per-instance and so never
    enters the cached template) or listed in `_cache_key_flags` (so that each
    variant gets its own cache entry).  Either fix the offending model or, if
    the difference is a genuine structural toggle, add the controlling flag to
    that class's `_cache_key_flags`.
    """


def _validate_eq_cache_env_default() -> bool:
    raw = os.environ.get("HYDROGEN_VALIDATE_EQUATION_CACHE", "")
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


# Paranoid guard: when enabled, EVERY cache replay (`collect_equations` Path A)
# is checked against a freshly built `declare_equations()` and any mismatch
# raises `EquationCacheValidationError`.  This catches templates that are not
# instance-invariant -- e.g. a third instance whose baked literals differ from
# the first two that happened to agree (the ordinary two-instance validation
# would have promoted such a class to 'cached' and then silently mis-replayed).
# Off by default (it doubles equation-building work); turn it on in CI / tests.
_VALIDATE_EQUATION_CACHE = _validate_eq_cache_env_default()


def set_equation_cache_validation(enabled: bool) -> bool:
    """Enable/disable the paranoid equation-cache validation guard.

    Returns the previous setting so callers (tests) can restore it.  When the
    guard is on, each cache replay is re-derived and compared, raising
    `EquationCacheValidationError` on any divergence.
    """
    global _VALIDATE_EQUATION_CACHE
    prev = _VALIDATE_EQUATION_CACHE
    _VALIDATE_EQUATION_CACHE = bool(enabled)
    return prev


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


def match_name_index(names, req):
    """First index in ``names`` matching ``req``: exact, else dotted-suffix,
    else bare-suffix.  Returns ``None`` if nothing matches.

    This is the shared name-resolution rule used both by recorded-variable
    lookups (:meth:`Model.resolve_vars`) and the service layer's parameter /
    variable matching, so a convenient suffix (``"wall_0_0.C_1"``) resolves the
    same way everywhere.
    """
    for i, n in enumerate(names):
        if n == req:
            return i
    for i, n in enumerate(names):
        if n.endswith("." + req) or n.endswith(req):
            return i
    return None


def match_name_indices(names, req):
    """All indices in ``names`` matching ``req`` (exact if any exact match
    exists, otherwise every dotted-/bare-suffix match).

    Use this to gather a repeated, per-instance quantity -- e.g. every
    ``wall_*.m_dot_a_leak`` across a multi-segment pipe -- in one call.
    """
    exact = [i for i, n in enumerate(names) if n == req]
    if exact:
        return exact
    return [i for i, n in enumerate(names)
            if n.endswith("." + req) or n.endswith(req)]


class Model:
    # Shared symbols used by every Model/Variable instance.  Previously each
    # `Variable.__init__` re-allocated a fresh triple, which for a system with
    # thousands of variables was thousands of unused sympy `Symbol` objects on
    # the heap.  Sympy's `Symbol` cache makes these structurally identical, but
    # holding distinct Python objects per variable still wastes memory.
    # The first three entries are the time triplet `(t, t_prev, dt)`.  The four
    # trailing entries are GLOBAL integration-scheme coefficients shared by every
    # `DifferentialVariable` closure (see `DifferentialVariable.declare_equations`):
    #
    #     x = sch_p0 * x_prev + dt * sch_a * der + (sch_p1 + dt * sch_b) * der_prev
    #
    # They live in the `t`-block (not the parameter block) because, like `dt`,
    # they are model-wide runtime scalars the stepper rewrites per stage rather
    # than per-component compile-time constants.  Crank-Nicolson is the default
    # (`sch_p0, sch_p1, sch_a, sch_b = 1, 0, 1/2, 1/2`); the TR-BDF2 stepper
    # temporarily swaps in its trapezoidal / BDF2 stage coefficients.
    t_symbols = [
        sp.symbols('t', real=True),
        sp.symbols('t_prev', real=True),
        sp.symbols('dt', real=True),
        sp.symbols('sch_p0', real=True),
        sp.symbols('sch_p1', real=True),
        sp.symbols('sch_a', real=True),
        sp.symbols('sch_b', real=True),
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
    #
    # The cache is keyed by `(cls, flag_values)` where `flag_values` are the
    # current values of the per-instance flags named in `_cache_key_flags`.
    # Most classes' `declare_equations` structure is fully determined by the
    # class, so the default is an empty tuple (key == just the class).  A class
    # whose equation structure (or which symbols/variables exist) depends on a
    # constructor flag should mark that flag `ParamSpec(structural=True)` in its
    # `Annotated` type hint; `cache_key_flag_names()` then derives the cache key
    # automatically so two instances with different flag values get DIFFERENT
    # cache entries instead of one replaying the other's template.  This
    # `_cache_key_flags` attribute is for *computed* keys that are NOT
    # constructor arguments (e.g. a private `_perm_key` summarising an injected
    # model's structural identity); it is merged with the structural args by
    # `cache_key_flag_names()`.  Flag values must be hashable.
    _cache_key_flags: tuple = ()

    def __init__(self):
        self.can_evaluate = True
        self.components = {}
        # Variable-pair connections registered by `declare_equations` via
        # `add_connection`.  These are short-circuited at instantiate time
        # (union-find) instead of being threaded through the symbolic
        # trivial-equation reducer, which is much faster for large systems
        # where the bulk of the trivial equations are connection equalities.
        self.connections = []
        # Set once `declare_equations()` has been run for its wiring side
        # effects (so `ensure_equations_declared()` -- used by serialization to
        # wire a model without compiling it -- stays idempotent and never
        # double-wires ports).  `instantiate()` runs `declare_equations()` on
        # its own; this flag only gates the explicit "wire early" helper.
        self._equations_declared = False
        # Typed `Port` registry.  Optional layer on top of `components`;
        # ports merely group existing Variable references and add type/
        # multiplicity checks at `connect()` time.  See `hydrogen/ports.py`.
        self.ports = {}
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
            # Per-recorded-step solver diagnostics, kept index-aligned with
            # `time` / `state`.  `step_wall_time` is the wall-clock seconds the
            # adaptive solver spent producing that step (including any rejected
            # retries); `step_error` is the controller's accepted local-error
            # estimate for that step.  Entries that don't correspond to a solved
            # step -- the initial state from `initialise()` and any manual
            # `next_step()` call -- are recorded as `nan`.
            'step_wall_time': [],
            'step_error': [],
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

    def declare_events(self):
        """Override to declare *explicit time events* for this component.

        Return a list of absolute model times (floats) at which this component's
        driving signal is non-smooth -- a value jump (e.g. a `Step`) or a slope
        kink (e.g. the corners of a `Ramp`).  The integrator clips its step size
        so it never integrates *across* such an instant; instead it lands just
        before it (``t* - event_eps``) and just after it (``t* + event_eps``),
        so each committed step stays on a single smooth branch of the signal.

        This is what lets the adaptive controller keep a clean local-error
        estimate at a discontinuity instead of stalling / rejecting on the kink.
        Times outside the run's ``[t_start, stop_time]`` window are ignored.
        Default: no events.
        """
        return []

    def add_port(self, name, port):
        """Register a `hydrogen.ports.Port` on this Model under `name`.

        Returns the port so users can chain in-place:

            inlet = self.add_port('inlet', FluidPort_phm(self, channels={...}, ...))

        Aliasing a port under a second name (e.g. to expose an inherited
        port via a different label) is not supported; declare a fresh port
        bound to the same channels instead.
        """
        from .ports import Port  # local import to avoid a circular module-load
        if not isinstance(port, Port):
            raise TypeError(
                f"add_port expects a hydrogen.ports.Port instance, got "
                f"{type(port).__name__}"
            )
        if name in self.ports:
            raise ValueError(
                f"Port {name!r} already declared on {getattr(self, 'name', type(self).__name__)}"
            )
        port.name = name
        self.ports[name] = port
        return port

    def connect(self, port_a, port_b):
        """Wire two ports together; emits one `add_connection` per channel.

        See `hydrogen/ports.py` for the channel/orientation semantics.  The
        sign on each `add_connection` is picked automatically:

          * ACROSS channels      ->  sign = +1 (direct equality always)
          * FLOW channels        ->  sign = +1 if orientations differ
                                       (`out` -> `in`, the typical case)
                                     sign = -1 if orientations match
                                       (two `in` or two `out` ports;
                                       sum-to-zero on the flow variable,
                                       which is the Kirchhoff / Modelica
                                       connector convention)

        Two extra checks fire before any connection is recorded so wiring
        mistakes raise at parent-time rather than producing a confusing
        Newton failure:

          * `kind` mismatch                            -> PortKindMismatchError
          * a `medium` mismatch on either side (when   -> PortMediumMismatchError
            both ports declare a non-None medium)
          * either port already wired                  -> PortAlreadyConnectedError
          * required channel missing on the other side -> PortChannelMissingError
        """
        from .ports import (
            PortChannelMissingError,
            PortKindMismatchError,
            PortMediumMismatchError,
        )

        # Idempotent re-wire: if these two ports are already wired to each
        # other (e.g. `declare_equations()` ran once for serialization and is
        # run again by `instantiate()`), do nothing -- so the ports and the
        # `add_connection` list are never doubled.  Wiring a port to a *new*
        # partner still raises `PortAlreadyConnectedError` below as usual.
        if port_a._connected_to is port_b and port_b._connected_to is port_a:
            return

        if port_a.kind != port_b.kind:
            raise PortKindMismatchError(
                f"Cannot connect ports of different kinds: "
                f"{port_a._path()} ({port_a.kind!r}) <-> "
                f"{port_b._path()} ({port_b.kind!r})"
            )
        if (port_a.medium is not None and port_b.medium is not None
                and port_a.medium is not port_b.medium):
            raise PortMediumMismatchError(
                f"Cannot connect fluid ports with different media: "
                f"{port_a._path()} (medium={port_a.medium}) <-> "
                f"{port_b._path()} (medium={port_b.medium})"
            )

        flow_set = set(port_a.flow_channels)
        if flow_set != set(port_b.flow_channels):
            raise PortChannelMissingError(
                f"Flow-channel sets disagree on matching-kind ports: "
                f"{port_a._path()} {port_a.flow_channels} vs "
                f"{port_b._path()} {port_b.flow_channels}"
            )

        port_a._mark_connected(port_b)
        port_b._mark_connected(port_a)

        same_orientation = (port_a.flow_orientation == port_b.flow_orientation)
        # Iterate `port_a.channels` in insertion order so adjacent runs of
        # `connect()` produce the same connection list (the UF pass is
        # order-independent in result but the pretty-printed counters
        # depend on it).
        for ch_name in port_a.channels:
            if ch_name not in port_b.channels:
                raise PortChannelMissingError(
                    f"Channel {ch_name!r} present on {port_a._path()} but missing "
                    f"on {port_b._path()}"
                )
            va = port_a.channels[ch_name]
            vb = port_b.channels[ch_name]
            sign = -1 if (ch_name in flow_set and same_orientation) else +1
            self.add_connection(va, vb, sign=sign)

    def add_connection(self, var_a, var_b, sign=+1):
        """Declare a linear constraint between two `Variable`s, resolved via
        signed union-find BEFORE the symbolic-equation machinery runs.

        With the default `sign=+1` this is the classical "same value" wire:
        the constraint is `var_a - var_b == 0` and the two symbols collapse
        into one shared representative.  With `sign=-1` the constraint is
        `var_a + var_b == 0` (the value of one is the *negation* of the
        other); the two symbols collapse to a single representative plus a
        recorded sign, and every downstream `xreplace` emits `s -> rep`
        for one and `s -> -rep` for the other.

        Either form is functionally equivalent to returning the same sympy
        expression from `declare_equations()`, but much cheaper at scale:
          * no sympy `Add` is built per connection,
          * the connection never enters the equation list,
          * the trivial-equation reducer doesn't have to discover it.

        Use `sign=+1` for "same physical port wired together" (pipe-to-pipe
        segment continuity, splitter outlet -> child inlet, ...).  Use
        `sign=-1` when two ports of the same flow orientation are wired
        (e.g. a junction port whose `m_dot` is "into me" connected to a
        pipe inlet whose `m_dot` is also "into me") -- the resulting
        constraint encodes "what enters one face must enter the other from
        the opposite side".  The high-level `Model.connect()` /
        `hydrogen.ports.Port` layer picks the sign automatically.
        """
        if sign not in (+1, -1):
            raise ValueError(f"connection sign must be +1 or -1, got {sign!r}")
        self.connections.append((var_a, var_b, sign))

    def collect_connections(self):
        """Recursively gather every `(var_a, var_b, sign)` connection registered
        in the tree.  Mirror of `collect_equations` but for connections only.

        Legacy 2-tuple entries (from code that pre-dates signed union-find)
        are widened to 3-tuples with `sign=+1` on the fly so external
        callers and any in-tree composite that bypasses `add_connection`
        keep working without modification.

        IMPORTANT: connections are declared *inside* `declare_equations`
        (because that's where the user has access to sub-components), so
        callers MUST call `collect_equations` first to flush them onto each
        component's `connections` list.
        """
        def _norm(c):
            # Accept either (a, b) or (a, b, sign); always emit (a, b, sign).
            if len(c) == 2:
                return (c[0], c[1], +1)
            return c

        conns = [_norm(c) for c in self.connections]
        for c in self.components.values():
            if isinstance(c, Model) and c.is_composite():
                conns.extend(c.collect_connections())
        return conns

    def _is_wired(self):
        """True if `declare_equations()` has already run for its wiring side
        effects -- i.e. this model (or a nested composite) carries any
        `add_connection` entry or has a connected port.

        This reflects the *actual* wiring state (not just the
        `_equations_declared` flag), so it also recognises a model wired by a
        plain `instantiate()` or a hand call to `declare_equations()`.
        """
        if self.connections:
            return True
        for comp in self.components.values():
            for port in getattr(comp, "ports", {}).values():
                if getattr(port, "is_connected", False):
                    return True
            if isinstance(comp, Model) and comp.is_composite() and comp._is_wired():
                return True
        return False

    def ensure_equations_declared(self):
        """Idempotently run `declare_equations()` once for its wiring side
        effects (port `connect()` / `add_connection`), so callers that need the
        model **wired but not compiled** -- chiefly serialization -- don't have
        to call `declare_equations()` (or `instantiate()`) by hand first.

        It is a no-op if the model is already wired (via a prior
        `instantiate()`, a hand `declare_equations()` call, or an earlier
        `ensure_equations_declared()`), so it never double-wires a port.

        NB: like the framework's once-only `declare_equations()` contract, do
        not call this and *then* `instantiate()` the same object -- instantiate
        re-runs `declare_equations()` and would re-wire already-connected ports.
        Serialization dumps a built model and (re)builds a fresh one to run, so
        this never arises there.
        """
        if self._equations_declared or self._is_wired():
            self._equations_declared = True
            return
        self.declare_equations()
        self._equations_declared = True

    def is_composite(self):
        for c in self.components.values():
            if isinstance(c, Model):
                return True
        return False

    def differential_variables(self, _prefix=""):
        """Dotted names of every `DifferentialVariable` in this model's subtree.

        A model with NONE is *quasi-static* -- its state is purely algebraic,
        so it has no time-integrated memory; a model with one or more is
        *dynamic* -- it carries state that evolves in time (a thermal mass, a
        stored gas inventory, a diffusion front, ...).

        Walks the constructed component tree, so it works on any built instance
        WITHOUT a full `instantiate()` (handy for UI categorisation).  Each
        differential state is reported once: a `DifferentialVariable` named
        ``x`` adds an algebraic ``der_x`` companion, which is a plain
        `Variable` and so is not double-counted.
        """
        found = []
        for name, c in self.components.items():
            path = f"{_prefix}.{name}" if _prefix else name
            if isinstance(c, DifferentialVariable):
                found.append(path)
            elif isinstance(c, Model) and c.is_composite():
                found.extend(c.differential_variables(_prefix=path))
        return found

    def is_dynamic(self):
        """``True`` if this model carries any `DifferentialVariable` (dynamic);
        ``False`` if it is purely algebraic (quasi-static).

        Short-circuits on the first differential state found; see
        `differential_variables` for the full list.
        """
        for c in self.components.values():
            if isinstance(c, DifferentialVariable):
                return True
            if isinstance(c, Model) and c.is_composite() and c.is_dynamic():
                return True
        return False

    def _iter_ports(self):
        """Yield every `Port` registered anywhere in this Model's subtree."""
        for p in getattr(self, "ports", {}).values():
            yield p
        for c in self.components.values():
            if isinstance(c, Model):
                yield from c._iter_ports()

    def _warn_unconnected_required_ports(self):
        """Emit a `PortNotConnectedWarning` for each `require_connection` port
        that ended up unwired.  Such a port leaves its across-variable
        unclosed, so the system is singular; warning here turns an opaque
        "Factor is exactly singular" into an actionable message."""
        import warnings

        from .ports import PortNotConnectedWarning

        for p in self._iter_ports():
            if getattr(p, "require_connection", False) and not p.is_connected:
                warnings.warn(
                    f"Port {p._path()} (kind={p.kind!r}) was declared "
                    f"require_connection=True but is not connected to anything. "
                    f"Its across-variable is left unclosed, so the system will be "
                    f"singular. Connect a boundary/component to this port.",
                    PortNotConnectedWarning,
                    stacklevel=2,
                )

    def get_vars_references(self):
        vars_references = []
        param_references = []
        for _, c in self.components.items():
            if isinstance(c, ParameterAlias):
                # The target Parameter is owned (and accounted for) by
                # whichever Model has it directly in its `components`;
                # the alias only forwards reads.
                continue
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
            if isinstance(c, ParameterAlias):
                # Alias: the target Parameter is owned by another Model in
                # the tree and that Model's `assign_symbols` walk assigns
                # the symbol there.  Skip here to avoid double-naming.
                continue
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

        `ParameterAlias` entries are included under their LOCAL path: this
        is what lets the cache replay correctly across instances that
        share the same alias structure but point at different target
        Parameters (e.g. `pipe1.pipe_segment_*.A_in -> pipe1.A` vs
        `pipe2.pipe_segment_*.A_in -> pipe2.A`).
        """
        out = {}
        for name, c in self.components.items():
            if isinstance(c, ParameterAlias):
                sym = c.symbol
                if sym is not None:
                    out[(name,)] = sym
            elif isinstance(c, (Variable, Parameter)):
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
        # Cache key: the class PLUS the values of any per-instance flags the
        # class declares as equation-structure-affecting (`_cache_key_flags`,
        # default empty).  Keying by class alone would let two instances of the
        # same class but with different structure-affecting flags (e.g. a
        # `dynamic=True` vs `dynamic=False` wall) collide and replay each
        # other's template.  Including the flag values gives each variant its
        # own entry, so caching still fires WITHIN a variant while staying
        # correct ACROSS variants.
        key = (cls, tuple(getattr(self, name)
                          for name in cache_key_flag_names(cls)))
        # `eq_cache is None` -> running outside `instantiate()` (e.g. via
        # `get_current_system()` or a unit test).  Just rebuild equations
        # every time; the cache exists purely as an `instantiate()` speed-up.
        cache = eq_cache.get(key) if eq_cache is not None else None

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
                replayed = [eq.xreplace(mapping) if eq.free_symbols else eq
                            for eq in cache['eqs']]
                if _VALIDATE_EQUATION_CACHE:
                    self._validate_replayed_equations(replayed)
                eqs.extend(replayed)
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
                        eq_cache[key] = {'state': 'no-cache'}
                    else:
                        sym_paths = self._build_sym_paths()
                        sym_to_path = {s: p for p, s in sym_paths.items()}
                        used_syms = set()
                        for eq in run_eqs:
                            used_syms.update(eq.free_symbols)
                        eq_cache[key] = {
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

    def _validate_replayed_equations(self, replayed):
        """Paranoid guard: assert a cache-replayed template equals a freshly
        built `declare_equations()` for THIS instance.

        Only invoked when `_VALIDATE_EQUATION_CACHE` is on.  A mismatch means
        the cached template is not instance-invariant (a structure-affecting
        value is baked as a literal rather than a `Parameter`, or a structural
        toggle is missing from `_cache_key_flags`).  Re-deriving the equations
        must be side-effect free here: only classes without `add_connection`
        side effects ever reach the 'cached' state, so `declare_equations()`
        appends nothing to `self.connections`; we snapshot/restore its length
        defensively so the guard can never corrupt instantiate state.
        """
        n_conns_before = len(self.connections)
        fresh = self.declare_equations()
        if len(self.connections) > n_conns_before:
            del self.connections[n_conns_before:]

        same = (len(fresh) == len(replayed)
                and all(a == b for a, b in zip(replayed, fresh)))
        if same:
            return

        diff_idx = next(
            (i for i in range(min(len(fresh), len(replayed)))
             if replayed[i] != fresh[i]),
            min(len(fresh), len(replayed)),
        )
        detail = ""
        if diff_idx < min(len(fresh), len(replayed)):
            detail = (f"\n  first divergence at equation #{diff_idx}:"
                      f"\n    cached : {replayed[diff_idx]}"
                      f"\n    fresh  : {fresh[diff_idx]}")
        raise EquationCacheValidationError(
            f"Cached equation template for {type(self).__name__} does not "
            f"match a freshly built declare_equations() for this instance "
            f"({len(replayed)} cached vs {len(fresh)} fresh equations); the "
            f"template is not instance-invariant.  Represent any "
            f"structure-affecting value as a Parameter, or add its controlling "
            f"flag to {type(self).__name__}._cache_key_flags.{detail}"
        )

    @line_profiler.profile
    def get_current_system(self):
        vars, prev_vars, params, vars_map, params_map = self.assign_symbols(top_level=True)
        equations = self.collect_equations()
        return equations, vars, prev_vars, params

    # --- trivial-equation reduction ---------------------------------------------------

    @staticmethod
    def _classify_linear(eq, param_set=None):
        """Try to express `eq` as `c0 + sum(ci * vi) == 0` where every `vi` is a
        sympy Symbol and every coefficient is a Number.

        Returns `(const_term, {symbol: coeff})` if successful, otherwise `None`.

        Implemented as a recursive structural walk over `Add`/`Mul`/`Symbol`/`Number`
        nodes -- crucially this never builds a `Poly` (which is the slow operation
        the previous `is_polynomial()`/`as_poly().degree()` path was triggering on
        every equation, including the many CoolProp-laden ones that are obviously
        non-linear).

        When `param_set` is given (a set of Parameter symbols), linearity is
        classified in the VARIABLES only: any subexpression whose free symbols
        are all in `param_set` (plus pure Numbers) is treated as part of the
        constant coefficient field.  This recognises e.g. `a + k*b` (with `k` a
        Parameter) as linear in `{a, b}` with coefficients `{a: 1, b: k}` -- a
        wiring whose Jacobian entries are constant w.r.t. the Newton state and
        so eliminable just like the numeric-coefficient case.  When `param_set`
        is None the legacy behaviour holds: every symbol is an indeterminate and
        only Numbers may be coefficients.
        """
        if param_set is not None:
            # Any subexpression free of NON-parameter symbols (numbers and/or
            # parameters only) is a constant coefficient.
            if eq.free_symbols <= param_set:
                return eq, {}
            if isinstance(eq, sp.Symbol):
                # Has a non-parameter free symbol and is a bare Symbol -> it is
                # a variable indeterminate (a parameter symbol would have been
                # caught by the subset check above).
                return sp.S.Zero, {eq: sp.S.One}
            if eq.is_Mul:
                coeff = sp.S.One
                sym = None
                for arg in eq.args:
                    if arg.free_symbols <= param_set:
                        coeff = coeff * arg          # number or parameter expr
                    elif isinstance(arg, sp.Symbol):
                        if sym is not None:
                            return None              # var*var -> nonlinear
                        sym = arg
                    else:
                        return None                  # var**2, f(var), ... -> nonlinear
                if sym is None:
                    return coeff, {}
                return sp.S.Zero, {sym: coeff}
            if eq.is_Add:
                const = sp.S.Zero
                coeffs = {}
                for term in eq.args:
                    sub = Model._classify_linear(term, param_set)
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

    def _dynamic_param_symbols(self):
        """Set of parameter symbols whose VALUE changes within a step (the
        `cur`/`prev` leaves of every `Input`).

        Used by the parameter-coefficient linear eliminators to refuse a
        substitution whose RHS depends on a time-varying parameter: the
        prev-step mirror (`xreplace(current_to_prev)`) only maps variable
        cur->prev symbols, so a dynamic parameter would be frozen at its
        current-level value on the previous time level -- wrong for an Input
        whose `cur` and `prev` genuinely differ."""
        dyn = set()
        for inp in self._collect_inputs():
            for leaf in ('cur', 'prev'):
                comp = inp.components.get(leaf)
                s = getattr(comp, 'symbol', None) if comp is not None else None
                if s is not None:
                    dyn.add(s)
        return dyn

    def _differential_state_symbols(self):
        """Set of symbols backing `DifferentialVariable` STATES.

        The linear eliminators must never pivot on (i.e. substitute away) a
        differential state: the time steppers advance these states through
        their `prev` slots (`_get_diff_state_indices`; TR-BDF2's BDF2 stage
        folds `c2*x_n + c1*x_gamma` into the state's prev value), and the
        Crank-Nicolson closure equation references `x_prev`.  If the state's
        symbol is substituted out, the prev-slot writes land on a slot no
        surviving equation reads, silently freezing the state's dynamics
        (observed as broken mass conservation when a global pressure state
        was inlined as `mean(pc)` by the linear-block pass).

        Derivative companions (`der_x`) are deliberately NOT included: the
        steppers tolerate an inlined derivative definition (see
        `_get_diff_state_indices`)."""
        states = set()

        def _walk(node):
            if isinstance(node, DifferentialVariable) and node.symbol is not None:
                states.add(node.symbol)
            if isinstance(node, Model):
                for child in node.components.values():
                    _walk(child)

        _walk(self)
        return states

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

        # Iterative post-order DFS: a long substitution chain (e.g. the
        # ~N-deep face-pressure chain of a finely-segmented pipe) used to
        # blow Python's recursion limit around N ~ 1000 cells.
        def resolve(root):
            stack = [(root, False)]
            while stack:
                k, expanded = stack.pop()
                if k in cache:
                    continue
                v = substitutions[k]
                deps = v.free_symbols & keys
                if expanded:
                    visiting.discard(k)
                    cache[k] = v.xreplace({d: cache[d] for d in deps}) \
                        if deps else v
                    continue
                todo = [d for d in deps if d not in cache]
                if not todo:
                    cache[k] = v.xreplace({d: cache[d] for d in deps}) \
                        if deps else v
                    continue
                if k in visiting:
                    raise ValueError(
                        f"Cycle in trivial-equation substitutions involving {k}")
                visiting.add(k)
                stack.append((k, True))
                for d in todo:
                    if d in visiting:
                        raise ValueError(
                            f"Cycle in trivial-equation substitutions "
                            f"involving {d}")
                    stack.append((d, False))
            return cache[root]

        for k in list(substitutions.keys()):
            substitutions[k] = resolve(k)
        return substitutions

    @line_profiler.profile
    def remove_trivial_equations(self, equations, var_symbols, allow_param_coeffs=False):
        """Eliminate trivially-linear equations (`a*x + b*y + c == 0` with `a,b,c`
        constants and `x,y` symbols) without invoking `sp.solve`/`sp.Poly`.

        When `allow_param_coeffs` is True the notion of "constant" is widened
        from pure Numbers to "Numbers and Parameters", so wirings like
        `0 = a + k*b` (with `k` a Parameter) become eligible.  Two guards keep
        this safe: (1) the eliminated variable's own coefficient must still be a
        nonzero Number, so we never bake `1/parameter` (a possible runtime
        division-by-zero) into a substitution; (2) substitutions whose RHS
        references a time-varying (`Input`) parameter are skipped, because the
        prev-step mirror can't carry such a parameter to the old time level.

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
        diff_state_set = self._differential_state_symbols()
        current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}
        # Parameter-coefficient mode: classify linearity in the variables while
        # treating Parameters as part of the constant coefficient field.
        param_set = set(self.raw_param_symbols) if allow_param_coeffs else None
        dynamic_param_set = self._dynamic_param_symbols() if allow_param_coeffs else set()

        print("Identifying trivial equations (structural)")
        for idx, eq in enumerate(equations):
            classified = Model._classify_linear(eq, param_set)
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
            #
            # In `allow_param_coeffs` mode we ALSO accept the single-variable case
            # (`len == 1`) -- but only when the constant `c0` references a
            # parameter.  This is exactly the `var = parameter-expression` wiring
            # (e.g. `0 = a - k`) that the legacy classifier handled implicitly by
            # treating the parameter as a pseudo-variable; once parameters are
            # folded into the constant field that equation drops to one variable,
            # so without this branch enabling the flag would ELIMINATE FEWER
            # equations.  Pure-number fixes (`0 = 2*a - 5`) stay excluded, matching
            # legacy.
            if allow_param_coeffs:
                if len(coeffs) == 2:
                    pass
                elif len(coeffs) == 1 and getattr(c0, "free_symbols", set()):
                    pass
                else:
                    new_eqs.append(eq)
                    continue
            elif len(coeffs) != 2:
                new_eqs.append(eq)
                continue

            # Only true Variables that are still surviving and not already eliminated
            # this pass are eligible.  Differential states are never eligible:
            # eliminating one breaks the stepper's prev-slot advancement (see
            # `_differential_state_symbols`).
            candidates = [
                s for s in coeffs
                if s in raw_var_set and s in var_set and s not in removed_vars
                and s not in diff_state_set
            ]
            if allow_param_coeffs:
                # Guard 1 (safe division): only eliminate a variable whose own
                # coefficient is a nonzero Number.  The OTHER term's coefficient
                # may be a parameter expression, but dividing by it could be a
                # runtime division-by-zero, so it never becomes the pivot.
                candidates = [
                    s for s in candidates
                    if getattr(coeffs[s], "is_Number", False) and coeffs[s] != 0
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

            # Guard 2 (prev-step mirror): a substitution that references a
            # time-varying parameter can't be mirrored to the previous time
            # level correctly (see `_dynamic_param_symbols`), so leave the
            # equation in the system.
            if allow_param_coeffs and (sol.free_symbols & dynamic_param_set):
                new_eqs.append(eq)
                continue

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

            # Apply the closed substitutions in a single pass: equations whose
            # `free_symbols` are disjoint from the eliminated set keep their
            # identity (no tree walk), the rest get ONE direct `xreplace`.
            #
            # A previous per-template placeholder cache here was ~19x SLOWER on
            # channel / wall-heavy models (measured 26.8 s vs 1.4 s at N=300,
            # identical output): normalising each instance to a template,
            # hashing that CoolProp-laden tree for the cache key, and rebinding
            # placeholders cost far more than a single `xreplace` -- even at a
            # ~98% template-hit rate, because the rebind is itself a full tree
            # walk so the cache never avoids the dominant cost.
            n_skipped = 0
            n_applied = 0
            new_eqs_out = []
            for e in new_eqs:
                if e.free_symbols & sub_keys:
                    new_eqs_out.append(e.xreplace(substitutions))
                    n_applied += 1
                else:
                    new_eqs_out.append(e)
                    n_skipped += 1
            print(
                f"Applying substitutions to {n_applied}/{len(new_eqs)} equations "
                f"(skipped {n_skipped} disjoint)"
            )
            new_eqs = new_eqs_out
        else:
            print("No substitutions applied")

        updated_var_symbols = [v for v in var_symbols_list if v not in removed_vars]
        return new_eqs, updated_var_symbols, substitutions

    # --- duplicate-equation reduction -------------------------------------------------

    @line_profiler.profile
    def remove_duplicate_equations(self, equations, var_symbols):
        """Collapse pairs (or groups) of equations of the shape
        `alpha * var + R == 0` whose `(alpha, R)` are structurally identical
        apart from the linear leaf `var`.

        Such pairs imply `var_a == var_b` (one degree of redundancy per
        match: `var_a` and `var_b` differ only by a single rename in two
        otherwise-identical equations).  Removing one equation and rewriting
        `var_b -> var_a` in every survivor leaves the system square AND
        smaller.

        The canonical motivating case is a `StraightPipe`'s per-segment
        face-velocity closures: after `add_connection` unifies `(p, h,
        m_dot)` across an internal interface and `StraightPipe`'s shared
        `A` Parameter makes both sides reference the same area symbol, the
        two closures
            m_dot - rho_ph(p, h) * A * w_out(seg_k)         == 0
            m_dot - rho_ph(p, h) * A * w_in(seg_{k+1})      == 0
        are structurally identical apart from the `w_*` leaf.  This pass
        unifies `w_in(seg_{k+1}) := w_out(seg_k)` and drops one of the two
        equations, saving `N - 1` variables AND equations per pipe.

        Strategy:
          1. For each equation, enumerate every leaf Variable that appears
             strictly linearly with an in-leaf-constant coefficient (so the
             equation decomposes as `coeff * var + rest == 0`).
          2. Bucket equations by `(coeff, rest)` SymPy structural identity.
          3. The first equation registered in each bucket becomes the
             representative; subsequent matches schedule `dup_var ->
             keeper_var` substitutions and drop their equation.
          4. Close substitution chains, apply via `xreplace` to survivors,
             and mirror onto the prev-step companion symbols so the
             time-stepping bookkeeping stays consistent.

        Safety:
          * Only acts on `coeff != 0` (structural zero check, matching the
            convention used by `remove_trivial_equations`).
          * The implicit division by `coeff` it represents is safe for the
            face-closure use case because `coeff = -rho * A` is non-zero on
            the physical domain (positive density, positive area); users
            with constructions where the candidate coefficient can pass
            through zero should disable this pass via
            `instantiate(max_remove_duplicate_passes=0)`.
        """
        raw_var_set = set(self.raw_var_symbols)
        var_set = set(var_symbols)
        diff_state_set = self._differential_state_symbols()
        current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}

        # `signatures[(coeff, rest)] = (var, idx)` -- first equation registered
        # for each structural template gets to keep its variable.
        signatures = {}
        duplicate_subs = {}     # var_to_eliminate -> keeper (current AND prev)
        drop_indices = set()
        removed_vars = set()

        def _decompose(eq):
            """Enumerate (var, coeff, rest) splits of `eq` where `var` is a
            current-step Variable leaf appearing strictly linearly.

            Sign-canonicalization: `a*x + R == 0` and `-a*x - R == 0` encode
            the same constraint, so we normalise the signature by flipping
            BOTH `coeff` and `rest` whenever `coeff` carries a top-level
            minus sign (`could_extract_minus_sign()`).  Without this,
            adjacent pipe-segment face closures under the "flow into me"
            convention end up in opposite-sign sympy forms after the signed
            UF substitution (`m_dot_out - rho*A*w_out = 0` on the in-face
            of seg_{k+1} vs the out-face's `-m_dot_in_{k+1} + rho*A*w_out =
            0`); their `(coeff, rest)` would otherwise hash to different
            buckets and the dedup pass would silently miss the match.

            The keeper-var bookkeeping is unaffected: dedup still records
            `dup_var -> keeper_var` (and the prev-step mirror), only the
            hash key is canonicalised.
            """
            out = []
            for s in eq.free_symbols:
                if s not in raw_var_set or s not in var_set or s in removed_vars:
                    continue
                coeff = eq.coeff(s, 1)
                if coeff == 0:
                    continue
                if s in coeff.free_symbols:
                    continue
                rest = eq - coeff * s
                if s in rest.free_symbols:
                    continue
                # Canonicalise (coeff, rest) by sign so `(a, R)` and `(-a, -R)`
                # land in the same bucket.  Use sympy's built-in heuristic
                # `could_extract_minus_sign` -- it's the same predicate sympy
                # itself uses for `Add` canonicalisation.
                try:
                    if coeff.could_extract_minus_sign():
                        coeff = -coeff
                        rest = -rest
                except AttributeError:
                    # `coeff` may be a plain Python number (e.g. `-1`); fall
                    # back to a numeric sign check.
                    if coeff < 0:
                        coeff = -coeff
                        rest = -rest
                out.append((s, coeff, rest))
            return out

        print("Identifying duplicate equations (structural)")
        for idx, eq in enumerate(equations):
            cands = _decompose(eq)
            if not cands:
                continue

            matched = False
            for var, coeff, rest in cands:
                sig = (coeff, rest)
                existing = signatures.get(sig)
                if existing is None:
                    continue
                keeper_var, keeper_idx = existing
                if keeper_var is var:
                    # Exact structural duplicate: same coeff, rest AND leaf, so
                    # `coeff*var + rest` is the identical equation.  Drop the
                    # redundant copy outright (no rename needed).  This is the
                    # capacitive-node-merge case -- two heat-capacity surface
                    # nodes wired together (e.g. adjacent dynamic wall layers)
                    # collapse their two now-identical state closures into one.
                    if idx != keeper_idx:
                        drop_indices.add(idx)
                        matched = True
                        break
                    continue
                if keeper_var in removed_vars:
                    continue
                # Differential states must survive renames: the steppers
                # advance them through their prev slots (see
                # `_differential_state_symbols`).  If the dup leaf is a diff
                # state, either flip the rename direction (algebraic keeper
                # absorbed by the state) or -- when both are diff states --
                # leave the pair alone.
                if var in diff_state_set:
                    if keeper_var in diff_state_set:
                        continue
                    duplicate_subs[keeper_var] = var
                    removed_vars.add(keeper_var)
                    var_prev = current_to_prev.get(var)
                    keeper_prev = current_to_prev.get(keeper_var)
                    if var_prev is not None and keeper_prev is not None:
                        duplicate_subs[keeper_prev] = var_prev
                        removed_vars.add(keeper_prev)
                    signatures[sig] = (var, idx)
                    drop_indices.add(keeper_idx)
                    matched = True
                    break
                duplicate_subs[var] = keeper_var
                removed_vars.add(var)
                var_prev = current_to_prev.get(var)
                keeper_prev = current_to_prev.get(keeper_var)
                if var_prev is not None and keeper_prev is not None:
                    duplicate_subs[var_prev] = keeper_prev
                    removed_vars.add(var_prev)
                drop_indices.add(idx)
                matched = True
                break

            if matched:
                continue

            # No match -- register EVERY decomposition of this equation so
            # future equations can match on any of them.  This matters
            # because the matching decomposition may not be the
            # `_decompose` iteration order's first (e.g. the face closure
            # has BOTH a `var=m_dot, coeff=1, rest=-rho*A*w` decomposition
            # AND a `var=w, coeff=-rho*A, rest=m_dot` decomposition; the
            # latter is the cross-segment-aligned one).
            for var, coeff, rest in cands:
                sig = (coeff, rest)
                if sig not in signatures:
                    signatures[sig] = (var, idx)

        if not duplicate_subs and not drop_indices:
            print("No duplicate equations found")
            return equations, var_symbols, {}

        print(
            f"Identified {len(drop_indices)} duplicate equation(s) / "
            f"{len(duplicate_subs)} substitution(s)"
        )
        Model._close_substitutions(duplicate_subs)
        sub_keys = set(duplicate_subs.keys())

        new_eqs = []
        for idx, eq in enumerate(equations):
            if idx in drop_indices:
                continue
            if eq.free_symbols.isdisjoint(sub_keys):
                new_eqs.append(eq)
            else:
                new_eqs.append(eq.xreplace(duplicate_subs))

        removed_set = set(duplicate_subs.keys())
        new_var_symbols = [s for s in var_symbols if s not in removed_set]

        return new_eqs, new_var_symbols, duplicate_subs

    # --- Tearing (greedy feedback vertex set within BLT blocks) ----------------------

    @staticmethod
    def _tear_block_greedy(block_rows, block_cols, n_local):
        """Greedy heuristic tearing for one BLT block.

        Identifies a small set of "tearing" variables whose removal makes the
        intra-block dependency graph acyclic.  The non-tear variables can then
        (in principle) be solved in topological order given the tear values --
        which is what would let an outer Newton iterate over only the tear
        variables rather than the full block.

        Algorithm (MTK/Cellier-Elmqvist style greedy):
          1. Build the directed dependency graph on the block's variables: edge
             var_j -> var_k iff equation matched(j) references var k (j != k,
             both inside the block).
          2. While there are SCCs of size > 1 in the remaining graph:
               * pick the vertex with the largest (in_deg + out_deg) inside any
                 nontrivial SCC -- this is the variable that, if torn, breaks
                 the most cycles per removal.
               * mark it as a tearing variable and remove it from the graph.
          3. Return the list of tearing variable indices (block-local).

        `block_rows`/`block_cols` are the LOCAL (within-block) coordinates of
        the block's Jacobian nonzeros (i.e. eq-row to var-col edges, both
        relabelled to 0..n_local-1 by `_build_blt_plan`'s `eq_local`/`var_local`
        translation).  The function returns a list of local var indices.

        This is heuristic and small-graphs-only -- for blocks beyond a few
        hundred variables you'd want a smarter algorithm.  For hydrogen's
        typical loop blocks (pipe-tree split joints, MixingJunction loops),
        the input is well under that scale.
        """
        # Note on cost: for a 250-var block this runs in a few milliseconds;
        # we cap the loop to `n_local` iterations as a safety net for
        # pathological inputs.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        # Build var-> var directed adjacency from the block's local J pattern.
        # We assume `_build_blt_plan` ran a perfect matching on the BLOCK so
        # that local-row i corresponds to local-var i (the diagonal entry).
        # In practice we pass the diag+lower coords for the block; we just
        # drop self-loops below.
        edges_src = []
        edges_dst = []
        for r, c in zip(block_rows, block_cols):
            if r != c:
                edges_src.append(r)
                edges_dst.append(c)
        if not edges_src:
            return []

        # Use scipy CSR + connected_components for SCC detection.
        n = n_local
        adj_data = np.ones(len(edges_src), dtype=np.int8)
        adj = csr_matrix(
            (adj_data, (edges_src, edges_dst)), shape=(n, n)
        )

        tear_vars = []
        removed = np.zeros(n, dtype=bool)

        for _it in range(n):
            # Find SCCs of the surviving subgraph.
            keep_mask = ~removed
            if not keep_mask.any():
                break
            sub = adj[keep_mask][:, keep_mask]
            n_sub = sub.shape[0]
            if sub.nnz == 0:
                break
            n_comp, labels = connected_components(
                sub, directed=True, connection='strong', return_labels=True
            )
            # Map local labels back to global indices.
            local_to_global = np.flatnonzero(keep_mask)
            # Find sizes of each SCC; we only need to tear inside SCCs of size>1.
            counts = np.bincount(labels)
            big_sccs = np.where(counts > 1)[0]
            if big_sccs.size == 0:
                break
            # Pick the SCC with the largest count -- tearing inside the
            # biggest cycle yields the largest structural reduction.
            target_scc = int(big_sccs[np.argmax(counts[big_sccs])])
            target_locals = np.where(labels == target_scc)[0]
            target_globals = local_to_global[target_locals]

            # Score each candidate by total degree within the surviving graph.
            # Higher = more cycles depend on this var.
            sub_for_target = adj[target_globals][:, target_globals]
            out_deg = np.asarray(
                sub_for_target.sum(axis=1)
            ).ravel()
            in_deg = np.asarray(
                sub_for_target.sum(axis=0)
            ).ravel()
            total_deg = in_deg + out_deg
            pick_local = int(np.argmax(total_deg))
            pick_global = int(target_globals[pick_local])
            tear_vars.append(pick_global)
            removed[pick_global] = True

        return sorted(tear_vars)

    def _compute_tearing(self):
        """Run greedy tearing on every BLT block of size > 1 and report stats.

        Populates `self._blt_plan['tear_vars_per_block']` with one list per
        block (empty for 1x1s).  Does not currently change the Newton solve
        path -- converting tearing into a runtime speedup requires either
        symbolic re-derivation (so the reduced block can be lambdified at a
        smaller size) or a nested-Newton solver with implicit differentiation
        through the inner substitution; both are substantial follow-ups beyond
        the structural analysis here.  See doc/passes_blt_tearing.md for the
        full discussion.
        """
        plan = self._blt_plan
        if plan is None:
            return
        block_vars = plan['block_vars']
        diag_local_rows = plan['diag_local_rows']
        diag_local_cols = plan['diag_local_cols']
        block_n = plan['block_n']

        tear_per_block = []
        total_n = 0
        total_tear = 0
        biggest_after = 0
        for b in range(plan['n_blocks']):
            n_b = int(block_n[b])
            if n_b <= 1:
                tear_per_block.append([])
                continue
            tears = self._tear_block_greedy(
                diag_local_rows[b], diag_local_cols[b], n_b
            )
            tear_per_block.append(tears)
            total_n += n_b
            total_tear += len(tears)
            reduced = n_b - len(tears)
            biggest_after = max(biggest_after, reduced)
        plan['tear_vars_per_block'] = tear_per_block

        n_loop_blocks = sum(1 for ts in tear_per_block if ts)
        if total_n:
            print(
                f"Tearing analysis: {n_loop_blocks} block(s) with cycles; "
                f"{total_tear}/{total_n} variables flagged as tear "
                f"({100.0 * total_tear / total_n:.1f}%); biggest "
                f"post-tear residual block: {biggest_after} variables"
            )

    # --- BLT (Block Lower Triangular) decomposition ----------------------------------

    @staticmethod
    def _compute_blt_decomposition(rows, cols, n):
        """Structural BLT decomposition of a square sparse pattern.

        Given the COO triplet `(rows, cols)` of a `n x n` (eqs x vars) Jacobian
        pattern, find a permutation that puts the system in block lower
        triangular form: blocks on the diagonal correspond to strongly-
        connected components (SCCs) of the var-dependency graph induced by a
        maximum bipartite matching.

        Returns either None (rectangular or structurally singular) or a dict:

          * 'n_blocks'   -> int
          * 'block_vars' -> list[np.ndarray] of var indices per block, in
                            topological *solve* order (ascending label).
          * 'block_eqs'  -> list[np.ndarray] of eq indices per block, in the
                            same order as 'block_vars'.  Note: every block
                            has `len(block_eqs[b]) == len(block_vars[b])`
                            because the matching is perfect.
          * 'match'      -> np.ndarray shape (n,): match[i] = j means equation
                            i is matched to variable j.
          * 'labels'     -> np.ndarray shape (n,): SCC label per variable.
                            Lower label = earlier solve order.

        Algorithm: Hopcroft-Karp matching (scipy) -> directed var graph ->
        Tarjan SCCs (scipy).  Both are O(E + V) up to log factors for sparse
        patterns; on the pipe_tree N=3 benchmark this is ~50 ms total.
        """
        if rows.size == 0:
            return None
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import (
            connected_components,
            maximum_bipartite_matching,
        )

        data = np.ones(rows.size, dtype=np.int8)
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        A.sum_duplicates()

        # match[i] = j (or -1) -> eq i is matched to var j
        match = maximum_bipartite_matching(A, perm_type='column')
        if (match < 0).any():
            # Structurally singular -- no perfect matching exists.  We surface
            # this rather than silently producing wrong block structure;
            # caller is expected to disable BLT and fall back.
            return None

        # Var graph: edge var_j -> var_k iff eq match^{-1}(j) references var k.
        # In CSR over (eqs x vars), row i's nonzero cols are A.indices[indptr[i]:indptr[i+1]].
        # We loop rows once and emit (match[i], k) per nonzero k in row i (k != match[i]).
        indptr = A.indptr
        indices = A.indices
        # Pre-size: at most nnz - n entries (subtract one self-loop per row).
        nnz = indices.size
        edge_src = np.empty(nnz, dtype=np.int64)
        edge_dst = np.empty(nnz, dtype=np.int64)
        cursor = 0
        for i in range(n):
            j = match[i]
            for k in indices[indptr[i]:indptr[i + 1]]:
                if k != j:
                    edge_src[cursor] = j
                    edge_dst[cursor] = k
                    cursor += 1
        edge_src = edge_src[:cursor]
        edge_dst = edge_dst[:cursor]

        if cursor == 0:
            # No cross-variable dependencies: every variable is its own SCC,
            # already in some order.  Skip the SCC scan -- much cheaper to
            # just use the identity labelling.
            labels = np.arange(n, dtype=np.int64)
            n_components = n
        else:
            var_graph = csr_matrix(
                (np.ones(cursor, dtype=np.int8), (edge_src, edge_dst)),
                shape=(n, n),
            )
            n_components, labels = connected_components(
                var_graph, directed=True, connection='strong', return_labels=True
            )
        # scipy gives SCC labels in REVERSE topological order: sinks (no
        # outgoing edges, solvable first) get LOW labels.  We iterate
        # ascending so blocks come out in solve order automatically.

        block_vars_lists = [[] for _ in range(n_components)]
        for j in range(n):
            block_vars_lists[labels[j]].append(j)

        # inv_match[j] = i: var j is matched to eq i.
        inv_match = np.empty(n, dtype=np.int64)
        inv_match[match] = np.arange(n, dtype=np.int64)

        block_vars = [np.asarray(sorted(vs), dtype=np.int64) for vs in block_vars_lists]
        block_eqs = [inv_match[vs] for vs in block_vars]

        return {
            'n_blocks': int(n_components),
            'block_vars': block_vars,
            'block_eqs': block_eqs,
            'match': match,
            'labels': labels,
        }

    def _build_blt_plan(self, blt, dense_cutoff=32):
        """Pre-compute per-block index arrays for fast block-wise Newton solve.

        Splits the Jacobian's nonzeros into:
          * 'diag_*' per block: (row, col) both in this block (the block's own
                                local Jacobian, used to actually solve the sub-system)
          * 'lower_*' per block: row in this block, col in any earlier block
                                 (off-diagonal coupling that gets subtracted from
                                  the local rhs before solving the sub-system)

        Entries with col in a LATER block would violate BLT-by-construction
        (they couldn't exist in a true block lower triangle).  If any are
        found we surface it as a programmer error -- the BLT decomposition
        was inconsistent with the actual nonzero pattern.

        `dense_cutoff` controls the per-block solver selection:
            n == 1                          -> 'scalar' (single divide)
            1 < n <= dense_cutoff           -> 'dense'  (numpy.linalg.solve)
            n  > dense_cutoff               -> 'sparse' (scipy splu)
        """
        n_blocks = blt['n_blocks']
        block_vars = blt['block_vars']
        block_eqs = blt['block_eqs']
        labels = blt['labels']

        # var_block[j] = block label of variable j
        var_block = labels

        # eq_block[i] = block label of equation i (== block of var match^{-1}(i))
        eq_block = np.empty(self.n_v, dtype=np.int64)
        for b in range(n_blocks):
            eq_block[block_eqs[b]] = b

        rows = self._jac_sparse_rows
        cols = self._jac_sparse_cols
        jac_row_block = eq_block[rows]
        jac_col_block = var_block[cols]

        # Sanity check: no upper-block entries allowed by BLT construction.
        upper_mask = jac_col_block > jac_row_block
        if upper_mask.any():
            raise RuntimeError(
                f"BLT structural plan violated: {int(upper_mask.sum())} "
                f"Jacobian entry(s) reference a var in a later block.  "
                f"This means the matching/SCC pass disagrees with the "
                f"sparsity pattern -- please report with a reproducer."
            )

        # Global-symbol -> local-within-block index lookups.
        eq_local = np.zeros(self.n_v, dtype=np.int64)
        var_local = np.zeros(self.n_v, dtype=np.int64)
        for b in range(n_blocks):
            eq_local[block_eqs[b]] = np.arange(block_eqs[b].size, dtype=np.int64)
            var_local[block_vars[b]] = np.arange(block_vars[b].size, dtype=np.int64)

        diag_jac_idx = []
        diag_local_rows = []
        diag_local_cols = []
        lower_jac_idx = []
        lower_local_rows = []
        lower_global_cols = []
        block_solver = []

        for b in range(n_blocks):
            in_block_rows = jac_row_block == b

            diag_mask = in_block_rows & (jac_col_block == b)
            di = np.nonzero(diag_mask)[0]
            diag_jac_idx.append(di)
            diag_local_rows.append(eq_local[rows[di]])
            diag_local_cols.append(var_local[cols[di]])

            lower_mask = in_block_rows & (jac_col_block < b)
            li = np.nonzero(lower_mask)[0]
            lower_jac_idx.append(li)
            lower_local_rows.append(eq_local[rows[li]])
            lower_global_cols.append(cols[li].astype(np.int64, copy=False))

            n_b = block_vars[b].size
            if n_b == 1:
                block_solver.append('scalar')
            elif n_b <= dense_cutoff:
                block_solver.append('dense')
            else:
                block_solver.append('sparse')

        block_n = np.asarray([v.size for v in block_vars], dtype=np.int64)

        # Block-wise solve only beats `splu` on the whole system in two
        # regimes:
        #   * Pure forward substitution (largest block is 1)            -> chain
        #     of scalar divides, ~10x faster than splu.
        #   * Many small-to-medium independent blocks dominate          -> e.g.
        #     after tearing reduces a loop to a few coupled vars while leaving
        #     the rest as 1x1.
        # Otherwise (one dominant SCC or hundreds of 2..32 blocks with Python
        # dispatch overhead per block) splu's single C call wins.  We therefore
        # decide solve strategy here and fall back to the flat sparse path
        # automatically; the BLT structure stays attached on the model so
        # downstream passes (tearing, scaling) can still consult it.
        largest_block = int(block_n.max())
        n_scalar = int((block_n == 1).sum())
        if largest_block == 1:
            solve_mode = 'triangular'      # spsolve_triangular on global matrix
        elif n_scalar / max(1, self.n_v) >= 0.95:
            solve_mode = 'blockwise'       # tiny tail of larger blocks
        else:
            solve_mode = 'monolithic'      # splu on the whole permuted matrix

        # Pre-compute the BLT permutation arrays for `triangular` mode: we
        # need to permute COO indices into the BLT order so the resulting
        # CSR matrix is genuinely lower-triangular.
        var_perm = np.empty(self.n_v, dtype=np.int64)   # old_col -> new_col
        eq_perm = np.empty(self.n_v, dtype=np.int64)    # old_row -> new_row
        cursor = 0
        for b in range(n_blocks):
            for v in block_vars[b]:
                var_perm[v] = cursor
                cursor += 1
        cursor = 0
        for b in range(n_blocks):
            for e in block_eqs[b]:
                eq_perm[e] = cursor
                cursor += 1
        # Inverse perms: new_idx -> old_idx (used to scatter solved delta back)
        var_perm_inv = np.empty(self.n_v, dtype=np.int64)
        var_perm_inv[var_perm] = np.arange(self.n_v, dtype=np.int64)
        eq_perm_inv = np.empty(self.n_v, dtype=np.int64)
        eq_perm_inv[eq_perm] = np.arange(self.n_v, dtype=np.int64)

        # COO indices in BLT order (precomputed; values come at runtime).
        perm_rows = eq_perm[self._jac_sparse_rows]
        perm_cols = var_perm[self._jac_sparse_cols]

        return {
            'n_blocks': n_blocks,
            'block_vars': block_vars,
            'block_eqs': block_eqs,
            'block_n': block_n,
            'largest_block': largest_block,
            'n_scalar': n_scalar,
            'diag_jac_idx': diag_jac_idx,
            'diag_local_rows': diag_local_rows,
            'diag_local_cols': diag_local_cols,
            'lower_jac_idx': lower_jac_idx,
            'lower_local_rows': lower_local_rows,
            'lower_global_cols': lower_global_cols,
            'block_solver': block_solver,
            'solve_mode': solve_mode,
            'var_perm': var_perm,
            'var_perm_inv': var_perm_inv,
            'eq_perm': eq_perm,
            'eq_perm_inv': eq_perm_inv,
            'perm_rows': perm_rows,
            'perm_cols': perm_cols,
        }

    # --- linear-block (multi-var) elimination ----------------------------------------

    @line_profiler.profile
    def remove_linear_block_equations(self, equations, var_symbols, allow_param_coeffs=False):
        """Eliminate equations linear in >=3 surviving Variables.

        Extension of `remove_trivial_equations` (which only handles 2-var
        linear equations).  For an equation
            c0 + c1*v1 + c2*v2 + ... + cK*vK == 0  (K >= 2, all ci constant)
        we pick one `v_p` as pivot and substitute
            v_p = -(c0 + sum_{i != p} ci * vi) / cp
        into every other equation in the system, then drop the original
        equation and `v_p`.

        Pivot heuristic: among the linear eq's Variables, pick the one that
        appears in the FEWEST other equations (linear or nonlinear).  This
        keeps the substitution from blowing up the AST size of equations it
        gets pushed into.

        Multiple candidates in the same pass: pivots are applied
        SEQUENTIALLY, with each new substitution xreplace'd into the
        remaining candidate equations before the next pivot is picked.
        This is critical for correctness when later candidates reference
        earlier pivots -- without the in-loop xreplace, dropping the
        earlier pivot's contribution from those eqs silently breaks the
        constraints they express.

        Safety:
          * Refuses to pivot on a variable whose coefficient `cp` is a
            sympy expression containing free symbols (would be dividing by
            something that could vanish).  Plain Numbers (incl. negatives)
            are fine.
          * Only Variables (not Parameters) are eligible as pivots.

        When `allow_param_coeffs` is True, Parameters are folded into the
        constant coefficient field (so `c1*v1 + k*v2 + ... == 0` with `k` a
        Parameter is recognised as linear).  The existing "numeric pivot
        coefficient only" guard already makes this division-safe; substitutions
        whose RHS references a time-varying (`Input`) parameter are additionally
        skipped so the prev-step mirror stays correct.
        """
        raw_var_set = set(self.raw_var_symbols)
        var_set = set(var_symbols)
        diff_state_set = self._differential_state_symbols()
        current_to_prev = {v.symbol: v.prev_symbol for v in self.raw_vars_references}
        param_set = set(self.raw_param_symbols) if allow_param_coeffs else None
        dynamic_param_set = self._dynamic_param_symbols() if allow_param_coeffs else set()

        # Pass 1: count var usage across ALL eqs (linear + nonlinear) and
        # find candidate indices.  Usage drives pivot selection: prefer
        # eliminating a var that doesn't bloat many other equations.
        var_usage = {}
        candidate_indices = []
        for idx, eq in enumerate(equations):
            for s in eq.free_symbols:
                if s in var_set:
                    var_usage[s] = var_usage.get(s, 0) + 1
            res = Model._classify_linear(eq, param_set)
            if res is None:
                continue
            c0, coeffs = res
            n_surviving = sum(
                1 for s, c in coeffs.items()
                if c != 0 and s in raw_var_set and s in var_set
            )
            if n_surviving >= 3:
                candidate_indices.append(idx)

        if not candidate_indices:
            print("No multi-var linear equations found")
            return equations, var_symbols, {}

        print(f"Linear-block: {len(candidate_indices)} candidate equation(s) "
              f"(>=3 surviving vars)")

        # Pass 2: sequential pivot picking.  Maintain a MUTABLE per-candidate
        # equation form so each new pivot's substitution gets folded into
        # every remaining candidate before the next pivot decision.  This
        # makes the algorithm equivalent to one-pivot-per-outer-pass, but
        # without the lambdify-grade cost of re-running the whole instantiate
        # outer loop per pivot.
        eqs_state = {idx: equations[idx] for idx in candidate_indices}
        substitutions = {}
        drop_indices = set()
        removed_vars = set()

        for idx in candidate_indices:
            eq_now = eqs_state[idx]
            res = Model._classify_linear(eq_now, param_set)
            if res is None:
                # Earlier pivot substitution turned this candidate nonlinear
                # (rare but possible if RHS contained a product).  Leave it
                # in the system; the next outer pass will re-classify.
                continue
            c0, coeffs = res
            live = {
                s: c for s, c in coeffs.items()
                if c != 0 and s in raw_var_set and s in var_set
                and s not in removed_vars
            }
            if len(live) < 2:
                continue

            # Differential states must survive as pivot targets: substituting
            # one away breaks the stepper's prev-slot advancement (see
            # `_differential_state_symbols`).
            pivot_pool = {s: c for s, c in live.items() if s not in diff_state_set}
            if not pivot_pool:
                continue

            pivot = min(
                pivot_pool.keys(),
                key=lambda s: (var_usage.get(s, 0), s.name),
            )
            cp = live[pivot]
            try:
                if cp.free_symbols:
                    continue
            except AttributeError:
                pass

            # v_p = -(c0 + sum_{j != p} cj * vj) / cp.  Build from the
            # CURRENT (mutated) form of the eq, NOT the original coeffs --
            # earlier xreplaces may have rewritten the symbols.
            rest = -c0
            for s, c in live.items():
                if s is pivot:
                    continue
                rest = rest - c * s
            sol = rest / cp

            # Guard (prev-step mirror): skip substitutions whose RHS references
            # a time-varying parameter -- it can't be carried to the previous
            # time level by `xreplace(current_to_prev)`.
            if allow_param_coeffs and (sol.free_symbols & dynamic_param_set):
                continue

            substitutions[pivot] = sol
            removed_vars.add(pivot)
            drop_indices.add(idx)

            pivot_prev = current_to_prev.get(pivot)
            if pivot_prev is not None:
                sol_prev = sol.xreplace(current_to_prev) if sol.free_symbols else sol
                substitutions[pivot_prev] = sol_prev
                removed_vars.add(pivot_prev)

            # Fold this pivot's substitution into ALL OTHER remaining
            # candidate eqs so the next iteration's `_classify_linear` sees
            # the up-to-date form.  Only touch eqs that actually mention
            # the pivot -- xreplace is O(tree) and pipe trees have many
            # candidates that don't share vars.
            pivot_sub = {pivot: sol}
            for other in candidate_indices:
                if other in drop_indices or other == idx:
                    continue
                eq_other = eqs_state[other]
                if pivot in eq_other.free_symbols:
                    eqs_state[other] = eq_other.xreplace(pivot_sub)

        if not substitutions:
            print("No linear-block pivots picked")
            return equations, var_symbols, {}

        print(f"Linear-block: {len(drop_indices)} equation(s) dropped, "
              f"{len(substitutions)} substitution(s)")
        Model._close_substitutions(substitutions)
        sub_keys = set(substitutions.keys())

        new_eqs = []
        for idx, eq in enumerate(equations):
            if idx in drop_indices:
                continue
            if eq.free_symbols.isdisjoint(sub_keys):
                new_eqs.append(eq)
            else:
                new_eqs.append(eq.xreplace(substitutions))

        removed_set = set(substitutions.keys())
        new_var_symbols = [s for s in var_symbols if s not in removed_set]
        return new_eqs, new_var_symbols, substitutions

    # --- compilation ------------------------------------------------------------------

    @line_profiler.profile
    def instantiate(self, cse=True, aditional_modules=None, max_remove_trival_passes=1,
                    lambda_cache_dir=None, max_remove_duplicate_passes=5,
                    enable_blt=True, max_remove_linear_block_passes=3,
                    enable_var_scaling=True, eliminate_param_linear=None,
                    numba=False):
        # Step B's `declare_equations()` template cache is scoped to this single
        # `instantiate()` call to avoid cross-call contamination (e.g. Air's
        # `Air_rho_ph` Function nodes leaking into a subsequent Hydrogen
        # instantiation).  Set a fresh dict on entry and reset on exit
        # regardless of whether `instantiate()` returns or raises.
        _eq_cache_token = _eq_cache_var.set({})
        try:
            # Patch `Min`/`Max` canonicalisation to skip its costly
            # `factor_terms` domination retry for the whole call.  All Min/Max
            # nodes (built by `declare_equations`, rewritten by the reduction
            # passes, regrouped by the lambdify classifier) are then simplified
            # consistently, so structural template invariants hold while the
            # per-`xreplace` re-canonicalisation cost collapses.
            with _cheap_minmax_simplification():
                return self._instantiate_impl(
                    cse=cse,
                    aditional_modules=aditional_modules,
                    max_remove_trival_passes=max_remove_trival_passes,
                    lambda_cache_dir=lambda_cache_dir,
                    max_remove_duplicate_passes=max_remove_duplicate_passes,
                    enable_blt=enable_blt,
                    max_remove_linear_block_passes=max_remove_linear_block_passes,
                    enable_var_scaling=enable_var_scaling,
                    eliminate_param_linear=eliminate_param_linear,
                    numba=numba,
                )
        finally:
            _eq_cache_var.reset(_eq_cache_token)

    @line_profiler.profile
    def _instantiate_impl(self, cse=True, aditional_modules=None,
                          max_remove_trival_passes=1, lambda_cache_dir=None,
                          max_remove_duplicate_passes=5, enable_blt=True,
                          max_remove_linear_block_passes=3,
                          enable_var_scaling=True, eliminate_param_linear=None,
                          numba=False):
        if aditional_modules is None:
            aditional_modules = []
        # Parameter-coefficient linear elimination (treat Parameters as constant
        # coefficients in the trivial / linear-block reducers).  Opt-in: defaults
        # to the `HYDROGEN_PARAM_LINEAR_ELIM` env var so it can be A/B-tested
        # without editing call sites.  See `remove_trivial_equations`.
        if eliminate_param_linear is None:
            eliminate_param_linear = os.environ.get(
                "HYDROGEN_PARAM_LINEAR_ELIM", "").strip().lower() in (
                "1", "true", "yes", "on")
        self._eliminate_param_linear = eliminate_param_linear
        if eliminate_param_linear:
            print("Parameter-coefficient linear elimination: ENABLED")
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

        _t_instantiate_start = time.time()
        # Ordered (phase_name, seconds) records, rendered as a timing table at
        # the end of instantiation.
        _timings = []
        start_time = time.time()
        _t0 = time.time()
        vars, prev_vars, params, vars_map, params_map = self.assign_symbols(top_level=True)
        _t_assign = time.time() - _t0
        _timings.append(("assign_symbols", _t_assign, []))
        _t0 = time.time()
        # `collect_equations` consults `_eq_cache_var` (set by `instantiate`)
        # so that only the first 1-2 instances of each `Model` subclass build
        # their CoolProp-laden symbolic equations from scratch; subsequent
        # siblings replay via a path-based symbol remap + a single `xreplace`
        # per eq.
        self.all_raw_equations = self.collect_equations()
        _t_collect = time.time() - _t0
        _timings.append(("collect_equations", _t_collect, []))
        # Connections are now resolved (`connect()` ran inside the
        # `declare_equations` walk above), so port wiring state is final:
        # warn about any port that asked to be connected but wasn't.
        self._warn_unconnected_required_ports()
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
        #
        # The unionfind is SIGNED: each non-root carries a sign (+/-1)
        # relative to its parent so we can collapse both `a == b` wires
        # (`sign=+1`, the legacy form) AND `a + b == 0` wires (`sign=-1`,
        # emitted by `Model.connect()` whenever two ports of the same flow
        # orientation are wired -- e.g. junction-`in` to pipe-`in`).  The
        # all-`+1` workload reduces structurally to plain UF (one extra
        # integer multiply per `find()`), so this is a strict generalisation
        # of the old code.
        _t0_collect_conn = time.time()
        connections = self.collect_connections()
        _timings.append(("collect_connections", time.time() - _t0_collect_conn, []))
        if connections:
            uf_start = time.time()
            uf_parent = {}
            uf_sign = {}    # sign of this node relative to its parent (+/-1)
            inconsistent_loops = []

            def find(s):
                """Return (root, sign_of_s_relative_to_root).  Path-compress."""
                parent = uf_parent.get(s, s)
                if parent is s:
                    return s, +1
                root, sign_parent = find(parent)
                sign_self = sign_parent * uf_sign.get(s, +1)
                # Path-compression: rewire `s` directly under the root and
                # collapse the accumulated sign onto `uf_sign[s]`.
                uf_parent[s] = root
                uf_sign[s] = sign_self
                return root, sign_self

            # Symbols backing a `DifferentialVariable` state MUST survive
            # connection elimination.  The time integrator locates states by
            # walking the component tree for `DifferentialVariable` instances
            # (`_get_diff_state_indices` / `_get_diff_var_index_pairs`); if a
            # state's symbol is absorbed into a plain (algebraic) representative
            # -- e.g. a fluid cell's `T_wall_i` alias swallowing a wall layer's
            # differential `T_a` purely because its name sorts first -- the
            # state silently drops out of the integrated set and the TR-BDF2
            # stage handling no longer advances it, corrupting its dynamics.
            # Rank state symbols ahead of algebraic ones so they are kept.
            diff_state_symbols = set()

            # state-symbol -> (der-symbol, der-prev-symbol) for capacitive-node
            # merging (see the connection_subs loop below).
            diff_state_der = {}

            def _collect_diff_states(node):
                if isinstance(node, DifferentialVariable) and node.symbol is not None:
                    diff_state_symbols.add(node.symbol)
                    dv = node.der_variable
                    if dv is not None and dv.symbol is not None:
                        diff_state_der[node.symbol] = (dv.symbol, dv.prev_symbol)
                if isinstance(node, Model):
                    for child in node.components.values():
                        _collect_diff_states(child)

            _collect_diff_states(self)

            def _rep_key(s):
                # Smaller key wins as representative; differential-state symbols
                # rank first (0) so they absorb algebraic partners, never the
                # other way round.  Name breaks ties deterministically.
                return (0 if s in diff_state_symbols else 1, s.name)

            def union(a, b, sign):
                """Add the constraint `a == sign * b`."""
                ra, sa = find(a)
                rb, sb = find(b)
                if ra is rb:
                    # Cycle: must be consistent.  `sa*ra == sign * sb*rb`
                    # plus `ra is rb` requires `sa == sign * sb`.  An
                    # inconsistent cycle forces `ra == -ra`, i.e. the whole
                    # equivalence class collapses to zero -- almost certainly
                    # a wiring mistake, so we surface it as a diagnostic.
                    if sa != sign * sb:
                        inconsistent_loops.append((a, b))
                    return
                # Want:  sa * ra == sign * sb * rb
                #   ->   ra == (sign * sb / sa) * rb   (sa, sb in {+1, -1})
                rel_ab = sign * sb * sa  # +/-1
                # Deterministic: lower-ranked symbol wins as representative
                # (differential states first, then by name).
                if _rep_key(rb) < _rep_key(ra):
                    # Flip so the preferred root absorbs the other.
                    uf_parent[ra] = rb
                    uf_sign[ra] = rel_ab
                else:
                    uf_parent[rb] = ra
                    uf_sign[rb] = rel_ab

            raw_var_set = set(self.raw_var_symbols)
            deferred_eqs = []
            for var_a, var_b, sign in connections:
                sa, sb = var_a.symbol, var_b.symbol
                if sa is None or sb is None:
                    continue
                if sa not in raw_var_set or sb not in raw_var_set:
                    # One side is a Parameter / t -- can't union with a non-Variable;
                    # defer to the symbolic trivial reducer.
                    deferred_eqs.append(sa - sign * sb)
                    continue
                union(sa, sb, sign)

            if inconsistent_loops:
                preview = ", ".join(
                    f"{a.name}<->{b.name}" for a, b in inconsistent_loops[:5]
                )
                raise ValueError(
                    f"Inconsistent signed-connection cycle(s) detected "
                    f"({len(inconsistent_loops)} pair(s), e.g. {preview}). "
                    f"This usually means two ports of the same flow orientation "
                    f"were wired in a loop that forces a variable to zero -- "
                    f"check the topology before instantiate()."
                )

            current_to_prev = {var.symbol: var.prev_symbol for var in self.raw_vars_references}
            connection_subs = {}
            for s in list(uf_parent.keys()):
                rep, sign_to_rep = find(s)
                if rep is s:
                    continue
                # `s -> sign_to_rep * rep`.  sympy's xreplace handles the
                # `-rep` case via a `Mul(-1, rep)` rewrite, so downstream
                # consumers (trivial reducer, dedup pass, lambdify) need no
                # changes -- they just see a regular sympy expression.
                if sign_to_rep == +1:
                    connection_subs[s] = rep
                else:
                    connection_subs[s] = -rep
                ps = current_to_prev.get(s)
                pr = current_to_prev.get(rep)
                if ps is not None and pr is not None:
                    connection_subs[ps] = pr if sign_to_rep == +1 else -pr

                # Capacitive-node merge: when the eliminated symbol `s` and its
                # representative `rep` are BOTH differential states (two heat-
                # capacity surface nodes wired together, e.g. adjacent dynamic
                # wall layers sharing an interface), forcing `s == sign*rep`
                # also forces `der_s == sign*der_rep`.  Aliasing the derivative
                # companions makes both nodes' `C*der = ...` ODEs constrain the
                # SAME derivative, so the capacities add into one combined node
                # ((C_s + C_rep)*der = RHS_s + RHS_rep) -- the correct index
                # reduction.  Without this the eliminated node's derivative is
                # orphaned and the Jacobian is structurally singular.
                if s in diff_state_der and rep in diff_state_der:
                    der_s, der_s_prev = diff_state_der[s]
                    der_rep, der_rep_prev = diff_state_der[rep]
                    connection_subs[der_s] = der_rep if sign_to_rep == +1 else -der_rep
                    if der_s_prev is not None and der_rep_prev is not None:
                        connection_subs[der_s_prev] = (
                            der_rep_prev if sign_to_rep == +1 else -der_rep_prev)

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
            n_signed = sum(1 for _, _, s in connections if s == -1)
            _t_uf = time.time() - uf_start
            print(f"add_connection short-circuited {len(connection_subs)} symbols "
                  f"from {len(connections)} pairs "
                  f"({n_signed} sign-flipped, {len(deferred_eqs)} deferred) "
                  f"in {_t_uf:.2f} s")
            _timings.append(("add_connection short-circuit", _t_uf, []))

        if max_remove_trival_passes > 0:
            print("Removing trivial equations")
            curent_size = len(self.improved_vars)
            print(f"Original variables: {curent_size}")
            start_time = time.time()
            _trivial_pass_times = []

            for i in range(max_remove_trival_passes):
                print(f"Removing trivial pass {i+1}")
                _t_pass = time.time()
                self.improved_equations, self.all_improved_symbols, pass_subs = self.remove_trivial_equations(self.improved_equations, self.all_improved_symbols, allow_param_coeffs=eliminate_param_linear)
                _trivial_pass_times.append(time.time() - _t_pass)
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
            _timings.append(("remove trivial equations", stop_time - start_time, _trivial_pass_times))

        # Duplicate-equation reduction.  Runs AFTER trivial reduction so the
        # equation list is in its most-reduced form (which only helps when
        # the trivial reducer rewrote symbols that show up in duplicate
        # signatures; for the headline use case -- StraightPipe face
        # closures -- the relevant equations are nonlinear and survive
        # trivial reduction unchanged).
        #
        # Iterated up to `max_remove_duplicate_passes` times: pass N's
        # substitutions can EXPOSE new duplicates for pass N+1.  Concretely,
        # `TwoPortSegment`'s face-medium closures (rho, T, mu, k) get
        # collapsed in pass 1 -- which in turn rewrites the rho leaves
        # inside neighbouring `m_dot = rho * A * w` closures so that the
        # face-velocity duplicates become visible in pass 2.  Passes are
        # idempotent once no further substitutions fire, so the loop
        # always terminates well below the cap for sane inputs.
        if max_remove_duplicate_passes > 0:
            print("Removing duplicate equations")
            dup_start = time.time()
            before_n = len(self.improved_equations)
            total_subs = 0
            _dup_pass_times = []
            for pass_idx in range(max_remove_duplicate_passes):
                print(f"Duplicate-equation pass {pass_idx + 1}")
                pass_before_n = len(self.improved_equations)
                _t_pass = time.time()
                self.improved_equations, self.all_improved_symbols, dup_subs = (
                    self.remove_duplicate_equations(
                        self.improved_equations, self.all_improved_symbols)
                )
                _dup_pass_times.append(time.time() - _t_pass)
                if not dup_subs:
                    break
                # Stitch the new substitutions into the running improve_subs
                # map exactly the same way the trivial-pass loop does.
                for prev_key in list(self.improve_subs.keys()):
                    self.improve_subs[prev_key] = self.improve_subs[prev_key].xreplace(dup_subs)
                self.improve_subs.update(dup_subs)
                total_subs += len(dup_subs)
                if len(self.improved_equations) == pass_before_n:
                    break
            improved_symbols_set = set(self.all_improved_symbols)
            self.improved_vars = [s for s in self.raw_var_symbols if s in improved_symbols_set]
            after_n = len(self.improved_equations)
            _t_dup = time.time() - dup_start
            print(
                f"Duplicate-equation reduction: {before_n - after_n} equation(s) / "
                f"{total_subs} substitution(s) over {pass_idx + 1} pass(es) in "
                f"{_t_dup:.2f} s"
            )
            _timings.append(("remove duplicate equations", _t_dup, _dup_pass_times))

        # Multi-variable linear-block elimination.  Extends the trivial
        # reducer (2-var) to any-N-var linear equations: mass balances at
        # splitters / junctions, geometric area relations, anything where
        # the dedup pass left a linear sub-block of >= 3 unknowns.  Iterated
        # because each pass's substitutions can expose new linear-3+
        # candidates (a multi-var eq with one symbolic subexpression
        # collapses to fewer vars after substitution).
        if max_remove_linear_block_passes > 0:
            print("Removing multi-var linear equations")
            lb_start = time.time()
            before_n = len(self.improved_equations)
            total_lb_subs = 0
            _lb_pass_times = []
            for pass_idx in range(max_remove_linear_block_passes):
                print(f"Linear-block pass {pass_idx + 1}")
                pass_before_n = len(self.improved_equations)
                _t_pass = time.time()
                self.improved_equations, self.all_improved_symbols, lb_subs = (
                    self.remove_linear_block_equations(
                        self.improved_equations, self.all_improved_symbols,
                        allow_param_coeffs=eliminate_param_linear)
                )
                _lb_pass_times.append(time.time() - _t_pass)
                if not lb_subs:
                    break
                for prev_key in list(self.improve_subs.keys()):
                    self.improve_subs[prev_key] = self.improve_subs[prev_key].xreplace(lb_subs)
                self.improve_subs.update(lb_subs)
                total_lb_subs += len(lb_subs)
                if len(self.improved_equations) == pass_before_n:
                    break
            improved_symbols_set = set(self.all_improved_symbols)
            self.improved_vars = [s for s in self.raw_var_symbols
                                  if s in improved_symbols_set]
            after_n = len(self.improved_equations)
            _t_lb = time.time() - lb_start
            print(
                f"Linear-block reduction: {before_n - after_n} equation(s) / "
                f"{total_lb_subs} substitution(s) over {pass_idx + 1} pass(es) "
                f"in {_t_lb:.2f} s"
            )
            _timings.append(("remove linear-block equations", _t_lb, _lb_pass_times))

        self.n_v = len(self.improved_vars)
        self.n_p = len(self.raw_param_symbols)
        self.n_t = len(self.t_symbols)
        print(f"Remaining variables and equations: {self.n_v}")
        self.values = np.zeros(2 * self.n_v + self.n_p + self.n_t)
        # Default the global integration scheme to Crank-Nicolson so every solve
        # (including `initialise`'s t=0 consistency solve) sees a well-posed
        # closure.  The TR-BDF2 stepper overrides these per stage and restores
        # them afterwards.
        self.set_scheme_coeffs(*_CN_COEFFS)
        self.delta_values = np.zeros(self.n_v)
        # Bind each surviving Parameter to its slot in `self.values` so that
        # `Parameter.set_value()` can write a single index directly.  Slot
        # order == position in `raw_param_references`, the same invariant
        # `set_param_values`/`initialise`/`_refresh_inputs` rely on.
        _param_base = 2 * self.n_v
        for _i, _p in enumerate(self.raw_param_references):
            _p.bind_value_slot(self.values, _param_base + _i)
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
        #
        # For a segmented model the N interior cells emit N structurally
        # IDENTICAL equations that all collapse to one template.  The naive way
        # to discover that -- build the placeholder-substituted `template`
        # expression (a full sympy `xreplace`, which rebuilds the whole tree)
        # for EVERY equation and group by expression equality -- costs one
        # sympy tree rebuild + one sympy hash per equation, i.e. O(N * expr)
        # sympy work that dominates this phase at large N even though there are
        # only a handful of distinct templates.
        #
        # Instead we compute a cheap *structural key* in a single preorder walk
        # (a flat token stream with each symbol replaced by its first-occurrence
        # index) and use it purely as a fast cache for the expensive
        # `xreplace`: the FIRST equation of each key materialises the real
        # `template`, and subsequent equations with the same key reuse that same
        # template object (an O(1) dict hit -- its sympy hash is memoised on the
        # object) instead of rebuilding + re-hashing the tree.
        #
        # The grouping itself stays keyed by the template EXPRESSION, exactly as
        # the legacy path, so behaviour is identical.  This matters because the
        # structural key is intentionally allowed to be *finer* than template
        # equality: commutative reorderings (e.g. `a*b + a` vs `c*d + d`) yield
        # different token streams but the SAME canonical sympy template, and
        # grouping by the template expression re-merges them.  Keying the final
        # groups by the token stream instead would wrongly split (and, on
        # re-key, silently drop) those instances.
        key_to_template = {}   # structural key -> representative template expr
        template_to_instances = {}
        n_constant_dropped = 0
        _Symbol = sp.Symbol
        for i, eq in enumerate(improved_equations_list):
            # Single preorder walk (matches `sp.preorder_traversal` order):
            # emit a flat token stream capturing the tree structure and collect
            # the equation's symbols in first-occurrence order.
            sym_order = []
            sym_to_idx = {}
            tokens = []
            tok_append = tokens.append
            stack = [eq]
            while stack:
                node = stack.pop()
                if isinstance(node, _Symbol):
                    idx = sym_to_idx.get(node)
                    if idx is None:
                        idx = len(sym_order)
                        sym_to_idx[node] = idx
                        sym_order.append(node)
                    tok_append(idx)          # placeholder slot for this symbol
                    continue
                if node.is_Number:
                    tok_append(node)         # exact numeric literal (kept as-is)
                    continue
                args = node.args
                tok_append((type(node), len(args)))
                # Push children reversed so they pop in natural preorder order.
                for a in reversed(args):
                    stack.append(a)
            if not sym_order:
                # Constant equation -- drop a tautological `0` row; keep a
                # non-zero constant as an infeasibility row the solver surfaces.
                if eq == 0:
                    n_constant_dropped += 1
                    continue
            key = tuple(tokens)
            template = key_to_template.get(key)
            if template is None:
                # First equation of this structural key: build the placeholder
                # template exactly as the legacy path did (same first-occurrence
                # placeholder order) -- byte-identical templates, hence
                # unchanged lambda-cache keys.
                template = (eq.xreplace({s: _placeholder(k)
                                         for k, s in enumerate(sym_order)})
                            if sym_order else eq)
                key_to_template[key] = template
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
            args_mat = sp.Matrix(placeholders)
            label = f"template_{tid:02d}"
            cached_func = None
            key = None
            if self._lambda_cache_dir is not None:
                # Key on the TEMPLATE (residual) alone.  The Jacobian rows are a
                # deterministic function of it -- `sp.diff` wrt the fixed
                # placeholders -- so differentiating the whole block and
                # serialising it (the old key input) just to LOOK UP the cache
                # was pure overhead that ran on every instantiate, hit or miss.
                # For heavy templates (e.g. the `compressible` momentum/energy
                # rows) that `sp.diff` + pickle dominated the "lambdify" phase
                # even when every template was a cache hit.  Keying on the
                # template defers `sp.diff` to the miss path below.
                key = lambda_cache_key(args_mat, template,
                                       self._lambda_modules_sig, cse)
                namespace = self._build_lambdify_namespace(all_modules)
                cached_func = load_lambdified_source(self._lambda_cache_dir, key, namespace)
                if cached_func is not None:
                    print(f"  [lambda-cache HIT  for {label}: {key[:8]}]")
            # The differentiated block is only needed to LAMBDIFY (cache miss)
            # or to scan for numba twins; skip the expensive `sp.diff` on a
            # plain cache hit.
            exprs_list = None
            if cached_func is None or numba:
                exprs_list = [template] + [sp.diff(template, ph)
                                           for ph in placeholders]
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
        use_parallel_lambdify = (
            cache_misses
            and worker_procs > 1
            and self._lambda_cache_dir is not None
            and _mp_fork_available()
        )
        if use_parallel_lambdify:
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
            if worker_procs > 1 and not _mp_fork_available():
                print("  [parallel lambdify skipped: 'fork' is unavailable on "
                      "this platform; using sequential fallback]")
            # Sequential fallback (also used when worker_procs <= 1 or the user
            # set HYDROGEN_PARALLEL_LAMBDIFY=0).
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

        # Structural pattern caching for the per-Newton-iter sparse solve.
        # `precompute_csc_pattern` returns the COO->CSC permutation +
        # canonical `(indices, indptr)` once; the inner loop then builds
        # the CSC matrix without re-sorting / dedup-scanning every iter.
        # Saves ~30-50us per Newton iter on the run_system workload --
        # small in absolute terms here (CoolProp dominates) but free, and
        # the relative win grows once the per-iter symbolic work drops.
        if self._jac_nnz > 0:
            from .numerics import precompute_csc_pattern
            n = self.n_v
            (
                self._jac_csc_perm,
                self._jac_csc_indices,
                self._jac_csc_indptr,
            ) = precompute_csc_pattern(
                self._jac_sparse_rows, self._jac_sparse_cols, (n, n)
            )
            print(
                f"Sparse Jacobian CSC pattern cached: nnz={self._jac_nnz}, "
                f"shape=({n}, {n})"
            )
        else:
            self._jac_csc_perm = None
            self._jac_csc_indices = None
            self._jac_csc_indptr = None

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
        _timings.append(("per-template lambdify", elapsed, []))

        # ---- BLT decomposition + block-wise solve plan (Pass: BLT) ---------
        # Splits the post-dedup Jacobian into strongly-connected blocks and
        # routes the per-Newton-iteration linear solve through a block-wise
        # forward substitution.  For feed-forward topologies (pipe trees,
        # series pipe chains) the SCC pass produces hundreds of 1x1 blocks
        # and the entire Newton step becomes O(nnz) flops instead of one
        # whole-system splu factorisation; loop topologies collapse the
        # loop into a single dense/sparse block while leaving the rest as
        # 1x1.  Enabled by default; pass `enable_blt=False` to fall back
        # to the flat `fast_sparse_solve` path used pre-BLT.
        self._blt_plan = None
        if enable_blt and self._jac_nnz > 0 and self.n_v == n_eq:
            t_blt = time.time()
            blt = self._compute_blt_decomposition(
                self._jac_sparse_rows, self._jac_sparse_cols, self.n_v
            )
            if blt is not None:
                self._blt_plan = self._build_blt_plan(blt)
                bn = self._blt_plan['block_n']
                n_blocks = self._blt_plan['n_blocks']
                n_scalar = int((bn == 1).sum())
                n_small = int(((bn > 1) & (bn <= 32)).sum())
                n_large = int((bn > 32).sum())
                largest = int(bn.max())
                mode = self._blt_plan['solve_mode']
                _t_blt = time.time() - t_blt
                print(
                    f"BLT decomposition: {n_blocks} block(s) over {self.n_v} "
                    f"variables (1x1: {n_scalar}, 2..32: {n_small}, >32: "
                    f"{n_large}; largest block {largest}; solve_mode={mode}) "
                    f"in {_t_blt:.2f} s"
                )
                _timings.append(("BLT decomposition", _t_blt, []))
                if largest > 1:
                    t_tear = time.time()
                    self._compute_tearing()
                    _t_tear = time.time() - t_tear
                    print(f"  tearing analysis in {_t_tear:.2f} s")
                    _timings.append(("tearing analysis", _t_tear, []))
            else:
                print("BLT skipped: no perfect matching (structurally singular)")
        elif enable_blt:
            print(f"BLT skipped: n_v={self.n_v}, n_eq={n_eq}, nnz={self._jac_nnz}")

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

        # ---- Optional numba JIT of the vectorised templates -----------------
        # Re-lambdify each vectorised template with a numba-friendly printer
        # (nested minimum/maximum/where instead of reduce/select) and the
        # medium callbacks' `@njit` twins (attached by `TabulatedMedium` as
        # `_hydrogen_numba`), then compile the whole template in nopython
        # mode.  Templates whose medium calls have no njit twin, or that fail
        # numba typing, silently keep the numpy path.
        if numba:
            _t0_numba = time.time()
            if n_vec:
                self._numba_jit_templates(
                    prep_per_template, aditional_modules, cse)
            else:
                print("  [numba: no vectorised templates to jit]")
            _timings.append(("numba JIT", time.time() - _t0_numba, []))

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
                    if isinstance(out_list, (list, tuple)):
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

        # Membership must be tested against a SET: `improved_vars` is a list, so
        # the previous `var.symbol in self.improved_vars` was an O(n) scan run
        # once per raw var -> O(n^2) overall, which dominated "other" at large
        # segment counts.
        _t0_active = time.time()
        _improved_var_set = set(self.improved_vars)
        self.active_vars_references = [var for var in self.raw_vars_references
                                       if var.symbol in _improved_var_set]
        _timings.append(("active vars references", time.time() - _t0_active, []))

        # Per-variable scale vector for the Newton convergence metric.
        # Per-Variable `scale` attribute wins; otherwise we fall back to
        # max(|initial_value|, 1.0) -- a Modelica-style "nominal" default
        # that auto-rescales mixed-magnitude systems (pressure ~1e5,
        # mass flow ~1e-3) so they share a comparable convergence
        # threshold.  When `enable_var_scaling=False` we leave the metric
        # un-scaled (==1.0) so the legacy unscaled L2 norm is recovered.
        if enable_var_scaling:
            scales = []
            for v in self.active_vars_references:
                s = getattr(v, 'scale', None)
                if s is None:
                    s = max(abs(float(v.value)), 1.0)
                scales.append(float(s) if float(s) > 0 else 1.0)
            self.var_scales = np.asarray(scales, dtype=float)
        else:
            self.var_scales = np.ones(self.n_v, dtype=float)
        # Pre-compute the inverse and a scratch buffer so the inner Newton
        # loop's scaled-norm metric is a single fused mul (no array alloc,
        # no division) -- otherwise the 338-var/20-step bench shows a
        # ~25ms hot-loop regression from `delta / scales` allocating fresh
        # output every iter.
        self._var_inv_scales = 1.0 / self.var_scales
        self._scaled_delta_buf = np.empty(self.n_v, dtype=float)
        # Only flag scaling as ACTIVE if at least 2 orders of magnitude
        # separate the smallest and largest scale -- below that the
        # un-scaled L2 norm is fine and the per-iter mul is pure overhead.
        if self.n_v and enable_var_scaling:
            log_s = np.log10(self.var_scales)
            span = float(log_s.max() - log_s.min())
            self._var_scaling_active = span >= 2.0
            print(
                f"Var scaling: span log10 = [{log_s.min():.1f}, "
                f"{log_s.max():.1f}] ({self.n_v} vars; "
                f"{int((self.var_scales != 1.0).sum())} non-unit scales; "
                f"{'active' if self._var_scaling_active else 'inactive (span<2)'})"
            )
        else:
            self._var_scaling_active = False

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

        # --- time-dependent Input signals --------------------------------------
        # Each `Input` contributed two ordinary Parameters (`cur`/`prev`) into
        # the param block.  Cache (input, slot_cur, slot_prev) so that
        # `_refresh_inputs` can rewrite those two slots in place every time the
        # integrator moves the time level -- no full `set_param_values` needed.
        # Slot index == position in `raw_param_references`, which is the exact
        # order the param block is laid out (the same invariant `initialise`
        # relies on for `set_param_values`).
        self._input_refs = []
        inputs = self._collect_inputs()
        if inputs:
            param_index = {id(p): i for i, p in enumerate(self.raw_param_references)}
            for inp in inputs:
                i_cur = param_index.get(id(inp.components['cur']))
                i_prev = param_index.get(id(inp.components['prev']))
                if i_cur is not None and i_prev is not None:
                    self._input_refs.append((inp, i_cur, i_prev))

        # --- explicit time events ----------------------------------------------
        # Aggregate every component's `declare_events()` (signal jumps / kinks)
        # so `iter_run` can clip the step size to land just before/after each
        # instant instead of integrating across the discontinuity.
        self._event_times = self._collect_events()

        _t0_gc = time.time()
        gc.collect()
        _timings.append(("gc.collect", time.time() - _t0_gc, []))

        _t_total = time.time() - _t_instantiate_start
        _measured = sum(s for _, s, _ in _timings)
        _timings.append(("other", max(_t_total - _measured, 0.0), []))

        def _pct(secs):
            return 100.0 * secs / _t_total if _t_total > 0 else 0.0

        # Build display rows first so the name column can be sized to fit the
        # indented per-pass sub-rows too.
        _rows = []  # (label, seconds, pct)
        for name, secs, runs in _timings:
            _rows.append((name, secs, _pct(secs)))
            if len(runs) > 1:
                for _i, _rt in enumerate(runs):
                    _rows.append((f"  pass {_i + 1}", _rt, _pct(_rt)))

        _name_w = max([len(lbl) for lbl, _, _ in _rows] + [len("Total")])
        _sep = "+" + "-" * (_name_w + 2) + "+----------+--------+"
        print("\nInstantiation timing breakdown")
        print(_sep)
        print(f"| {'Phase'.ljust(_name_w)} | {'Time [s]':>8} | {'%':>5}  |")
        print(_sep)
        for lbl, secs, pct in _rows:
            print(f"| {lbl.ljust(_name_w)} | {secs:8.2f} | {pct:5.1f}  |")
        print(_sep)
        print(f"| {'Total'.ljust(_name_w)} | {_t_total:8.2f} | {100.0:5.1f}  |")
        print(_sep)

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

    def _numba_jit_templates(self, prep_per_template, aditional_modules, cse):
        """Compile the vectorised equation templates in numba nopython mode.

        For each vectorised template, re-lambdify with `NumbaFriendlyPrinter`
        (nested ``minimum``/``maximum``/``where`` instead of ``reduce`` /
        ``select``) against the medium callbacks' ``@njit`` twins, wrap in
        ``numba.njit`` and compile eagerly on dummy arrays.  Successful
        templates replace their numpy lambda in `self._template_lambdas`;
        anything else (no njit twin available, numba typing failure) keeps
        the numpy path.  Purely a runtime optimisation -- values are
        identical (the printer rewrites are exact and the twins mirror the
        numpy table evaluation).
        """
        try:
            import numba as _nb
        except ImportError:
            print("  [numba requested but not installed; keeping numpy path]")
            return
        from .numerics import NumbaFriendlyPrinter, lambdify_compat

        t0 = time.time()
        # Mirror the printer settings sympy's lambdify uses internally;
        # `allow_unknown_functions` makes the medium callbacks (`Water_rho_ph`
        # etc.) print as plain calls resolved from the injected modules.
        printer = NumbaFriendlyPrinter({
            "fully_qualified_modules": False, "inline": True,
            "allow_unknown_functions": True,
        })
        medium_fns = {}
        for m in aditional_modules:
            if isinstance(m, dict):
                medium_fns.update({k: v for k, v in m.items() if callable(v)})

        n_ok = 0
        skipped = []
        for tid, use_vec in enumerate(self._vec_template_use):
            if not use_vec:
                continue
            prep = prep_per_template[tid]
            # Scan the WHOLE lambdified block (residual + Jacobian rows):
            # differentiating a property call introduces new function names
            # (`Water_T_ph` -> `Water_dT_ph_dp`) that only appear in the
            # derivative rows.
            needed = {type(f).__name__
                      for expr in prep["block"]
                      for f in expr.atoms(sp.Function)}
            needed_medium = needed & set(medium_fns)
            twins = {}
            missing = None
            for name in needed_medium:
                twin = getattr(medium_fns[name], "_hydrogen_numba", None)
                if twin is None:
                    missing = name
                    break
                twins[name] = twin
            if missing is not None:
                skipped.append((prep["label"], f"no njit twin: {missing}"))
                continue
            try:
                fn = lambdify_compat(
                    prep["args_mat"], tuple(prep["block"]),
                    modules=[twins, "numpy"], cse=cse, docstring_limit=-1,
                    printer=printer,
                )
                jit = _nb.njit(cache=False, nogil=True)(fn)
                dummy = [np.full(2, 1.05) for _ in range(prep["n_ph"])]
                with np.errstate(all="ignore"):
                    jit(*dummy)                       # eager compile
            except Exception as exc:                  # typing/lowering error
                skipped.append(
                    (prep["label"], f"{type(exc).__name__}: {exc}"))
                continue
            self._template_lambdas[tid] = jit
            n_ok += 1
        n_vec = sum(self._vec_template_use)
        print(f"  [numba: {n_ok}/{n_vec} vectorised templates jitted in "
              f"{time.time() - t0:.1f} s"
              + (f"; {len(skipped)} kept numpy path" if skipped else "")
              + "]")
        for label, why in skipped:
            print(f"    - {label}: {str(why)[:160]}")
        self._numba_template_stats = {"jitted": n_ok, "skipped": skipped}

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
        self._refresh_inputs()

    @property
    def time(self):
        return self.t_symbols[0]

    @property
    def t_prev(self):
        return self.t_symbols[1]

    @property
    def dt(self):
        return self.t_symbols[2]

    # --- integration-scheme coefficient symbols (shared by every diff closure) --
    @property
    def sch_p0(self):
        return self.t_symbols[3]

    @property
    def sch_p1(self):
        return self.t_symbols[4]

    @property
    def sch_a(self):
        return self.t_symbols[5]

    @property
    def sch_b(self):
        return self.t_symbols[6]

    def set_scheme_coeffs(self, p0, p1, a, b):
        """Write the four global integration-scheme coefficients into the
        `t`-block.  See `DifferentialVariable.declare_equations` for the closure
        they parameterise; `_CN_COEFFS` restores Crank-Nicolson."""
        base = 2 * self.n_v + self.n_p
        self.values[base + 3] = p0
        self.values[base + 4] = p1
        self.values[base + 5] = a
        self.values[base + 6] = b

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
        self._refresh_inputs()

    def set_t_prev(self, t_prev):
        self.values[2 * self.n_v + self.n_p + 1] = t_prev
        self._refresh_inputs()

    # --- time-dependent Input signals ------------------------------------------------

    def _collect_inputs(self):
        """Every `Input` reachable in the component tree (depth-first)."""
        found = []

        def _walk(node):
            for c in node.components.values():
                if isinstance(c, Input):
                    found.append(c)
                elif isinstance(c, Model):
                    _walk(c)

        _walk(self)
        return found

    def _collect_events(self):
        """Aggregate every component's `declare_events()` over the subtree.

        Walks the whole component tree (depth-first, like `_collect_inputs`),
        calling `declare_events()` on each `Model` node and flattening the
        returned event times into one sorted, de-duplicated list.  Coincident
        events (within `event_eps`) collapse to one so two components that kink
        at the same instant only create one pair of step boundaries.
        """
        times = []

        def _walk(node):
            for t_ev in node.declare_events():
                t_ev = float(t_ev)
                if math.isfinite(t_ev):
                    times.append(t_ev)
            for c in node.components.values():
                if isinstance(c, Model):
                    _walk(c)

        _walk(self)
        return sorted(times)

    def _refresh_inputs(self):
        """Re-sample every `Input` at the current `(t, t_prev)` and write the
        results straight into their two parameter slots in `self.values`.

        Called from the time setters so the lambdified residual/Jacobian
        always sees `u(t_{k+1})` and `u(t_k)` consistent with the time level
        being solved.  No-op (single attribute read) for systems without any
        `Input`, so the hot path is unaffected.
        """
        refs = getattr(self, "_input_refs", None)
        if not refs:
            return
        base = 2 * self.n_v
        t = self.values[base + self.n_p]
        t_prev = self.values[base + self.n_p + 1]
        for inp, i_cur, i_prev in refs:
            cur = float(inp.func(t))
            prev = float(inp.func(t_prev))
            inp.components['cur'].value = cur
            inp.components['prev'].value = prev
            self.values[base + i_cur] = cur
            self.values[base + i_prev] = prev

    # --- evaluation / Newton solve / time stepping -----------------------------------

    def eval_residuals(self, vars):
        self.set_vars_values(vars)
        return self.lambdified_eqs(*self.values)

    def _safe_residual_norm(self):
        """`||F(x)||` at the CURRENT variable values, returning `+inf` instead
        of raising / NaN when the state is thermodynamically invalid.

        This is the merit function the backtracking line search minimises.
        Returning `+inf` for an infeasible state (e.g. a Newton step that
        overshoots a boiling density cliff into negative density -- which makes
        a property call raise or produce NaN) is exactly what lets the line
        search REJECT that step and backtrack to a feasible one.
        """
        try:
            r = np.asarray(self.eval_residuals(self.get_vars_values())).reshape(-1)
        except Exception:
            return np.inf
        n = float(fast_error_norm(r))
        return n if np.isfinite(n) else np.inf

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

    def record_state(self, step_wall_time=np.nan, step_error=np.nan):
        self.record['time'].append(self.get_t_value())
        full_state = self.lambdified_raw_vars(*self.values)
        self.record['state'].append(full_state)
        self.record['step_wall_time'].append(float(step_wall_time))
        self.record['step_error'].append(float(step_error))

    def next_step(self, step_wall_time=np.nan, step_error=np.nan):
        self.set_prev_vars_values(self.get_vars_values())
        self.set_t_prev(self.get_t_value())
        # Note: `solve_dae_step` already advanced `t` by `dt`. Advancing here too would
        # double-count, so we deliberately do not call `set_t` again.
        self.record_state(step_wall_time=step_wall_time, step_error=step_error)

    # --- recorded-state access (shared with the service layer) ------------

    def latest_state(self):
        """The most recently recorded full-state vector (as a numpy array), or
        ``None`` if nothing has been recorded yet (i.e. before
        ``initialise()`` / the first ``next_step()``)."""
        state = self.record.get('state')
        if not state:
            return None
        return np.asarray(state[-1])

    def resolve_vars(self, vars=None):
        """Map requested variable name(s) to ``(display_name, column)`` pairs.

        Names match exactly, else by dotted-suffix, else by bare-suffix (see
        :func:`match_name_index`) -- the same rule the service layer uses, so a
        convenient suffix like ``"wall_0_0.C_1"`` resolves identically here.
        ``vars=None`` returns every recorded variable (full names, in order); a
        single string is treated as a one-element list.  Each requested name
        resolves to its *first* match (unresolved names are skipped).
        """
        names = self.record.get('vars_names', [])
        if vars is None:
            return [(n, i) for i, n in enumerate(names)]
        if isinstance(vars, str):
            vars = [vars]
        out = []
        for req in vars:
            i = match_name_index(names, req)
            if i is not None:
                out.append((req, i))
        return out

    def get_state(self, vars=None):
        """Latest recorded value of each requested variable as a
        ``{display_name: float}`` dict (``vars=None`` -> every variable).

        Mirrors the service client's ``get_state``; matching is by suffix.
        Returns ``{}`` if no state has been recorded yet.
        """
        row = self.latest_state()
        if row is None:
            return {}
        return {name: float(row[i]) for name, i in self.resolve_vars(vars)}

    def state_value(self, name):
        """Latest recorded value of the *first* variable matching ``name``
        (suffix match) as a float.

        Raises ``KeyError`` if nothing has been recorded yet or no variable
        matches.
        """
        row = self.latest_state()
        if row is None:
            raise KeyError(
                "no recorded state yet; call initialise() (or next_step()) first")
        i = match_name_index(self.record.get('vars_names', []), name)
        if i is None:
            raise KeyError(f"no recorded variable matching {name!r}")
        return float(row[i])

    def state_values(self, name):
        """Latest recorded values of *all* variables matching ``name`` (suffix
        match) as a 1-D numpy array -- e.g. to sum a per-segment quantity such
        as ``state_values("m_dot_a_leak").sum()``.

        Raises ``KeyError`` if nothing has been recorded yet or no variable
        matches.
        """
        row = self.latest_state()
        if row is None:
            raise KeyError(
                "no recorded state yet; call initialise() (or next_step()) first")
        idx = match_name_indices(self.record.get('vars_names', []), name)
        if not idx:
            raise KeyError(f"no recorded variable matching {name!r}")
        return np.array([row[i] for i in idx])

    # --- recorded-history (timeseries) access -----------------------------
    #
    # These are the *timeseries* analogues of the latest-state accessors
    # above: where `state_value` / `state_values` return the most recent
    # value(s), `series` / `series_values` return the whole recorded column(s)
    # over time, and `record_time` gives the matching time axis.  All accept an
    # optional ``start:stop:stride`` window so a long adaptive run can be
    # subsampled without copying the full history first.

    def record_time(self, start=0, stop=None, stride=1):
        """Recorded time axis as a 1-D numpy array (optionally sliced).

        Pair with :meth:`series` / :meth:`series_values` (which accept the same
        window) to get aligned ``(t, y)`` arrays for plotting or integration.
        """
        sl = slice(start, stop, max(1, stride or 1))
        return np.asarray(self.record.get('time', [])[sl], dtype=float)

    def series(self, name, start=0, stop=None, stride=1):
        """1-D recorded timeseries of the *first* variable matching ``name``
        (suffix match) -- the timeseries analogue of :meth:`state_value`.

        Raises ``KeyError`` if nothing has been recorded yet or no variable
        matches.
        """
        state_all = self.record.get('state', [])
        if not state_all:
            raise KeyError(
                "no recorded state yet; call initialise() (or next_step()) first")
        i = match_name_index(self.record.get('vars_names', []), name)
        if i is None:
            raise KeyError(f"no recorded variable matching {name!r}")
        sl = slice(start, stop, max(1, stride or 1))
        return np.asarray(state_all)[sl, i].astype(float)

    def series_values(self, name, start=0, stop=None, stride=1):
        """2-D recorded timeseries (rows = time, columns = matches) of *all*
        variables matching ``name`` (suffix match) -- the timeseries analogue
        of :meth:`state_values`.

        Sum a per-instance quantity across segments in one call, e.g.
        ``series_values("m_dot_a_leak").sum(axis=1)``.  Raises ``KeyError`` if
        nothing has been recorded yet or no variable matches.
        """
        state_all = self.record.get('state', [])
        if not state_all:
            raise KeyError(
                "no recorded state yet; call initialise() (or next_step()) first")
        idx = match_name_indices(self.record.get('vars_names', []), name)
        if not idx:
            raise KeyError(f"no recorded variable matching {name!r}")
        sl = slice(start, stop, max(1, stride or 1))
        return np.asarray(state_all)[sl][:, idx].astype(float)

    def interp_series(self, name, t, *, left=np.nan, right=np.nan,
                      start=0, stop=None, stride=1):
        """Linearly resample the recorded timeseries of the *first* variable
        matching ``name`` onto arbitrary query time(s) ``t``.

        An adaptive run lands on a non-uniform time grid that differs between
        runs / strategies, so to compare whole *trajectories* -- not just the
        final state -- resample each onto a shared grid first::

            tq = np.linspace(t0, t1, 501)
            p_a = run_a.interp_series("tank_3.gas.p", tq)
            p_b = run_b.interp_series("tank_3.gas.p", tq)
            max_abs = np.max(np.abs(p_a - p_b))

        ``t`` may be a scalar (returns ``float``) or array-like (returns an
        ndarray).  Query times outside the recorded span yield ``left`` /
        ``right`` (default ``nan``) instead of being silently clamped to the
        end values.  Raises ``KeyError`` if nothing is recorded yet or no
        variable matches.
        """
        tp = self.record_time(start, stop, stride)
        fp = self.series(name, start, stop, stride)
        t_arr = np.asarray(t, dtype=float)
        out = np.interp(t_arr, tp, fp, left=left, right=right)
        return float(out) if t_arr.ndim == 0 else out

    def interp_state(self, t, vars=None, *, left=np.nan, right=np.nan,
                     start=0, stop=None, stride=1):
        """Linearly resample many recorded variables onto query time(s) ``t``.

        The timeseries analogue of :meth:`get_state`: where ``get_state``
        returns each requested variable's *latest* value, this returns each one
        resampled onto ``t``.  Returns ``{display_name: values}`` where
        ``values`` is a ``float`` (scalar ``t``) or ndarray (array-like ``t``);
        ``vars=None`` resamples every recorded variable.  Matching and
        out-of-range semantics follow :meth:`interp_series`.
        """
        tp = self.record_time(start, stop, stride)
        state_all = self.record.get('state', [])
        if not state_all:
            raise KeyError(
                "no recorded state yet; call initialise() (or next_step()) first")
        sl = slice(start, stop, max(1, stride or 1))
        arr = np.asarray(state_all)[sl].astype(float)
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        out = {}
        for disp, i in self.resolve_vars(vars):
            vals = np.interp(t_arr, tp, arr[:, i], left=left, right=right)
            out[disp] = float(vals) if scalar else vals
        return out

    def expand_names(self, template, *index_specs):
        """Expand a ``str.format`` template into an ordered list of variable
        names by formatting it with the Cartesian product of ``index_specs``
        (one iterable -- a ``range`` or list -- per replacement field; the last
        spec varies fastest).

        Use this when a quantity is indexed at one *or more* positions in the
        name, e.g.::

            expand_names("wall_0_0.C_{}", range(1, 4))
            #  -> ['wall_0_0.C_1', 'wall_0_0.C_2', 'wall_0_0.C_3']
            expand_names("wall_{}_0.C_{}", range(2), [1, 2])
            #  -> ['wall_0_0.C_1', 'wall_0_0.C_2', 'wall_1_0.C_1', 'wall_1_0.C_2']

        The result can be handed straight to :meth:`get_record` /
        :meth:`get_series`; :meth:`series_grid` does exactly that and also
        reshapes the result to the index grid.
        """
        specs = [list(s) for s in index_specs]
        return [template.format(*combo) for combo in itertools.product(*specs)]

    def series_grid(self, template, *index_specs, start=0, stop=None,
                    stride=1, reshape=True):
        """Recorded timeseries for a *grid* of variables, named by formatting
        ``template`` with the Cartesian product of ``index_specs`` (see
        :meth:`expand_names`).

        Returns a numpy array whose first axis is time and whose remaining
        axes follow ``index_specs`` -- so a single varying index gives the same
        ``(n_time, n)`` shape you'd otherwise build by hand::

            C = system.series_grid("wall_0_0.C_{}", range(1, n_nodes + 1))
            #  shape (n_time, n_nodes); C[t, j] is node j+1 at time t

        and two indices give ``(n_time, len(spec_0), len(spec_1))``.  Pass
        ``reshape=False`` to keep the flat ``(n_time, n_names)`` layout.  Every
        generated name must resolve (exact, else suffix match) -- a missing one
        raises ``KeyError`` so the returned grid shape is always exact.
        """
        specs = [list(s) for s in index_specs]
        names = [template.format(*combo) for combo in itertools.product(*specs)]
        state_all = self.record.get('state', [])
        if not state_all:
            raise KeyError(
                "no recorded state yet; call initialise() (or next_step()) first")
        var_names = self.record.get('vars_names', [])
        cols, missing = [], []
        for nm in names:
            i = match_name_index(var_names, nm)
            (cols if i is not None else missing).append(i if i is not None else nm)
        if missing:
            raise KeyError(f"no recorded variable(s) matching {missing!r}")
        sl = slice(start, stop, max(1, stride or 1))
        rows = np.asarray(state_all)[sl][:, cols].astype(float)
        if reshape and len(specs) > 1:
            rows = rows.reshape((rows.shape[0],) + tuple(len(s) for s in specs))
        return rows

    def get_record(self, vars=None, start=0, stop=None, stride=1):
        """A slice of recorded history, row-major (mirrors the service client's
        ``get_record``): ``{"names": [...], "time": ndarray, "rows": ndarray}``
        where ``rows[k]`` is the selected variables at ``time[k]``.

        Each requested name resolves to its *first* match (use
        :meth:`series_values` to gather every match of a name).
        ``vars=None`` returns every recorded variable, in order.
        """
        cols = self.resolve_vars(vars)
        time = self.record_time(start, stop, stride)
        state_all = self.record.get('state', [])
        sl = slice(start, stop, max(1, stride or 1))
        if state_all and cols:
            rows = np.asarray(state_all)[sl][:, [i for _, i in cols]].astype(float)
        else:
            rows = np.empty((len(time), len(cols)), dtype=float)
        return {"names": [name for name, _ in cols], "time": time, "rows": rows}

    def get_series(self, vars=None, start=0, stop=None, stride=1):
        """A slice of recorded history, column-major (mirrors the service
        client's ``get_series``): ``{"time": ndarray, "series": {name: ndarray}}``.

        This is the convenient shape for plotting a chosen variable / list over
        time.  Each requested name resolves to its *first* match;
        ``vars=None`` returns every recorded variable.
        """
        rec = self.get_record(vars, start=start, stop=stop, stride=stride)
        series = {name: rec["rows"][:, k] for k, name in enumerate(rec["names"])}
        return {"time": rec["time"], "series": series}

    @line_profiler.profile
    def initialise(self, n=1, relaxation=1.0, tol=1e-6, max_iter=100,
                   line_search=False, steady=False, steady_dt=1e6):
        """Set the system to a Newton-consistent state at t = 0.

        For weakly-coupled or smoothly-conditioned problems the default
        `relaxation=1.0` (full Newton step) is fine. For systems with stiff
        startup transients (e.g. a pressure vessel charging from a much higher
        upstream pressure where the pipe's default initial guesses are far
        from the boundary-driven values), pass a smaller `relaxation` to
        damp the first few iterations and avoid overshooting into infeasible
        thermodynamic states, or pass `line_search=True` to let a backtracking
        line search pick the step length automatically (see `custom_solve`).

        With the default `steady=False` the solve runs at `dt = 0`, which
        PINS every `DifferentialVariable` at its seeded value (the closure
        degenerates to `x = x_prev`) and lets the derivative companions
        absorb whatever imbalance the seeding carries.  That is the right
        semantics when the initial state is genuinely transient (e.g. a
        vessel that starts charging at t = 0: `der_x` comes out as the true
        initial rate).  But when the run is meant to START FROM STEADY STATE,
        a seeding inconsistency leaves large spurious derivative values that
        poison adaptive-stepper error estimates on the first steps.  Pass
        `steady=True` to append a relaxation to the true steady state: one
        L-stable implicit-Euler step of size `steady_dt` (default 1e6 s) with
        the clock held at t = 0 (Inputs keep their t=0 values), after which
        every surviving derivative is ~0 and the states sit on the steady
        solution.
        """
        self.set_t_values([0.0, 0.0, 0.0])
        init_values = np.array([var.value for var in self.active_vars_references])
        init_params = np.array([param.value for param in self.raw_param_references])
        self.set_vars_values(init_values)
        self.set_prev_vars_values(init_values)
        self.set_param_values(init_params)
        self.custom_solve(tol=tol, max_iter=max_iter, relaxation=relaxation,
                          line_search=line_search)
        self.next_step()
        if steady:
            self.set_scheme_coeffs(1.0, 0.0, 1.0, 0.0)   # implicit Euler
            self.set_dt(steady_dt)
            self.set_t(0.0)
            self.set_t_prev(0.0)
            self.custom_solve(tol=tol, max_iter=max_iter, relaxation=relaxation,
                              line_search=line_search)
            self.set_scheme_coeffs(*_CN_COEFFS)
            self.set_t_values([0.0, 0.0, 0.0])
            self.next_step()

    def update_delta(self):
        # BLT path: per-block forward substitution.  For feed-forward
        # topologies (linear pipe chain, pipe tree) this is much faster
        # than a whole-system splu because most blocks are 1x1 (single
        # divide).  For loop-containing systems the loop becomes one block
        # solved with dense/sparse LU and the rest stay 1x1.
        if getattr(self, "_blt_plan", None) is not None:
            self._update_delta_blt()
            return
        # Sparse path: evaluate Jacobian nonzero values and solve via
        # scipy's SuperLU on the full system.  Used when BLT is disabled
        # or didn't apply.  For the pipe-tree case the Jacobian is < 7%
        # dense, so SuperLU wins both in time and memory vs a dense LU.
        if getattr(self, "_lambdified_jac_values", None) is not None:
            self.set_vars_values(self.get_vars_values())  # no-op write to refresh `values`
            jac_vals = np.asarray(self._lambdified_jac_values(*self.values)).reshape(-1)
            r = np.asarray(self.eval_residuals(self.get_vars_values())).reshape(-1)
            n_eq = self.delta_values.size  # square system: n_eq == n_v
            if self._jac_csc_perm is not None:
                self.delta_values[:] = fast_sparse_solve_cached(
                    jac_vals,
                    self._jac_csc_perm,
                    self._jac_csc_indices,
                    self._jac_csc_indptr,
                    (n_eq, self.n_v),
                    r,
                )
            else:
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

    def _update_delta_blt(self):
        """BLT-guided Newton-step solve.  Dispatches on `plan['solve_mode']`:

          * 'triangular' (largest block == 1): permute Jacobian into BLT order
            and run scipy's `spsolve_triangular` -- a C-level forward
            substitution that's ~4x faster than `splu` on fully-1x1 systems.
          * 'blockwise'  (>= 95% of vars in 1x1 blocks with a tiny tail of
            larger blocks): per-block forward substitution loop.  Useful
            mainly post-tearing.
          * 'monolithic' (one dominant SCC OR many medium-sized blocks):
            permute into BLT order and run `splu` on the permuted matrix.
            The permutation gives SuperLU a head-start on the fill-reducing
            ordering AND keeps the BLT structure intact for future
            block-aware passes; the per-iteration cost is at parity with
            the un-permuted `splu` path.
        """
        self.set_vars_values(self.get_vars_values())  # refresh `values`
        jac_vals = np.asarray(
            self._lambdified_jac_values(*self.values)
        ).reshape(-1)
        r = np.asarray(self.eval_residuals(self.get_vars_values())).reshape(-1)

        plan = self._blt_plan
        mode = plan['solve_mode']
        if mode == 'triangular':
            self._blt_solve_triangular(jac_vals, r)
        elif mode == 'blockwise':
            self._blt_solve_blockwise(jac_vals, r)
        else:
            self._blt_solve_monolithic(jac_vals, r)

    def _blt_solve_triangular(self, jac_vals, r):
        """Pure forward substitution on a BLT-permuted matrix (all 1x1 blocks).

        After permutation the matrix is strictly lower-triangular with the
        Jacobian's diagonal entries on the diagonal.  `spsolve_triangular`
        runs the entire forward substitution as one C call with no Python
        loop overhead.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve_triangular
        plan = self._blt_plan
        n = self.n_v
        A = csr_matrix(
            (jac_vals, (plan['perm_rows'], plan['perm_cols'])),
            shape=(n, n),
        )
        # Permute r by row permutation
        r_perm = r[plan['eq_perm_inv']]
        delta_perm = spsolve_triangular(A, r_perm, lower=True,
                                        unit_diagonal=False)
        # Inverse-permute delta back into original variable order
        self.delta_values[:] = delta_perm[plan['var_perm']]

    def _blt_solve_monolithic(self, jac_vals, r):
        """Permute Jacobian into BLT row/col order, then run splu on the
        permuted CSC.  Equivalent in speed to splu on the unpermuted matrix
        (SuperLU re-orders internally) but keeps the BLT framing intact for
        block-aware passes layered above.
        """
        n = self.n_v
        if self._jac_csc_perm is not None:
            self.delta_values[:] = fast_sparse_solve_cached(
                jac_vals,
                self._jac_csc_perm,
                self._jac_csc_indices,
                self._jac_csc_indptr,
                (n, n),
                r,
            )
        else:
            self.delta_values[:] = fast_sparse_solve(
                jac_vals,
                self._jac_sparse_rows,
                self._jac_sparse_cols,
                (n, n),
                r,
            )

    def _blt_solve_blockwise(self, jac_vals, r):
        """Per-block forward substitution.  Used when most vars are in 1x1
        blocks but a small tail of larger blocks needs dense/sparse LU."""
        plan = self._blt_plan
        block_vars = plan['block_vars']
        block_eqs = plan['block_eqs']
        block_n = plan['block_n']
        diag_jac_idx = plan['diag_jac_idx']
        diag_local_rows = plan['diag_local_rows']
        diag_local_cols = plan['diag_local_cols']
        lower_jac_idx = plan['lower_jac_idx']
        lower_local_rows = plan['lower_local_rows']
        lower_global_cols = plan['lower_global_cols']
        block_solver = plan['block_solver']

        delta = np.empty(self.n_v)
        _coo = None
        _splu = None

        for b in range(plan['n_blocks']):
            n_b = int(block_n[b])
            vidx = block_vars[b]
            eidx = block_eqs[b]

            r_local = r[eidx].copy()
            li = lower_jac_idx[b]
            if li.size:
                contribs = jac_vals[li] * delta[lower_global_cols[b]]
                np.add.at(r_local, lower_local_rows[b], -contribs)

            di = diag_jac_idx[b]
            solver = block_solver[b]
            if solver == 'scalar':
                delta[vidx[0]] = r_local[0] / jac_vals[di[0]]
            elif solver == 'dense':
                A = np.zeros((n_b, n_b))
                A[diag_local_rows[b], diag_local_cols[b]] = jac_vals[di]
                delta[vidx] = np.linalg.solve(A, r_local)
            else:
                if _coo is None:
                    from scipy.sparse import coo_matrix as _coo
                    from scipy.sparse.linalg import splu as _splu
                A = _coo(
                    (jac_vals[di], (diag_local_rows[b], diag_local_cols[b])),
                    shape=(n_b, n_b),
                ).tocsc()
                delta[vidx] = _splu(A).solve(r_local)

        self.delta_values[:] = delta

    @line_profiler.profile
    def custom_solve(self, tol=1e-6, max_iter=100, relaxation=1.0,
                     raise_on_no_convergence=False, line_search=False,
                     ls_beta=0.5, ls_c=1e-4, ls_max_backtracks=30,
                     ls_grow=np.inf):
        """Damped Newton solve.

        Two step-length policies:

        * Default (`line_search=False`): a fixed `relaxation` damping -- every
          step is `x <- x - relaxation * J^-1 F`.  Cheap (no extra residual
          evals) and fine for smoothly-conditioned problems, but on a stiff
          property cliff (boiling/flashing density) the right `relaxation` has
          to be hand-tuned or the full step overshoots into an infeasible state.

        * `line_search=True`: feasibility-guarded backtracking.  Start from a
          step of `relaxation` (default 1.0 = full Newton) and halve it
          (`ls_beta`) until the trial state is FEASIBLE -- i.e. `||F||` is
          finite.  On a stiff property cliff (boiling/flashing) a full Newton
          step overshoots into an infeasible state (e.g. negative density),
          which makes a property call raise / return NaN and scores `+inf`
          (see `_safe_residual_norm`); that step is rejected and the search
          backs off until it lands in a valid region.  Where the full step is
          already feasible (the common case) `lam` stays at `relaxation`, so
          this is identical to plain Newton -- it only damps where it must,
          with no hand-tuned `relaxation`.  Costs `1 + n_backtracks` extra
          residual evals per iteration.

          A decrease test (Armijo, or the residual-growth cap `ls_grow`) is
          NOT applied by default: the mixed-unit residual (Pa ~ 1e5 dwarfs
          continuity ~ 1e-3) makes the raw norm a poor merit, and capping
          growth rejects perfectly good Newton steps in the ill-conditioned
          single-phase region -- benchmarked at ~4-18x more iterations than
          the pure feasibility guard.  `ls_grow` defaults to `+inf` (pure
          feasibility); pass a finite value to additionally reject any step
          that grows `||F||` past `ls_grow x ||F(x)||` (rarely useful here).

        `ls_c` is unused (kept for signature stability).
        """
        guess = self.get_vars_values()
        error_norm = np.inf
        i = 0
        # Pre-fetch the scale vector once; the inner Newton loop must be
        # tight.  When scaling is inactive (span < 2 orders of magnitude
        # or disabled), we hit the unscaled fast path that's identical
        # to the legacy code.
        is_scaled = getattr(self, '_var_scaling_active', False)
        if is_scaled:
            inv_scales = self._var_inv_scales
            buf = self._scaled_delta_buf
        while error_norm > tol and i < max_iter:
            try:
                self.update_delta()
            except PropertyEvaluationError as exc:
                # A Newton iterate drove the state outside the medium's valid
                # thermodynamic domain (e.g. CoolProp cannot flash an enthalpy
                # below its minimum), so the Jacobian/residual evaluation
                # raised.  Treat this exactly like Newton non-convergence:
                # record the failure so `raise_on_no_convergence` callers
                # (every adaptive strategy) reject the step and retry at a
                # smaller dt instead of crashing the whole run.
                self._last_solve_error_norm = np.inf
                self._last_solve_iters = i
                if raise_on_no_convergence:
                    raise NewtonConvergenceFailure(np.inf, i, max_iter, tol) from exc
                return guess
            except RuntimeError as exc:
                # SuperLU (`splu`) raises `RuntimeError("Factor is exactly
                # singular")` when the Jacobian is degenerate -- most often
                # because a Newton overshoot produced NaN/Inf entries (a
                # smoothly-extrapolating surrogate medium won't raise a
                # domain error the way CoolProp does, so the non-finite value
                # only surfaces here at the factorization).  A numerically
                # singular pivot is unrecoverable at this dt, so fall through
                # to the same non-convergence path: let the adaptive stepper
                # shrink dt and retry instead of aborting the whole run.
                if "singular" not in str(exc).lower():
                    raise
                self._last_solve_error_norm = np.inf
                self._last_solve_iters = i
                if raise_on_no_convergence:
                    raise NewtonConvergenceFailure(np.inf, i, max_iter, tol) from exc
                return guess
            delta = self.delta_values
            # (Scaled) norm of the FULL Newton step, computed before any
            # damping; the actual step taken is `lam * delta`.
            if is_scaled:
                np.multiply(delta, inv_scales, out=buf)
                full_step_norm = fast_error_norm(buf)
            else:
                full_step_norm = fast_error_norm(delta)
            if line_search:
                x0 = guess.copy()
                f0 = self._safe_residual_norm()
                # By default (`ls_grow=inf`) the only thing that triggers a
                # backtrack is INFEASIBILITY: an overshoot into e.g. negative
                # density scores +inf and fails the `f_try <= f_cap` test below,
                # while every feasible (finite) trial passes -- so Newton keeps
                # its natural full step wherever it is valid.  A finite
                # `ls_grow` additionally caps residual growth at `ls_grow x f0`
                # (skipped while f0 is infinite, i.e. recovering from a bad
                # start, so any finite trial is accepted).
                if not np.isfinite(f0) or not np.isfinite(ls_grow):
                    f_cap = np.inf
                else:
                    f_cap = ls_grow * f0
                lam = relaxation
                for _ls in range(ls_max_backtracks):
                    guess[:] = x0
                    guess -= lam * delta
                    f_try = self._safe_residual_norm()
                    if np.isfinite(f_try) and f_try <= f_cap:
                        break
                    lam *= ls_beta
                # On exhaustion the most-damped (smallest-lam) step is already
                # applied to `guess`; convergence is judged on what we took.
                error_norm = lam * full_step_norm
            else:
                delta *= relaxation
                error_norm = relaxation * full_step_norm
                guess -= delta
            i += 1
        # Expose the last solve's outcome so `solve_adaptive_step` (and any
        # diagnostic tooling) can inspect convergence without re-evaluating
        # the residual.  Both attributes are always set, even on success.
        self._last_solve_error_norm = float(error_norm)
        self._last_solve_iters = i
        if raise_on_no_convergence and error_norm > tol:
            raise NewtonConvergenceFailure(error_norm, i, max_iter, tol)
        return guess

    @line_profiler.profile
    def solve_dae_step(self, dt, relaxation=1.0, tol=1e-6, max_iter=100,
                       raise_on_no_convergence=False, line_search=False):
        self.set_dt(dt)
        self.set_t(self.get_t_value() + dt)
        self.custom_solve(tol=tol, max_iter=max_iter, relaxation=relaxation,
                          raise_on_no_convergence=raise_on_no_convergence,
                          line_search=line_search)

    # --- adaptive time stepping -------------------------------------------------------
    #
    # `solve_adaptive_step(dt_target, strategy=...)` tries `dt_target`, asks the
    # chosen strategy whether the result is acceptable, and either commits or
    # restores + retries at smaller `dt`.  It NEVER calls `next_step()` -- the
    # caller commits explicitly, mirroring the fixed-step pattern.  A rejected
    # step never reaches `record`.
    #
    # Three strategies live in the `_ADAPTIVE_STRATEGIES` registry plus a "fixed"
    # short-circuit that just calls `solve_dae_step` once.  Each strategy is
    # responsible for its own internal solve loop because Richardson needs to
    # take MULTIPLE solves per accepted step (one full + two halves) -- it's
    # cleaner to push the loop into each strategy than to invent a multi-step
    # protocol that all three share.

    def _snapshot_state(self):
        """Capture the minimum state needed to retry a step: current vars,
        previous vars (used by CN), the time triplet, and the snapshotted
        `der_x` value of every active `DifferentialVariable` (used by the
        predictor-corrector strategy)."""
        return {
            "values": self.values[:self.n_v].copy(),
            "prev_values": self.values[self.n_v:2 * self.n_v].copy(),
            "t": self.get_t_value(),
            "t_prev": self.get_t_prev_value(),
            "dt": self.get_dt_value(),
        }

    def _restore_state(self, snap):
        self.set_vars_values(snap["values"])
        self.set_prev_vars_values(snap["prev_values"])
        self.set_t(snap["t"])
        self.set_t_prev(snap["t_prev"])
        self.set_dt(snap["dt"])

    def _get_diff_var_index_pairs(self):
        """Returns `[(state_idx, der_idx), ...]` into `active_vars_references`,
        one entry per `DifferentialVariable` whose state AND derivative both
        survived trivial-equation reduction.  Cached after first call."""
        if getattr(self, "_diff_var_index_pairs_cache", None) is not None:
            return self._diff_var_index_pairs_cache
        refs = self.active_vars_references
        id_to_idx = {id(v): i for i, v in enumerate(refs)}
        pairs = []
        # Walk the component tree, picking up every DifferentialVariable.
        def _walk(node):
            if isinstance(node, DifferentialVariable):
                state_idx = id_to_idx.get(id(node))
                der_idx = id_to_idx.get(id(node.der_variable))
                if state_idx is not None and der_idx is not None:
                    pairs.append((state_idx, der_idx))
            if isinstance(node, Model):
                for child in node.components.values():
                    _walk(child)
        _walk(self)
        self._diff_var_index_pairs_cache = pairs
        return pairs

    def _get_diff_state_indices(self):
        """Returns the state-vector indices of EVERY `DifferentialVariable`
        whose state survived reduction, regardless of whether its derivative
        companion was eliminated.  Cached after first call.

        Unlike `_get_diff_var_index_pairs` (which needs the *derivative* slot
        too, e.g. for the predictor-corrector estimate), this is the set the
        TR-BDF2 stepper must advance: the BDF2 stage rewrites each diff state's
        `prev` slot, which is well-defined even when the trivial-equation
        reducer inlined the derivative definition."""
        if getattr(self, "_diff_state_index_cache", None) is not None:
            return self._diff_state_index_cache
        refs = self.active_vars_references
        id_to_idx = {id(v): i for i, v in enumerate(refs)}
        idxs = []

        def _walk(node):
            if isinstance(node, DifferentialVariable):
                state_idx = id_to_idx.get(id(node))
                if state_idx is not None:
                    idxs.append(state_idx)
            if isinstance(node, Model):
                for child in node.components.values():
                    _walk(child)

        _walk(self)
        self._diff_state_index_cache = np.array(idxs, dtype=int)
        return self._diff_state_index_cache

    def _get_var_atols(self, fallback_atol):
        """Per-variable absolute tolerance vector aligned with `active_vars_references`.
        Falls back to `fallback_atol` (the strategy's global value) for any
        variable that didn't override it via `Variable(..., atol=...)`."""
        return np.array(
            [v.atol if v.atol is not None else fallback_atol
             for v in self.active_vars_references],
            dtype=float,
        )

    def _tr_bdf2_step(self, dt, snap, diff_state_idx, pair_state_idx, pair_der_idx,
                      relaxation, tol, max_iter, line_search):
        """Take ONE TR-BDF2 step of size `dt` from the snapshot `snap` (state
        `x_n` at time `t_n`).  Returns ``(est, est_state_idx)`` where `est` is
        the embedded local-error estimate aligned with `est_state_idx`.

        TR-BDF2 is a two-stage, L-stable, stiffly-accurate, second-order
        one-step method:

          1. Trapezoidal sub-step of size ``gamma*dt`` to ``t_n + gamma*dt``
             (`gamma = 2 - sqrt(2)`), reusing the Crank-Nicolson closure.
          2. BDF2 sub-step to ``t_n + dt`` using the start value `x_n` and the
             stage value `x_gamma`.

        Both implicit stages share the diagonal coefficient ``1 - sqrt(2)/2``
        (the SDIRK property), so the Newton Jacobian structure is identical
        across stages.  Raises `NewtonConvergenceFailure` if either stage solve
        fails to converge, so the caller can reject and shrink `dt`.

        The BDF2 stage is expressed through the generalised closure WITHOUT a
        second history slot: it sets the scheme coefficients to ``(1, 0, d, 0)``
        -- which zeroes EVERY differential variable's `der_prev` term -- and
        folds the BDF2 history combination ``c2*x_n + c1*x_gamma`` into each diff
        state's `prev` slot.  Because it only ever writes the (always-surviving)
        state `prev` slot, it is correct whether or not the trivial-equation
        reducer inlined a variable's derivative definition.
        """
        nv = self.n_v
        g = _TRBDF2_GAMMA
        # Start every (re)try from the clean pre-step state: x_n in cur AND prev,
        # t == t_prev == t_n.
        self._restore_state(snap)
        t_n = snap["t"]
        x_n = snap["values"][diff_state_idx].copy()

        # ---- Stage 1: trapezoidal rule, sub-step of size gamma*dt -------------
        self.set_scheme_coeffs(*_CN_COEFFS)
        self.solve_dae_step(g * dt, relaxation=relaxation, tol=tol,
                            max_iter=max_iter, raise_on_no_convergence=True,
                            line_search=line_search)
        x_gamma = self.values[diff_state_idx].copy()
        # Capture the trapezoidal-stage slope for the embedded estimate (only
        # for diff vars whose derivative companion survived reduction).
        f_gamma = (self.values[pair_der_idx].copy()
                   if pair_der_idx.size else None)

        # ---- Stage 2: BDF2 to t_n + dt via prev-folding ----------------------
        # Closure with (p0,p1,a,b) = (1, 0, d, 0) is  x = x_prev + d*dt*der;
        # writing x_prev := c2*x_n + c1*x_gamma reproduces the BDF2 update
        # x = c2*x_n + c1*x_gamma + d*dt*f_{n+1} for every diff state.
        self.set_scheme_coeffs(1.0, 0.0, _TRBDF2_D, 0.0)
        self.values[nv + diff_state_idx] = _TRBDF2_C2 * x_n + _TRBDF2_C1 * x_gamma
        self.set_t_prev(t_n)                          # whole step starts at t_n
        self.set_dt(dt)
        self.set_t(t_n + dt)                          # resample inputs at t_{n+1}
        self.custom_solve(tol=tol, max_iter=max_iter, relaxation=relaxation,
                          raise_on_no_convergence=True, line_search=line_search)

        # ---- Embedded local-error estimate -----------------------------------
        if pair_der_idx.size:
            # Proper TR-BDF2 estimate from the three stage slopes f_n, f_gamma,
            # f_{n+1} (read straight from the derivative slots).  The weights
            # are the second divided difference of f at nodes {0, gamma, 1},
            # scaled by K so the estimate equals TR-BDF2's leading
            # O(dt^3 * y''') truncation term.
            f_n = snap["values"][pair_der_idx]
            f_np1 = self.values[pair_der_idx]
            est = _TRBDF2_K * dt * (
                _TRBDF2_E0 * f_n + _TRBDF2_E1 * f_gamma + _TRBDF2_E2 * f_np1)
            return est, pair_state_idx
        # Fallback for the (degenerate) case where every derivative was inlined:
        # recover f_{n+1} from the BDF2 increment and compare the order-2 result
        # to a backward-Euler predictor x_n + dt*f_{n+1} (an O(dt^2) estimate).
        x_np1 = self.values[diff_state_idx]
        f_np1 = (x_np1 - _TRBDF2_C2 * x_n - _TRBDF2_C1 * x_gamma) / (_TRBDF2_D * dt)
        est = x_np1 - (x_n + dt * f_np1)
        return est, diff_state_idx

    # --- live run controller (mutable mid-run from the service host) -------- #
    _RUN_CTRL_NUMERIC = (
        "dt_min", "dt_max", "grow", "shrink", "max_retries",
        "relaxation", "tol", "max_iter", "stop_time",
    )
    _RUN_CTRL_BOOL = ("line_search",)

    def _init_run_control(self, *, strategy, dt=None, stop_time=None, **kwargs):
        """Seed the per-run controller read by :meth:`iter_run` each step."""
        if isinstance(strategy, dict):
            strat = dict(strategy)
        else:
            strat = {"name": str(strategy)}
        ctrl = {"strategy": strat, "fixed_dt": dt, "stop_time": stop_time}
        for key in self._RUN_CTRL_NUMERIC:
            if key in kwargs and kwargs[key] is not None:
                ctrl[key] = float(kwargs[key])
        for key in self._RUN_CTRL_BOOL:
            if key in kwargs:
                ctrl[key] = bool(kwargs[key])
        self._run_ctrl = ctrl

    def get_run_control(self) -> dict:
        """Snapshot of the live run controller (empty when idle)."""
        ctrl = getattr(self, "_run_ctrl", None)
        return dict(ctrl) if ctrl else {}

    def clear_run_control(self):
        self._run_ctrl = None

    def update_run_control(self, **updates) -> dict:
        """Merge runtime controller knobs (honoured on the next step).

        When ``dt_max`` is lowered or ``dt_min`` raised, ``_dt_hint`` is
        clamped so the next step respects the new bounds immediately.
        """
        ctrl = getattr(self, "_run_ctrl", None)
        if not ctrl:
            raise RuntimeError("no active run controller")
        applied: dict = {}
        strat = updates.pop("strategy", None)
        if strat is not None:
            cur = ctrl.get("strategy") or {}
            if isinstance(cur, dict):
                base = dict(cur)
            else:
                base = {"name": str(cur)}
            if isinstance(strat, dict):
                base.update(strat)
            else:
                base = {"name": str(strat)}
            ctrl["strategy"] = base
            applied["strategy"] = dict(base)
        for key in self._RUN_CTRL_NUMERIC:
            if key not in updates:
                continue
            val = updates.pop(key)
            if val is None:
                ctrl.pop(key, None)
            else:
                ctrl[key] = float(val)
            applied[key] = ctrl.get(key)
        for key in self._RUN_CTRL_BOOL:
            if key not in updates:
                continue
            ctrl[key] = bool(updates.pop(key))
            applied[key] = ctrl[key]
        if updates:
            bad = ", ".join(sorted(updates))
            raise ValueError(f"unknown run-control key(s): {bad}")
        dt_max = ctrl.get("dt_max")
        dt_min = ctrl.get("dt_min")
        hint = getattr(self, "_dt_hint", None)
        if hint is not None:
            if dt_max is not None and hint > dt_max:
                self._dt_hint = float(dt_max)
            if dt_min is not None and self._dt_hint < dt_min:
                self._dt_hint = float(dt_min)
        return applied

    def set_dt_max(self, dt_max: float) -> float:
        """Raise or lower the adaptive step ceiling for the current run."""
        self.update_run_control(dt_max=float(dt_max))
        return float(self._run_ctrl["dt_max"])

    def solve_adaptive_step(self, dt_target, strategy="predictor_corrector",
                            dt_min=1e-9, dt_max=None,
                            grow=1.5, shrink=0.5, max_retries=20,
                            relaxation=1.0, tol=1e-6, max_iter=100,
                            line_search=False):
        """Take one accepted adaptive step toward `dt_target`.

        Returns `(dt_used, info)` where `info` is a dict with at least
        `{"strategy", "rejections", "metric", "n_iters"}` so the caller can
        log/plot rejection rates and dt history.

        The caller MUST call `next_step()` after a successful return to
        commit the new state to `record` and advance `t_prev`.

        Parameters
        ----------
        dt_target : float
            Hint for the dt to try first.  The controller may grow beyond it
            (up to `dt_max`) on a string of easy steps, so pass the LARGEST
            dt you'd accept, not a conservative one.
        strategy : str | dict | None
            One of `"predictor_corrector"` (default), `"derivative_limit"`,
            `"richardson"`, or `"fixed"`.  Pass a dict to also set tuning
            params, e.g. `{"name": "richardson", "tol_local": 1e-5}`.
        dt_min, dt_max : float
            Hard floor / ceiling.  `dt_max` defaults to `4 * dt_target`.
        grow, shrink : float
            Multiplicative factors used by the simple controllers
            (`derivative_limit`, `predictor_corrector`).  `richardson` uses
            its own `(tol/err)^(1/3)` formula and ignores these.
        max_retries : int
            Hard cap on rejection-and-retry iterations within a single
            adaptive step.  Raises `RuntimeError` if exceeded.
        relaxation, tol, max_iter, line_search
            Forwarded to every internal `solve_dae_step` / `custom_solve`.
            `line_search=True` enables feasibility-guarded backtracking, which
            damps Newton overshoots into infeasible thermodynamic states (see
            `custom_solve`) instead of relying on dt rejection alone.

        Raises
        ------
        This method always raises when it cannot produce a converged accepted
        step: `NewtonConvergenceFailure` for a `"fixed"` step (no dt to shrink),
        or `RuntimeError` for an adaptive strategy that exhausts `max_retries`
        or hits `dt_min`.  Callers (`run` / the service loop) decide whether to
        propagate or report it.
        """
        if dt_max is None:
            dt_max = 4.0 * dt_target

        name, params = _normalise_adaptive_strategy(strategy)
        if name == "fixed":
            # A fixed step has no `dt` to shrink, so Newton non-convergence is
            # unrecoverable here -- raise it (like the adaptive strategies raise
            # `RuntimeError` when they exhaust retries) rather than silently
            # marching on an unconverged state.  The driver (`run` / the service
            # loop) decides whether to propagate or report the failure.
            self.solve_dae_step(dt_target, relaxation=relaxation, tol=tol,
                                max_iter=max_iter, line_search=line_search,
                                raise_on_no_convergence=True)
            return dt_target, {"strategy": "fixed", "rejections": 0,
                               "metric": 0.0,
                               "n_iters": getattr(self, "_last_solve_iters", -1)}
        impl = _ADAPTIVE_STRATEGIES[name]
        return impl(self, dt_target=dt_target, dt_min=dt_min, dt_max=dt_max,
                    grow=grow, shrink=shrink, max_retries=max_retries,
                    relaxation=relaxation, tol=tol, max_iter=max_iter,
                    line_search=line_search, params=params)

    def iter_run(self, *, stop_time=None, strategy="richardson", dt=None,
                 dt_min=1e-9, dt_max=None, dt_start=None,
                 grow=1.5, shrink=0.5, max_retries=20,
                 relaxation=1.0, tol=1e-6, max_iter=100, line_search=False,
                 event_eps=1e-6):
        """Generator kernel shared by `Model.run` and the service run loop.

        Each `next()` selects this step's `dt` (the adaptive `_dt_hint` clipped
        to `dt_max`, `stop_time` and the next pending event boundary, or the
        fixed `dt`), takes ONE accepted step via `solve_adaptive_step`, commits
        it with `next_step()` (recording the per-step wall time / error), and
        yields the step `info` dict augmented with `{"dt", "step_wall_time"}`.

        Explicit time events (signal jumps / kinks declared by components via
        `declare_events`, e.g. a `Ramp`'s corners) become *step boundaries*: for
        each event time ``te`` the adaptive stepper is forced to land at
        ``te - event_eps`` and ``te + event_eps``, so no committed step ever
        integrates across the discontinuity.  The tiny crossing step between
        those two boundaries spans only ``~2*event_eps`` of the jump, which is
        negligible.  Event clipping applies to the adaptive strategies only;
        `"fixed"` stepping keeps its constant `dt`.

        The generator is intentionally OPEN-ended: it never decides when to stop.
        The driver loop owns ALL stop conditions (stop_time / steps / max_steps /
        wall_budget / cooperative stop) and simply stops pulling from the
        generator -- which is exactly what lets the service loop interleave
        socket polling, pause, MPI lock-step, streaming and pacing around each
        `next()` while `Model.run` keeps its own simpler bookkeeping.  Callers
        MUST therefore apply their own stop check (e.g. `t >= stop_time`) BEFORE
        each `next()`, so the final `dt` clip never produces a zero-length step.
        """
        name, _ = _normalise_adaptive_strategy(strategy)
        is_fixed = (name == "fixed")
        t_start = self.get_t_value()
        if dt_max is None and not is_fixed:
            if stop_time is not None:
                dt_max = (stop_time - t_start) / 20.0
            elif dt_start is not None or dt is not None:
                dt_max = 4.0 * (dt_start if dt_start is not None else dt)
        init_target = (dt if dt is not None
                       else dt_start if dt_start is not None
                       else dt_max if dt_max is not None else 1.0)

        self._init_run_control(
            strategy=strategy, dt=dt, stop_time=stop_time,
            dt_min=dt_min, dt_max=dt_max, grow=grow, shrink=shrink,
            max_retries=max_retries, relaxation=relaxation, tol=tol,
            max_iter=max_iter, line_search=line_search,
        )

        # Build the sorted list of event step boundaries (te +/- event_eps) that
        # fall strictly inside the run window.  `guard` is used both to merge
        # near-coincident boundaries (avoiding zero-length crossing steps) and
        # to skip the boundary we have just landed on.
        guard = event_eps * 1e-3
        boundaries = []
        if not is_fixed:
            for te in (getattr(self, "_event_times", None) or []):
                for b in (te - event_eps, te + event_eps):
                    if b <= t_start:
                        continue
                    if stop_time is not None and b >= stop_time:
                        continue
                    boundaries.append(b)
            boundaries.sort()
            if boundaries:
                merged = [boundaries[0]]
                for b in boundaries[1:]:
                    if b - merged[-1] > guard:
                        merged.append(b)
                boundaries = merged
        bptr = 0

        while True:
            ctrl = self._run_ctrl
            live_stop = ctrl.get("stop_time", stop_time)
            live_dt_max = ctrl.get("dt_max", dt_max)
            live_dt_min = ctrl.get("dt_min", dt_min)
            live_grow = ctrl.get("grow", grow)
            live_shrink = ctrl.get("shrink", shrink)
            live_max_retries = int(ctrl.get("max_retries", max_retries))
            live_relaxation = ctrl.get("relaxation", relaxation)
            live_tol = ctrl.get("tol", tol)
            live_max_iter = int(ctrl.get("max_iter", max_iter))
            live_line_search = ctrl.get("line_search", line_search)
            live_strategy = ctrl.get("strategy", strategy)

            t = self.get_t_value()
            if is_fixed:
                dt_try = ctrl.get("fixed_dt", dt)
            else:
                dt_try = getattr(self, "_dt_hint", init_target)
                if live_dt_max is not None:
                    dt_try = min(dt_try, live_dt_max)
            if live_stop is not None:
                dt_try = min(dt_try, live_stop - t)
            # Clip to the next pending event boundary (advancing past any we
            # have already reached).  `t` is monotonic across yields, so the
            # pointer never has to rewind.
            while bptr < len(boundaries) and boundaries[bptr] <= t + guard:
                bptr += 1
            if bptr < len(boundaries):
                dt_try = min(dt_try, boundaries[bptr] - t)

            adaptive_dt_max = (live_dt_max if live_dt_max is not None
                               else 4.0 * dt_try)
            t_step0 = time.perf_counter()
            dt_used, info = self.solve_adaptive_step(
                dt_try, strategy=live_strategy, dt_min=live_dt_min,
                dt_max=adaptive_dt_max, grow=live_grow, shrink=live_shrink,
                max_retries=live_max_retries, relaxation=live_relaxation,
                tol=live_tol, max_iter=live_max_iter,
                line_search=live_line_search)
            step_wall_time = time.perf_counter() - t_step0
            self.next_step(step_wall_time=step_wall_time,
                           step_error=info.get("metric", np.nan))
            info = dict(info)
            info["dt"] = dt_used
            info["step_wall_time"] = step_wall_time
            yield info

    def run(self, stop_time=None, *, steps=None, strategy="richardson",
            dt=None, dt_min=1e-9, dt_max=None, dt_start=None,
            grow=1.5, shrink=0.5, max_retries=20,
            relaxation=1.0, tol=1e-6, max_iter=100, line_search=False,
            max_steps=None, wall_budget=None, on_step=None,
            raise_on_no_convergence=True, event_eps=1e-6):
        """Integrate from the current time, owning the whole step loop.

        This is the high-level driver for scripts/examples: it pulls steps from
        the shared `iter_run` generator (which picks each step's `dt`, takes the
        step, and commits it with `next_step()`) and repeats until a stop
        condition is met.  The service run loop drives the SAME `iter_run`
        generator, so the per-step dt controller is implemented once.

        Stop conditions (give at least one; the loop ends on whichever trips
        first): advance until `t >= stop_time`, or until `steps` accepted steps
        have been taken, or `max_steps`, or `wall_budget` seconds of wall-clock.

        Parameters
        ----------
        stop_time : float | None
            Integrate until the model time reaches this (in the model's time
            unit).  Steps are clipped so the last one lands exactly on it.
        steps, max_steps : int | None
            Cap on the number of accepted steps (`steps` is the natural
            fixed-`dt` budget; `max_steps` is an extra hard safety cap).
        strategy : str | dict | None
            Passed through to :meth:`solve_adaptive_step`.  `"fixed"` requires
            `dt`; the adaptive strategies (`"richardson"`, `"derivative_limit"`,
            `"predictor_corrector"`) use `dt`/`dt_start` only as the first
            target and then self-adapt (carrying `dt` forward via `_dt_hint`).
        dt : float | None
            Fixed step (`strategy="fixed"`) or the initial adaptive target.
        dt_min, dt_max : float
            Hard floor / ceiling on the adaptive `dt`.  When `dt_max` is None it
            defaults to `(stop_time - t_start) / 20` (so dt can grow to ~5% of
            the span) or, failing that, `4 * dt_start`.
        dt_start : float | None
            First adaptive target when `dt` is not given (defaults to `dt_max`).
        grow, shrink, max_retries, relaxation, tol, max_iter
            Forwarded to :meth:`solve_adaptive_step`.
        wall_budget : float | None
            Abort (cleanly) after this many seconds of wall-clock.
        on_step : callable | None
            Called after every committed step as `on_step(model, info)` where
            `info` carries `{"step", "t", "dt", "rejections", "metric",
            "strategy"}`.  Return `False` to request a cooperative stop (this is
            how the service injects streaming / stop / pause).
        raise_on_no_convergence : bool
            If True, a Newton/adaptive failure propagates; if False the loop
            stops and the failure is reported in the returned summary.
        event_eps : float
            Half-width of the crossing step forced around each explicit time
            event (see :meth:`iter_run`); the stepper lands at `te +/- event_eps`
            so it never integrates across a declared discontinuity.

        Returns
        -------
        dict
            `{"strategy", "steps", "rejections", "t_start", "t_end",
            "wall_time", "stop_reason", "error"}`.
        """
        if stop_time is None and steps is None and max_steps is None:
            raise ValueError(
                "run() needs at least one stop condition: stop_time, steps, "
                "or max_steps")
        name, _ = _normalise_adaptive_strategy(strategy)
        is_fixed = (name == "fixed")
        if is_fixed and dt is None:
            raise ValueError("run(strategy='fixed', ...) requires a dt")

        t_start = self.get_t_value()
        steps_gen = self.iter_run(
            stop_time=stop_time, strategy=strategy, dt=dt, dt_min=dt_min,
            dt_max=dt_max, dt_start=dt_start, grow=grow, shrink=shrink,
            max_retries=max_retries, relaxation=relaxation, tol=tol,
            max_iter=max_iter, line_search=line_search, event_eps=event_eps)

        n_steps = 0
        n_rej = 0
        reason = "stop_time"
        err = None
        # Components that expose a `runtime_diagnostics()` hook (e.g. the
        # advective SegmentedChannel's cell-Peclet check) are polled after each
        # committed step and dropped once they report they are done.
        diag_components = self._collect_runtime_diagnostics()
        t_wall0 = time.perf_counter()
        while True:
            t = self.get_t_value()
            if stop_time is not None and t >= stop_time - 1e-9:
                reason = "stop_time"
                break
            if steps is not None and n_steps >= steps:
                reason = "steps"
                break
            if max_steps is not None and n_steps >= max_steps:
                reason = "max_steps"
                break
            if (wall_budget is not None
                    and time.perf_counter() - t_wall0 > wall_budget):
                reason = "wall_budget"
                break

            try:
                info = next(steps_gen)
            except (NewtonConvergenceFailure, RuntimeError) as e:
                if raise_on_no_convergence:
                    raise
                reason = "error"
                err = str(e)
                break

            n_steps += 1
            n_rej += int(info.get("rejections", 0))
            if diag_components:
                diag_components = [c for c in diag_components
                                  if not c.runtime_diagnostics()]
            if on_step is not None:
                cont = on_step(self, {
                    "step": n_steps, "t": self.get_t_value(),
                    "dt": info.get("dt"),
                    "rejections": info.get("rejections", 0),
                    "metric": info.get("metric", 0.0), "strategy": name})
                if cont is False:
                    reason = "callback"
                    break

        return {
            "strategy": name,
            "steps": n_steps,
            "rejections": n_rej,
            "t_start": t_start,
            "t_end": self.get_t_value(),
            "wall_time": time.perf_counter() - t_wall0,
            "stop_reason": reason,
            "error": err,
        }

    def set_initial_time(self, t):
        for c in self.components.values():
            if isinstance(c, Model):
                c.set_initial_time(t)

    def _collect_runtime_diagnostics(self):
        """Recursively gather sub-components exposing a ``runtime_diagnostics()``
        hook (polled by `run` after every committed step)."""
        found = []
        for c in self.components.values():
            if hasattr(c, "runtime_diagnostics") and callable(
                    getattr(c, "runtime_diagnostics")):
                found.append(c)
            if isinstance(c, Model):
                found.extend(c._collect_runtime_diagnostics())
        return found

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


# --- Adaptive-step strategy registry --------------------------------------------------
#
# Each strategy is a callable with the signature
#   strategy(model, dt_target, dt_min, dt_max, grow, shrink, max_retries,
#            relaxation, tol, max_iter, params) -> (dt_used, info_dict)
# and is responsible for its own rejection-and-retry loop.  The outer
# `Model.solve_adaptive_step` just dispatches into here based on the strategy
# name and forwards the controller knobs.  This split lets `richardson` do its
# own multi-solve dance (full step + two half-steps) without contorting a
# single-solve protocol that the simpler strategies would share.

# --- Integration-scheme coefficients --------------------------------------------------
#
# Crank-Nicolson is the default closure; `(p0, p1, a, b)` parameterise
# `DifferentialVariable.declare_equations`.  `set_scheme_coeffs(*_CN_COEFFS)`
# restores it after the TR-BDF2 stepper borrows the coefficient slots.
_CN_COEFFS = (1.0, 0.0, 0.5, 0.5)

# TR-BDF2 (gamma = 2 - sqrt(2)).  Trapezoidal sub-step to gamma*dt, then a BDF2
# sub-step `x = c2*x_n + c1*x_gamma + d*dt*f_{n+1}` to dt.  The trapezoidal and
# BDF2 stages share the diagonal coefficient `d = 1 - sqrt(2)/2 = gamma/2` (SDIRK).
_TRBDF2_GAMMA = 2.0 - 2.0 ** 0.5                # 0.5857864376...
_TRBDF2_C1 = 0.5 + 2.0 ** 0.5 / 2.0            # 1.2071067811...  ( x_gamma )
_TRBDF2_C2 = 0.5 - 2.0 ** 0.5 / 2.0            # -0.2071067811... ( x_n )
_TRBDF2_D = 1.0 - 2.0 ** 0.5 / 2.0             # 0.2928932188...  ( dt * f_{n+1} )
# Embedded error estimate `K * dt * (E0*f_n + E1*f_gamma + E2*f_{n+1})`.  The
# divided-difference weights E0/E1/E2 and the scalar K are derived so the
# estimate equals TR-BDF2's leading O(dt^3 * y''') local truncation term.
_TRBDF2_E0 = 1.0 / _TRBDF2_GAMMA                                    #  1.7071067811...
_TRBDF2_E1 = -1.0 / (_TRBDF2_GAMMA * (1.0 - _TRBDF2_GAMMA))         # -4.1213203435...
_TRBDF2_E2 = 1.0 / (1.0 - _TRBDF2_GAMMA)                            #  2.4142135623...
_TRBDF2_K = 2.0 ** 0.5 - 4.0 / 3.0                                  #  0.0807611844...


_DEFAULT_STRATEGY_PARAMS = {
    # `derivative_limit`: 1% relative state change per step is a generous
    # default appropriate for engineering transients; a tighter `rel_tol`
    # is unstable on systems with variables that pass through zero
    # (an oscillator's velocity at its peak amplitude has |x_new|/|x_old|
    # arbitrarily large).  Use `predictor_corrector` or `richardson` if
    # you need a tighter principled bound.
    "derivative_limit":   {"rel_tol": 1e-2, "atol": 1e-6},
    # `predictor_corrector`: the FE-CN mismatch is O(dt^2 * y'') -- which
    # is `dt`-times LARGER than the actual CN local error of O(dt^3 * y''').
    # So `tol_local=1e-2` here corresponds to a CN local error budget of
    # ~`tol_local * dt` (i.e. 1e-4 at dt=0.01).  Tighten to 1e-3 if you want
    # accuracy bounded a la fixed-dt 1e-3.
    "predictor_corrector": {"tol_local": 1e-2, "atol": 1e-8},
    # `richardson`: the half-vs-full mismatch IS the CN local error itself,
    # so `tol_local=1e-4` directly bounds local truncation per step.
    "richardson":         {"tol_local": 1e-4, "atol": 1e-8, "safety": 0.9, "order": 2},
    # `tr_bdf2`: L-stable, stiffly-accurate second-order one-step method whose
    # built-in embedded estimate IS the local truncation error, so `tol_local`
    # bounds local error directly (same calibration as `richardson`) -- but at
    # ~2 implicit solves/step instead of 3, and without CN's trapezoidal ringing
    # on stiff/discontinuous transients.
    "tr_bdf2":            {"tol_local": 1e-4, "atol": 1e-8, "safety": 0.9, "order": 2},
}


def _normalise_adaptive_strategy(strategy):
    """`strategy` is one of: None / "fixed" / "name" / {"name": "...", **params}."""
    if strategy is None:
        strategy = "predictor_corrector"
    if isinstance(strategy, str):
        name, user_params = strategy, {}
    elif isinstance(strategy, dict):
        s = dict(strategy)
        name = s.pop("name", None)
        if name is None:
            raise ValueError("strategy dict must include a 'name' key")
        user_params = s
    else:
        raise TypeError(
            f"strategy must be None, str, or dict, got {type(strategy).__name__}")
    if name == "fixed":
        return name, {}
    if name not in _ADAPTIVE_STRATEGIES:
        raise ValueError(
            f"unknown adaptive strategy {name!r}; choose from "
            f"{sorted(['fixed', *_ADAPTIVE_STRATEGIES])}")
    # Defaults < user overrides
    merged = dict(_DEFAULT_STRATEGY_PARAMS[name])
    merged.update(user_params)
    return name, merged


def _adaptive_scale(x_pre, x_new, atols):
    """Symmetric scaling vector for the step-acceptance metric:
    `scale_i = max(|x_pre_i|, |x_new_i|, atol_i)`.

    Using BOTH endpoints (not just `|x_pre|`) makes the metric well-behaved
    for variables that grow from zero: `z_osc(0) = 0 -> z_osc(dt) = -1.55`
    on a harmonic oscillator otherwise produces an "infinite" relative
    change.  With the symmetric scale the relative metric becomes the
    standard Hairer-Wanner step-error norm.
    """
    return np.maximum(np.maximum(np.abs(x_pre), np.abs(x_new)), atols)


def _try_step(model, dt, snap, relaxation, tol, max_iter, line_search=False):
    """Restore `snap`, attempt a single solve at `dt`, return `(succeeded, dt)`.

    Always restores BEFORE attempting so the caller can pass the same `snap`
    to multiple `_try_step` calls in the same retry loop without explicit
    bookkeeping.  Catches `NewtonConvergenceFailure` and returns False so the
    strategy can treat non-convergence as a rejection signal.
    """
    model._restore_state(snap)
    try:
        model.solve_dae_step(dt, relaxation=relaxation, tol=tol,
                             max_iter=max_iter, raise_on_no_convergence=True,
                             line_search=line_search)
        return True
    except NewtonConvergenceFailure:
        return False


def _strategy_derivative_limit(model, *, dt_target, dt_min, dt_max, grow, shrink,
                               max_retries, relaxation, tol, max_iter,
                               line_search, params):
    """(B) Reject when any active variable's per-step relative change exceeds
    `rel_tol`.  Cheapest strategy: zero extra implicit solves per accepted step.
    Best for problems where engineers can express physical limits naturally
    (e.g. "no variable should move more than 1% per step")."""
    rel_tol = params["rel_tol"]
    atol_global = params["atol"]
    atols = model._get_var_atols(atol_global)
    snap = model._snapshot_state()
    dt = max(dt_min, min(dt_target, dt_max))
    rejections = 0
    for _ in range(max_retries):
        ok = _try_step(model, dt, snap, relaxation, tol, max_iter, line_search)
        if not ok:
            rejections += 1
            dt = max(dt_min, dt * shrink)
            if dt <= dt_min:
                raise RuntimeError(
                    f"derivative_limit: hit dt_min={dt_min} after Newton "
                    f"non-convergence ({rejections} rejections)")
            continue
        x_new = model.get_vars_values()
        x_pre = snap["values"]
        scale = _adaptive_scale(x_pre, x_new, atols)
        metric = float(np.max(np.abs(x_new - x_pre) / scale))
        if metric <= rel_tol:
            # Accept; suggest a slightly larger dt next time if we're well below
            # the limit.  The hint lives on the model so the next call picks it up.
            if metric < 0.25 * rel_tol:
                dt_hint = min(dt * grow, dt_max)
            else:
                dt_hint = dt
            model._dt_hint = dt_hint
            return dt, {"strategy": "derivative_limit", "rejections": rejections,
                        "metric": metric,
                        "n_iters": model._last_solve_iters}
        rejections += 1
        dt = max(dt_min, dt * shrink)
        if dt <= dt_min:
            raise RuntimeError(
                f"derivative_limit: hit dt_min={dt_min} with metric={metric:.3e} "
                f"still above rel_tol={rel_tol}")
    raise RuntimeError(
        f"derivative_limit: exceeded {max_retries} retries (last metric={metric:.3e})")


def _strategy_predictor_corrector(model, *, dt_target, dt_min, dt_max, grow, shrink,
                                  max_retries, relaxation, tol, max_iter,
                                  line_search, params):
    """(P) Compare the implicit Crank-Nicolson result to a cheap explicit-Euler
    predictor `x_pred = x_pre + dt * der_pre` for every `DifferentialVariable`.
    The mismatch IS a calibrated O(dt^2) local-error estimator -- principled
    without paying for an extra implicit solve.

    Auto-falls back to `derivative_limit` if there are no `DifferentialVariable`s
    in the active set (the predictor would have nothing to compare to).
    """
    pairs = model._get_diff_var_index_pairs()
    if not pairs:
        return _strategy_derivative_limit(
            model, dt_target=dt_target, dt_min=dt_min, dt_max=dt_max,
            grow=grow, shrink=shrink, max_retries=max_retries,
            relaxation=relaxation, tol=tol, max_iter=max_iter,
            line_search=line_search,
            params={"rel_tol": params["tol_local"], "atol": params["atol"]})
    state_idx = np.array([p[0] for p in pairs])
    der_idx = np.array([p[1] for p in pairs])
    tol_local = params["tol_local"]
    atol_global = params["atol"]
    atols = model._get_var_atols(atol_global)
    snap = model._snapshot_state()
    der_pre = snap["values"][der_idx].copy()    # slope at start of step
    x_pre = snap["values"][state_idx].copy()
    atols_diff = atols[state_idx]
    dt = max(dt_min, min(dt_target, dt_max))
    rejections = 0
    for _ in range(max_retries):
        ok = _try_step(model, dt, snap, relaxation, tol, max_iter, line_search)
        if not ok:
            rejections += 1
            dt = max(dt_min, dt * shrink)
            if dt <= dt_min:
                raise RuntimeError(
                    f"predictor_corrector: hit dt_min={dt_min} after Newton "
                    f"non-convergence ({rejections} rejections)")
            continue
        x_new_full = model.get_vars_values()
        x_new_diff = x_new_full[state_idx]
        x_pred = x_pre + dt * der_pre
        scale = _adaptive_scale(x_pre, x_new_diff, atols_diff)
        metric = float(np.max(np.abs(x_new_diff - x_pred) / scale))
        if metric <= tol_local:
            dt_hint = min(dt * grow, dt_max) if metric < 0.25 * tol_local else dt
            model._dt_hint = dt_hint
            return dt, {"strategy": "predictor_corrector",
                        "rejections": rejections, "metric": metric,
                        "n_iters": model._last_solve_iters}
        rejections += 1
        dt = max(dt_min, dt * shrink)
        if dt <= dt_min:
            raise RuntimeError(
                f"predictor_corrector: hit dt_min={dt_min} with "
                f"metric={metric:.3e} still above tol_local={tol_local}")
    raise RuntimeError(
        f"predictor_corrector: exceeded {max_retries} retries "
        f"(last metric={metric:.3e})")


def _strategy_richardson(model, *, dt_target, dt_min, dt_max, grow, shrink,
                         max_retries, relaxation, tol, max_iter,
                         line_search, params):
    """(R) Step-doubling: take one full step at `dt`, restore and take two
    half-steps at `dt/2`, compare.  Costs 3 implicit solves per accepted step
    (vs 1 for the simpler strategies) but gives a properly-calibrated
    `O(dt^(p+1))` local-error estimate, so the controller uses the standard
    `dt_new = dt * safety * (tol/err)^(1/(p+1))` formula instead of a fixed
    grow/shrink ratio.

    Commits the HALF-STEP result (more accurate of the two), so an accepted
    Richardson step has the same accuracy as a fixed-`dt/2` run.
    """
    tol_local = params["tol_local"]
    atol_global = params["atol"]
    safety = params["safety"]
    p = params["order"]                         # CN is order 2
    atols = model._get_var_atols(atol_global)
    snap = model._snapshot_state()
    dt = max(dt_min, min(dt_target, dt_max))
    rejections = 0
    for _ in range(max_retries):
        # Full step at dt
        ok_full = _try_step(model, dt, snap, relaxation, tol, max_iter, line_search)
        if not ok_full:
            rejections += 1
            dt = max(dt_min, dt * shrink)
            if dt <= dt_min:
                raise RuntimeError(
                    f"richardson: hit dt_min={dt_min} (full-step Newton "
                    f"non-convergence after {rejections} rejections)")
            continue
        x_full = model.get_vars_values().copy()
        # Two half-steps at dt/2 from the same snapshot.  After the first
        # half-step we DON'T call next_step() -- we just reuse the existing
        # state as the prev_values for the second half.  This matches what
        # an outer fixed-dt/2 loop would do across two `solve_dae_step` calls.
        ok1 = _try_step(model, dt / 2, snap, relaxation, tol, max_iter, line_search)
        if not ok1:
            rejections += 1
            dt = max(dt_min, dt * shrink)
            continue
        # Manually advance prev_values to current values for the second half.
        mid_snap = model._snapshot_state()
        model.set_prev_vars_values(mid_snap["values"])
        model.set_t_prev(mid_snap["t"])
        ok2 = False
        try:
            model.solve_dae_step(dt / 2, relaxation=relaxation, tol=tol,
                                 max_iter=max_iter, raise_on_no_convergence=True,
                                 line_search=line_search)
            ok2 = True
        except NewtonConvergenceFailure:
            pass
        if not ok2:
            rejections += 1
            dt = max(dt_min, dt * shrink)
            continue
        x_half_half = model.get_vars_values().copy()
        # Error estimate
        scale = _adaptive_scale(x_full, x_half_half, atols)
        err = float(np.max(np.abs(x_full - x_half_half) / scale))
        # Standard Richardson controller for an order-p method
        if err > 0:
            ratio = (tol_local / err) ** (1.0 / (p + 1))
        else:
            ratio = grow
        dt_next = max(dt_min, min(dt_max, dt * safety * ratio))
        if err <= tol_local:
            # Accept the half-step result (already in `model`).  Restore the
            # CN bookkeeping: prev_values must be the ORIGINAL pre-step state
            # (before we did any half-stepping), not the mid-step value.
            model.set_prev_vars_values(snap["prev_values"])
            model.set_t_prev(snap["t_prev"])
            model._dt_hint = dt_next
            return dt, {"strategy": "richardson", "rejections": rejections,
                        "metric": err, "n_iters": model._last_solve_iters}
        rejections += 1
        dt = max(dt_min, dt_next)              # use the controller's hint
        if dt <= dt_min:
            raise RuntimeError(
                f"richardson: hit dt_min={dt_min} with err={err:.3e} "
                f"still above tol_local={tol_local}")
    raise RuntimeError(
        f"richardson: exceeded {max_retries} retries (last err={err:.3e})")


def _strategy_tr_bdf2(model, *, dt_target, dt_min, dt_max, grow, shrink,
                      max_retries, relaxation, tol, max_iter,
                      line_search, params):
    """(T) TR-BDF2: L-stable, stiffly-accurate, second-order one-step method.

    Each accepted step is two implicit sub-solves (a trapezoidal stage to
    `gamma*dt` then a BDF2 stage to `dt`); the method's built-in embedded
    estimate gives a calibrated `O(dt^3)` local-error norm, so the controller
    uses the standard `dt * safety * (tol/err)^(1/(p+1))` formula (`p=2`).

    Unlike Crank-Nicolson (used by the other strategies), TR-BDF2 damps
    high-frequency stiff modes monotonically (L-stability) instead of ringing,
    which makes it the robust default for stiff or sharply-forced transients.
    Requires at least one `DifferentialVariable`.
    """
    diff_state_idx = model._get_diff_state_indices()
    if diff_state_idx.size == 0:
        raise RuntimeError(
            "tr_bdf2 requires at least one DifferentialVariable in the active "
            "set; use 'predictor_corrector' or 'richardson' for purely "
            "algebraic / derivative-limited control")
    pairs = model._get_diff_var_index_pairs()
    pair_state_idx = np.array([p[0] for p in pairs], dtype=int)
    pair_der_idx = np.array([p[1] for p in pairs], dtype=int)
    tol_local = params["tol_local"]
    atol_global = params["atol"]
    safety = params["safety"]
    p = params["order"]                         # TR-BDF2 is order 2
    atols = model._get_var_atols(atol_global)
    snap = model._snapshot_state()
    dt = max(dt_min, min(dt_target, dt_max))
    rejections = 0
    err = float("inf")
    try:
        for _ in range(max_retries):
            try:
                est, est_idx = model._tr_bdf2_step(
                    dt, snap, diff_state_idx, pair_state_idx, pair_der_idx,
                    relaxation, tol, max_iter, line_search)
            except NewtonConvergenceFailure:
                rejections += 1
                dt = max(dt_min, dt * shrink)
                if dt <= dt_min:
                    raise RuntimeError(
                        f"tr_bdf2: hit dt_min={dt_min} after Newton "
                        f"non-convergence ({rejections} rejections)")
                continue
            x_pre = snap["values"][est_idx]
            x_new = model.get_vars_values()[est_idx]
            scale = _adaptive_scale(x_pre, x_new, atols[est_idx])
            err = float(np.max(np.abs(est) / scale))
            ratio = (tol_local / err) ** (1.0 / (p + 1)) if err > 0 else grow
            dt_next = max(dt_min, min(dt_max, dt * safety * ratio))
            if err <= tol_local:
                # Accept.  Restore CN bookkeeping: prev_values must be the
                # ORIGINAL pre-step state (the TR-BDF2 stage 2 overwrote the
                # der prev slot with x_gamma).  `iter_run`/`run` then call
                # `next_step()` to advance prev_values to the new state.
                model.set_prev_vars_values(snap["prev_values"])
                model.set_t_prev(snap["t_prev"])
                model._dt_hint = dt_next
                return dt, {"strategy": "tr_bdf2", "rejections": rejections,
                            "metric": err, "n_iters": model._last_solve_iters}
            rejections += 1
            dt = max(dt_min, dt_next)           # use the controller's hint
            if dt <= dt_min:
                raise RuntimeError(
                    f"tr_bdf2: hit dt_min={dt_min} with err={err:.3e} "
                    f"still above tol_local={tol_local}")
        raise RuntimeError(
            f"tr_bdf2: exceeded {max_retries} retries (last err={err:.3e})")
    finally:
        # Always leave the model on the default Crank-Nicolson closure so any
        # later solve (other strategies, `initialise`, fixed steps) is well-posed.
        model.set_scheme_coeffs(*_CN_COEFFS)


_ADAPTIVE_STRATEGIES = {
    "derivative_limit":    _strategy_derivative_limit,
    "predictor_corrector": _strategy_predictor_corrector,
    "richardson":          _strategy_richardson,
    "tr_bdf2":             _strategy_tr_bdf2,
}


class Parameter(Model):
    """Compile-time-known scalar that is passed to the lambdified residual but never solved for.

    Pass a Python scalar (`float`, `int`, ...) to declare a fresh Parameter
    owned by this `Model`.  Alternatively, pass an EXISTING `Parameter`
    instance owned by another `Model` to declare an *alias* into it: the
    constructor short-circuits via `__new__` and returns a
    `ParameterAlias` that transparently forwards `.symbol`, `.value`, and
    `.unit` to the underlying target.  This lets a child component
    reference a Parameter owned by its parent (e.g. all `TwoPortSegment`s
    of a `StraightPipe` referencing the same `A` symbol) without the
    component author having to special-case the float-vs-Parameter
    dispatch -- everything looks idiomatic at the call site:

        self.add_component('A_in', Parameter(self.A_in, "m^2"))

    works regardless of whether `self.A_in` is a `float` or a `Parameter`.
    """

    def __new__(cls, value=None, unit=None, description=None):
        # Unwrap alias chains so we never build alias-to-alias references --
        # the target is always a concrete `Parameter`.
        while isinstance(value, ParameterAlias):
            value = value._target
        if isinstance(value, Parameter):
            # Returning an instance whose class is NOT `cls` causes Python
            # to skip the subsequent `__init__` call, so the alias is not
            # re-initialised as a real Parameter.
            return ParameterAlias(value)
        return super().__new__(cls)

    def __init__(self, value, unit=None, description=None):
        self.components = {}
        self.symbol = None
        self.value = value
        self.unit = unit
        #: Optional human-readable description (authored once via the owning
        #: component's `PARAMS` spec; see `hydrogen.paramspec`).
        self.description = description
        self.can_evaluate = False
        # Filled in by the top-level Model at `instantiate()` so `set_value`
        # can write straight into the live solver buffer (a single slot)
        # instead of re-pushing the whole parameter vector.  `None` until
        # bound (e.g. before instantiate, or for an unused Parameter).
        self._value_array = None
        self._value_index = None

    def bind_value_slot(self, value_array, index):
        """Wire this Parameter to its slot in the owning model's `values` array.

        Called once per `instantiate()` for every Parameter that survives into
        the residual.  After binding, `set_value` keeps `self.value` and the
        live solver buffer in sync.
        """
        self._value_array = value_array
        self._value_index = index

    def set_value(self, value):
        """Update this Parameter and push it straight into the solver buffer.

        Equivalent to assigning `.value` *and* refreshing only this parameter's
        slot, so the next solve sees the new value without re-pushing the whole
        param vector.  Before `instantiate()` (unbound) it just records the
        value, which `initialise()` later snapshots into the buffer.
        """
        self.value = value
        if self._value_array is not None:
            self._value_array[self._value_index] = value

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


class ParameterAlias:
    """Transparent alias to a `Parameter` owned by another `Model`.

    Instances are produced automatically by `Parameter(value, ...)` when
    the supplied `value` is itself a `Parameter` -- see the dispatch in
    `Parameter.__new__`.  An alias exposes the read-only surface
    (`symbol`, `value`, `unit`) that `declare_equations()` actually
    consumes, plus the structural hooks that the framework uses for
    composition (`set_name`, `is_composite`, `get_vars_references`,
    `collect_equations`).  Critically, an alias is NOT a `Parameter`
    subclass, so the framework's structural walks (`assign_symbols`,
    `get_vars_references`, `_build_sym_paths`) can skip it cheaply via
    an `isinstance(..., ParameterAlias)` check -- the underlying target
    is reached via the OWNER's component tree exactly once and its
    symbol gets assigned there.

    Two `TwoPortSegment`s of the same `StraightPipe` end up holding two
    separate `ParameterAlias` objects, both pointing at the parent's
    single `A` Parameter.  Reading `self['A_in'].symbol` from either
    segment returns the SAME SymPy symbol, which is exactly what the
    duplicate-equation pass needs to collapse the per-face `m_dot =
    rho * A * w` closures across interior interfaces.
    """

    __slots__ = ('_target', 'name')

    def __init__(self, target):
        self._target = target
        self.name = None

    # --- forwarded attributes -----------------------------------------

    @property
    def symbol(self):
        return self._target.symbol

    @property
    def value(self):
        return self._target.value

    @property
    def unit(self):
        return getattr(self._target, 'unit', None)

    @property
    def description(self):
        return getattr(self._target, 'description', None)

    @property
    def target(self):
        return self._target

    def set_value(self, value):
        # Aliases never own a slot; forward to the concrete target, which is
        # the Parameter that actually lives in the param block.
        self._target.set_value(value)

    # --- composition hooks expected by the framework -------------------

    def set_name(self, name):
        self.name = name

    def is_composite(self):
        return False

    def get_vars_references(self):
        # The owner's component tree accounts for the target Parameter;
        # aliases contribute nothing of their own.
        return [], []

    def get_vars_len(self):
        return 0

    def check(self):
        return True

    def get_equations(self):
        return []

    def collect_equations(self):
        return []

    def collect_connections(self):
        return []

    def setup_t_values_referece(self, t_values):
        pass

    def __repr__(self):
        return f"ParameterAlias(target={self._target!r})"


class Variable(Model):
    """Algebraic unknown; a Newton solve adjusts it to satisfy the system's residuals.

    `atol` is an optional per-variable absolute tolerance consulted by
    `Model.solve_adaptive_step` when computing the step-acceptance metric.
    Leave as `None` to inherit the strategy's global `atol` (typical case);
    set explicitly only when one variable's units make the global value
    inappropriate -- e.g. mass flow `kg/s` (~1e-3) needs a much smaller
    `atol` than pressure `Pa` (~1e5).  See "Performance & tuning" in the
    README for guidance.

    `scale` is an optional per-variable typical magnitude.  When set (or
    when `enable_var_scaling=True` is passed to `instantiate()` and the
    initial value can be used as a proxy), the Newton convergence metric
    becomes `||delta / scale||_2` rather than the raw `||delta||_2`,
    preventing fast-converging large-magnitude variables (pressure ~1e5)
    from masking unconverged small-magnitude variables (mass flow ~1e-3).
    Leave as `None` to inherit `max(|initial_value|, 1.0)` automatically.
    """

    def __init__(self, value, unit=None, atol=None, scale=None):
        self.components = {}
        self.value = value
        self._symbold_index = None
        self.symbol = None
        self.prev_value = value
        self.prev_symbol = None
        self.unit = unit
        self.atol = atol
        self.scale = scale
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

    def __init__(self, value, unit=None, atol=None, scale=None):
        super().__init__(value, unit, atol=atol, scale=scale)
        self.der_variable = Variable(0.0, None)

    def declare_equations(self):
        # Generalised one-step closure parameterised by the four global scheme
        # coefficients (`sch_p0, sch_p1, sch_a, sch_b`):
        #
        #     x = p0 * x_prev + dt * a * der + (p1 + dt * b) * der_prev
        #
        # This single compiled residual covers every supported integration
        # method by changing only the runtime coefficient VALUES (and, for
        # multi-stage methods, the value carried in `der_prev`):
        #
        #   Crank-Nicolson : (p0,p1,a,b) = (1, 0, 1/2, 1/2)
        #   implicit Euler : (p0,p1,a,b) = (1, 0,   1,   0)
        #   TR-BDF2 stage 1: (p0,p1,a,b) = (1, 0, 1/2, 1/2) with dt -> gamma*dt
        #   TR-BDF2 stage 2: (p0,p1,a,b) = (c2, c1, d, 0), der_prev := x_gamma
        #
        # The TR-BDF2 BDF2 stage `x = c2*x_prev + c1*x_gamma + d*dt*der` reuses
        # the (otherwise-unused-in-that-stage) `der_prev` slot to carry the
        # trapezoidal stage value `x_gamma`; that keeps the closure to a single
        # extra-history slot without growing the state vector.
        x       = self.symbol
        x_prev  = self.prev_symbol
        der     = self.der_variable.symbol
        der_prev = self.der_variable.prev_symbol
        eq1 = x - (
            self.sch_p0 * x_prev
            + self.dt * self.sch_a * der
            + (self.sch_p1 + self.dt * self.sch_b) * der_prev
        )
        return [eq1]

    def get_derivative_variable(self):
        return self.der_variable


class Input(Model):
    """Time-dependent driving signal `u(t)` that is supplied, not solved for.

    An `Input` sits alongside `Parameter` in a `Model`'s component list, but
    where a `Parameter` is a single compile-time constant, an `Input` is a
    *known function of time* that the framework re-evaluates as the
    integrator advances.  It is NOT a `Variable`: it never enters the Jacobian
    and the Newton solve never touches it.

    Crucially -- and unlike a `Parameter` -- an `Input` carries TWO values at
    once so it can appear on either side of a Crank-Nicolson balance:

        * `self['u'].symbol`       -> the value at the new level   u(t_{k+1})
        * `self['u'].prev_symbol`  -> the value at the old level   u(t_k)

    This lets a `DifferentialVariable`'s RHS depend on the input correctly:
    the trapezoidal closure `x = x_prev + 0.5*dt*(der_x + der_x_prev)` already
    pairs `der_x` (built from `u.symbol`) with `der_x_prev` (built from
    `u.prev_symbol`), so the source term is integrated at full second-order
    accuracy without the user managing any history.

    The signal is given as a Python callable `func(t) -> float`.  The
    framework calls it automatically whenever the time level changes (each
    solve step, at `initialise`, and on adaptive-step restore); user code
    never sets the value by hand.

    Implementation note: an `Input` is realised as a two-leaf composite of
    ordinary `Parameter`s (`cur` and `prev`).  That makes it ride the
    existing symbol-assignment, reference-collection and lambdify-argument
    plumbing for free -- the two leaves simply live in the parameter block
    of the state vector, and `Model._refresh_inputs` rewrites their two slots
    in place from `func` every time `t`/`t_prev` move.

    Example
    -------
        class HeatedMass(Model):
            def declare_components(self):
                self.add_component('T', DifferentialVariable(300.0, "K"))
                self.add_component('C', Parameter(500.0, "J/K"))
                # ambient temperature ramp, a known driver:
                self.add_component('T_amb', Input(lambda t: 300.0 + 0.5 * t, "K"))
                self.add_component('G', Parameter(2.0, "W/K"))

            def declare_equations(self):
                der_T = self['der_T'].symbol
                T     = self['T'].symbol
                C, G  = self['C'].symbol, self['G'].symbol
                T_amb = self['T_amb'].symbol        # u(t_{k+1})
                return [C * der_T - G * (T_amb - T)]
    """

    def __init__(self, func, unit=None, value=None):
        if not callable(func):
            raise TypeError("Input(func, ...) expects a callable t -> float")
        self.func = func
        self._unit = unit
        if value is None:
            v0 = float(func(0.0))
        else:
            v0 = float(value)
        self._init_value = v0
        super().__init__()

    def declare_components(self):
        # `cur` holds u(t_{k+1}); `prev` holds u(t_k).  Both are real
        # Parameters (floats, never aliases) so they slot into the param
        # block exactly like any other compile-time scalar.
        self.add_component('cur', Parameter(self._init_value, self._unit))
        self.add_component('prev', Parameter(self._init_value, self._unit))

    # --- read-only surface consumed by declare_equations -------------------

    @property
    def symbol(self):
        """SymPy symbol carrying the value at the new time level u(t_{k+1})."""
        return self.components['cur'].symbol

    @property
    def prev_symbol(self):
        """SymPy symbol carrying the value at the old time level u(t_k)."""
        return self.components['prev'].symbol

    @property
    def value(self):
        return self.components['cur'].value

    @property
    def prev_value(self):
        return self.components['prev'].value

    @property
    def unit(self):
        return self._unit

    def evaluate(self, t):
        """Sample the underlying signal at time `t` (handy for assertions/plots)."""
        return float(self.func(t))

    def __repr__(self):
        unit_str = f" {self._unit}" if self._unit else ""
        name = getattr(self, 'name', None) or 'Input'
        return f"Input({name}){unit_str}"
