"""Numerical primitives and lambdify compatibility shim."""

from __future__ import annotations

import inspect

import numba
import numpy as np
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla
import sympy as sp
from sympy.printing.numpy import NumPyPrinter as _NumPyPrinter

# Gravitational acceleration used by fluid components for buoyancy terms.
G_const = 9.81


def smooth_max(a, b, eps):
    """Smooth (C-infinity) approximation of ``max(a, b)``.

    ``0.5 * (a + b + sqrt((a - b)**2 + eps**2))`` -- a hyperbola that rounds the
    kink of a hard ``max`` over a blend width ``~eps`` around ``a == b``, so the
    result stays differentiable and Newton keeps a continuous Jacobian.  The
    bias is at most ``eps/2`` (at ``a == b``) and decays to ~0 more than a few
    ``eps`` away from the corner; ``smooth_max(a, b, eps) >= max(a, b)`` always.

    ``eps`` MUST be in the SAME units / magnitude as ``a`` and ``b`` (e.g. Pa for
    pressures, or a small fraction of a signal's range) -- there is no sensible
    default, so it is a required argument.  Intended for building symbolic
    (sympy) residual expressions.
    """
    return 0.5 * (a + b + sp.sqrt((a - b) ** 2 + eps ** 2))


def smooth_min(a, b, eps):
    """Smooth (C-infinity) approximation of ``min(a, b)``; see :func:`smooth_max`.

    ``0.5 * (a + b - sqrt((a - b)**2 + eps**2))``.  ``smooth_min`` is bounded
    ABOVE by ``min(a, b)`` (bias at most ``-eps/2`` at ``a == b``).  ``eps`` must
    match the units / magnitude of ``a`` and ``b``.
    """
    return 0.5 * (a + b - sp.sqrt((a - b) ** 2 + eps ** 2))


def lambdify_compat(args, expr, modules=None, cse=True, docstring_limit=-1,
                    printer=None):
    """Wrapper around `sympy.lambdify` that tolerates older sympy versions.

    Older sympy releases don't accept `cse` or `docstring_limit` keywords. We probe the
    signature once and only forward the kwargs the installed version actually supports.
    """
    supported_kwargs = inspect.signature(sp.lambdify).parameters
    kwargs = {"modules": modules}
    if "cse" in supported_kwargs:
        kwargs["cse"] = cse
    if "docstring_limit" in supported_kwargs:
        kwargs["docstring_limit"] = docstring_limit
    if printer is not None:
        kwargs["printer"] = printer
    return sp.lambdify(args, expr, **kwargs)


class NumbaFriendlyPrinter(_NumPyPrinter):
    """NumPy code printer emitting only constructs numba's nopython mode
    supports.

    The stock `NumPyPrinter` prints `Min`/`Max` as ``reduce(numpy.minimum,
    [...])`` (numba: no `functools.reduce`) and `Piecewise` as
    ``numpy.select([...], [...], default=nan)`` (numba: arrays-only, no
    scalar branches).  Both have exact rewrites as nested binary ufunc /
    ``numpy.where`` calls, which numba compiles fine and which evaluate to
    the identical values (`select` also evaluates all branches).
    """

    def _print_Min(self, expr):
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        out = self._print(expr.args[-1])
        fn = self._module_format(self._module + ".minimum")
        for a in reversed(expr.args[:-1]):
            out = f"{fn}({self._print(a)}, {out})"
        return out

    def _print_Max(self, expr):
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        out = self._print(expr.args[-1])
        fn = self._module_format(self._module + ".maximum")
        for a in reversed(expr.args[:-1]):
            out = f"{fn}({self._print(a)}, {out})"
        return out

    def _print_Piecewise(self, expr):
        pairs = expr.args
        if pairs[-1].cond == sp.true:
            out = self._print(pairs[-1].expr)
            pairs = pairs[:-1]
        else:
            out = self._module_format(self._module + ".nan")
        fn = self._module_format(self._module + ".where")
        for arg in reversed(pairs):
            out = (f"{fn}({self._print(arg.cond)}, "
                   f"{self._print(arg.expr)}, {out})")
        return out


@numba.jit(nopython=True)
def fast_error_norm(vars):
    """Numba-compiled `||vars||_2`, used as the Newton convergence metric."""
    return np.linalg.norm(vars)


@numba.jit(nopython=True)
def fast_linear_solve(A, b):
    """Numba-compiled dense linear solve, used in each Newton iteration."""
    return np.linalg.solve(A, b)


def _equilibrated_splu_solve(A, b):
    """Solve ``A x = b`` (A a CSC matrix) after two-sided infinity-norm
    equilibration.

    Thermofluid Jacobians mix wildly different physical scales in one system:
    pressures ~1e5 Pa, enthalpies ~1e6 J/kg, densities ~1e0..1e3, velocities
    ~1e-2 m/s and viscosities ~1e-4 Pa*s.  Unscaled, the condition number can
    reach ~1e14 (dominated by an all-pressures near-null mode), at which point
    SuperLU reports a *spurious* "Factor is exactly singular" -- a below-
    threshold pivot -- even though the system is perfectly solvable.  This is
    especially easy to trip inside the two-phase dome, where the density (and
    hence the momentum/continuity coupling) swings 100x across a few cells.

    A single pass of row- then column- max-norm scaling
    ``(Dr A Dc) y = Dr b,  x = Dc y`` brings the condition number down by ~9
    orders of magnitude (measured ~3.5e14 -> ~1.9e5) at O(nnz) cost, negligible
    beside the factorisation itself, and never changes the true solution.
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    n = A.shape[0]
    row_idx = A.indices                       # CSC: row index of each nonzero
    col_idx = np.repeat(np.arange(n, dtype=np.intp), np.diff(A.indptr))
    data = A.data
    absd = np.abs(data)
    r_max = np.zeros(n)
    np.maximum.at(r_max, row_idx, absd)
    r_max[r_max == 0.0] = 1.0
    dr = 1.0 / r_max
    data = data * dr[row_idx]
    absd = np.abs(data)
    c_max = np.zeros(n)
    np.maximum.at(c_max, col_idx, absd)
    c_max[c_max == 0.0] = 1.0
    dc = 1.0 / c_max
    data = data * dc[col_idx]
    A_s = _sp.csc_matrix((data, A.indices, A.indptr), shape=A.shape, copy=False)
    y = _spla.splu(A_s).solve(dr * b)
    return dc * y


def fast_sparse_solve(values, rows, cols, shape, b):
    """SuperLU sparse linear solve used when a sparse Jacobian is available.

    `values`/`rows`/`cols` are the COO triplets emitted by the sparse Jacobian
    evaluator; we build a CSC matrix in place (CSC is what SuperLU expects) and
    call `splu` (after equilibration; see `_equilibrated_splu_solve`).  Reusing
    `splu` is non-trivial because the symbolic factorisation depends on the
    value pattern, so we re-factorise per Newton iteration -- still much cheaper
    than a dense solve once the Jacobian is significantly sparser than ~10%.
    """
    A = _sp.csc_matrix((values, (rows, cols)), shape=shape)
    return _equilibrated_splu_solve(A, b)


def precompute_csc_pattern(rows, cols, shape):
    """Precompute the COO->CSC reordering for a fixed-sparsity Jacobian.

    Returns `(perm, indices, indptr)` such that, given a fresh `values`
    array aligned with the original `(rows, cols)` triplets,
        data_csc = values[perm]
        csc_matrix((data_csc, indices, indptr), shape, copy=False)
    builds a CSC matrix WITHOUT re-sorting or scanning for duplicate
    entries -- both expensive `csc_matrix(...)` constructor steps the
    sparse Jacobian evaluator was paying every Newton iteration.

    Assumes the (rows, cols) pattern has no duplicates (true for our
    sparse Jacobian, which emits each non-zero exactly once).  This pass
    is run once at the end of `instantiate()` and cached on the model.
    """
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    cols = np.ascontiguousarray(cols, dtype=np.int32)
    n_row, n_col = shape
    nnz = rows.size
    # Build indptr by counting column occupancies, then prefix-sum.
    indptr = np.zeros(n_col + 1, dtype=np.int32)
    np.add.at(indptr[1:], cols, 1)
    np.cumsum(indptr, out=indptr)
    # Permutation: stable sort by (col, row) so that for each column,
    # the row indices come out in ascending order (CSC canonical form).
    perm = np.lexsort((rows, cols)).astype(np.int32)
    indices = rows[perm]
    return perm, indices, indptr


def fast_sparse_solve_cached(values, perm, indices, indptr, shape, b):
    """Sparse linear solve that reuses a cached CSC structure.

    `perm`, `indices`, `indptr` come from `precompute_csc_pattern`.
    The CSC constructor here skips all sorting/dedup -- about a 3-5x
    speed-up over `fast_sparse_solve` for the construct step (the splu
    cost itself is unchanged).
    """
    data = values[perm]
    A = _sp.csc_matrix((data, indices, indptr), shape=shape, copy=False)
    return _equilibrated_splu_solve(A, b)
