"""Numerical primitives and lambdify compatibility shim."""

from __future__ import annotations

import inspect

import numba
import numpy as np
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla
import sympy as sp

# Gravitational acceleration used by fluid components for buoyancy terms.
G_const = 9.81


def lambdify_compat(args, expr, modules=None, cse=True, docstring_limit=-1):
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
    return sp.lambdify(args, expr, **kwargs)


@numba.jit(nopython=True)
def fast_error_norm(vars):
    """Numba-compiled `||vars||_2`, used as the Newton convergence metric."""
    return np.linalg.norm(vars)


@numba.jit(nopython=True)
def fast_linear_solve(A, b):
    """Numba-compiled dense linear solve, used in each Newton iteration."""
    return np.linalg.solve(A, b)


def fast_sparse_solve(values, rows, cols, shape, b):
    """SuperLU sparse linear solve used when a sparse Jacobian is available.

    `values`/`rows`/`cols` are the COO triplets emitted by the sparse Jacobian
    evaluator; we build a CSC matrix in place (CSC is what SuperLU expects) and
    call `splu`.  Reusing `splu` is non-trivial because the symbolic
    factorisation depends on the value pattern, so we re-factorise per Newton
    iteration -- still much cheaper than a dense solve once the Jacobian is
    significantly sparser than ~10%.
    """
    A = _sp.csc_matrix((values, (rows, cols)), shape=shape)
    return _spla.splu(A).solve(np.asarray(b).reshape(-1))


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
    return _spla.splu(A).solve(np.asarray(b).reshape(-1))
