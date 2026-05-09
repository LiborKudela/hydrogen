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
