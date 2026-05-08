"""Unit tests for `hydrogen.numerics`."""

from __future__ import annotations

import numpy as np
import sympy as sp

from hydrogen.numerics import G_const, fast_error_norm, fast_linear_solve, lambdify_compat


def test_g_const_is_standard_gravity():
    assert abs(G_const - 9.81) < 1e-9


def test_lambdify_compat_evaluates_correctly():
    x, y = sp.symbols('x y', real=True)
    f = lambdify_compat([x, y], x ** 2 + y, modules='numpy')
    assert f(3.0, 1.0) == 10.0


def test_lambdify_compat_handles_matrix():
    x, y = sp.symbols('x y', real=True)
    m = sp.Matrix([x + y, x - y])
    f = lambdify_compat([x, y], m, modules='numpy')
    out = np.asarray(f(2.0, 0.5)).reshape(-1)
    assert np.allclose(out, [2.5, 1.5])


def test_fast_error_norm_matches_numpy():
    a = np.array([3.0, 4.0])
    assert abs(fast_error_norm(a) - 5.0) < 1e-12


def test_fast_linear_solve_matches_numpy():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([4.0, 5.0])
    x = fast_linear_solve(A, b)
    assert np.allclose(A @ x, b, atol=1e-12)
