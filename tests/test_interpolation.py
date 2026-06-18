"""Tests for the symbolic interpolation utilities.

Covered:

  * 1D piecewise-linear values, node-exactness, segment slopes;
  * 2D bilinear values, node-exactness, both partial derivatives;
  * both extrapolation policies (``constant`` -> flat shelf, zero slope;
    ``linear`` -> straight ramp, edge slope) in 1D and 2D;
  * the symbolic surface: ``sympy.diff`` of an interpolated expression matches
    the analytic derivative interpolator (this is what the Newton Jacobian
    relies on);
  * an end-to-end solve where a model residual contains an interpolated term,
    proving eval + derivative flow correctly through ``instantiate`` /
    ``lambdify`` / the Newton solve.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from hydrogen.model import Model, Variable
from hydrogen.utilities import Interpolation1D, Interpolation2D


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------

X1 = [0.0, 1.0, 2.0, 4.0]
Y1 = [0.0, 2.0, 6.0, 6.0]   # slopes: 2, 4, 0


def test_1d_is_node_exact():
    f = Interpolation1D(X1, Y1)
    for x, y in zip(X1, Y1):
        assert f.eval(x) == pytest.approx(y)


def test_1d_linear_within_segments():
    f = Interpolation1D(X1, Y1)
    assert f.eval(0.5) == pytest.approx(1.0)    # slope 2 from (0,0)
    assert f.eval(1.5) == pytest.approx(4.0)    # slope 4 from (1,2)
    assert f.eval(3.0) == pytest.approx(6.0)    # slope 0 flat top


def test_1d_segment_derivatives():
    f = Interpolation1D(X1, Y1)
    assert f.derivative(0.5) == pytest.approx(2.0)
    assert f.derivative(1.5) == pytest.approx(4.0)
    assert f.derivative(3.0) == pytest.approx(0.0)


def test_1d_extrapolate_constant():
    f = Interpolation1D(X1, Y1, extrapolate="constant")
    assert f.eval(-5.0) == pytest.approx(Y1[0])
    assert f.eval(99.0) == pytest.approx(Y1[-1])
    assert f.derivative(-5.0) == pytest.approx(0.0)
    assert f.derivative(99.0) == pytest.approx(0.0)


def test_1d_extrapolate_linear():
    f = Interpolation1D(X1, Y1, extrapolate="linear")
    # below: continue first segment (slope 2) -> y = 0 + 2*(x-0)
    assert f.eval(-1.0) == pytest.approx(-2.0)
    assert f.derivative(-1.0) == pytest.approx(2.0)
    # above: continue last segment (slope 0) -> stays at 6
    assert f.eval(10.0) == pytest.approx(6.0)
    assert f.derivative(10.0) == pytest.approx(0.0)


def test_1d_vectorised_eval():
    f = Interpolation1D(X1, Y1)
    out = f.eval(np.array([0.5, 1.5, 3.0]))
    assert np.allclose(out, [1.0, 4.0, 6.0])


def test_1d_symbolic_derivative_matches_analytic():
    f = Interpolation1D(X1, Y1, extrapolate="linear")
    x = sp.Symbol("x")
    d_expr = sp.diff(f(x), x)
    d_fn = sp.lambdify(x, d_expr, modules=["numpy"] + f.modules)
    for xq in (-1.0, 0.5, 1.5, 3.0, 10.0):
        assert d_fn(xq) == pytest.approx(f.derivative(xq))


def test_1d_invalid_inputs():
    with pytest.raises(ValueError):
        Interpolation1D([0.0], [1.0])                       # too few
    with pytest.raises(ValueError):
        Interpolation1D([0.0, 0.0], [1.0, 2.0])             # not increasing
    with pytest.raises(ValueError):
        Interpolation1D([0.0, 1.0], [1.0], )                # length mismatch
    with pytest.raises(ValueError):
        Interpolation1D(X1, Y1, extrapolate="spline")       # bad policy


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

X2 = [0.0, 1.0, 2.0]
Y2 = [0.0, 2.0]
# z = x + 3*y  -> exactly representable by bilinear, slopes d/dx=1, d/dy=3
Z2 = [[0.0, 6.0],
      [1.0, 7.0],
      [2.0, 8.0]]


def test_2d_is_node_exact():
    g = Interpolation2D(X2, Y2, Z2)
    for i, x in enumerate(X2):
        for j, y in enumerate(Y2):
            assert g.eval(x, y) == pytest.approx(Z2[i][j])


def test_2d_bilinear_interior():
    g = Interpolation2D(X2, Y2, Z2)
    # plane z = x + 3y
    assert g.eval(0.5, 1.0) == pytest.approx(0.5 + 3.0)
    assert g.eval(1.5, 0.5) == pytest.approx(1.5 + 1.5)


def test_2d_partial_derivatives():
    g = Interpolation2D(X2, Y2, Z2)
    assert g.derivative(0.5, 1.0, axis=0) == pytest.approx(1.0)   # d/dx
    assert g.derivative(0.5, 1.0, axis=1) == pytest.approx(3.0)   # d/dy


def test_2d_extrapolate_constant():
    g = Interpolation2D(X2, Y2, Z2, extrapolate="constant")
    # both axes out of range -> clamp to nearest corner z[0,0]=0
    assert g.eval(-1.0, -1.0) == pytest.approx(0.0)
    # far corner clamp -> z[-1,-1] = 8
    assert g.eval(5.0, 5.0) == pytest.approx(8.0)
    # x out of range, y in range: value clamps x but still varies with y
    assert g.eval(-1.0, 1.0) == pytest.approx(0.0 + 3.0)
    # derivative is zero only in the out-of-range direction
    assert g.derivative(-1.0, 1.0, axis=0) == pytest.approx(0.0)
    assert g.derivative(-1.0, 1.0, axis=1) == pytest.approx(3.0)


def test_2d_extrapolate_linear():
    g = Interpolation2D(X2, Y2, Z2, extrapolate="linear")
    # plane continues: z = x + 3y everywhere
    assert g.eval(-1.0, -1.0) == pytest.approx(-1.0 - 3.0)
    assert g.eval(5.0, 5.0) == pytest.approx(5.0 + 15.0)
    assert g.derivative(-1.0, -1.0, axis=0) == pytest.approx(1.0)
    assert g.derivative(5.0, 5.0, axis=1) == pytest.approx(3.0)


def test_2d_symbolic_derivatives_match_analytic():
    g = Interpolation2D(X2, Y2, Z2, extrapolate="linear")
    x, y = sp.symbols("x y")
    dx_fn = sp.lambdify((x, y), sp.diff(g(x, y), x), modules=["numpy"] + g.modules)
    dy_fn = sp.lambdify((x, y), sp.diff(g(x, y), y), modules=["numpy"] + g.modules)
    for xq, yq in ((0.5, 1.0), (1.5, 0.5), (-1.0, 3.0)):
        assert dx_fn(xq, yq) == pytest.approx(g.derivative(xq, yq, axis=0))
        assert dy_fn(xq, yq) == pytest.approx(g.derivative(xq, yq, axis=1))


def test_2d_invalid_inputs():
    with pytest.raises(ValueError):
        Interpolation2D(X2, Y2, [[0.0, 1.0]])               # bad z shape
    with pytest.raises(ValueError):
        Interpolation2D([0.0], Y2, [[0.0, 1.0]])            # too few x


# ---------------------------------------------------------------------------
# End-to-end: an interpolated term inside a solved model residual
# ---------------------------------------------------------------------------

class _InterpRoot(Model):
    """Solve `f(x) = target` for `x`; exercises eval + Jacobian derivative
    through instantiate/lambdify/Newton."""

    def __init__(self, interp, target, x0):
        self._interp = interp
        self._target = target
        self._x0 = x0
        super().__init__()

    def declare_components(self):
        self.add_component('x', Variable(self._x0))

    def declare_equations(self):
        x = self['x'].symbol
        return [self._interp(x) - self._target]


def _solve_root(interp, target, x0):
    model = _InterpRoot(interp, target, x0)
    model.instantiate(aditional_modules=interp.modules, max_remove_trival_passes=3)
    model.initialise()
    names = list(model.record['vars_names'])
    state = np.asarray(model.record['state'])
    idx = next(i for i, n in enumerate(names) if n.endswith('.x'))
    return state[-1, idx]


def test_model_solve_with_interpolated_residual():
    # f piecewise linear, strictly increasing so the root is unique and the
    # Newton step needs the correct (nonzero) derivative to converge.
    f = Interpolation1D([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 6.0, 12.0])
    # target 4.0 lies in segment (1,2): y = 2 + 4*(x-1) -> x = 1.5
    x_sol = _solve_root(f, target=4.0, x0=1.2)
    assert x_sol == pytest.approx(1.5, rel=1e-8)
    assert f.eval(x_sol) == pytest.approx(4.0, rel=1e-8)
