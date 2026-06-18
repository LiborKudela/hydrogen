"""Symbolic linear/bilinear interpolation tables.

These utilities expose lookup tables to the symbolic model layer exactly the
way :class:`hydrogen.medium.CoolPropMedium` exposes thermophysical properties:
as a ``sympy.Function`` whose

  * numeric evaluation defers to a fast NumPy interpolator, and
  * symbolic derivative (``fdiff``) defers to a matching derivative
    interpolator,

so an interpolated quantity can appear anywhere in a model's residual and the
Newton Jacobian picks up the correct partial derivatives automatically.

Two public classes are provided:

  * :class:`Interpolation1D` -- piecewise-linear ``y = f(x)`` from a 1D table.
  * :class:`Interpolation2D` -- bilinear ``z = f(x, y)`` from a 2D grid.

Both accept an ``extrapolate`` policy controlling behaviour outside the table:

  * ``"constant"`` -- clamp to the nearest edge value; the derivative is zero
    in any direction whose coordinate is out of range (a flat shelf).
  * ``"linear"``   -- continue the slope of the nearest edge segment; the
    derivative is the (constant) edge-segment slope (a straight ramp).

Usage
-----
    from hydrogen.utilities import Interpolation1D

    eta = Interpolation1D([0.0, 1.0, 2.0], [0.5, 0.8, 0.9],
                          extrapolate="constant")

    # inside Model.declare_equations:
    #     m_dot = self['m_dot'].symbol
    #     return [self['P'].symbol - eta(m_dot) * ref_power]

    # at instantiate time, hand the table's callbacks to lambdify:
    #     model.instantiate(aditional_modules=eta.modules)

Multiple tables (and tables alongside a medium) are combined by concatenating
their ``.modules`` lists::

    model.instantiate(aditional_modules=eta.modules + medium.modules)
"""

from __future__ import annotations

import hashlib

import numpy as np
import sympy as sp

__all__ = ["Interpolation1D", "Interpolation2D"]

_EXTRAPOLATION_MODES = ("constant", "linear")


def _make_symbolic_function(name, arg_names, eval_func, deriv_funcs):
    """Build a uniquely-named ``sympy.Function`` subclass backed by NumPy.

    Parameters
    ----------
    name : str
        Unique class name; this is also the key lambdify will look up in the
        ``modules`` namespace, so it must match the registered callback.
    arg_names : sequence[str]
        Names of the positional arguments, used to name derivative classes
        (``d{name}_d{arg}``) and to map ``argindex`` -> argument.
    eval_func : callable
        ``eval_func(*floats) -> float`` numeric evaluator.
    deriv_funcs : dict[int, callable]
        ``{argindex (1-based) -> derivative evaluator}``.  An empty dict marks
        a leaf (e.g. a first-derivative class), so differentiating it again
        raises ``NotImplementedError`` -- the Newton solve only needs first
        derivatives.
    """
    # Cache derivative classes so repeated `fdiff` calls (one per Jacobian
    # nonzero) reuse the same class object instead of rebuilding it.
    _deriv_cache: dict[int, type] = {}

    class _Interp(sp.Function):

        @classmethod
        def eval(cls, *args):
            # Fold to a number only when every argument is numeric; otherwise
            # stay symbolic so the expression survives until lambdify.
            if all(getattr(a, "is_number", False) for a in args):
                return sp.Float(float(eval_func(*[float(a) for a in args])))
            return None

        def fdiff(self, argindex=1):
            if argindex not in deriv_funcs:
                raise NotImplementedError(
                    f"Derivative of {name} w.r.t. argument {argindex} is not defined"
                )
            cls = _deriv_cache.get(argindex)
            if cls is None:
                wrt = arg_names[argindex - 1]
                cls = _make_symbolic_function(
                    f"d{name}_d{wrt}", arg_names, deriv_funcs[argindex], {}
                )
                _deriv_cache[argindex] = cls
            return cls(*self.args)

    return type(name, (_Interp,), {})


def _stable_suffix(*chunks) -> str:
    """Deterministic 8-hex-char id from the table contents.

    Keeping the symbolic-function name stable for identical data lets the
    on-disk lambdify source cache hit across processes/runs.
    """
    h = hashlib.md5()
    for c in chunks:
        h.update(np.ascontiguousarray(c, dtype=float).tobytes())
        h.update(b"|")
    return h.hexdigest()[:8]


class Interpolation1D:
    """Piecewise-linear lookup table ``y = f(x)`` usable as a symbolic function.

    Parameters
    ----------
    x : array-like
        Strictly increasing sample abscissae (length >= 2).
    y : array-like
        Sample ordinates, same length as ``x``.
    extrapolate : {"constant", "linear"}
        Out-of-range policy (see module docstring).
    name : str, optional
        Base name for the generated symbolic function; a content hash is
        appended to keep it unique and cache-stable.  Defaults to ``"interp1d"``.

    Notes
    -----
    Call the instance with a sympy expression to get the symbolic value
    (``f(expr)``); call :meth:`eval` / :meth:`derivative` for plain numbers.
    Pass :attr:`modules` to ``Model.instantiate(aditional_modules=...)``.
    """

    def __init__(self, x, y, extrapolate="constant", name=None):
        if extrapolate not in _EXTRAPOLATION_MODES:
            raise ValueError(
                f"extrapolate must be one of {_EXTRAPOLATION_MODES}, got {extrapolate!r}"
            )
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("x and y must be 1D")
        if xs.size < 2:
            raise ValueError("need at least two samples for interpolation")
        if xs.size != ys.size:
            raise ValueError(f"x and y length mismatch: {xs.size} vs {ys.size}")
        if not np.all(np.diff(xs) > 0):
            raise ValueError("x must be strictly increasing")

        self._xs = xs
        self._ys = ys
        self.extrapolate = extrapolate

        base = name or "interp1d"
        suffix = _stable_suffix(xs, ys, [_EXTRAPOLATION_MODES.index(extrapolate)])
        self.name = f"{base}_{suffix}"
        dname = f"d{self.name}_dx"

        self._func_class = _make_symbolic_function(
            self.name, ["x"], self._eval_value, {1: self._eval_deriv}
        )
        self.modules = [
            {self.name: self._eval_value},
            {dname: self._eval_deriv},
        ]

    # --- numeric evaluators (scalar; the solver vectorises them) -----------

    def _segment(self, x):
        n = self._xs.size
        i = int(np.searchsorted(self._xs, x, side="right")) - 1
        return min(max(i, 0), n - 2)

    def _eval_value(self, x):
        x = float(x)
        xs, ys = self._xs, self._ys
        if self.extrapolate == "constant":
            if x <= xs[0]:
                return float(ys[0])
            if x >= xs[-1]:
                return float(ys[-1])
        i = self._segment(x)
        slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
        return float(ys[i] + slope * (x - xs[i]))

    def _eval_deriv(self, x):
        x = float(x)
        xs, ys = self._xs, self._ys
        if self.extrapolate == "constant" and (x < xs[0] or x > xs[-1]):
            return 0.0
        i = self._segment(x)
        return float((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))

    # --- public surface ---------------------------------------------------

    def __call__(self, x_expr):
        """Return the symbolic interpolated value ``f(x_expr)``."""
        return self._func_class(x_expr)

    @property
    def function(self):
        """The underlying ``sympy.Function`` subclass."""
        return self._func_class

    def eval(self, x):
        """Numeric value ``f(x)`` (scalar or NumPy array)."""
        return self._apply(self._eval_value, x)

    def derivative(self, x):
        """Numeric derivative ``df/dx`` (scalar or NumPy array)."""
        return self._apply(self._eval_deriv, x)

    @staticmethod
    def _apply(fn, x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return fn(float(arr))
        return np.array([fn(float(v)) for v in arr.ravel()]).reshape(arr.shape)


class Interpolation2D:
    """Bilinear lookup table ``z = f(x, y)`` usable as a symbolic function.

    Parameters
    ----------
    x : array-like
        Strictly increasing grid abscissae along the first axis (length >= 2).
    y : array-like
        Strictly increasing grid abscissae along the second axis (length >= 2).
    z : array-like, shape ``(len(x), len(y))``
        Sample values; ``z[i, j] == f(x[i], y[j])``.
    extrapolate : {"constant", "linear"}
        Out-of-range policy applied independently per axis (see module
        docstring).  ``"constant"`` clamps the out-of-range coordinate to the
        nearest edge (zero partial in that direction); ``"linear"`` continues
        the nearest edge cell's slope.
    name : str, optional
        Base name for the generated symbolic function.

    Notes
    -----
    Call the instance with two sympy expressions (``f(x_expr, y_expr)``) to get
    the symbolic value; use :meth:`eval` / :meth:`derivative` for numbers.
    """

    def __init__(self, x, y, z, extrapolate="constant", name=None):
        if extrapolate not in _EXTRAPOLATION_MODES:
            raise ValueError(
                f"extrapolate must be one of {_EXTRAPOLATION_MODES}, got {extrapolate!r}"
            )
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        zs = np.asarray(z, dtype=float)
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("x and y must be 1D")
        if xs.size < 2 or ys.size < 2:
            raise ValueError("need at least two samples along each axis")
        if zs.shape != (xs.size, ys.size):
            raise ValueError(
                f"z must have shape (len(x), len(y)) = {(xs.size, ys.size)}, got {zs.shape}"
            )
        if not np.all(np.diff(xs) > 0):
            raise ValueError("x must be strictly increasing")
        if not np.all(np.diff(ys) > 0):
            raise ValueError("y must be strictly increasing")

        self._xs = xs
        self._ys = ys
        self._zs = zs
        self.extrapolate = extrapolate

        base = name or "interp2d"
        suffix = _stable_suffix(xs, ys, zs.ravel(),
                                [_EXTRAPOLATION_MODES.index(extrapolate)])
        self.name = f"{base}_{suffix}"
        dx_name = f"d{self.name}_dx"
        dy_name = f"d{self.name}_dy"

        self._func_class = _make_symbolic_function(
            self.name, ["x", "y"],
            self._eval_value, {1: self._eval_dx, 2: self._eval_dy},
        )
        self.modules = [
            {self.name: self._eval_value},
            {dx_name: self._eval_dx},
            {dy_name: self._eval_dy},
        ]

    # --- numeric evaluators (scalar; the solver vectorises them) -----------

    @staticmethod
    def _cell(grid, q):
        n = grid.size
        i = int(np.searchsorted(grid, q, side="right")) - 1
        return min(max(i, 0), n - 2)

    def _clamp(self, x, y):
        xs, ys = self._xs, self._ys
        xc = min(max(x, xs[0]), xs[-1])
        yc = min(max(y, ys[0]), ys[-1])
        return xc, yc

    def _corners(self, i, j):
        z = self._zs
        return z[i, j], z[i + 1, j], z[i, j + 1], z[i + 1, j + 1]

    def _eval_value(self, x, y):
        x, y = float(x), float(y)
        if self.extrapolate == "constant":
            x, y = self._clamp(x, y)
        xs, ys = self._xs, self._ys
        i, j = self._cell(xs, x), self._cell(ys, y)
        tx = (x - xs[i]) / (xs[i + 1] - xs[i])
        ty = (y - ys[j]) / (ys[j + 1] - ys[j])
        z00, z10, z01, z11 = self._corners(i, j)
        return float(
            (1 - tx) * (1 - ty) * z00
            + tx * (1 - ty) * z10
            + (1 - tx) * ty * z01
            + tx * ty * z11
        )

    def _eval_dx(self, x, y):
        x, y = float(x), float(y)
        xs, ys = self._xs, self._ys
        if self.extrapolate == "constant":
            if x < xs[0] or x > xs[-1]:
                return 0.0
            x, y = self._clamp(x, y)
        i, j = self._cell(xs, x), self._cell(ys, y)
        ty = (y - ys[j]) / (ys[j + 1] - ys[j])
        z00, z10, z01, z11 = self._corners(i, j)
        return float(((1 - ty) * (z10 - z00) + ty * (z11 - z01)) / (xs[i + 1] - xs[i]))

    def _eval_dy(self, x, y):
        x, y = float(x), float(y)
        xs, ys = self._xs, self._ys
        if self.extrapolate == "constant":
            if y < ys[0] or y > ys[-1]:
                return 0.0
            x, y = self._clamp(x, y)
        i, j = self._cell(xs, x), self._cell(ys, y)
        tx = (x - xs[i]) / (xs[i + 1] - xs[i])
        z00, z10, z01, z11 = self._corners(i, j)
        return float(((1 - tx) * (z01 - z00) + tx * (z11 - z10)) / (ys[j + 1] - ys[j]))

    # --- public surface ---------------------------------------------------

    def __call__(self, x_expr, y_expr):
        """Return the symbolic interpolated value ``f(x_expr, y_expr)``."""
        return self._func_class(x_expr, y_expr)

    @property
    def function(self):
        """The underlying ``sympy.Function`` subclass."""
        return self._func_class

    def eval(self, x, y):
        """Numeric value ``f(x, y)`` (scalars or broadcastable NumPy arrays)."""
        return self._apply(self._eval_value, x, y)

    def derivative(self, x, y, axis):
        """Numeric partial derivative w.r.t. ``axis`` (0 -> x, 1 -> y)."""
        if axis == 0:
            return self._apply(self._eval_dx, x, y)
        if axis == 1:
            return self._apply(self._eval_dy, x, y)
        raise ValueError("axis must be 0 (x) or 1 (y)")

    @staticmethod
    def _apply(fn, x, y):
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        if xa.ndim == 0 and ya.ndim == 0:
            return fn(float(xa), float(ya))
        xb, yb = np.broadcast_arrays(xa, ya)
        out = np.array([fn(float(a), float(b))
                        for a, b in zip(xb.ravel(), yb.ravel())])
        return out.reshape(xb.shape)
