# `hydrogen.utilities`

Cross-cutting helpers that are not tied to a single physics domain.

## Symbolic interpolation tables

`Interpolation1D` and `Interpolation2D` turn a lookup table into a symbolic
function that the model layer treats exactly like a CoolProp medium property:

- evaluating it on plain numbers runs a fast NumPy interpolation, and
- differentiating it symbolically (what the Newton **Jacobian** does) yields a
  matching derivative interpolation.

This means an interpolated quantity can appear anywhere inside a model's
residual equations and the solver gets the correct partial derivatives for
free — no finite differencing.

### How it works

Each table builds a uniquely-named `sympy.Function` subclass whose

- `eval` folds to a number when all arguments are numeric, and
- `fdiff(argindex)` returns a derivative `sympy.Function` backed by the
  analytic slope of the interpolant.

The numeric callbacks are exposed in a `.modules` list with one `{name: fn}`
dict per function (the value itself plus each first derivative). You hand that
list to `Model.instantiate(aditional_modules=...)`, concatenating it with a
medium's modules if you use both:

```python
model.instantiate(aditional_modules=table.modules + medium.modules)
```

The generated function name embeds a hash of the table contents, so identical
tables get a stable name across runs (good for the on-disk lambdify cache) and
distinct tables never collide in the lambdify namespace.

### 1D — `Interpolation1D(x, y, extrapolate=..., name=...)`

Piecewise-linear `y = f(x)` over a strictly increasing `x` grid.

```python
from hydrogen.utilities import Interpolation1D

# pump efficiency vs normalised flow
eta = Interpolation1D([0.0, 0.5, 1.0, 1.5],
                      [0.30, 0.70, 0.82, 0.65],
                      extrapolate="constant")

# inside Model.declare_equations:
#     q = self['q'].symbol
#     return [self['eta'].symbol - eta(q)]

# numeric checks / plotting:
eta.eval(0.75)        # -> 0.76
eta.derivative(0.75)  # -> segment slope
```

### 2D — `Interpolation2D(x, y, z, extrapolate=..., name=...)`

Bilinear `z = f(x, y)` over a strictly increasing `x`/`y` grid, with
`z[i, j] == f(x[i], y[j])`.

```python
from hydrogen.utilities import Interpolation2D

table = Interpolation2D(x=[0.0, 1.0, 2.0],
                        y=[0.0, 1.0],
                        z=[[0.0, 1.0],
                           [2.0, 3.0],
                           [4.0, 5.0]],
                        extrapolate="linear")

# inside a model:  table(self['a'].symbol, self['b'].symbol)
table.eval(0.5, 0.5)            # bilinear value
table.derivative(0.5, 0.5, 0)   # ∂/∂x
table.derivative(0.5, 0.5, 1)   # ∂/∂y
```

### Extrapolation outside the table

Both classes take `extrapolate`:

- `"constant"` — clamp to the nearest edge value. The derivative is **zero** in
  any direction whose coordinate is out of range (a flat shelf).
- `"linear"` — continue the slope of the nearest edge segment/cell. The
  derivative is the constant edge slope (a straight ramp).

In 2D the policy is applied independently per axis, so a point that is in range
along `x` but out of range along `y` is clamped/extended only along `y`.
