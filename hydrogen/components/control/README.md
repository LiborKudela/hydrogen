# Control components (signal blocks)

A Modelica.Blocks-style control domain for the `hydrogen` solver: source,
maths, and controller blocks that produce and process **causal real
signals**, wired together through a single typed connector. Use it to build
setpoints and feedback controllers that drive actuators in other domains
(for example a valve opening).

## Layout

```
control/
├── __init__.py            # public API re-exports
├── control_components.py   # connector + block implementations
└── README.md               # this file
```

## Connector

- **`RealSignal`** — a causal real-valued signal interface carrying one
  across channel `value` and **no flow channel**. Connecting two signal
  ports is therefore a pure value-equality (the two backing variables
  collapse to one symbol), exactly the Modelica `connect(a.y, b.u)`
  semantics.
  - **Outputs** (`RealSignal.as_output`) are built with `allow_fanout=True`,
    so one block output can drive **many** inputs (one-to-many). This is the
    one place the otherwise single-use port rule is relaxed — it is sound
    only because a signal has no through variable, so extra wires add
    consistent equalities rather than violating Kirchhoff.
  - **Inputs** (`RealSignal.as_input`) are single-use and
    `require_connection=True`: an unconnected input leaves its variable
    unclosed (singular system), so `instantiate()` warns by name.

Causality is *not* enforced at connect-time — like every hydrogen variable, a
signal is solved acausally in the assembled DAE.

## Blocks

By convention a single-input/single-output block exposes ports `u` (input)
and `y` (output).

| Class | Equation / behavior |
| --- | --- |
| `Constant(k)` | `y = k` |
| `Step(height, start_time, offset)` | step at `start_time` |
| `Ramp(height, duration, start_time, offset)` | linear rise over `duration` |
| `Sine(amplitude, freq, phase, offset, start_time)` | sinusoid |
| `Gain(k)` | `y = k*u` |
| `Add(k1, k2)` | `y = k1*u1 + k2*u2` |
| `Feedback()` | `y = u1 - u2` |
| `Sum(n, weights)` | `y = sum_i k_i*u_i` |
| `Product()` | `y = u1*u2` |
| `Limiter(lo, hi, eps)` | smooth saturation to `[lo, hi]` |
| `Integrator(k, y_start)` | `der(y) = k*u` |
| `FirstOrder(T, k)` | `T*der(y) + y = k*u` |
| `PID(kp, ki, kd, Tf)` | parallel PID with filtered derivative |

Time-dependent sources (`Step`, `Ramp`, `Sine`) are backed by the framework
`Input` (a two-level `u(t_k)` / `u(t_{k+1})` signal), so they remain
second-order accurate under Crank-Nicolson even when feeding a differential
block. Stateful blocks (`Integrator`, `FirstOrder`, `PID`) use
`DifferentialVariable`.

`Limiter` uses a *smooth* min/max (blend scale `eps`) instead of a hard clip
so the Jacobian stays continuous; keep `eps` small relative to the signal
range. `PID` has no output saturation / anti-windup yet — drive its output
through a `Limiter` when commanding a bounded actuator such as a `[0, 1]`
valve opening.

## Usage

```python
from hydrogen.components import Constant, PID, Feedback, Limiter, RealSignal
# equivalently:
from hydrogen.components.control import Constant, PID

# inside a Model.declare_equations():
#   self.connect(self['ctrl'].ports['y'], self['valve'].ports['opening'])
```

A signal output may fan out to several consumers without a junction:

```python
self.connect(self['src'].ports['y'], self['gain_a'].ports['u'])
self.connect(self['src'].ports['y'], self['gain_b'].ports['u'])  # same output, ok
```

## Adding a new physics domain

Create a sibling subfolder containing the domain's Python module(s) and a
`README.md`, declare the domain's own `Port` subclass(es) inside it, then
re-export its public symbols from `hydrogen/components/__init__.py`.
