"""Signal-block component library (Modelica.Blocks-style control domain).

This domain models **causal real signals**: dimensionless (or unit-tagged)
scalars produced by source/maths/controller blocks and consumed by other
blocks or by actuators in another domain (e.g. a valve opening).  Unlike the
fluid / thermal connectors, a signal connector carries **no through
variable** -- wiring two signal ports is a pure value-equality, exactly the
Modelica `connect(a.y, b.u)` semantics.

Connector
---------
`RealSignal` -- one across channel ``value``, no flow channel.  An OUTPUT is
built with ``allow_fanout=True`` so a single block output can drive several
inputs (one-to-many), as in Modelica; an INPUT is single-use and
``require_connection=True`` (an unconnected input leaves its backing Variable
unclosed -> singular system, which `instantiate()` then warns about by name).

Blocks
------
Sources   : `Constant`, `Step`, `Ramp`, `Sine`
Maths     : `Gain`, `Add`, `Feedback`, `Sum`, `Product`, `Limiter`
Continuous: `Integrator`, `FirstOrder`, `PID`

Time-dependent sources are backed by the framework's `Input` (a two-level
`u(t_k)` / `u(t_{k+1})` signal) so they stay second-order accurate under
Crank-Nicolson even when feeding a differential block.  See the domain
`README.md` for the full overview and worked examples.
"""

from __future__ import annotations

import math
from typing import Annotated

import sympy as sp

from ...model import DifferentialVariable, Input, Model, Parameter, Variable
from ...paramspec import ParamSpec
from ...ports import Port

# Channel name carried by every signal connector (the connector's value).
_SIGNAL_CHANNEL = "value"

# Shared metadata for the optional signal `unit` tag every block carries.  The
# blocks have no common annotated `__init__`, so this spec is authored once and
# referenced from each subclass signature via ``Annotated[...]`` -- a single
# source of truth for the catalog and `declare_components`.
_SPEC_SIGNAL_UNIT = ParamSpec(
    "Optional unit tag for the signal channel (e.g. 'Pa', 'K', '1'); None "
    "leaves the signal untagged / dimensionless.")


class RealSignal(Port):
    """Causal real-valued signal connector (Modelica RealInput / RealOutput).

    A single across channel ``value`` and no flow channel, so `Model.connect`
    emits one value-equality per wire (the two backing Variables collapse to
    one symbol).  Causality (which side "writes" the value) is emergent from
    the assembled DAE rather than enforced here -- the same acausal treatment
    every other hydrogen variable gets.

    Use `RealSignal.as_output(...)` / `RealSignal.as_input(...)` rather than
    the raw constructor so the fan-out / require-connection conventions are
    applied consistently.
    """

    kind = "signal_real"
    required_channels = (_SIGNAL_CHANNEL,)
    flow_channels = ()

    @classmethod
    def as_output(cls, owner, var, *, name=None):
        """An output port: may fan out to many inputs."""
        return cls(owner, channels={_SIGNAL_CHANNEL: var}, name=name,
                   allow_fanout=True)

    @classmethod
    def as_input(cls, owner, var, *, name=None, require_connection=True):
        """An input port: single-use, warns if left unconnected."""
        return cls(owner, channels={_SIGNAL_CHANNEL: var}, name=name,
                   require_connection=require_connection)


class Block(Model):
    """Base class for signal blocks: helpers to declare signal in/out ports.

    Subclasses call `_add_input` / `_add_output` from `declare_components`
    to create a backing `Variable` (or use an existing one) and bind a
    `RealSignal` port to it, then return their residuals from
    `declare_equations`.  The port name equals the variable name, so a SISO
    block exposes ports ``u`` (input) and ``y`` (output) by convention.
    """

    #: Abstract base -- excluded from the component catalog / registry.
    _catalog_abstract = True

    def _add_input(self, name="u", *, init=0.0, unit=None, require_connection=True):
        self.add_component(name, Variable(init, unit))
        self.add_port(name, RealSignal.as_input(
            self, self[name], name=name, require_connection=require_connection))
        return self[name].symbol

    def _add_output(self, name="y", *, var=None, init=0.0, unit=None):
        if var is None:
            self.add_component(name, Variable(init, unit))
            var = self[name]
        self.add_port(name, RealSignal.as_output(self, var, name=name))
        return var.symbol


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class Constant(Block):
    """Constant signal: ``y = k``."""

    UI_ICON = "constant.svg"

    def __init__(self, k: Annotated[float, ParamSpec("Constant output value.")] = 0.0,
                 unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None):
        self.k = k
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('k', Parameter(self.k, self.unit))
        self._add_output('y', init=self.k, unit=self.unit)

    def declare_equations(self):
        return [self['y'].symbol - self['k'].symbol]


class _TimeSource(Block):
    """Base for time-driven sources: a two-level `Input` drives ``y``.

    Subclasses implement `signal(t) -> float`.  The output `y` is an
    algebraic Variable closed by ``y - drive`` where ``drive`` is the
    framework `Input`; routing the signal through `Input` (rather than a raw
    time symbol) is what keeps a downstream differential block second-order
    accurate under Crank-Nicolson.
    """

    unit = None

    def signal(self, t):  # pragma: no cover - overridden
        raise NotImplementedError

    def declare_components(self):
        self.add_component('drive', Input(self.signal, self.unit))
        self._add_output('y', init=self.signal(0.0), unit=self.unit)

    def declare_equations(self):
        return [self['y'].symbol - self['drive'].symbol]


class Step(_TimeSource):
    """Step signal: ``y = offset`` for ``t < start_time`` then ``offset + height``."""

    UI_ICON = "step.svg"

    def __init__(
        self,
        height: Annotated[float, ParamSpec("Size of the step (added to the "
                         "offset at t >= start_time).")] = 1.0,
        start_time: Annotated[float, ParamSpec("Time at which the step "
                             "occurs.", unit="s")] = 0.0,
        offset: Annotated[float, ParamSpec("Baseline output before the "
                         "step.")] = 0.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        self.height = height
        self.start_time = start_time
        self.offset = offset
        self.unit = unit
        super().__init__()

    def signal(self, t):
        return self.offset + (self.height if t >= self.start_time else 0.0)

    def declare_events(self):
        # The output jumps by `height` at `start_time`; integrate around it.
        return [self.start_time]


class Ramp(_TimeSource):
    """Ramp signal: flat ``offset``, linear rise of ``height`` over ``duration``, then flat."""

    UI_ICON = "ramp.svg"

    def __init__(
        self,
        height: Annotated[float, ParamSpec("Total rise over the ramp.")] = 1.0,
        duration: Annotated[float, ParamSpec("Time taken for the linear rise "
                           "(> 0).", unit="s")] = 1.0,
        start_time: Annotated[float, ParamSpec("Time at which the rise "
                             "begins.", unit="s")] = 0.0,
        offset: Annotated[float, ParamSpec("Baseline output before the "
                         "ramp.")] = 0.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if duration <= 0.0:
            raise ValueError("Ramp duration must be > 0")
        self.height = height
        self.duration = duration
        self.start_time = start_time
        self.offset = offset
        self.unit = unit
        super().__init__()

    def signal(self, t):
        if t <= self.start_time:
            return self.offset
        if t >= self.start_time + self.duration:
            return self.offset + self.height
        return self.offset + self.height * (t - self.start_time) / self.duration

    def declare_events(self):
        # Slope kinks at both ends of the linear rise (output is continuous but
        # its derivative jumps), which is exactly where Richardson would stall.
        return [self.start_time, self.start_time + self.duration]


class Sine(_TimeSource):
    """Sine signal: ``y = offset + amplitude * sin(2*pi*freq*t + phase)`` for ``t >= start_time``."""

    UI_ICON = "sine.svg"

    def __init__(
        self,
        amplitude: Annotated[float, ParamSpec("Peak amplitude of the "
                            "sine.")] = 1.0,
        freq: Annotated[float, ParamSpec("Frequency.", unit="Hz")] = 1.0,
        phase: Annotated[float, ParamSpec("Phase offset.", unit="rad")] = 0.0,
        offset: Annotated[float, ParamSpec("Constant offset added to the "
                         "sine.")] = 0.0,
        start_time: Annotated[float, ParamSpec("Time before which the output "
                             "stays at offset.", unit="s")] = 0.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        self.amplitude = amplitude
        self.freq = freq
        self.phase = phase
        self.offset = offset
        self.start_time = start_time
        self.unit = unit
        super().__init__()

    def signal(self, t):
        if t < self.start_time:
            return self.offset
        return self.offset + self.amplitude * math.sin(
            2.0 * math.pi * self.freq * (t - self.start_time) + self.phase)

    def declare_events(self):
        # Switch-on at `start_time`: a kink (and a value jump if the initial
        # phase is non-zero).  No event for the smooth sine that follows.
        return [self.start_time]


class SmoothRamp(_TimeSource):
    """Ramp with rounded corners -- a C-infinity alternative to `Ramp`.

    Same overall shape as `Ramp` (flat ``offset``, a rise of ``height`` spread
    over ``duration`` from ``start_time``, then flat) but the two sharp corners
    of the linear ramp are smoothed with a ``tanh`` blend of width ``corner``
    (a fraction of ``duration``).  It is the integral of a smoothed boxcar rate
    ``0.5*(height/duration)*[tanh((t-a)/w) - tanh((t-b)/w)]`` with
    ``a=start_time``, ``b=start_time+duration`` and ``w=corner*duration``:

        y(t) = offset + height/2
               + 0.5*(height/duration)*w*[logcosh((t-a)/w) - logcosh((t-b)/w)]

    Because it has no kinks, it needs **no events**: the adaptive controller
    sails through it on a smooth local-error estimate.  Use it instead of a
    `Ramp` when a slightly rounded command is acceptable and you would rather
    avoid event handling entirely.  As ``corner -> 0`` it converges to `Ramp`.
    """

    UI_ICON = "ramp.svg"

    def __init__(
        self,
        height: Annotated[float, ParamSpec("Total rise over the ramp.")] = 1.0,
        duration: Annotated[float, ParamSpec("Time taken for the (rounded) "
                           "rise (> 0).", unit="s")] = 1.0,
        start_time: Annotated[float, ParamSpec("Time about which the rise "
                             "begins.", unit="s")] = 0.0,
        offset: Annotated[float, ParamSpec("Baseline output before the "
                         "ramp.")] = 0.0,
        corner: Annotated[float, ParamSpec("Corner-rounding width as a fraction "
                         "of duration (0 < corner <= 0.5); smaller is sharper.")]
        = 0.1,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if duration <= 0.0:
            raise ValueError("SmoothRamp duration must be > 0")
        if not 0.0 < corner <= 0.5:
            raise ValueError("SmoothRamp corner must be in (0, 0.5]")
        self.height = height
        self.duration = duration
        self.start_time = start_time
        self.offset = offset
        self.corner = corner
        self.unit = unit
        super().__init__()

    @staticmethod
    def _logcosh(x):
        # Numerically stable log(cosh(x)) = |x| + log((1 + exp(-2|x|))/2),
        # avoiding cosh overflow for large |x|.
        ax = abs(x)
        return ax + math.log1p(math.exp(-2.0 * ax)) - math.log(2.0)

    def signal(self, t):
        w = self.corner * self.duration
        a = self.start_time
        b = self.start_time + self.duration
        slope = self.height / self.duration
        return (self.offset + 0.5 * self.height
                + 0.5 * slope * w
                * (self._logcosh((t - a) / w) - self._logcosh((t - b) / w)))


# ---------------------------------------------------------------------------
# Maths
# ---------------------------------------------------------------------------


class Gain(Block):
    """Scalar gain: ``y = k * u``."""

    UI_ICON = "gain.svg"

    def __init__(self, k: Annotated[float, ParamSpec("Gain factor applied to "
                "the input.")] = 1.0,
                unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None):
        self.k = k
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('k', Parameter(self.k, "-"))
        self._u = self._add_input('u', unit=self.unit)
        self._add_output('y', unit=self.unit)

    def declare_equations(self):
        return [self['y'].symbol - self['k'].symbol * self['u'].symbol]


class Add(Block):
    """Weighted sum of two inputs: ``y = k1*u1 + k2*u2``."""

    UI_ICON = "add.svg"

    def __init__(
        self,
        k1: Annotated[float, ParamSpec("Weight on the first input u1.")] = 1.0,
        k2: Annotated[float, ParamSpec("Weight on the second input u2.")] = 1.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        self.k1 = k1
        self.k2 = k2
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('k1', Parameter(self.k1, "-"))
        self.add_component('k2', Parameter(self.k2, "-"))
        self._add_input('u1', unit=self.unit)
        self._add_input('u2', unit=self.unit)
        self._add_output('y', unit=self.unit)

    def declare_equations(self):
        return [self['y'].symbol
                - self['k1'].symbol * self['u1'].symbol
                - self['k2'].symbol * self['u2'].symbol]


class Feedback(Block):
    """Difference junction: ``y = u1 - u2`` (setpoint minus measurement)."""

    UI_ICON = "feedback.svg"

    def __init__(self, unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None):
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self._add_input('u1', unit=self.unit)
        self._add_input('u2', unit=self.unit)
        self._add_output('y', unit=self.unit)

    def declare_equations(self):
        return [self['y'].symbol - (self['u1'].symbol - self['u2'].symbol)]


class Sum(Block):
    """N-input weighted sum: ``y = sum_i k_i * u_i``."""

    UI_ICON = "sum.svg"

    def __init__(
        self,
        n: Annotated[int, ParamSpec("Number of inputs (>= 1).")],
        weights: Annotated[list | None, ParamSpec("Per-input weights (length "
                          "n); None = all ones.")] = None,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if n < 1:
            raise ValueError("Sum needs at least one input")
        self.n = n
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError("len(weights) must equal n")
        self.weights = list(weights)
        self.unit = unit
        super().__init__()

    def declare_components(self):
        for i in range(self.n):
            self.add_component(f'k_{i}', Parameter(self.weights[i], "-"))
            self._add_input(f'u_{i}', unit=self.unit)
        self._add_output('y', unit=self.unit)

    def declare_equations(self):
        acc = sum(self[f'k_{i}'].symbol * self[f'u_{i}'].symbol
                  for i in range(self.n))
        return [self['y'].symbol - acc]


class Product(Block):
    """Multiply two inputs: ``y = u1 * u2``."""

    UI_ICON = "product.svg"

    def __init__(self, unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None):
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self._add_input('u1', unit=self.unit)
        self._add_input('u2', unit=self.unit)
        self._add_output('y', unit=self.unit)

    def declare_equations(self):
        return [self['y'].symbol - self['u1'].symbol * self['u2'].symbol]


class Limiter(Block):
    """Smoothly saturate a signal to ``[lo, hi]``.

    A hard ``clip`` has kinks at both bounds that hurt Newton convergence, so
    this uses a smooth min/max with a small blending scale ``eps``
    (``smooth_max(a,b) = 0.5*(a+b+sqrt((a-b)^2+eps^2))``).  The error vs a
    hard clip is ~``eps/4`` only within ~``eps`` of each bound; pick ``eps``
    small relative to the signal range (default ``1e-3``).  Ideal for clamping
    a valve opening to ``[0, 1]``.
    """

    UI_ICON = "limiter.svg"

    def __init__(
        self,
        lo: Annotated[float, ParamSpec("Lower saturation bound.")] = 0.0,
        hi: Annotated[float, ParamSpec("Upper saturation bound (> lo).")] = 1.0,
        eps: Annotated[float, ParamSpec("Smoothing scale of the soft min/max "
                      "blend; keep small relative to the signal range.")] = 1e-3,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if hi <= lo:
            raise ValueError("Limiter requires hi > lo")
        self.lo = lo
        self.hi = hi
        self.eps = eps
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('lo', Parameter(self.lo, self.unit))
        self.add_component('hi', Parameter(self.hi, self.unit))
        self.add_component('eps', Parameter(self.eps, self.unit))
        self._add_input('u', unit=self.unit)
        self._add_output('y', init=min(max(0.0, self.lo), self.hi), unit=self.unit)

    def declare_equations(self):
        u = self['u'].symbol
        lo = self['lo'].symbol
        hi = self['hi'].symbol
        eps = self['eps'].symbol

        def smooth_max(a, b):
            return 0.5 * (a + b + sp.sqrt((a - b) ** 2 + eps ** 2))

        def smooth_min(a, b):
            return 0.5 * (a + b - sp.sqrt((a - b) ** 2 + eps ** 2))

        y_clamped = smooth_min(smooth_max(u, lo), hi)
        return [self['y'].symbol - y_clamped]


# ---------------------------------------------------------------------------
# Continuous (stateful)
# ---------------------------------------------------------------------------


class Integrator(Block):
    """Time integrator: ``der(y) = k * u``, ``y(0) = y_start``."""

    UI_ICON = "integrator.svg"

    def __init__(
        self,
        k: Annotated[float, ParamSpec("Integral gain on the input.")] = 1.0,
        y_start: Annotated[float, ParamSpec("Initial output (integration "
                          "constant).")] = 0.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        self.k = k
        self.y_start = y_start
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('k', Parameter(self.k, "-"))
        self._add_input('u', unit=self.unit)
        self.add_component('y', DifferentialVariable(self.y_start, self.unit))
        self.add_port('y', RealSignal.as_output(self, self['y'], name='y'))

    def declare_equations(self):
        return [self['der_y'].symbol - self['k'].symbol * self['u'].symbol]


class FirstOrder(Block):
    """First-order lag: ``T*der(y) + y = k*u`` (time constant ``T``, gain ``k``)."""

    UI_ICON = "first_order.svg"

    def __init__(
        self,
        T: Annotated[float, ParamSpec("Time constant (> 0).", unit="s")],
        k: Annotated[float, ParamSpec("Steady-state gain.")] = 1.0,
        y_start: Annotated[float, ParamSpec("Initial output.")] = 0.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if T <= 0.0:
            raise ValueError("FirstOrder time constant T must be > 0")
        self.T = T
        self.k = k
        self.y_start = y_start
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('T', Parameter(self.T, "s"))
        self.add_component('k', Parameter(self.k, "-"))
        self._add_input('u', unit=self.unit)
        self.add_component('y', DifferentialVariable(self.y_start, self.unit))
        self.add_port('y', RealSignal.as_output(self, self['y'], name='y'))

    def declare_equations(self):
        T = self['T'].symbol
        k = self['k'].symbol
        return [T * self['der_y'].symbol + self['y'].symbol - k * self['u'].symbol]


class PID(Block):
    """Parallel PID controller with a filtered derivative.

        y = kp*e + ki*x_i + kd*der(x_d)

    where the single input ``u`` is the control error ``e`` (put a `Feedback`
    in front to form ``setpoint - measurement``), ``x_i`` is the integral
    state (``der(x_i) = e``) and ``x_d`` is a first-order derivative filter
    (``der(x_d) = (e - x_d)/Tf``) so ``der(x_d) -> de/dt`` as ``Tf -> 0`` --
    a proper, noise-tolerant derivative instead of a raw (unbounded) one.

    Note: this is a textbook PID without output saturation / anti-windup, so
    drive its output through a `Limiter` when commanding a bounded actuator
    (e.g. a ``[0, 1]`` valve opening); integrator anti-windup is a planned
    enhancement.
    """

    UI_ICON = "pid.svg"

    def __init__(
        self,
        kp: Annotated[float, ParamSpec("Proportional gain.")] = 1.0,
        ki: Annotated[float, ParamSpec("Integral gain.", unit="1/s")] = 0.0,
        kd: Annotated[float, ParamSpec("Derivative gain.", unit="s")] = 0.0,
        Tf: Annotated[float, ParamSpec("Derivative-filter time constant (> 0); "
                     "der(x_d) -> de/dt as Tf -> 0.", unit="s")] = 0.1,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if Tf <= 0.0:
            raise ValueError("PID derivative-filter time Tf must be > 0")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Tf = Tf
        self.unit = unit
        super().__init__()

    def declare_components(self):
        self.add_component('kp', Parameter(self.kp, "-"))
        self.add_component('ki', Parameter(self.ki, "1/s"))
        self.add_component('kd', Parameter(self.kd, "s"))
        self.add_component('Tf', Parameter(self.Tf, "s"))
        self._add_input('u', unit=self.unit)
        self.add_component('x_i', DifferentialVariable(0.0, self.unit))
        self.add_component('x_d', DifferentialVariable(0.0, self.unit))
        self._add_output('y', unit=self.unit)

    def declare_equations(self):
        e = self['u'].symbol
        kp = self['kp'].symbol
        ki = self['ki'].symbol
        kd = self['kd'].symbol
        Tf = self['Tf'].symbol
        der_x_d = self['der_x_d'].symbol
        eq_int = self['der_x_i'].symbol - e
        eq_der = der_x_d - (e - self['x_d'].symbol) / Tf
        eq_out = self['y'].symbol - (kp * e + ki * self['x_i'].symbol + kd * der_x_d)
        return [eq_int, eq_der, eq_out]
