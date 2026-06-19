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

import sympy as sp

from ...model import DifferentialVariable, Input, Model, Parameter, Variable
from ...ports import Port

# Channel name carried by every signal connector (the connector's value).
_SIGNAL_CHANNEL = "value"


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

    def __init__(self, k=0.0, unit=None):
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

    def __init__(self, height=1.0, start_time=0.0, offset=0.0, unit=None):
        self.height = height
        self.start_time = start_time
        self.offset = offset
        self.unit = unit
        super().__init__()

    def signal(self, t):
        return self.offset + (self.height if t >= self.start_time else 0.0)


class Ramp(_TimeSource):
    """Ramp signal: flat ``offset``, linear rise of ``height`` over ``duration``, then flat."""

    def __init__(self, height=1.0, duration=1.0, start_time=0.0, offset=0.0, unit=None):
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


class Sine(_TimeSource):
    """Sine signal: ``y = offset + amplitude * sin(2*pi*freq*t + phase)`` for ``t >= start_time``."""

    def __init__(self, amplitude=1.0, freq=1.0, phase=0.0, offset=0.0,
                 start_time=0.0, unit=None):
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


# ---------------------------------------------------------------------------
# Maths
# ---------------------------------------------------------------------------


class Gain(Block):
    """Scalar gain: ``y = k * u``."""

    def __init__(self, k=1.0, unit=None):
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

    def __init__(self, k1=1.0, k2=1.0, unit=None):
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

    def __init__(self, unit=None):
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

    def __init__(self, n, weights=None, unit=None):
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

    def __init__(self, unit=None):
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

    def __init__(self, lo=0.0, hi=1.0, eps=1e-3, unit=None):
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

    def __init__(self, k=1.0, y_start=0.0, unit=None):
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

    def __init__(self, T, k=1.0, y_start=0.0, unit=None):
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

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, Tf=0.1, unit=None):
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
