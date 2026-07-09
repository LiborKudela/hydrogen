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

import csv as _csv
import math
from typing import Annotated

import numpy as np
import sympy as sp

from ...model import DifferentialVariable, Input, Model, Parameter, Variable
from ...numerics import smooth_max, smooth_min
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


class CsvTable(_TimeSource):
    """Replay one column of a CSV file as a time-interpolated signal ``y(t)``.

    A data-driven source (the analogue of Modelica's ``CombiTimeTable``): reads
    `filename` once at construction, uses `time_column` as the time base and
    `value_column` as the data, and emits ``y(t)`` by linear interpolation,
    held flat outside the recorded span.  An optional affine transform
    ``value_scale * raw + value_offset`` rescales the column (e.g. degC -> K
    with ``value_offset=273.15``), and `time_scale` converts the time column
    into seconds.

    Columns may be named (matched against the header row) or given as integer
    indices.  Because the underlying `Input` carries the value at both ends of
    each step, a downstream differential block stays second-order accurate.

    Typical use -- drive a `flow.TemperatureInlet` from measured data::

        tin = CsvTable("PipeDataULg151202.csv", value_column="water_inlet",
                       value_offset=273.15, unit="K")
        self.connect(tin.ports["y"], inlet.ports["T_set"])
    """

    UI_ICON = "constant.svg"

    def __init__(
        self,
        filename: Annotated[str, ParamSpec("Path to the CSV file to read.")],
        value_column: Annotated[str, ParamSpec("Column (header name or integer "
                               "index) replayed as the output signal.")],
        time_column: Annotated[str, ParamSpec("Column (header name or integer "
                              "index) holding the time base.")] = "time",
        value_scale: Annotated[float, ParamSpec("Output = value_scale * raw + "
                              "value_offset.")] = 1.0,
        value_offset: Annotated[float, ParamSpec("Output = value_scale * raw + "
                               "value_offset (e.g. 273.15 for degC -> K).")]
        = 0.0,
        time_scale: Annotated[float, ParamSpec("Multiplier converting the time "
                             "column into seconds.")] = 1.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        self.filename = filename
        self.value_column = value_column
        self.time_column = time_column
        self.value_scale = value_scale
        self.value_offset = value_offset
        self.time_scale = time_scale
        self.unit = unit
        # The CSV is read lazily (on the first `signal` evaluation) rather than
        # at construction: this lets the component be *built* with an as-yet
        # unset / placeholder `filename` (e.g. when the canvas introspects it to
        # discover its single output port), while a real run still surfaces a
        # missing / malformed file the moment the solver samples the source.
        self._t = None
        self._v = None
        super().__init__()

    def _load(self):
        with open(self.filename, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            raise ValueError(f"CsvTable: {self.filename!r} has no data rows")
        header = list(rows[0].keys())

        def resolve(spec):
            if spec in header:
                return spec
            try:
                return header[int(spec)]
            except (ValueError, IndexError, TypeError):
                raise KeyError(
                    f"CsvTable: column {spec!r} not found in header {header}")

        tc = resolve(self.time_column)
        vc = resolve(self.value_column)
        t = np.array([float(r[tc]) for r in rows]) * self.time_scale
        v = np.array([float(r[vc]) for r in rows]) * self.value_scale \
            + self.value_offset
        order = np.argsort(t, kind="stable")
        return t[order], v[order]

    def signal(self, t):
        # np.interp holds the end values flat outside the span (no extrapolation).
        if self._t is None:
            # An unset `filename` means the source isn't configured yet (e.g. the
            # canvas is only introspecting the block to place its output port);
            # emit the flat offset rather than reading a file.  A *configured*
            # file that fails to load still raises -- a real run must not run on
            # silently-wrong data.
            if self.filename is None:
                return self.value_offset
            self._t, self._v = self._load()
        return float(np.interp(float(t), self._t, self._v))


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

        y_clamped = smooth_min(smooth_max(u, lo, eps), hi, eps)
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
    """Parallel PID controller with a filtered derivative, output limits and
    anti-windup.

        v = kp*e + x_i + kd*der(x_d),      e = reference - feedback
        y = clamp(v, y_min, y_max)         (when `limited`, else y = v)

    The controller exposes three signal ports: two inputs ``reference`` (the
    setpoint) and ``feedback`` (the measurement), and one output ``y``.  The
    control error ``e = reference - feedback`` is formed internally (no need
    for a separate `Feedback` block).

    States:
      * ``x_i`` -- the integral action, in OUTPUT units, so ``der(x_i) = ki*e``
        (equivalently ``ki*integral(e)``).  Its initial value is ``y_start``,
        which therefore seeds the controller's **initial output** (bumpless
        start when the error begins near zero -- needs ``ki > 0`` to be held).
      * ``x_d`` -- a first-order derivative filter, ``der(x_d) = (e - x_d)/Tf``,
        so ``kd*der(x_d) -> kd*de/dt`` as ``Tf -> 0`` -- a proper, noise-tolerant
        derivative instead of a raw (unbounded) one.

    Output limiting (``limited=True``):
      * The output is smoothly saturated to ``[y_min, y_max]`` (a soft min/max
        so Newton keeps a smooth Jacobian near the bounds).
      * **Back-calculation anti-windup** feeds the saturation error back into
        the integrator, ``der(x_i) = ki*(e + Ni*(y - v))``: while saturated the
        term ``y - v`` is non-zero and bleeds the integral state back toward the
        limit so it cannot wind up.  ``Ni`` is the (dimensionless) anti-windup
        gain ratio, so the back-calculation gain ``Ni*ki`` scales WITH the
        integral gain -- and, crucially, vanishes for a pure P / PD controller
        (``ki = 0``), where there is no integrator to unwind.  A pure-P loop
        with limits is then exactly ``y = clamp(kp*e + y_start, y_min, y_max)``.
        With ``limited=False`` there is no saturation and ``der(x_i) = ki*e``
        (a textbook PID).
    """

    UI_ICON = "pid.svg"

    def __init__(
        self,
        kp: Annotated[float, ParamSpec("Proportional gain.")] = 1.0,
        ki: Annotated[float, ParamSpec("Integral gain.", unit="1/s")] = 0.0,
        kd: Annotated[float, ParamSpec("Derivative gain.", unit="s")] = 0.0,
        Tf: Annotated[float, ParamSpec("Derivative-filter time constant (> 0); "
                     "der(x_d) -> de/dt as Tf -> 0.", unit="s")] = 0.1,
        y_start: Annotated[float, ParamSpec("Initial output: seeds the integral "
                          "state x_i, so the controller starts at this value "
                          "(needs ki > 0 to be held).")] = 0.0,
        limited: Annotated[bool, ParamSpec("If true, saturate the output to "
                          "[y_min, y_max] with back-calculation anti-windup; if "
                          "false the output is unbounded.", structural=True)]
        = False,
        y_min: Annotated[float, ParamSpec("Lower output limit (used when "
                        "limited).")] = 0.0,
        y_max: Annotated[float, ParamSpec("Upper output limit (used when "
                        "limited); must be > y_min.")] = 1.0,
        Ni: Annotated[float, ParamSpec("Anti-windup gain ratio (dimensionless, "
                     ">= 0, used when limited): the back-calculation gain is "
                     "Ni*ki, so anti-windup scales with the integral gain and "
                     "vanishes for a pure P/PD controller (ki = 0). 0 disables "
                     "it.")] = 1.0,
        unit: Annotated[str, _SPEC_SIGNAL_UNIT] = None,
    ):
        if Tf <= 0.0:
            raise ValueError("PID derivative-filter time Tf must be > 0")
        self.limited = bool(limited)
        if self.limited:
            if y_max <= y_min:
                raise ValueError("PID output limits require y_max > y_min")
            if Ni < 0.0:
                raise ValueError("PID anti-windup ratio Ni must be >= 0")
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Tf = Tf
        self.y_start = y_start
        self.y_min = y_min
        self.y_max = y_max
        self.Ni = Ni
        self.unit = unit
        # Soft-saturation blend width: a small fraction of the limit span so the
        # rounding is imperceptible yet keeps the min/max corners differentiable.
        self._sat_eps = 1e-3 * (y_max - y_min) if self.limited else 0.0
        super().__init__()

    def declare_components(self):
        self.add_component('kp', Parameter(self.kp, "-"))
        self.add_component('ki', Parameter(self.ki, "1/s"))
        self.add_component('kd', Parameter(self.kd, "s"))
        self.add_component('Tf', Parameter(self.Tf, "s"))
        self._add_input('reference', unit=self.unit)
        self._add_input('feedback', unit=self.unit)
        # Integral state carries the integral action directly (output units), so
        # its initial value y_start seeds the controller's initial output.
        self.add_component('x_i', DifferentialVariable(self.y_start, self.unit))
        self.add_component('x_d', DifferentialVariable(0.0, self.unit))
        if self.limited:
            self.add_component('y_min', Parameter(self.y_min, self.unit))
            self.add_component('y_max', Parameter(self.y_max, self.unit))
            self.add_component('Ni', Parameter(self.Ni, "-"))
            # Unsaturated command, recorded as a leaf variable (plottable) and
            # used by the anti-windup back-calculation.  NOTE: the OUTPUT clamp
            # deliberately does NOT read this leaf -- it saturates the live
            # expression `v_expr` instead (see `declare_equations`), so the
            # output's Jacobian reflects the true saturation state at every
            # Newton iterate rather than the leaf's stale (corner) start value.
            self.add_component('v', Variable(self.y_start, self.unit))
        self._add_output('y', init=self._saturated_start(), unit=self.unit)

    def _saturated_start(self):
        """Initial output, clamped to the limits so it starts inside the valid
        range even if `y_start` was left at a default outside [y_min, y_max]."""
        if not self.limited:
            return self.y_start
        return min(max(self.y_start, self.y_min), self.y_max)

    def declare_equations(self):
        e = self['reference'].symbol - self['feedback'].symbol
        kp = self['kp'].symbol
        ki = self['ki'].symbol
        kd = self['kd'].symbol
        Tf = self['Tf'].symbol
        der_x_d = self['der_x_d'].symbol
        x_i = self['x_i'].symbol
        y = self['y'].symbol

        eq_der = der_x_d - (e - self['x_d'].symbol) / Tf
        v_expr = kp * e + x_i + kd * der_x_d

        if not self.limited:
            eq_int = self['der_x_i'].symbol - ki * e
            eq_out = y - v_expr
            return [eq_int, eq_der, eq_out]

        v = self['v'].symbol
        y_min = self['y_min'].symbol
        y_max = self['y_max'].symbol
        Ni = self['Ni'].symbol
        eps = self._sat_eps

        # Record the unsaturated command as a leaf (for plotting / anti-windup).
        eq_v = v - v_expr
        # Saturate the LIVE expression, not the leaf `v`.  This is the key to a
        # robust initialisation: `v_expr` is evaluated from the current state, so
        # if the proportional term kp*e is huge (large error, high gain) the
        # clamp is already deep in saturation where its slope ~ 0 -- the output
        # decouples and stays at the limit at EVERY Newton iterate.  Clamping the
        # leaf `v` instead would use its stale start value (y_start, at the clip
        # corner where the slope ~ 0.5), leaking the huge unsaturated command
        # into the first Newton step and driving the coupled plant (e.g. a
        # compressible pipe) to non-physical states (negative pressure).
        eq_out = y - smooth_min(smooth_max(v_expr, y_min, eps), y_max, eps)
        # Back-calculation anti-windup, gain Ni*ki so it scales with the
        # integral gain: (y - v_expr) is 0 unless saturated, where it bleeds the
        # integral back toward the active limit.  With ki = 0 the whole
        # integrator (and its windup) disappears -> a clean limited P/PD loop.
        eq_int = self['der_x_i'].symbol - ki * (e + Ni * (y - v_expr))
        return [eq_int, eq_der, eq_v, eq_out]
