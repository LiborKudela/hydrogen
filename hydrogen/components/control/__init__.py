"""Control / signal-block component library (Modelica.Blocks-style).

A causal real-signal domain: source, maths, and controller blocks wired
together through the `RealSignal` connector (a single value channel, no
flow).  Outputs may fan out to many inputs.  Use it to build setpoints and
feedback controllers that drive actuators in other domains -- e.g. a valve
opening.  See `README.md` for the overview.
"""

from .control_components import (
    Add,
    Block,
    Constant,
    Feedback,
    FirstOrder,
    Gain,
    Integrator,
    Limiter,
    PID,
    Product,
    Ramp,
    RealSignal,
    Sine,
    Step,
    Sum,
)

__all__ = [
    # connector
    "RealSignal",
    # base
    "Block",
    # sources
    "Constant",
    "Step",
    "Ramp",
    "Sine",
    # maths
    "Gain",
    "Add",
    "Feedback",
    "Sum",
    "Product",
    "Limiter",
    # continuous
    "Integrator",
    "FirstOrder",
    "PID",
]
