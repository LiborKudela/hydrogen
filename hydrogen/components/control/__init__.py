"""Control / signal-block component library (Modelica.Blocks-style).

A causal real-signal domain: source, maths, and controller blocks wired
together through the `RealSignal` connector (a single value channel, no
flow).  Outputs may fan out to many inputs.  Use it to build setpoints and
feedback controllers that drive actuators in other domains -- e.g. a valve
opening.  See `README.md` for the overview.

Components are imported from their defining module (no flat re-exports)::

    from hydrogen.components.control.control_components import PID, Gain, Step
"""
