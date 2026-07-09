"""Inline flow sensors for the `thermofluid` domain.

Every sensor here is a **lossless two-port pass-through**: it sits inline on a
`(p, h, m_dot)` line, imposes no pressure drop, adds no enthalpy, and conserves
mass, so it can be dropped between any two fluid components without perturbing
the flow.  It reads one quantity of the stream and publishes it on a
`control.RealSignal` OUTPUT port named ``y`` (green signal wire); that output
may drive a control block (e.g. a `control.PID` feedback) or simply be recorded
/ plotted.  Leaving ``y`` unconnected is fine -- it is an output, closed by its
own defining equation.

Sensors
-------
* `MassFlowSensor`    -- axial mass-flow rate                    ``y`` [kg/s]
* `MassTotalizer`     -- cumulative mass that has passed (an
                         integrator, ``der(y) = m_dot``)         ``y`` [kg]
* `TemperatureSensor` -- fluid temperature; static ``T(p, h)`` by
                         default, or total / stagnation
                         temperature when ``total=True``         ``y`` [K]
* `PressureSensor`    -- fluid pressure; static ``p`` by default,
                         or total / stagnation pressure
                         ``p + rho*w^2/2`` when ``total=True``   ``y`` [Pa]
* `VolumeFlowSensor`  -- volumetric flow rate ``m_dot / rho``    ``y`` [m^3/s]

Sign convention (package-wide "flow into me"): the inlet face reports
``m_dot_in > 0`` for forward flow (inlet -> outlet), so the "axial" flow the
rate/total sensors publish is ``m_dot_in``.
"""

from __future__ import annotations

from typing import Annotated

import sympy as sp

from ...medium import CoolPropMedium
from ...model import DifferentialVariable, Model, Parameter, Variable
from ...paramspec import ParamSpec, merged_param_specs
from ..control.control_components import RealSignal
from .ports import FluidPort_phm


class _FlowSensor(Model):
    """Base class: a lossless ``(p, h, m_dot)`` pass-through plus a ``y`` signal.

    Subclasses declare any measurement parameters / internal variables in
    :meth:`_declare_signal` (which must also create the ``y`` output) and return
    the residual(s) closing ``y`` from :meth:`_measure_equations`.  The base
    handles the two fluid ports and the pass-through equalities.
    """

    #: Abstract base -- excluded from the component catalog / registry.
    _catalog_abstract = True

    #: Draw the bare P&ID symbol on the canvas (no labelled box / editable
    #: ports); inherited by every concrete sensor.  Ports land on the icon:
    #: ``inlet`` left, ``outlet`` right, the ``y`` signal on the bottom edge.
    UI_ICON_ONLY = True

    #: SI unit tag of the published signal (subclass overrides).
    _SIGNAL_UNIT = "1"

    def __init__(self, medium: CoolPropMedium):
        self.medium = medium
        self._h_std = float(medium.eval_h_pT(101325.0, 293.15))
        super().__init__()

    # -- pass-through plumbing ------------------------------------------------
    def declare_components(self):
        self.add_component('p_in', Variable(101325.0, "Pa"))
        self.add_component('h_in', Variable(self._h_std, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('p_out', Variable(101325.0, "Pa"))
        self.add_component('h_out', Variable(self._h_std, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'],
                      'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'],
                      'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self._declare_signal()

    def _declare_signal(self):
        """Add the ``y`` output Variable and its `RealSignal` output port.

        Overridden by sensors that need extra parameters / internal variables
        or an integrated (differential) ``y``.
        """
        self.add_component('y', Variable(0.0, self._SIGNAL_UNIT))
        self.add_port('y', RealSignal.as_output(self, self['y'], name='y'))

    def declare_equations(self):
        # Lossless pass-through: pressure and enthalpy are equal across the
        # sensor (union-find variable-equality, exactly like `flow.Splitter`),
        # and mass is conserved under "flow into me" (m_dot_in + m_dot_out = 0).
        self.add_connection(self['p_out'], self['p_in'])
        self.add_connection(self['h_out'], self['h_in'])
        eqs = [self['m_dot_in'].symbol + self['m_dot_out'].symbol]
        eqs.extend(self._measure_equations())
        return eqs

    def _measure_equations(self):  # pragma: no cover - overridden
        raise NotImplementedError


class MassFlowSensor(_FlowSensor):
    """Inline mass-flow-rate sensor: ``y = m_dot`` (axial) [kg/s].

    Reports the forward mass-flow rate through the line (positive inlet ->
    outlet).  A perfect, drop-free flow meter.
    """

    UI_ICON = "mass_flow_sensor.svg"
    _SIGNAL_UNIT = "kg/s"

    def _measure_equations(self):
        return [self['y'].symbol - self['m_dot_in'].symbol]


class MassTotalizer(_FlowSensor):
    """Inline mass totalizer: ``y`` = cumulative mass that has passed [kg].

    An integrating flow meter -- ``der(y) = m_dot`` -- so ``y`` accumulates the
    net mass carried through the line since ``t = 0`` (starting from
    ``y_start``).  With forward flow ``y`` rises; reverse flow makes it fall.
    """

    UI_ICON = "mass_totalizer.svg"
    _SIGNAL_UNIT = "kg"

    def __init__(
        self,
        medium: CoolPropMedium,
        y_start: Annotated[float, ParamSpec("Initial totalized mass "
                          "(integration constant).", unit="kg")] = 0.0,
    ):
        self.y_start = y_start
        super().__init__(medium)

    def _declare_signal(self):
        self.add_component('y', DifferentialVariable(self.y_start, "kg"))
        self.add_port('y', RealSignal.as_output(self, self['y'], name='y'))

    def _measure_equations(self):
        return [self['der_y'].symbol - self['m_dot_in'].symbol]


class TemperatureSensor(_FlowSensor):
    """Inline temperature sensor: ``y`` = fluid temperature [K].

    By default it reports the **static** (thermodynamic) temperature
    ``T = T(p, h)`` from the port state.  Set ``total=True`` to instead report
    the **total / stagnation** temperature, which folds the flow's kinetic
    energy back into the enthalpy before the property lookup::

        w        = m_dot / (rho * A),   A = pi * D^2 / 4
        T_total  = T(p, h + w^2 / 2)

    so the sensor needs the pipe bore ``D`` to turn the mass flow into a
    velocity.  At low speed ``w^2/2 << h`` and the two readings coincide; the
    distinction only matters in fast (high-Mach) flow.
    """

    UI_ICON = "temperature_sensor.svg"
    _SIGNAL_UNIT = "K"

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Pipe bore at the sensor (sets the flow "
                    "area used to recover velocity for the total-temperature "
                    "correction).", unit="m")] = 0.01,
        total: Annotated[bool, ParamSpec("If true, report the total "
                        "(stagnation) temperature T(p, h + w^2/2); if false, "
                        "the static temperature T(p, h).",
                        structural=True)] = False,
    ):
        self.D = D
        self.total = bool(total)
        super().__init__(medium)

    def _declare_signal(self):
        spec = merged_param_specs(type(self))
        self.add_component('D', Parameter(self.D, **spec['D'].param_kwargs()))
        self.add_component('y', Variable(293.15, "K"))
        self.add_port('y', RealSignal.as_output(self, self['y'], name='y'))

    def _measure_equations(self):
        p = self['p_in'].symbol
        h = self['h_in'].symbol
        if not self.total:
            return [self['y'].symbol - self.medium.T_ph(p, h)]
        # Stagnation: add the kinetic energy w^2/2 to the enthalpy first.
        rho = self.medium.rho_ph(p, h)
        area = sp.pi * self['D'].symbol ** 2 / 4
        w = self['m_dot_in'].symbol / (rho * area)
        h_total = h + w ** 2 / 2
        return [self['y'].symbol - self.medium.T_ph(p, h_total)]


class PressureSensor(_FlowSensor):
    """Inline pressure sensor: ``y`` = fluid pressure [Pa].

    By default it reports the **static** pressure ``y = p`` at the port.  Set
    ``total=True`` to instead report the **total / stagnation** pressure, which
    adds the flow's dynamic pressure::

        w       = m_dot / (rho * A),   A = pi * D^2 / 4
        p_total = p + rho * w^2 / 2 = p + m_dot^2 / (2 * rho * A^2)

    (the low-Mach / isentropic result ``dp = rho*dh`` applied to the
    stagnation-enthalpy rise ``w^2/2``), so the sensor needs the pipe bore
    ``D`` to turn the mass flow into a velocity.  At low speed the dynamic term
    is tiny and the two readings coincide.
    """

    UI_ICON = "pressure_sensor.svg"
    _SIGNAL_UNIT = "Pa"

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Pipe bore at the sensor (sets the flow "
                    "area used to recover velocity for the total-pressure "
                    "correction).", unit="m")] = 0.01,
        total: Annotated[bool, ParamSpec("If true, report the total "
                        "(stagnation) pressure p + rho*w^2/2; if false, the "
                        "static pressure p.", structural=True)] = False,
    ):
        self.D = D
        self.total = bool(total)
        super().__init__(medium)

    def _declare_signal(self):
        spec = merged_param_specs(type(self))
        self.add_component('D', Parameter(self.D, **spec['D'].param_kwargs()))
        self.add_component('y', Variable(101325.0, "Pa"))
        self.add_port('y', RealSignal.as_output(self, self['y'], name='y'))

    def _measure_equations(self):
        p = self['p_in'].symbol
        if not self.total:
            return [self['y'].symbol - p]
        # Dynamic pressure rho*w^2/2 = m_dot^2 / (2*rho*A^2).
        rho = self.medium.rho_ph(p, self['h_in'].symbol)
        area = sp.pi * self['D'].symbol ** 2 / 4
        q_dyn = self['m_dot_in'].symbol ** 2 / (2 * rho * area ** 2)
        return [self['y'].symbol - (p + q_dyn)]


class VolumeFlowSensor(_FlowSensor):
    """Inline volumetric-flow-rate sensor: ``y = m_dot / rho`` [m^3/s].

    Converts the mass-flow rate to a volumetric rate using the local density
    ``rho(p, h)`` at the port state.  Written multiplicatively as
    ``y * rho = m_dot`` so the residual never divides by the density.
    """

    UI_ICON = "volume_flow_sensor.svg"
    _SIGNAL_UNIT = "m^3/s"

    def _measure_equations(self):
        rho = self.medium.rho_ph(self['p_in'].symbol, self['h_in'].symbol)
        return [self['y'].symbol * rho - self['m_dot_in'].symbol]
