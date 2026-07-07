"""Fluid-flow components of the `thermofluid` library, built on `hydrogen.model`.

Components: AmbientInlet, AmbientOutlet, TwoPortSegment, AdiabaticPump,
PressureOutlet, Splitter, PressureSource, PressureVessel, MixingJunction,
LoopBuffer, StraightPipe.

The typed connectors (`FluidPort_phm`, `ThermalPort_TQ`, `PermeationPort_pN`)
live in the sibling `ports` module; the leaky `TwoPortSegment` exposes a heat
port and a permeation leak port, so they are shared across this package.
"""

from __future__ import annotations

import re
import warnings
from typing import Annotated, Callable

import numpy as np
import sympy as sp

from ...medium import CoolPropMedium
from ...model import DifferentialVariable, Model, Parameter, Variable
from ...numerics import G_const
from ...paramspec import ParamSpec, merged_param_specs
from ..control.control_components import RealSignal
from .ports import FluidPort_phm, PermeationPort_pN, ThermalPort_TQ

import math as _math


class _SymLib:
    """sympy math namespace for the axial-dispersion closure (symbolic residual)."""
    Abs = staticmethod(sp.Abs)
    sqrt = staticmethod(sp.sqrt)
    log = staticmethod(sp.log)
    Max = staticmethod(sp.Max)
    Min = staticmethod(sp.Min)


class _NumLib:
    """Plain-float math namespace mirroring `_SymLib` for the run-time check."""
    Abs = staticmethod(abs)
    sqrt = staticmethod(_math.sqrt)
    log = staticmethod(_math.log)
    Max = staticmethod(max)
    Min = staticmethod(min)


def _reference_cp(medium, p=101325.0, T=293.15):
    """Reference isobaric heat capacity cp = (dh/dT)_p [J/(kg*K)] at a standard
    state, used as a *constant* in the Nusselt correlation's Prandtl number and
    the axial-diffusion term.  A fixed cp is adequate for these (it keeps the
    residual free of the medium's T_ph 2nd derivatives) and is robust: the
    analytic partial is tried first, with a central finite difference of
    ``h(p, T)`` as a fallback so the value is never missing."""
    try:
        cp = float(medium.eval_dh_pT_dT(p, T))
        if np.isfinite(cp) and cp > 0.0:
            return cp
    except Exception:
        pass
    try:
        dT = 0.5
        cp = (float(medium.eval_h_pT(p, T + dT))
              - float(medium.eval_h_pT(p, T - dT))) / (2.0 * dT)
        if np.isfinite(cp) and cp > 0.0:
            return cp
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Shared parameter metadata reused across several flow components (single
# source of truth consumed by both `declare_components` and the catalog).
# ---------------------------------------------------------------------------
_SPEC_MULTIPHASE = ParamSpec(
    "Thermodynamic-property mode: 'single' (single-phase, faster) or 'HEM' "
    "(homogeneous equilibrium, handles boiling / flashing).",
    choices=("single", "HEM"), structural=True)
_SPEC_HEAT_PORT = ParamSpec(
    "If true, expose a thermal `wall` port for conjugate heat transfer; if "
    "false the segment is adiabatic (q = 0).", structural=True)
_SPEC_LEAKY = ParamSpec(
    "If true, expose a permeation `leak` port whose mass-flow is subtracted "
    "from the continuity balance; if false the wall is sealed.", structural=True)
_SPEC_Z_IN = ParamSpec("Elevation of the inlet face (gravity head).",
                       unit="m", default=0.0)
_SPEC_Z_OUT = ParamSpec("Elevation of the outlet face (gravity head).",
                        unit="m", default=0.0)
_SPEC_EPSILON = ParamSpec("Absolute wall roughness (friction).", unit="m",
                          default=1e-6)
_SPEC_L = ParamSpec("Flow length.", unit="m", default=1.0)
_SPEC_COUNT = ParamSpec(
    "Number of identical parallel segments this one component represents "
    "(multiplicity >= 1).  A live Parameter (NOT structural): retuning it "
    "scales the extensive flow / heat / leak by the new multiplicity without "
    "re-instantiating the model.", unit="1", default=1.0)


class AmbientInlet(Model):
    """Mass-flow-imposed inlet matched to ambient (p, T) conditions.

    Holds the ambient pressure and temperature as parameters/variables, computes the
    corresponding enthalpy and entropy via the medium's property functions, and emits
    isentropic + energy + continuity equations to determine `(p_out, h_out, m_dot_out)`.

    Port (matches the (p, h, m_dot) convention used everywhere in this package):
        p_out, h_out, m_dot_out   - drives the downstream component

    The component's internal `D` (port diameter) sets the throat area used for the
    kinetic-energy correction in the isentropic energy balance.  Mass flow is set
    by the `m_flow` parameter and propagates through `m_dot_out`.
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        p_ambient: Annotated[float, ParamSpec("Ambient (reservoir) pressure.",
                            unit="Pa")] = 101325,
        T_ambient: Annotated[float, ParamSpec("Ambient (reservoir) "
                            "temperature.", unit="K")] = 293.15,
        m_flow: Annotated[float, ParamSpec("Imposed mass flow rate delivered "
                         "to the outlet.", unit="kg/s")] = 0.1,
        D: Annotated[float, ParamSpec("Port (throat) diameter setting the "
                    "kinetic-energy correction area.", unit="m")] = 0.07,
    ):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self.m_flow = m_flow
        self.D = D
        super().__init__()

    def declare_components(self):
        # Constructor-arg Parameters pull their unit/description from this
        # class's ParamSpec (authored once in the __init__ annotation);
        # computed Parameters/Variables below keep their explicit unit.
        spec = merged_param_specs(type(self))
        self.add_component('p_ambient', Parameter(self.p_ambient, **spec['p_ambient'].param_kwargs()))
        self.add_component('T_ambient', Variable(self.T_ambient, "K"))
        self.add_component('T_ambient_set', Parameter(self.T_ambient, **spec['T_ambient'].param_kwargs()))
        self.add_component('h_ambient', Parameter(self.medium.h_pT(self['p_ambient'].value, self['T_ambient'].value), "J/kg"))
        self.add_component('s_ambient', Parameter(self.medium.s_ph(self['p_ambient'].value, self['h_ambient'].value), "J/kg/K"))
        self.add_component('m_flow', Parameter(self.m_flow, **spec['m_flow'].param_kwargs()))
        self.add_component('D', Parameter(self.D, **spec['D'].param_kwargs()))
        self.add_component('p_out', Variable(self.p_ambient * 0.99, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(self.p_ambient, self.T_ambient) * 0.99, "J/kg"))
        self.add_component('m_dot_out', Variable(self.m_flow, "kg/s"))
        self.add_component('w_out', Variable(0.2, "m/s"))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        A = np.pi * self['D'].symbol ** 2 / 4
        
        # Continuity (mass-flow imposed):
        eq1 = self['m_flow'].symbol + self['m_dot_out'].symbol

        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq_w = self['m_dot_out'].symbol + rho_out * self['w_out'].symbol * A

        h_in = self['h_ambient'].symbol
        s_in = self['s_ambient'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq2 = s_in - s_out
        eq3 = h_in - (self['h_out'].symbol + self['w_out'].symbol ** 2 / 2)
        eq4 = self['T_ambient'].symbol - self['T_ambient_set'].symbol

        return [eq1, eq_w, eq2, eq3, eq4]


class TemperatureInlet(Model):
    """Mass-flow inlet whose enthalpy tracks a commanded *temperature* signal.

    Imposes the mass flow `m_flow` (continuity) and sets the boundary enthalpy
    from a temperature driven through a `RealSignal` INPUT port ``T_set`` -- wire
    it to a `control.CsvTable` (or any signal block) to replay a measured /
    prescribed inlet-temperature history::

        CsvTable(water_inlet) --y--> T_set [TemperatureInlet] --outlet--> pipe

    The enthalpy closure is ``h_out = medium.h_pT(p_out, T_set)`` evaluated at
    the *local* boundary pressure, which the downstream network determines (so
    terminate the line with a `PressureOutlet` to fix the pressure level).  It
    is the live-temperature dual of `AmbientInlet`: that one bakes a fixed
    enthalpy and pins the pressure isentropically; this one imposes mass flow +
    a time-varying enthalpy and lets pressure float.  The kinetic-energy
    correction is dropped (negligible for liquids / low-Mach flows).

    Ports:
        outlet (p, h, m_dot)  - drives the downstream component
        T_set  (signal in)    - commanded inlet temperature [K]
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        m_flow: Annotated[float, ParamSpec("Imposed mass flow rate delivered "
                         "to the outlet.", unit="kg/s")] = 0.1,
        p_init: Annotated[float, ParamSpec("Initial boundary-pressure guess "
                         "(the level is set downstream).", unit="Pa")]
                = 101325.0,
        T_init: Annotated[float, ParamSpec("Initial temperature: seeds the "
                         "enthalpy guess and the `T_set` signal.", unit="K")]
                = 293.15,
    ):
        self.medium = medium
        self.m_flow = m_flow
        self.p_init = p_init
        self.T_init = T_init
        self._h_init = float(medium.eval_h_pT(p_init, T_init))
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('m_flow', Parameter(self.m_flow, **spec['m_flow'].param_kwargs()))
        # Commanded inlet temperature: an algebraic Variable closed by the
        # `T_set` signal input (require_connection -> an unwired port is flagged).
        self.add_component('T_set', Variable(self.T_init, "K"))
        self.add_component('p_out', Variable(self.p_init, "Pa"))
        self.add_component('h_out', Variable(self._h_init, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_port('T_set', RealSignal.as_input(
            self, self['T_set'], name='T_set'))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        # Continuity (mass flow imposed); "flow into me" => m_dot_out = -m_flow.
        eq_cont = self['m_flow'].symbol + self['m_dot_out'].symbol
        # Enthalpy from the commanded temperature at the local boundary pressure.
        eq_h = self['h_out'].symbol - self.medium.h_pT(
            self['p_out'].symbol, self['T_set'].symbol)
        return [eq_cont, eq_h]


class AmbientOutlet(Model):
    """Outlet to ambient pressure (and matched isentropic conditions).

    Same equations as `AmbientInlet` minus the `T_ambient` constraint.
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        p_ambient: Annotated[float, ParamSpec("Ambient (reservoir) pressure.",
                            unit="Pa")] = 101325,
        T_ambient: Annotated[float, ParamSpec("Ambient (reservoir) "
                            "temperature.", unit="K")] = 293.15,
        m_flow: Annotated[float, ParamSpec("Imposed mass flow rate drawn from "
                         "the inlet.", unit="kg/s")] = 0.1,
        D: Annotated[float, ParamSpec("Port (throat) diameter setting the "
                    "kinetic-energy correction area.", unit="m")] = 0.07,
    ):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self.m_flow = m_flow
        self.D = D
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('p_ambient', Parameter(self.p_ambient, **spec['p_ambient'].param_kwargs()))
        self.add_component('T_ambient', Parameter(self.T_ambient, **spec['T_ambient'].param_kwargs()))
        self.add_component('h_ambient', Parameter(self.medium.h_pT(self['p_ambient'].value, self['T_ambient'].value), "J/kg"))
        self.add_component('s_ambient', Parameter(self.medium.s_ph(self['p_ambient'].value, self['h_ambient'].value), "J/kg/K"))
        self.add_component('m_flow', Parameter(self.m_flow, **spec['m_flow'].param_kwargs()))
        self.add_component('D', Parameter(self.D, **spec['D'].param_kwargs()))
        self.add_component('p_out', Variable(self.p_ambient * 0.99, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(self.p_ambient, self.T_ambient) * 0.99, "J/kg"))
        self.add_component('m_dot_out', Variable(self.m_flow, "kg/s"))
        self.add_component('w_out', Variable(0.2, "m/s"))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        A = np.pi * self['D'].symbol ** 2 / 4
        eq1 = self['m_flow'].symbol + self['m_dot_out'].symbol

        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq_w = self['m_dot_out'].symbol + rho_out * self['w_out'].symbol * A

        h_in = self['h_ambient'].symbol
        s_in = self['s_ambient'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq2 = s_in - s_out
        eq3 = h_in - (self['h_out'].symbol + self['w_out'].symbol ** 2 / 2)

        return [eq1, eq_w, eq2, eq3]


class ClosedEnd(Model):
    """Dead-end / capped boundary: zero mass flow, pressure & enthalpy free.

    Terminates a duct with a solid cap.  Imposes only `m_dot = 0` at the
    boundary plane and lets the upstream component's momentum / energy balance
    determine `(p, h)` there.  Use it to seal one end of a pressurised line so
    the only mass leaving is whatever exits through other ports.

    Port (the standard `(p, h, m_dot)` triple):
        p_in, h_in, m_dot_in    - the capped face (m_dot pinned to zero)
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        p_init: Annotated[float, ParamSpec("Initial pressure guess at the "
                         "capped face.", unit="Pa")] = 101325.0,
        T_init: Annotated[float, ParamSpec("Initial temperature guess at the "
                         "capped face.", unit="K")] = 293.15,
    ):
        self.medium = medium
        self.p_init = p_init
        self.T_init = T_init
        self._h_init = float(medium.eval_h_pT(p_init, T_init))
        super().__init__()

    def declare_components(self):
        self.add_component('p_in', Variable(self.p_init, "Pa"))
        self.add_component('h_in', Variable(self._h_init, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        return [self['m_dot_in'].symbol] # m_dot_in = 0


class TwoPortSegment(Model):
    """One discrete control volume of a duct with continuity, momentum, and energy.

    Port convention (the standard `(p, h, m_dot)` triple used everywhere):
        p_in,  h_in,  m_dot_in     - upstream face
        p_out, h_out, m_dot_out    - downstream face

    Sign convention -- Modelica "flow into me":
        Every face's `m_dot` is positive when fluid ENTERS the component
        through that face.  Forward axial flow (in-face -> out-face) is
        therefore reported as `m_dot_in > 0`, `m_dot_out < 0`, and the
        continuity equation `m_dot_in + m_dot_out == 0` holds (Kirchhoff
        on the control volume).  The unsigned "axial flow rate" used
        internally by momentum / energy terms is recovered as
        `m_dot_axial = m_dot_in = -m_dot_out`.
    Two **internal** algebraic Variables `w_in` / `w_out` carry the face
    velocities so the friction, momentum, kinetic-energy, and heat-transfer
    expressions can reference them as leaf SymPy symbols (lean lambdified
    code).  The link between them is a pair of closures
    `m_dot = rho * w * A` per face.

    `f_factor_func(Re, epsilon, Dh)` returns the friction factor symbolically.

    `q_inflow_func(w, p, h, rho, T, mu, k, fr, T_wall, Dh, area)` returns the
    heat input rate.

    Each of the seven geometry slots (`A_in`, `A_out`, `P_in`, `P_out`, `z_in`,
    `z_out`, `L`) may be passed either as a plain Python scalar OR as an
    existing `Parameter` instance owned by a parent `Model`.

    Face thermodynamic properties (`T`, `rho`, `mu`, `k` at the in/out faces)
    are exposed as explicit algebraic Variables with closure equations
    `rho_in - rho_ph(p_in, h_in) == 0` (etc.).

    There are three orthogonal (not affecting each other) interface toggles:
    Switching these toggles invalidates the cached template equations.
    `multiphase`:
       - `single`: single-phase (rho, mu, k are functions of p and h).
          Deals faster with single-phase flow. Does not handle boiling or flashing well.

      - 'HEM': homogeneous equilibrium model (rho, mu, k are functions of p, h, and quality).
        Handles boiling and flashing well. Leans on the line serach of Newton
        solver to get the fast and reliable marching within dome region.
     
    `heat_port`:
        - `True`: a `wall` `ThermalPort_TQ` is exposed, and the wall-temperature
          is closed by the connected boundary which must be connected otherwize the system
          is underdetermined and singular.
        - `False`: adiabatic (T_wall = T_avg, q = 0).
           The `ThermalPort_TQ` is removed and the segment is adiabatic.
    `leaky`:
        - `True`: a permeation `leak` port is exposed, and the leak mass-flow
          is subtracted from continuity.
        - `False`: the permeation `leak` port is removed and the segment is not leaky.
    """

    #: Allowed values for the `multiphase` flag.
    _MULTIPHASE_MODES = ("single", "HEM")

    def __init__(
        self,
        medium: CoolPropMedium,
        A_in: Annotated[float, ParamSpec("Inlet-face flow area.", unit="m^2",
                       default=1e-3)],
        A_out: Annotated[float, ParamSpec("Outlet-face flow area.", unit="m^2",
                        default=1e-3)],
        P_in: Annotated[float, ParamSpec("Inlet-face wetted perimeter "
                       "(hydraulic diameter).", unit="m", default=0.1)],
        P_out: Annotated[float, ParamSpec("Outlet-face wetted perimeter "
                        "(hydraulic diameter).", unit="m", default=0.1)],
        z_in: Annotated[float, _SPEC_Z_IN],
        z_out: Annotated[float, _SPEC_Z_OUT],
        L: Annotated[float, _SPEC_L],
        epsilon: Annotated[float, _SPEC_EPSILON],
        f_factor_func: Annotated[Callable, ParamSpec(
            "Callable f(Re, epsilon, Dh) returning the Darcy friction factor "
            "symbolically.")],
        q_inflow_func: Annotated[Callable, ParamSpec(
            "Callable returning the radial heat input rate symbolically.")],
        multiphase: Annotated[str, _SPEC_MULTIPHASE] = "single",
        heat_port: Annotated[bool, _SPEC_HEAT_PORT] = False,
        leaky: Annotated[bool, _SPEC_LEAKY] = False,
        count: Annotated[float, _SPEC_COUNT] = 1.0,
    ):
        self.medium = medium
        if multiphase not in self._MULTIPHASE_MODES:
            raise ValueError(
                f"multiphase must be one of {self._MULTIPHASE_MODES}, got {multiphase!r}")
        self.multiphase = multiphase
        self.heat_port = bool(heat_port)  # adds radial heat transfer term
        self.leaky = bool(leaky)          # adds radial permeation mass-flow term
        # Multiplicity: this one segment stands in for `count` identical parallel
        # segments.  A live `Parameter` (not baked into `A`/`P`) so the parallel
        # multiplicity can be retuned without re-instantiating: its symbol scales
        # the EXTENSIVE terms only -- the `m_dot = rho*A*w` face closures and the
        # convective contact area -- leaving every intensive quantity (velocity,
        # Re, friction, specific enthalpy, hydraulic diameter) unchanged.
        self.count = count
        self.A_in = A_in
        self.A_out = A_out
        self.P_in = P_in
        self.P_out = P_out
        self.z_in = z_in
        self.z_out = z_out
        self.L = L
        self.epsilon = epsilon
        self.f_factor_func = f_factor_func
        self.q_inflow_func = q_inflow_func
        h_std = float(medium.eval_h_pT(101325.0, 293.15))
        self._h_std = h_std
        self._rho_std = float(medium.eval_rho_ph(101325.0, h_std))
        self._mu_std = float(medium.eval_mu_ph(101325.0, h_std))
        self._k_std = float(medium.eval_k_ph(101325.0, h_std))
        # Reference isobaric heat capacity used as a constant in the Nusselt
        # correlation's Prandtl number (see `_reference_cp`).
        self._cp_std = _reference_cp(medium)
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self._h_std, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('w_in', Variable(0.1, "m/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self._h_std, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_component('w_out', Variable(0.1, "m/s"))
        self.add_component('T_in',   Variable(293.15,         "K"))
        self.add_component('T_out',  Variable(293.15,         "K"))
        self.add_component('rho_in', Variable(self._rho_std,  "kg/m^3"))
        self.add_component('rho_out',Variable(self._rho_std,  "kg/m^3"))
        self.add_component('mu_in',  Variable(self._mu_std,   "Pa*s"))
        self.add_component('mu_out', Variable(self._mu_std,   "Pa*s"))
        self.add_component('k_in',   Variable(self._k_std,    "W/m/K"))
        self.add_component('k_out',  Variable(self._k_std,    "W/m/K"))
        self.add_component('count', Parameter(self.count, **spec['count'].param_kwargs()))
        self.add_component('A_in', Parameter(self.A_in, **spec['A_in'].param_kwargs()))
        self.add_component('A_out', Parameter(self.A_out, **spec['A_out'].param_kwargs()))
        self.add_component('P_in', Parameter(self.P_in, **spec['P_in'].param_kwargs()))
        self.add_component('P_out', Parameter(self.P_out, **spec['P_out'].param_kwargs()))
        self.add_component('z_in', Parameter(self.z_in, **spec['z_in'].param_kwargs()))
        self.add_component('z_out', Parameter(self.z_out, **spec['z_out'].param_kwargs()))
        self.add_component('L', Parameter(self.L, **spec['L'].param_kwargs()))
        self.add_component('T_wall', Variable(293.15, "K"))
        self.add_component('q_inflow', Variable(0.0, "W"))

        if self.heat_port: # expose port for radial heat transfer
            self.add_port('wall', ThermalPort_TQ(
                self,
                channels={'T': self['T_wall'], 'Q_dot': self['q_inflow']},
                flow_orientation='in',
                require_connection=True,
            ))

        if self.leaky: # expose port for radial leak (e.g. permeation) mass-flow
            self.add_component('m_dot_leak', Variable(0.0, "kg/s"))
            self.add_port('leak', PermeationPort_pN(
                self,
                channels={'p_partial': self['p_in'], 'm_dot_leak': self['m_dot_leak']},
                flow_orientation='in',
                require_connection=True,
            ))
        
        # Always expose both inlet and outlet ports.
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def _property_funcs(self):
        """Return the `(T_ph, rho_ph, mu_ph, k_ph)` property callables for this
        segment's `multiphase` mode.

        * ``"single"`` -> the medium's plain CoolProp `(p, h)` properties.
        * ``"HEM"``    -> the smooth homogeneous-equilibrium variants
          (`*_ph_hem`): same values, but with smoothed, consistent partials so
          the Jacobian stays well-conditioned through the saturation dome.
        """
        m = self.medium
        if self.multiphase == "HEM":
            try:
                return m.T_ph_hem, m.rho_ph_hem, m.mu_ph_hem, m.k_ph_hem
            except AttributeError as exc:
                raise AttributeError(
                    f"multiphase='HEM' requires a medium exposing *_ph_hem "
                    f"property functions (e.g. CoolPropMedium); {type(m).__name__} "
                    f"does not."
                ) from exc
        return m.T_ph, m.rho_ph, m.mu_ph, m.k_ph

    def declare_equations(self):
        
        # convenience local references for writing the equations:
        T_in    = self['T_in'].symbol
        T_out   = self['T_out'].symbol
        rho_in  = self['rho_in'].symbol
        rho_out = self['rho_out'].symbol
        mu_in   = self['mu_in'].symbol
        mu_out  = self['mu_out'].symbol
        k_in    = self['k_in'].symbol
        k_out   = self['k_out'].symbol
        w_in = self['w_in'].symbol
        w_out = self['w_out'].symbol
        m_dot_in = self['m_dot_in'].symbol
        m_dot_out = self['m_dot_out'].symbol

        # useful expressions for the equations:
        p_avg = (self['p_in'].symbol + self['p_out'].symbol) / 2
        h_avg = (self['h_in'].symbol + self['h_out'].symbol) / 2
        T_avg = (T_in + T_out) / 2
        mu_avg = (mu_in + mu_out) / 2
        rho_avg = (rho_in + rho_out) / 2
        k_avg = (k_in + k_out) / 2
        w_avg = (w_in + w_out) / 2
        A_avg = (self['A_in'].symbol + self['A_out'].symbol) / 2
        m_dot_avg = (m_dot_in - m_dot_out) / 2
        P_avg = (self['P_in'].symbol + self['P_out'].symbol) / 2
        Dh_in = 4 * self['A_in'].symbol / self['P_in'].symbol
        Dh_out = 4 * self['A_out'].symbol / self['P_out'].symbol
        Dh_avg = (Dh_in + Dh_out) / 2
        Re_avg = rho_avg * abs(w_avg) * Dh_avg / mu_avg + 1

        # Mass-flow continuity.
        eq_continuity = m_dot_in + m_dot_out
        if self.leaky:
            eq_continuity = eq_continuity + self['m_dot_leak'].symbol

        # `count` parallel segments: the SAME per-segment velocity `w` flows
        # through `count` times the single-segment area, so the EXTENSIVE mass
        # flow scales by `count` while `w` (and hence Re, friction, Dh) stays
        # the per-pipe intensive value.  Scaling the face closures here (rather
        # than baking `count` into `A`) keeps `count` a live, re-tunable
        # Parameter.
        count = self['count'].symbol

        # m_dot <-> w closures as equations 
        # adds variables but these vars are used multiple times so it reduces
        # complexity of many expresions -> simulations speeds up.
        eq_w_in = m_dot_in - count * rho_in * self['A_in'].symbol * w_in
        eq_w_out = m_dot_out + count * rho_out * self['A_out'].symbol * w_out

        # Momentum conservation equation.
        # Hook allows to resuse this class to implement different components
        # (e.g. valves, pumps, etc.).
        f_avg = self.f_factor_func(Re_avg, self.epsilon, Dh_avg)
        eq_momentum = self._momentum_eq(
            p_in=self['p_in'].symbol, p_out=self['p_out'].symbol,
            A_in=self['A_in'].symbol, A_out=self['A_out'].symbol, A_avg=A_avg,
            rho_avg=rho_avg, w_in=w_in, w_out=w_out, w_avg=w_avg,
            m_dot_avg=m_dot_avg, f_avg=f_avg, Dh_avg=Dh_avg,
        )

        # Total wall-contact area over the `count` parallel segments, so the
        # radial heat `q` scales by `count` too -- keeping the SPECIFIC heat
        # `q_specific = q / m_dot` (and the energy balance) intensive.
        area_conv = count * P_avg * self['L'].symbol

        #TODO: comment this section.
        q = self.q_inflow_func(
            w_avg, p_avg, h_avg, rho_avg, T_avg, mu_avg, k_avg, 
            f_avg, self['T_wall'].symbol, Dh_avg, area_conv
        )
        
        # regularization to avoid division by zero
        w_eps = 1e-4
        m_eps = 1e-6
        sign_w = w_avg / sp.sqrt(w_avg ** 2 + w_eps ** 2)
        m_dot_reg = sp.sqrt(m_dot_avg ** 2 + m_eps ** 2)
        q_specific = sign_w * q / m_dot_reg

        # energy conservation equation
        eq_energy = self['h_in'].symbol + w_in ** 2 / 2 + q_specific - (self['h_out'].symbol + w_out ** 2 / 2)
        eq_q_diag = self['q_inflow'].symbol - q

        # face property closures 
        # adds variables but removes complexity at runtime -> simulations speeds up
        p_in_sym = self['p_in'].symbol
        p_out_sym = self['p_out'].symbol
        h_in_sym = self['h_in'].symbol
        h_out_sym = self['h_out'].symbol
        T_ph, rho_ph, mu_ph, k_ph = self._property_funcs()
        eq_T_in    = T_in    - T_ph(p_in_sym, h_in_sym)
        eq_rho_in  = rho_in  - rho_ph(p_in_sym, h_in_sym)
        eq_mu_in   = mu_in   - mu_ph(p_in_sym, h_in_sym)
        eq_k_in    = k_in    - k_ph(p_in_sym, h_in_sym)
        eq_T_out   = T_out   - T_ph(p_out_sym, h_out_sym)
        eq_rho_out = rho_out - rho_ph(p_out_sym, h_out_sym)
        eq_mu_out  = mu_out  - mu_ph(p_out_sym, h_out_sym)
        eq_k_out   = k_out   - k_ph(p_out_sym, h_out_sym)

        # base set of equations (conservatoin + closures)
        eqs = [
            eq_continuity, eq_w_in, eq_w_out, eq_momentum, eq_energy, eq_q_diag,
            eq_T_in,  eq_rho_in,  eq_mu_in,  eq_k_in,
            eq_T_out, eq_rho_out, eq_mu_out, eq_k_out,
        ]

        # without heat port the wall temperature equals the mean fluid temperature
        if not self.heat_port:
            T_avg = (self['T_in'].symbol + self['T_out'].symbol) / 2
            eqs.append(self['T_wall'].symbol - T_avg)
        return eqs

    def _momentum_eq(self, *, p_in, p_out, A_in, A_out, A_avg, rho_avg,
                     w_in, w_out, w_avg, m_dot_avg, f_avg, Dh_avg):
        """Default momentum residual: distributed duct balance.

        Force balance on the control volume in the axial direction:

            p_in*A_in - p_out*A_out
                - f_avg*(L/Dh)*(rho/2)*|w|*w * A_avg     (Darcy friction)
                - m_dot_avg*(w_out - w_in)               (momentum flux)
                - g*(z_out - z_in)*A_avg*rho             (buoyancy)
            == 0

        Subclasses (e.g. `Valve`) override this to substitute a different
        pressure/flow relation while reusing every other segment equation.
        """
        delta_P_friction = f_avg * (self['L'].symbol / Dh_avg) * (rho_avg * abs(w_avg) * w_avg / 2)
        momentum_flux = m_dot_avg * (w_out - w_in)
        buoyancy_force = -G_const * (self['z_out'].symbol - self['z_in'].symbol) * A_avg * rho_avg
        return p_in * A_in - p_out * A_out - delta_P_friction * A_avg - momentum_flux + buoyancy_force

class StraightPipe(Model):
    """1D pipe split into `n_segments` equal-length `TwoPortSegment`s with optional heat transfer and hydrogen permeation.

    Uses the Churchill correlation for the friction factor and a smoothed
    Gnielinski / laminar Nusselt blend for the wall heat-transfer coefficient.

    Two orthogonal segment interfaces are exposed via per-segment ports:

      * `heat_port=True` -> a `wall` `ThermalPort_TQ` per segment, collected by
        `segment_wall_ports` (conjugate heat transfer).
      * `leaky=True`     -> a `leak` `PermeationPort_pN` per segment, collected
        by `segment_leak_ports` (hydrogen permeation through the wall).

    The friction factor is the Churchill correlation evaluated on a Reynolds
    number floored away from zero (see `TwoPortSegment`), so a dead-ended /
    stagnant line (Re -> 0) stays well-conditioned regardless of `leaky`.

    A pipe may be heated, leaky, both, or neither.

    `count=N` simulates `N` identical pipes in parallel as one component: every
    segment scales its extensive flow / heat / leak by the shared `count`
    Parameter (keeping the hydraulic diameter, so velocity / Reynolds number /
    friction / per-pipe pressure drop are unchanged), while the total mass flow,
    wall heat, and permeation scale by `N`.  `count` is a LIVE Parameter -- it
    can be retuned (`set_param`) without re-instantiating -- and may be a plain
    scalar or a parent-owned Parameter shared with the matching walls /
    boundaries (see `Pipe`).
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Pipe bore (inner) diameter.", unit="m",
                    default=0.01)],
        L: Annotated[float, ParamSpec("Total pipe length.", unit="m",
                    default=1.0)],
        epsilon: Annotated[float, _SPEC_EPSILON],
        z_in: Annotated[float, _SPEC_Z_IN],
        z_out: Annotated[float, _SPEC_Z_OUT],
        n_segments: Annotated[int, ParamSpec("Number of equal-length finite-"
                             "volume segments (>= 1).", unit="1",
                             structural=True)] = 3,
        heat_port: Annotated[bool, ParamSpec("If true, each segment exposes a "
                            "thermal `wall` port for conjugate heat transfer.",
                            structural=True)] = False,
        adiabatic: Annotated[bool | None, ParamSpec("Deprecated legacy flag; "
                            "use `heat_port` instead. None leaves it unset.")]
                   = None,
        multiphase: Annotated[str, _SPEC_MULTIPHASE] = "single",
        leaky: Annotated[bool, ParamSpec("If true, each segment exposes a "
                        "permeation `leak` port.", structural=True)] = False,
        count: Annotated[float, ParamSpec("Number of identical parallel pipes "
                        "this one component represents (multiplicity >= 1); "
                        "scales the flow area and wetted perimeter by N so the "
                        "hydraulic diameter, velocity, friction and per-pipe "
                        "pressure drop are unchanged while the total mass flow / "
                        "heat / leak scale by N -- N pipes for one pipe's "
                        "equations.  A live Parameter: retunable without a "
                        "rebuild.", unit="1", default=1.0)] = 1.0,
    ):
        self.medium = medium
        if multiphase not in TwoPortSegment._MULTIPHASE_MODES:
            raise ValueError(
                f"multiphase must be one of {TwoPortSegment._MULTIPHASE_MODES}, "
                f"got {multiphase!r}")
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments!r}")
        # `count` may be a plain scalar or a parent-owned `Parameter` (the `Pipe`
        # assembly shares ONE `count` symbol across the pipe, its walls and its
        # boundaries); validate against the underlying numeric value.
        count_num = getattr(count, 'value', count)
        if not (count_num >= 1):
            raise ValueError(
                f"StraightPipe: count must be >= 1, got {count!r}")
        # Forwarded verbatim to every segment so the whole pipe shares one
        # thermodynamic-property mode (see `TwoPortSegment.multiphase`).
        self.multiphase = multiphase
        self.D = D
        self.L = L
        self.epsilon = epsilon
        self.z_in = z_in
        self.z_out = z_out
        self.n_segments = n_segments
        # `heat_port` is the single switch for the segment thermal interface:
        #   * False (default): each segment is a plain adiabatic `TwoPortSegment`
        #     (T_wall = T_avg, q = 0); the pipe has no thermal connectivity.
        #   * True: each segment is a heated `TwoPortSegment` exposing a `wall`
        #     `ThermalPort_TQ`; the convective heat is driven by whatever is
        #     wired to those ports (a `CylindricalWall`, a `FixedTemperature`,
        #     ...).  Reproduce the old "convect to a fixed wall temperature"
        #     behaviour by setting heat_port=True and connecting a
        #     `FixedTemperature` to each segment wall port.
        #
        # `adiabatic` is the deprecated legacy flag.  It only ever toggled the
        # heat source against a fixed 293.15 K `T_wall` Parameter, which has
        # been removed; map it onto `heat_port` and warn.
        if adiabatic is not None:
            if adiabatic and heat_port:
                raise ValueError(
                    "StraightPipe: pass either the deprecated `adiabatic=` or the new "
                    "`heat_port=`, not both."
                )
            if not adiabatic:
                warnings.warn(
                    "StraightPipe(adiabatic=False): the legacy fixed-293.15 K wall "
                    "heating has been removed. Use heat_port=True and connect a thermal "
                    "boundary (e.g. FixedTemperature) or a wall to each segment's `wall` "
                    "port to model heat transfer.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # adiabatic=True simply means "no heat port" == default.
        self.heat_port = heat_port
        # `leaky=True`: every segment exposes a gas-permeation `leak`
        # `PermeationPort_pN` (collected via `segment_leak_ports`), so a parent
        # wires each to its own leaky `CylindricalWall`.  Orthogonal to
        # `heat_port` -- a pipe may be heated, leaky, both, or neither.
        self.leaky = bool(leaky)
        # Multiplicity: this one pipe stands in for `count` identical parallel
        # pipes.  `count` is a live `Parameter` (own, or shared from the `Pipe`
        # assembly): it is passed down to every `TwoPortSegment`, whose face
        # closures scale the extensive `m_dot` / heat / leak by it while keeping
        # `Dh = 4A/P = D`, velocity, Re, friction and per-pipe pressure drop
        # unchanged.  Because it is a Parameter (not baked into `A`/`P`), the
        # multiplicity can be retuned without re-instantiating the model.
        self.count = count
        # Std-state reference properties for the Nusselt-correlation Prandtl
        # number (a constant cp; see `_reference_cp`).  `get_q_inflow` is a
        # method of this pipe (passed down to each `TwoPortSegment` as their
        # `q_inflow_func`), so the reference values live here.
        h_std = float(self.medium.eval_h_pT(101325.0, 293.15))
        self._mu_std = float(self.medium.eval_mu_ph(101325.0, h_std))
        self._k_std = float(self.medium.eval_k_ph(101325.0, h_std))
        self._cp_std = _reference_cp(self.medium)
        super().__init__()

    def get_churchill_f_factor(self, Re, epsilon, D):
        term1 = (8.0 / Re) ** 12
        A = (-2.457 * sp.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / Re) ** 16
        term2 = 1.0 / (A + B) ** 1.5
        return (term1 + term2) ** (1.0 / 12.0) * 8

    def calculate_nu(self, Re, Pr, fr):
        # Laminar fully-developed Nu = 3.66 below Re~2300; Gnielinski turbulent
        # correlation above Re~3100; linear blend across the transition band.
        nu_turb = (fr / 8) * (Re - 1000) * Pr / (
            1 + 12.7 * (fr / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        if Re <= 2300:
            return 3.66
        if Re >= 3100:
            return nu_turb
        phi = (Re - 2300.0) / 800.0
        return (1.0 - phi) * 3.66 + phi * nu_turb

    def calculate_nu_smooth(self, Re, Pr, fr):
        # Smooth (differentiable) Nusselt number: laminar fully-developed
        # Nu = 3.66 below Re~2300, blended into the Gnielinski turbulent
        # correlation above Re~3100.
        #
        # NOTE: earlier revisions had the two regimes INVERTED -- the Gnielinski
        # turbulent formula was applied in the laminar branch and the laminar
        # constant 3.66 was returned for turbulent Re.  That under-predicted the
        # turbulent Nusselt number by ~30x (e.g. Nu=3.66 instead of ~106 at
        # Re=1.4e4), leaving heated pipes far too weakly coupled to their walls.
        nu_lam = 3.66
        nu_turb = (fr / 8) * (Re - 1000) * Pr / (
            1 + 12.7 * (fr / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        phi = sp.Max(0.0, sp.Min(1.0, (Re - 2300.0) / 800.0))
        return (1.0 - phi) * nu_lam + phi * nu_turb

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        # `Dh` (hydraulic diameter; == D for a circular pipe) and `area` (the
        # inner wetted surface of ONE segment, = mean perimeter x segment
        # length) arrive as SymPy expressions built from the SEGMENT's own
        # geometry Parameter symbols.  Using them -- instead of the parent
        # pipe's Python floats `self.D` / `self.L` / `self.n_segments` --
        # keeps this heat term free of baked-in literals, so the cached
        # heated-segment template is instance-invariant across pipes of
        # different diameter or length.  (`area` already accounts for the
        # per-segment length, so there is no `n_segments` over-counting.)
        Re = w * Dh * rho / mu
        # Prandtl number Pr = mu*cp/k (cp at the std reference state; see
        # `_reference_cp`).  Earlier this read `mu*rho/k`, which is
        # dimensionally not the Prandtl number and badly under-predicted Nu
        # for liquids (cp >> rho-scaled value).
        cp = self._cp_std if self._cp_std else self._k_std / self._mu_std
        Pr = mu * cp / k
        nu = self.calculate_nu_smooth(Re, Pr, fr)
        alpha = nu * k / Dh
        return alpha * area * (T_wall - T)

    def get_q_inflow_adiabatic(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        return 0.0

    def declare_components(self):
        # Pipe-level constitutive Parameters.  Hoisting these out of the
        # per-segment `TwoPortSegment`s and passing them down as shared
        # `Parameter` references is what makes every segment's equations
        # reference the SAME SymPy symbols for area, perimeter, and segment
        # length -- a precondition for `Model.remove_duplicate_equations`
        # to recognise the per-face `m_dot = rho*A*w` closures of adjacent
        # segments as structurally identical (apart from a single `w` leaf)
        # and collapse them.
        spec = merged_param_specs(type(self))
        self.add_component('D', Parameter(self.D, **spec['D'].param_kwargs()))
        self.add_component('L', Parameter(self.L, **spec['L'].param_kwargs()))
        self.add_component('epsilon', Parameter(self.epsilon, **spec['epsilon'].param_kwargs()))
        self.add_component('z_in', Parameter(self.z_in, **spec['z_in'].param_kwargs()))
        self.add_component('z_out', Parameter(self.z_out, **spec['z_out'].param_kwargs()))
        # Derived shared geometry.  We register them as their own Parameters
        # (rather than building `pi*D**2/4` as a SymPy expression every time
        # a segment references it) because Parameter symbols are flat
        # leaves: the per-segment friction / momentum / energy expression
        # trees stay shallow, and the equation-dedup signature comparison
        # is a single-symbol hash rather than a full subtree walk.
        # SINGLE-pipe flow area and wetted perimeter.  The `count` multiplicity
        # is NOT folded in here (that would bake it into a constant `Parameter`
        # leaf and force a rebuild to change it); instead each segment scales
        # its extensive terms by the shared `count` Parameter below.  Keeping
        # `A`/`P` per-pipe leaves `Dh = 4A/P = D` and every intensive quantity
        # identical to a single pipe.
        A_value = np.pi * self.D ** 2 / 4
        P_value = np.pi * self.D
        L_segment_value = self.L / self.n_segments
        # Multiplicity shared by every segment (own fresh Parameter, or an alias
        # into the parent assembly's `count` when one was passed down).
        self.add_component('count', Parameter(self.count, **spec['count'].param_kwargs()))
        self.add_component('A', Parameter(A_value, "m^2"))
        self.add_component('P', Parameter(P_value, "m"))
        self.add_component('L_segment', Parameter(L_segment_value, "m"))

        # Pipe-level port Variables.
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

        # `N+1` axial stations shared between adjacent segments: segment `k`'s
        # `z_out` and segment `k+1`'s `z_in` reference the same `Parameter`.
        # Not strictly required for the dedup pass to fire on the face
        # closures (which don't depend on `z`), but it halves the elevation-
        # Parameter inventory of a pipe and keeps a single source of truth
        # for each interface elevation.
        dz = (self.z_out - self.z_in) / self.n_segments
        for j in range(self.n_segments + 1):
            self.add_component(f'z_{j}', Parameter(self.z_in + j * dz, "m"))

        A_param = self['A']
        P_param = self['P']
        L_seg_param = self['L_segment']
        # `heat_port` is a per-segment flag on `TwoPortSegment` (registered in
        # its `_cache_key_flags`, so the per-class equation cache stays correct
        # in models mixing heated and adiabatic pipes).  It selects the
        # wall-temperature closure; the convective heat CORRELATION is injected
        # separately via `q_inflow_func` (zero heat when adiabatic).
        q_f = self.get_q_inflow if self.heat_port else self.get_q_inflow_adiabatic
        # Single Churchill friction factor for every pipe.  Zero-flow
        # robustness is handled once, by the `Re_avg` floor in
        # `TwoPortSegment.declare_equations`, so a dead-ended / stagnant line
        # (leaky or not) keeps a finite friction factor and a non-singular
        # Jacobian as Re -> 0.
        fr_f = self.get_churchill_f_factor
        for i in range(self.n_segments):
            self.add_component(
                f'pipe_segment_{i}',
                TwoPortSegment(
                    self.medium,
                    A_in=A_param, A_out=A_param,
                    P_in=P_param, P_out=P_param,
                    z_in=self[f'z_{i}'], z_out=self[f'z_{i + 1}'],
                    L=L_seg_param,
                    epsilon=self.epsilon,
                    f_factor_func=fr_f,
                    q_inflow_func=q_f,
                    multiphase=self.multiphase,
                    heat_port=self.heat_port,
                    leaky=self.leaky,
                    count=self['count'],
                ),
            )

    def declare_equations(self):
        # Wire the pipe-level (p, h, m_dot)_in/out ports to the matching ports of
        # the first / last segment.  Using `segment_0.{h,m_dot}_in` (rather than
        # `_out`) is important: the inlet ports of the pipe must represent the
        # axial station at the actual pipe entrance, otherwise upstream
        # connections (a `Splitter`, another pipe, a vessel) compare values
        # across mismatched stations and the resulting "mass conservation"
        # through the wiring is off by the small frictional density change
        # across the first segment.
        #
        # All of these are pure variable-equality constraints, so we route them
        # through `add_connection` (union-find at instantiate time) rather than
        # building a sympy `Add` per pair and letting the trivial reducer eat them.
        for port in ('p_in', 'h_in', 'm_dot_in'):
            self.add_connection(self[port], self['pipe_segment_0'][port])
        # Inter-segment wiring.  We deliberately do NOT union `w` between
        # adjacent segments via `add_connection`: each segment owns a pair
        # of closure equations `m_dot = rho * A * w` (one per face), and at
        # an internal interface both equations reduce to the same statement
        # after `(p, h, m_dot)` are unioned -- collapsing the two `w`
        # symbols via `add_connection` would leave the system over-
        # determined by one equation per shared face.
        #
        # Instead, `Model.remove_duplicate_equations` (run during
        # `instantiate()`, controlled by the same-named flag) detects the
        # two face closures as structurally identical apart from their `w`
        # leaf and unifies them -- removing both the redundant `w` symbol
        # AND its closure equation per internal interface (`N - 1`
        # eliminations per pipe).  For this to fire, both segments must
        # reference the SAME SymPy symbols for area / perimeter /
        # segment-length, which is exactly what `declare_components` above
        # wires up via shared `A` / `P` / `L_segment` Parameters.
        # Under "flow into me", the m_dot at seg_i's out-face and seg_{i+1}'s
        # in-face describe the same physical interface but from opposite
        # control volumes (each measures fluid flowing INTO its own
        # component) -- they're equal in magnitude with opposite sign, so
        # we route the m_dot wire through signed UF (`sign=-1`) while p
        # and h stay direct (across variables -- the face's pressure and
        # enthalpy are single-valued regardless of which side looks at it).
        for i in range(self.n_segments - 1):
            self.add_connection(
                self[f'pipe_segment_{i}']['p_out'],
                self[f'pipe_segment_{i + 1}']['p_in'],
            )
            self.add_connection(
                self[f'pipe_segment_{i}']['h_out'],
                self[f'pipe_segment_{i + 1}']['h_in'],
            )
            self.add_connection(
                self[f'pipe_segment_{i}']['m_dot_out'],
                self[f'pipe_segment_{i + 1}']['m_dot_in'],
                sign=-1,
            )
        last = f'pipe_segment_{self.n_segments - 1}'
        for port in ('p_out', 'h_out', 'm_dot_out'):
            self.add_connection(self[port], self[last][port])
        return []

    @property
    def segment_wall_ports(self):
        """List of the per-segment `wall` `ThermalPort_TQ` (only when
        `heat_port=True`).  A parent model connects these to walls/boundaries,
        e.g. `self.connect(pipe.segment_wall_ports[i], wall_i.ports['port_a'])`.
        """
        if not self.heat_port:
            raise AttributeError(
                "StraightPipe was built with heat_port=False; it has no segment "
                "wall ports. Pass heat_port=True to expose them."
            )
        return [self[f'pipe_segment_{i}'].ports['wall'] for i in range(self.n_segments)]

    @property
    def segment_leak_ports(self):
        """List of the per-segment `leak` `PermeationPort_pN` (only when
        `leaky=True`).  A parent model wires each to a permeable wall's inner
        surface, e.g.
        `self.connect(pipe.segment_leak_ports[i], wall_i.ports['leak_a'])`.
        """
        if not self.leaky:
            raise AttributeError(
                "StraightPipe was built with leaky=False; it has no segment "
                "leak ports. Pass leaky=True to expose them."
            )
        return [self[f'pipe_segment_{i}'].ports['leak'] for i in range(self.n_segments)]


class SegmentedChannel(Model):
    """A duct of `N` finite-volume cells sharing `N+1` faces, deduplicated by
    construction.

    This is the staggered-grid replacement for an explicit chain of
    `TwoPortSegment`s wired by `StraightPipe`.  Instead of every segment owning
    its own in- and out-face property/velocity closures (and relying on the
    global duplicate-equation reduction to collapse the `5*(N-1)` redundant
    interface closures), the channel owns **one** set of closures per shared
    face:

        per face j (0..N):   M_j = count*rho_j*A*w_j        (axial mass flow)
                             T_j  = T_ph(p_j, h_j)
                             rho_j = rho_ph(p_j, h_j)
                             mu_j  = mu_ph(p_j, h_j)
                             k_j   = k_ph(p_j, h_j)

    and one balance set per cell i (between faces i and i+1):

        continuity:  M_i - M_{i+1} (+ m_dot_leak_i) == 0
        momentum:    distributed duct balance (see `_momentum_eq`)
        energy:      h_i + w_i^2/2 + q_specific - (h_{i+1} + w_{i+1}^2/2) == 0
        q_diag:      q_inflow_i - q == 0
        wall temp:   (adiabatic only) T_wall_i - (T_i + T_{i+1})/2 == 0

    `M_j` is the signed **axial** mass flow (positive downstream); the cell maps
    it onto the "flow into me" convention used by the per-cell balances as
    `m_dot_in = M_i`, `m_dot_out = -M_{i+1}`.  The boundary into-me flows
    `m_dot_in`/`m_dot_out` (the inlet/outlet port channels) satisfy
    `m_dot_in = M_0` and `m_dot_out = -M_N`.

    With `N=1` and `dynamic="static"` the equation set is identical to a single
    `TwoPortSegment`.

    Dynamic levels (``dynamic=``):

    Pick the level by *what you want to simulate* -- each adds one storage
    mechanism (and cost) on top of the previous:

      * ``"static"`` (default) -- steady operating points.  Quasi-steady mass /
        momentum / energy, exactly the `TwoPortSegment` physics (the only level
        that is not cell-centred; it uses the face-collocated per-cell
        template).  All media.
      * ``"advective"`` -- temperature / composition fronts.  Transient
        **energy** only: ``hc_i`` differential, mass and momentum quasi-steady,
        ``pc_i`` algebraic (set by the staggered momentum).  The level for
        (near-)incompressible transients (validated on the ULg water
        benchmark).
      * ``"compressible"`` -- pack/unpack, pressurisation, mass storage (gas,
        liquid AND the HEM two-phase dome), no pressure waves.  Transient
        **mass + energy** via the **low-Mach pressure split** (Paolucci 1982;
        Majda & Sethian 1985; the single-pressure-state structure of
        ThermoPower's ``Flow1D``, Casella 2006): ONE pipe-level thermodynamic
        pressure state ``p_pipe`` (anchored to the mean cell pressure) carries
        the pressure dynamics, while each cell keeps the exact quasi-steady
        spatial profile ``pc_i`` from the momentum chain (friction + gravity),
        so the EoS sees the correct local pressure (saturation-temperature
        glide along a boiling channel included).  The per-cell mass balance
        ``V*(rho_p*dp_pipe/dt + rho_h*dhc_i/dt) = M_i - M_{i+1}`` lets the
        flow field go non-uniform as fluid is stored/released.  Acoustics are
        filtered by construction, which is exactly what keeps this level
        well-conditioned at any Mach number.
      * ``"acoustic"`` -- pressure-wave phenomena: water hammer, surge, fast
        valve events, for all phases.  Per-cell ``pc_i`` differential states
        plus transient momentum ``rho*L*dw_j/dt`` on EVERY face (end faces
        included, so waves reflect correctly off the port boundary
        conditions).  The mass/energy rows carry constant reference scaling
        and the momentum rows are written in pressure units, which keeps the
        per-cell pressure Jacobian well-conditioned even for a subcooled
        liquid (see `_declare_primitive`).  Optional wall physics:
        ``wall_elasticity`` (Korteweg hoop compliance -- the classic
        elastic-line wave-speed correction, required for quantitative water
        hammer), ``cavitation`` (discrete vapor cavities / column separation
        via a smoothed complementarity clamp at ``p_vap``),
        ``unsteady_friction`` (Brunone) and ``viscoelastic_wall``
        (Kelvin-Voigt, polymer pipes).  Resolving the waves needs
        ``dt <~ L/(N*c)``; the implicit integrator remains stable at larger
        ``dt`` (waves are then damped, not amplified).

    All three dynamic levels share one **primitive** ``(p, h)`` cell-centred
    finite-volume formulation (see `_declare_primitive`): cells carry
    ``(pc_i, hc_i)`` with EoS-derived ``Tc_i``/``rhoc_i``/``kc_i``; momentum is
    a **staggered face** balance (face ``j`` relates the adjacent cell
    pressures -- half a cell to the port pressures at the ends -- to the face
    velocity ``w_j``); face enthalpies are *reconstructed* with the
    upwind-biased ``advection_scheme`` stencil (default ``'U2D1'``); axial
    diffusion (Fourier + ``dispersion_func``) telescopes to a compact
    cell-temperature Laplacian.  ``rho_p = drho/dp|_h`` and
    ``rho_h = drho/dh|_p`` come from the medium as symbolic functions with
    consistent second derivatives, so the Newton Jacobian of the storage terms
    is exact in every phase.
    """

    _MULTIPHASE_MODES = ("single", "HEM")
    _DYNAMIC_LEVELS = ("static", "advective", "compressible", "acoustic")
    #: Instance flags that change the emitted equation structure, so the
    #: per-class template cache keeps heated / leaky / multiphase / sized
    #: variants distinct (see `Model.collect_equations`).
    _cache_key_flags = ("multiphase", "heat_port", "leaky", "dynamic", "N",
                        "advection_scheme", "wall_elasticity",
                        "unsteady_friction", "viscoelastic_wall",
                        "cavitation")

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Pipe bore (inner) diameter.", unit="m",
                    default=0.01)],
        L: Annotated[float, ParamSpec("Total flow length.", unit="m",
                    default=1.0)],
        epsilon: Annotated[float, _SPEC_EPSILON],
        z_in: Annotated[float, _SPEC_Z_IN],
        z_out: Annotated[float, _SPEC_Z_OUT],
        N: Annotated[int, ParamSpec("Number of finite-volume cells (>= 1); the "
                    "channel has N+1 shared faces.", unit="1",
                    structural=True)] = 1,
        f_factor_func: Annotated[Callable, ParamSpec(
            "Callable f(Re, epsilon, Dh) returning the Darcy friction factor "
            "symbolically.")] = None,
        q_inflow_func: Annotated[Callable, ParamSpec(
            "Callable returning the radial heat input rate symbolically.")] = None,
        dispersion_func: Annotated[Callable, ParamSpec(
            "Callable f(w, Dh, alpha, nu) returning the effective axial "
            "diffusivity [m^2/s] for the advective level, from the local face "
            "velocity `w`, hydraulic diameter `Dh`, molecular thermal "
            "diffusivity `alpha=k/(rho*cp)` and kinematic viscosity `nu=mu/rho`. "
            "Default: `get_general_dispersion`, a regime-blended (laminar "
            "Taylor-Aris / turbulent Taylor) model valid for any medium.")] = None,
        advection_scheme: Annotated[str, ParamSpec(
            "Face-enthalpy reconstruction for the advective level "
            "('U<n_up>D<n_down>'); default 'U2D1'.", structural=True)] = "U2D1",
        multiphase: Annotated[str, _SPEC_MULTIPHASE] = "single",
        heat_port: Annotated[bool, _SPEC_HEAT_PORT] = False,
        leaky: Annotated[bool, _SPEC_LEAKY] = False,
        dynamic: Annotated[str, ParamSpec("Dynamic modelling level: 'static' "
                          "(quasi-steady, default), 'advective', "
                          "'compressible', 'acoustic'.",
                          choices=("static", "advective", "compressible",
                                   "acoustic"),
                          structural=True)] = "static",
        p_init: Annotated[float, ParamSpec(
            "Initial fluid pressure [Pa]. Seeds the cell and face pressures.",
            unit="Pa")] = 101325.0,
        wall_elasticity: Annotated[bool, ParamSpec(
            "Korteweg hoop compliance of the pipe wall (mass-storage levels "
            "only): a pressure rise stretches the wall and enlarges the "
            "cross-section, adding rho*D*c1/(e*E) to the effective "
            "compressibility.  Lowers the pressure-wave speed to the classic "
            "elastic-line value 1/a^2 = rho_p + rho*D*c1/(e*E) -- required "
            "for quantitative water hammer in real pipes.",
            structural=True)] = False,
        wall_E: Annotated[float, ParamSpec(
            "Young's modulus of the structural wall (wall_elasticity).",
            unit="Pa", relevant_when={"wall_elasticity": True})] = 200e9,
        wall_e: Annotated[float, ParamSpec(
            "Structural wall thickness (wall_elasticity).", unit="m",
            relevant_when={"wall_elasticity": True})] = 0.002,
        wall_c1: Annotated[float, ParamSpec(
            "Pipe-anchoring constraint factor c1 in the Korteweg formula: "
            "1 (expansion joints, default), 1-nu/2 (anchored upstream only), "
            "1-nu^2 (anchored throughout).",
            relevant_when={"wall_elasticity": True})] = 1.0,
        unsteady_friction: Annotated[bool, ParamSpec(
            "TESTING: instantaneous-acceleration (Brunone) unsteady wall "
            "friction on the acoustic level; adds "
            "k_uf*rho*L*(dw/dt + a*sign(w)*dw/dx) to the face pressure drop. "
            "Improves the damping of water-hammer oscillation tails.",
            structural=True)] = False,
        k_uf: Annotated[float, ParamSpec(
            "Brunone unsteady-friction coefficient (unsteady_friction).",
            relevant_when={"unsteady_friction": True})] = 0.033,
        viscoelastic_wall: Annotated[bool, ParamSpec(
            "TESTING: Kelvin-Voigt viscoelastic wall creep (mass-storage "
            "levels): one retarded-strain state per cell, "
            "tau_ve*deps/dt = J_ve*(pc - p_init) - eps, feeding 2*rho*deps/dt "
            "into the mass storage.  For polymer (PE/PVC) pipes; see Covas et "
            "al. (2004).", structural=True)] = False,
        J_ve: Annotated[float, ParamSpec(
            "Kelvin-Voigt creep compliance of the wall hoop strain per Pa of "
            "gauge pressure (viscoelastic_wall).", unit="1/Pa",
            relevant_when={"viscoelastic_wall": True})] = 0.0,
        tau_ve: Annotated[float, ParamSpec(
            "Kelvin-Voigt retardation time (viscoelastic_wall).", unit="s",
            relevant_when={"viscoelastic_wall": True})] = 1.0,
        cavitation: Annotated[bool, ParamSpec(
            "Vapor-cavity (column-separation) handling, acoustic level only: "
            "each cell carries a discrete vapor-cavity volume V_cav_i >= 0 "
            "closed by the smoothed complementarity (pc_i - p_vap) >= 0, the "
            "implicit-FV analogue of the classic DVCM/DGCM (Wylie & Streeter; "
            "Bergant & Simpson 1999).  When a rarefaction wave pulls a cell "
            "down to the vapor pressure, the cell pressure clamps there and "
            "the cavity absorbs the flow imbalance; on collapse the "
            "water-hammer shock is re-emitted.  The surrounding liquid stays "
            "single-phase, so the two-phase-dome property cliff never enters "
            "the Jacobian.", structural=True)] = False,
        p_vap: Annotated[float, ParamSpec(
            "Cavity opening pressure [Pa] (cavitation=True): the fluid's "
            "saturation pressure at the operating temperature.  `Pipe` fills "
            "this in automatically from `T_wall_init`; a bare "
            "SegmentedChannel needs it explicitly.", unit="Pa",
            relevant_when={"cavitation": True})] = None,
        cav_eps: Annotated[float, ParamSpec(
            "Dimensionless smoothing of the cavity complementarity switch: "
            "the exact condition (p - p_vap)*V_cav = 0 is regularised to the "
            "smooth hyperbola a*b = cav_eps^2/2 in scaled variables.  Smaller "
            "= sharper clamp, larger = easier Newton convergence through "
            "cavity opening / collapse (1e-3 is already too sharp for plain "
            "Newton on a hard water-hammer cavity).",
            relevant_when={"cavitation": True})] = 1e-2,
        fsi: Annotated[bool, ParamSpec(
            "RESERVED: full fluid-structure interaction (axial pipe motion, "
            "junction coupling).  Not implemented -- raises.",
            structural=True)] = False,
        count: Annotated[float, _SPEC_COUNT] = 1.0,
    ):
        self.medium = medium
        if multiphase not in self._MULTIPHASE_MODES:
            raise ValueError(
                f"multiphase must be one of {self._MULTIPHASE_MODES}, "
                f"got {multiphase!r}")
        if dynamic not in self._DYNAMIC_LEVELS:
            raise ValueError(
                f"dynamic must be one of {self._DYNAMIC_LEVELS}, got {dynamic!r}")
        if int(N) != N or N < 1:
            raise ValueError(f"N must be an integer >= 1, got {N!r}")
        self.multiphase = multiphase
        self.heat_port = bool(heat_port)
        self.leaky = bool(leaky)
        self.dynamic = dynamic
        # The three dynamic levels share one primitive `(p, h)` cell-centred
        # finite-volume formulation (see `_declare_primitive`); each level adds
        # one storage mechanism on top of the previous:
        #   * `advective`   -- transient cell energy only (`hc_i` states); mass
        #     and momentum quasi-steady, `pc_i` algebraic.
        #   * `compressible`-- + mass storage carried by ONE pipe-level
        #     thermodynamic-pressure state `p_pipe` (low-Mach pressure split:
        #     the cells see `pc_i = p_pipe + quasi-steady momentum profile`, so
        #     density/storage respond to pressure without any per-cell acoustic
        #     mode -- the singular low-Mach coupling is designed out).
        #   * `acoustic`    -- per-cell `pc_i` differential states + face
        #     velocity `w_j` momentum inertia on EVERY face: the full 1-D
        #     compressible staggered scheme that resolves pressure waves.
        # `static` is the sole non-cell-centred level (face-collocated template).
        self._cell_centered = dynamic in ("advective", "compressible",
                                          "acoustic")
        self._pressure_split = dynamic == "compressible"
        self._mass_storage = dynamic in ("compressible", "acoustic")
        self._momentum_inertia = dynamic == "acoustic"
        self.p_init = float(p_init)
        if fsi:
            raise NotImplementedError(
                "SegmentedChannel(fsi=True): full fluid-structure interaction "
                "(axial pipe motion, junction coupling) is not implemented. "
                "Korteweg hoop compliance is available via wall_elasticity=True;"
                " for full FSI see Tijsseling (1996), 'Fluid-structure "
                "interaction in liquid-filled pipe systems: a review'.")
        self.fsi = False
        self.wall_elasticity = bool(wall_elasticity)
        self.wall_E = float(wall_E)
        self.wall_e = float(wall_e)
        self.wall_c1 = float(wall_c1)
        if self.wall_elasticity and (self.wall_E <= 0 or self.wall_e <= 0):
            raise ValueError(
                f"SegmentedChannel(wall_elasticity=True): wall_E and wall_e "
                f"must be > 0, got wall_E={wall_E!r}, wall_e={wall_e!r}")
        self.unsteady_friction = bool(unsteady_friction)
        self.k_uf = float(k_uf)
        self.viscoelastic_wall = bool(viscoelastic_wall)
        self.J_ve = float(J_ve)
        self.tau_ve = float(tau_ve)
        if self.viscoelastic_wall and self.tau_ve <= 0:
            raise ValueError("SegmentedChannel(viscoelastic_wall=True): "
                             f"tau_ve must be > 0, got {tau_ve!r}")
        self.cavitation = bool(cavitation)
        self.cav_eps = float(cav_eps)
        self.p_vap = None if p_vap is None else float(p_vap)
        if self.cavitation:
            if dynamic != "acoustic":
                raise ValueError(
                    f"SegmentedChannel(cavitation=True) requires "
                    f"dynamic='acoustic' (the only level with per-cell "
                    f"pressure states that a cavity can clamp), got "
                    f"dynamic={dynamic!r}")
            if self.p_vap is None:
                raise ValueError(
                    "SegmentedChannel(cavitation=True): p_vap is required -- "
                    "pass the saturation pressure at the operating "
                    "temperature, e.g. CoolProp.PropsSI('P', 'T', T_op, 'Q', "
                    "0, fluid).  (The Pipe assembly computes it automatically "
                    "from T_wall_init.)")
            if self.p_vap <= 0:
                raise ValueError(
                    f"SegmentedChannel(cavitation=True): p_vap must be > 0, "
                    f"got {p_vap!r}")
            if self.cav_eps <= 0:
                raise ValueError(
                    f"SegmentedChannel(cavitation=True): cav_eps must be > 0, "
                    f"got {cav_eps!r}")
            if self.p_vap >= float(p_init):
                warnings.warn(
                    f"SegmentedChannel(cavitation=True): p_vap="
                    f"{self.p_vap:.4g} Pa >= p_init={float(p_init):.4g} Pa -- "
                    f"the channel starts cavitated, which the closed-cavity "
                    f"initial state (V_cav=0) contradicts.  Expect "
                    f"initialisation trouble; raise p_init or check p_vap.",
                    stacklevel=2)
        self.N = int(N)
        # `n_segments` alias so the channel is a drop-in for `StraightPipe`
        # consumers that read `.n_segments` and per-cell port lists.
        self.n_segments = self.N
        self.D = D
        self.L = L
        self.epsilon = epsilon
        self.z_in = z_in
        self.z_out = z_out
        self.count = count
        # Default the constitutive hooks (mirrors StraightPipe): Churchill
        # friction, and a Gnielinski/laminar Nusselt heat term that is zeroed
        # when there is no heat port.
        self.f_factor_func = f_factor_func or self.get_churchill_f_factor
        self.q_inflow_func = q_inflow_func or (
            self.get_q_inflow if self.heat_port else self.get_q_inflow_adiabatic)
        # Axial diffusion closure, used only by the `advective`/`compressible`
        # levels.  Defaults to `get_general_dispersion`: a regime-blended
        # (laminar Taylor--Aris / turbulent Taylor) effective diffusivity built
        # entirely from local properties, so it is valid for any medium
        # (including the HEM multiphase mixture) across laminar, transitional
        # and turbulent flow.  The physical dispersion it adds is what keeps the
        # central axial-diffusion stencil non-oscillatory at sharp fronts; on a
        # coarse grid (cell-Peclet >> 2) it can still oscillate, so the channel
        # emits a build-time and run-time cell-Peclet warning (see
        # `_warn_cell_peclet` / `runtime_diagnostics`).  Pass
        # `dispersion_func=<channel>.get_conduction_only` to disable it.
        #
        # The `static` (quasi-steady) level has NO axial-diffusion term at all
        # (`_cell_residuals` is a pure face-to-face energy balance), so it never
        # uses a dispersion closure -- leave it `None` there rather than
        # silently binding `get_general_dispersion` to a slot that is dead for
        # static, which would misleadingly suggest static disperses.
        self._dispersion_is_custom = dispersion_func is not None
        if self._cell_centered:
            self.dispersion_func = dispersion_func or self.get_general_dispersion
        else:
            self.dispersion_func = None
        self._peclet_warned = False
        self.advection_scheme = advection_scheme
        self._n_up, self._n_down = self._parse_advection_scheme(advection_scheme)
        h_std = float(medium.eval_h_pT(101325.0, 293.15))
        self._h_std = h_std
        self._rho_std = float(medium.eval_rho_ph(101325.0, h_std))
        self._mu_std = float(medium.eval_mu_ph(101325.0, h_std))
        self._k_std = float(medium.eval_k_ph(101325.0, h_std))
        # Std-state isobaric heat capacity, used as a (constant) reference both
        # for the Nusselt-correlation Prandtl number and for the advective
        # level's axial diffusion (a 2nd-order diffusive correction, so a fixed
        # cp is adequate and keeps the residual free of T_ph derivatives, whose
        # 2nd derivatives the medium does not provide).  See `_reference_cp`.
        self._cp_std = _reference_cp(medium)
        super().__init__()

    # --- constitutive correlations (same as StraightPipe) ------------------
    def get_churchill_f_factor(self, Re, epsilon, D):
        term1 = (8.0 / Re) ** 12
        A = (-2.457 * sp.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / Re) ** 16
        term2 = 1.0 / (A + B) ** 1.5
        return (term1 + term2) ** (1.0 / 12.0) * 8

    def calculate_nu_smooth(self, Re, Pr, fr):
        # Smooth (differentiable) Nusselt number: laminar fully-developed
        # Nu = 3.66 below Re~2300, blended into the Gnielinski turbulent
        # correlation above Re~3100.  (Earlier revisions had the laminar /
        # turbulent regimes inverted; see `StraightPipe.calculate_nu_smooth`.)
        nu_lam = 3.66
        nu_turb = (fr / 8) * (Re - 1000) * Pr / (
            1 + 12.7 * (fr / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        phi = sp.Max(0.0, sp.Min(1.0, (Re - 2300.0) / 800.0))
        return (1.0 - phi) * nu_lam + phi * nu_turb

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        Re = w * Dh * rho / mu
        # Prandtl number Pr = mu*cp/k (cp at the std reference state; see
        # `_reference_cp`).  Earlier this read `mu*rho/k`, which is
        # dimensionally not the Prandtl number and badly under-predicted Nu
        # for liquids (cp >> rho-scaled value).
        cp = self._cp_std if self._cp_std else self._k_std / self._mu_std
        Pr = mu * cp / k
        nu = self.calculate_nu_smooth(Re, Pr, fr)
        alpha = nu * k / Dh
        return alpha * area * (T_wall - T)

    def get_q_inflow_adiabatic(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        return 0.0

    # Regime blend for the general axial-dispersion model.  Taylor--Aris (the
    # laminar w^2/alpha shear term) is only valid below transition; the turbulent
    # Taylor scaling D ~ Dh*u* takes over above it.  `phi` ramps 0->1 across
    # [_RE_LAM, _RE_TURB] so the two branches join smoothly (and, crucially, the
    # laminar term is switched off before it blows up at high Re).
    _RE_LAM = 2300.0
    _RE_TURB = 4000.0
    #: Taylor (1954) turbulent axial-dispersion constant, D = 10.1*R*u* =
    #: 5.05*Dh*u* with the friction velocity u* = |w|*sqrt(f/8).
    _C_TAYLOR_TURB = 5.05

    def get_conduction_only(self, w, Dh, alpha, nu=None):
        """Axial diffusivity = molecular thermal diffusivity only (no shear
        dispersion).  Pass this to disable the general model."""
        return alpha

    def get_taylor_aris_dispersion(self, w, Dh, alpha, nu=None):
        """Laminar Taylor--Aris effective axial diffusivity ``[m^2/s]``:
        ``D_eff = alpha + (w*Dh)^2 / (192*alpha)`` (valid for laminar flow only;
        diverges as ``w`` grows -- use `get_general_dispersion` across regimes).
        """
        return alpha + (w * Dh) ** 2 / (192 * alpha)

    def get_general_dispersion(self, w, Dh, alpha, nu):
        """General effective axial diffusivity ``[m^2/s]``, valid across laminar,
        transitional and turbulent flow and for any medium (the closure is built
        purely from the *local* molecular diffusivity ``alpha=k/(rho*cp)`` and
        kinematic viscosity ``nu=mu/rho``, so it works unchanged for the HEM
        multiphase mixture).

        ``D_eff = (1-phi)*[alpha + (w*Dh)^2/(192*alpha)]  (laminar Taylor-Aris)``
        ``      + phi*[alpha + C*Dh*|w|*sqrt(f/8)]        (turbulent Taylor)``

        with ``phi`` a smooth ramp over ``Re in [_RE_LAM, _RE_TURB]``, ``f`` the
        Darcy friction factor (`f_factor_func`) and ``C = _C_TAYLOR_TURB``.  Both
        branches include molecular conduction (``alpha``), so the Fourier term is
        never dropped.
        """
        return self._general_dispersion(w, Dh, alpha, nu, _SymLib)

    def _general_dispersion(self, w, Dh, alpha, nu, lib):
        """Shared laminar/turbulent-blend implementation, evaluated with either
        the sympy namespace (`_SymLib`, symbolic residual) or the numeric
        namespace (`_NumLib`, run-time Peclet check)."""
        Re = lib.Abs(w) * Dh / nu + 1.0            # +1 floor: keep Re, f finite
        f = self._darcy_f(Re, self.epsilon, Dh, lib)
        D_lam = alpha + (w * Dh) ** 2 / (192.0 * alpha)
        u_star = lib.Abs(w) * lib.sqrt(f / 8.0)
        D_turb = alpha + self._C_TAYLOR_TURB * Dh * u_star
        phi = lib.Max(0.0, lib.Min(
            1.0, (Re - self._RE_LAM) / (self._RE_TURB - self._RE_LAM)))
        return (1.0 - phi) * D_lam + phi * D_turb

    @staticmethod
    def _darcy_f(Re, epsilon, D, lib):
        """Churchill Darcy friction factor, evaluated with `lib` (sympy or
        numeric); mirrors `get_churchill_f_factor` so the run-time Peclet check
        matches the symbolic residual."""
        term1 = (8.0 / Re) ** 12
        A = (-2.457 * lib.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / Re) ** 16
        return (term1 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0) * 8

    # --- advective-level face reconstruction (cell -> face) ----------------
    @staticmethod
    def _parse_advection_scheme(scheme):
        """Parse ``'U<n_up>D<n_down>'`` into ``(n_up, n_down)`` cell counts."""
        s = str(scheme).upper().strip()
        m = re.fullmatch(r"U(\d+)D(\d+)", s)
        if not m:
            raise ValueError(
                f"advection_scheme must look like 'U2D1' (upwind/downwind cell "
                f"counts), got {scheme!r}")
        n_up, n_down = int(m.group(1)), int(m.group(2))
        if n_up < 1:
            raise ValueError("advection_scheme needs at least one upwind cell "
                             "(n_up >= 1)")
        return n_up, n_down

    @staticmethod
    def _interp_stencil(positions):
        """Coefficients that interpolate the value at ``x = 0`` from samples at
        ``positions`` (in units of ``dx``).  Solves the Vandermonde system so the
        result is exact for polynomials up to degree ``len(positions) - 1``
        (e.g. ``[-1.5, -0.5, 0.5] -> [-1/8, 6/8, 3/8]``, the U2D1 face value)."""
        x = np.asarray(positions, dtype=float)
        n = x.size
        V = np.vander(x, n, increasing=True).T  # V[k, i] = x[i] ** k
        rhs = np.zeros(n)
        rhs[0] = 1.0
        return np.linalg.solve(V, rhs)

    def _face_h_recon(self, j, hc):
        """Sign-aware reconstructed enthalpy at interior/outlet face ``j`` from
        the cell-centre enthalpies ``hc`` (a list of cell symbols, ``len == N``).

        Face ``j`` sits between cell ``j-1`` (left) and cell ``j`` (right).  The
        upwind side is chosen by the sign of the face mass flow ``M_j`` via a
        smooth blend so the Jacobian stays continuous.  The stencil uses up to
        ``n_up`` upwind and ``n_down`` downwind cells and automatically drops to
        a lower order near the channel ends where neighbours are missing."""
        N = self.N
        n_up, n_down = self._n_up, self._n_down

        def one_sided(forward):
            # Cell offsets and their positions relative to the face (units dx).
            if forward:  # flow +x: upwind is the left side (cells j-1, j-2, ...)
                offs = [j - 1 - k for k in range(n_up)] + [j + k for k in range(n_down)]
                pos = [-0.5 - k for k in range(n_up)] + [0.5 + k for k in range(n_down)]
            else:        # flow -x: upwind is the right side (cells j, j+1, ...)
                offs = [j + k for k in range(n_up)] + [j - 1 - k for k in range(n_down)]
                pos = [0.5 + k for k in range(n_up)] + [-0.5 - k for k in range(n_down)]
            keep = [(o, p) for o, p in zip(offs, pos) if 0 <= o <= N - 1]
            if not keep:
                # The whole upwind side is past a channel end (e.g. reverse flow
                # at the outlet with no downwind cell): fall back to the nearest
                # in-domain cell (first-order, boundary value).
                nearest = min(max(j - 1, 0), N - 1)
                return hc[nearest]
            offs = [o for o, _ in keep]
            pos = [p for _, p in keep]
            c = self._interp_stencil(pos)
            return sum(float(ck) * hc[o] for ck, o in zip(c, offs))

        h_fwd = one_sided(True)
        h_bwd = one_sided(False)
        M_j = self[f'M_{j}'].symbol
        theta = (1 + M_j / sp.sqrt(M_j ** 2 + 1e-12)) / 2  # ~1 forward, ~0 reverse
        return theta * h_fwd + (1 - theta) * h_bwd

    def _dispersion_numeric(self, w, Dh, alpha, nu):
        """Evaluate the effective axial diffusivity on plain floats (for the
        Peclet checks).  Uses the fast numeric branch for the built-in general
        model and otherwise coerces the (possibly symbolic) custom hook."""
        if self.dispersion_func == self.get_general_dispersion:
            return self._general_dispersion(w, Dh, alpha, nu, _NumLib)
        try:
            return float(self.dispersion_func(w, Dh, alpha, nu))
        except (TypeError, ValueError):
            return float(alpha)

    def _warn_cell_peclet(self, w_ref=1.0):
        """Best-effort *build-time* cell-Peclet check for the central diffusion
        stencil.  The reconstruction is non-oscillatory only while
        ``Pe_cell = |w|*dx/D_eff <= 2``; evaluate it at a reference velocity on
        the std-state properties (the true, velocity-dependent ``Pe`` is checked
        again at run time by `runtime_diagnostics`)."""
        try:
            cp = self._cp_std
            alpha = self._k_std / (self._rho_std * cp)
            nu = self._mu_std / self._rho_std
            Dh = self.D                       # circular bore: Dh = 4A/P = D
            D_eff = self._dispersion_numeric(w_ref, Dh, alpha, nu)
            dx = self.L / self.N
            pe = abs(w_ref) * dx / D_eff
        except Exception:  # property eval or hook may not support floats
            return
        if pe > 2.0:
            warnings.warn(
                f"SegmentedChannel(dynamic={self.dynamic!r}): estimated cell "
                f"Peclet ~ {pe:.1f} > 2 at w_ref={w_ref} m/s (dx={dx:.3g} m). "
                f"Central axial diffusion may oscillate; increase N "
                f"(currently {self.N}).",
                stacklevel=2)

    def _max_cell_peclet_runtime(self):
        """Largest interior-face cell-Peclet ``|w|*dx/D_eff`` from the *current*
        solved state (actual velocities / properties), or ``None`` if it cannot
        be evaluated."""
        cp = self._cp_std
        Dh = self.D
        dx = self.L / self.N
        worst = None
        for j in range(1, self.N):
            try:
                w = float(self[f'w_{j}'].value)
                rho = 0.5 * (float(self[f'rhoc_{j - 1}'].value)
                             + float(self[f'rhoc_{j}'].value))
                k = 0.5 * (float(self[f'kc_{j - 1}'].value)
                           + float(self[f'kc_{j}'].value))
                mu = float(self[f'mu_{j}'].value)
            except (KeyError, TypeError, ValueError):
                continue
            if rho <= 0.0 or k <= 0.0:
                continue
            alpha = k / (rho * cp)
            nu = mu / rho if mu > 0.0 else self._mu_std / rho
            D_eff = self._dispersion_numeric(w, Dh, alpha, nu)
            if D_eff <= 0.0:
                continue
            pe = abs(w) * dx / D_eff
            worst = pe if worst is None else max(worst, pe)
        return worst

    def runtime_diagnostics(self):
        """Run-time cell-Peclet check against the *actual* solved velocities,
        invoked automatically once per committed step by `Model.run`.

        Emits a single warning if the true ``Pe_cell`` exceeds 2 at the chosen
        segmentation (so the user learns the grid is too coarse for the flow
        they are actually running, which the build-time estimate at a guessed
        reference velocity can miss).  Returns ``True`` once it no longer needs
        to be called (warned, not applicable, or check budget exhausted)."""
        if not self._cell_centered or self._peclet_warned:
            return True
        self._peclet_checks = getattr(self, "_peclet_checks", 0) + 1
        pe = self._max_cell_peclet_runtime()
        if pe is not None and pe > 2.0:
            dx = self.L / self.N
            warnings.warn(
                f"SegmentedChannel(dynamic={self.dynamic!r}): run-time cell "
                f"Peclet ~ {pe:.1f} > 2 (dx={dx:.3g} m, N={self.N}). The axial "
                f"advection/diffusion may show unphysical oscillations at sharp "
                f"fronts; increase N (finer grid) to bring Pe_cell <~ 2.",
                stacklevel=2)
            self._peclet_warned = True
            return True
        # Give up quietly after a bounded number of checks if never triggered.
        return self._peclet_checks >= 200

    def _property_funcs(self):
        m = self.medium
        if self.multiphase == "HEM":
            try:
                return m.T_ph_hem, m.rho_ph_hem, m.mu_ph_hem, m.k_ph_hem
            except AttributeError as exc:
                raise AttributeError(
                    f"multiphase='HEM' requires a medium exposing *_ph_hem "
                    f"property functions (e.g. CoolPropMedium); "
                    f"{type(m).__name__} does not.") from exc
        return m.T_ph, m.rho_ph, m.mu_ph, m.k_ph

    def _momentum_eq(self, cell, *, p_in, p_out, A_in, A_out, A_avg, rho_avg,
                     w_in, w_out, w_avg, m_dot_avg, f_avg, Dh_avg, L, z_in, z_out):
        """Default per-cell momentum residual (distributed duct balance).

        Identical to `TwoPortSegment._momentum_eq`, but parameterised on the
        cell's own segment length `L` and station elevations `z_in`/`z_out` so a
        single channel can carry many cells.  Subclasses (valves / pumps)
        override to substitute a different pressure/flow relation.
        """
        delta_P_friction = f_avg * (L / Dh_avg) * (rho_avg * abs(w_avg) * w_avg / 2)
        momentum_flux = m_dot_avg * (w_out - w_in)
        buoyancy_force = -G_const * (z_out - z_in) * A_avg * rho_avg
        return p_in * A_in - p_out * A_out - delta_P_friction * A_avg - momentum_flux + buoyancy_force

    def declare_components(self):
        spec = merged_param_specs(type(self))
        N = self.N
        A_value = np.pi * self.D ** 2 / 4
        P_value = np.pi * self.D
        L_segment_value = self.L / N

        self.add_component('count', Parameter(self.count, **spec['count'].param_kwargs()))
        self.add_component('A', Parameter(A_value, "m^2"))
        self.add_component('P', Parameter(P_value, "m"))
        self.add_component('L_segment', Parameter(L_segment_value, "m"))

        # N+1 station elevations shared between adjacent cells.
        dz = (self.z_out - self.z_in) / N
        for j in range(N + 1):
            self.add_component(f'z_{j}', Parameter(self.z_in + j * dz, "m"))

        # Nominal scales for the derivative companions (Modelica-style
        # `nominal`): a `der_x` starts at 0, so the default variable scale
        # max(|init|, 1) = 1 would make the Newton step-norm metric demand
        # absurd absolute accuracy on fast derivatives.  Worse, the mass
        # storage rows determine the pressure derivatives only through the
        # tiny compressive coefficient V*rho_p, so ~1e-11 property-
        # interpolation noise in the residual maps to O(1e-7) Pa/s wiggle on
        # `der_p` -- physically harmless (a ~1e-18 kg/s spurious mass rate)
        # but enough to keep an unscaled step norm above tolerance forever.
        # The scales below are natural magnitudes: pressure/velocity change
        # per acoustic cell-transit time (acoustic level) or per second
        # (pipe-level pressure of the compressible level).
        if self._cell_centered:
            rho_ref_s, _rp_s, c_ref_s = self._ref_state_floats()
            L_cell = self.L / N
            self._der_pc_scale = max(rho_ref_s * c_ref_s ** 2 / L_cell, 1.0)
            self._der_w_scale = max(c_ref_s / L_cell, 1.0)
            self._der_P_scale = max(self.p_init, 1e5)
            self._der_hc_scale = max(abs(self._h_std), 1e5)
            if self.cavitation:
                # Cavity complementarity scales: `a = (pc - p_vap)/S_p_cav`,
                # `b = V_cav/S_V_cav` are both O(1) at the operating point /
                # for a cell-sized cavity, so the Fischer-Burmeister row is
                # dimensionless and well-conditioned in every regime.
                self._S_p_cav = max(abs(self.p_init - self.p_vap),
                                    0.05 * self.p_init, 1e3)
                # `count` may be a live Parameter (Pipe multiplicity); the
                # scales are constant floats, so take its numeric value.
                cnt = float(getattr(self.count, "value", self.count))
                self._S_V_cav = cnt * A_value * L_segment_value
                # Cavity growth-rate scale = pipe area times the Joukowsky
                # velocity swing that matches the pressure scale.
                dw_jouk = self._S_p_cav / (rho_ref_s * c_ref_s)
                self._der_Vcav_scale = cnt * A_value * max(dw_jouk, 0.1)
                # Width of the smooth EoS-pressure floor (`_p_eos_cav`).
                self._cav_w = 0.05 * self._S_p_cav

        # N+1 shared faces, each carrying its own (single) closure set.  On the
        # acoustic level EVERY face velocity is a momentum state (transient
        # d(rho*w)/dt inertia -- including the end faces, so pressure waves
        # reflect correctly off the port boundary conditions); on the other
        # levels `w_j` is algebraic (quasi-steady momentum).
        for j in range(N + 1):
            self.add_component(f'p_{j}', Variable(self.p_init, "Pa"))
            self.add_component(f'h_{j}', Variable(self._h_std, "J/kg"))
            self.add_component(f'M_{j}', Variable(0.0, "kg/s"))
            if self._momentum_inertia:
                dv_w = DifferentialVariable(0.1, "m/s")
                dv_w.der_variable.scale = self._der_w_scale
                self.add_component(f'w_{j}', dv_w)
            else:
                self.add_component(f'w_{j}', Variable(0.1, "m/s"))
            self.add_component(f'T_{j}', Variable(293.15, "K"))
            self.add_component(f'rho_{j}', Variable(self._rho_std, "kg/m^3"))
            self.add_component(f'mu_{j}', Variable(self._mu_std, "Pa*s"))
            self.add_component(f'k_{j}', Variable(self._k_std, "W/m/K"))

        # Per-cell wall-temperature / heat-diagnostic (+ leak) variables.
        for i in range(N):
            self.add_component(f'T_wall_{i}', Variable(293.15, "K"))
            self.add_component(f'q_inflow_{i}', Variable(0.0, "W"))
            if self.leaky:
                self.add_component(f'm_dot_leak_{i}', Variable(0.0, "kg/s"))

        # Primitive cell-centred thermodynamic state (advective / compressible /
        # acoustic): each cell carries the primary pair `(pc_i, hc_i)` and the
        # EoS-derived `Tc_i`/`rhoc_i`/`kc_i`, plus one axial-diffusion flux
        # `F_diff_j` per face.  `hc_i` is always a differential state (transient
        # energy).  `pc_i` is:
        #   * algebraic on `advective` (pinned by the quasi-steady momentum);
        #   * algebraic on `compressible` too -- the mass storage there is
        #     carried by ONE pipe-level thermodynamic pressure state `p_pipe`
        #     (low-Mach pressure split; see `_declare_primitive`), while the
        #     cells keep the exact quasi-steady spatial profile;
        #   * a per-cell differential state on `acoustic` (resolves waves).
        if self._cell_centered:
            for i in range(N):
                dv_hc = DifferentialVariable(self._h_std, "J/kg")
                # `der_hc` reaches O(h/residence-time) magnitudes during
                # transients; the default scale of 1 makes the Newton step
                # norm demand 1e-8 J/kg/s ABSOLUTE accuracy on it, which
                # property-interpolation noise can hold hostage at small dt
                # (observed as non-convergent +/- 1e-7 flip-flop on water).
                dv_hc.der_variable.scale = self._der_hc_scale
                self.add_component(f'hc_{i}', dv_hc)
                if self._momentum_inertia:
                    dv_pc = DifferentialVariable(self.p_init, "Pa")
                    dv_pc.der_variable.scale = self._der_pc_scale
                    self.add_component(f'pc_{i}', dv_pc)
                else:
                    self.add_component(f'pc_{i}', Variable(self.p_init, "Pa"))
                self.add_component(f'Tc_{i}', Variable(293.15, "K"))
                self.add_component(f'rhoc_{i}', Variable(self._rho_std, "kg/m^3"))
                self.add_component(f'kc_{i}', Variable(self._k_std, "W/m/K"))
                if self.viscoelastic_wall and self._mass_storage:
                    self.add_component(f'eps_ve_{i}', DifferentialVariable(
                        0.0, "1"))
                if self.cavitation:
                    # Discrete vapor-cavity volume (starts closed).  Explicit
                    # scales: the state starts at 0, so the default
                    # max(|init|, 1) would treat a 1 m^3 cavity as "typical".
                    dv_vc = DifferentialVariable(0.0, "m^3",
                                                 scale=self._S_V_cav)
                    dv_vc.der_variable.scale = self._der_Vcav_scale
                    self.add_component(f'V_cav_{i}', dv_vc)
            for j in range(N + 1):
                self.add_component(f'F_diff_{j}', Variable(0.0, "W"))
            # Compressible level: the single pipe-level thermodynamic-pressure
            # state (anchored to the mean cell pressure; drives all storage).
            if self._pressure_split:
                dv_P = DifferentialVariable(self.p_init, "Pa")
                dv_P.der_variable.scale = self._der_P_scale
                self.add_component('p_pipe', dv_P)
            # Wall-physics knobs (live Parameters so the equation templates
            # stay instance-invariant across different wall properties).
            if self.wall_elasticity:
                spec_w = merged_param_specs(type(self))
                self.add_component('wall_E', Parameter(
                    self.wall_E, **spec_w['wall_E'].param_kwargs()))
                self.add_component('wall_e', Parameter(
                    self.wall_e, **spec_w['wall_e'].param_kwargs()))
                self.add_component('wall_c1', Parameter(
                    self.wall_c1, **spec_w['wall_c1'].param_kwargs()))
            if self.unsteady_friction:
                self.add_component('k_uf', Parameter(
                    self.k_uf,
                    **merged_param_specs(type(self))['k_uf'].param_kwargs()))
            if self.viscoelastic_wall and self._mass_storage:
                spec_v = merged_param_specs(type(self))
                self.add_component('J_ve', Parameter(
                    self.J_ve, **spec_v['J_ve'].param_kwargs()))
                self.add_component('tau_ve', Parameter(
                    self.tau_ve, **spec_v['tau_ve'].param_kwargs()))
            if self.cavitation:
                self.add_component('p_vap', Parameter(
                    self.p_vap,
                    **merged_param_specs(type(self))['p_vap'].param_kwargs()))
            self._warn_cell_peclet()

        # Boundary "into me" mass flows = the inlet/outlet port channels.
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))

        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_0'], 'h': self['h_0'], 'm_dot': self['m_dot_in']},
            flow_orientation='in', medium=self.medium))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self[f'p_{N}'], 'h': self[f'h_{N}'], 'm_dot': self['m_dot_out']},
            flow_orientation='in', medium=self.medium))

        if self.heat_port:
            for i in range(N):
                self.add_port(f'wall_{i}', ThermalPort_TQ(
                    self,
                    channels={'T': self[f'T_wall_{i}'], 'Q_dot': self[f'q_inflow_{i}']},
                    flow_orientation='in', require_connection=True))
        if self.leaky:
            for i in range(N):
                self.add_port(f'leak_{i}', PermeationPort_pN(
                    self,
                    channels={'p_partial': self[f'p_{i}'],
                              'm_dot_leak': self[f'm_dot_leak_{i}']},
                    flow_orientation='in', require_connection=True))

    def _p_eos_cav(self, p):
        """EoS-side smooth pressure floor at ``p_vap`` (cavitation only).

        The DVCM complementarity keeps the *solution* at ``pc >= p_vap``, but
        Newton ITERATES (and the last few Pa of the smoothed clamp) can still
        graze below the saturation line, where the single-phase property
        partials of a real-fluid backend are undefined (HEOS returns NaN,
        the tabular backends have invalid cells) -- one bad iterate then
        poisons the Jacobian.  Property calls therefore see

            p_eos = p_vap + (x + sqrt(x^2 + w^2))/2,   x = p - p_vap

        (a smooth max, ``w = _cav_w``): identical to ``p`` a few kPa above
        the clamp, floored at ``p_vap + w/2`` below it, exact symbolic
        derivatives everywhere.  The balance equations keep the true ``p``;
        only the EoS lookups are floored, and when the cavity is open the
        cell sits at ``p ~ p_vap`` anyway, so the floor changes nothing
        physical.  Identity when `cavitation` is off."""
        if not self.cavitation:
            return p
        pv = self['p_vap'].symbol
        x = p - pv
        return pv + (x + sp.sqrt(x ** 2 + self._cav_w ** 2)) / 2

    def _wall_compliance_term(self, rhoci, Dh):
        """Korteweg hoop-compliance contribution to the effective cell
        compressibility ``d(rho*A)/dp / A = rho_p + rho*D*c1/(e*E)``: the wall
        stretches under pressure, enlarging the cross-section (quasi-static --
        the hoop breathing frequency is far above water-hammer frequencies).
        Returns 0 when `wall_elasticity` is off."""
        if not self.wall_elasticity:
            return 0
        E = self['wall_E'].symbol
        e = self['wall_e'].symbol
        c1 = self['wall_c1'].symbol
        return rhoci * Dh * c1 / (e * E)

    def _ref_state_floats(self):
        """Reference floats at ``(p_init, h_std)`` for constant residual row
        scaling: ``(rho_ref, rho_p_ref, c_ref)``.  ``rho_p_ref`` includes the
        Korteweg wall compliance when enabled, and ``c_ref`` is the isentropic
        sound speed ``1/sqrt(rho_p + rho_h/rho)`` corrected for the wall (the
        classic elastic-line wave speed)."""
        p0, h0 = self.p_init, self._h_std
        try:
            rho_ref = float(self.medium.eval_rho_ph(p0, h0))
            rho_p = float(self.medium.eval_drho_ph_dp(p0, h0))
            rho_h = float(self.medium.eval_drho_ph_dh(p0, h0))
        except Exception:
            rho_ref, rho_p, rho_h = self._rho_std, 1e-6, 0.0
        rho_p_eff = rho_p
        if self.wall_elasticity:
            rho_p_eff = rho_p + rho_ref * self.D * self.wall_c1 / (
                self.wall_e * self.wall_E)
        if not (rho_p_eff > 0.0):
            rho_p_eff = 1e-7
        denom = rho_p_eff + rho_h / rho_ref
        c_ref = _math.sqrt(1.0 / denom) if denom > 0 else 1000.0
        return rho_ref, rho_p_eff, c_ref

    def _corr_template(self, func, key, args):
        """Evaluate a per-cell/-face correlation hook (``func``) through a cached
        placeholder template so its (expensive) symbolic body is canonicalised
        by sympy only ONCE per call site, not once per cell/face.

        The primitive (cell-centred) levels call ``f_factor_func`` /
        ``dispersion_func`` / ``q_inflow_func`` inside per-cell and per-face
        loops.  Those hooks are pure symbolic functions of their positional args
        (Churchill friction with ``(...)**12/**16/**1.5``, the Taylor blend with
        ``Max/Min``, the Gnielinski Nusselt) whose *structure* depends only on
        the arg count, not the arg values -- so rebuilding them N times re-fires
        sympy's Pow/assumption/Max-Min canonicalisation machinery N times, which
        is the dominant cost of `collect_equations` for the transient levels
        (the static level already sidesteps this via its cell template).

        The first call on a given ``key`` builds ``func(*dummies)`` once and
        caches ``(dummies, expr)``; every call then returns ``expr`` with the
        real args substituted for the dummies under ``sp.evaluate(False)`` -- a
        structural leaf swap that skips re-canonicalisation, exactly mirroring
        the static `declare_equations` template/`xreplace` path.  The dummies are
        fully covered by the substitution map, so none leak into the residuals.
        """
        cache = self.__dict__.setdefault('_corr_tmpl_cache', {})
        entry = cache.get(key)
        if entry is None or len(entry[0]) != len(args):
            dummies = tuple(sp.Dummy() for _ in args)
            expr = sp.sympify(func(*dummies))
            entry = (dummies, expr)
            cache[key] = entry
        dummies, expr = entry
        with sp.evaluate(False):
            return expr.xreplace(dict(zip(dummies, args)))

    def _primitive_cell_real_syms(self, i):
        """Leaf symbols for primitive cell ``i`` (dynamic levels)."""
        d = {
            'pci': self[f'pc_{i}'].symbol,
            'hci': self[f'hc_{i}'].symbol,
            'Tci': self[f'Tc_{i}'].symbol,
            'rhoci': self[f'rhoc_{i}'].symbol,
            'kci': self[f'kc_{i}'].symbol,
            'der_hci': self[f'der_hc_{i}'].symbol,
            'q_inflow': self[f'q_inflow_{i}'].symbol,
            'T_wall': self[f'T_wall_{i}'].symbol,
            'M_in': self[f'M_{i}'].symbol,
            'M_out': self[f'M_{i + 1}'].symbol,
            'h_in': self[f'h_{i}'].symbol,
            'h_out': self[f'h_{i + 1}'].symbol,
            'w_in': self[f'w_{i}'].symbol,
            'w_out': self[f'w_{i + 1}'].symbol,
            'F_in': self[f'F_diff_{i}'].symbol,
            'F_out': self[f'F_diff_{i + 1}'].symbol,
        }
        d['m_dot_leak'] = (self[f'm_dot_leak_{i}'].symbol if self.leaky else None)
        if self._momentum_inertia:
            d['der_pc'] = self[f'der_pc_{i}'].symbol
            if self.cavitation:
                d['V_cav'] = self[f'V_cav_{i}'].symbol
                d['der_V_cav'] = self[f'der_V_cav_{i}'].symbol
        if self.viscoelastic_wall and self._mass_storage:
            d['eps_ve'] = self[f'eps_ve_{i}'].symbol
            d['der_eps_ve'] = self[f'der_eps_ve_{i}'].symbol
        return d

    def _primitive_cell_placeholder_syms(self):
        """Placeholder leaves for one primitive cell, keyed like `_primitive_cell_real_syms`."""
        keys = (
            'pci', 'hci', 'Tci', 'rhoci', 'kci', 'der_hci', 'q_inflow', 'T_wall',
            'M_in', 'M_out', 'h_in', 'h_out', 'w_in', 'w_out', 'F_in', 'F_out',
        )
        d = {k: sp.Symbol(f'__prim_{k}__', real=True) for k in keys}
        d['m_dot_leak'] = (sp.Symbol('__prim_m_dot_leak__', real=True)
                           if self.leaky else None)
        if self._momentum_inertia:
            d['der_pc'] = sp.Symbol('__prim_der_pc__', real=True)
            if self.cavitation:
                d['V_cav'] = sp.Symbol('__prim_V_cav__', real=True)
                d['der_V_cav'] = sp.Symbol('__prim_der_V_cav__', real=True)
        if self.viscoelastic_wall and self._mass_storage:
            d['eps_ve'] = sp.Symbol('__prim_eps_ve__', real=True)
            d['der_eps_ve'] = sp.Symbol('__prim_der_eps_ve__', real=True)
        return d

    def _primitive_diff_real_syms(self, j):
        """Leaf symbols for interior diffusion face ``j`` (1 <= j < N)."""
        return {
            'Fd': self[f'F_diff_{j}'].symbol,
            'Tc_L': self[f'Tc_{j - 1}'].symbol,
            'Tc_R': self[f'Tc_{j}'].symbol,
            'rhoc_L': self[f'rhoc_{j - 1}'].symbol,
            'rhoc_R': self[f'rhoc_{j}'].symbol,
            'kc_L': self[f'kc_{j - 1}'].symbol,
            'kc_R': self[f'kc_{j}'].symbol,
            'w': self[f'w_{j}'].symbol,
            'mu': self[f'mu_{j}'].symbol,
        }

    def _primitive_diff_placeholder_syms(self):
        keys = ('Fd', 'Tc_L', 'Tc_R', 'rhoc_L', 'rhoc_R', 'kc_L', 'kc_R', 'w', 'mu')
        return {k: sp.Symbol(f'__prim_diff_{k}__', real=True) for k in keys}

    def _primitive_diff_residual(self, *, Fd, Tc_L, Tc_R, rhoc_L, rhoc_R, kc_L, kc_R,
                                 w, mu, count, A, L_seg, Dh, cp):
        """Axial diffusion flux at one interior face."""
        rho_f = (rhoc_L + rhoc_R) / 2
        k_f = (kc_L + kc_R) / 2
        alpha_f = k_f / (rho_f * cp)
        nu_f = mu / rho_f
        D_eff = self._corr_template(
            self.dispersion_func, 'disp', (w, Dh, alpha_f, nu_f))
        kappa_eff = rho_f * cp * D_eff
        return Fd + count * A * kappa_eff * (Tc_R - Tc_L) / L_seg

    def _primitive_mom_interior_real_syms(self, j, z):
        """Leaf symbols for interior face momentum ``j`` (1 <= j < N)."""
        d = {
            'w': self[f'w_{j}'].symbol,
            'rho': self[f'rho_{j}'].symbol,
            'mu': self[f'mu_{j}'].symbol,
            'pL': self[f'pc_{j - 1}'].symbol,
            'pR': self[f'pc_{j}'].symbol,
            'zL': (z[j - 1] + z[j]) / 2,
            'zR': (z[j] + z[j + 1]) / 2,
        }
        if self._momentum_inertia:
            d['der_w'] = self[f'der_w_{j}'].symbol
            if self.unsteady_friction:
                d['w_prev'] = self[f'w_{j - 1}'].symbol
                d['w_next'] = self[f'w_{j + 1}'].symbol
        return d

    def _primitive_mom_interior_placeholder_syms(self):
        keys = ['w', 'rho', 'mu', 'pL', 'pR', 'zL', 'zR']
        d = {k: sp.Symbol(f'__prim_mom_{k}__', real=True) for k in keys}
        if self._momentum_inertia:
            d['der_w'] = sp.Symbol('__prim_mom_der_w__', real=True)
            if self.unsteady_friction:
                d['w_prev'] = sp.Symbol('__prim_mom_w_prev__', real=True)
                d['w_next'] = sp.Symbol('__prim_mom_w_next__', real=True)
        return d

    def _primitive_mom_residual(self, *, w, rho, mu, pL, pR, zL, zR, L_mom, Dh,
                                der_w=None, w_prev=None, w_next=None, c_ref=None):
        """Staggered face momentum balance (pressure units)."""
        Re = rho * abs(w) * Dh / mu + 1
        f = self._corr_template(self.f_factor_func, 'f', (Re, self.epsilon, Dh))
        dp_fric = f * (L_mom / Dh) * (rho * abs(w) * w / 2)
        dp_grav = G_const * (zR - zL) * rho
        if self._momentum_inertia:
            dp_uf = 0
            if self.unsteady_friction:
                k_uf = self['k_uf'].symbol
                dwdx = (w_next - w_prev) / (2 * L_mom)
                sgn_w = w / sp.sqrt(w ** 2 + 1e-6)
                dp_uf = k_uf * rho * L_mom * (der_w + c_ref * sgn_w * dwdx)
            return rho * L_mom * der_w - (pL - pR - dp_fric - dp_grav - dp_uf)
        return pL - pR - dp_fric - dp_grav

    def _primitive_cell_residuals(self, *, pci, hci, Tci, rhoci, kci, der_hci,
                                  q_inflow, T_wall, M_in, M_out, h_in, h_out,
                                  w_in, w_out, F_in, F_out,
                                  m_dot_leak=None, der_pc=None, V_cav=None,
                                  der_V_cav=None, eps_ve=None, der_eps_ve=None,
                                  V_cell, area_conv, Dh, count, cp,
                                  T_ph, rho_ph, mu_ph, k_ph, drho_dp, drho_dh,
                                  S_mass, S_energy, c_ref, N):
        """Primitive cell-centred residuals as a pure function of leaf symbols."""
        eqs = []
        pei = self._p_eos_cav(pci)
        eqs.append(rhoci - rho_ph(pei, hci))
        eqs.append(Tci - T_ph(pei, hci))
        eqs.append(kci - k_ph(pei, hci))

        m_dot_in = M_in
        m_dot_out = -M_out
        cont = m_dot_in + m_dot_out
        if self.leaky:
            cont = cont + m_dot_leak

        rho_p = drho_dp(pei, hci)
        rho_h = drho_dh(pei, hci)
        rho_p_eff = rho_p + self._wall_compliance_term(rhoci, Dh)

        muci = mu_ph(pei, hci)
        w_avg = (w_in + w_out) / 2
        Re_cell = rhoci * abs(w_avg) * Dh / muci + 1
        f_cell = self._corr_template(
            self.f_factor_func, 'f', (Re_cell, self.epsilon, Dh))
        q = self._corr_template(
            self.q_inflow_func, 'q',
            (w_avg, pci, hci, rhoci, Tci, muci, kci,
             f_cell, T_wall, Dh, area_conv))
        flux = (m_dot_in * (h_in + w_in ** 2 / 2)
                + m_dot_out * (h_out + w_out ** 2 / 2)
                + q + F_in - F_out)
        if self.leaky:
            flux = flux + m_dot_leak * hci

        ve_storage = 0
        if self.viscoelastic_wall and self._mass_storage:
            J = self['J_ve'].symbol
            tau = self['tau_ve'].symbol
            eqs.append(tau * der_eps_ve + eps_ve - J * (pci - self.p_init))
            ve_storage = 2 * rhoci * der_eps_ve

        if self._momentum_inertia:
            cav_storage = 0
            if self.cavitation:
                cav_storage = rhoci * der_V_cav
            mass_row = (V_cell * (rho_p_eff * der_pc + rho_h * der_hci
                                  + ve_storage) - cav_storage - cont)
            eqs.append(S_mass * mass_row)
            dU_dt = V_cell * ((rho_p * hci - 1) * der_pc
                              + (rho_h * hci + rhoci) * der_hci)
            if self.cavitation:
                dU_dt = dU_dt - rhoci * hci * der_V_cav
            eqs.append(S_energy * (dU_dt - flux))
            if self.cavitation:
                a_cav = (pci - self['p_vap'].symbol) / self._S_p_cav
                b_cav = V_cav / self._S_V_cav
                r_cav = sp.sqrt(a_cav ** 2 + b_cav ** 2 + self.cav_eps ** 2)
                phi = a_cav + b_cav - r_cav
                dphi_dt = ((1 - a_cav / r_cav) * der_pc / self._S_p_cav
                           + (1 - b_cav / r_cav) * der_V_cav / self._S_V_cav)
                tau_cav = (self.L / N) / c_ref
                eqs.append(phi + tau_cav * dphi_dt)
        elif self._pressure_split:
            der_P = self['der_p_pipe'].symbol
            eqs.append(V_cell * (rho_p_eff * der_P + rho_h * der_hci
                                 + ve_storage) - cont)
            dU_dt = V_cell * ((rho_p * hci - 1) * der_P
                              + (rho_h * hci + rhoci) * der_hci)
            eqs.append(dU_dt - flux)
        else:
            eqs.append(cont)
            dU_dt = V_cell * (rho_h * hci + rhoci) * der_hci
            eqs.append(dU_dt - flux)
        eqs.append(q_inflow - q)
        if not self.heat_port:
            eqs.append(T_wall - Tci)
        return eqs

    def _declare_primitive(self, eqs, N, count, A, P, L_seg, Dh,
                           T_ph, rho_ph, mu_ph, k_ph):
        """Unified primitive-``(p, h)`` residuals for the three dynamic levels.

        Each cell carries the primary pair ``(pc_i, hc_i)`` (plus EoS closures
        for ``Tc_i``/``rhoc_i``/``kc_i``); interior face pressures are the mean of
        the two adjacent cell pressures; face enthalpies are the upwind
        reconstruction; axial diffusion is the compact cell-temperature Laplacian;
        and momentum is a **staggered face** balance relating each face velocity
        ``w_j`` to the adjacent cell-pressure drop (half a cell to the port
        pressures at the ends).  Using the cell-to-cell pressure gradient
        directly keeps the pressure-velocity coupling free of the odd-even
        (checkerboard) mode.  Each level adds one storage mechanism:

        * ``advective`` -- transient energy only.  Quasi-steady mass
          (``M_i - M_{i+1} = 0`` per cell) and momentum; ``pc_i`` algebraic.

        * ``compressible`` -- + mass storage via the **low-Mach pressure split**
          (Paolucci 1982; Majda & Sethian 1985; the single-pressure-state
          structure of ThermoPower's ``Flow1D``, Casella 2006): ONE pipe-level
          thermodynamic pressure state ``p_pipe`` (anchored to the mean cell
          pressure) carries the pressure *dynamics*, while the cells keep the
          exact quasi-steady spatial *profile* ``pc_i`` from the staggered
          momentum (friction + gravity), so the EoS sees the correct local
          pressure.  The mass balance per cell is
          ``V*(rho_p*dp_pipe/dt + rho_h*dhc_i/dt) = M_i - M_{i+1}`` (+leak),
          which lets the flow field go non-uniform as the fluid stores/releases
          mass -- WITHOUT a per-cell acoustic pressure mode.  This designs out
          the singular low-Mach coupling (a per-cell pressure backed by the
          tiny compressive storage ``V*rho_p``): the only pressure DOF is
          pinned algebraically by the momentum profile, so the level is
          well-conditioned for gas, subcooled liquid AND the HEM dome alike.
          The neglected ``d(delta_p_i)/dt`` storage is O(dp_friction/p) --
          second-order small.  Acoustics are filtered by construction.

        * ``acoustic`` -- per-cell ``pc_i`` differential states AND transient
          momentum ``rho*L*dw_j/dt`` on EVERY face (end faces included, so
          waves reflect correctly off the port boundary conditions): the full
          1-D compressible staggered semi-implicit structure (RELAP5 family;
          Casulli-type), resolving pressure-wave propagation (water hammer).
          The mass/energy rows are scaled by constant reference factors
          ``1/(V*rho_p_ref)`` / ``1/V`` and the momentum row is written in
          pressure units, so the Jacobian stays numerically well-conditioned
          in every regime (the raw conservative rows leave ``dpc/dt`` with a
          ~1e-10 coefficient, which is what previously made the per-cell
          scheme rank-deficient in float64).

        Optional wall physics (all levels with mass storage):
        `wall_elasticity` (Korteweg hoop compliance -- adds
        ``rho*D*c1/(e*E)`` to the effective compressibility, lowering the wave
        speed to the elastic-line value), `viscoelastic_wall` (Kelvin-Voigt
        retarded strain state per cell, for polymer pipes) and, on the
        acoustic momentum, `unsteady_friction` (Brunone instantaneous-
        acceleration model).

        Optional column separation (`cavitation`, acoustic only): per-cell
        discrete vapor-cavity volume ``V_cav_i`` displacing liquid storage
        ``rho*dV_cav/dt`` in the mass balance, closed by the smoothed
        Fischer-Burmeister complementarity ``(pc_i - p_vap) >= 0 _|_
        V_cav_i >= 0`` -- the implicit-FV analogue of the classic DVCM/DGCM
        (Wylie & Streeter; Bergant & Simpson 1999).  The clamp keeps the
        liquid EoS out of the two-phase dome, so the acoustic Jacobian stays
        well-conditioned straight through cavitation and cavity collapse.
        """
        hc = [self[f'hc_{i}'].symbol for i in range(N)]
        pc = [self[f'pc_{i}'].symbol for i in range(N)]
        z = [self[f'z_{j}'].symbol for j in range(N + 1)]
        V_cell = count * A * L_seg
        cp = self._cp_std
        area_conv = count * P * L_seg
        # Symbolic density partials for the primitive chain rule (HEM-smoothed
        # variants inside the dome).  Both `rho_p` and `rho_h` expose consistent
        # second derivatives, so the Newton Jacobian of the cell balances is
        # exact.
        if self.multiphase == "HEM":
            drho_dp = self.medium.drho_ph_hem_dp
            drho_dh = self.medium.drho_ph_hem_dh
        else:
            drho_dp = self.medium.drho_ph_dp
            drho_dh = self.medium.drho_ph_dh

        # Constant reference floats for residual row scaling (acoustic) and the
        # Brunone convective term.  Constant scalars multiply whole residual
        # rows, so they change nothing about the solution manifold -- they only
        # equilibrate the Jacobian.
        rho_ref, rho_p_ref, c_ref = self._ref_state_floats()
        V_val = (_math.pi * self.D ** 2 / 4.0) * (self.L / N)
        S_mass = 1.0 / (V_val * rho_p_ref)
        S_energy = 1.0 / (V_val * rho_ref)

        def cell_z(i):
            return (z[i] + z[i + 1]) / 2

        # Face enthalpy reconstruction (interior + outlet); `h_0` stays the free
        # inlet port input.
        for j in range(1, N + 1):
            eqs.append(self[f'h_{j}'].symbol - self._face_h_recon(j, hc))

        # Interior face pressure = mean of the two adjacent cell pressures.
        # (`p_0` / `p_N` are the inlet / outlet port pressures.)
        for j in range(1, N):
            eqs.append(self[f'p_{j}'].symbol - (pc[j - 1] + pc[j]) / 2)

        # Axial diffusion fluxes: compact cell-temperature Laplacian (ends
        # insulated).
        for j in range(N + 1):
            if j == 0 or j == N:
                eqs.append(self[f'F_diff_{j}'].symbol)
        if N > 1:
            diff_ref = self._primitive_diff_placeholder_syms()
            diff_tpl = [self._primitive_diff_residual(
                count=count, A=A, L_seg=L_seg, Dh=Dh, cp=cp, **diff_ref)]
            with sp.evaluate(False):
                for j in range(1, N):
                    real = self._primitive_diff_real_syms(j)
                    mapping = {ph: real[k] for k, ph in diff_ref.items()}
                    eqs.extend(eq.xreplace(mapping) for eq in diff_tpl)
        elif N == 1:
            pass  # only insulated end faces (already zeroed above)

        # Per-cell EoS closures + energy (+ optional mass) balances.
        cell_kw = dict(
            V_cell=V_cell, area_conv=area_conv, Dh=Dh, count=count, cp=cp,
            T_ph=T_ph, rho_ph=rho_ph, mu_ph=mu_ph, k_ph=k_ph,
            drho_dp=drho_dp, drho_dh=drho_dh,
            S_mass=S_mass, S_energy=S_energy, c_ref=c_ref, N=N)
        if N == 1:
            real = self._primitive_cell_real_syms(0)
            eqs.extend(self._primitive_cell_residuals(**real, **cell_kw))
        else:
            ref = self._primitive_cell_placeholder_syms()
            cell_tpl = self._primitive_cell_residuals(**ref, **cell_kw)
            with sp.evaluate(False):
                for i in range(N):
                    real = self._primitive_cell_real_syms(i)
                    mapping = {ph: real[k] for k, ph in ref.items()
                               if real.get(k) is not None}
                    eqs.extend(eq.xreplace(mapping) for eq in cell_tpl)

        # Compressible level: anchor the pipe-level thermodynamic pressure to
        # the mean cell pressure.  `p_pipe` is thereby pinned ALGEBRAICALLY by
        # the momentum profile (a Dirichlet-like anchor through the ports), so
        # its time derivative is a well-conditioned numerical derivative -- the
        # pressure level is never left floating on the tiny compressive
        # storage.  (Anchoring to the outlet port pressure directly is
        # structurally fragile: with a fixed-pressure outlet the anchor row
        # collapses against the boundary equation during trivial reduction.)
        if self._pressure_split:
            eqs.append(self['p_pipe'].symbol - sum(pc) / N)

        # Staggered face momentum: drive w_j from the adjacent cell-pressure
        # drop (half a cell to the port pressures at the ends).  Quasi-steady
        # duct friction on `advective`/`compressible`; on `acoustic` EVERY face
        # carries the transient inertia `rho*L*dw/dt = -dp/dx - friction +
        # gravity` (the classic water-hammer momentum equation, in PRESSURE
        # units so the pressure coupling keeps O(1) Jacobian rows).
        half_L = L_seg / 2
        # End faces (only two; build directly).
        for j in (0, N):
            if j == 0:
                pL, pR = self['p_0'].symbol, pc[0]
                zL, zR = z[0], cell_z(0)
                L_mom = half_L
            else:
                pL, pR = pc[N - 1], self[f'p_{N}'].symbol
                zL, zR = cell_z(N - 1), z[N]
                L_mom = half_L
            w_j = self[f'w_{j}'].symbol
            rho_f = self[f'rho_{j}'].symbol
            mu_f = self[f'mu_{j}'].symbol
            mom_kw = dict(w=w_j, rho=rho_f, mu=mu_f, pL=pL, pR=pR,
                          zL=zL, zR=zR, L_mom=L_mom, Dh=Dh, c_ref=c_ref)
            if self._momentum_inertia:
                mom_kw['der_w'] = self[f'der_w_{j}'].symbol
                if self.unsteady_friction:
                    if j == 0:
                        mom_kw['w_prev'] = w_j
                        mom_kw['w_next'] = self['w_1'].symbol
                    else:
                        mom_kw['w_prev'] = self[f'w_{N - 1}'].symbol
                        mom_kw['w_next'] = w_j
            eqs.append(self._primitive_mom_residual(**mom_kw))
        # Interior faces: one template, rename leaves per face.
        if N > 1:
            mom_ref = self._primitive_mom_interior_placeholder_syms()
            mom_tpl = [self._primitive_mom_residual(
                L_mom=L_seg, Dh=Dh, c_ref=c_ref, **mom_ref)]
            with sp.evaluate(False):
                for j in range(1, N):
                    real = self._primitive_mom_interior_real_syms(j, z)
                    mapping = {ph: real[k] for k, ph in mom_ref.items()
                               if real.get(k) is not None}
                    eqs.extend(eq.xreplace(mapping) for eq in mom_tpl)

        return eqs

    def declare_equations(self):
        N = self.N
        count = self['count'].symbol
        A = self['A'].symbol
        P = self['P'].symbol
        L_seg = self['L_segment'].symbol
        Dh = 4 * A / P
        T_ph, rho_ph, mu_ph, k_ph = self._property_funcs()
        eqs = []

        # --- per-face closures (ONE sept per shared face) -------------------
        for j in range(N + 1):
            p = self[f'p_{j}'].symbol
            h = self[f'h_{j}'].symbol
            M = self[f'M_{j}'].symbol
            w = self[f'w_{j}'].symbol
            T = self[f'T_{j}'].symbol
            rho = self[f'rho_{j}'].symbol
            mu = self[f'mu_{j}'].symbol
            k = self[f'k_{j}'].symbol
            # Face EoS lookups get the same (cavitation-only) smooth pressure
            # floor as the cells (`_p_eos_cav`; identity otherwise).
            pe = self._p_eos_cav(p)
            eqs.append(M - count * rho * A * w)
            eqs.append(T - T_ph(pe, h))
            eqs.append(rho - rho_ph(pe, h))
            eqs.append(mu - mu_ph(pe, h))
            eqs.append(k - k_ph(pe, h))

        # Boundary into-me flows (inlet/outlet port channels).
        eqs.append(self['m_dot_in'].symbol - self['M_0'].symbol)
        eqs.append(self['m_dot_out'].symbol + self[f'M_{N}'].symbol)

        # --- dynamic levels: one primitive (p, h) cell-centred scheme -------
        # The advective / compressible / acoustic levels all share the primitive
        # formulation (staggered momentum, upwind face-enthalpy reconstruction,
        # axial diffusion) and differ in the storage mechanism: none / one
        # pipe-level pressure state (low-Mach split) / per-cell pressure +
        # all-face momentum inertia.  Per-cell and per-face residuals are
        # structurally identical across cells/faces (only leaf symbols differ),
        # so they are built once on placeholders and renamed per instance --
        # mirroring the static-level template path.
        if self._cell_centered:
            return self._declare_primitive(
                eqs, N, count, A, P, L_seg, Dh, T_ph, rho_ph, mu_ph, k_ph)

        # --- per-cell balances (static level only) ------------------------
        # The N cells are structurally identical -- same correlations, only the
        # per-cell leaf symbols differ.  Building each from scratch re-fires
        # sympy's assumption + Max/Min canonicalisation machinery N times, which
        # is the dominant cost of `collect_equations` for long pipes.  Instead
        # build ONE cell on placeholder symbols and rename its leaves per cell.
        #
        # The rename is a pure symbol->symbol swap, done under
        # `sp.evaluate(False)`: the template is already canonical so no
        # re-evaluation is needed.  This is ~20x faster than rebuilding -- and,
        # crucially, a *plain* `xreplace` (evaluate on) is actually SLOWER than
        # rebuilding because it reconstructs every parent node and pays the full
        # canonicalisation cost again.  For N == 1 (valves) there is nothing to
        # amortise, and valve hooks read per-cell symbols directly, so just
        # build the single cell normally.
        if N == 1:
            eqs.extend(self._cell_residuals(
                0, A=A, P=P, L_seg=L_seg, count=count, Dh=Dh,
                **self._cell_real_syms(0)))
            return eqs

        ref = self._cell_placeholder_syms()
        template = self._cell_residuals(
            0, A=A, P=P, L_seg=L_seg, count=count, Dh=Dh, **ref)
        with sp.evaluate(False):
            for i in range(N):
                real = self._cell_real_syms(i)
                mapping = {ph: real[k] for k, ph in ref.items()
                           if real[k] is not None}
                eqs.extend(eq.xreplace(mapping) for eq in template)
        return eqs

    #: Per-cell residual symbol fields whose ``in``/``out`` ends map to faces
    #: ``stem_{i}`` / ``stem_{i+1}`` (used to remap a cell template onto each
    #: cell's own symbols).
    _CELL_FACE_FIELDS = (
        ('p_in', 'p_out', 'p'), ('h_in', 'h_out', 'h'),
        ('w_in', 'w_out', 'w'), ('rho_in', 'rho_out', 'rho'),
        ('mu_in', 'mu_out', 'mu'), ('k_in', 'k_out', 'k'),
        ('T_in', 'T_out', 'T'), ('M_in', 'M_out', 'M'),
    )

    def _cell_real_syms(self, i):
        """The actual leaf symbols for cell ``i`` keyed by residual field.

        Static level uses this for the face-nodal template; dynamic
        (cell-centred) levels use `_primitive_cell_real_syms` instead.
        """
        d = {}
        for k_in, k_out, stem in self._CELL_FACE_FIELDS:
            d[k_in] = self[f'{stem}_{i}'].symbol
            d[k_out] = self[f'{stem}_{i + 1}'].symbol
        d['z_in'] = self[f'z_{i}'].symbol
        d['z_out'] = self[f'z_{i + 1}'].symbol
        d['T_wall'] = self[f'T_wall_{i}'].symbol
        d['q_inflow'] = self[f'q_inflow_{i}'].symbol
        d['m_dot_leak'] = self[f'm_dot_leak_{i}'].symbol if self.leaky else None
        return d

    def _cell_placeholder_syms(self):
        """Fresh placeholder symbols for one cell, keyed like `_cell_real_syms`.

        ``m_dot_leak`` is ``None`` when the channel is not leaky so it is never
        referenced by `_cell_residuals` and never enters the rename map.
        """
        keys = [k for pair in self._CELL_FACE_FIELDS for k in pair[:2]]
        keys += ['z_in', 'z_out', 'T_wall', 'q_inflow']
        d = {k: sp.Symbol(f'__cell_{k}__', real=True) for k in keys}
        d['m_dot_leak'] = (sp.Symbol('__cell_m_dot_leak__', real=True)
                           if self.leaky else None)
        return d

    def _cell_residuals(self, cell, *, p_in, p_out, h_in, h_out, w_in, w_out,
                        rho_in, rho_out, mu_in, mu_out, k_in, k_out, T_in, T_out,
                        M_in, M_out, z_in, z_out, T_wall, q_inflow, m_dot_leak,
                        A, P, L_seg, count, Dh):
        """Static-level residuals for a single cell as a pure function of its
        symbols.

        Shared by the direct (N == 1) path and the template build; expressed in
        terms of the passed symbols so the same code serves real and
        placeholder symbols.  Per-cell hooks (`f_factor_func`,
        `q_inflow_func`, `_momentum_eq`) must be pure functions of their
        arguments for the template rename to be valid (true for the base
        channel; the valve subclasses are N == 1 and take the direct path).
        """
        w_eps = 1e-4
        m_eps = 1e-6

        # "flow into me" mapping for this cell.
        m_dot_in = M_in
        m_dot_out = -M_out

        rho_avg = (rho_in + rho_out) / 2
        mu_avg = (mu_in + mu_out) / 2
        k_avg = (k_in + k_out) / 2
        w_avg = (w_in + w_out) / 2
        A_avg = A
        m_dot_avg = (m_dot_in - m_dot_out) / 2
        p_avg = (p_in + p_out) / 2
        h_avg = (h_in + h_out) / 2
        T_avg = (T_in + T_out) / 2
        P_avg = P
        Dh_avg = Dh
        Re_avg = rho_avg * abs(w_avg) * Dh_avg / mu_avg + 1

        eqs = []
        # continuity
        cont = m_dot_in + m_dot_out
        if self.leaky:
            cont = cont + m_dot_leak
        eqs.append(cont)

        # momentum
        f_avg = self.f_factor_func(Re_avg, self.epsilon, Dh_avg)
        eqs.append(self._momentum_eq(
            cell, p_in=p_in, p_out=p_out, A_in=A, A_out=A, A_avg=A_avg,
            rho_avg=rho_avg, w_in=w_in, w_out=w_out, w_avg=w_avg,
            m_dot_avg=m_dot_avg, f_avg=f_avg, Dh_avg=Dh_avg, L=L_seg,
            z_in=z_in, z_out=z_out))

        # energy (quasi-steady, static level)
        area_conv = count * P_avg * L_seg
        q = self.q_inflow_func(
            w_avg, p_avg, h_avg, rho_avg, T_avg, mu_avg, k_avg,
            f_avg, T_wall, Dh_avg, area_conv)
        sign_w = w_avg / sp.sqrt(w_avg ** 2 + w_eps ** 2)
        m_dot_reg = sp.sqrt(m_dot_avg ** 2 + m_eps ** 2)
        q_specific = sign_w * q / m_dot_reg
        eqs.append(h_in + w_in ** 2 / 2 + q_specific - (h_out + w_out ** 2 / 2))
        eqs.append(q_inflow - q)

        if not self.heat_port:
            eqs.append(T_wall - T_avg)
        return eqs

    @property
    def segment_wall_ports(self):
        if not self.heat_port:
            raise AttributeError(
                "SegmentedChannel was built with heat_port=False; it has no "
                "segment wall ports. Pass heat_port=True to expose them.")
        return [self.ports[f'wall_{i}'] for i in range(self.N)]

    @property
    def segment_leak_ports(self):
        if not self.leaky:
            raise AttributeError(
                "SegmentedChannel was built with leaky=False; it has no segment "
                "leak ports. Pass leaky=True to expose them.")
        return [self.ports[f'leak_{i}'] for i in range(self.N)]


class AdiabaticPump(TwoPortSegment):
    #TODO: make it into real pupm with a real friction
    # model and make the izoentropic case a special case of real pump.
    """Adiabatic pump segment using a custom friction model `f = a_iz / (Re*Dh)`."""

    def __init__(self, medium: CoolPropMedium, A_in, A_out, P_in, P_out, z_in, z_out):
        super().__init__(medium, A_in, A_out, P_in, P_out, z_in, z_out, 1, 0.0, self.get_f_from_a_iz, self.get_q_inflow)
        self.adiabatic = True

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        return 0.0

    def get_f_from_a_iz(self, Re_avg, epsilon, Dh_avg):
        return self['a_iz'].symbol / (Re_avg * Dh_avg)

    def declare_components(self):
        super().declare_components()
        self.add_component('a_iz', Variable(0.0, "W"))


class Valve(SegmentedChannel):
    """Base class for control valves: an adiabatic throttle whose pressure/
    flow relation is set by a sizing coefficient and a 0..1 opening signal.

    Built on `SegmentedChannel` (a single `N=1` cell) so it reuses continuity,
    the *adiabatic* energy
    balance -- which makes an equal-area valve isenthalpic, the correct
    throttling physics -- and the face-property closures; only the momentum
    relation is swapped, via the `_momentum_eq` hook.  Heat transfer is off
    (`q = 0`) and wall friction is irrelevant, so the inherited `L` / `epsilon`
    geometry is inert (a nominal `L = 1 m`, `epsilon = 0` are passed); the
    flow area used by the `m_dot = rho*A*w` closures comes from the connecting
    diameter `D`.

    The `opening` (0 = shut, 1 = full open) is an algebraic `Variable` exposed
    on a `RealSignal` INPUT port named ``opening``, so a control block drives
    it: wire a `control.Constant` for a fixed position, or
    `control.PID -> control.Limiter` for closed-loop control.  The port is
    ``require_connection=True`` -- an unconnected opening leaves the Variable
    unclosed (singular system), which `instantiate()` flags by name.

    Subclasses implement `_valve_flow(dp, rho_avg, theta) -> m_dot` (axial
    mass flow [kg/s] for pressure drop ``dp = p_in - p_out``).
    """

    #: P&ID-style SVG symbol for the UI canvas (a filename in
    #: ``hydrogen/components/icons/``; surfaced via the catalog as ``"icon"``).
    #: Inherited by the concrete valve subclasses below.
    UI_ICON = "valve.svg"

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Connecting diameter setting the "
                    "m_dot = rho*A*w flow area.", unit="m", default=0.01)],
        opening: Annotated[float, ParamSpec("Initial opening fraction (0 = "
                          "shut, 1 = full open); driven at runtime by the "
                          "`opening` signal port.")] = 1.0,
        z_in=0.0,
        z_out=0.0,
        dp_eps: Annotated[float, ParamSpec("Pressure-drop regulariser keeping "
                         "the sign*sqrt(|dp|) flow law smooth through dp = 0.",
                         unit="Pa")] = 1.0,
        multiphase: Annotated[str, _SPEC_MULTIPHASE] = "single",
    ):
        # Store the constructor scalars under their own names so the reflective
        # serializer can recover them (it maps each __init__ arg to a like-named
        # attribute); `D` / `opening` are not otherwise kept by the base.
        self.D = D
        self.opening = opening
        self.dp_eps = dp_eps
        # A single finite-volume cell; the channel computes A = pi*D^2/4 and
        # P = pi*D from the connecting diameter.  Friction / heat are inert (the
        # momentum hook is replaced by the valve flow law), so a nominal L and
        # zero roughness are passed.  `multiphase="HEM"` makes the face/cell
        # property closures dome-safe -- needed when the valve discharges a
        # flashing (cavitating) liquid whose outlet state is two-phase.
        super().__init__(medium, D=D, L=1.0, epsilon=0.0, z_in=z_in,
                         z_out=z_out, N=1, f_factor_func=self._no_friction,
                         q_inflow_func=self._no_heat, multiphase=multiphase)

    @staticmethod
    def _no_friction(Re, epsilon, Dh):
        return 0.0

    @staticmethod
    def _no_heat(w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        return 0.0

    def declare_components(self):
        super().declare_components()
        # Opening setpoint driven through the signal port; regulariser for the
        # sign*sqrt(|dp|) flow law.  Both are leaf symbols, so the valve's
        # equation template stays instance-invariant / cache-safe.
        self.add_component('opening', Variable(self.opening, "-"))
        self.add_component('dp_eps', Parameter(self.dp_eps,
                                               **merged_param_specs(type(self))['dp_eps'].param_kwargs()))
        self.add_port('opening', RealSignal.as_input(
            self, self['opening'], name='opening'))

    def _momentum_eq(self, cell=None, *, p_in, p_out, rho_avg, m_dot_avg, **_):
        dp = p_in - p_out
        theta = self['opening'].symbol
        return m_dot_avg - self._valve_flow(dp, rho_avg, theta)

    def _valve_flow(self, dp, rho_avg, theta):  # pragma: no cover - abstract
        raise NotImplementedError


class IncompressibleValve(Valve):
    """Liquid / low-Mach valve sized by metric Kv.

        m_dot = (theta^n * Kv / 36000) * sign(dp) * sqrt(rho * |dp_eff|)

    where ``Kv`` is the metric flow coefficient (m^3/h of water at 1 bar),
    ``theta`` the 0..1 opening and ``n = trim_exp`` the inherent trim
    characteristic (1 = linear, the default; ~2-3 approximates a ball /
    quick-closing segment valve, whose flow coefficient collapses in the
    last part of the stroke -- important when replaying a measured valve
    position for a water-hammer event).  Calibration: Kv=1, water
    (rho=1000), dp=1 bar gives 0.2778 kg/s = 1 m^3/h.  Valid while
    compressibility is negligible (``Delta p << p_1``); use
    `CompressibleValve` for gases at large pressure ratio.

    Optional LIQUID CHOKING / valve cavitation (ISA 75.01 / IEC 60534-2-1),
    enabled by passing ``p_vap``: when the vena-contracta pressure reaches
    the vapor pressure the flow saturates at

        dp_eff = min(dp, dp_choked),   dp_choked = FL^2 * (p_up - FF*p_vap)

    with ``FL`` the liquid pressure-recovery factor (~0.9 full-bore ball) and
    ``FF`` the liquid critical-pressure-ratio factor
    (``0.96 - 0.28*sqrt(p_vap/p_crit)``): the downstream pressure then no
    longer influences the flow, which is the correct physics for a valve
    discharging near/below the vapor pressure (flashing).  The upstream
    density is used in the choked law (the downstream face may be a
    two-phase mixture -- combine with ``multiphase="HEM"`` so the outlet
    property closures are dome-safe).  The choke clamp and the upstream
    selection use smooth min/max (`p_eps`), and the discontinuous
    ``sign(dp)*sqrt(|dp|)`` is regularised as
    ``dp / (dp^2 + dp_eps^2)^(1/4)``, so the Jacobian stays smooth through
    ``dp = 0`` and the onset of choking.
    """

    UI_ICON = "valve.svg"
    #: `choked` (p_vap given or not) and the baked trim exponent change the
    #: emitted equation structure -> keep template-cache variants distinct.
    _cache_key_flags = SegmentedChannel._cache_key_flags + (
        "choked", "trim_exp")
    #: `choked` is COMPUTED (True iff `p_vap` was given), not a constructor
    #: arg, so its catalog-literal description lives here.
    PARAMS = {"choked": ParamSpec(
        "Computed flag: True when `p_vap` is given, i.e. the ISA "
        "liquid-choking (valve cavitation / flashing) clamp is active in "
        "the momentum equation.")}

    def __init__(
        self,
        medium: CoolPropMedium,
        Kv: Annotated[float, ParamSpec("Metric flow coefficient (m^3/h of "
                     "water at 1 bar pressure drop).", unit="m^3/h",
                     default=1.0)],
        D, opening=1.0, z_in=0.0, z_out=0.0, dp_eps=1.0,
        trim_exp: Annotated[float, ParamSpec(
            "Inherent trim characteristic exponent n in Kv_eff = Kv*theta^n: "
            "1 = linear (default), ~2-3 approximates a ball valve.",
            structural=True)] = 1.0,
        p_vap: Annotated[float, ParamSpec(
            "Fluid vapor pressure at the operating temperature [Pa]; enables "
            "the ISA liquid-choking (valve cavitation / flashing) clamp.  "
            "None (default) = no choking.", unit="Pa")] = None,
        FL: Annotated[float, ParamSpec(
            "Liquid pressure-recovery factor (ISA 75.01): ~0.9 full-bore "
            "ball, ~0.85-0.9 globe.", relevant_when={"choked": True})] = 0.9,
        FF: Annotated[float, ParamSpec(
            "Liquid critical-pressure-ratio factor "
            "(0.96 - 0.28*sqrt(p_vap/p_crit)).",
            relevant_when={"choked": True})] = 0.96,
        p_eps: Annotated[float, ParamSpec(
            "Smoothing scale [Pa] of the choke clamp and upstream-pressure "
            "selection.", unit="Pa",
            relevant_when={"choked": True})] = 1.0,
        multiphase: str = "single",
    ):
        self.Kv = Kv
        self.trim_exp = float(trim_exp)
        if self.trim_exp <= 0:
            raise ValueError(
                f"IncompressibleValve: trim_exp must be > 0, got {trim_exp!r}")
        self.choked = p_vap is not None
        self.FL = float(FL)
        self.FF = float(FF)
        self.p_eps = float(p_eps)
        # The SegmentedChannel base has its own `p_vap` constructor argument
        # (pipe cavitation, default None) and assigns the attribute inside
        # super().__init__ -- BEFORE declare_components() runs.  Keep the
        # valve's value in a private slot that declare_components reads, and
        # restore the public attribute afterwards for the serializer.
        self._p_vap_valve = None if p_vap is None else float(p_vap)
        super().__init__(medium, D, opening=opening, z_in=z_in, z_out=z_out,
                         dp_eps=dp_eps, multiphase=multiphase)
        self.p_vap = self._p_vap_valve

    def declare_components(self):
        super().declare_components()
        spec = merged_param_specs(type(self))
        self.add_component('Kv', Parameter(self.Kv,
                                           **spec['Kv'].param_kwargs()))
        if self.choked:
            self.add_component('p_vap', Parameter(
                self._p_vap_valve, **spec['p_vap'].param_kwargs()))
            self.add_component('FL', Parameter(
                self.FL, **spec['FL'].param_kwargs()))
            self.add_component('FF', Parameter(
                self.FF, **spec['FF'].param_kwargs()))
            self.add_component('p_eps', Parameter(
                self.p_eps, **spec['p_eps'].param_kwargs()))

    def _valve_flow(self, dp, rho_avg, theta):
        C = self['Kv'].symbol / 36000.0
        dp_eps = self['dp_eps'].symbol
        th = theta if self.trim_exp == 1.0 else theta ** self.trim_exp
        if not self.choked:
            return (C * th * sp.sqrt(rho_avg)
                    * dp / (dp ** 2 + dp_eps ** 2) ** 0.25)

        FL = self['FL'].symbol
        FF = self['FF'].symbol
        pv = self['p_vap'].symbol
        p_eps = self['p_eps'].symbol
        p_in = self['p_0'].symbol
        p_out = self['p_1'].symbol
        rho_in = self['rho_0'].symbol
        rho_out = self['rho_1'].symbol

        def smax(a, b):
            return 0.5 * (a + b + sp.sqrt((a - b) ** 2 + p_eps ** 2))

        def smin(a, b):
            return 0.5 * (a + b - sp.sqrt((a - b) ** 2 + p_eps ** 2))

        # Upstream pressure / (liquid) density picked smoothly by flow
        # direction -- the downstream face may be a flashing two-phase
        # mixture whose density must NOT enter the sizing law.
        s = dp / sp.sqrt(dp ** 2 + dp_eps ** 2)             # smooth sign(dp)
        frac = 0.5 * (1 + s)                                # ~1 fwd, ~0 rev
        p_up = smax(p_in, p_out)
        rho_up = frac * rho_in + (1 - frac) * rho_out
        # ISA liquid-choking clamp (floored at p_eps so the sqrt stays real
        # even if an iterate drags p_up below FF*p_vap).
        dp_choke = FL ** 2 * smax(p_up - FF * pv, p_eps)
        dp_used = smin(smax(dp, -dp_choke), dp_choke)
        return (C * th * sp.sqrt(rho_up)
                * dp_used / (dp_used ** 2 + dp_eps ** 2) ** 0.25)


class CompressibleValve(Valve):
    """Gas valve sized by Kv with IEC 60534-2-1 compressibility (expansion
    factor ``Y`` and choked flow).

        m_dot  = sign(dp) * (theta*Kv/36000) * Y * sqrt(rho_up * dp_eff)
        x      = |dp| / p_up                  (pressure-drop ratio)
        Fgamma = gamma / 1.4
        x_eff  = min(x, Fgamma*xT)            (choke clamp -> flow saturates)
        Y      = 1 - x_eff / (3*Fgamma*xT)    (1 .. 2/3)
        dp_eff = x_eff * p_up

    ``xT`` is the valve's pressure-differential-ratio factor (~0.7 for many
    globe valves) and ``gamma = cp/cv`` the gas specific-heat ratio.  Reduces
    to the incompressible Kv law as ``x -> 0`` (``Y -> 1``).  Upstream
    pressure / density and the choke clamp are selected with smooth min/max so
    the residual stays differentiable through flow reversal and the onset of
    choking.  ``p_eps`` sets the smoothing scale on pressures.
    """

    UI_ICON = "valve.svg"

    def __init__(
        self,
        medium: CoolPropMedium,
        Kv: Annotated[float, ParamSpec("Metric flow coefficient (m^3/h of "
                     "water at 1 bar pressure drop).", unit="m^3/h",
                     default=1.0)],
        D: Annotated[float, ParamSpec("Connecting diameter setting the "
                    "m_dot = rho*A*w flow area.", unit="m", default=0.01)],
        xT: Annotated[float, ParamSpec("Pressure-differential-ratio factor "
                     "(~0.7 for many globe valves); sets the choke point.",
                     unit="1")] = 0.7,
        gamma: Annotated[float, ParamSpec("Gas specific-heat ratio cp/cv.",
                        unit="1")] = 1.4,
        opening: Annotated[float, ParamSpec("Initial opening fraction (0 = "
                          "shut, 1 = full open); driven at runtime by the "
                          "`opening` signal port.")] = 1.0,
        z_in: Annotated[float, ParamSpec("Elevation of the inlet face.",
                       unit="m")] = 0.0,
        z_out: Annotated[float, ParamSpec("Elevation of the outlet face.",
                        unit="m")] = 0.0,
        dp_eps: Annotated[float, ParamSpec("Pressure-drop regulariser keeping "
                         "the sign*sqrt(|dp|) flow law smooth through dp = 0.",
                         unit="Pa")] = 1.0,
        p_eps: Annotated[float, ParamSpec("Smoothing scale on pressures for "
                        "the smooth min/max upstream selection and choke "
                        "clamp.", unit="Pa")] = 1.0,
    ):
        self.Kv = Kv
        self.xT = xT
        self.gamma = gamma
        self.p_eps = p_eps
        super().__init__(medium, D, opening=opening, z_in=z_in, z_out=z_out,
                         dp_eps=dp_eps)

    def declare_components(self):
        super().declare_components()
        spec = merged_param_specs(type(self))
        self.add_component('Kv', Parameter(self.Kv, **spec['Kv'].param_kwargs()))
        self.add_component('xT', Parameter(self.xT, **spec['xT'].param_kwargs()))
        self.add_component('gamma', Parameter(self.gamma, **spec['gamma'].param_kwargs()))
        self.add_component('p_eps', Parameter(self.p_eps, **spec['p_eps'].param_kwargs()))

    def _valve_flow(self, dp, rho_avg, theta):
        Kv = self['Kv'].symbol
        xT = self['xT'].symbol
        gamma = self['gamma'].symbol
        dp_eps = self['dp_eps'].symbol      # Pa: sqrt(|dp|) regulariser
        p_eps = self['p_eps'].symbol        # Pa: smooth max/min scale
        # Single-cell channel: face 0 is the inlet, face 1 the outlet.
        p_in = self['p_0'].symbol
        p_out = self['p_1'].symbol
        rho_in = self['rho_0'].symbol
        rho_out = self['rho_1'].symbol

        # All smooth-max/min arguments below are PRESSURES (Pa), so `p_eps`
        # (Pa) is the only blending scale -- no dimensionless/dimensional mix.
        def smax(a, b):
            return 0.5 * (a + b + sp.sqrt((a - b) ** 2 + p_eps ** 2))

        def smin(a, b):
            return 0.5 * (a + b - sp.sqrt((a - b) ** 2 + p_eps ** 2))

        s = dp / sp.sqrt(dp ** 2 + dp_eps ** 2)            # smooth sign(dp)
        frac = 0.5 * (1 + s)                                # ~1 fwd, ~0 rev
        p_up = smax(p_in, p_out)                            # upstream pressure
        rho_up = frac * rho_in + (1 - frac) * rho_out       # upstream density
        Fg = gamma / 1.4
        x_choke = Fg * xT                                   # critical x
        dp_choke = x_choke * p_up                           # Pa, > 0

        # Cap the (signed) pressure drop at +/- the choke value in Pascals,
        # then take the always-real regularised sign*sqrt(|.|).  Because the
        # cap and the sqrt are both in Pa, the radicand is never negative.
        dp_used = smin(smax(dp, -dp_choke), dp_choke)
        x_eff = sp.sqrt(dp_used ** 2 + dp_eps ** 2) / p_up  # |dp_used| / p_up
        Y = 1 - x_eff / (3 * x_choke)                       # 2/3 .. 1
        g = dp_used / (dp_used ** 2 + dp_eps ** 2) ** 0.25  # sign*sqrt(|dp_used|)
        C = Kv / 36000.0
        return C * theta * Y * sp.sqrt(rho_up) * g


class PressureOutlet(Model):
    """Fixed-pressure outlet boundary.

    Imposes `p_in = p_ambient` at the boundary plane and lets the upstream component
    determine `(h_in, m_dot_in)`. This is the pressure-imposed dual of `PressureSource`:
    use it to terminate plumbing that exhausts to atmosphere or any other fixed-pressure
    sink, when you want the *system* to determine the mass flow rather than imposing it.

    Port (matches the (p, h, m_dot) convention used everywhere):
        p_in, h_in, m_dot_in    - external connection inputs (driven by upstream)
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        p_ambient: Annotated[float, ParamSpec("Imposed outlet (sink) "
                            "pressure.", unit="Pa")] = 101325.0,
        T_ambient: Annotated[float, ParamSpec("Sink temperature (sets the "
                            "initial enthalpy guess).", unit="K")] = 293.15,
    ):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self._h_ambient = float(medium.eval_h_pT(p_ambient, T_ambient))
        super().__init__()

    def declare_components(self):
        self.add_component('p_ambient', Parameter(self.p_ambient,
                                                  **merged_param_specs(type(self))['p_ambient'].param_kwargs()))
        self.add_component('p_in', Variable(self.p_ambient, "Pa"))
        self.add_component('h_in', Variable(self._h_ambient, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        return [self['p_in'].symbol - self['p_ambient'].symbol]


class Splitter(Model):
    """K-way ideal flow splitter: one inlet port -> K outlet ports.

    Lumped-junction model with no pressure drop, no enthalpy loss, no kinetic-energy
    mixing penalty:

        p_out_k = p_in                                          for k = 0..K-1
        h_out_k = h_in                                          for k = 0..K-1
        m_dot_in = sum_k m_dot_out_k                            (continuity)

    The pressure / enthalpy equalities are linear and get reduced away as trivial
    substitutions during instantiation, so at runtime the splitter contributes only
    the single mass-balance residual that ties `m_dot_in` to the K branch flow
    rates.  The individual `m_dot_out_k` are free unknowns; their values come from
    whatever each branch is wired into downstream.  For a balanced symmetric tree
    (each branch sees identical downstream conditions) symmetry forces
    `m_dot_out_k = m_dot_in / K`.

    Note: with `(p, h, m_dot)` ports the splitter no longer needs to know
    anything about port areas -- mass-flow conservation is geometry-free.
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        K: Annotated[int, ParamSpec("Number of outlet branches.", unit="1",
                    default=2)],
    ):
        self.medium = medium
        self.K = K
        self._h_init = float(medium.eval_h_pT(101325.0, 293.15))
        super().__init__()

    def declare_components(self):
        self.add_component('p_in', Variable(101325.0, "Pa"))
        self.add_component('h_in', Variable(self._h_init, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        for k in range(self.K):
            self.add_component(f'p_out_{k}', Variable(101325.0, "Pa"))
            self.add_component(f'h_out_{k}', Variable(self._h_init, "J/kg"))
            self.add_component(f'm_dot_out_{k}', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        for k in range(self.K):
            self.add_port(f'outlet_{k}', FluidPort_phm(
                self,
                channels={'p': self[f'p_out_{k}'], 'h': self[f'h_out_{k}'],
                          'm_dot': self[f'm_dot_out_{k}']},
                flow_orientation='in',
                medium=self.medium,
            ))

    def declare_equations(self):
        # The K pressure and K enthalpy equalities are pure variable-equality
        # constraints -- short-circuit them via union-find instead of building
        # 2K sympy Add nodes for the trivial reducer to chew through later.
        for k in range(self.K):
            self.add_connection(self[f'p_out_{k}'], self['p_in'])
            self.add_connection(self[f'h_out_{k}'], self['h_in'])

        # Mass balance under "flow into me": every port's m_dot measures
        # fluid entering the splitter, so net inflow sums to zero.  Forward
        # operation produces m_dot_in > 0 and each m_dot_out_k < 0.
        m_sum = self['m_dot_in'].symbol
        for k in range(self.K):
            m_sum = m_sum + self[f'm_dot_out_{k}'].symbol
        return [m_sum]


class PressureSource(Model):
    """Stagnation reservoir / fixed-pressure inlet boundary.

    Holds the upstream (p_source, T_source) constant and lets the *downstream* system
    determine the mass flow. Unlike `AmbientInlet`, no mass-flow constraint is imposed
    locally; the boundary plane satisfies only:

        s(p_out, h_out) = s_total                         (isentropic acceleration)
        h_total         = h_out + (m_dot_out / (rho * A))**2 / 2   (steady energy balance)

    The only boundary *parameters* are the reservoir state `(p_source, T_source)`;
    the stagnation enthalpy / entropy are not free inputs but *computed* from them
    by two closure equations:

        h_total = h(p_source, T_source)
        s_total = s(p_source, h_total)

    so a caller slides the operating point by setting `p_source` (and `T_source`)
    alone — the solver keeps `h_total` / `s_total` consistent.  The third closure
    (`p_out`) comes from whatever this source is wired into downstream.
    Use this when you want flow to be driven by a pressure differential — e.g. filling
    a vessel from a pressurised line: as the vessel back-pressure rises the inlet
    velocity naturally decays toward zero.

    `A` is the cross-sectional area of the boundary plane, needed to translate
    the port's mass flow rate `m_dot_out` into the velocity used in the kinetic-
    energy term.  Set it to the area of the downstream port the source is wired
    into; for low-Mach flows the answer is barely sensitive to the exact value
    because the KE correction is typically <1% of the stagnation pressure.

    `p_control`:
        - `False` (default): `p_source` is a `Parameter` the caller sets
          directly (e.g. ``src["p_source"].set_value(...)``).
        - `True`: `p_source` becomes an algebraic `Variable` exposed on a
          `RealSignal` INPUT port named ``p_set``, so a control block drives the
          supply pressure: wire a `control.Ramp` to pressurise over time, a
          `control.Constant` for a fixed value, or a `control.PID` for
          closed-loop control.  The port is ``require_connection=True`` -- an
          unconnected ``p_set`` leaves the supply pressure unclosed (singular
          system), which `instantiate()` flags by name.

    Port (matches the (p, h, m_dot) convention used everywhere):
        p_out, h_out, m_dot_out    - drives the downstream component
        p_set (signal, if `p_control`) - commanded supply pressure [Pa]
    """

    #: P&ID-style SVG symbol for the UI canvas (file in
    #: ``hydrogen/components/icons/``; surfaced via the catalog as ``"icon"``).
    UI_ICON = "pressure_source.svg"

    def __init__(
        self,
        medium: CoolPropMedium,
        p_source: Annotated[float, ParamSpec("Reservoir (stagnation) supply "
                           "pressure.", unit="Pa")] = 101325,
        T_source: Annotated[float, ParamSpec("Reservoir (stagnation) "
                           "temperature.", unit="K")] = 293.15,
        A: Annotated[float, ParamSpec("Cross-sectional area of the boundary "
                    "plane (kinetic-energy term); set to the downstream port "
                    "area.", unit="m^2")] = 1e-3,
        p_control: Annotated[bool, ParamSpec("If true, expose `p_source` on a "
                            "`p_set` signal input so a control block drives "
                            "the supply pressure; if false it is a fixed "
                            "parameter.", structural=True)] = False,
    ):
        self.medium = medium
        self.p_source = p_source
        self.T_source = T_source
        self.A = A
        self.p_control = bool(p_control)
        self._h_total = float(medium.eval_h_pT(p_source, T_source))
        self._s_total = float(medium.eval_s_ph(p_source, self._h_total))
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        if self.p_control:
            # Supply pressure driven by an external signal (e.g. a control.Ramp).
            self.add_component('p_source', Variable(self.p_source, "Pa"))
            self.add_port('p_set', RealSignal.as_input(
                self, self['p_source'], name='p_set'))
        else:
            self.add_component('p_source', Parameter(self.p_source, **spec['p_source'].param_kwargs()))
        self.add_component('T_source', Parameter(self.T_source, **spec['T_source'].param_kwargs()))
        self.add_component('A', Parameter(self.A, **spec['A'].param_kwargs()))
        # Stagnation enthalpy / entropy are *computed* from (p_source, T_source)
        # by the closure equations below; these values are only initial guesses.
        self.add_component('h_total', Variable(self._h_total, "J/kg"))
        self.add_component('s_total', Variable(self._s_total, "J/kg/K"))
        # Initial guesses near the stagnation state - downstream pulls them off-stagnation.
        self.add_component('p_out', Variable(self.p_source, "Pa"))
        self.add_component('h_out', Variable(self._h_total, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        # Internal velocity Variable; leaf symbol in the isentropic energy
        # balance so `w_out**2 / 2` lambdifies to a flat expression rather
        # than the deep `m_dot / (rho_ph(p, h) * A)` chain.
        self.add_component('w_out', Variable(1.0, "m/s"))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        p_src = self['p_source'].symbol
        T_src = self['T_source'].symbol
        h_total = self['h_total'].symbol
        s_total = self['s_total'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        # m_dot <-> w closure (nonlinear in rho; trivial reducer leaves it
        # alone).  Sign: under "flow into me", `m_dot_out` is positive when
        # fluid enters through the out-face (axial backward) while `w_out`
        # stays axial forward, so the two have opposite physical signs and
        # sum to zero.
        eq_w = self['m_dot_out'].symbol + rho_out * self['w_out'].symbol * self['A'].symbol
        eq_isentropic = s_total - s_out
        eq_energy = h_total - (self['h_out'].symbol + self['w_out'].symbol ** 2 / 2)
        # Stagnation state is fixed by the reservoir (p_source, T_source), so the
        # caller only sets pressure/temperature; h_total / s_total follow.
        eq_h_total = h_total - self.medium.h_pT(p_src, T_src)
        eq_s_total = s_total - self.medium.s_ph(p_src, h_total)
        return [eq_w, eq_isentropic, eq_energy, eq_h_total, eq_s_total]


class PressureVessel(Model):
    """Lumped-volume rigid-wall pressure vessel that fills through a single port.

    Pressure rises adiabatically as mass and energy accumulate in the control volume.

    Differential states:
        m  - total mass in the vessel                       [kg]
        U  - total internal energy in the vessel            [J]

    Algebraic states (vessel-average):
        p  - pressure                                       [Pa]
        h  - specific enthalpy                              [J/kg]

    Port (matches the (p, h, m_dot) convention used everywhere in this package):
        p_in, h_in, m_dot_in    - external connection inputs

    Equations:
        dm/dt = m_dot_in                                    (continuity)
        dU/dt = m_dot_in * h_in                             (energy, adiabatic)
        m    = rho(p, h) * V                                (density closure)
        U    = m*h - p*V                                    (since u = h - p/rho, m/rho = V)
        p_in = p                                            (no port-throttling)

    `leaky`:
        - `True`: a permeation `leak` `PermeationPort_pN` is exposed and its
          mass-flow enters the mass/energy balance:

              dm/dt = m_dot_in + m_dot_leak
              dU/dt = m_dot_in*h_in + m_dot_leak*h    (leaked mass carries h)

          Under "flow into me" permeation makes `m_dot_leak < 0` (gas leaves the
          volume), so the vessel slowly loses mass to a wall while an upstream
          source makes it up.  The leak port publishes the vessel pressure `p`
          as the partial pressure (pure-gas assumption).
        - `False` (default): the `leak` port is removed; the vessel is sealed.

    Conjugate heat / multi-wall coupling (`heat_ports`, `leak_ports`):
        Set `heat_ports > 0` to expose that many `heat_{k}` `ThermalPort_TQ`s,
        each publishing the bulk gas temperature `T = T(p, h)` and feeding heat
        into the energy balance::

            dU/dt += sum_k Q_dot_wall_k          (Q into the gas through port k)

        The conjugate partner (e.g. a wall surface, optionally through a
        `ThermalConductor` film of conductance `h*A`) sets the relation between
        `Q_dot_wall_k` and the surface temperature.  This is how a walled tank
        couples its gas to a barrel and to its end caps -- one `heat_{k}` per
        wall (the framework's `connect()` is pairwise, so each wall needs its
        own port).

        Set `leak_ports > 0` to expose that many *additional* `leak_{k}`
        permeation ports (independent of the single `leaky` `leak` port); each
        publishes `p` as the partial pressure and feeds the mass / energy
        balance exactly like `leaky`.  Used by the `Tank` assembly to drive one
        permeation chain per wall.

    Notes / simplifications:
      * Rigid wall (V constant), no shaft work, no outflow.  Heat exchange is
        only through the optional `heat_{k}` ports (otherwise adiabatic).
      * Inflow kinetic energy is neglected.  For typical vessel-filling regimes
        the contribution `(m_dot_in / (rho * A))**2 / 2` is several orders of
        magnitude below `h_in`; if you need it, add it to the energy balance
        below (using the `A_in` parameter the component already carries).
      * Reverse flow is not modeled.  If `m_dot_in` becomes negative the energy
        balance will still integrate, but `h_in` would no longer represent the
        true outflow enthalpy (you'd need an upwinding switch on `h_in <-> h`).
    """

    #: P&ID-style SVG symbol for the UI canvas (file in
    #: ``hydrogen/components/icons/``; surfaced via the catalog as ``"icon"``).
    UI_ICON = "pressure_vessel.svg"

    def __init__(
        self,
        medium: CoolPropMedium,
        V: Annotated[float, ParamSpec("Internal control volume of the "
                    "vessel.", unit="m^3", default=0.001)],
        A_in: Annotated[float, ParamSpec("Inlet port area (used if inflow "
                       "kinetic energy is added).", unit="m^2", default=1e-3)],
        p_init: Annotated[float, ParamSpec("Initial vessel pressure.",
                         unit="Pa")] = 101325.0,
        T_init: Annotated[float, ParamSpec("Initial vessel temperature.",
                         unit="K")] = 293.15,
        leaky: Annotated[bool, ParamSpec("If true, expose a permeation `leak` "
                        "port that feeds the mass / energy balance; if false "
                        "the vessel is sealed.", structural=True)] = False,
        heat_ports: Annotated[int, ParamSpec("Number of conjugate `heat_{k}` "
                            "thermal ports (each publishes the bulk gas "
                            "temperature and feeds heat into the energy "
                            "balance); 0 = adiabatic.", unit="1",
                            structural=True)] = 0,
        leak_ports: Annotated[int, ParamSpec("Number of additional `leak_{k}` "
                            "permeation ports (each publishes the vessel "
                            "pressure as partial pressure and feeds the mass / "
                            "energy balance); 0 = none.", unit="1",
                            structural=True)] = 0,
        count: Annotated[float, _SPEC_COUNT] = 1.0,
    ):
        if int(heat_ports) < 0 or int(leak_ports) < 0:
            raise ValueError(
                f"PressureVessel: heat_ports / leak_ports must be >= 0; got "
                f"heat_ports={heat_ports}, leak_ports={leak_ports}")
        self.medium = medium
        self.V = V
        self.A_in = A_in
        self.p_init = p_init
        self.T_init = T_init
        self.leaky = bool(leaky)
        self.heat_ports = int(heat_ports)
        self.leak_ports = int(leak_ports)
        # Multiplicity: this one vessel stands in for `count` identical parallel
        # vessels.  A live `Parameter` (own, or shared from the `Tank` assembly)
        # that scales the stored control volume `V` -- and hence the extensive
        # mass / energy -- by N, leaving the intensive (p, h, T) states
        # unchanged.  Retunable without re-instantiating.
        self.count = count
        # Pre-compute thermodynamically consistent initial conditions so the t=0 Newton
        # solve starts near a converged state.  Seed the extensive states (m, U)
        # at the N-vessel scale so the Newton start is consistent with `V_total`.
        count_num = getattr(count, 'value', count)
        V_total = count_num * V
        self.h_init = float(medium.eval_h_pT(p_init, T_init))
        self.rho_init = float(medium.eval_rho_ph(p_init, self.h_init))
        self.m_init = self.rho_init * V_total
        self.U_init = self.m_init * self.h_init - p_init * V_total  # U = m*u = m*h - p*V
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('V', Parameter(self.V, **spec['V'].param_kwargs()))
        self.add_component('count', Parameter(self.count, **spec['count'].param_kwargs()))
        self.add_component('A_in', Parameter(self.A_in, **spec['A_in'].param_kwargs()))

        # Differential states (auto-attaches `der_m`, `der_U` companions).
        self.add_component('m', DifferentialVariable(self.m_init, "kg"))
        self.add_component('U', DifferentialVariable(self.U_init, "J"))

        # Vessel-average algebraic states.
        self.add_component('p', Variable(self.p_init, "Pa"))
        self.add_component('h', Variable(self.h_init, "J/kg"))

        # Inlet port (driven by the upstream component via the parent's connection eqs).
        self.add_component('p_in', Variable(self.p_init, "Pa"))
        self.add_component('h_in', Variable(self.h_init, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))

        if self.leaky:  # expose a permeation leak port (publishes vessel p)
            self.add_component('m_dot_leak', Variable(0.0, "kg/s"))
            self.add_port('leak', PermeationPort_pN(
                self,
                channels={'p_partial': self['p'], 'm_dot_leak': self['m_dot_leak']},
                flow_orientation='in',
                require_connection=True,
            ))

        # Conjugate heat ports: publish the bulk gas temperature, draw heat from
        # the energy balance.  `T = T(p, h)` is closed in `declare_equations`.
        if self.heat_ports > 0:
            self.add_component('T', Variable(self.T_init, "K"))
            for k in range(self.heat_ports):
                self.add_component(f'Q_dot_wall_{k}', Variable(0.0, "W"))
                self.add_port(f'heat_{k}', ThermalPort_TQ(
                    self,
                    channels={'T': self['T'], 'Q_dot': self[f'Q_dot_wall_{k}']},
                    flow_orientation='in',
                    require_connection=True,
                ))

        # Extra permeation ports (one per leaky wall), each publishing vessel p.
        for k in range(self.leak_ports):
            self.add_component(f'm_dot_leak_{k}', Variable(0.0, "kg/s"))
            self.add_port(f'leak_{k}', PermeationPort_pN(
                self,
                channels={'p_partial': self['p'], 'm_dot_leak': self[f'm_dot_leak_{k}']},
                flow_orientation='in',
                require_connection=True,
            ))

    def declare_equations(self):
        m = self['m'].symbol
        U = self['U'].symbol
        p = self['p'].symbol
        h = self['h'].symbol
        # `count` parallel vessels share the same intensive (p, h) state but
        # store `count` times the volume, so the EXTENSIVE control volume that
        # closes (m, U) <-> (p, h) is `count * V`.
        V = self['count'].symbol * self['V'].symbol
        p_in = self['p_in'].symbol
        h_in = self['h_in'].symbol

        rho = self.medium.rho_ph(p, h)

        # With m_dot at the port, the inflow mass rate is the port value itself
        # -- no density/area product needed.
        m_in_dot = self['m_dot_in'].symbol

        # A leaky vessel also gains/loses mass (and its carried enthalpy)
        # through the permeation port(s); `m_dot_leak < 0` when gas permeates
        # out.  Sum the single `leak` (if `leaky`) and any indexed `leak_{k}`.
        m_leak = self['m_dot_leak'].symbol if self.leaky else 0
        for k in range(self.leak_ports):
            m_leak = m_leak + self[f'm_dot_leak_{k}'].symbol

        # Conjugate wall heat into the gas (0 when adiabatic).
        Q_wall = 0
        for k in range(self.heat_ports):
            Q_wall = Q_wall + self[f'Q_dot_wall_{k}'].symbol

        # Continuity: m grows at the inflow + leak mass rate.
        eq_mass = self['der_m'].symbol - (m_in_dot + m_leak)
        # Energy: open system; inflow and leaked mass carry enthalpy, walls add heat.
        eq_energy = self['der_U'].symbol - (m_in_dot * h_in + m_leak * h + Q_wall)

        # Algebraic closure linking (m, U) to (p, h) via the equation of state.
        eq_density = m - rho * V
        eq_energy_state = U - m * h + p * V

        # Port pressure equality: vessel pressure feeds back as the upstream's back-pressure.
        eq_port_p = p_in - p

        eqs = [eq_mass, eq_energy, eq_density, eq_energy_state, eq_port_p]

        # Bulk-temperature closure (only when a conjugate heat port needs it).
        if self.heat_ports > 0:
            eqs.append(self['T'].symbol - self.medium.T_ph(p, h))
        return eqs


class MixingJunction(Model):
    """`N`-port well-mixed junction with smooth donor-cell upwind on enthalpy flux.

    Operates in two modes, selected via the `dynamic` flag:

        dynamic=True  (default)  -- mass + internal energy are differential
                                    states; the junction has compressible
                                    storage.  Required for closed loops
                                    (breaks rank-deficiency the same way
                                    `LoopBuffer` does) and for fast transients.
        dynamic=False            -- purely algebraic "ideal mixer" with
                                    instant mass and energy balance.  Smaller
                                    symbolic system, no initial conditions,
                                    no `V`.  Requires that the surrounding
                                    network determines `p` (at least one
                                    pressure boundary somewhere) and that the
                                    attached `m_dot`s aren't all independently
                                    over-prescribed (the algebraic mass
                                    balance must have a degree of freedom to
                                    close).  An explicit `h`-anchor
                                    regularization (see `h_anchor_strength`)
                                    keeps the energy balance well-conditioned
                                    at exact zero flow.

    Both modes share the same port API, the same smooth-blend port-enthalpy
    closure, and the same flow-reversal handling -- only the bulk mass /
    energy balance and the EoS closures differ.

    Sign convention (junction-centric, consistent with `m_dot_in` semantics
    elsewhere in the package -- positive m_dot means "flow into me"):

        m_dot_k > 0   ->   flow INTO the junction through port k (inflow)
        m_dot_k < 0   ->   flow OUT OF the junction through port k (outflow)

    Wire each port to the matching `_out` port of the adjacent component
    (`add_connection(pipe.m_dot_out, junction.m_dot_k)`); the directions then
    align automatically.

    Stream-variable port convention (per port, k = 0 .. N-1):

        p_k       - port pressure (always == p, glued via union-find)
        h_k       - port CARRIER enthalpy: the value any connected component
                    actually sees on its side of the wire.
                      * in outflow: h_k == h (well-mixed value flows out)
                      * in inflow : h_k == h_set_k (whatever upstream supplies)
                    blended smoothly across the zero-crossing.
        h_set_k   - port STREAM-IN enthalpy: the value the upstream component
                    *would* push into the junction if flow were inward at this
                    port.  The connected component MUST set `h_set_k` (e.g.
                    via `add_connection(src.h_set_out, junction.h_set_k)` or
                    by writing a residual `h_set_k - h_upstream == 0`); the
                    junction itself does not constrain it.  Modelica calls
                    the analogous concept a "stream variable" + `inStream()`.
        m_dot_k   - port mass flow rate (positive = into the junction).

    Decoupling the "stream-in" enthalpy from the "carrier" enthalpy is what
    makes a single, well-conditioned smooth-blend closure work at every flow
    direction; without it, the inflow-side residual collapses to `0 = 0`
    while the upstream's own `h = h_set` constraint persists, over-determining
    the global system.  See the design notes in `tests/test_mixing_junction.py`.

    Algebraic well-mixed conditions (both modes):
        p  - pressure                                             [Pa]
        h  - specific enthalpy                                    [J/kg]

    Dynamic-mode-only differential states:
        m  - total mass in the junction                           [kg]
        U  - total internal energy in the junction                [J]

    Equations (returned):

        # mass balance
        if dynamic:    dm/dt = sum_k m_dot_k
        if not dynamic: 0    = sum_k m_dot_k                     (algebraic)

        # energy balance, smooth donor-cell upwind on the per-port enthalpy
        # flux:
        flux_total = sum_k [ m_inflow_k * h_set_k - m_outflow_k * h ]

            if dynamic:    dU/dt = flux_total
            if not dynamic: 0    = flux_total
                                 + h_anchor_strength * (h_init - h)
                                                                   (algebraic
                                                                   + regularization)

        where the smooth `max`s are
            |m_dot_k|_smooth = sqrt(m_dot_k**2 + m_dot_eps**2)
            m_inflow_k       = (m_dot_k + |m_dot_k|_smooth) / 2   ~ max(m_dot_k, 0)
            m_outflow_k      = (|m_dot_k|_smooth - m_dot_k) / 2   ~ max(-m_dot_k, 0)

        # EoS closures (dynamic mode only -- they tie storage to (p, h))
        m = rho(p, h) * V
        U = m * h - p * V

        # per-port carrier-enthalpy closure (both modes, smooth blend,
        # always non-degenerate):
        h_k = alpha_k * h_set_k + (1 - alpha_k) * h
        where alpha_k = (1 + m_dot_k / |m_dot_k|_smooth) / 2  ~ step(m_dot_k > 0)

            - alpha_k -> 1 (port in heavy inflow):   h_k == h_set_k
            - alpha_k -> 0 (port in heavy outflow):  h_k == h
            - alpha_k = 0.5 (zero-flow):             h_k = (h_set_k + h) / 2

    Port pressure equalities (both modes, via union-find, eliminated
    structurally):
        p_k == p   for every k.

    Regularization of the quasi-static energy balance
    -------------------------------------------------
    At exact zero net flow the donor-cell coefficients collapse to
    `m_inflow_k = m_outflow_k = m_dot_eps / 2`, so the energy balance reduces
    to `(m_dot_eps / 2) * (sum h_set_k - N * h) = 0`, which still determines
    `h` but with a Jacobian row scaled by `m_dot_eps` -- tiny.  Newton can
    still converge but the step quality suffers.  We add an explicit
    `h`-anchor term that smoothly pulls `h` toward `h_init` only when the
    flux terms are small:

        flux_total + h_anchor_strength * (h_init - h)

    Default `h_anchor_strength = m_dot_eps` keeps the perturbation below
    `m_dot_eps * |h| ~ 1e-6 * 1e5 = 0.1 J/kg` at typical scales (well below
    Newton tol), while bumping the Jacobian floor up to `m_dot_eps` so the
    solver doesn't crawl at startup or after a reversal pass.  Pass
    `h_anchor_strength=0` to disable.  In `dynamic=True` mode this term is
    ignored because the EoS closure `U = m*h - p*V` already pins `h`
    independently of the flux balance.

    Other notes / simplifications:
      * Rigid wall (V constant in dynamic mode), no heat loss, no shaft work,
        no port-throttling.
      * Inflow / outflow kinetic energy is neglected (consistent with
        `PressureVessel` and `LoopBuffer`).
      * `m_dot_eps` controls the width of the smoothed direction-switch
        region.  Default `1e-6` kg/s is several orders below typical flow
        rates yet large enough to keep `d alpha / d m_dot` well-conditioned
        across the zero-crossing.  Tighten it (smaller value) for sharper
        switching at the cost of stiffer Newton iterations near reversal;
        loosen it for smoother but less direction-faithful blending.
    """

    #: When True, `declare_components` publishes one `FluidPort_phm` per port
    #: (`port_0` .. `port_{N-1}`) so the junction wires like any other
    #: thermofluid component.  Subclasses that expose their own directional
    #: ports (e.g. `LoopBuffer`) set this False to avoid duplicate connectors.
    _AUTO_FLUID_PORTS = True

    # `dynamic` toggles the bulk balance between differential storage (with
    # `der_m`/`der_U` states + EoS closures) and a purely algebraic mixer, so
    # the equation structure is NOT determined by the class alone.  Keying the
    # equation-template cache on it keeps a model mixing dynamic and
    # quasi-static junctions correct.  (This class also emits `add_connection`
    # side effects, which already routes it to the 'no-cache' path -- but the
    # key is declared defensively so correctness never relies on that.)
    def __init__(
        self,
        medium: CoolPropMedium,
        N: Annotated[int, ParamSpec("Number of ports (>= 2).", unit="1",
                    default=2)],
        V: Annotated[float | None, ParamSpec("Control volume (required for "
                    "dynamic=True storage; None for the algebraic mixer).",
                    unit="m^3", default=1e-3)] = None,
        p_init: Annotated[float, ParamSpec("Initial junction pressure.",
                         unit="Pa")] = 101325.0,
        T_init: Annotated[float, ParamSpec("Initial junction temperature.",
                         unit="K")] = 293.15,
        m_dot_eps: Annotated[float, ParamSpec("Smoothing scale of the donor-"
                            "cell direction switch (smaller = sharper but "
                            "stiffer).", unit="kg/s")] = 1e-6,
        dynamic: Annotated[bool, ParamSpec("If true, mass and internal energy "
                          "are differential storage states; if false a purely "
                          "algebraic ideal mixer.", structural=True)] = True,
        h_anchor_strength: Annotated[float | None, ParamSpec("Enthalpy "
                          "regulariser for the quasi-static energy balance; "
                          "None defaults to m_dot_eps (algebraic) or 0 "
                          "(dynamic).", unit="kg/s")] = None,
    ):
        if N < 2:
            raise ValueError(f"MixingJunction needs at least 2 ports, got N={N}")
        if dynamic and V is None:
            raise ValueError(
                "dynamic=True MixingJunction requires V (control volume) for "
                "the mass and internal-energy storage states.  Pass V=<m^3>, "
                "or set dynamic=False to use the purely-algebraic mixer."
            )
        self.medium = medium
        self.N = N
        self.V = V
        self.p_init = p_init
        self.T_init = T_init
        self.m_dot_eps = m_dot_eps
        self.dynamic = dynamic
        # Default the regularization to `m_dot_eps` for the quasi-static case
        # and to `0` for dynamic (where the EoS closure already conditions `h`).
        if h_anchor_strength is None:
            h_anchor_strength = 0.0 if dynamic else m_dot_eps
        self.h_anchor_strength = h_anchor_strength
        # Pre-compute thermodynamically consistent initial conditions.  Used
        # for variable seeds in both modes and for the storage states + EoS
        # closure in dynamic mode.
        self.h_init = float(medium.eval_h_pT(p_init, T_init))
        if dynamic:
            self.rho_init = float(medium.eval_rho_ph(p_init, self.h_init))
            self.m_init = self.rho_init * V
            self.U_init = self.m_init * self.h_init - p_init * V
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        # Well-mixed algebraic states (both modes).
        self.add_component('p', Variable(self.p_init, "Pa"))
        self.add_component('h', Variable(self.h_init, "J/kg"))

        # Direction-switch smoothing scale.  Held as a Parameter (not a baked
        # literal) so its appearance in `sqrt(m_dot**2 + m_dot_eps**2)` keeps
        # the equation template free of instance-varying numeric literals.
        self.add_component('m_dot_eps', Parameter(self.m_dot_eps, **spec['m_dot_eps'].param_kwargs()))

        # Storage states + volume parameter, dynamic mode only.
        if self.dynamic:
            self.add_component('V', Parameter(self.V, **spec['V'].param_kwargs()))
            self.add_component('m', DifferentialVariable(self.m_init, "kg"))
            self.add_component('U', DifferentialVariable(self.U_init, "J"))
        else:
            # Quasi-static `h`-anchor regularization constants, promoted to
            # Parameters for the same template-invariance reason (they appear
            # in the algebraic energy balance below).  `h_init` is computed
            # (no constructor arg), so it keeps an explicit unit.
            self.add_component('h_init', Parameter(self.h_init, "J/kg"))
            self.add_component('h_anchor_strength', Parameter(self.h_anchor_strength, **spec['h_anchor_strength'].param_kwargs()))

        # N ports.  Each port has both a CARRIER enthalpy (`h_k`, what the
        # downstream component actually sees through the wire) and a STREAM-IN
        # enthalpy (`h_set_k`, what the upstream would push into the junction
        # if flow were inward).  See the class docstring for why both are
        # needed for clean flow reversal.
        for k in range(self.N):
            self.add_component(f'p_{k}',     Variable(self.p_init, "Pa"))
            self.add_component(f'h_{k}',     Variable(self.h_init, "J/kg"))
            self.add_component(f'h_set_{k}', Variable(self.h_init, "J/kg"))
            self.add_component(f'm_dot_{k}', Variable(0.0, "kg/s"))

        # Publish each port as a standard FluidPort_phm so the junction wires
        # like every other thermofluid component (UI palette + Model.connect),
        # instead of requiring callers to hand-wire the `p_k`/`h_set_k`/`m_dot_k`
        # triple.  The across `h` channel binds to the per-port STREAM-IN
        # `h_set_k`: a connected upstream supplies its enthalpy there, which the
        # smooth donor-cell closure consumes to dictate the carrier `h_k`.  The
        # flow `m_dot_k` uses the junction-centric "into me" orientation shared
        # by every port in the package.
        if self._AUTO_FLUID_PORTS:
            for k in range(self.N):
                self.add_port(f'port_{k}', FluidPort_phm(
                    self,
                    channels={'p': self[f'p_{k}'],
                              'h': self[f'h_set_{k}'],
                              'm_dot': self[f'm_dot_{k}']},
                    flow_orientation='in',
                    medium=self.medium,
                ))

    def declare_equations(self):
        # No port-throttling: every port's pressure equals the well-mixed
        # pressure.  Routed via union-find so these never appear as residuals.
        for k in range(self.N):
            self.add_connection(self[f'p_{k}'], self['p'])

        p = self['p'].symbol
        h = self['h'].symbol
        m_dot_eps = self['m_dot_eps'].symbol

        # Accumulate mass / energy fluxes across ports, and emit one smoothed
        # direction-switch closure per port (these pieces are mode-independent).
        port_eqs = []
        sum_m_dot = 0
        sum_energy_flux = 0
        for k in range(self.N):
            m_dot_k = self[f'm_dot_{k}'].symbol
            h_k = self[f'h_{k}'].symbol
            h_set_k = self[f'h_set_{k}'].symbol
            # Smoothed |m_dot|, max(m_dot, 0), max(-m_dot, 0), step(m_dot > 0).
            abs_m = sp.sqrt(m_dot_k ** 2 + m_dot_eps ** 2)
            m_inflow_k = (m_dot_k + abs_m) / 2
            m_outflow_k = (abs_m - m_dot_k) / 2
            alpha_k = (1 + m_dot_k / abs_m) / 2

            sum_m_dot = sum_m_dot + m_dot_k
            # Donor-cell upwind on the energy flux: inflow brings in fluid
            # with the upstream's `h_set_k`; outflow carries away fluid with
            # the well-mixed `h`.  Smoothly weighted by the m_dot magnitudes
            # so the residual is C^0 across the zero-crossing.
            sum_energy_flux = sum_energy_flux + m_inflow_k * h_set_k - m_outflow_k * h
            # Carrier-h closure: a single, always-non-degenerate blend.
            # `(1 - alpha_k)` and `alpha_k` are smooth, strictly-positive
            # rational expressions of `m_dot_k`, so this residual stays a
            # proper Newton constraint at every flow direction -- including
            # the zero-crossing where it collapses to `h_k = (h + h_set_k)/2`.
            port_eqs.append(h_k - alpha_k * h_set_k - (1 - alpha_k) * h)

        if self.dynamic:
            # Mass + energy STORAGE residuals (Crank-Nicolson auto-couples
            # der_m, der_U to m, U).  EoS closures tie storage to (p, h).
            m = self['m'].symbol
            U = self['U'].symbol
            V = self['V'].symbol
            rho = self.medium.rho_ph(p, h)
            eq_mass = self['der_m'].symbol - sum_m_dot
            eq_energy = self['der_U'].symbol - sum_energy_flux
            eq_density = m - rho * V
            eq_energy_state = U - m * h + p * V
            return [eq_mass, eq_energy, eq_density, eq_energy_state] + port_eqs

        # Quasi-static mode: mass + energy balance as algebraic constraints,
        # no EoS, no V.  The energy balance carries an explicit `h`-anchor
        # regularization that smoothly pulls `h` toward `h_init` only when
        # the flux terms are tiny -- see the class docstring for the rationale
        # and the order-of-magnitude analysis.  Setting `h_anchor_strength=0`
        # disables it (use only if the network is guaranteed to never sit
        # at zero net flow).
        eq_mass = sum_m_dot
        eq_energy = sum_energy_flux + self['h_anchor_strength'].symbol * (self['h_init'].symbol - h)
        return [eq_mass, eq_energy] + port_eqs


class LoopBuffer(MixingJunction):
    """Two-port well-mixed buffer with directional `_in` / `_out` ports.

    Thin subclass over `MixingJunction(dynamic=True, N=2)` that aliases the
    indexed `_k` ports to the directional naming used throughout the rest
    of the package (`m_dot_in` > 0 means "flow into me", `m_dot_out` > 0
    means "flow out of me").  Apart from the names, everything physical
    -- mass + internal-energy storage, EoS closure, smooth donor-cell
    upwind, alpha-blend carrier -- comes from `MixingJunction`.

    Why this exists at all
    ----------------------
    Closed-loop pipe+pump topologies are structurally rank-deficient in
    steady state (continuity is implied because `rho*w` is conserved
    through every segment; adiabatic energy is implied because
    `h + w**2/2` is conserved).  The buffer's `m` and `U` differential
    states attach real residuals to the loop (`dm/dt = m_dot_in - m_dot_out`,
    `dU/dt = m_dot_in*h_in - m_dot_out*h` in forward flow) so that even
    at steady state `(m, U)` are pinned by initial conditions rather than
    collapsing to `0 = 0`.  See `tutorials/loop_pump_pipe.py` for the
    canonical usage.

    Bidirectional flow
    ------------------
    Unlike the legacy `LoopBuffer` (which was forward-only and would
    silently give wrong physics for `m_dot_in < 0`), this version
    inherits `MixingJunction`'s smooth donor-cell upwind, so flow
    reversal is handled cleanly:

      * Forward (`m_dot_in > 0`):  dU/dt contribution at inlet
        `=  m_dot_in * h_in  -  0 * h         = m_dot_in * h_in`
        (upstream's `h_in` flows in).
      * Reverse (`m_dot_in < 0`):  dU/dt contribution at inlet
        `=  0 * h_in  -  |m_dot_in| * h        = m_dot_in * h`
        (buffer's bulk `h` flows out through the inlet).

    The transition is C^0 across the zero-crossing thanks to the
    `sqrt(m_dot**2 + m_dot_eps**2)` smoothing inherited from
    `MixingJunction`, so Newton stays well-conditioned during reversal
    transients.

    Port API
    --------
    Modelica "flow into me" convention -- both ports' `m_dot_*` are positive
    when fluid enters the buffer through that face:

        p_in,  h_in,  m_dot_in       (driven by upstream;
                                      m_dot_in  > 0 = flow INTO buffer at inlet)
        p_out, h_out, m_dot_out      (drives downstream;
                                      m_dot_out > 0 = flow INTO buffer at outlet
                                                      = reverse axial flow)

    Mapping onto the inherited `MixingJunction` ports (the junction already
    uses "into me" semantics on every indexed port, so the two-port aliases
    collapse to direct unions with no sign flips):

        p_in       <-> p_0          (unioned)
        m_dot_in   <-> m_dot_0      (unioned, both "into buffer at port 0")
        h_in       <-> h_set_0      (h_in plays the role of the inlet's
                                     stream-in enthalpy -- this is what
                                     forward flow physically carries into
                                     the buffer)
        p_out      <-> p_1          (unioned)
        h_out      <-> h_1          (carrier; collapses to bulk h via the
                                     pin on h_set_1 below)
        m_dot_out  <-> m_dot_1      (unioned, both "into buffer at port 1")
        h_set_1     pinned to h     (the buffer's port API does NOT expose
                                     a separate downstream stream-in; this
                                     pin collapses the port-1 blend to
                                     `h_1 == h`, preserving the legacy
                                     `h_out == h` semantics in both
                                     directions)

    Reverse-flow physics caveat
    ---------------------------
    Because `h_set_1` is pinned to bulk `h`, reverse flow at the outlet
    is modelled as "buffer absorbing fluid back at its own h" -- the
    outlet contributes nothing to dU/dt in reverse.  For a network where
    the *downstream* component would push physically different `h` back
    into the buffer through the outlet port, use `MixingJunction(N=2)`
    directly and wire its `h_set_1` to the downstream's stream-in
    enthalpy.  All the inlet reversal physics, however, is exact.
    """

    # LoopBuffer exposes its own directional `inlet`/`outlet` ports below, so it
    # suppresses the inherited indexed `port_k` connectors to avoid duplicates.
    _AUTO_FLUID_PORTS = False

    def __init__(self, medium: CoolPropMedium, V,
                 p_init=101325.0, T_init=293.15, m_dot_eps=1e-6):
        super().__init__(medium=medium, N=2, V=V,
                         p_init=p_init, T_init=T_init,
                         m_dot_eps=m_dot_eps, dynamic=True)

    def declare_components(self):
        # Inherit the indexed _k ports + storage states + p, h, V.
        super().declare_components()
        # Add the directional alias ports.  Seeded with the same values as
        # the indexed ports so Newton starts on (or near) the manifold.
        self.add_component('p_in',      Variable(self.p_init, "Pa"))
        self.add_component('h_in',      Variable(self.h_init, "J/kg"))
        self.add_component('m_dot_in',  Variable(0.0, "kg/s"))
        self.add_component('p_out',     Variable(self.p_init, "Pa"))
        self.add_component('h_out',     Variable(self.h_init, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['p_in'], 'h': self['h_in'], 'm_dot': self['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        # Pull in the parent's storage residuals + port blends.
        eqs = super().declare_equations()

        # Inlet alias (port 0, no sign flip on m_dot).  Unioning `h_in` with
        # `h_set_0` makes h_in serve as the inlet's stream-in -- matches the
        # original "upstream supplies h_in" semantics, and the parent's
        # smooth-blend port closure now handles the reversal automatically.
        self.add_connection(self['p_in'],     self['p_0'])
        self.add_connection(self['m_dot_in'], self['m_dot_0'])
        self.add_connection(self['h_in'],     self['h_set_0'])

        # Outlet alias (port 1).  `h_out` is unioned with the carrier `h_1`;
        # we then pin h_set_1 to bulk h below, which makes the parent's
        # port-1 blend `h_1 = alpha_1*h_set_1 + (1-alpha_1)*h` collapse to
        # `h_1 == h` for every direction -- so `h_out == h` always, just
        # like the legacy LoopBuffer.
        self.add_connection(self['p_out'],    self['p_1'])
        self.add_connection(self['h_out'],    self['h_1'])

        # Under "flow into me" the buffer's `m_dot_out` and the parent's
        # `m_dot_1` are measuring the same physical quantity (both positive
        # when fluid enters the buffer at port 1), so they're unioned with
        # no sign flip -- UF eliminates one of {m_dot_out, m_dot_1} cleanly
        # at instantiate time.
        self.add_connection(self['m_dot_out'], self['m_dot_1'])
        # Pin downstream stream-in to bulk h (no separate API for it).
        # Trivially linear -- the reducer eliminates h_set_1 = h.
        eqs.append(self['h_set_1'].symbol - self['h'].symbol)

        return eqs



