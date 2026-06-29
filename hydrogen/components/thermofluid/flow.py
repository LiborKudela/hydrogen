"""Fluid-flow components of the `thermofluid` library, built on `hydrogen.model`.

Components: AmbientInlet, AmbientOutlet, TwoPortSegment, AdiabaticPump,
PressureOutlet, Splitter, PressureSource, PressureVessel, MixingJunction,
LoopBuffer, StraightPipe.

The typed connectors (`FluidPort_phm`, `ThermalPort_TQ`, `PermeationPort_pN`)
live in the sibling `ports` module; the leaky `TwoPortSegment` exposes a heat
port and a permeation leak port, so they are shared across this package.
"""

from __future__ import annotations

import warnings
from typing import Annotated

import numpy as np
import sympy as sp

from ...medium import CoolPropMedium
from ...model import DifferentialVariable, Model, Parameter, Variable
from ...numerics import G_const
from ...paramspec import ParamSpec, merged_param_specs
from ..control.control_components import RealSignal
from .ports import FluidPort_phm, PermeationPort_pN, ThermalPort_TQ

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

    #: The two callables can't carry a meaningful scalar type annotation, so
    #: their metadata stays here (merged with the Annotated specs below).
    PARAMS = {
        "f_factor_func": ParamSpec("Callable f(Re, epsilon, Dh) returning the "
                                  "Darcy friction factor symbolically."),
        "q_inflow_func": ParamSpec("Callable returning the radial heat input "
                                  "rate symbolically."),
    }

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
        f_factor_func,
        q_inflow_func,
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
        super().__init__()

    def get_churchill_f_factor(self, Re, epsilon, D):
        term1 = (8.0 / Re) ** 12
        A = (-2.457 * sp.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / Re) ** 16
        term2 = 1.0 / (A + B) ** 1.5
        return (term1 + term2) ** (1.0 / 12.0) * 8

    def calculate_nu(self, Re, Pr, fr):
        U = Re / 1000
        if Re <= 2300:
            return (fr / 8) * (Re - 1000) * Pr / (1 + 12.7 * (fr / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        if 2300 < Re <= 3100:
            return 3.52 * U ** 4 - 45.148 * U ** 3 + 212.13 * U ** 2 - 427.45 * U + 316.08
        return 3.66

    def calculate_nu_smooth(self, Re, Pr, fr):
        # `calculate_nu` smoothed via narrow transition windows so the symbolic system is
        # differentiable across the laminar/transitional/turbulent boundaries.
        U = Re / 1000
        nu_1 = (fr / 8) * (Re - 1000) * Pr / (1 + 12.7 * (fr / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        nu_2 = 3.52 * U ** 4 - 45.148 * U ** 3 + 212.13 * U ** 2 - 427.45 * U + 316.08
        nu_3 = 3.66
        TRANSITION_WIDTH = 100
        TRANSITION_CENTER_1 = 2250
        TRANSITION_CENTER_2 = 3050
        w1 = sp.Max(0.0, sp.Min(1.0, 1.0 - (Re - TRANSITION_CENTER_1) / TRANSITION_WIDTH))
        w3 = sp.Max(0.0, sp.Min(1.0, (Re - TRANSITION_CENTER_2) / TRANSITION_WIDTH))
        w2 = 1.0 - w1 - w3
        return w1 * nu_1 + w2 * nu_2 + w3 * nu_3

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
        Pr = mu * rho / k
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
      * ``"static"`` (default) -- quasi-steady mass / momentum / energy, exactly
        the `TwoPortSegment` physics.
      * ``"advective"`` / ``"compressible"`` -- transient enthalpy advection /
        + mass storage.  Scaffolded; not yet implemented.
    """

    _MULTIPHASE_MODES = ("single", "HEM")
    _DYNAMIC_LEVELS = ("static", "advective", "compressible")
    #: Instance flags that change the emitted equation structure, so the
    #: per-class template cache keeps heated / leaky / multiphase / sized
    #: variants distinct (see `Model.collect_equations`).
    _cache_key_flags = ("multiphase", "heat_port", "leaky", "dynamic", "N")

    PARAMS = {
        "f_factor_func": ParamSpec("Callable f(Re, epsilon, Dh) returning the "
                                  "Darcy friction factor symbolically."),
        "q_inflow_func": ParamSpec("Callable returning the radial heat input "
                                  "rate symbolically."),
    }

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
        f_factor_func=None,
        q_inflow_func=None,
        multiphase: Annotated[str, _SPEC_MULTIPHASE] = "single",
        heat_port: Annotated[bool, _SPEC_HEAT_PORT] = False,
        leaky: Annotated[bool, _SPEC_LEAKY] = False,
        dynamic: Annotated[str, ParamSpec("Dynamic modelling level: 'static' "
                          "(quasi-steady, default), 'advective', "
                          "'compressible'.",
                          choices=("static", "advective", "compressible"),
                          structural=True)] = "static",
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
        if dynamic != "static":
            raise NotImplementedError(
                f"SegmentedChannel dynamic={dynamic!r} is scaffolded but not yet "
                f"implemented; use 'static' (quasi-steady) for now.")
        if int(N) != N or N < 1:
            raise ValueError(f"N must be an integer >= 1, got {N!r}")
        self.multiphase = multiphase
        self.heat_port = bool(heat_port)
        self.leaky = bool(leaky)
        self.dynamic = dynamic
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
        h_std = float(medium.eval_h_pT(101325.0, 293.15))
        self._h_std = h_std
        self._rho_std = float(medium.eval_rho_ph(101325.0, h_std))
        self._mu_std = float(medium.eval_mu_ph(101325.0, h_std))
        self._k_std = float(medium.eval_k_ph(101325.0, h_std))
        super().__init__()

    # --- constitutive correlations (same as StraightPipe) ------------------
    def get_churchill_f_factor(self, Re, epsilon, D):
        term1 = (8.0 / Re) ** 12
        A = (-2.457 * sp.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / Re) ** 16
        term2 = 1.0 / (A + B) ** 1.5
        return (term1 + term2) ** (1.0 / 12.0) * 8

    def calculate_nu_smooth(self, Re, Pr, fr):
        U = Re / 1000
        nu_1 = (fr / 8) * (Re - 1000) * Pr / (1 + 12.7 * (fr / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        nu_2 = 3.52 * U ** 4 - 45.148 * U ** 3 + 212.13 * U ** 2 - 427.45 * U + 316.08
        nu_3 = 3.66
        TRANSITION_WIDTH = 100
        TRANSITION_CENTER_1 = 2250
        TRANSITION_CENTER_2 = 3050
        w1 = sp.Max(0.0, sp.Min(1.0, 1.0 - (Re - TRANSITION_CENTER_1) / TRANSITION_WIDTH))
        w3 = sp.Max(0.0, sp.Min(1.0, (Re - TRANSITION_CENTER_2) / TRANSITION_WIDTH))
        w2 = 1.0 - w1 - w3
        return w1 * nu_1 + w2 * nu_2 + w3 * nu_3

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        Re = w * Dh * rho / mu
        Pr = mu * rho / k
        nu = self.calculate_nu_smooth(Re, Pr, fr)
        alpha = nu * k / Dh
        return alpha * area * (T_wall - T)

    def get_q_inflow_adiabatic(self, w, p, h, rho, T, mu, k, fr, T_wall, Dh, area):
        return 0.0

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

        # N+1 shared faces, each carrying its own (single) closure set.
        for j in range(N + 1):
            self.add_component(f'p_{j}', Variable(101325, "Pa"))
            self.add_component(f'h_{j}', Variable(self._h_std, "J/kg"))
            self.add_component(f'M_{j}', Variable(0.0, "kg/s"))
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

    def declare_equations(self):
        N = self.N
        count = self['count'].symbol
        A = self['A'].symbol
        P = self['P'].symbol
        L_seg = self['L_segment'].symbol
        Dh = 4 * A / P
        T_ph, rho_ph, mu_ph, k_ph = self._property_funcs()
        eqs = []

        # --- per-face closures (ONE set per shared face) -------------------
        for j in range(N + 1):
            p = self[f'p_{j}'].symbol
            h = self[f'h_{j}'].symbol
            M = self[f'M_{j}'].symbol
            w = self[f'w_{j}'].symbol
            T = self[f'T_{j}'].symbol
            rho = self[f'rho_{j}'].symbol
            mu = self[f'mu_{j}'].symbol
            k = self[f'k_{j}'].symbol
            eqs.append(M - count * rho * A * w)
            eqs.append(T - T_ph(p, h))
            eqs.append(rho - rho_ph(p, h))
            eqs.append(mu - mu_ph(p, h))
            eqs.append(k - k_ph(p, h))

        # Boundary into-me flows (inlet/outlet port channels).
        eqs.append(self['m_dot_in'].symbol - self['M_0'].symbol)
        eqs.append(self['m_dot_out'].symbol + self[f'M_{N}'].symbol)

        # --- per-cell balances --------------------------------------------
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
        """The actual leaf symbols for cell ``i`` keyed by residual field."""
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
        """Residuals for a single cell as a pure function of its symbols.

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

        # energy
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
        # zero roughness are passed.
        super().__init__(medium, D=D, L=1.0, epsilon=0.0, z_in=z_in,
                         z_out=z_out, N=1, f_factor_func=self._no_friction,
                         q_inflow_func=self._no_heat)

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
    """Liquid / low-Mach valve sized by metric Kv with a linear opening trim.

        m_dot = (theta * Kv / 36000) * sign(dp) * sqrt(rho * |dp|)   [kg/s, Pa]

    where ``Kv`` is the metric flow coefficient (m^3/h of water at 1 bar) and
    ``theta`` the 0..1 opening.  Calibration: Kv=1, water (rho=1000), dp=1 bar
    gives 0.2778 kg/s = 1 m^3/h.  Valid while compressibility is negligible
    (``Delta p << p_1``); use `CompressibleValve` for gases at large pressure
    ratio.  The discontinuous ``sign(dp)*sqrt(|dp|)`` is regularised as
    ``dp / (dp^2 + dp_eps^2)^(1/4)`` so the Jacobian stays smooth through
    ``dp = 0`` (flow reversal).
    """

    UI_ICON = "valve.svg"

    def __init__(
        self,
        medium: CoolPropMedium,
        Kv: Annotated[float, ParamSpec("Metric flow coefficient (m^3/h of "
                     "water at 1 bar pressure drop).", unit="m^3/h",
                     default=1.0)],
        D, opening=1.0, z_in=0.0, z_out=0.0, dp_eps=1.0,
    ):
        self.Kv = Kv
        super().__init__(medium, D, opening=opening, z_in=z_in, z_out=z_out,
                         dp_eps=dp_eps)

    def declare_components(self):
        super().declare_components()
        self.add_component('Kv', Parameter(self.Kv,
                                           **merged_param_specs(type(self))['Kv'].param_kwargs()))

    def _valve_flow(self, dp, rho_avg, theta):
        C = self['Kv'].symbol / 36000.0
        dp_eps = self['dp_eps'].symbol
        return C * theta * sp.sqrt(rho_avg) * dp / (dp ** 2 + dp_eps ** 2) ** 0.25


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



