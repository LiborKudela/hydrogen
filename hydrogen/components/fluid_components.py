"""Reusable fluid-system components built on top of `hydrogen.model`.

Module layout:

  1. `FluidPort_phm` -- the typed port that this library exposes on every
     component.  Lives at the top of the module (rather than in the
     generic `hydrogen.ports`) so that each physics-domain library owns its
     own port kinds; `hydrogen.ports` only defines the generic `Port`
     base class and the shared error hierarchy.
  2. Components -- AmbientInlet, AmbientOutlet, TwoPortSegment,
     AdiabaticPump, PressureOutlet, Splitter, PressureSource,
     PressureVessel, MixingJunction, LoopBuffer, StraightPipe.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..medium import CoolPropMedium
from ..model import DifferentialVariable, Model, Parameter, Variable
from ..numerics import G_const
from ..ports import Port


# ---------------------------------------------------------------------------
# Fluid port -- (p, h, m_dot) interface used by every component below
# ---------------------------------------------------------------------------


class FluidPort_phm(Port):
    """Compressible-fluid interface carrying `(p, h, m_dot)`.

    * `p`       - port pressure                   [Pa]   (across)
    * `h`       - port specific enthalpy          [J/kg] (across)
    * `m_dot`   - port mass flow rate             [kg/s] (THROUGH;
                  positive = "INTO me" under the Modelica
                  "flow into me" convention used package-wide)

    All standard fluid components in this module declare either an
    `outlet` or an `inlet` port of this kind.  Both faces use
    `flow_orientation='in'` (positive m_dot enters the component),
    so `Model.connect()` emits a sum-to-zero on the flow channel
    when two same-orientation ports are wired -- the Kirchhoff /
    Modelica connector convention.

    Two FluidPort_phm of different `medium` are refused at
    connect-time (`PortMediumMismatchError`) to catch air<->hydrogen
    cross-wiring before it produces a confusing CoolProp NameError
    in the lambdified residual.
    """

    kind = "fluid_phm"
    required_channels = ("p", "h", "m_dot")
    flow_channels = ("m_dot",)


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

    def __init__(self, medium: CoolPropMedium, p_ambient=101325, T_ambient=293.15, m_flow=0.1, D=0.07):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self.m_flow = m_flow
        self.D = D
        super().__init__()

    def declare_components(self):
        self.add_component('p_ambient', Parameter(self.p_ambient, "Pa"))
        self.add_component('T_ambient', Variable(self.T_ambient, "K"))
        self.add_component('h_ambient', Parameter(self.medium.h_pT(self['p_ambient'].value, self['T_ambient'].value), "J/kg"))
        self.add_component('s_ambient', Parameter(self.medium.s_ph(self['p_ambient'].value, self['h_ambient'].value), "J/kg/K"))
        self.add_component('m_flow', Parameter(self.m_flow, "kg/s"))
        self.add_component('D', Parameter(self.D, "m"))
        self.add_component('p_out', Variable(self.p_ambient * 0.99, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(self.p_ambient, self.T_ambient) * 0.99, "J/kg"))
        self.add_component('m_dot_out', Variable(self.m_flow, "kg/s"))
        # Internal velocity Variable; kept as a leaf symbol so the isentropic
        # energy balance below sees `w_out**2 / 2` as a 1-leaf expression
        # rather than a nested `m_dot/(rho*A)` chain that bloats the Jacobian.
        self.add_component('w_out', Variable(0.2, "m/s"))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['p_out'], 'h': self['h_out'], 'm_dot': self['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        A = np.pi * self['D'].symbol ** 2 / 4
        # Continuity (mass-flow imposed): under "flow into me", positive
        # `m_dot_out` means fluid entering the inlet through its out-face
        # (reverse flow).  The user-facing `m_flow` parameter still means
        # "physical outflow rate" (positive forward), so we pin
        # `m_dot_out = -m_flow`, i.e. `m_flow + m_dot_out == 0`.  Trivial
        # and collapses to a Parameter substitution at instantiate time.
        eq1 = self['m_flow'].symbol + self['m_dot_out'].symbol

        # m_dot <-> w closure (nonlinear in rho, so the trivial reducer
        # leaves it alone -- keeps `w_out` as a leaf symbol in eq3 below).
        # Sign: `w_out` is axial forward, `m_dot_out` is "into me at the
        # out-face" (axial backward), so they sum to zero.
        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq_w = self['m_dot_out'].symbol + rho_out * self['w_out'].symbol * A

        h_in = self['h_ambient'].symbol
        s_in = self['s_ambient'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq2 = s_in - s_out
        eq3 = h_in - (self['h_out'].symbol + self['w_out'].symbol ** 2 / 2)
        eq4 = self['T_ambient'].symbol - self.T_ambient

        return [eq1, eq_w, eq2, eq3, eq4]


class AmbientOutlet(Model):
    """Outlet to ambient pressure (and matched isentropic conditions).

    Same equations as `AmbientInlet` minus the `T_ambient` constraint.
    """

    def __init__(self, medium: CoolPropMedium, p_ambient=101325, T_ambient=293.15, m_flow=0.1, D=0.07):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self.m_flow = m_flow
        self.D = D
        super().__init__()

    def declare_components(self):
        self.add_component('p_ambient', Parameter(self.p_ambient, "Pa"))
        self.add_component('T_ambient', Parameter(self.T_ambient, "K"))
        self.add_component('h_ambient', Parameter(self.medium.h_pT(self['p_ambient'].value, self['T_ambient'].value), "J/kg"))
        self.add_component('s_ambient', Parameter(self.medium.s_ph(self['p_ambient'].value, self['h_ambient'].value), "J/kg/K"))
        self.add_component('m_flow', Parameter(self.m_flow, "kg/s"))
        self.add_component('D', Parameter(self.D, "m"))
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
        # "Flow into me" sign convention: `m_dot_out` is positive when fluid
        # enters through the boundary's out-face.  The user-facing `m_flow`
        # remains "physical outflow rate" (positive forward), hence the `+`
        # in both continuity and the m_dot<->w closure (see `AmbientInlet`).
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
    `m_dot = rho * w * A` per face, which the trivial-equation reducer
    can NOT collapse (rho is a nonlinear function of `(p, h)`), so the
    `w_in` / `w_out` symbols stay leaf-shaped through code generation.

    Why both?  Putting `m_dot` on the port unifies mass-flow across joints
    regardless of cross-sectional-area mismatch (`add_connection` of
    velocities silently breaks mass conservation when areas differ).
    Keeping `w` as a Variable (instead of a derived expression
    `m_dot / (rho * A)` substituted everywhere) keeps the per-equation
    expression tree shallow -- otherwise every `w` reference inflates to a
    nested `m_dot / (rho_ph(p, h) * A)` chain, which both slows the trivial
    reducer (chained substitutions through hundreds of equations) and
    bloats the lambdified Jacobian (more CoolProp evaluations per non-zero).

    `f_factor_func(Re, epsilon, Dh)` returns the friction factor symbolically.
    `q_inflow_func(w, p, h, rho, T, mu, k, fr, T_wall)` returns the heat input rate.

    Each of the seven geometry slots (`A_in`, `A_out`, `P_in`, `P_out`, `z_in`,
    `z_out`, `L`) may be passed either as a plain Python scalar OR as an
    existing `Parameter` instance owned by a parent `Model`.  The
    `Parameter(...)` constructor itself handles the dispatch (see
    `Parameter.__new__`): a scalar produces a fresh local Parameter, an
    existing Parameter produces a transparent `ParameterAlias` so two
    sibling segments end up with the SAME SymPy symbol in their equations.
    This is what lets `Model.remove_duplicate_equations` collapse the per-
    face `m_dot = rho*A*w` closures across internal interfaces of a uniform
    `StraightPipe`.

    Face thermodynamic properties (`T`, `rho`, `mu`, `k` at the in/out faces)
    are exposed as explicit algebraic Variables with closure equations
    `rho_in - rho_ph(p_in, h_in) == 0` (etc.).  Materialising them as leaf
    symbols serves two purposes:
      1. The downstream momentum / energy / friction expressions reference
         the leaves directly, so CoolProp calls appear in the residual
         exactly four times per face (rho, T, mu, k) -- not once per use.
      2. After `add_connection` unifies the upstream-face `(p, h)` of segment
         k+1 with the downstream-face of segment k, the two segments' face
         closures become structurally identical.  The iterated
         `Model.remove_duplicate_equations` pass collapses them, leaving
         one closure per *physical* interface rather than per *segment side*,
         and likewise unifies the face Variables.  For a uniform N-segment
         pipe that removes ~4*(N-1) CoolProp evaluations per Newton iteration.
    """

    def __init__(self, medium: CoolPropMedium, A_in, A_out, P_in, P_out, z_in, z_out, L, epsilon, f_factor_func, q_inflow_func):
        self.medium = medium
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
        # Cache the standard-state property values used as initial guesses
        # for the face-property Variables.  Computed once per segment via
        # the medium's scalar `eval_*_ph` callbacks so each Variable starts
        # the Newton solve at a physically reasonable order of magnitude
        # (otherwise the very first `eval_residual` would see e.g. mu=1
        # while the actual value is ~1e-5 -- enough to push the line-
        # search step into ridiculous regions).
        h_std = float(medium.eval_h_pT(101325.0, 293.15))
        self._h_std = h_std
        self._rho_std = float(medium.eval_rho_ph(101325.0, h_std))
        self._mu_std = float(medium.eval_mu_ph(101325.0, h_std))
        self._k_std = float(medium.eval_k_ph(101325.0, h_std))
        super().__init__()

    def declare_components(self):
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self._h_std, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('w_in', Variable(0.1, "m/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self._h_std, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_component('w_out', Variable(0.1, "m/s"))
        # Face thermodynamic properties as leaf Variables.  See class
        # docstring for the rationale; closure equations are declared in
        # `declare_equations` and the iterated duplicate-equation pass
        # collapses them across shared interfaces of adjacent segments.
        self.add_component('T_in',   Variable(293.15,         "K"))
        self.add_component('T_out',  Variable(293.15,         "K"))
        self.add_component('rho_in', Variable(self._rho_std,  "kg/m^3"))
        self.add_component('rho_out',Variable(self._rho_std,  "kg/m^3"))
        self.add_component('mu_in',  Variable(self._mu_std,   "Pa*s"))
        self.add_component('mu_out', Variable(self._mu_std,   "Pa*s"))
        self.add_component('k_in',   Variable(self._k_std,    "W/m/K"))
        self.add_component('k_out',  Variable(self._k_std,    "W/m/K"))
        self.add_component('A_in', Parameter(self.A_in, "m^2"))
        self.add_component('A_out', Parameter(self.A_out, "m^2"))
        self.add_component('P_in', Parameter(self.P_in, "m"))
        self.add_component('P_out', Parameter(self.P_out, "m"))
        self.add_component('z_in', Parameter(self.z_in, "m"))
        self.add_component('z_out', Parameter(self.z_out, "m"))
        self.add_component('L', Parameter(self.L, "m"))
        self.add_component('T_wall', Parameter(293.15, "K"))
        self.add_component('q_inflow', Variable(0.0, "W"))
        # Directional fluid ports.  BOTH faces use the Modelica "flow into
        # me" convention -- positive m_dot means fluid entering the segment
        # through that face -- so `flow_orientation='in'` on both, and the
        # continuity equation `m_dot_in + m_dot_out == 0` (Kirchhoff on the
        # CV) replaces the old axial-positive `m_dot_in - m_dot_out == 0`.
        # See class docstring.
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
        # Face properties as LEAF symbols.  Closure equations below bind
        # each Variable to its CoolProp evaluation; everything downstream
        # references the leaf so the residual / Jacobian see no CoolProp
        # calls anywhere except in those 8 closures.
        T_in    = self['T_in'].symbol
        T_out   = self['T_out'].symbol
        rho_in  = self['rho_in'].symbol
        rho_out = self['rho_out'].symbol
        mu_in   = self['mu_in'].symbol
        mu_out  = self['mu_out'].symbol
        k_in    = self['k_in'].symbol
        k_out   = self['k_out'].symbol

        # Local aliases -- leaf symbols only, so every reference below stays
        # cheap to evaluate and cheap to differentiate.
        w_in = self['w_in'].symbol
        w_out = self['w_out'].symbol
        m_dot_in = self['m_dot_in'].symbol
        m_dot_out = self['m_dot_out'].symbol

        p_avg = (self['p_in'].symbol + self['p_out'].symbol) / 2
        h_avg = (self['h_in'].symbol + self['h_out'].symbol) / 2
        T_avg = (T_in + T_out) / 2
        mu_avg = (mu_in + mu_out) / 2
        rho_avg = (rho_in + rho_out) / 2
        k_avg = (k_in + k_out) / 2
        w_avg = (w_in + w_out) / 2
        A_avg = (self['A_in'].symbol + self['A_out'].symbol) / 2
        # Axial-positive mass-flow average: under the "flow into me"
        # convention `m_dot_out` measures fluid *entering* through the out-
        # face (i.e. axially backward), so the forward-flow rate is
        # `m_dot_in == -m_dot_out`.  After continuity collapses one of
        # `{m_dot_in, m_dot_out}` via the trivial reducer (or via signed UF
        # if the residual is routed through `add_connection`), this
        # difference evaluates to the surviving leaf with axial orientation
        # -- which is what every downstream momentum / energy term wants.
        m_dot_avg = (m_dot_in - m_dot_out) / 2

        Dh_in = 4 * self['A_in'].symbol / self['P_in'].symbol
        Dh_out = 4 * self['A_out'].symbol / self['P_out'].symbol
        Dh_avg = (Dh_in + Dh_out) / 2

        Re_avg = rho_avg * abs(w_avg) * Dh_avg / mu_avg

        # Mass-flow continuity (Kirchhoff on the CV under "flow into me":
        # net inflow sums to zero).  Linear in `m_dot_in, m_dot_out` with
        # unit coefficients, so the trivial-equation reducer eliminates one
        # leaf at instantiate time and the residual contributes nothing at
        # runtime.
        eq_continuity = m_dot_in + m_dot_out

        # m_dot <-> w closures, one per face.  Nonlinear in (rho, w) so the
        # trivial reducer leaves them alone -- which is exactly what we want
        # so that `w_in` / `w_out` keep being leaf symbols in the heavy
        # friction / momentum / energy expressions below.  When two sibling
        # segments share their geometry Parameters (e.g. all segments of a
        # `StraightPipe` referencing the same `A` symbol via
        # `ParameterAlias`), the equation-deduplication pass run during
        # `instantiate()` collapses `w_out(seg_k)` and `w_in(seg_{k+1})` to
        # a single variable at every internal interface, dropping one
        # equation per face.
        #
        # Sign of `w_out` term: `w_out` is the axial velocity (positive
        # forward), but `m_dot_out` is "into me at out-face" (positive
        # backward) -- opposite orientations.  So the closure carries a
        # `+ rho*A*w_out` term (sum-to-zero), unlike the `- rho*A*w_in` on
        # the inlet face where both quantities point inward.
        eq_w_in = m_dot_in - rho_in * self['A_in'].symbol * w_in
        eq_w_out = m_dot_out + rho_out * self['A_out'].symbol * w_out

        # Momentum
        f_avg = self.f_factor_func(Re_avg, self.epsilon, Dh_avg)
        delta_P_friction = f_avg * (self['L'].symbol / Dh_avg) * (rho_avg * abs(w_avg) * w_avg / 2)
        momentum_flux = m_dot_avg * (w_out - w_in)
        buoyancy_force = -G_const * (self['z_out'].symbol - self['z_in'].symbol) * A_avg * rho_avg
        eq_momentum = self['p_in'].symbol * self['A_in'].symbol - self['p_out'].symbol * self['A_out'].symbol - delta_P_friction * A_avg - momentum_flux + buoyancy_force

        # Energy
        q = self.q_inflow_func(w_avg, p_avg, h_avg, rho_avg, T_avg, mu_avg, k_avg, f_avg, self['T_wall'].symbol)
        # Steady CV energy balance assuming forward flow:
        #     (h + w^2/2)|in + q = (h + w^2/2)|out
        # Under reverse flow, fluid enters at "out" and leaves at "in", so q must
        # be credited at the actual upstream port.  A smoothed sign(w_avg) picks
        # the correct side:
        #     forward (w_avg > 0):  +q -> heat added to outflow at "out"
        #     reverse (w_avg < 0):  -q -> heat added to outflow at "in"
        # `w_eps` keeps the Jacobian smooth across zero-flow crossings (kink
        # otherwise hurts Newton convergence near startup / flow reversal).
        # Pick `w_eps` well below realistic operating velocities; for hydrogen /
        # air plumbing 1e-4 m/s is several orders below any real flow yet large
        # enough to keep d sign_w / d w_avg well-conditioned.
        w_eps = 1e-4
        sign_w = w_avg / sp.sqrt(w_avg ** 2 + w_eps ** 2)
        eq_energy = self['h_in'].symbol + w_in ** 2 / 2 + sign_w * q - (self['h_out'].symbol + w_out ** 2 / 2)

        # `q_inflow` stays the raw magnitude of heat transfer (direction-
        # independent) so users get a meaningful "wall heat input" diagnostic
        # regardless of which way the fluid happens to be flowing.
        eq_q_diag = self['q_inflow'].symbol - q

        # Face-property closures.  Each is `leaf - X_ph(p, h) == 0`, i.e.
        # linear in its dedicated leaf with a constant (= 1) coefficient
        # and a (p, h)-only "rest" term.  Two adjacent segments meeting at
        # an internal interface end up with structurally identical closures
        # (after `add_connection` unifies (p, h) across the face and
        # `_decompose` in the dedup pass picks the leaf as `var`); the
        # iterated `Model.remove_duplicate_equations` then collapses the
        # closure pair and unifies the two leaves.
        p_in_sym = self['p_in'].symbol
        p_out_sym = self['p_out'].symbol
        h_in_sym = self['h_in'].symbol
        h_out_sym = self['h_out'].symbol
        eq_T_in    = T_in    - self.medium.T_ph(p_in_sym, h_in_sym)
        eq_rho_in  = rho_in  - self.medium.rho_ph(p_in_sym, h_in_sym)
        eq_mu_in   = mu_in   - self.medium.mu_ph(p_in_sym, h_in_sym)
        eq_k_in    = k_in    - self.medium.k_ph(p_in_sym, h_in_sym)
        eq_T_out   = T_out   - self.medium.T_ph(p_out_sym, h_out_sym)
        eq_rho_out = rho_out - self.medium.rho_ph(p_out_sym, h_out_sym)
        eq_mu_out  = mu_out  - self.medium.mu_ph(p_out_sym, h_out_sym)
        eq_k_out   = k_out   - self.medium.k_ph(p_out_sym, h_out_sym)

        return [
            eq_continuity, eq_w_in, eq_w_out, eq_momentum, eq_energy, eq_q_diag,
            eq_T_in,  eq_rho_in,  eq_mu_in,  eq_k_in,
            eq_T_out, eq_rho_out, eq_mu_out, eq_k_out,
        ]


class AdiabaticPump(TwoPortSegment):
    """Adiabatic pump segment using a custom friction model `f = a_iz / (Re*Dh)`."""

    def __init__(self, medium: CoolPropMedium, A_in, A_out, P_in, P_out, z_in, z_out):
        super().__init__(medium, A_in, A_out, P_in, P_out, z_in, z_out, 1, 0.0, self.get_f_from_a_iz, self.get_q_inflow)
        self.adiabatic = True

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall):
        return 0.0

    def get_f_from_a_iz(self, Re_avg, epsilon, Dh_avg):
        return self['a_iz'].symbol / (Re_avg * Dh_avg)

    def declare_components(self):
        super().declare_components()
        self.add_component('a_iz', Variable(0.0, "W"))


class PressureOutlet(Model):
    """Fixed-pressure outlet boundary.

    Imposes `p_in = p_ambient` at the boundary plane and lets the upstream component
    determine `(h_in, m_dot_in)`. This is the pressure-imposed dual of `PressureSource`:
    use it to terminate plumbing that exhausts to atmosphere or any other fixed-pressure
    sink, when you want the *system* to determine the mass flow rather than imposing it.

    Port (matches the (p, h, m_dot) convention used everywhere):
        p_in, h_in, m_dot_in    - external connection inputs (driven by upstream)
    """

    def __init__(self, medium: CoolPropMedium, p_ambient=101325.0, T_ambient=293.15):
        self.medium = medium
        self.p_ambient = p_ambient
        self.T_ambient = T_ambient
        self._h_ambient = float(medium.eval_h_pT(p_ambient, T_ambient))
        super().__init__()

    def declare_components(self):
        self.add_component('p_ambient', Parameter(self.p_ambient, "Pa"))
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

    def __init__(self, medium: CoolPropMedium, K):
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

    where `h_total = h(p_source, T_source)` and `s_total = s(p_source, h_total)`. The
    third closure (`p_out`) comes from whatever this source is wired into downstream.
    Use this when you want flow to be driven by a pressure differential — e.g. filling
    a vessel from a pressurised line: as the vessel back-pressure rises the inlet
    velocity naturally decays toward zero.

    `A` is the cross-sectional area of the boundary plane, needed to translate
    the port's mass flow rate `m_dot_out` into the velocity used in the kinetic-
    energy term.  Set it to the area of the downstream port the source is wired
    into; for low-Mach flows the answer is barely sensitive to the exact value
    because the KE correction is typically <1% of the stagnation pressure.

    Port (matches the (p, h, m_dot) convention used everywhere):
        p_out, h_out, m_dot_out    - drives the downstream component
    """

    def __init__(self, medium: CoolPropMedium, p_source=101325, T_source=293.15, A=1e-3):
        self.medium = medium
        self.p_source = p_source
        self.T_source = T_source
        self.A = A
        self._h_total = float(medium.eval_h_pT(p_source, T_source))
        self._s_total = float(medium.eval_s_ph(p_source, self._h_total))
        super().__init__()

    def declare_components(self):
        self.add_component('p_source', Parameter(self.p_source, "Pa"))
        self.add_component('T_source', Parameter(self.T_source, "K"))
        self.add_component('h_total', Parameter(self._h_total, "J/kg"))
        self.add_component('s_total', Parameter(self._s_total, "J/kg/K"))
        self.add_component('A', Parameter(self.A, "m^2"))
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
        return [eq_w, eq_isentropic, eq_energy]


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

    Notes / simplifications:
      * Rigid wall (V constant), no heat loss, no shaft work, no outflow.
      * Inflow kinetic energy is neglected.  For typical vessel-filling regimes
        the contribution `(m_dot_in / (rho * A))**2 / 2` is several orders of
        magnitude below `h_in`; if you need it, add it to the energy balance
        below (using the `A_in` parameter the component already carries).
      * Reverse flow is not modeled.  If `m_dot_in` becomes negative the energy
        balance will still integrate, but `h_in` would no longer represent the
        true outflow enthalpy (you'd need an upwinding switch on `h_in <-> h`).
    """

    def __init__(self, medium: CoolPropMedium, V, A_in, p_init=101325.0, T_init=293.15):
        self.medium = medium
        self.V = V
        self.A_in = A_in
        self.p_init = p_init
        self.T_init = T_init
        # Pre-compute thermodynamically consistent initial conditions so the t=0 Newton
        # solve starts near a converged state.
        self.h_init = float(medium.eval_h_pT(p_init, T_init))
        self.rho_init = float(medium.eval_rho_ph(p_init, self.h_init))
        self.m_init = self.rho_init * V
        self.U_init = self.m_init * self.h_init - p_init * V  # U = m*u = m*h - p*V
        super().__init__()

    def declare_components(self):
        self.add_component('V', Parameter(self.V, "m^3"))
        self.add_component('A_in', Parameter(self.A_in, "m^2"))

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

    def declare_equations(self):
        m = self['m'].symbol
        U = self['U'].symbol
        p = self['p'].symbol
        h = self['h'].symbol
        V = self['V'].symbol
        p_in = self['p_in'].symbol
        h_in = self['h_in'].symbol

        rho = self.medium.rho_ph(p, h)

        # With m_dot at the port, the inflow mass rate is the port value itself
        # -- no density/area product needed.
        m_in_dot = self['m_dot_in'].symbol

        # Continuity: m grows at the inflow mass rate.
        eq_mass = self['der_m'].symbol - m_in_dot
        # Energy: adiabatic open system with inflow enthalpy (KE neglected).
        eq_energy = self['der_U'].symbol - m_in_dot * h_in

        # Algebraic closure linking (m, U) to (p, h) via the equation of state.
        eq_density = m - rho * V
        eq_energy_state = U - m * h + p * V

        # Port pressure equality: vessel pressure feeds back as the upstream's back-pressure.
        eq_port_p = p_in - p

        return [eq_mass, eq_energy, eq_density, eq_energy_state, eq_port_p]


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

    def __init__(self, medium: CoolPropMedium, N, V=None,
                 p_init=101325.0, T_init=293.15,
                 m_dot_eps=1e-6, dynamic=True, h_anchor_strength=None):
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
        # Well-mixed algebraic states (both modes).
        self.add_component('p', Variable(self.p_init, "Pa"))
        self.add_component('h', Variable(self.h_init, "J/kg"))

        # Storage states + volume parameter, dynamic mode only.
        if self.dynamic:
            self.add_component('V', Parameter(self.V, "m^3"))
            self.add_component('m', DifferentialVariable(self.m_init, "kg"))
            self.add_component('U', DifferentialVariable(self.U_init, "J"))

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

    def declare_equations(self):
        # No port-throttling: every port's pressure equals the well-mixed
        # pressure.  Routed via union-find so these never appear as residuals.
        for k in range(self.N):
            self.add_connection(self[f'p_{k}'], self['p'])

        p = self['p'].symbol
        h = self['h'].symbol
        m_dot_eps = self.m_dot_eps

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
        eq_energy = sum_energy_flux + self.h_anchor_strength * (self.h_init - h)
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
    collapsing to `0 = 0`.  See `examples/loop_pump_pipe.py` for the
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


class StraightPipe(Model):
    """1D pipe split into `n_segments` equal-length `TwoPortSegment`s with optional heat transfer.

    Uses the Churchill correlation for the friction factor and a smoothed
    Gnielinski / laminar Nusselt blend for the wall heat-transfer coefficient.
    """

    def __init__(self, medium: CoolPropMedium, D, L, epsilon, z_in, z_out, n_segments=3, adiabatic=False):
        self.medium = medium
        self.D = D
        self.L = L
        self.epsilon = epsilon
        self.z_in = z_in
        self.z_out = z_out
        self.n_segments = n_segments
        self.adiabatic = adiabatic
        super().__init__()

    def get_churchill_f_factor(self, Re, epsilon, D):
        term1 = (8.0 / (Re + 1)) ** 12
        A = (-2.457 * sp.log((7.0 / Re) ** 0.9 + 0.27 * epsilon / D)) ** 16
        B = (37530.0 / (Re + 1)) ** 16
        term2 = 1.0 / (A + B) ** 1.5
        f = (term1 + term2) ** (1.0 / 12.0)
        return f * 8

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

    def get_q_inflow(self, w, p, h, rho, T, mu, k, fr, T_wall):
        Re = w * self.D * rho / mu
        Pr = mu * rho / k
        nu = self.calculate_nu_smooth(Re, Pr, fr)
        alpha = nu * k / self.D
        area = np.pi * self.D * self.L
        return alpha * area * (T_wall - T)

    def get_q_inflow_adiabatic(self, w, p, h, rho, T, mu, k, fr, T_wall):
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
        self.add_component('D', Parameter(self.D, "m"))
        self.add_component('L', Parameter(self.L, "m"))
        self.add_component('epsilon', Parameter(self.epsilon, "m"))
        self.add_component('z_in', Parameter(self.z_in, "m"))
        self.add_component('z_out', Parameter(self.z_out, "m"))
        # Derived shared geometry.  We register them as their own Parameters
        # (rather than building `pi*D**2/4` as a SymPy expression every time
        # a segment references it) because Parameter symbols are flat
        # leaves: the per-segment friction / momentum / energy expression
        # trees stay shallow, and the equation-dedup signature comparison
        # is a single-symbol hash rather than a full subtree walk.
        A_value = np.pi * self.D ** 2 / 4
        P_value = np.pi * self.D
        L_segment_value = self.L / self.n_segments
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
        for i in range(self.n_segments):
            fr_f = self.get_churchill_f_factor
            q_f = self.get_q_inflow if not self.adiabatic else self.get_q_inflow_adiabatic
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
