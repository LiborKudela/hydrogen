"""Reusable fluid-system components built on top of `hydrogen.model`."""

from __future__ import annotations

import numpy as np
import sympy as sp

from .medium import CoolPropMedium
from .model import DifferentialVariable, Model, Parameter, Variable
from .numerics import G_const


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

    def declare_equations(self):
        A = np.pi * self['D'].symbol ** 2 / 4
        # Continuity (mass-flow imposed): the port carries m_dot_out, the
        # constitutive parameter `m_flow` sets its value.  This is trivial and
        # collapses to a Parameter substitution at instantiate time.
        eq1 = self['m_flow'].symbol - self['m_dot_out'].symbol

        # m_dot <-> w closure (nonlinear in rho, so the trivial reducer
        # leaves it alone -- keeps `w_out` as a leaf symbol in eq3 below).
        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq_w = self['m_dot_out'].symbol - rho_out * self['w_out'].symbol * A

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

    def declare_equations(self):
        A = np.pi * self['D'].symbol ** 2 / 4
        eq1 = self['m_flow'].symbol - self['m_dot_out'].symbol

        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        eq_w = self['m_dot_out'].symbol - rho_out * self['w_out'].symbol * A

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

    `m_dot` is the mass flow rate [kg/s] (signed by physical flow direction).
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
        super().__init__()

    def declare_components(self):
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('w_in', Variable(0.1, "m/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))
        self.add_component('w_out', Variable(0.1, "m/s"))
        self.add_component('A_in', Parameter(self.A_in, "m^2"))
        self.add_component('A_out', Parameter(self.A_out, "m^2"))
        self.add_component('P_in', Parameter(self.P_in, "m"))
        self.add_component('P_out', Parameter(self.P_out, "m"))
        self.add_component('z_in', Parameter(self.z_in, "m"))
        self.add_component('z_out', Parameter(self.z_out, "m"))
        self.add_component('L', Parameter(self.L, "m"))
        self.add_component('T_wall', Parameter(293.15, "K"))
        self.add_component('q_inflow', Variable(0.0, "W"))

    def declare_equations(self):
        # volume averages
        T_in = self.medium.T_ph(self['p_in'].symbol, self['h_in'].symbol)
        rho_in = self.medium.rho_ph(self['p_in'].symbol, self['h_in'].symbol)
        mu_in = self.medium.mu_ph(self['p_in'].symbol, self['h_in'].symbol)
        k_in = self.medium.k_ph(self['p_in'].symbol, self['h_in'].symbol)

        T_out = self.medium.T_ph(self['p_out'].symbol, self['h_out'].symbol)
        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        mu_out = self.medium.mu_ph(self['p_out'].symbol, self['h_out'].symbol)
        k_out = self.medium.k_ph(self['p_out'].symbol, self['h_out'].symbol)

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
        m_dot_avg = (m_dot_in + m_dot_out) / 2

        Dh_in = 4 * self['A_in'].symbol / self['P_in'].symbol
        Dh_out = 4 * self['A_out'].symbol / self['P_out'].symbol
        Dh_avg = (Dh_in + Dh_out) / 2

        Re_avg = rho_avg * abs(w_avg) * Dh_avg / mu_avg

        # Mass-flow continuity end-to-end (trivially `m_dot_in == m_dot_out`;
        # collapsed by the trivial-equation reducer at instantiate time so it
        # contributes nothing at runtime).
        eq_continuity = m_dot_in - m_dot_out

        # m_dot <-> w closures, one per face.  Nonlinear in (rho, w) so the
        # trivial reducer leaves them alone -- which is exactly what we want
        # so that `w_in` / `w_out` keep being leaf symbols in the heavy
        # friction / momentum / energy expressions below.
        eq_w_in = m_dot_in - rho_in * self['A_in'].symbol * w_in
        eq_w_out = m_dot_out - rho_out * self['A_out'].symbol * w_out

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

        return [eq_continuity, eq_w_in, eq_w_out, eq_momentum, eq_energy, eq_q_diag]


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

    def declare_equations(self):
        # The K pressure and K enthalpy equalities are pure variable-equality
        # constraints -- short-circuit them via union-find instead of building
        # 2K sympy Add nodes for the trivial reducer to chew through later.
        for k in range(self.K):
            self.add_connection(self[f'p_out_{k}'], self['p_in'])
            self.add_connection(self[f'h_out_{k}'], self['h_in'])

        m_out = 0
        for k in range(self.K):
            m_out = m_out + self[f'm_dot_out_{k}'].symbol
        return [self['m_dot_in'].symbol - m_out]


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

    def declare_equations(self):
        h_total = self['h_total'].symbol
        s_total = self['s_total'].symbol
        s_out = self.medium.s_ph(self['p_out'].symbol, self['h_out'].symbol)
        rho_out = self.medium.rho_ph(self['p_out'].symbol, self['h_out'].symbol)
        # m_dot <-> w closure (nonlinear in rho; trivial reducer leaves it alone).
        eq_w = self['m_dot_out'].symbol - rho_out * self['w_out'].symbol * self['A'].symbol
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


class LoopBuffer(Model):
    """Two-port well-mixed lumped-volume buffer for closed-loop topologies.

    Like `PressureVessel` but with both an inflow and an outflow port, so it
    can sit *inline* in a flow path.  The point of having one of these in a
    closed loop is to break the structural rank-deficiency that an
    otherwise-pure pipe+pump loop carries:

      * In a fully-wired closed loop, the steady-state continuity equations
        are linearly dependent (`rho*w` is conserved through every segment,
        so the loop-closing continuity equation is implied by the others).
      * In an adiabatic closed loop, the steady-state energy equations are
        likewise dependent (`h + w**2/2` is conserved through every
        adiabatic segment).

    The buffer breaks both: its `m` and `U` are *state* variables that
    enter their own residuals (`dm/dt = m_in - m_out`,
    `dU/dt = m_in*h_in - m_out*h_out`).  Even at steady state where
    `dm/dt = dU/dt = 0`, those residuals constrain `(m, U)` (and via the
    EoS closure `(p, h)`) to specific values rather than collapsing to
    `0 = 0`, so the global Jacobian stays full rank.

    Differential states:
        m  - total mass in the buffer                            [kg]
        U  - total internal energy in the buffer                 [J]

    Algebraic states (well-mixed = single (p, h) inside the volume):
        p  - pressure                                            [Pa]
        h  - specific enthalpy                                   [J/kg]

    Ports (matches the (p, h, m_dot) convention used everywhere):
        p_in,  h_in,  m_dot_in       (driven by the upstream component)
        p_out, h_out, m_dot_out      (drives the downstream component)

    Equations (returned):
        dm/dt = m_dot_in - m_dot_out                        (continuity)
        dU/dt = m_dot_in * h_in - m_dot_out * h             (energy)
        m  = rho(p, h) * V                                  (density closure)
        U  = m*h - p*V                                      (internal-energy closure)

    Port equalities, declared via `add_connection` (so they collapse out of
    the symbolic Jacobian via union-find rather than living as residuals):
        p_in  == p
        p_out == p              (no inlet/outlet throttling)
        h_out == h              (well-mixed: outflow carries the vessel's h)

    `h_in` and the two mass flow rates `m_dot_in`, `m_dot_out` stay as free
    port variables, fixed by whatever the buffer is wired into.

    Notes / simplifications:
      * Rigid wall (V constant), no heat loss, no shaft work.
      * Inflow / outflow kinetic energy is neglected (consistent with
        `PressureVessel`).  For typical loop applications `h >> w**2/2`.
      * Reverse flow is not modelled.  If `m_dot_in` becomes negative the
        energy balance still integrates, but the inlet would be carrying
        downstream conditions, not the prescribed `h_in`.
    """

    def __init__(self, medium: CoolPropMedium, V,
                 p_init=101325.0, T_init=293.15):
        self.medium = medium
        self.V = V
        self.p_init = p_init
        self.T_init = T_init
        # Pre-compute thermodynamically consistent initial conditions so the
        # t=0 Newton solve starts on (or very near) the algebraic manifold.
        self.h_init = float(medium.eval_h_pT(p_init, T_init))
        self.rho_init = float(medium.eval_rho_ph(p_init, self.h_init))
        self.m_init = self.rho_init * V
        self.U_init = self.m_init * self.h_init - p_init * V  # u = h - p/rho, m/rho = V
        super().__init__()

    def declare_components(self):
        self.add_component('V', Parameter(self.V, "m^3"))

        # Differential states (auto-attaches `der_m`, `der_U` companions).
        self.add_component('m', DifferentialVariable(self.m_init, "kg"))
        self.add_component('U', DifferentialVariable(self.U_init, "J"))

        # Vessel-average algebraic states.
        self.add_component('p', Variable(self.p_init, "Pa"))
        self.add_component('h', Variable(self.h_init, "J/kg"))

        # Inlet / outlet ports (driven by adjacent components).
        self.add_component('p_in', Variable(self.p_init, "Pa"))
        self.add_component('h_in', Variable(self.h_init, "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('p_out', Variable(self.p_init, "Pa"))
        self.add_component('h_out', Variable(self.h_init, "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))

    def declare_equations(self):
        # Port equalities via union-find: p_in == p_out == p and h_out == h.
        # `h_in` stays free (it is whatever the upstream feeds into the buffer).
        self.add_connection(self['p_in'], self['p'])
        self.add_connection(self['p_out'], self['p'])
        self.add_connection(self['h_out'], self['h'])

        m = self['m'].symbol
        U = self['U'].symbol
        p = self['p'].symbol
        h = self['h'].symbol
        V = self['V'].symbol
        h_in = self['h_in'].symbol
        m_in_dot = self['m_dot_in'].symbol
        m_out_dot = self['m_dot_out'].symbol

        rho = self.medium.rho_ph(p, h)

        # Continuity / energy: net imbalance feeds the differential states.
        eq_mass = self['der_m'].symbol - (m_in_dot - m_out_dot)
        eq_energy = self['der_U'].symbol - (m_in_dot * h_in - m_out_dot * h)

        # Algebraic closure linking (m, U) to (p, h) via the equation of state.
        eq_density = m - rho * V
        eq_energy_state = U - m * h + p * V

        return [eq_mass, eq_energy, eq_density, eq_energy_state]


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
        L_segments = self.L / self.n_segments
        A = np.pi * self.D ** 2 / 4
        P = np.pi * self.D
        dz = (self.z_out - self.z_in) / self.n_segments
        self.add_component('p_in', Variable(101325, "Pa"))
        self.add_component('h_in', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('m_dot_in', Variable(0.0, "kg/s"))
        self.add_component('p_out', Variable(101325, "Pa"))
        self.add_component('h_out', Variable(self.medium.h_pT(101325, 293.15), "J/kg"))
        self.add_component('m_dot_out', Variable(0.0, "kg/s"))

        for i in range(self.n_segments):
            z_in = self.z_in + i * dz
            z_out = self.z_in + (i + 1) * dz
            fr_f = self.get_churchill_f_factor
            q_f = self.get_q_inflow if not self.adiabatic else self.get_q_inflow_adiabatic
            self.add_component(
                f'pipe_segment_{i}',
                TwoPortSegment(self.medium, A, A, P, P, z_in, z_out, L_segments, self.epsilon, fr_f, q_f),
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
        # adjacent segments: each segment owns a pair of closure equations
        # `m_dot = rho * A * w` (one per face), and at an internal interface
        # both equations would reduce to the same statement after `(p, h,
        # m_dot)` are unioned -- collapsing the two `w` symbols into one
        # would leave the system over-determined by one equation per shared
        # face.  Letting `w_out(seg_k)` and `w_in(seg_{k+1})` stay distinct
        # algebraic Variables that independently converge to the same value
        # costs `N - 1` extra unknowns per pipe but keeps the equation count
        # square.
        for i in range(self.n_segments - 1):
            for io in ('p', 'h', 'm_dot'):
                self.add_connection(
                    self[f'pipe_segment_{i}'][f'{io}_out'],
                    self[f'pipe_segment_{i + 1}'][f'{io}_in'],
                )
        last = f'pipe_segment_{self.n_segments - 1}'
        for port in ('p_out', 'h_out', 'm_dot_out'):
            self.add_connection(self[port], self[last][port])
        return []
