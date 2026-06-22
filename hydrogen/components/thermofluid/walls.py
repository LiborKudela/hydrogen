"""Wall heat-transfer components of the `thermofluid` library, built on
`hydrogen.model`.

Module layout:

  1. Boundary conditions -- `FixedTemperature`, `FixedHeatFlow`,
     `ConvectiveBoundary`.  Small single-port models used to drive a
     thermal network for testing and for the worked examples.
  2. Passive elements -- `ThermalConductor`: a massless conductance used
     to wire a prescribed-temperature source onto a capacitive node
     (driving a heat capacity through a conductance is well-posed; wiring
     a temperature straight onto it is a high-index constraint).
  3. Components -- `TwoNodeWall` is the shared base: a wall lumped into two
     surface heat-capacity nodes with conduction between them.  `FlatWall`
     (plane slab) and `CylindricalWall` (hollow tube) subclass it and
     supply only the geometry-specific node capacity and conductance
     (Cartesian vs radial).  `CylindricalWall(leaky=True)` additionally
     permeates a gas radially, exposing two `PermeationPort_pN` surfaces (see
     the sibling `permeation` module for the injected flux models).

The typed connectors (`ThermalPort_TQ`, `PermeationPort_pN`, ...) live in the
sibling `ports` module.

Sign convention -- Modelica "flow into me":
    Every port's `Q_dot` is positive when heat flows INTO the component
    that owns the port through that face.  When two same-orientation
    ports are wired, `Model.connect()` emits a sum-to-zero on the flow
    channel (`Q_dot_a + Q_dot_b == 0`), i.e. the heat leaving one
    component enters the other -- the thermal analogue of the
    Kirchhoff / Modelica connector rule.
"""

from __future__ import annotations

from typing import Annotated

import sympy as sp

from ...model import DifferentialVariable, Model, Parameter, Variable
from ...paramspec import ParamSpec, merged_param_specs
from .ports import PermeationPort_pN, ThermalPort_TQ


# ---------------------------------------------------------------------------
# Boundary conditions (single-port drivers for a thermal network)
# ---------------------------------------------------------------------------


class FixedTemperature(Model):
    """Imposes a fixed temperature at its port; supplies whatever heat is needed.

    A temperature reservoir: it pins its own port temperature to `T_set`
    and leaves `Q_dot` free, so the connected component draws (or dumps)
    exactly the heat required to hold the boundary temperature.

    Port (`heat`):
        T, Q_dot   - `T` is pinned to `T_set`; `Q_dot` is determined by
                     the connected network.

    Equation:
        T_port - T_set == 0
    """

    def __init__(self, T_set: Annotated[float, ParamSpec("Boundary "
                "temperature held at the port.", unit="K")] = 293.15):
        self.T_set = T_set
        super().__init__()

    def declare_components(self):
        self.add_component('T_set', Parameter(self.T_set,
                                              **merged_param_specs(type(self))['T_set'].param_kwargs()))
        self.add_component('T_port', Variable(self.T_set, "K"))
        self.add_component('Q_dot_port', Variable(0.0, "W"))
        self.add_port('heat', ThermalPort_TQ(
            self,
            channels={'T': self['T_port'], 'Q_dot': self['Q_dot_port']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        return [self['T_port'].symbol - self['T_set'].symbol]


class FixedHeatFlow(Model):
    """Imposes a fixed heat flow rate INTO the connected component.

    A heat-rate source: the user-facing `Q_flow` parameter is the heat
    rate delivered to the *connected* component (positive = heating it).
    Under "flow into me", the boundary's own `Q_dot_port` measures heat
    entering THIS boundary, so it is pinned to `-Q_flow`; the
    same-orientation `connect()` sum-to-zero then makes the partner's
    `Q_dot` equal `+Q_flow`.

    Use `Q_flow=0` for a perfectly insulated (adiabatic) surface.

    Port (`heat`):
        T, Q_dot   - `T` is free (floats to whatever the surface reaches);
                     `Q_dot` pinned so the partner receives `Q_flow`.

    Equation:
        Q_dot_port + Q_flow == 0
    """

    def __init__(
        self,
        Q_flow: Annotated[float, ParamSpec("Heat rate delivered into the "
                         "connected component (positive = heating); 0 = "
                         "adiabatic.", unit="W")] = 0.0,
        T_init: Annotated[float, ParamSpec("Initial port temperature guess "
                         "(floats freely).", unit="K")] = 293.15,
    ):
        self.Q_flow = Q_flow
        self.T_init = T_init
        super().__init__()

    def declare_components(self):
        self.add_component('Q_flow', Parameter(self.Q_flow,
                                               **merged_param_specs(type(self))['Q_flow'].param_kwargs()))
        self.add_component('T_port', Variable(self.T_init, "K"))
        self.add_component('Q_dot_port', Variable(-self.Q_flow, "W"))
        self.add_port('heat', ThermalPort_TQ(
            self,
            channels={'T': self['T_port'], 'Q_dot': self['Q_dot_port']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        return [self['Q_dot_port'].symbol + self['Q_flow'].symbol]


class ConvectiveBoundary(Model):
    """Newton-cooling boundary: `Q_into_partner = h * A * (T_inf - T_surface)`.

    Convective exchange with a far-field fluid at `T_inf` through a film
    coefficient `h` over area `A`.  The surface temperature is the port
    temperature (set by the connected component); the heat delivered to
    that component is `h * A * (T_inf - T_port)` (positive when the
    far-field is hotter than the surface).

    Under "flow into me" the boundary's own `Q_dot_port` is the negative
    of the heat delivered to the partner, hence the closure below.

    Port (`heat`):
        T, Q_dot   - `T` follows the surface; `Q_dot` set by the
                     convective law.

    Equation:
        Q_dot_port - h * A * (T_port - T_inf) == 0
    """

    def __init__(
        self,
        h: Annotated[float, ParamSpec("Convective film (heat-transfer) "
                    "coefficient.", unit="W/m^2/K", default=10.0)],
        A: Annotated[float, ParamSpec("Convective exchange area.", unit="m^2",
                    default=1.0)],
        T_inf: Annotated[float, ParamSpec("Far-field fluid temperature.",
                        unit="K")] = 293.15,
    ):
        self.h = h
        self.A = A
        self.T_inf = T_inf
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('h', Parameter(self.h, **spec['h'].param_kwargs()))
        self.add_component('A', Parameter(self.A, **spec['A'].param_kwargs()))
        self.add_component('T_inf', Parameter(self.T_inf, **spec['T_inf'].param_kwargs()))
        self.add_component('T_port', Variable(self.T_inf, "K"))
        self.add_component('Q_dot_port', Variable(0.0, "W"))
        self.add_port('heat', ThermalPort_TQ(
            self,
            channels={'T': self['T_port'], 'Q_dot': self['Q_dot_port']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        h = self['h'].symbol
        A = self['A'].symbol
        T_inf = self['T_inf'].symbol
        T_port = self['T_port'].symbol
        Q_dot_port = self['Q_dot_port'].symbol
        # Q delivered to the partner = h*A*(T_inf - T_port); the boundary's
        # own "into me" Q_dot is the negation of that.
        return [Q_dot_port - h * A * (T_port - T_inf)]


# ---------------------------------------------------------------------------
# Passive elements
# ---------------------------------------------------------------------------


class ThermalConductor(Model):
    """Pure thermal conductance (no storage): `Q = G * (T_a - T_b)`.

    A massless two-port resistor.  Heat entering face ``a`` conducts
    straight through to face ``b`` with conductance `G` `[W/K]`; the
    component stores no energy, so the two face heat flows sum to zero.

    For a slab of conductivity `k`, area `A`, thickness `L` use
    `G = k * A / L`.  For a contact/film resistance `R` use `G = 1 / R`.

    This is the element you put *between* a `FixedTemperature` source and a
    capacitive node (a `FlatWall` surface): a prescribed temperature wired
    straight onto a heat capacity is a high-index constraint, but driving
    the capacity through a conductance is well-posed.

    Ports (`heat_a`, `heat_b`, both `flow_orientation='in'`):
        T, Q_dot   - face temperatures and the heat entering each face.

    Equations:
        Q_dot_a - G * (T_a - T_b) == 0      (conduction law)
        Q_dot_a + Q_dot_b        == 0       (no storage)
    """

    def __init__(
        self,
        G: Annotated[float, ParamSpec("Thermal conductance (G = k*A/L, or 1/R "
                    "for a contact resistance).", unit="W/K", default=1.0)],
        T_init: Annotated[float, ParamSpec("Initial face-temperature guess.",
                         unit="K")] = 293.15,
    ):
        self.G = G
        self.T_init = T_init
        super().__init__()

    def declare_components(self):
        self.add_component('G', Parameter(self.G,
                                          **merged_param_specs(type(self))['G'].param_kwargs()))
        self.add_component('T_a', Variable(self.T_init, "K"))
        self.add_component('T_b', Variable(self.T_init, "K"))
        self.add_component('Q_dot_a', Variable(0.0, "W"))
        self.add_component('Q_dot_b', Variable(0.0, "W"))
        self.add_port('heat_a', ThermalPort_TQ(
            self,
            channels={'T': self['T_a'], 'Q_dot': self['Q_dot_a']},
            flow_orientation='in',
        ))
        self.add_port('heat_b', ThermalPort_TQ(
            self,
            channels={'T': self['T_b'], 'Q_dot': self['Q_dot_b']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        G = self['G'].symbol
        T_a = self['T_a'].symbol
        T_b = self['T_b'].symbol
        Q_dot_a = self['Q_dot_a'].symbol
        Q_dot_b = self['Q_dot_b'].symbol
        eq_law = Q_dot_a - G * (T_a - T_b)
        eq_balance = Q_dot_a + Q_dot_b
        return [eq_law, eq_balance]


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class TwoNodeWall(Model):
    """Base class for a wall lumped into two surface nodes with conduction
    between them.

    Captures everything the flat and cylindrical walls share -- the two
    surface temperatures, the two heat ports, and the first-law energy
    balance at each node.  The `dynamic` flag selects between two modes:

      * `dynamic=True` (default) -- each node carries thermal mass, so the
        wall stores energy and the surface temperatures are DIFFERENTIAL
        states (transient / heat-up behaviour)::

            port_a  ->  [ C_node | T_a ]== G ==[ T_b | C_node ]  <-  port_b
                           surface A                  surface B

            C_node * dT_a/dt = Q_dot_a - G * (T_a - T_b)
            C_node * dT_b/dt = Q_dot_b - G * (T_b - T_a)

      * `dynamic=False` -- quasi-static: the node capacities and their ODEs
        are removed, so the wall is a pure (massless) conductance and the
        surface temperatures are ALGEBRAIC states (instantaneous steady
        conduction)::

            port_a  ->  [ T_a ]== G ==[ T_b ]  <-  port_b

            0 = Q_dot_a - G * (T_a - T_b)        (=> Q_dot_a + Q_dot_b = 0)
            0 = Q_dot_b - G * (T_b - T_a)

        In this mode a prescribed temperature (`FixedTemperature`) may be
        wired straight onto a face -- there is no capacity, so it is not a
        high-index constraint (unlike the dynamic mode, which needs a
        `ThermalConductor` between a temperature source and the node).

    Surface temperatures (differential if `dynamic`, else algebraic):
        T_a, T_b  - the surface-node temperatures            [K]

    Algebraic states (port heat flows, set by the connected network):
        Q_dot_a, Q_dot_b  - heat flow INTO each surface      [W]

    Ports (both `flow_orientation='in'`, positive `Q_dot` = heat into the
    wall):
        port_a   - T = T_a, Q_dot = Q_dot_a
        port_b   - T = T_b, Q_dot = Q_dot_b

    Only two things vary between geometries, so subclasses provide exactly
    these (everything else lives here):

      * `_declare_geometry()`  - register the geometry `Parameter`s the
        capacity/conductance expressions need (the shared material
        parameters `rho`, `cp`, `k` are already declared by the base).
      * `_node_capacity()`     - return the symbolic per-node heat
        capacity `C_node` `[J/K]` (only used when `dynamic=True`).
      * `_conductance()`       - return the symbolic node-to-node
        conductance `G` `[W/K]`.

    Notes / simplifications:
      * Two-node lumped model: captures the first internal mode (a
        gradient through the wall) but not the full continuous profile.
      * Material properties are constant; conduction is one-dimensional.
      * All material/geometry inputs are parameters: pass plain scalars
        (a fresh local `Parameter` is created) or an existing
        `Parameter`/`ParameterAlias` to share a parent's symbol.
    """

    #: Single-source metadata for the shared material params (see
    #: `hydrogen.paramspec`): read both by the component catalog and by
    #: `declare_components` below, so the units live in exactly one place.
    #: These args have no shared `__init__` to annotate (subclasses define
    #: their own constructor), so the specs stay here.  `dynamic` is marked
    #: ``structural`` so `cache_key_flag_names()` keys the equation-template
    #: cache on it (it toggles the ODE vs algebraic form).
    PARAMS = {
        "rho": ParamSpec("Density of the wall material.", unit="kg/m^3"),
        "cp": ParamSpec("Specific heat capacity of the wall material.",
                        unit="J/(kg*K)"),
        "k": ParamSpec("Thermal conductivity of the wall material.",
                       unit="W/(m*K)"),
        "T_init": ParamSpec("Initial wall temperature.", unit="K"),
        "dynamic": ParamSpec(
            "If true the wall has thermal mass (heat-up transient); if false "
            "it conducts quasi-statically with no capacity.", structural=True),
    }

    def declare_components(self):
        # Shared material parameters; subclasses add geometry on top.  Units /
        # descriptions are pulled straight from `PARAMS` so they are authored
        # once (here) and reused by the catalog without instantiating.
        spec = merged_param_specs(type(self))
        self.add_component('rho', Parameter(self.rho, **spec['rho'].param_kwargs()))
        self.add_component('cp', Parameter(self.cp, **spec['cp'].param_kwargs()))
        self.add_component('k', Parameter(self.k, **spec['k'].param_kwargs()))
        self._declare_geometry()

        # Surface-node temperatures.  Dynamic -> DifferentialVariable (carries
        # thermal mass, auto-attaches der_T_a/der_T_b); quasi-static -> plain
        # algebraic Variable (no capacity, no ODE).
        TNode = DifferentialVariable if self._is_dynamic() else Variable
        self.add_component('T_a', TNode(self.T_init, "K"))
        self.add_component('T_b', TNode(self.T_init, "K"))

        # Port heat flows (algebraic; determined by the connected network).
        self.add_component('Q_dot_a', Variable(0.0, "W"))
        self.add_component('Q_dot_b', Variable(0.0, "W"))

        self.add_port('port_a', ThermalPort_TQ(
            self,
            channels={'T': self['T_a'], 'Q_dot': self['Q_dot_a']},
            flow_orientation='in',
        ))
        self.add_port('port_b', ThermalPort_TQ(
            self,
            channels={'T': self['T_b'], 'Q_dot': self['Q_dot_b']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        T_a = self['T_a'].symbol
        T_b = self['T_b'].symbol
        Q_dot_a = self['Q_dot_a'].symbol
        Q_dot_b = self['Q_dot_b'].symbol

        G = self._conductance()

        if not self._is_dynamic():
            # Quasi-static: massless conductance, algebraic node temperatures.
            eq_a = Q_dot_a - G * (T_a - T_b)
            eq_b = Q_dot_b - G * (T_b - T_a)
            return [eq_a, eq_b]

        # Dynamic: first-law energy balance at each surface node with thermal
        # mass.  Conduction is positive from the hotter node to the colder one.
        der_T_a = self['der_T_a'].symbol
        der_T_b = self['der_T_b'].symbol
        C_node = self._node_capacity()
        eq_a = C_node * der_T_a - (Q_dot_a - G * (T_a - T_b))
        eq_b = C_node * der_T_b - (Q_dot_b - G * (T_b - T_a))
        return [eq_a, eq_b]

    def _is_dynamic(self):
        """Whether this wall carries thermal mass (default True)."""
        return getattr(self, 'dynamic', True)

    # --- geometry hooks (subclass responsibility) -------------------------

    def _declare_geometry(self):
        """Register the geometry `Parameter`s for this wall shape."""
        raise NotImplementedError

    def _node_capacity(self):
        """Return the symbolic per-node heat capacity `C_node` [J/K]."""
        raise NotImplementedError

    def _conductance(self):
        """Return the symbolic node-to-node conductance `G` [W/K]."""
        raise NotImplementedError


class FlatWall(TwoNodeWall):
    """Plane wall as two surface heat capacities with conduction between them.

    A slab of area `A` and thickness `L` made of a material with density
    `rho`, specific heat `cp`, and thermal conductivity `k`, lumped into
    one node at each surface (see `TwoNodeWall` for the shared states,
    ports, and energy balance):

        C_node = rho * cp * A * L / 2          [J/K]   (per surface node)
        G      = k * A / L                     [W/K]   (node-to-node)

    `rho`, `cp`, `k`, `A`, `L` are all parameters.  `dynamic` toggles between
    the transient (capacitive) and quasi-static (massless) modes -- see
    `TwoNodeWall`.
    """

    def __init__(
        self, rho, cp, k,
        A: Annotated[float, ParamSpec("Heat-transfer (conduction) area of the "
                    "slab.", unit="m^2")],
        L: Annotated[float, ParamSpec("Slab thickness (the conduction "
                    "length).", unit="m")],
        T_init=293.15, dynamic=True,
    ):
        self.rho = rho
        self.cp = cp
        self.k = k
        self.A = A
        self.L = L
        self.T_init = T_init
        self.dynamic = dynamic
        super().__init__()

    def _declare_geometry(self):
        spec = merged_param_specs(type(self))
        self.add_component('A', Parameter(self.A, **spec['A'].param_kwargs()))
        self.add_component('L', Parameter(self.L, **spec['L'].param_kwargs()))

    def _node_capacity(self):
        # Half the slab's thermal mass per surface node.
        return self['rho'].symbol * self['cp'].symbol * self['A'].symbol * self['L'].symbol / 2

    def _conductance(self):
        # Plane-wall conduction across the full thickness.
        return self['k'].symbol * self['A'].symbol / self['L'].symbol


class CylindricalWall(TwoNodeWall):
    """Hollow-cylinder (tube) wall as two surface heat capacities with radial
    conduction between them.  The circular-geometry counterpart of `FlatWall`.

    An annular wall of length `length` between inner radius `r_in` and outer
    radius `r_out`, made of a material with density `rho`, specific heat
    `cp`, and thermal conductivity `k`, lumped into one node at each
    cylindrical surface (`port_a` inner, `port_b` outer).  See
    `TwoNodeWall` for the shared states, ports, and energy balance; only
    the geometry-derived terms differ:

        V      = pi * (r_out**2 - r_in**2) * length     [m^3]
        C_node = rho * cp * V / 2                         [J/K]   (per node)
        G      = 2 * pi * k * length / ln(r_out / r_in)  [W/K]   (node-to-node)

    The `2*pi*k*length/ln(r_out/r_in)` term is the exact steady radial
    conductance of a cylindrical shell -- the analogue of `k*A/L` for a
    flat slab.

    `rho`, `cp`, `k`, `r_in`, `r_out`, `length` are all parameters.
    Requires `r_out > r_in > 0`.  `dynamic` toggles between the transient
    (capacitive) and quasi-static (massless) modes -- see `TwoNodeWall`.

    Gas permeation (`leaky=True`)
    -----------------------------
    Set `leaky=True` and inject a `permeation_flux` strategy to make the wall
    ALSO conduct a gas radially, exposing two `PermeationPort_pN`s -- one per
    surface, mirroring the two thermal ports:

        leak_a  - inner surface: p_partial = p_partial_a, m_dot = m_dot_a_leak (into wall)
        leak_b  - outer surface: p_partial = p_partial_b, m_dot = m_dot_b_leak (into wall)

    The wall stays permeation-physics-agnostic: it owns the ports, the two
    surface partial-pressure variables, and the two leak mass-flows, but the
    pressure-gradient -> mass-flow CORRELATION is supplied by the injected
    `permeation_flux` object (e.g. a steady Richardson flux or a transient
    diffusion chain from the `permeation` module).  That object carries the
    permeant + `TransportFit` and must provide:

        * `cache_key`        - a hashable identity for its equation structure
                               (keyed into the wall's `_cache_key_flags` so the
                               equation cache stays correct across flux models).
        * `declare(wall)`    - register any extra components it needs (Arrhenius
                               `Parameter`s, transient state vars).
        * `equations(wall)`  - return the residual equations, binding
                               `m_dot_a_leak` / `m_dot_b_leak` to the computed
                               fluxes (reads geometry, `T_a`/`T_b`, `p_partial_a`/
                               `p_partial_b` off the wall).

    `p_in_init` / `p_out_init` seed the two surface partial pressures (only used
    when `leaky=True`).
    """

    #: Adds the permeation toggle + the injected flux model's structural identity
    #: to the thermal `dynamic` key, so a model mixing plain / leaky / steady /
    #: transient walls caches each equation variant correctly.
    #: `dynamic` (inherited, structural) and `leaky` (structural, below) are
    #: derived automatically; `_perm_key` is the computed structural identity
    #: of the injected flux model and is not a constructor argument.
    _cache_key_flags = ("_perm_key",)

    #: The injected flux model can't carry a scalar type annotation, so its
    #: metadata stays here (merged with the Annotated specs below).
    PARAMS = {
        "permeation_flux": ParamSpec(
            "Permeation flux model (SteadyRichardson / TransientDiffusion) "
            "supplying the pressure-gradient -> mass-flow law; required when "
            "leaky.", relevant_when={"leaky": True}),
    }

    def __init__(
        self, rho, cp, k,
        r_in: Annotated[float, ParamSpec("Inner radius of the tube wall (bore "
                       "side).", unit="m")],
        r_out: Annotated[float, ParamSpec("Outer radius of the tube wall.",
                        unit="m")],
        length: Annotated[float, ParamSpec("Axial length of the wall "
                         "segment.", unit="m")],
        T_init=293.15,
        dynamic=True,
        leaky: Annotated[bool, ParamSpec("Enable radial gas permeation "
                        "(requires a `permeation_flux`).",
                        structural=True)] = False,
        permeation_flux=None,
        p_in_init: Annotated[float, ParamSpec("Initial inner-surface partial "
                            "pressure (only used when leaky).", unit="Pa")]
                   = 101325.0,
        p_out_init: Annotated[float, ParamSpec("Initial outer-surface partial "
                             "pressure (only used when leaky).", unit="Pa")]
                    = 101325.0,
    ):
        if not (r_out > r_in > 0):
            raise ValueError(
                f"CylindricalWall requires r_out > r_in > 0; got "
                f"r_in={r_in}, r_out={r_out}"
            )
        if leaky and permeation_flux is None:
            raise ValueError(
                "CylindricalWall(leaky=True) requires a `permeation_flux` model "
                "(e.g. SteadyRichardson(material) or TransientDiffusion(material))."
            )
        self.rho = rho
        self.cp = cp
        self.k = k
        self.r_in = r_in
        self.r_out = r_out
        self.length = length
        self.T_init = T_init
        self.dynamic = dynamic
        self.leaky = bool(leaky)
        self.permeation_flux = permeation_flux if self.leaky else None
        self.p_in_init = p_in_init
        self.p_out_init = p_out_init
        # Structural cache-key contribution from the injected flux model (its
        # equations change the system, so two walls with different flux models
        # must not replay each other's cached template).  `None` when not leaky.
        self._perm_key = self.permeation_flux.cache_key if self.leaky else None
        super().__init__()

    def declare_components(self):
        super().declare_components()  # thermal: material, geometry, T_a/T_b, ports
        if not self.leaky:
            return
        # Two surface H2 partial pressures + two leak mass-flows (into the wall),
        # one per surface.  The flux model reads/binds these.
        self.add_component('p_partial_a', Variable(self.p_in_init, "Pa"))
        self.add_component('p_partial_b', Variable(self.p_out_init, "Pa"))
        self.add_component('m_dot_a_leak', Variable(0.0, "kg/s"))
        self.add_component('m_dot_b_leak', Variable(0.0, "kg/s"))
        # Let the injected correlation register its own parameters / states.
        self.permeation_flux.declare(self)
        self.add_port('leak_a', PermeationPort_pN(
            self,
            channels={'p_partial': self['p_partial_a'], 'm_dot_leak': self['m_dot_a_leak']},
            flow_orientation='in',
            require_connection=True,
        ))
        self.add_port('leak_b', PermeationPort_pN(
            self,
            channels={'p_partial': self['p_partial_b'], 'm_dot_leak': self['m_dot_b_leak']},
            flow_orientation='in',
            require_connection=True,
        ))

    def declare_equations(self):
        eqs = list(super().declare_equations())
        if self.leaky:
            eqs += list(self.permeation_flux.equations(self))
        return eqs

    def _declare_geometry(self):
        spec = merged_param_specs(type(self))
        self.add_component('r_in', Parameter(self.r_in, **spec['r_in'].param_kwargs()))
        self.add_component('r_out', Parameter(self.r_out, **spec['r_out'].param_kwargs()))
        self.add_component('length', Parameter(self.length, **spec['length'].param_kwargs()))

    def _node_capacity(self):
        # Annular thermal mass split across the two surface nodes.
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        length = self['length'].symbol
        V = sp.pi * (r_out ** 2 - r_in ** 2) * length
        return self['rho'].symbol * self['cp'].symbol * V / 2

    def _conductance(self):
        # Exact radial conduction conductance of a hollow cylinder.
        k = self['k'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        length = self['length'].symbol
        return 2 * sp.pi * k * length / sp.log(r_out / r_in)
