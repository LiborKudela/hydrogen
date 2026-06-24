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
     surface heat-capacity nodes with conduction between them, plus the
     (optional) gas-permeation plumbing.  `FlatWall` (plane slab),
     `CylindricalWall` (hollow tube) and `SphericalWall` (hollow shell)
     subclass it and supply only the geometry-specific node capacity,
     conductance, and -- when permeation is used -- the geometric permeation
     conductance and finite-volume shell layout (Cartesian / radial /
     spherical).  Setting `leaky=True` on ANY of them makes the wall also
     permeate a gas through its thickness, exposing two `PermeationPort_pN`
     surfaces (see the sibling `permeation` module for the injected,
     geometry-agnostic flux models).

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
        count: Annotated[float, ParamSpec("Number of identical parallel "
                        "surfaces this boundary represents (multiplicity >= 1); "
                        "scales the exchange area, hence the heat, by N.  A live "
                        "Parameter (not structural).", unit="1",
                        default=1.0)] = 1.0,
    ):
        self.h = h
        self.A = A
        self.T_inf = T_inf
        self.count = count
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('h', Parameter(self.h, **spec['h'].param_kwargs()))
        self.add_component('A', Parameter(self.A, **spec['A'].param_kwargs()))
        self.add_component('count', Parameter(self.count, **spec['count'].param_kwargs()))
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
        count = self['count'].symbol
        T_inf = self['T_inf'].symbol
        T_port = self['T_port'].symbol
        Q_dot_port = self['Q_dot_port'].symbol
        # Q delivered to the partner = h*(count*A)*(T_inf - T_port); the
        # boundary's own "into me" Q_dot is the negation of that.  `count`
        # scales the area so N parallel surfaces exchange N times the heat.
        return [Q_dot_port - h * count * A * (T_port - T_inf)]


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
        count: Annotated[float, ParamSpec("Number of identical parallel "
                        "conductors this one represents (multiplicity >= 1); "
                        "scales the conductance by N.  A live Parameter (not "
                        "structural).", unit="1", default=1.0)] = 1.0,
    ):
        self.G = G
        self.T_init = T_init
        self.count = count
        super().__init__()

    def declare_components(self):
        spec = merged_param_specs(type(self))
        self.add_component('G', Parameter(self.G, **spec['G'].param_kwargs()))
        self.add_component('count', Parameter(self.count, **spec['count'].param_kwargs()))
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
        count = self['count'].symbol
        T_a = self['T_a'].symbol
        T_b = self['T_b'].symbol
        Q_dot_a = self['Q_dot_a'].symbol
        Q_dot_b = self['Q_dot_b'].symbol
        # `count` parallel conductors -> total conductance is N*G.
        eq_law = Q_dot_a - count * G * (T_a - T_b)
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

    Gas permeation (`leaky=True`)
    -----------------------------
    Set `leaky=True` and inject a `permeation_flux` strategy to make the wall
    ALSO conduct a gas through its thickness, exposing two `PermeationPort_pN`s
    -- one per surface, mirroring the two thermal ports:

        leak_a  - inner surface: p_partial = p_partial_a, m_dot = m_dot_a_leak (into wall)
        leak_b  - outer surface: p_partial = p_partial_b, m_dot = m_dot_b_leak (into wall)

    The wall stays permeation-physics-agnostic: it owns the ports, the two
    surface partial-pressure variables, and the two leak mass-flows, but the
    pressure-gradient -> mass-flow CORRELATION is supplied by the injected
    `permeation_flux` object (e.g. a steady Richardson flux or a transient
    diffusion chain from the `permeation` module).  That object is itself
    geometry-agnostic: it reads the wall's surface temperatures / partial
    pressures and asks the wall for the two shape-dependent permeation terms
    (`_perm_geom_conductance()` and `_perm_shell_volumes(n)`), so the same flux
    model drives a flat, cylindrical or spherical wall unchanged.

    `p_in_init` / `p_out_init` seed the two surface partial pressures (only used
    when `leaky=True`).

    Geometry hooks (subclass responsibility)
    ----------------------------------------
    Only the shape-specific terms vary between geometries, so subclasses provide
    exactly these (everything else lives here):

      * `_declare_geometry()`        - register the geometry `Parameter`s the
        capacity/conductance expressions need (the shared material
        parameters `rho`, `cp`, `k` are already declared by the base).
      * `_node_capacity()`           - return the symbolic per-node heat
        capacity `C_node` `[J/K]` (only used when `dynamic=True`).
      * `_conductance()`             - return the symbolic node-to-node thermal
        conductance `G` `[W/K]`.
      * `_perm_geom_conductance()`   - return the symbolic geometric permeation
        conductance `K` `[m]` such that the steady molar leak is
        `Phi(T) * K * (c_a - c_b)` (only needed when `leaky=True`).
      * `_perm_shell_volumes(n)`     - return the `n` symbolic finite-volume
        shell volumes `[m^3]` for the transient diffusion chain, laid out on
        the shape's equal-conductance node spacing (only needed when `leaky=True`
        with a `TransientDiffusion` flux).

    Notes / simplifications:
      * Two-node lumped model: captures the first internal mode (a
        gradient through the wall) but not the full continuous profile.
      * Material properties are constant; conduction is one-dimensional.
      * All material/geometry inputs are parameters: pass plain scalars
        (a fresh local `Parameter` is created) or an existing
        `Parameter`/`ParameterAlias` to share a parent's symbol.
    """

    #: Single-source metadata for the shared material + permeation params (see
    #: `hydrogen.paramspec`): read both by the component catalog and by
    #: `declare_components` below, so the units live in exactly one place.
    #: These args have no shared `__init__` to annotate (subclasses define
    #: their own constructor), so the specs stay here.  `dynamic` and `leaky`
    #: are marked ``structural`` so `cache_key_flag_names()` keys the
    #: equation-template cache on them (`dynamic` toggles the ODE vs algebraic
    #: thermal form; `leaky` toggles the whole permeation equation set).
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
        "leaky": ParamSpec(
            "Enable gas permeation through the wall thickness (requires a "
            "`permeation_flux`).", structural=True),
        "permeation_flux": ParamSpec(
            "Permeation flux model (SteadyRichardson / TransientDiffusion) "
            "supplying the pressure-gradient -> mass-flow law; required when "
            "leaky.", relevant_when={"leaky": True}),
        "p_in_init": ParamSpec("Initial inner-surface partial pressure (only "
                               "used when leaky).", unit="Pa",
                               relevant_when={"leaky": True}),
        "p_out_init": ParamSpec("Initial outer-surface partial pressure (only "
                                "used when leaky).", unit="Pa",
                                relevant_when={"leaky": True}),
    }

    #: The injected flux model's structural identity contributes to the
    #: equation-template cache key (alongside the derived structural `dynamic` /
    #: `leaky` flags), so a model mixing plain / leaky / steady / transient
    #: walls caches each equation variant correctly.  `_perm_key` is the
    #: computed identity of the injected flux model (set in `_setup_permeation`),
    #: not a constructor argument, so it is listed explicitly here.
    _cache_key_flags = ("_perm_key",)

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

        # Gas permeation (optional): two surface partial pressures + two leak
        # mass-flows + two permeation ports, mirroring the thermal pair.  The
        # geometry-agnostic flux model registers any extra components it needs.
        self._declare_permeation()

    def _declare_permeation(self):
        """Register the permeation ports/states when `leaky=True` (no-op
        otherwise).  Shared by every wall shape; the injected `permeation_flux`
        adds its own parameters/states and the geometry is supplied by the
        subclass `_perm_*` hooks."""
        if not self._is_leaky():
            return
        # Two surface partial pressures + two leak mass-flows (into the wall),
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
        T_a = self['T_a'].symbol
        T_b = self['T_b'].symbol
        Q_dot_a = self['Q_dot_a'].symbol
        Q_dot_b = self['Q_dot_b'].symbol

        G = self._conductance()

        if not self._is_dynamic():
            # Quasi-static: massless conductance, algebraic node temperatures.
            eqs = [Q_dot_a - G * (T_a - T_b),
                   Q_dot_b - G * (T_b - T_a)]
        else:
            # Dynamic: first-law energy balance at each surface node with
            # thermal mass.  Conduction is positive hotter node -> colder one.
            der_T_a = self['der_T_a'].symbol
            der_T_b = self['der_T_b'].symbol
            C_node = self._node_capacity()
            eqs = [C_node * der_T_a - (Q_dot_a - G * (T_a - T_b)),
                   C_node * der_T_b - (Q_dot_b - G * (T_b - T_a))]

        # Permeation residuals (bind m_dot_*_leak to the injected flux model),
        # geometry-agnostic: the flux model asks the wall for its shape terms.
        if self._is_leaky():
            eqs += list(self.permeation_flux.equations(self))
        return eqs

    def _is_dynamic(self):
        """Whether this wall carries thermal mass (default True)."""
        return getattr(self, 'dynamic', True)

    def _is_leaky(self):
        """Whether this wall also permeates a gas (default False)."""
        return getattr(self, 'leaky', False)

    def _setup_permeation(self, leaky, permeation_flux, p_in_init, p_out_init):
        """Validate + record the permeation settings; call from a subclass
        ``__init__`` (before ``super().__init__()``).

        Sets ``leaky`` / ``permeation_flux`` / ``p_in_init`` / ``p_out_init``
        and the computed ``_perm_key`` (the flux model's structural cache-key
        contribution; ``None`` when not leaky).
        """
        leaky = bool(leaky)
        if leaky and permeation_flux is None:
            raise ValueError(
                f"{type(self).__name__}(leaky=True) requires a `permeation_flux` "
                "model (e.g. SteadyRichardson(material) or "
                "TransientDiffusion(material))."
            )
        self.leaky = leaky
        self.permeation_flux = permeation_flux if leaky else None
        self.p_in_init = p_in_init
        self.p_out_init = p_out_init
        # Structural cache-key contribution from the injected flux model (its
        # equations change the system, so two walls with different flux models
        # must not replay each other's cached template).  `None` when not leaky.
        self._perm_key = self.permeation_flux.cache_key if leaky else None

    # --- thermal geometry hooks (subclass responsibility) -----------------

    def _declare_geometry(self):
        """Register the geometry `Parameter`s for this wall shape."""
        raise NotImplementedError

    def _node_capacity(self):
        """Return the symbolic per-node heat capacity `C_node` [J/K]."""
        raise NotImplementedError

    def _conductance(self):
        """Return the symbolic node-to-node conductance `G` [W/K]."""
        raise NotImplementedError

    # --- permeation geometry hooks (subclass responsibility when leaky) ----

    def _perm_geom_conductance(self):
        """Return the symbolic geometric permeation conductance `K` [m] such
        that the steady molar leak is `Phi(T) * K * (c_a - c_b)` and the
        finite-volume chain's uniform conductance is `(n+1) * D(T) * K`."""
        raise NotImplementedError

    def _perm_shell_volumes(self, n):
        """Return the `n` symbolic finite-volume shell volumes [m^3] for the
        transient diffusion chain, on this shape's equal-conductance node
        spacing (node `j` = 1..n)."""
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

    Gas permeation (`leaky=True`).  See `TwoNodeWall`; the plane-slab geometry
    terms are::

        K      = A / L                         [m]    (permeation conductance)
        V_j    = A * L / (n + 1)               [m^3]  (uniform FV shell volume)

    so the steady molar leak is `Phi(T) * A / L * (c_a - c_b)` -- the classic
    `Phi/t * A * (p_in**(1/n) - p_out**(1/n))` slab law.
    """

    def __init__(
        self, rho, cp, k,
        A: Annotated[float, ParamSpec("Heat-transfer (conduction) area of the "
                    "slab.", unit="m^2")],
        L: Annotated[float, ParamSpec("Slab thickness (the conduction "
                    "length).", unit="m")],
        T_init=293.15, dynamic=True,
        leaky=False, permeation_flux=None,
        p_in_init=101325.0, p_out_init=101325.0,
    ):
        self.rho = rho
        self.cp = cp
        self.k = k
        self.A = A
        self.L = L
        self.T_init = T_init
        self.dynamic = dynamic
        self._setup_permeation(leaky, permeation_flux, p_in_init, p_out_init)
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

    def _perm_geom_conductance(self):
        # Plane-slab permeation conductance: N_dot = Phi * (A/L) * (c_a - c_b).
        return self['A'].symbol / self['L'].symbol

    def _perm_shell_volumes(self, n):
        # Equal-thickness finite volumes: n interior nodes evenly spaced across
        # the slab (the two boundary half-cells carry no storage), so each cell
        # is A * Delta with Delta = L/(n+1).
        A = self['A'].symbol
        L = self['L'].symbol
        return [A * L / (n + 1) for _ in range(1, n + 1)]


class CylindricalWall(TwoNodeWall):
    """Hollow-cylinder (tube) wall as two surface heat capacities with radial
    conduction between them.  The circular-geometry counterpart of `FlatWall`.

    An annular wall of length `length` between inner radius `r_in` and outer
    radius `r_out`, made of a material with density `rho`, specific heat
    `cp`, and thermal conductivity `k`, lumped into one node at each
    cylindrical surface (`port_a` inner, `port_b` outer).  See
    `TwoNodeWall` for the shared states, ports, and energy balance; only
    the geometry-derived terms differ:

        V      = f * pi * (r_out**2 - r_in**2) * length     [m^3]
        C_node = rho * cp * V / 2                            [J/K]   (per node)
        G      = f * 2 * pi * k * length / ln(r_out / r_in) [W/K]   (node-to-node)

    The `2*pi*k*length/ln(r_out/r_in)` term is the exact steady radial
    conductance of a (full) cylindrical shell -- the analogue of `k*A/L` for a
    flat slab.

    `angle_fraction` (`f`) is the fraction of the full 2*pi tube the wall sweeps
    (1 = full tube, 0.5 = half tube, 0.25 = quarter, ...).  It multiplies every
    extensive term -- mass / heat capacity, conductance, and (when leaky) the
    permeation conductance and shell volumes -- so a partial sector behaves like
    the matching slice of the full tube.

    `count` (`N >= 1`) is a multiplicity: the single component stands in for `N`
    identical parallel tubes.  It multiplies the SAME extensive terms as
    `angle_fraction` (the effective factor is `f = angle_fraction * count`), so
    `N` identical tubes simulate with one set of equations.  Because both
    capacity and conductance scale by `N`, every time constant is unchanged and
    the intensive states (temperatures, partial pressures) match a single tube
    exactly -- only the absolute heat / leak flows scale by `N`.

    `rho`, `cp`, `k`, `r_in`, `r_out`, `length`, `angle_fraction`, `count` are
    all parameters.  Requires `r_out > r_in > 0`, `0 < angle_fraction <= 1`, and
    `count >= 1`.
    `dynamic` toggles between the transient (capacitive) and quasi-static
    (massless) modes -- see `TwoNodeWall`.

    Gas permeation (`leaky=True`).  See `TwoNodeWall` for the shared plumbing
    (ports, surface states, injected flux model); the radial geometry terms
    this wall supplies are::

        K   = f * 2*pi*length / ln(r_out/r_in)               [m]   (permeation conductance)
        V_j = f * pi*(r(j+0.5)**2 - r(j-0.5)**2)*length      [m^3] (equal-ln FV shells)

    so the steady molar leak is `f * 2*pi*Phi(T)*length/ln(r_out/r_in)
    * (p_in**(1/n) - p_out**(1/n))` -- the exact radial Richardson rate, scaled
    by the swept fraction.
    """

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
        angle_fraction: Annotated[float, ParamSpec("Fraction of the full 2*pi "
                       "tube the wall sweeps (1 = full tube, 0.5 = half, ...); "
                       "scales all extensive terms (mass, conductance, leak).",
                       unit="1")] = 1.0,
        count: Annotated[float, ParamSpec("Number of identical parallel tubes "
                       "this one component represents (multiplicity >= 1); "
                       "scales every extensive term so N tubes simulate as one "
                       "without extra equations.", unit="1")] = 1.0,
        leaky=False,
        permeation_flux=None,
        p_in_init=101325.0,
        p_out_init=101325.0,
    ):
        if not (r_out > r_in > 0):
            raise ValueError(
                f"CylindricalWall requires r_out > r_in > 0; got "
                f"r_in={r_in}, r_out={r_out}"
            )
        if not (0 < angle_fraction <= 1):
            raise ValueError(
                f"CylindricalWall requires 0 < angle_fraction <= 1; got "
                f"angle_fraction={angle_fraction}"
            )
        if not (getattr(count, 'value', count) >= 1):
            raise ValueError(
                f"CylindricalWall requires count >= 1; got count={count}")
        self.rho = rho
        self.cp = cp
        self.k = k
        self.r_in = r_in
        self.r_out = r_out
        self.length = length
        self.angle_fraction = angle_fraction
        self.count = count
        self.T_init = T_init
        self.dynamic = dynamic
        self._setup_permeation(leaky, permeation_flux, p_in_init, p_out_init)
        super().__init__()

    def _declare_geometry(self):
        spec = merged_param_specs(type(self))
        self.add_component('r_in', Parameter(self.r_in, **spec['r_in'].param_kwargs()))
        self.add_component('r_out', Parameter(self.r_out, **spec['r_out'].param_kwargs()))
        self.add_component('length', Parameter(self.length, **spec['length'].param_kwargs()))
        self.add_component('angle_fraction', Parameter(
            self.angle_fraction, **spec['angle_fraction'].param_kwargs()))
        self.add_component('count', Parameter(
            self.count, **spec['count'].param_kwargs()))

    def _node_capacity(self):
        # Annular thermal mass split across the two surface nodes (sector f).
        f = self['angle_fraction'].symbol * self['count'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        length = self['length'].symbol
        V = f * sp.pi * (r_out ** 2 - r_in ** 2) * length
        return self['rho'].symbol * self['cp'].symbol * V / 2

    def _conductance(self):
        # Exact radial conduction conductance of a hollow cylinder sector.
        f = self['angle_fraction'].symbol * self['count'].symbol
        k = self['k'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        length = self['length'].symbol
        return f * 2 * sp.pi * k * length / sp.log(r_out / r_in)

    def _perm_geom_conductance(self):
        # Exact radial permeation conductance of a hollow cylinder sector
        # (analogue of the thermal G with k -> Phi): N_dot = Phi * K * (c_a - c_b).
        f = self['angle_fraction'].symbol * self['count'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        length = self['length'].symbol
        return f * 2 * sp.pi * length / sp.log(r_out / r_in)

    def _perm_shell_volumes(self, n):
        # Equal-ln radius spacing makes the n+1 series conductances equal (so
        # they telescope to the exact Richardson resistance).  Shell j spans
        # the radii at fractional indices j-0.5 .. j+0.5, scaled by the sector f.
        f = self['angle_fraction'].symbol * self['count'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        length = self['length'].symbol
        Delta = sp.log(r_out / r_in) / (n + 1)

        def r_at(x):
            return r_in * sp.exp(x * Delta)

        return [f * sp.pi * (r_at(j + 0.5) ** 2 - r_at(j - 0.5) ** 2) * length
                for j in range(1, n + 1)]


class SphericalWall(TwoNodeWall):
    """Hollow-sphere (shell) wall as two surface heat capacities with radial
    conduction between them.  The spherical counterpart of `CylindricalWall`.

    A spherical shell between inner radius `r_in` and outer radius `r_out`,
    made of a material with density `rho`, specific heat `cp`, and thermal
    conductivity `k`, lumped into one node at each surface (`port_a` inner,
    `port_b` outer).  See `TwoNodeWall` for the shared states, ports, and
    energy balance; only the geometry-derived terms differ:

        V      = f * 4/3 * pi * (r_out**3 - r_in**3)           [m^3]
        C_node = rho * cp * V / 2                              [J/K]   (per node)
        G      = f * 4 * pi * k * r_in * r_out / (r_out - r_in) [W/K]  (node-to-node)

    The `4*pi*k*r_in*r_out/(r_out-r_in)` term is the exact steady radial
    conductance of a (full) spherical shell -- the analogue of `k*A/L` (flat)
    and `2*pi*k*length/ln(r_out/r_in)` (cylinder).

    `angle_fraction` (`f`) is the fraction of the full sphere (4*pi solid angle)
    the wall covers (1 = full shell, 0.5 = hemisphere, 0.25 = quarter, ...).  It
    multiplies every extensive term -- mass / heat capacity, conductance, and
    (when leaky) the permeation conductance and shell volumes -- so a partial
    cap behaves like the matching slice of the full shell.

    `count` (`N >= 1`) is a multiplicity: the single component stands in for `N`
    identical parallel shells.  It multiplies the SAME extensive terms as
    `angle_fraction` (the effective factor is `f = angle_fraction * count`), so
    `N` identical shells simulate with one set of equations and unchanged time
    constants -- only the absolute heat / leak flows scale by `N`.

    `rho`, `cp`, `k`, `r_in`, `r_out`, `angle_fraction`, `count` are all
    parameters.
    Requires `r_out > r_in > 0` and `0 < angle_fraction <= 1`.  `dynamic`
    toggles between the transient (capacitive) and quasi-static (massless)
    modes -- see `TwoNodeWall`.

    Gas permeation (`leaky=True`).  See `TwoNodeWall` for the shared plumbing;
    the spherical geometry terms this wall supplies are::

        K   = f * 4*pi*r_in*r_out / (r_out - r_in)             [m]   (permeation conductance)
        V_j = f * 4/3*pi*(r(j+0.5)**3 - r(j-0.5)**3)           [m^3] (equal-1/r FV shells)

    so the steady molar leak is `f * 4*pi*Phi(T)*r_in*r_out/(r_out-r_in)
    * (p_in**(1/n) - p_out**(1/n))`, scaled by the covered fraction.
    """

    def __init__(
        self, rho, cp, k,
        r_in: Annotated[float, ParamSpec("Inner radius of the spherical shell "
                       "(bore side).", unit="m")],
        r_out: Annotated[float, ParamSpec("Outer radius of the spherical "
                        "shell.", unit="m")],
        T_init=293.15,
        dynamic=True,
        angle_fraction: Annotated[float, ParamSpec("Fraction of the full sphere "
                       "(4*pi solid angle) the wall covers (1 = full shell, "
                       "0.5 = hemisphere, ...); scales all extensive terms "
                       "(mass, conductance, leak).", unit="1")] = 1.0,
        count: Annotated[float, ParamSpec("Number of identical parallel shells "
                       "this one component represents (multiplicity >= 1); "
                       "scales every extensive term so N shells simulate as one "
                       "without extra equations.", unit="1")] = 1.0,
        leaky=False,
        permeation_flux=None,
        p_in_init=101325.0,
        p_out_init=101325.0,
    ):
        if not (r_out > r_in > 0):
            raise ValueError(
                f"SphericalWall requires r_out > r_in > 0; got "
                f"r_in={r_in}, r_out={r_out}"
            )
        if not (0 < angle_fraction <= 1):
            raise ValueError(
                f"SphericalWall requires 0 < angle_fraction <= 1; got "
                f"angle_fraction={angle_fraction}"
            )
        if not (getattr(count, 'value', count) >= 1):
            raise ValueError(
                f"SphericalWall requires count >= 1; got count={count}")
        self.rho = rho
        self.cp = cp
        self.k = k
        self.r_in = r_in
        self.r_out = r_out
        self.angle_fraction = angle_fraction
        self.count = count
        self.T_init = T_init
        self.dynamic = dynamic
        self._setup_permeation(leaky, permeation_flux, p_in_init, p_out_init)
        super().__init__()

    def _declare_geometry(self):
        spec = merged_param_specs(type(self))
        self.add_component('r_in', Parameter(self.r_in, **spec['r_in'].param_kwargs()))
        self.add_component('r_out', Parameter(self.r_out, **spec['r_out'].param_kwargs()))
        self.add_component('angle_fraction', Parameter(
            self.angle_fraction, **spec['angle_fraction'].param_kwargs()))
        self.add_component('count', Parameter(
            self.count, **spec['count'].param_kwargs()))

    def _node_capacity(self):
        # Shell thermal mass split across the two surface nodes (cap fraction f).
        f = self['angle_fraction'].symbol * self['count'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        V = f * sp.Rational(4, 3) * sp.pi * (r_out ** 3 - r_in ** 3)
        return self['rho'].symbol * self['cp'].symbol * V / 2

    def _conductance(self):
        # Exact radial conduction conductance of a hollow sphere cap.
        f = self['angle_fraction'].symbol * self['count'].symbol
        k = self['k'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        return f * 4 * sp.pi * k * r_in * r_out / (r_out - r_in)

    def _perm_geom_conductance(self):
        # Exact radial permeation conductance of a hollow sphere cap (analogue
        # of the thermal G with k -> Phi): N_dot = Phi * K * (c_a - c_b).
        f = self['angle_fraction'].symbol * self['count'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        return f * 4 * sp.pi * r_in * r_out / (r_out - r_in)

    def _perm_shell_volumes(self, n):
        # Equal-(1/r) spacing makes the n+1 series conductances equal (the
        # spherical conduction coordinate is u = 1/r), so they telescope to the
        # exact spherical resistance.  Shell j spans radii at indices
        # j-0.5 .. j+0.5, scaled by the cap fraction f.
        f = self['angle_fraction'].symbol * self['count'].symbol
        r_in = self['r_in'].symbol
        r_out = self['r_out'].symbol
        u_in = 1 / r_in
        u_out = 1 / r_out
        Delta_u = (u_in - u_out) / (n + 1)

        def r_at(x):  # radius decreases in u, i.e. grows with x
            return 1 / (u_in - x * Delta_u)

        return [f * sp.Rational(4, 3) * sp.pi
                * (r_at(j + 0.5) ** 3 - r_at(j - 0.5) ** 3)
                for j in range(1, n + 1)]
