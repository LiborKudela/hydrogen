"""Reusable heat-transfer components built on top of `hydrogen.model`.

Module layout (mirrors the fluid library so each physics domain is
self-contained and readable top-to-bottom):

  1. `ThermalPort_TQ` -- the typed port this library exposes on every
     component.  It lives at the top of the module rather than in the
     generic `hydrogen.ports` so the heat-transfer domain owns its own
     connector kind; `hydrogen.ports` only defines the generic `Port`
     base class and the shared error hierarchy.
  2. Boundary conditions -- `FixedTemperature`, `FixedHeatFlow`,
     `ConvectiveBoundary`.  Small single-port models used to drive a
     thermal network for testing and for the worked examples.
  3. Passive elements -- `ThermalConductor`: a massless conductance used
     to wire a prescribed-temperature source onto a capacitive node
     (driving a heat capacity through a conductance is well-posed; wiring
     a temperature straight onto it is a high-index constraint).
  4. Components -- `TwoNodeWall` is the shared base: a wall lumped into two
     surface heat-capacity nodes with conduction between them.  `FlatWall`
     (plane slab) and `CylindricalWall` (hollow tube) subclass it and
     supply only the geometry-specific node capacity and conductance
     (Cartesian vs radial).

Sign convention -- Modelica "flow into me":
    Every port's `Q_dot` is positive when heat flows INTO the component
    that owns the port through that face.  When two same-orientation
    ports are wired, `Model.connect()` emits a sum-to-zero on the flow
    channel (`Q_dot_a + Q_dot_b == 0`), i.e. the heat leaving one
    component enters the other -- the thermal analogue of the
    Kirchhoff / Modelica connector rule.
"""

from __future__ import annotations

import sympy as sp

from ...model import DifferentialVariable, Model, Parameter, Variable
from ...ports import Port


# ---------------------------------------------------------------------------
# Thermal port -- (T, Q_dot) interface used by every component below
# ---------------------------------------------------------------------------


class ThermalPort_TQ(Port):
    """Heat-transfer interface carrying `(T, Q_dot)`.

    * `T`       - port temperature                [K]   (across)
    * `Q_dot`   - heat flow rate                   [W]   (THROUGH;
                  positive = "INTO me" under the Modelica
                  "flow into me" convention used package-wide)

    Both faces of every component here use `flow_orientation='in'`
    (positive `Q_dot` enters the component), so `Model.connect()` emits a
    sum-to-zero on the flow channel when two same-orientation ports are
    wired -- the heat-conduction analogue of the Kirchhoff / Modelica
    connector convention.

    The thermal domain carries no `medium`, so two `ThermalPort_TQ` of
    any owners may be connected as long as their `kind` matches.
    """

    kind = "thermal_TQ"
    required_channels = ("T", "Q_dot")
    flow_channels = ("Q_dot",)


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

    def __init__(self, T_set=293.15):
        self.T_set = T_set
        super().__init__()

    def declare_components(self):
        self.add_component('T_set', Parameter(self.T_set, "K"))
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

    def __init__(self, Q_flow=0.0, T_init=293.15):
        self.Q_flow = Q_flow
        self.T_init = T_init
        super().__init__()

    def declare_components(self):
        self.add_component('Q_flow', Parameter(self.Q_flow, "W"))
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

    def __init__(self, h, A, T_inf=293.15):
        self.h = h
        self.A = A
        self.T_inf = T_inf
        super().__init__()

    def declare_components(self):
        self.add_component('h', Parameter(self.h, "W/m^2/K"))
        self.add_component('A', Parameter(self.A, "m^2"))
        self.add_component('T_inf', Parameter(self.T_inf, "K"))
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

    def __init__(self, G, T_init=293.15):
        self.G = G
        self.T_init = T_init
        super().__init__()

    def declare_components(self):
        self.add_component('G', Parameter(self.G, "W/K"))
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

    # The `dynamic` flag toggles `declare_equations` between an ODE form (with
    # `der_*` states) and an algebraic form, so the equation structure is NOT
    # determined by the class alone.  Declaring it here makes the per-class
    # equation template cache key on `(class, dynamic)`, so a model mixing
    # dynamic and quasi-static walls of the same geometry caches each variant
    # correctly instead of replaying one onto the other.
    _cache_key_flags = ("dynamic",)

    def declare_components(self):
        # Shared material parameters; subclasses add geometry on top.
        self.add_component('rho', Parameter(self.rho, "kg/m^3"))
        self.add_component('cp', Parameter(self.cp, "J/kg/K"))
        self.add_component('k', Parameter(self.k, "W/m/K"))
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

    def __init__(self, rho, cp, k, A, L, T_init=293.15, dynamic=True):
        self.rho = rho
        self.cp = cp
        self.k = k
        self.A = A
        self.L = L
        self.T_init = T_init
        self.dynamic = dynamic
        super().__init__()

    def _declare_geometry(self):
        self.add_component('A', Parameter(self.A, "m^2"))
        self.add_component('L', Parameter(self.L, "m"))

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
    """

    def __init__(self, rho, cp, k, r_in, r_out, length, T_init=293.15, dynamic=True):
        if not (r_out > r_in > 0):
            raise ValueError(
                f"CylindricalWall requires r_out > r_in > 0; got "
                f"r_in={r_in}, r_out={r_out}"
            )
        self.rho = rho
        self.cp = cp
        self.k = k
        self.r_in = r_in
        self.r_out = r_out
        self.length = length
        self.T_init = T_init
        self.dynamic = dynamic
        super().__init__()

    def _declare_geometry(self):
        self.add_component('r_in', Parameter(self.r_in, "m"))
        self.add_component('r_out', Parameter(self.r_out, "m"))
        self.add_component('length', Parameter(self.length, "m"))

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
