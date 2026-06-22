"""Typed connectors exposed by the `thermofluid` component library.

This domain models everything around pipes, vessels, and walls -- bulk fluid
flow, wall heat conduction, and hydrogen permeation -- so it owns three port
kinds, collected here so the flow / wall / permeation modules can share them
without import cycles:

  * `FluidPort_phm`      -- compressible flow, `(p, h, m_dot)`.
  * `ThermalPort_TQ`     -- heat transfer, `(T, Q_dot)`.
  * `PermeationPort_pN`  -- gas permeation leak, `(p_partial, m_dot_leak)`.

`hydrogen.ports` keeps only the generic `Port` base class and the shared error
hierarchy; concrete connector kinds live next to the components that use them.
"""

from __future__ import annotations

from ...ports import Port


class FluidPort_phm(Port):
    """Compressible-fluid interface carrying `(p, h, m_dot)`.

    * `p`       - port pressure                   [Pa]   (across)
    * `h`       - port specific enthalpy          [J/kg] (across)
    * `m_dot`   - port mass flow rate             [kg/s] (THROUGH;
                  positive = "INTO me" under the Modelica
                  "flow into me" convention used package-wide)

    All standard flow components declare either an `outlet` or an `inlet`
    port of this kind.  Both faces use `flow_orientation='in'` (positive
    m_dot enters the component), so `Model.connect()` emits a sum-to-zero on
    the flow channel when two same-orientation ports are wired -- the
    Kirchhoff / Modelica connector convention.

    Two FluidPort_phm of different `medium` are refused at connect-time
    (`PortMediumMismatchError`) to catch air<->hydrogen cross-wiring before it
    produces a confusing CoolProp NameError in the lambdified residual.
    """

    kind = "fluid_phm"
    required_channels = ("p", "h", "m_dot")
    flow_channels = ("m_dot",)


class ThermalPort_TQ(Port):
    """Heat-transfer interface carrying `(T, Q_dot)`.

    * `T`       - port temperature                [K]   (across)
    * `Q_dot`   - heat flow rate                   [W]   (THROUGH;
                  positive = "INTO me" under the Modelica
                  "flow into me" convention used package-wide)

    Both faces of a wall/boundary use `flow_orientation='in'` (positive
    `Q_dot` enters the component), so `Model.connect()` emits a sum-to-zero on
    the flow channel when two same-orientation ports are wired -- the
    heat-conduction analogue of the Kirchhoff / Modelica connector convention.

    Carries no `medium`, so two `ThermalPort_TQ` of any owners may be connected
    as long as their `kind` matches.
    """

    kind = "thermal_TQ"
    required_channels = ("T", "Q_dot")
    flow_channels = ("Q_dot",)


class PermeationPort_pN(Port):
    """Gas-permeation interface carrying `(p_partial, m_dot_leak)`.

    * `p_partial`   - partial pressure of the permeating species [Pa] (across)
    * `m_dot_leak`  - leak mass-flow rate                   [kg/s] (THROUGH;
                      positive = "INTO me" under the package-wide Modelica
                      "flow into me" convention)

    Wiring a fluid volume's leak port to a wall's leak port unifies the
    partial pressure (across) and sums the two leak mass-flows to zero
    (Kirchhoff): the gas the wall takes in equals the gas the fluid loses.  The
    connector carries no `medium`, so any two `PermeationPort_pN` of matching
    `kind` may be connected.

    Exposed by BOTH a leaky `flow.TwoPortSegment` and a leaky `walls.CylindricalWall`,
    which is why it lives here in the shared domain `ports` module rather than in
    either the flow or the walls module.
    """

    kind = "permeation_pN"
    required_channels = ("p_partial", "m_dot_leak")
    flow_channels = ("m_dot_leak",)
