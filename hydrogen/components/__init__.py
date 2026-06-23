"""Component libraries grouped by usage domain.

Each domain lives in its own subpackage (subfolder) that bundles the domain's
Python module(s) and a `README.md`. A domain library is self-contained: it
declares its own typed `Port` subclass(es) alongside the components that use
them, so a reader sees the connector contract and the implementations together.

Currently shipped domains:

  * `thermofluid/` -- everything modelled around pipes, vessels, and walls:
    compressible flow (`flow`), wall heat conduction (`walls`), and hydrogen
    permeation (`permeation`).  These compose into the same physical objects
    (a heated, leaky pipe is one component), so they share one domain and its
    connectors (`FluidPort_phm`, `ThermalPort_TQ`, `PermeationPort_pN`).
  * `power/` -- coupled (conjugate) power-engineering models composed from the
    thermofluid domain.  Exposes `ConjugatePipe` (a flow pipe wrapped
    segment-by-segment in a cylindrical metal wall).
  * `control/` -- Modelica.Blocks-style signal blocks (sources, maths,
    controllers) wired through the `RealSignal` connector.  Used to build
    setpoints / feedback controllers that drive actuators in other domains.

This `__init__` deliberately does **not** re-export the domain APIs.  Each
component is imported from the module where it is defined, so the import path
mirrors the package layout, e.g.::

    from hydrogen.components.thermofluid.assemblies import Pipe, Tank
    from hydrogen.components.thermofluid.flow import StraightPipe
    from hydrogen.components.control.control_components import PID

Tooling that needs the full set of shipped components should use the catalog
helpers in :mod:`hydrogen.serialization` (``component_catalog`` /
``component_spec``), which discover components by walking these subpackages.
"""
