"""Thermo-fluid component library.

A single domain for everything modelled around pipes, vessels, and walls --
bulk compressible flow, wall heat conduction, and hydrogen permeation -- which
in practice compose into the same physical objects (a heated, leaky pipe is one
component, not three).  Organised into submodules:

  * `ports`       -- the typed connectors: `FluidPort_phm`, `ThermalPort_TQ`,
    `PermeationPort_pN`.
  * `flow`        -- compressible-flow components: `TwoPortSegment`,
    `StraightPipe`, vessels, valves, junctions, sources/outlets.
  * `sensors`     -- inline, drop-free flow instruments that publish a reading
    on a `control.RealSignal` output: `MassFlowSensor`, `MassTotalizer`,
    `TemperatureSensor` (static or total temperature), `VolumeFlowSensor`.
  * `walls`       -- lumped wall conduction: `TwoNodeWall`, `FlatWall`,
    `CylindricalWall`, `SphericalWall` (any `leaky=True` for gas permeation),
    plus thermal boundary conditions.
  * `permeation`  -- gas-permeation materials (`Permeant`, `TransportFit`),
    flux models (`SteadyRichardson`, `TransientDiffusion`, `SpecifiedFlux`)
    injected into a leaky wall, and the `FixedPartialPressure` boundary.  A leaky flow volume
    is just `PressureVessel(leaky=True)` / `StraightPipe(leaky=True)`.
  * `local_loss`  -- pluggable local- (minor-) pressure-loss correlations
    (`FixedK`, `SuddenExpansion`, `SuddenContraction`, ...) supplying the
    dimensionless coefficient `K` a `flow.LocalResistance` / `assemblies.LocalLoss`
    consumes; a UI offers them as a dropdown.

  * `assemblies`  -- batteries-included composites: `Pipe` (a flowing pipe
    wrapped per-segment in a `WallLayer` stack, optionally permeable), `Valve` /
    `LocalLoss` (a single-cell throttle / minor loss wrapped in an equivalent
    cylindrical `WallLayer` stack) and `Tank` (a lumped-gas pressure vessel with
    a cylindrical barrel + spherical caps, conjugate heat, and optional
    permeation) built from the modules above.

See `README.md` in this folder for the full domain overview.

Components are imported from their defining submodule (no flat re-exports),
so the import path mirrors the package layout, e.g.::

    from hydrogen.components.thermofluid.flow import StraightPipe, Valve
    from hydrogen.components.thermofluid.walls import CylindricalWall
    from hydrogen.components.thermofluid.assemblies import Pipe, Tank
    from hydrogen.components.thermofluid.ports import FluidPort_phm
"""
