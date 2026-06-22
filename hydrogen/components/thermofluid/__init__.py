"""Thermo-fluid component library.

A single domain for everything modelled around pipes, vessels, and walls --
bulk compressible flow, wall heat conduction, and hydrogen permeation -- which
in practice compose into the same physical objects (a heated, leaky pipe is one
component, not three).  Organised into submodules:

  * `ports`       -- the typed connectors: `FluidPort_phm`, `ThermalPort_TQ`,
    `PermeationPort_pN`.
  * `flow`        -- compressible-flow components: `TwoPortSegment`,
    `StraightPipe`, vessels, valves, junctions, sources/outlets.
  * `walls`       -- lumped wall conduction: `TwoNodeWall`, `FlatWall`,
    `CylindricalWall`, `SphericalWall` (any `leaky=True` for gas permeation),
    plus thermal boundary conditions.
  * `permeation`  -- gas-permeation materials (`Permeant`, `TransportFit`),
    flux models (`SteadyRichardson`, `TransientDiffusion`) injected into a
    leaky wall, and the `FixedPartialPressure` boundary.  A leaky flow volume
    is just `PressureVessel(leaky=True)` / `StraightPipe(leaky=True)`.

  * `assemblies`  -- batteries-included composites: `Pipe` (a flowing pipe
    wrapped per-segment in a `WallLayer` stack, optionally permeable) and `Tank`
    (a lumped-gas pressure vessel with a cylindrical barrel + spherical caps,
    conjugate heat, and optional permeation) built from the modules above.

See `README.md` in this folder for the full domain overview.
"""

from .assemblies import Pipe, Tank, WallLayer
from .flow import (
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    ClosedEnd,
    CompressibleValve,
    IncompressibleValve,
    LoopBuffer,
    MixingJunction,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    StraightPipe,
    TwoPortSegment,
    Valve,
)
from .permeation import (
    H2,
    H2_IN_AUSTENITIC,
    H2_IN_AISI_304,
    H2_IN_AISI_316,
    HELIUM,
    NITROGEN,
    FixedPartialPressure,
    Permeant,
    PermeationFlux,
    SteadyRichardson,
    TransientDiffusion,
    TransportFit,
)
from .ports import FluidPort_phm, PermeationPort_pN, ThermalPort_TQ
from .walls import (
    ConvectiveBoundary,
    CylindricalWall,
    FixedHeatFlow,
    FixedTemperature,
    FlatWall,
    SphericalWall,
    ThermalConductor,
    TwoNodeWall,
)

__all__ = [
    # ports
    "FluidPort_phm",
    "ThermalPort_TQ",
    "PermeationPort_pN",
    # flow components
    "AmbientInlet",
    "AmbientOutlet",
    "ClosedEnd",
    "TwoPortSegment",
    "AdiabaticPump",
    "StraightPipe",
    "Valve",
    "IncompressibleValve",
    "CompressibleValve",
    "PressureOutlet",
    "PressureSource",
    "PressureVessel",
    "Splitter",
    "MixingJunction",
    "LoopBuffer",
    # wall / thermal components
    "FixedTemperature",
    "FixedHeatFlow",
    "ConvectiveBoundary",
    "ThermalConductor",
    "TwoNodeWall",
    "FlatWall",
    "CylindricalWall",
    "SphericalWall",
    # composite assemblies
    "Pipe",
    "Tank",
    "WallLayer",
    # permeation materials
    "Permeant",
    "TransportFit",
    "H2",
    "HELIUM",
    "NITROGEN",
    "H2_IN_AUSTENITIC",
    "H2_IN_AISI_304",
    "H2_IN_AISI_316",
    # permeation models / boundary
    "PermeationFlux",
    "SteadyRichardson",
    "TransientDiffusion",
    "FixedPartialPressure",
]
