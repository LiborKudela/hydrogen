"""Component libraries grouped by physics domain.

Each physics domain lives in its own subpackage (subfolder) that bundles
the domain's Python module(s) and a `README.md`. A domain library is
self-contained: it declares its own typed `Port` subclass(es) alongside
the components that use them, so a reader of the fluid library sees the
connector contract and the component implementations in one place.

Currently shipped domains:

  * `fluid/` -- compressible-fluid plumbing: ambient inlets/outlets,
    two-port pipe segments, splitters, junctions, pressure vessels, the
    loop buffer, the heated `StraightPipe` wrapper.  Exposes
    `FluidPort_phm` and all fluid component classes.
  * `thermal/` -- lumped heat transfer: two-node `FlatWall` / `CylindricalWall`
    conduction models plus `FixedTemperature` / `FixedHeatFlow` /
    `ConvectiveBoundary` boundary conditions.  Exposes `ThermalPort_TQ` and
    these classes.
  * `power/` -- coupled (conjugate) power-engineering models composed from the
    fluid and thermal domains.  Exposes `ConjugatePipe` (a fluid pipe wrapped
    segment-by-segment in a cylindrical metal wall).

This `__init__` re-exports the top-level public API of every shipped
domain so existing call sites can keep writing
`from hydrogen.components import StraightPipe` (or `FluidPort_phm`).
Future cross-domain libraries (thermal, electrical, ...) plug in by
adding their subpackage here and re-exporting the symbols they want at
this package level.
"""

from .fluid import (
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    FluidPort_phm,
    HeatedSegment,
    LoopBuffer,
    MixingJunction,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    StraightPipe,
    TwoPortSegment,
)
from .power import ConjugatePipe
from .thermal import (
    ConvectiveBoundary,
    CylindricalWall,
    FixedHeatFlow,
    FixedTemperature,
    FlatWall,
    ThermalConductor,
    ThermalPort_TQ,
    TwoNodeWall,
)

__all__ = [
    # ports (re-exported from the `fluid` domain)
    "FluidPort_phm",
    # fluid components
    "AmbientInlet",
    "AmbientOutlet",
    "TwoPortSegment",
    "HeatedSegment",
    "AdiabaticPump",
    "StraightPipe",
    "PressureOutlet",
    "PressureSource",
    "PressureVessel",
    "Splitter",
    "MixingJunction",
    "LoopBuffer",
    # ports (re-exported from the `thermal` domain)
    "ThermalPort_TQ",
    # thermal components
    "FixedTemperature",
    "FixedHeatFlow",
    "ConvectiveBoundary",
    "ThermalConductor",
    "TwoNodeWall",
    "FlatWall",
    "CylindricalWall",
    # power components
    "ConjugatePipe",
]
