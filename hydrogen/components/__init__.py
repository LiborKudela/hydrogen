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

This `__init__` re-exports the top-level public API of every shipped
domain so existing call sites can keep writing
`from hydrogen.components import StraightPipe` (or `FluidPort_phm`).
"""

from .control import (
    Add,
    Constant,
    Feedback,
    FirstOrder,
    Gain,
    Integrator,
    Limiter,
    PID,
    Product,
    Ramp,
    RealSignal,
    Sine,
    Step,
    Sum,
)
from .materials import AISI_304, AISI_316, WallMaterial
from .power import ConjugatePipe
from .thermofluid import (
    H2,
    H2_IN_AUSTENITIC,
    H2_IN_AISI_304,
    H2_IN_AISI_316,
    HELIUM,
    NITROGEN,
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    ClosedEnd,
    CompressibleValve,
    ConvectiveBoundary,
    CylindricalWall,
    FixedHeatFlow,
    FixedPartialPressure,
    FixedTemperature,
    FlatWall,
    FluidPort_phm,
    IncompressibleValve,
    LoopBuffer,
    MixingJunction,
    Permeant,
    PermeationFlux,
    PermeationPort_pN,
    Pipe,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    SphericalWall,
    SteadyRichardson,
    StraightPipe,
    Tank,
    ThermalConductor,
    ThermalPort_TQ,
    TransientDiffusion,
    TransportFit,
    TwoNodeWall,
    TwoPortSegment,
    Valve,
    WallLayer,
)

__all__ = [
    # connectors (re-exported from the `thermofluid` domain)
    "FluidPort_phm",
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
    "SphericalWall",
    # composite assemblies
    "Pipe",
    "Tank",
    "WallLayer",
    # power components
    "ConjugatePipe",
    # permeation-library port
    "PermeationPort_pN",
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
    # wall materials
    "WallMaterial",
    "AISI_304",
    "AISI_316",
    # control-library port
    "RealSignal",
    # control components
    "Constant",
    "Step",
    "Ramp",
    "Sine",
    "Gain",
    "Add",
    "Feedback",
    "Sum",
    "Product",
    "Limiter",
    "Integrator",
    "FirstOrder",
    "PID",
]
