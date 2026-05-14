"""Component libraries grouped by physics domain.

Each submodule is a self-contained library: it declares its own typed
`Port` subclass(es) at the top of the file alongside the components that
use them, so a reader of the fluid library sees the connector contract
and the component implementations in one place.

Currently shipped libraries:

  * `fluid_components` -- compressible-fluid plumbing: ambient
    inlets/outlets, two-port pipe segments, splitters, junctions,
    pressure vessels, the loop buffer, the heated `StraightPipe`
    wrapper.  Exposes `FluidPort_phm` and all fluid component classes.

This `__init__` re-exports the top-level public API of every shipped
library so existing call sites can keep writing
`from hydrogen.components import StraightPipe` (or `FluidPort_phm`).
Future cross-domain libraries (thermal, electrical, ...) plug in by
adding their submodule here and re-exporting the symbols they want at
this package level.
"""

from .fluid_components import (
    AdiabaticPump,
    AmbientInlet,
    AmbientOutlet,
    FluidPort_phm,
    LoopBuffer,
    MixingJunction,
    PressureOutlet,
    PressureSource,
    PressureVessel,
    Splitter,
    StraightPipe,
    TwoPortSegment,
)

__all__ = [
    # ports (re-exported from `fluid_components`)
    "FluidPort_phm",
    # components
    "AmbientInlet",
    "AmbientOutlet",
    "TwoPortSegment",
    "AdiabaticPump",
    "StraightPipe",
    "PressureOutlet",
    "PressureSource",
    "PressureVessel",
    "Splitter",
    "MixingJunction",
    "LoopBuffer",
]
