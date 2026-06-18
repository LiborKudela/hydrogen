"""Fluid physics-domain component library.

Compressible-fluid plumbing built on `hydrogen.model`. This domain owns
its own typed connector (`FluidPort_phm`) and a set of components for
ambient inlets/outlets, two-port pipe segments, splitters, junctions,
pressure vessels, the loop buffer, and the heated `StraightPipe`
wrapper. See `README.md` in this folder for the full domain overview.
"""

from .fluid_components import (
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

__all__ = [
    # ports
    "FluidPort_phm",
    # components
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
]
