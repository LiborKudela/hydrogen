"""Power-engineering physics-domain component library.

Coupled (conjugate) models that compose the `fluid` and `thermal` domains
into the building blocks of power-plant / process plumbing.  Currently
ships the `ConjugatePipe`: a fluid `StraightPipe` whose every segment is
wrapped in a `CylindricalWall` with real metal thermal mass, plus a
configurable outer boundary.  See `README.md` in this folder for the full
domain overview.
"""

from .power_components import ConjugatePipe

__all__ = [
    "ConjugatePipe",
]
