"""Heat-transfer physics-domain component library.

Lumped thermal components built on `hydrogen.model`.  This domain owns
its own typed connector (`ThermalPort_TQ`, carrying `(T, Q_dot)`) and a
set of boundary conditions plus a two-node `FlatWall` conduction model.
See `README.md` in this folder for the full domain overview.
"""

from .thermal_components import (
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
    # port
    "ThermalPort_TQ",
    # boundary conditions
    "FixedTemperature",
    "FixedHeatFlow",
    "ConvectiveBoundary",
    # passive elements
    "ThermalConductor",
    # components
    "TwoNodeWall",
    "FlatWall",
    "CylindricalWall",
]
