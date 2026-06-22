"""Wall materials: the thermal property set of a solid wall.

A `WallMaterial` bundles the constants a metal (or other solid) wall needs for
heat conduction / storage -- `rho`, `cp`, `k` -- used by the thermal
`CylindricalWall` / `FlatWall`.

Permeation transport (the Arrhenius permeability `Phi(T)`, diffusivity `D(T)`,
and solubility `S(T)`) is *species-specific*: it depends on which gas is
dissolving in the wall, not just on the metal.  That data therefore lives with
the permeation physics in `hydrogen.components.thermofluid.permeation`
(`Permeant` + `TransportFit`), keeping this module a pure thermal-property
catalogue.  A leaky wall is built by pairing a thermal `WallMaterial` (for
`rho/cp/k`) with a `TransportFit` (for the gas-in-metal transport).

Data source for the AISI presets
--------------------------------
Thermal properties are representative engineering values for austenitic
stainless steels.  The matching hydrogen `TransportFit` presets (e.g.
`H2_IN_AUSTENITIC`) live in the permeation module and document their own Sandia
source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from ..paramspec import ParamSpec

#: Universal gas constant [J/mol/K].  Shared by the Arrhenius transport fits in
#: the permeation module.
R_GAS = 8.314462618


@dataclass(frozen=True)
class WallMaterial:
    """Thermal property set for a solid wall.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"AISI 316/316L"``).
    rho, cp, k : float
        Density [kg/m^3], specific heat [J/kg/K], thermal conductivity [W/m/K].

    Permeation transport is not stored here -- it is species-specific and lives
    in a `TransportFit` (see `hydrogen.components.thermofluid.permeation`).
    """

    name: Annotated[str, ParamSpec("Human-readable material label (e.g. "
                   "'AISI 316').")]
    rho: Annotated[float, ParamSpec("Density.", unit="kg/m^3")]
    cp: Annotated[float, ParamSpec("Specific heat capacity.", unit="J/(kg*K)")]
    k: Annotated[float, ParamSpec("Thermal conductivity.", unit="W/(m*K)")]

    def to_spec(self) -> dict:
        """Serializable value spec (see `hydrogen.serialization`)."""
        return {"__type__": "WallMaterial", "name": self.name,
                "rho": self.rho, "cp": self.cp, "k": self.k}

    @classmethod
    def from_spec(cls, d: dict) -> "WallMaterial":
        return cls(name=d["name"], rho=d["rho"], cp=d["cp"], k=d["k"])


#: AISI 316 / 316L austenitic stainless steel.
AISI_316 = WallMaterial(name="AISI 316/316L", rho=7990.0, cp=500.0, k=15.0)

#: AISI 304 / 304L austenitic stainless steel.
AISI_304 = WallMaterial(name="AISI 304/304L", rho=7900.0, cp=500.0, k=16.2)

#: Named materials a UI can offer as a choice list (the catalog surfaces these
#: as `presets`); a UI typically adds a "Custom" entry so the user can instead
#: enter `rho` / `cp` / `k` by hand.
WallMaterial.PRESETS = {
    "AISI 316/316L": AISI_316,
    "AISI 304/304L": AISI_304,
}

__all__ = ["WallMaterial", "AISI_304", "AISI_316", "R_GAS"]
