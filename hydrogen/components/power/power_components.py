"""Power-engineering components built on top of `hydrogen.model`.

This domain composes the `thermofluid` library into the coupled (conjugate)
models that power-plant / process plumbing needs, rather than introducing new
primitive physics.

  1. `ConjugatePipe` -- a fluid pipe whose every segment is wrapped in a single
     cylindrical metal wall (conjugate heat transfer).  It is a thin
     specialisation of `thermofluid.Pipe` (single, non-permeable `WallLayer`);
     `Pipe` itself is the general multi-layer / optionally-permeable assembly.

The fluid/thermal coupling is wired through the typed ports each domain already
owns (`FluidPort_phm`, `ThermalPort_TQ`); no new connector kind is introduced.
"""

from __future__ import annotations

from typing import Annotated

from ...medium import CoolPropMedium
from ...paramspec import ParamSpec
from ..materials import WallMaterial
from ..thermofluid.assemblies import Pipe, WallLayer


class ConjugatePipe(Pipe):
    """A fluid pipe coupled to a single metal wall with thermal mass (conjugate
    heat transfer).

    A convenience specialisation of `thermofluid.Pipe` with exactly one
    (non-permeable) `WallLayer`.  Every fluid segment exchanges heat with its
    own `CylindricalWall`, whose outer surface is terminated per `outer`
    (``"adiabatic"`` / ``"convective"`` / ``"expose"``).  Fluid connectivity is
    re-exposed as `inlet` / `outlet`, so a `ConjugatePipe` drops into a fluid
    network exactly where a `StraightPipe` would.  For multi-layer walls or wall
    permeation, use `thermofluid.Pipe` directly.

    Geometry / material:
        D            - inner (flow) diameter                       [m]
        L            - total pipe length                           [m]
        wall_thickness - metal wall radial thickness               [m]
        rho_wall, cp_wall, k_wall - wall material properties
        epsilon, z_in, z_out, n_segments - as for `StraightPipe`
        wall_dynamic - `True` (default) for capacitive metal walls (thermal
                       mass / heat-up transient); `False` for quasi-static
                       (massless) walls that conduct heat straight through.
        outer        - outer-surface termination: ``"adiabatic"`` (insulated),
                       ``"convective"`` (Newton cooling via `h_ext`, `T_ext`),
                       or ``"expose"`` (re-exposed as `wall_outer_{i}` ports).
    """

    #: Geometry / thermal fields shared with `Pipe` (D, L, epsilon, z_in,
    #: z_out, n_segments, T_wall_init, h_ext, T_ext) inherit their specs via
    #: the MRO; only the single-wall material fields are annotated here.
    def __init__(
        self,
        medium: CoolPropMedium,
        D,
        L,
        epsilon,
        z_in,
        z_out,
        n_segments,
        wall_thickness: Annotated[float, ParamSpec("Radial thickness of the "
                                 "metal wall (> 0).", unit="m", default=0.002)],
        rho_wall: Annotated[float, ParamSpec("Wall material density.",
                           unit="kg/m^3", default=7900.0)],
        cp_wall: Annotated[float, ParamSpec("Wall material specific heat "
                          "capacity.", unit="J/(kg*K)", default=500.0)],
        k_wall: Annotated[float, ParamSpec("Wall material thermal "
                         "conductivity.", unit="W/(m*K)", default=15.0)],
        T_wall_init=293.15,
        outer: Annotated[str, ParamSpec("Outer-surface thermal termination.",
                        choices=("adiabatic", "convective", "expose"),
                        structural=True)] = "adiabatic",
        h_ext=10.0,
        T_ext=293.15,
        wall_dynamic: Annotated[bool, ParamSpec("If true the wall carries "
                               "thermal mass (heat-up transient); if false it "
                               "conducts quasi-statically.", structural=True)]
                      = True,
    ):
        if wall_thickness <= 0:
            raise ValueError(
                f"ConjugatePipe: wall_thickness must be > 0; got {wall_thickness}"
            )
        material = WallMaterial(name="conjugate wall", rho=rho_wall, cp=cp_wall, k=k_wall)
        # Kept as attributes (not just forwarded) so the structural cache key
        # can read them back -- `collect_equations` resolves every structural /
        # literal flag via ``getattr(self, name)``.
        self.outer = outer
        self.wall_dynamic = wall_dynamic
        super().__init__(
            medium, D, L, epsilon, z_in, z_out, n_segments,
            layers=[WallLayer(material, wall_thickness, dynamic=wall_dynamic)],
            outer_thermal=outer, h_ext=h_ext, T_ext=T_ext, T_wall_init=T_wall_init,
        )
