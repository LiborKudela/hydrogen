"""Power-engineering components built on top of `hydrogen.model`.

This domain composes the existing `fluid` and `thermal` libraries into the
coupled (conjugate) models that power-plant / process plumbing needs, rather
than introducing new primitive physics.  The first component is the
`ConjugatePipe`: a fluid `StraightPipe` whose every segment is wrapped in a
`CylindricalWall`, so the metal wall has real thermal mass and the fluid
exchanges heat with it segment-by-segment.

Module layout:

  1. `ConjugatePipe` -- fluid pipe + per-segment cylindrical metal wall +
     a configurable outer boundary (insulated / convective / exposed).

The fluid/thermal coupling is wired through the typed ports that each domain
already owns (`FluidPort_phm`, `ThermalPort_TQ`); no new connector kind is
introduced here.
"""

from __future__ import annotations

import numpy as np

from ...medium import CoolPropMedium
from ...model import Model
from ..fluid.fluid_components import FluidPort_phm, StraightPipe
from ..thermal.thermal_components import (
    ConvectiveBoundary,
    CylindricalWall,
    FixedHeatFlow,
    ThermalPort_TQ,
)


class ConjugatePipe(Model):
    """A fluid pipe coupled to a metal wall with thermal mass (conjugate heat
    transfer).

    Internally builds a `StraightPipe(heat_port=True)` and, for every one of
    its `n_segments` segments, a `CylindricalWall` whose INNER surface
    (`port_a`) is wired to that segment's `wall` thermal port::

        fluid:   ===[ segment_0 ]===[ segment_1 ]=== ... ===[ segment_N-1 ]===
                       |  wall          |  wall                 |  wall
                    port_a           port_a                  port_a
        metal:    [ wall_0 ]        [ wall_1 ]      ...    [ wall_N-1 ]
                    |  port_b          |  port_b              |  port_b
                    outer_0           outer_1               outer_N-1

    The segment `wall` port publishes the convective heat `q` (W) and the wall
    inner-node temperature; the connection closes the segment's `T_wall` to
    the wall metal temperature and (same-orientation sum-to-zero) feeds `-q`
    into the wall node, so energy is conserved across the fluid/metal
    interface.

    The OUTER wall surface (`port_b` of every `CylindricalWall`) is terminated
    according to `outer`:

      * ``"adiabatic"`` (default) -- a `FixedHeatFlow(0)` per segment: the
        outer surface is perfectly insulated, so all heat the fluid gives up
        is stored in (or supplied from) the metal.  Self-contained and
        well-posed on its own.
      * ``"convective"`` -- a `ConvectiveBoundary(h_ext, A_outer, T_ext)` per
        segment, modelling Newton cooling to a far-field at `T_ext` through a
        film coefficient `h_ext` over each segment's outer area
        `2*pi*r_out*L_segment`.
      * ``"expose"`` -- no internal termination; instead each segment's outer
        node is re-exposed as a `wall_outer_{i}` `ThermalPort_TQ` on this
        component, for the parent model to wire (e.g. to another wall, a
        radiation model, ...).

    Fluid connectivity is re-exposed as `inlet` / `outlet` `FluidPort_phm`
    (bound to the inner `StraightPipe`'s port variables) so a `ConjugatePipe`
    drops into a fluid network exactly where a `StraightPipe` would.

    Geometry / material:
        D            - inner (flow) diameter                       [m]
        L            - total pipe length                           [m]
        wall_thickness - metal wall radial thickness               [m]
        rho_wall, cp_wall, k_wall - wall material properties
        epsilon, z_in, z_out, n_segments - as for `StraightPipe`
        wall_dynamic - `True` (default) for capacitive metal walls (thermal
                       mass / heat-up transient); `False` for quasi-static
                       (massless) walls that conduct heat straight through.
    """

    def __init__(
        self,
        medium: CoolPropMedium,
        D,
        L,
        epsilon,
        z_in,
        z_out,
        n_segments,
        wall_thickness,
        rho_wall,
        cp_wall,
        k_wall,
        T_wall_init=293.15,
        outer="adiabatic",
        h_ext=10.0,
        T_ext=293.15,
        wall_dynamic=True,
    ):
        if outer not in ("adiabatic", "convective", "expose"):
            raise ValueError(
                f"ConjugatePipe: outer must be 'adiabatic', 'convective' or "
                f"'expose'; got {outer!r}"
            )
        if wall_thickness <= 0:
            raise ValueError(
                f"ConjugatePipe: wall_thickness must be > 0; got {wall_thickness}"
            )
        self.medium = medium
        self.D = D
        self.L = L
        self.epsilon = epsilon
        self.z_in = z_in
        self.z_out = z_out
        self.n_segments = n_segments
        self.wall_thickness = wall_thickness
        self.rho_wall = rho_wall
        self.cp_wall = cp_wall
        self.k_wall = k_wall
        self.T_wall_init = T_wall_init
        self.outer = outer
        self.h_ext = h_ext
        self.T_ext = T_ext
        # When False the metal wall is quasi-static (no thermal mass): heat
        # passes straight through by conduction.  See `thermal.TwoNodeWall`.
        self.wall_dynamic = wall_dynamic
        # Derived radial geometry, shared by every segment wall.
        self.r_in = D / 2.0
        self.r_out = D / 2.0 + wall_thickness
        self.L_segment = L / n_segments
        super().__init__()

    def declare_components(self):
        self.add_component(
            'pipe',
            StraightPipe(
                self.medium,
                D=self.D,
                L=self.L,
                epsilon=self.epsilon,
                z_in=self.z_in,
                z_out=self.z_out,
                n_segments=self.n_segments,
                heat_port=True,
            ),
        )

        A_outer = 2.0 * np.pi * self.r_out * self.L_segment
        for i in range(self.n_segments):
            self.add_component(
                f'wall_{i}',
                CylindricalWall(
                    self.rho_wall,
                    self.cp_wall,
                    self.k_wall,
                    self.r_in,
                    self.r_out,
                    self.L_segment,
                    T_init=self.T_wall_init,
                    dynamic=self.wall_dynamic,
                ),
            )
            if self.outer == 'adiabatic':
                self.add_component(f'outer_{i}', FixedHeatFlow(0.0, T_init=self.T_wall_init))
            elif self.outer == 'convective':
                self.add_component(f'outer_{i}', ConvectiveBoundary(self.h_ext, A_outer, T_inf=self.T_ext))
            else:  # 'expose'
                self.add_port(f'wall_outer_{i}', ThermalPort_TQ(
                    self,
                    channels={'T': self[f'wall_{i}']['T_b'], 'Q_dot': self[f'wall_{i}']['Q_dot_b']},
                    flow_orientation='in',
                ))

        # Re-expose the fluid inlet/outlet so a ConjugatePipe is drop-in for a
        # StraightPipe.  Bound to the inner pipe's own port variables.
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={
                'p': self['pipe']['p_in'],
                'h': self['pipe']['h_in'],
                'm_dot': self['pipe']['m_dot_in'],
            },
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={
                'p': self['pipe']['p_out'],
                'h': self['pipe']['h_out'],
                'm_dot': self['pipe']['m_dot_out'],
            },
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        for i in range(self.n_segments):
            # Inner wall surface <-> fluid segment wall port (conjugate coupling).
            self.connect(
                self['pipe'][f'pipe_segment_{i}'].ports['wall'],
                self[f'wall_{i}'].ports['port_a'],
            )
            # Outer wall surface termination (unless exposed for external wiring).
            if self.outer in ('adiabatic', 'convective'):
                self.connect(
                    self[f'wall_{i}'].ports['port_b'],
                    self[f'outer_{i}'].ports['heat'],
                )
        return []
