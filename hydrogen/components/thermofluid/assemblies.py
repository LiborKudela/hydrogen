"""Composite, batteries-included thermofluid assemblies.

Where `flow` / `walls` / `permeation` provide the primitive components, this
module wires them into ready-to-use objects so a user model collapses to
`boundary - component - boundary`.

  * `WallLayer` -- one radial layer of a pipe wall (thermal material + radial
    thickness, optionally permeable via an injected flux model).
  * `Pipe` -- a flowing pipe wrapped, segment-by-segment, in a stack of
    cylindrical metal `WallLayer`s, with the wall outer surfaces terminated by
    internal thermal / partial-pressure boundaries.  The fluid `inlet` / `outlet`
    are the only ports the user wires.

`Pipe` is the high-level "just give me a pipe" component (the name is kept
deliberately plain so more physics can be folded in later); `flow.StraightPipe`
remains the lower-level flow-only primitive it is built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np

from ...medium import CoolPropMedium
from ...model import Model
from ...paramspec import ParamSpec
from ..materials import WallMaterial
from .flow import StraightPipe
from .permeation import FixedPartialPressure, PermeationFlux
from .ports import FluidPort_phm, ThermalPort_TQ
from .walls import ConvectiveBoundary, CylindricalWall, FixedHeatFlow, FixedTemperature

#: Allowed `Pipe.outer_thermal` modes -- defined at module scope so it is
#: resolvable inside the `Annotated[...]` choices of `Pipe.__init__` (annotation
#: strings are evaluated in module globals, not the class namespace).
_OUTER_THERMAL_MODES = ("adiabatic", "convective", "fixed", "expose")


@dataclass
class WallLayer:
    """One radial layer of a pipe wall.

    Attributes
    ----------
    material : WallMaterial
        Thermal property set (`rho/cp/k`) of this layer.
    thickness : float
        Radial thickness [m] (`> 0`).
    permeation : PermeationFlux | None
        If given, the layer is permeable: this flux model (carrying the
        `Permeant` + `TransportFit`) supplies the pressure-gradient ->
        mass-flow correlation.  `None` (default) makes the layer a pure thermal
        / permeation-barrier layer.
    dynamic : bool
        `True` (default) gives the layer thermal mass (heat-up transient);
        `False` makes it quasi-static (massless, conducts straight through).
    """

    material: Annotated[WallMaterial, ParamSpec("Thermal property set "
                       "(rho/cp/k) of this layer.")]
    thickness: Annotated[float, ParamSpec("Radial thickness of the layer "
                        "(> 0).", unit="m", default=0.002)]
    permeation: Annotated[PermeationFlux | None, ParamSpec(
        "Optional gas-permeation flux model; omit (None) for a pure thermal / "
        "barrier layer.")] = None
    dynamic: Annotated[bool, ParamSpec(
        "If true the layer carries thermal mass (heat-up transient); if "
        "false it conducts quasi-statically.")] = True

    def to_spec(self) -> dict:
        """Serializable value spec (see `hydrogen.serialization`)."""
        return {
            "__type__": "WallLayer",
            "material": self.material.to_spec(),
            "thickness": self.thickness,
            "permeation": (self.permeation.to_spec()
                           if self.permeation is not None else None),
            "dynamic": self.dynamic,
        }

    @classmethod
    def from_spec(cls, d: dict) -> "WallLayer":
        from .permeation import permeation_flux_from_spec

        perm = d.get("permeation")
        return cls(
            material=WallMaterial.from_spec(d["material"]),
            thickness=d["thickness"],
            permeation=(permeation_flux_from_spec(perm) if perm else None),
            dynamic=d.get("dynamic", True),
        )


class Pipe(Model):
    """A flowing pipe wrapped in a per-segment stack of cylindrical wall layers.

    Internally builds a `flow.StraightPipe(heat_port=True)` and, for each of its
    `n_segments` segments, a radial stack of `CylindricalWall`s -- one per
    `WallLayer`.  The walls are coupled conjugately: the innermost layer's inner
    surface exchanges heat (and, if permeable, mass) with the fluid segment; the
    layers are chained radially; the outermost surface is terminated by an
    internal boundary.  Only the fluid `inlet` / `outlet` are exposed::

        fluid:  ===[ segment_i ]===
                      | wall  | leak (if permeable)
                   port_a   leak_a
        metal:   [ wall_i_0 ]--[ wall_i_1 ]-- ... --[ wall_i_{K-1} ]
                                                      | port_b  | leak_b
                                                  outer_i      env_i

    Thermal coupling (always conjugate)
    -----------------------------------
    The fluid pipe is heated (`heat_port=True`); each segment's `wall` port is
    wired to its innermost wall layer.  The outermost layer's outer surface is
    terminated per `outer_thermal`:

      * ``"adiabatic"`` (default) -- `FixedHeatFlow(0)`: perfectly insulated.
      * ``"convective"`` -- `ConvectiveBoundary(h_ext, A_outer, T_ext)`: Newton
        cooling to a far-field at `T_ext` over each segment's outer area.
      * ``"fixed"`` -- `FixedTemperature(T_outer)`: the outer surface is held at
        `T_outer` (use this for a near-isothermal wall when the fluid sits near
        `T_outer`).
      * ``"expose"`` -- no internal termination; each segment's outer node is
        re-exposed as a `wall_outer_{i}` `ThermalPort_TQ` for the parent to wire.

    Gas permeation (optional, per layer)
    ------------------------------------
    Give a `WallLayer` a `permeation` flux model to make it leaky.  The leaky
    layers must form a contiguous stack starting at the bore (layer 0): gas
    permeates from the fluid through each leaky layer in series and vents at the
    last leaky layer's outer surface into an internal `FixedPartialPressure(p_ext)`.
    A non-leaky layer acts as a barrier and must sit outside every leaky one.

    Parameters
    ----------
    medium : CoolPropMedium
        Working fluid.
    D, L, epsilon, z_in, z_out, n_segments : as for `flow.StraightPipe`.
    layers : list[WallLayer]
        Radial wall stack, innermost first.  Inner radius of layer 0 is `D/2`.
    multiphase : str
        Forwarded to the `StraightPipe` (`"single"` / `"HEM"`).
    outer_thermal : str
        Outer-surface thermal termination (see above).
    h_ext, T_ext : float
        Film coefficient / far-field temperature for `outer_thermal="convective"`.
    T_outer : float
        Outer-surface temperature for `outer_thermal="fixed"`.
    p_ext : float
        External partial pressure the outermost leaky layer vents to [Pa].
    T_wall_init : float
        Initial wall temperature [K].
    p_init : float
        Initial bore partial-pressure guess for leaky layers [Pa].
    """

    _OUTER_MODES = _OUTER_THERMAL_MODES

    #: Named predicates referenced by `relevant_when` below, so a UI can gate
    #: list-content-dependent fields without understanding the list internals.
    CONDITIONS = {
        "any_layer_permeable": "at least one WallLayer has a `permeation` model",
    }

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Pipe bore (inner) diameter.",
                    unit="m", default=0.01)],
        L: Annotated[float, ParamSpec("Total pipe length.", unit="m",
                    default=1.0)],
        epsilon: Annotated[float, ParamSpec("Absolute wall roughness "
                          "(friction).", unit="m", default=1e-6)],
        z_in: Annotated[float, ParamSpec("Elevation of the inlet.", unit="m",
                       default=0.0)],
        z_out: Annotated[float, ParamSpec("Elevation of the outlet.",
                        unit="m", default=0.0)],
        n_segments: Annotated[int, ParamSpec("Number of axial finite-volume "
                             "segments.", unit="1", default=1)],
        layers: Annotated[list[WallLayer], ParamSpec("Radial wall stack, "
                         "innermost first; inner radius of layer 0 is D/2.")],
        multiphase: Annotated[str, ParamSpec("Thermodynamic-property mode of "
                             "the flow.", choices=("single", "HEM"))] = "single",
        outer_thermal: Annotated[str, ParamSpec("Thermal termination of the "
                                "outermost wall surface.",
                                choices=_OUTER_THERMAL_MODES)] = "adiabatic",
        h_ext: Annotated[float, ParamSpec("External convection coefficient.",
                        unit="W/(m^2*K)",
                        relevant_when={"outer_thermal": "convective"})] = 10.0,
        T_ext: Annotated[float, ParamSpec("Far-field external temperature.",
                        unit="K",
                        relevant_when={"outer_thermal": "convective"})] = 293.15,
        T_outer: Annotated[float, ParamSpec("Prescribed outer-surface "
                          "temperature.", unit="K",
                          relevant_when={"outer_thermal": "fixed"})] = 293.15,
        p_ext: Annotated[float, ParamSpec("Vent partial pressure at the outer "
                        "leaky surface.", unit="Pa",
                        relevant_when="any_layer_permeable")] = 0.0,
        T_wall_init: Annotated[float, ParamSpec("Initial wall temperature.",
                              unit="K")] = 293.15,
        p_init: Annotated[float, ParamSpec("Initial fluid pressure (and "
                         "leaky-wall inner partial pressure).", unit="Pa")]
                = 101325.0,
    ):
        if outer_thermal not in self._OUTER_MODES:
            raise ValueError(
                f"Pipe: outer_thermal must be one of {self._OUTER_MODES}, "
                f"got {outer_thermal!r}")
        layers = list(layers)
        if not layers:
            raise ValueError("Pipe: `layers` must contain at least one WallLayer.")
        for k, layer in enumerate(layers):
            if layer.thickness <= 0:
                raise ValueError(
                    f"Pipe: layer {k} thickness must be > 0; got {layer.thickness}")
        # Leaky layers must form a contiguous stack from the bore (layer 0).
        leaky = [layer.permeation is not None for layer in layers]
        n_leaky = sum(leaky)
        if leaky[:n_leaky] != [True] * n_leaky or any(leaky[n_leaky:]):
            raise ValueError(
                "Pipe: permeable layers must be contiguous starting at the bore "
                "(layer 0); a non-permeable layer cannot sit between two "
                f"permeable ones. Got permeable flags {leaky}.")

        self.medium = medium
        self.D = D
        self.L = L
        self.epsilon = epsilon
        self.z_in = z_in
        self.z_out = z_out
        self.n_segments = n_segments
        self.layers = layers
        self.multiphase = multiphase
        self.outer_thermal = outer_thermal
        self.h_ext = h_ext
        self.T_ext = T_ext
        self.T_outer = T_outer
        self.p_ext = p_ext
        self.T_wall_init = T_wall_init
        self.p_init = p_init
        self.n_leaky = n_leaky
        self.any_leaky = n_leaky > 0
        # Radial geometry: cumulative radii r[0]=D/2, r[k+1]=r[k]+thickness_k.
        self.L_segment = L / n_segments
        r = [D / 2.0]
        for layer in layers:
            r.append(r[-1] + layer.thickness)
        self.radii = r
        super().__init__()

    def declare_components(self):
        self.add_component('pipe', StraightPipe(
            self.medium, D=self.D, L=self.L, epsilon=self.epsilon,
            z_in=self.z_in, z_out=self.z_out, n_segments=self.n_segments,
            heat_port=True, leaky=self.any_leaky, multiphase=self.multiphase))

        K = len(self.layers)
        r = self.radii
        A_outer = 2.0 * np.pi * r[K] * self.L_segment
        for i in range(self.n_segments):
            for k, layer in enumerate(self.layers):
                self.add_component(f'wall_{i}_{k}', CylindricalWall(
                    layer.material.rho, layer.material.cp, layer.material.k,
                    r[k], r[k + 1], self.L_segment,
                    T_init=self.T_wall_init, dynamic=layer.dynamic,
                    leaky=layer.permeation is not None,
                    permeation_flux=layer.permeation,
                    p_in_init=self.p_init, p_out_init=self.p_ext))

            # Outer-surface thermal termination on the outermost layer.
            if self.outer_thermal == 'adiabatic':
                self.add_component(f'outer_{i}', FixedHeatFlow(0.0, T_init=self.T_wall_init))
            elif self.outer_thermal == 'convective':
                self.add_component(f'outer_{i}', ConvectiveBoundary(self.h_ext, A_outer, T_inf=self.T_ext))
            elif self.outer_thermal == 'fixed':
                self.add_component(f'outer_{i}', FixedTemperature(T_set=self.T_outer))
            else:  # 'expose'
                self.add_port(f'wall_outer_{i}', ThermalPort_TQ(
                    self,
                    channels={'T': self[f'wall_{i}_{K - 1}']['T_b'],
                              'Q_dot': self[f'wall_{i}_{K - 1}']['Q_dot_b']},
                    flow_orientation='in',
                ))

            # Outer-surface permeation vent on the last leaky layer.
            if self.any_leaky:
                self.add_component(f'env_{i}', FixedPartialPressure(p_partial=self.p_ext))

        # Re-expose the fluid inlet/outlet so a Pipe is drop-in for a StraightPipe.
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['pipe']['p_in'], 'h': self['pipe']['h_in'],
                      'm_dot': self['pipe']['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': self['pipe']['p_out'], 'h': self['pipe']['h_out'],
                      'm_dot': self['pipe']['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        K = len(self.layers)
        for i in range(self.n_segments):
            # --- thermal chain: fluid -> layer 0 -> ... -> layer K-1 -> outer ---
            self.connect(self['pipe'].segment_wall_ports[i],
                         self[f'wall_{i}_0'].ports['port_a'])
            for k in range(K - 1):
                self.connect(self[f'wall_{i}_{k}'].ports['port_b'],
                             self[f'wall_{i}_{k + 1}'].ports['port_a'])
            if self.outer_thermal != 'expose':
                self.connect(self[f'wall_{i}_{K - 1}'].ports['port_b'],
                             self[f'outer_{i}'].ports['heat'])

            # --- permeation chain: fluid -> leaky layers in series -> env ---
            if self.any_leaky:
                self.connect(self['pipe'].segment_leak_ports[i],
                             self[f'wall_{i}_0'].ports['leak_a'])
                for k in range(self.n_leaky - 1):
                    self.connect(self[f'wall_{i}_{k}'].ports['leak_b'],
                                 self[f'wall_{i}_{k + 1}'].ports['leak_a'])
                self.connect(self[f'wall_{i}_{self.n_leaky - 1}'].ports['leak_b'],
                             self[f'env_{i}'].ports['leak'])
        return []
