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
  * `Tank` -- a lumped-gas pressure vessel whose shell is a cylindrical barrel
    plus a (combined) spherical end cap, both wrapped in `WallLayer` stacks with
    conjugate convective heat transfer and optional gas permeation.  The barrel
    length is solved from the requested internal `volume` / inner `diameter`, so
    the wall geometry matches the control volume.  The fluid `inlet` is the only
    port the user wires.

`Pipe` / `Tank` are the high-level "just give me a pipe / tank" components (the
names are kept deliberately plain so more physics can be folded in later);
`flow.StraightPipe` / `flow.PressureVessel` remain the lower-level primitives
they are built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import sympy as sp

from ...medium import CoolPropMedium
from ...model import Model, Parameter, Variable
from ...paramspec import ParamSpec
from ..materials import WallMaterial
from .flow import PressureVessel, SegmentedChannel, StraightPipe
from .permeation import FixedPartialPressure, PermeationFlux
from .ports import FluidPort_phm, ThermalPort_TQ
from .walls import (
    ConvectiveBoundary,
    CylindricalWall,
    FixedHeatFlow,
    FixedTemperature,
    SphericalWall,
    ThermalConductor,
)

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
    capacity_split: Annotated[str, ParamSpec(
        "How the layer's thermal mass is lumped onto its two surface nodes: "
        "'uniform' (50/50) or 'fem_logmean' (FEM log-mean radius split, more "
        "accurate for thick layers).", choices=("uniform", "fem_logmean"))] = (
        "uniform")

    def to_spec(self) -> dict:
        """Serializable value spec (see `hydrogen.serialization`)."""
        return {
            "__type__": "WallLayer",
            "material": self.material.to_spec(),
            "thickness": self.thickness,
            "permeation": (self.permeation.to_spec()
                           if self.permeation is not None else None),
            "dynamic": self.dynamic,
            "capacity_split": self.capacity_split,
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
            capacity_split=d.get("capacity_split", "uniform"),
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
    count : int
        Number of identical pipes operating in parallel (>= 1).  Only one
        representative pipe is built; every extensive quantity -- flow area and
        wetted perimeter (via the `StraightPipe`'s `count`), each wall layer's
        thermal mass / conduction / permeation (via the `CylindricalWall`'s
        `count`), and the outer convective area -- is scaled by `count`.  The
        intensive states (pressure, enthalpy, temperature, velocity, Reynolds
        number) are identical to a single pipe, so N pipes cost the same
        equations as one while the total mass flow, heat and leak scale by N.
    """

    _OUTER_MODES = _OUTER_THERMAL_MODES

    #: P&ID-style SVG symbol rendered for this component on the UI canvas (a
    #: filename in ``hydrogen/components/icons/``; surfaced via the catalog as
    #: ``"icon"``).  Components without one fall back to the generic box.
    UI_ICON = "pipe.svg"

    #: Named predicates referenced by `relevant_when` below, so a UI can gate
    #: list-content-dependent fields without understanding the list internals.
    CONDITIONS = {
        "any_layer_permeable": "at least one WallLayer has a `permeation` model",
    }

    def __init__(
        self,
        medium: CoolPropMedium,
        D: Annotated[float, ParamSpec("Pipe bore (inner) diameter.",
                    unit="m", default=0.01, ui_label=True)],
        L: Annotated[float, ParamSpec("Total pipe length.", unit="m",
                    default=1.0, ui_label=True)],
        epsilon: Annotated[float, ParamSpec("Absolute wall roughness "
                          "(friction).", unit="m", default=1e-6)],
        z_in: Annotated[float, ParamSpec("Elevation of the inlet.", unit="m",
                       default=0.0)],
        z_out: Annotated[float, ParamSpec("Elevation of the outlet.",
                        unit="m", default=0.0)],
        n_segments: Annotated[int, ParamSpec("Number of axial finite-volume "
                             "segments.", unit="1", default=1,
                             structural=True)],
        layers: Annotated[list[WallLayer], ParamSpec("Radial wall stack, "
                         "innermost first; inner radius of layer 0 is D/2.  "
                         "Empty (no layers) = a bare pipe: the chosen "
                         "`outer_thermal` boundary acts directly on the fluid "
                         "surface and permeation is disabled.")],
        multiphase: Annotated[str, ParamSpec("Thermodynamic-property mode of "
                             "the flow.", choices=("single", "HEM"),
                             structural=True)] = "single",
        outer_thermal: Annotated[str, ParamSpec("Thermal termination of the "
                                "outermost wall surface.",
                                choices=_OUTER_THERMAL_MODES,
                                structural=True)] = "adiabatic",
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
                        relevant_when="any_layer_permeable")] = 1.0,
        T_wall_init: Annotated[float, ParamSpec("Initial wall temperature.",
                              unit="K")] = 293.15,
        p_init: Annotated[float, ParamSpec("Initial fluid pressure (and "
                         "leaky-wall inner partial pressure).", unit="Pa")]
                = 101325.0,
        count: Annotated[int, ParamSpec("Number of identical pipes operating in "
                        "parallel (multiplicity >= 1).  One representative pipe "
                        "is simulated and every extensive quantity (flow area, "
                        "wetted perimeter, wall mass, conductance, film, "
                        "permeation, outer area) is scaled by `count`, so N "
                        "identical pipes cost the same equations as one and "
                        "share the same pressure / temperature / velocity.",
                        unit="1", default=1, ui_label=True)] = 1,
        channel_engine: Annotated[str, ParamSpec("Internal flow discretisation "
                       "engine: 'segmented' (default; a single staggered "
                       "SegmentedChannel, deduplicated by construction -- much "
                       "faster to instantiate) or 'straight' (the legacy "
                       "StraightPipe chain of TwoPortSegments).",
                       choices=("segmented", "straight"), structural=True)]
                       = "segmented",
        dynamic: Annotated[str, ParamSpec("Flow dynamic level (segmented engine "
                       "only): 'static' (quasi-steady, default), 'advective' "
                       "(transient cell energy: conduction + dispersion + "
                       "enthalpy storage), 'compressible' (advective + per-cell "
                       "mass storage with a free cell pressure and staggered "
                       "momentum -- use for a gas / two-phase medium whose "
                       "density changes appreciably in time), or 'acoustic' "
                       "(compressible + transient interior-face momentum "
                       "inertia -- the exact all-regime option, well-conditioned "
                       "even for an incompressible liquid, but the most "
                       "expensive).",
                       choices=("static", "advective", "compressible",
                                "acoustic"),
                       structural=True)] = "static",
        dispersion: Annotated[str, ParamSpec("Axial diffusion/dispersion "
                       "closure for the advective level: 'general' (the default; "
                       "a regime-blended laminar Taylor-Aris / turbulent Taylor "
                       "model built from local properties, valid for any medium "
                       "and all flow regimes -- adds the physical dispersion "
                       "that damps sharp-front oscillations), 'conduction' "
                       "(molecular thermal diffusivity only), 'taylor_aris' "
                       "(laminar shear enhancement (w*Dh)^2/(192*alpha) only), "
                       "'turbulent' (laminar Taylor-Aris below Re~1000 blended "
                       "into Dh*|w|*(1.17e9*Re^-2.5 + 0.41) above), or "
                       "'constant' (impose `D_axial`).  Ignored for "
                       "dynamic='static'.",
                       choices=("general", "conduction", "taylor_aris",
                                "turbulent", "constant"),
                       structural=True)] = "general",
        D_axial: Annotated[float, ParamSpec("Imposed constant effective axial "
                       "diffusivity used when dispersion='constant'.",
                       unit="m^2/s",
                       relevant_when={"dispersion": "constant"})] = 0.0,
        advection_scheme: Annotated[str, ParamSpec("Face-enthalpy "
                       "reconstruction stencil for the advective level "
                       "('U<n_up>D<n_down>', e.g. 'U1D0' first-order upwind, "
                       "'U2D1' the default).", structural=True)] = "U2D1",
        wall_elasticity: Annotated[bool, ParamSpec(
                       "Korteweg hoop compliance of the pipe wall "
                       "(compressible/acoustic levels): lowers the pressure-"
                       "wave speed to the classic elastic-line value -- "
                       "required for quantitative water hammer in real pipes.",
                       structural=True)] = False,
        wall_E: Annotated[float, ParamSpec(
                       "Young's modulus of the structural wall "
                       "(wall_elasticity).", unit="Pa",
                       relevant_when={"wall_elasticity": True})] = 200e9,
        wall_e: Annotated[float | None, ParamSpec(
                       "Structural wall thickness for the Korteweg term; "
                       "None (default) takes the innermost layer's thickness.",
                       unit="m",
                       relevant_when={"wall_elasticity": True})] = None,
        wall_c1: Annotated[float, ParamSpec(
                       "Pipe-anchoring constraint factor in the Korteweg "
                       "formula (1 = expansion joints).",
                       relevant_when={"wall_elasticity": True})] = 1.0,
        unsteady_friction: Annotated[bool, ParamSpec(
                       "TESTING: Brunone unsteady wall friction on the "
                       "acoustic level.", structural=True)] = False,
        k_uf: Annotated[float, ParamSpec(
                       "Brunone unsteady-friction coefficient.",
                       relevant_when={"unsteady_friction": True})] = 0.033,
        viscoelastic_wall: Annotated[bool, ParamSpec(
                       "TESTING: Kelvin-Voigt viscoelastic wall creep "
                       "(polymer pipes; compressible/acoustic levels).",
                       structural=True)] = False,
        J_ve: Annotated[float, ParamSpec(
                       "Kelvin-Voigt hoop-strain creep compliance per Pa "
                       "of gauge pressure.", unit="1/Pa",
                       relevant_when={"viscoelastic_wall": True})] = 0.0,
        tau_ve: Annotated[float, ParamSpec(
                       "Kelvin-Voigt retardation time.", unit="s",
                       relevant_when={"viscoelastic_wall": True})] = 1.0,
        cavitation: Annotated[bool, ParamSpec(
                       "Vapor-cavity (column-separation) handling on the "
                       "acoustic level: per-cell discrete vapor cavities with "
                       "a smoothed complementarity clamp at p_vap (the "
                       "DVCM/DGCM of the water-hammer literature).  Lets a "
                       "water-hammer run continue straight through column "
                       "separation and cavity collapse.",
                       structural=True)] = False,
        p_vap: Annotated[float | None, ParamSpec(
                       "Cavity opening pressure; None (default) evaluates "
                       "the fluid's saturation pressure at T_wall_init via "
                       "CoolProp.", unit="Pa",
                       relevant_when={"cavitation": True})] = None,
        cav_eps: Annotated[float, ParamSpec(
                       "Dimensionless smoothing of the cavity "
                       "complementarity switch (smaller = sharper clamp).",
                       relevant_when={"cavitation": True})] = 1e-2,
    ):
        if channel_engine not in ("straight", "segmented"):
            raise ValueError(
                f"Pipe: channel_engine must be 'straight' or 'segmented', "
                f"got {channel_engine!r}")
        if dynamic != "static" and channel_engine != "segmented":
            raise ValueError(
                f"Pipe: dynamic={dynamic!r} requires channel_engine='segmented' "
                f"(the 'straight' engine is quasi-steady only).")
        if outer_thermal not in self._OUTER_MODES:
            raise ValueError(
                f"Pipe: outer_thermal must be one of {self._OUTER_MODES}, "
                f"got {outer_thermal!r}")
        if int(count) != count or count < 1:
            raise ValueError(
                f"Pipe: count must be an integer >= 1; got {count!r}")
        layers = list(layers)
        # `layers == []` is a *bare* pipe: no wall stack at all.  The segment's
        # internal convective film still carries heat, the chosen `outer_thermal`
        # boundary is applied directly to the fluid surface, and permeation is
        # implicitly disabled (there is no layer that could be permeable).
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
        self.count = int(count)
        self.channel_engine = channel_engine
        self.dynamic = dynamic
        if dispersion not in ("general", "conduction", "taylor_aris",
                              "turbulent", "constant"):
            raise ValueError(
                f"Pipe: dispersion must be 'general', 'conduction', "
                f"'taylor_aris', 'turbulent' or 'constant', got {dispersion!r}")
        if dispersion == "constant" and D_axial < 0.0:
            raise ValueError(
                f"Pipe: D_axial must be >= 0 for dispersion='constant'; "
                f"got {D_axial}")
        self.dispersion = dispersion
        self.D_axial = D_axial
        self.advection_scheme = advection_scheme
        self.wall_elasticity = bool(wall_elasticity)
        self.wall_E = wall_E
        # Default the Korteweg structural thickness to the innermost layer.
        if wall_elasticity and wall_e is None:
            if not layers:
                raise ValueError(
                    "Pipe(wall_elasticity=True): give wall_e explicitly or "
                    "provide at least one wall layer to take its thickness.")
            wall_e = layers[0].thickness
        self.wall_e = wall_e
        self.wall_c1 = wall_c1
        self.unsteady_friction = bool(unsteady_friction)
        self.k_uf = k_uf
        self.viscoelastic_wall = bool(viscoelastic_wall)
        self.J_ve = J_ve
        self.tau_ve = tau_ve
        self.cavitation = bool(cavitation)
        if self.cavitation and p_vap is None:
            # Default the cavity opening pressure to the saturation pressure
            # at the initial (wall = fluid) temperature.
            p_vap = self._saturation_pressure(medium, T_wall_init)
        self.p_vap = p_vap
        self.cav_eps = cav_eps
        self.n_leaky = n_leaky
        self.any_leaky = n_leaky > 0
        # Radial geometry: cumulative radii r[0]=D/2, r[k+1]=r[k]+thickness_k.
        self.L_segment = L / n_segments
        r = [D / 2.0]
        for layer in layers:
            r.append(r[-1] + layer.thickness)
        self.radii = r
        super().__init__()

    @staticmethod
    def _saturation_pressure(medium, T):
        """Saturation pressure [Pa] of `medium`'s fluid at temperature `T`,
        for the `cavitation` default `p_vap`.  Uses CoolProp directly (both
        `CoolPropMedium` and `FeosMedium` carry a CoolProp fluid name)."""
        fluid = getattr(medium, "transport_fluid", None) or getattr(
            medium, "medium", None)
        try:
            import CoolProp.CoolProp as CP
            return float(CP.PropsSI("P", "T", float(T), "Q", 0, str(fluid)))
        except Exception as exc:
            raise ValueError(
                f"Pipe(cavitation=True): could not evaluate the saturation "
                f"pressure of {fluid!r} at T_wall_init={T!r} K via CoolProp "
                f"({exc}); pass p_vap explicitly.") from exc

    def _dispersion_func(self):
        """Build the channel's `dispersion_func` callable from the `dispersion`
        / `D_axial` knobs (returns ``None`` for the conduction-only default so
        the channel keeps its own robust default)."""
        # 'general' (and 'static', which ignores dispersion) -> None, so the
        # channel uses its own robust default (`get_general_dispersion`).
        if self.dynamic == "static" or self.dispersion == "general":
            return None
        if self.dispersion == "conduction":
            return lambda w, Dh, alpha, nu: alpha
        if self.dispersion == "taylor_aris":
            return lambda w, Dh, alpha, nu: alpha + (w * Dh) ** 2 / (192 * alpha)
        if self.dispersion == "turbulent":
            # Legacy IBPSA/ULg correlation, now fed the *local* kinematic
            # viscosity `nu` (so Re is medium-consistent rather than lumped).
            def _disp(w, Dh, alpha, nu):
                Re = sp.Abs(w) * Dh / nu + 0.1
                lam = alpha + (Dh ** 2 / 4.0) * w ** 2 / (48.0 * alpha)
                turb = alpha + Dh * sp.Abs(w) * (1.17e9 * Re ** (-2.5) + 0.41)
                phi = sp.Max(0.0, sp.Min(1.0, (Re - 1000.0) / 1000.0))
                return (1.0 - phi) * lam + phi * turb

            return _disp
        D = self.D_axial                       # 'constant': impose D_eff = D_axial
        return lambda w, Dh, alpha, nu: D

    def declare_components(self):
        # `count` identical pipes in parallel are simulated as one representative
        # pipe with every EXTENSIVE quantity scaled by N: the flow area / wetted
        # perimeter (via StraightPipe's `count`), the per-segment wall mass /
        # conductance / permeation (via the walls' `count`), and the outer
        # convective area (via the boundary's `count`).  Intensive states
        # (p, h, T, velocity) are N-invariant.
        #
        # ONE shared `count` Parameter is owned here and aliased into every
        # sub-component, so the multiplicity is a single live knob: `set_param`
        # on `count` retunes all of them at once with NO re-instantiation.
        self.add_component('count', Parameter(float(self.count), unit="1",
                           description="Number of identical parallel pipes."))
        N = self['count']

        if self.channel_engine == "segmented":
            self.add_component('pipe', SegmentedChannel(
                self.medium, D=self.D, L=self.L, epsilon=self.epsilon,
                z_in=self.z_in, z_out=self.z_out, N=self.n_segments,
                heat_port=True, leaky=self.any_leaky, multiphase=self.multiphase,
                dynamic=self.dynamic, dispersion_func=self._dispersion_func(),
                advection_scheme=self.advection_scheme, p_init=self.p_init,
                wall_elasticity=self.wall_elasticity, wall_E=self.wall_E,
                wall_e=(self.wall_e if self.wall_e is not None else 0.002),
                wall_c1=self.wall_c1,
                unsteady_friction=self.unsteady_friction, k_uf=self.k_uf,
                viscoelastic_wall=self.viscoelastic_wall, J_ve=self.J_ve,
                tau_ve=self.tau_ve,
                cavitation=self.cavitation, p_vap=self.p_vap,
                cav_eps=self.cav_eps,
                count=N))
        else:
            self.add_component('pipe', StraightPipe(
                self.medium, D=self.D, L=self.L, epsilon=self.epsilon,
                z_in=self.z_in, z_out=self.z_out, n_segments=self.n_segments,
                heat_port=True, leaky=self.any_leaky, multiphase=self.multiphase,
                count=N))

        K = len(self.layers)
        r = self.radii
        # Per-pipe outer area; the `count` multiplicity is applied inside the
        # convective boundary (shared Parameter), not baked in here.  For a bare
        # pipe (K == 0) r[K] == r[0] == D/2, so this is the bore surface area --
        # the boundary acts straight on the fluid's inner wall.
        A_outer = 2.0 * np.pi * r[K] * self.L_segment

        def _surface(i):
            """(T, Q_dot) variables of the outermost solid surface for segment
            `i`: the last layer's outer face, or -- for a bare pipe -- the fluid
            segment's own inner-wall node."""
            if K > 0:
                w = self[f'wall_{i}_{K - 1}']
                return w['T_b'], w['Q_dot_b']
            if self.channel_engine == "segmented":
                return self['pipe'][f'T_wall_{i}'], self['pipe'][f'q_inflow_{i}']
            seg = self['pipe'][f'pipe_segment_{i}']
            return seg['T_wall'], seg['q_inflow']

        for i in range(self.n_segments):
            for k, layer in enumerate(self.layers):
                self.add_component(f'wall_{i}_{k}', CylindricalWall(
                    layer.material.rho, layer.material.cp, layer.material.k,
                    r[k], r[k + 1], self.L_segment,
                    T_init=self.T_wall_init, dynamic=layer.dynamic,
                    capacity_split=layer.capacity_split,
                    count=N,
                    leaky=layer.permeation is not None,
                    permeation_flux=layer.permeation,
                    p_in_init=self.p_init, p_out_init=self.p_ext))

            # Outer-surface thermal termination (on the outermost layer, or
            # directly on the fluid segment when there are no layers).
            if self.outer_thermal == 'adiabatic':
                self.add_component(f'outer_{i}', FixedHeatFlow(0.0, T_init=self.T_wall_init))
            elif self.outer_thermal == 'convective':
                self.add_component(f'outer_{i}', ConvectiveBoundary(self.h_ext, A_outer, T_inf=self.T_ext, count=N))
            elif self.outer_thermal == 'fixed':
                self.add_component(f'outer_{i}', FixedTemperature(T_set=self.T_outer))
            else:  # 'expose'
                if K == 0:
                    # The exposed port aliases the segment's own (T_wall,
                    # q_inflow); the caller closes them by wiring `wall_outer_{i}`
                    # to a boundary, so the segment's require_connection warning
                    # would be misleading -- silence it.
                    if self.channel_engine == "segmented":
                        self['pipe'].ports[f'wall_{i}'].require_connection = False
                    else:
                        self['pipe'][f'pipe_segment_{i}'].ports['wall'].require_connection = False
                T_s, Q_s = _surface(i)
                self.add_port(f'wall_outer_{i}', ThermalPort_TQ(
                    self,
                    channels={'T': T_s, 'Q_dot': Q_s},
                    flow_orientation='in',
                ))

            # Outer-surface permeation vent on the last leaky layer.
            if self.any_leaky:
                self.add_component(f'env_{i}', FixedPartialPressure(p_partial=self.p_ext))

        # Total permeation leak to the environment: the sum of every segment's
        # outer-surface vent flow (positive = mass leaving the pipe into the
        # environment).  Already includes the `count` multiplicity, since each
        # wall's permeation is scaled by N.  Only meaningful when permeable.
        if self.any_leaky:
            self.add_component('m_dot_leak_env', Variable(0.0, "kg/s"))

        # Re-expose the fluid inlet/outlet so a Pipe is drop-in for a StraightPipe.
        # The two engines name their boundary face variables differently:
        # StraightPipe uses pipe-level `p_in`/`p_out`; SegmentedChannel uses the
        # shared face vars `p_0`/`p_{N}` (m_dot still via `m_dot_in`/`m_dot_out`).
        if self.channel_engine == "segmented":
            in_p, in_h = self['pipe']['p_0'], self['pipe']['h_0']
            out_p = self['pipe'][f'p_{self.n_segments}']
            out_h = self['pipe'][f'h_{self.n_segments}']
        else:
            in_p, in_h = self['pipe']['p_in'], self['pipe']['h_in']
            out_p, out_h = self['pipe']['p_out'], self['pipe']['h_out']
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': in_p, 'h': in_h, 'm_dot': self['pipe']['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))
        self.add_port('outlet', FluidPort_phm(
            self,
            channels={'p': out_p, 'h': out_h, 'm_dot': self['pipe']['m_dot_out']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        K = len(self.layers)
        for i in range(self.n_segments):
            seg_wall = self['pipe'].segment_wall_ports[i]

            if K == 0:
                # Bare pipe: the boundary acts on the fluid segment directly (its
                # internal convective film is the only heat resistance).  'expose'
                # is served by the re-exposed `wall_outer_{i}` port declared
                # above, so there is nothing to wire internally.  No layers means
                # no permeation either, so we are done with this segment.
                if self.outer_thermal != 'expose':
                    self.connect(seg_wall, self[f'outer_{i}'].ports['heat'])
                continue

            # --- thermal chain: fluid -> layer 0 -> ... -> layer K-1 -> outer ---
            self.connect(seg_wall, self[f'wall_{i}_0'].ports['port_a'])
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

        eqs = []
        if self.any_leaky:
            total = sum(self[f'env_{i}']['m_dot_leak'].symbol
                        for i in range(self.n_segments))
            eqs.append(self['m_dot_leak_env'].symbol + total)
        return eqs


class Tank(Model):
    """A lumped-gas pressure vessel: a cylindrical barrel + a spherical end cap,
    each wrapped in a `WallLayer` stack, with conjugate heat and gas permeation.

    The fluid side is a single `flow.PressureVessel` control volume (mass +
    energy storage, fills through `inlet`).  Its shell is split into the two
    shapes a real tank is made of:

      * a **cylindrical barrel** (`CylindricalWall` stack), and
      * **spherical end caps**, modelled as ONE full `SphericalWall`
        (`angle_fraction=1`) -- two hemispheres in parallel are thermally /
        permeation-wise identical to a single full shell, which keeps the
        equation count down.

    Geometry matching
    -----------------
    You specify the internal `volume` `V` and inner `diameter` `D`; the barrel
    length is solved so the gas control volume equals the requested `V`::

        r = D / 2
        V_caps = 4/3 * pi * r**3              (the two hemispherical caps)
        L_cyl  = (V - V_caps) / (pi * r**2)   (barrel length)

    so `V = pi*r**2*L_cyl + 4/3*pi*r**3` exactly.  `V` must exceed the cap
    volume `V_caps` (otherwise the caps alone already overflow the tank for the
    given `D`).  The `PressureVessel` is created with this same `V`.

    Conjugate heat (always)
    -----------------------
    The gas exposes one conjugate `heat_{k}` port per wall (the framework's
    `connect()` is pairwise, so the barrel and the cap each need their own).
    Each is wired to its wall's inner surface through a massless
    `ThermalConductor` film of conductance `G = h_inner * A_inner`
    (`A_inner = 2*pi*r*L_cyl` for the barrel, `4*pi*r**2` for the caps), so the
    wall surface sits below the gas temperature by the film drop and the heat
    feeds back into the gas energy balance.  The outermost wall surface is
    terminated per `outer_thermal` (`adiabatic` / `convective` / `fixed` /
    `expose`), exactly as for `Pipe`.

    Gas permeation (optional, per layer)
    ------------------------------------
    Give a `WallLayer` a `permeation` flux model (same `TransportFit` machinery
    `Pipe` uses) to make it leaky.  Leaky layers must form a contiguous stack
    from the bore (layer 0).  Gas permeates from the vessel through each leaky
    layer in series and vents at the last leaky layer's outer surface into an
    internal `FixedPartialPressure(p_ext)`.  The barrel and the cap each get
    their own permeation chain, both driven by the vessel pressure and both
    feeding mass / energy back into the `PressureVessel` balance.

    Multiple identical tanks (`count`)
    ----------------------------------
    `count = N` simulates `N` identical tanks running in parallel (same inlet
    manifold, same boundary conditions) at the cost of ONE tank's equations.
    Every EXTENSIVE quantity is scaled by `N` -- the stored gas volume, the wall
    thermal mass, the thermal/permeation conductances, the inner films, and the
    outer exchange areas -- while the geometry (`volume` / `diameter`) still
    describes a SINGLE tank.  Because capacities and conductances scale together,
    all time constants are unchanged and the intensive states (`p`, `T`, surface
    temperatures, partial pressures) are identical to a single tank; only the
    absolute mass / heat / leak flows are `N` times larger.  This is exact for
    identical tanks (it cannot capture tank-to-tank variation or independent
    valving -- use separate `Tank`s for that).

    Parameters
    ----------
    medium : CoolPropMedium
        Stored gas.
    volume : float
        Internal (gas) control volume of a SINGLE tank [m^3].
    diameter : float
        Inner diameter of the barrel and caps [m]; inner radius of layer 0 is
        `diameter/2`.
    layers : list[WallLayer]
        Radial wall stack, innermost first (shared by barrel and caps).
    h_inner : float
        Internal convective film coefficient (gas <-> wall inner surface).
    count : int
        Number of identical parallel tanks (>= 1); see above.
    outer_thermal : str
        Outer-surface thermal termination (see `Pipe`).
    h_ext, T_ext : float
        Film coefficient / far-field temperature for `outer_thermal="convective"`.
    T_outer : float
        Outer-surface temperature for `outer_thermal="fixed"`.
    p_ext : float
        External partial pressure the outermost leaky layer vents to [Pa].
    T_wall_init : float
        Initial wall temperature [K].
    p_init, T_init : float
        Initial vessel pressure [Pa] / temperature [K].
    """

    _OUTER_MODES = _OUTER_THERMAL_MODES

    #: P&ID-style SVG symbol for the UI canvas (file in
    #: ``hydrogen/components/icons/``; surfaced via the catalog as ``"icon"``).
    UI_ICON = "pressure_vessel.svg"

    #: Named predicates referenced by `relevant_when` (see `Pipe`).
    CONDITIONS = {
        "any_layer_permeable": "at least one WallLayer has a `permeation` model",
    }

    #: The two shell shapes, in the order their conjugate `heat_{k}` ports are
    #: assigned on the gas volume (barrel -> port 0, caps -> port 1).
    _SHAPES = ("cyl", "sph")

    def __init__(
        self,
        medium: CoolPropMedium,
        volume: Annotated[float, ParamSpec("Internal (gas) control volume of "
                         "the tank.", unit="m^3", default=0.05,
                         ui_label=True)],
        diameter: Annotated[float, ParamSpec("Inner diameter of the barrel / "
                           "caps.", unit="m", default=0.3, ui_label=True)],
        layers: Annotated[list[WallLayer], ParamSpec("Radial wall stack, "
                         "innermost first; inner radius of layer 0 is "
                         "diameter/2.  Empty (no layers) = a bare tank: the "
                         "chosen `outer_thermal` boundary acts directly on the "
                         "inner film's outer face and permeation is disabled.")],
        h_inner: Annotated[float, ParamSpec("Internal convective film "
                          "coefficient (gas <-> wall inner surface).",
                          unit="W/(m^2*K)", default=50.0)],
        count: Annotated[int, ParamSpec("Number of identical tanks operating in "
                        "parallel (multiplicity >= 1).  One representative tank "
                        "is simulated and every extensive quantity (stored gas, "
                        "wall mass, conductance, film, permeation) is scaled by "
                        "`count`, so N identical tanks cost the same equations "
                        "as one and share the same pressure / temperature.",
                        unit="1", default=1, ui_label=True)] = 1,
        outer_thermal: Annotated[str, ParamSpec("Thermal termination of the "
                                "outermost wall surface.",
                                choices=_OUTER_THERMAL_MODES,
                                structural=True)] = "adiabatic",
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
                        relevant_when="any_layer_permeable")] = 1.0,
        T_wall_init: Annotated[float, ParamSpec("Initial wall temperature.",
                              unit="K")] = 293.15,
        p_init: Annotated[float, ParamSpec("Initial vessel pressure (and "
                         "leaky-wall inner partial pressure).", unit="Pa")]
                = 101325.0,
        T_init: Annotated[float, ParamSpec("Initial vessel temperature.",
                         unit="K")] = 293.15,
    ):
        if outer_thermal not in self._OUTER_MODES:
            raise ValueError(
                f"Tank: outer_thermal must be one of {self._OUTER_MODES}, "
                f"got {outer_thermal!r}")
        if diameter <= 0:
            raise ValueError(f"Tank: diameter must be > 0; got {diameter}")
        if int(count) != count or count < 1:
            raise ValueError(
                f"Tank: count must be an integer >= 1; got {count!r}")
        layers = list(layers)
        # `layers == []` is a *bare* tank: no wall stack on either shell.  The
        # inner convective film still carries heat, the chosen `outer_thermal`
        # boundary is applied directly to the film's outer face, and permeation
        # is implicitly disabled (there is no layer that could be permeable).
        for k, layer in enumerate(layers):
            if layer.thickness <= 0:
                raise ValueError(
                    f"Tank: layer {k} thickness must be > 0; got {layer.thickness}")
        # Leaky layers must form a contiguous stack from the bore (layer 0).
        leaky = [layer.permeation is not None for layer in layers]
        n_leaky = sum(leaky)
        if leaky[:n_leaky] != [True] * n_leaky or any(leaky[n_leaky:]):
            raise ValueError(
                "Tank: permeable layers must be contiguous starting at the bore "
                "(layer 0); a non-permeable layer cannot sit between two "
                f"permeable ones. Got permeable flags {leaky}.")

        # --- geometry: solve the barrel length so the gas volume == `volume` ---
        r_in0 = diameter / 2.0
        V_caps = (4.0 / 3.0) * np.pi * r_in0 ** 3
        if volume <= V_caps:
            raise ValueError(
                f"Tank: volume ({volume} m^3) must exceed the spherical-cap "
                f"volume ({V_caps:.6g} m^3) for inner diameter {diameter} m "
                f"(the caps alone already fill the tank). Increase volume or "
                f"decrease diameter.")
        L_cyl = (volume - V_caps) / (np.pi * r_in0 ** 2)

        self.medium = medium
        self.volume = volume
        self.diameter = diameter
        self.layers = layers
        self.h_inner = h_inner
        self.count = int(count)
        self.outer_thermal = outer_thermal
        self.h_ext = h_ext
        self.T_ext = T_ext
        self.T_outer = T_outer
        self.p_ext = p_ext
        self.T_wall_init = T_wall_init
        self.p_init = p_init
        self.T_init = T_init
        self.n_leaky = n_leaky
        self.any_leaky = n_leaky > 0
        self.L_cyl = L_cyl
        # Cumulative radii r[0]=D/2, r[k+1]=r[k]+thickness_k (shared by both shapes).
        r = [r_in0]
        for layer in layers:
            r.append(r[-1] + layer.thickness)
        self.radii = r
        super().__init__()

    def _make_wall(self, shape, k, r_in, r_out):
        """Build the radial-layer-`k` wall component for `shape` ('cyl'/'sph')."""
        layer = self.layers[k]
        common = dict(
            T_init=self.T_wall_init, dynamic=layer.dynamic,
            angle_fraction=1.0, count=self['count'],
            leaky=layer.permeation is not None,
            permeation_flux=layer.permeation,
            p_in_init=self.p_init, p_out_init=self.p_ext,
        )
        if shape == "cyl":
            return CylindricalWall(
                layer.material.rho, layer.material.cp, layer.material.k,
                r_in, r_out, self.L_cyl, **common)
        return SphericalWall(
            layer.material.rho, layer.material.cp, layer.material.k,
            r_in, r_out, **common)

    def declare_components(self):
        r = self.radii
        K = len(self.layers)

        # `count` identical tanks in parallel are simulated as one representative
        # tank with every EXTENSIVE quantity scaled by N: the stored gas volume
        # (via the vessel's `count`), the wall mass / conductance / permeation
        # (via the walls' `count`), the inner films (via the conductor's `count`)
        # and the outer exchange areas (via the boundary's `count`).  Intensive
        # states (p, h, T, partial pressures) are N-invariant.
        #
        # ONE shared `count` Parameter is owned here and aliased into every
        # sub-component, so the multiplicity is a single live knob: `set_param`
        # on `count` retunes all of them at once with NO re-instantiation.
        self.add_component('count', Parameter(float(self.count), unit="1",
                           description="Number of identical parallel tanks."))
        N = self['count']

        # Single lumped-gas control volume (N tanks' worth, applied inside the
        # vessel via the shared `count`); one conjugate heat port per shell
        # shape, plus one permeation port per shape when leaky.
        self.add_component('gas', PressureVessel(
            self.medium, V=self.volume, A_in=np.pi * r[0] ** 2,
            p_init=self.p_init, T_init=self.T_init,
            leaky=False,
            heat_ports=len(self._SHAPES),
            leak_ports=len(self._SHAPES) if self.any_leaky else 0,
            count=N))

        # Inner-surface convective-film conductance (per-tank G = h_inner *
        # A_inner; the N multiplicity is applied inside the conductor).
        A_in_cyl = 2.0 * np.pi * r[0] * self.L_cyl
        A_in_sph = 4.0 * np.pi * r[0] ** 2
        self.add_component('film_cyl', ThermalConductor(
            self.h_inner * A_in_cyl, T_init=self.T_init, count=N))
        self.add_component('film_sph', ThermalConductor(
            self.h_inner * A_in_sph, T_init=self.T_init, count=N))

        # Per-tank outer-surface areas; for a bare tank (K == 0) r[K] == r[0] is
        # the inner radius, so these equal the inner-film areas -- the boundary
        # acts straight on the film's outer face.  The N multiplicity is applied
        # inside the convective boundary.
        A_out = {'cyl': 2.0 * np.pi * r[K] * self.L_cyl,
                 'sph': 4.0 * np.pi * r[K] ** 2}

        def _surface(shape):
            """(T, Q_dot) variables of the outermost solid surface for `shape`:
            the last layer's outer face, or -- for a bare tank -- the inner
            film's outer face (the wetted surface itself)."""
            if K > 0:
                w = self[f'wall_{shape}_{K - 1}']
                return w['T_b'], w['Q_dot_b']
            f = self[f'film_{shape}']
            return f['T_b'], f['Q_dot_b']

        for shape in self._SHAPES:
            for k in range(K):
                self.add_component(f'wall_{shape}_{k}',
                                   self._make_wall(shape, k, r[k], r[k + 1]))

            # Outer-surface thermal termination (on the outermost layer, or
            # directly on the inner film's outer face when there are no layers).
            if self.outer_thermal == 'adiabatic':
                self.add_component(f'outer_{shape}',
                                   FixedHeatFlow(0.0, T_init=self.T_wall_init))
            elif self.outer_thermal == 'convective':
                self.add_component(f'outer_{shape}', ConvectiveBoundary(
                    self.h_ext, A_out[shape], T_inf=self.T_ext, count=N))
            elif self.outer_thermal == 'fixed':
                self.add_component(f'outer_{shape}',
                                   FixedTemperature(T_set=self.T_outer))
            else:  # 'expose'
                T_s, Q_s = _surface(shape)
                self.add_port(f'wall_outer_{shape}', ThermalPort_TQ(
                    self,
                    channels={'T': T_s, 'Q_dot': Q_s},
                    flow_orientation='in',
                ))

            # Outer-surface permeation vent on the last leaky layer.
            if self.any_leaky:
                self.add_component(f'env_{shape}',
                                   FixedPartialPressure(p_partial=self.p_ext))

        # Total permeation leak to the environment: the sum of the barrel and
        # cap vent flows (positive = mass leaving the tank into the
        # environment).  Already includes the `count` multiplicity, since each
        # wall's permeation is scaled by N.  Only meaningful when permeable.
        if self.any_leaky:
            self.add_component('m_dot_leak_env', Variable(0.0, "kg/s"))

        # Re-expose the vessel inlet as the tank's single fluid port.
        self.add_port('inlet', FluidPort_phm(
            self,
            channels={'p': self['gas']['p_in'], 'h': self['gas']['h_in'],
                      'm_dot': self['gas']['m_dot_in']},
            flow_orientation='in',
            medium=self.medium,
        ))

    def declare_equations(self):
        K = len(self.layers)
        films = {'cyl': 'film_cyl', 'sph': 'film_sph'}
        for j, shape in enumerate(self._SHAPES):
            # --- conjugate heat: gas -[film]-> layer 0 -> ... -> outer ---
            self.connect(self['gas'].ports[f'heat_{j}'],
                         self[films[shape]].ports['heat_a'])

            if K == 0:
                # Bare tank: the inner film's outer face IS the wetted surface;
                # apply the boundary to it directly (or, for 'expose', it is
                # re-exposed as wall_outer_{shape}).  No layers => no permeation,
                # so this shell is done.
                if self.outer_thermal != 'expose':
                    self.connect(self[films[shape]].ports['heat_b'],
                                 self[f'outer_{shape}'].ports['heat'])
                continue

            self.connect(self[films[shape]].ports['heat_b'],
                         self[f'wall_{shape}_0'].ports['port_a'])
            for k in range(K - 1):
                self.connect(self[f'wall_{shape}_{k}'].ports['port_b'],
                             self[f'wall_{shape}_{k + 1}'].ports['port_a'])
            if self.outer_thermal != 'expose':
                self.connect(self[f'wall_{shape}_{K - 1}'].ports['port_b'],
                             self[f'outer_{shape}'].ports['heat'])

            # --- permeation: gas -> leaky layers in series -> env ---
            if self.any_leaky:
                self.connect(self['gas'].ports[f'leak_{j}'],
                             self[f'wall_{shape}_0'].ports['leak_a'])
                for k in range(self.n_leaky - 1):
                    self.connect(self[f'wall_{shape}_{k}'].ports['leak_b'],
                                 self[f'wall_{shape}_{k + 1}'].ports['leak_a'])
                self.connect(self[f'wall_{shape}_{self.n_leaky - 1}'].ports['leak_b'],
                             self[f'env_{shape}'].ports['leak'])

        eqs = []
        if self.any_leaky:
            total = sum(self[f'env_{shape}']['m_dot_leak'].symbol
                        for shape in self._SHAPES)
            eqs.append(self['m_dot_leak_env'].symbol + total)
        return eqs
