"""Gas-permeation materials, flux models, and boundaries of the `thermofluid` library.

This module supplies the *species-specific* half of wall permeation; the wall
itself (`walls.CylindricalWall(leaky=True)`) and the flowing-pipe leak
(`flow.TwoPortSegment(leaky=True)` / `flow.StraightPipe(leaky=True)` /
`flow.PressureVessel(leaky=True)`) are permeation-agnostic plumbing that only
own ports, surface partial pressures, and leak mass-flows.  What actually makes
a leak a *hydrogen* (or helium, or nitrogen) leak lives here:

  * `Permeant` -- the permeating species: molar mass `M` and the surface-law
    exponent `solubility_exponent` (`n`).  Presets: `H2`, `HELIUM`, `NITROGEN`.
  * `TransportFit` -- the Arrhenius transport of one permeant through one wall
    material: permeability `Phi(T)`, diffusivity `D(T)`, solubility
    `S(T) = Phi/D`.  Presets: `H2_IN_AUSTENITIC`.
  * `PermeationFlux` (+ `SteadyRichardson`, `TransientDiffusion`) -- the
    pressure-gradient -> mass-flow correlation injected into a leaky
    `walls.CylindricalWall` (`leaky=True, permeation_flux=...`).
  * `FixedPartialPressure` -- a boundary that pins a permeation port's partial
    pressure (the permeation analogue of `walls.FixedTemperature`), e.g. for the
    outer (environment) surface of a wall.

Surface law (Sieverts vs Henry).  The dissolved-gas concentration just inside a
surface follows `C = S(T) * p ** (1/n)`:

  * `n = 2` -- **Sieverts' law**, for diatomic gases that dissociate into atoms
    in a dense metal lattice (H2, D2, N2, O2).  The square-root pressure
    dependence is the classic hydrogen-permeation behaviour.
  * `n = 1` -- **Henry's law**, for non-dissociating permeants (e.g. helium
    through a metal/glass, or any gas through a polymer/elastomer membrane).

The exponent is carried by the `Permeant`, so the same flux models cover every
species; only the permeant (and its `TransportFit`) changes.

Permeation physics (diffusion-limited / Richardson regime, cylindrical wall):

    N_dot = 2*pi*Phi(T)*L / ln(r_out/r_in) * (p_in**(1/n) - p_out**(1/n))  [mol/s]
    m_dot = M * N_dot                                       mass leak [kg/s]

`Phi`, `D`, `S = Phi/D` are Arrhenius and come from a `TransportFit`.  `T_film =
(T_a + T_b)/2` (the mean of the two wall-surface temperatures) sets `Phi(T)`,
`D(T)`.

Two flux models are provided:

  * `SteadyRichardson`   -- the algebraic flux above (needs only `Phi`); inner
    uptake == outer venting.
  * `TransientDiffusion` -- a finite-volume radial diffusion chain of `n_nodes`
    dissolved-concentration states with surface-law boundary concentrations;
    captures the wall charge-up / time-lag (needs `D`).  By construction (equal
    `ln` shell spacing, `n+1` series conductances that telescope to
    `2*pi*D*L/ln(r_out/r_in)`) its steady limit reproduces `SteadyRichardson`
    exactly, for any `n_nodes`.

Single-species assumption.  Each permeation network models one permeant.  The
inner partial pressure is supplied by the leak port -- for a pure gas the leaky
flow component publishes its own pressure as the partial pressure, so when a
mixture model lands later only the port wiring (`p_partial = x_i * p`) changes,
not the physics here.

Sign convention -- Modelica "flow into me": a port's `m_dot_leak` is positive
when gas flows INTO that component through the port.  The wall's inner uptake
`m_dot_a_leak` is positive into the wall; at the outer surface gas LEAVES the
wall, so `m_dot_b_leak` is negative and the connected environment boundary
absorbs the positive vented flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

import sympy as sp

from ...model import DifferentialVariable, Model, Parameter, Variable
from ...paramspec import ParamSpec, merged_param_specs
from ..materials import R_GAS
from .ports import PermeationPort_pN


# ---------------------------------------------------------------------------
# Permeant species + (permeant, material) transport fits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Permeant:
    """A permeating gas species.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"H2"``).
    M : float
        Molar mass [kg/mol], used to convert the molar permeation flux to a
        mass flow.
    solubility_exponent : float
        The exponent `n` in the surface law `C = S * p ** (1/n)`:
          * ``2.0`` -- Sieverts' law (diatomic gas dissociating in a metal;
            the classic square-root pressure dependence of H2 permeation),
          * ``1.0`` -- Henry's law (non-dissociating permeant: noble gases in
            metal/glass, or any gas through a polymer membrane).
    """

    name: Annotated[str, ParamSpec("Human-readable permeant label (e.g. "
                   "'H2').")]
    M: Annotated[float, ParamSpec("Molar mass (converts the molar flux to a "
                "mass flow).", unit="kg/mol")]
    solubility_exponent: Annotated[float, ParamSpec(
        "Surface-law exponent n in C = S*p**(1/n): 2 = Sieverts "
        "(dissociating diatomic gas), 1 = Henry (non-dissociating).",
        unit="1")] = 2.0

    def to_spec(self) -> dict:
        """Serializable value spec (see `hydrogen.serialization`)."""
        return {"__type__": "Permeant", "name": self.name, "M": self.M,
                "solubility_exponent": self.solubility_exponent}

    @classmethod
    def from_spec(cls, d: dict) -> "Permeant":
        return cls(name=d["name"], M=d["M"],
                   solubility_exponent=d.get("solubility_exponent", 2.0))


#: Molecular hydrogen -- dissociates in metals (Sieverts, n = 2).
H2 = Permeant(name="H2", M=2.01588e-3, solubility_exponent=2.0)

#: Helium -- monatomic, does not dissociate (Henry, n = 1).
HELIUM = Permeant(name="He", M=4.002602e-3, solubility_exponent=1.0)

#: Molecular nitrogen -- dissociates in metals (Sieverts, n = 2).
NITROGEN = Permeant(name="N2", M=28.0134e-3, solubility_exponent=2.0)

#: Named, ready-made permeants a UI can offer as a choice list (the catalog
#: surfaces these as ``presets``); a UI typically adds a "Custom" entry on top
#: so the user can instead fill `M` / `solubility_exponent` by hand.
Permeant.PRESETS = {
    "H2": H2,
    "He": HELIUM,
    "N2": NITROGEN,
}


@dataclass(frozen=True)
class TransportFit:
    """Arrhenius transport of one `Permeant` through one wall material.

    Permeation through a dense solid in the diffusion-limited (Richardson)
    regime is governed by two independent temperature-dependent properties
    (`Phi = D * S`, so the third is derived):

        Phi(T) = Phi0 * exp(-E_Phi / (R*T))   permeability [mol/(m*s*Pa^(1/n))]
        D(T)   = D0   * exp(-E_D   / (R*T))    diffusivity  [m^2/s]
        S(T)   = Phi(T) / D(T)                 solubility   [mol/(m^3*Pa^(1/n))]

    where `n = permeant.solubility_exponent`.  `Phi` sets the steady leak rate
    and `D` sets the transient diffusion time-lag.

    The `Phi` / `D` / `S` methods accept either a SymPy symbol (returning a
    symbolic expression for `declare_equations`) or a plain float (returning a
    float, for initial guesses and diagnostics).
    """

    permeant: Annotated[Permeant, ParamSpec("The diffusing species (sets the "
                       "molar mass + surface-law exponent).")]
    Phi0: Annotated[float, ParamSpec("Permeability Arrhenius pre-exponential "
                   "factor.", unit="mol/(m*s*Pa^(1/n))")]
    E_Phi: Annotated[float, ParamSpec("Permeability activation energy.",
                    unit="J/mol")]
    D0: Annotated[float, ParamSpec("Diffusivity Arrhenius pre-exponential "
                 "factor.", unit="m^2/s")]
    E_D: Annotated[float, ParamSpec("Diffusivity activation energy.",
                  unit="J/mol")]
    name: Annotated[str, ParamSpec("Optional human-readable label for the "
                   "fit.")] = ""

    @staticmethod
    def _exp(x):
        # math.exp is cheaper and returns a true float on the numeric path;
        # sp.exp keeps symbols symbolic.
        return math.exp(x) if isinstance(x, (int, float)) else sp.exp(x)

    def Phi(self, T):
        """Permeability Phi(T) [mol/(m*s*Pa^(1/n))]."""
        return self.Phi0 * self._exp(-self.E_Phi / (R_GAS * T))

    def D(self, T):
        """Diffusivity D(T) [m^2/s]."""
        return self.D0 * self._exp(-self.E_D / (R_GAS * T))

    def S(self, T):
        """Solubility S(T) = Phi/D [mol/(m^3*Pa^(1/n))]."""
        return self.Phi(T) / self.D(T)

    def to_spec(self) -> dict:
        """Serializable value spec (see `hydrogen.serialization`)."""
        return {"__type__": "TransportFit", "permeant": self.permeant.to_spec(),
                "Phi0": self.Phi0, "E_Phi": self.E_Phi,
                "D0": self.D0, "E_D": self.E_D, "name": self.name}

    @classmethod
    def from_spec(cls, d: dict) -> "TransportFit":
        return cls(permeant=Permeant.from_spec(d["permeant"]),
                   Phi0=d["Phi0"], E_Phi=d["E_Phi"], D0=d["D0"], E_D=d["E_D"],
                   name=d.get("name", ""))


# Sandia-averaged austenitic-SS hydrogen transport ("Technical Reference on
# Hydrogen Compatibility of Materials", San Marchi & Somerday, codes 2101/2103,
# Table 2.1; the averaged austenitic-SS fit, deuterium-corrected to hydrogen).
# Reported in MPa^-0.5; converted to SI Pa^-0.5 (pre-exponential / 1000 =
# (1e6)^0.5).  Permeability/solubility are ~independent of composition across
# stable austenitic steels, so 304 and 316 share this fit; D0 = Phi0/S0 and
# E_D = E_Phi - E_S.  Calibrated ~423-700 K / a few atm; extrapolation to room
# temperature / high pressure is conservative.  Literature scatter is large
# (factor ~5 at high T, up to ~50 near room T), driven by surface oxides and
# cold-work / strain-induced martensite.  Engineering values, not exact
# constants.
H2_IN_AUSTENITIC = TransportFit(
    permeant=H2,
    Phi0=1.2e-7, E_Phi=59.8e3,
    D0=6.7e-7, E_D=53.9e3,
    name="H2 in austenitic SS",
)


# Material-specific AISI 304 / 316 fits calibrated to a colleague's reference
# values AT 150 C (423.15 K):
#     AISI 304:  D = 1.33486e-13 m^2/s,  Phi = 2.26839e-12 mol/(m*s*MPa^0.5)
#     AISI 316:  D = 1.37277e-13 m^2/s,  Phi = 2.19456e-12 mol/(m*s*MPa^0.5)
# Only the pre-exponentials are re-solved to hit those two points; the
# activation energies are kept from `H2_IN_AUSTENITIC` (E_Phi=59.8, E_D=53.9
# kJ/mol), so the temperature *slopes* still follow the Sandia averaged fit and
# the magnitudes match the reference at 150 C.  Phi0 is in SI Pa^-0.5
# (reference MPa^-0.5 / 1000 = (1e6)^0.5).  NOTE: because the source D and Phi
# are independent literature correlations, the implied S = Phi/D (~16-17
# mol/m^3/MPa^0.5) does NOT equal the reference's separately-quoted solubility
# (~37.9) -- the reference triple is not self-consistent (Phi != D*S), so a
# single Arrhenius pair can reproduce at most two of the three.  These fits
# reproduce D and Phi (which set the diffusion time-lag and the steady leak).
H2_IN_AISI_304 = TransportFit(
    permeant=H2,
    Phi0=5.462953e-08, E_Phi=59.8e3,
    D0=6.009657e-07, E_D=53.9e3,
    name="H2 in AISI 304",
)

H2_IN_AISI_316 = TransportFit(
    permeant=H2,
    Phi0=5.285149e-08, E_Phi=59.8e3,
    D0=6.180331e-07, E_D=53.9e3,
    name="H2 in AISI 316",
)

#: Named (permeant, material) transport fits a UI can offer as a choice list.
#: A fit fully defines the transport, *including* its permeant, so selecting a
#: preset fills every field (the alternative is a fully "Custom" fit).
TransportFit.PRESETS = {
    "H2 in austenitic SS": H2_IN_AUSTENITIC,
    "H2 in AISI 304": H2_IN_AISI_304,
    "H2 in AISI 316": H2_IN_AISI_316,
}


# ---------------------------------------------------------------------------
# Permeation flux models (injected into a leaky CylindricalWall)
# ---------------------------------------------------------------------------


class PermeationFlux:
    """Base class for the pressure-gradient -> mass-flow correlation injected
    into a leaky `walls.CylindricalWall` (`leaky=True, permeation_flux=...`).

    A flux model carries the `TransportFit` (and thus the permeant + all the
    transport physics); the wall stays permeation-agnostic and only provides the
    geometry, surface temperatures `T_a`/`T_b`, surface partial pressures
    `p_partial_a`/`p_partial_b`, and the two leak mass-flows `m_dot_a_leak`
    (inner) / `m_dot_b_leak` (outer).  Subclasses implement the two hooks below.

    `T_film = (T_a + T_b)/2` is the temperature used for the Arrhenius
    `Phi(T)`, `D(T)`, `S = Phi/D`.  The surface concentration uses the
    permeant's law `C = S * p ** (1/n)` (Sieverts for `n=2`, Henry for `n=1`).
    """

    #: Hashable identity of the emitted equation structure; keyed into the
    #: wall's `_cache_key_flags` so distinct flux models never share a cache
    #: entry.  Subclasses set this (it always includes the surface-law exponent,
    #: since `p**0.5` vs `p**1` are structurally different residuals).
    cache_key: tuple = ()

    def declare(self, wall):
        """Register any extra components (Arrhenius `Parameter`s, transient
        state variables) the correlation needs, onto `wall`."""
        raise NotImplementedError

    def equations(self, wall):
        """Return the residual equations, binding `wall['m_dot_a_leak']` (inner,
        into the wall) and `wall['m_dot_b_leak']` (outer, into the wall) to the
        computed fluxes."""
        raise NotImplementedError

    # --- shared helpers ---------------------------------------------------

    @property
    def permeant(self) -> Permeant:
        return self.fit.permeant

    def _p_pow(self, p_symbol):
        """Surface concentration pressure factor `p ** (1/n)` for this permeant
        (`sqrt(p)` under Sieverts, linear `p` under Henry)."""
        return p_symbol ** (1.0 / self.permeant.solubility_exponent)

    def _phi0_unit(self) -> str:
        return f"mol/(m.s.Pa^{1.0 / self.permeant.solubility_exponent:g})"

    @staticmethod
    def _T_film(wall):
        return (wall['T_a'].symbol + wall['T_b'].symbol) / 2

    def _declare_phi(self, wall):
        wall.add_component('Phi0', Parameter(self.fit.Phi0, self._phi0_unit()))
        wall.add_component('E_Phi', Parameter(self.fit.E_Phi, "J/mol"))

    def _declare_diffusivity(self, wall):
        wall.add_component('D0', Parameter(self.fit.D0, "m^2/s"))
        wall.add_component('E_D', Parameter(self.fit.E_D, "J/mol"))

    @staticmethod
    def _phi(wall):
        return wall['Phi0'].symbol * sp.exp(-wall['E_Phi'].symbol / (R_GAS * PermeationFlux._T_film(wall)))

    @staticmethod
    def _diffusivity(wall):
        return wall['D0'].symbol * sp.exp(-wall['E_D'].symbol / (R_GAS * PermeationFlux._T_film(wall)))


class SteadyRichardson(PermeationFlux):
    """Algebraic radial permeation flux (the cylindrical analogue of
    `Phi/t * (p_in**(1/n) - p_out**(1/n))`).  Inner uptake == outer venting:

        N_dot = 2*pi*Phi(T)*L / ln(r_out/r_in)
                  * (p_partial_a**(1/n) - p_partial_b**(1/n))
    """

    def __init__(self, transport_fit: Annotated[TransportFit, ParamSpec(
            "Arrhenius transport (permeant + material).")]):
        self.fit = transport_fit
        self.cache_key = ("steady", self.permeant.solubility_exponent)

    def to_spec(self) -> dict:
        return {"__type__": "SteadyRichardson", "transport_fit": self.fit.to_spec()}

    @classmethod
    def from_spec(cls, d: dict) -> "SteadyRichardson":
        return cls(TransportFit.from_spec(d["transport_fit"]))

    def declare(self, wall):
        self._declare_phi(wall)

    def equations(self, wall):
        Phi = self._phi(wall)
        r_in = wall['r_in'].symbol
        r_out = wall['r_out'].symbol
        L = wall['length'].symbol
        c_a = self._p_pow(wall['p_partial_a'].symbol)
        c_b = self._p_pow(wall['p_partial_b'].symbol)
        N_dot = 2 * sp.pi * Phi * L / sp.log(r_out / r_in) * (c_a - c_b)
        M = self.permeant.M
        # Inner surface takes up `+M*N_dot`; the outer surface vents the same,
        # i.e. `-M*N_dot` flows INTO the wall there (it leaves).
        return [
            wall['m_dot_a_leak'].symbol - M * N_dot,
            wall['m_dot_b_leak'].symbol + M * N_dot,
        ]


class TransientDiffusion(PermeationFlux):
    """Finite-volume radial diffusion chain of `n_nodes` dissolved-gas states.

    Nodes 1..n sit at equal-`ln` radii; the `n+1` inter-node/boundary
    conductances are all equal and their series resistance telescopes to
    `ln(r_out/r_in)/(2*pi*D*L)`, so the steady limit recovers the exact
    `SteadyRichardson` flux for any `n_nodes`.  Captures the wall charge-up /
    time lag.
    """

    def __init__(
        self,
        transport_fit: Annotated[TransportFit, ParamSpec("Arrhenius transport "
                       "(permeant + material).")],
        n_nodes: Annotated[int, ParamSpec("Number of radial finite-volume "
                          "diffusion nodes.", unit="1")] = 5,
        C_init: Annotated[float, ParamSpec("Initial dissolved-gas "
                         "concentration in the wall.", unit="mol/m^3")] = 0.0,
    ):
        if int(n_nodes) < 1:
            raise ValueError(f"n_nodes must be >= 1, got {n_nodes!r}")
        self.fit = transport_fit
        self.n_nodes = int(n_nodes)
        self.C_init = C_init
        self.cache_key = ("transient", self.n_nodes, self.permeant.solubility_exponent)

    def to_spec(self) -> dict:
        return {"__type__": "TransientDiffusion", "transport_fit": self.fit.to_spec(),
                "n_nodes": self.n_nodes, "C_init": self.C_init}

    @classmethod
    def from_spec(cls, d: dict) -> "TransientDiffusion":
        return cls(TransportFit.from_spec(d["transport_fit"]),
                   n_nodes=d.get("n_nodes", 5), C_init=d.get("C_init", 0.0))

    def declare(self, wall):
        self._declare_phi(wall)
        self._declare_diffusivity(wall)
        for j in range(1, self.n_nodes + 1):
            wall.add_component(f'C_{j}', DifferentialVariable(self.C_init, "mol/m^3"))

    def equations(self, wall):
        Phi = self._phi(wall)
        D = self._diffusivity(wall)
        S = Phi / D  # solubility constant

        r_in = wall['r_in'].symbol
        r_out = wall['r_out'].symbol
        L = wall['length'].symbol
        ln_ratio = sp.log(r_out / r_in)
        c_a = self._p_pow(wall['p_partial_a'].symbol)
        c_b = self._p_pow(wall['p_partial_b'].symbol)

        n = self.n_nodes
        Delta = ln_ratio / (n + 1)
        g = 2 * sp.pi * D * L / Delta                 # uniform conductance [m^3/s]
        C_in_surf = S * c_a                             # surface-law boundary (inner)
        C_out_surf = S * c_b                            # surface-law boundary (outer)

        C = {j: wall[f'C_{j}'].symbol for j in range(1, n + 1)}
        der = {j: wall[f'der_C_{j}'].symbol for j in range(1, n + 1)}

        def C_at(j):
            if j == 0:
                return C_in_surf
            if j == n + 1:
                return C_out_surf
            return C[j]

        def r_at(x):  # radius at fractional shell index (equal-ln spacing)
            return r_in * sp.exp(x * Delta)

        eqs = []
        for j in range(1, n + 1):
            V_j = sp.pi * (r_at(j + 0.5) ** 2 - r_at(j - 0.5) ** 2) * L
            flux_in = g * (C_at(j - 1) - C[j])
            flux_out = g * (C[j] - C_at(j + 1))
            eqs.append(V_j * der[j] - (flux_in - flux_out))

        # Inner uptake = inner-surface molar flux INTO the wall (`+`); outer
        # venting = outer-surface flux leaving the wall, so `-M*N_out` flows
        # INTO the wall there.
        M = self.permeant.M
        N_in = g * (C_in_surf - C[1])
        N_out = g * (C[n] - C_out_surf)
        eqs.append(wall['m_dot_a_leak'].symbol - M * N_in)
        eqs.append(wall['m_dot_b_leak'].symbol + M * N_out)
        return eqs


# ---------------------------------------------------------------------------
# Partial-pressure boundary
# ---------------------------------------------------------------------------


class FixedPartialPressure(Model):
    """Imposes a fixed partial pressure at its `leak` port; supplies whatever
    leak flow is needed.  The permeation analogue of `walls.FixedTemperature`.

    Pins its own `p_partial` to `p_set` and leaves `m_dot_leak` free, so the
    connected wall surface drives (or absorbs) exactly the permeation flow.
    Use it on a wall's outer surface for venting to a fixed external partial
    pressure (e.g. ~0 Pa to air).

    Port (`leak`):
        p_partial, m_dot_leak   - `p_partial` is pinned to `p_set`; `m_dot_leak`
                                  is set by the connected network.
    """

    def __init__(self, p_partial: Annotated[float, ParamSpec("Partial "
                "pressure of the permeant held at the leak port (e.g. ~0 Pa "
                "to vent to air).", unit="Pa")] = 101325.0):
        self.p_set = p_partial
        super().__init__()

    def declare_components(self):
        # Constructor arg is `p_partial`; the pinned setpoint Parameter pulls
        # its unit/description from that spec.
        self.add_component('p_set', Parameter(self.p_set,
                                              **merged_param_specs(type(self))['p_partial'].param_kwargs()))
        self.add_component('p_partial', Variable(self.p_set, "Pa"))
        self.add_component('m_dot_leak', Variable(0.0, "kg/s"))
        self.add_port('leak', PermeationPort_pN(
            self,
            channels={'p_partial': self['p_partial'], 'm_dot_leak': self['m_dot_leak']},
            flow_orientation='in',
        ))

    def declare_equations(self):
        return [self['p_partial'].symbol - self['p_set'].symbol]


_FLUX_SPECS = {
    "SteadyRichardson": SteadyRichardson,
    "TransientDiffusion": TransientDiffusion,
}


def permeation_flux_from_spec(d: dict) -> PermeationFlux:
    """Rebuild a `PermeationFlux` from its value spec (see `to_spec`)."""
    t = d.get("__type__")
    cls = _FLUX_SPECS.get(t)
    if cls is None:
        raise ValueError(
            f"unknown permeation flux spec type {t!r}; "
            f"known: {sorted(_FLUX_SPECS)}"
        )
    return cls.from_spec(d)


__all__ = [
    "PermeationPort_pN",
    "Permeant",
    "H2",
    "HELIUM",
    "NITROGEN",
    "TransportFit",
    "H2_IN_AUSTENITIC",
    "H2_IN_AISI_304",
    "H2_IN_AISI_316",
    "PermeationFlux",
    "SteadyRichardson",
    "TransientDiffusion",
    "permeation_flux_from_spec",
    "FixedPartialPressure",
]
