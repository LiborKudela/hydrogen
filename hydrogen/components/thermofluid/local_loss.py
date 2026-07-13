"""Pluggable local- (minor-) pressure-loss correlations for the `thermofluid`
library.

A *local* (minor) loss is the pressure drop a fitting adds beyond wall friction:
a bend, a sudden area change, an orifice, a diffuser, a tee, ...  It is quoted
as a dimensionless loss coefficient ``K`` (``zeta``) referenced to a velocity
head::

    Δp = K * rho * v**2 / 2

where ``v = m_dot / (rho * A)`` is the reference velocity at the component's
connecting bore area ``A = pi*D**2/4``.  Inverting the (regularised) law gives
the mass flow the `flow.LocalResistance` cell / `assemblies.LocalLoss` assembly
solve::

    m_dot = A * sign(Δp) * sqrt(2 * rho * |Δp| / K)

so a single sqrt law is shared across the static and dynamic body levels (it is
linear in the flow coefficient, exactly like the non-choked valve law, so it
composes per staggered face on the dynamic levels).

The coefficient itself is supplied by a *pluggable* `LocalLossModel`: a small
value object -- exactly the same pattern as a `permeation.PermeationFlux` -- that
a UI offers as a dropdown, auto-prompting for the fields of whichever
correlation is chosen.  A model implements four hooks:

  * ``declare(comp)``   -- register its (live, tunable) `Parameter`s onto the
    consuming component;
  * ``zeta(comp)``      -- return the symbolic dimensionless coefficient ``K``
    (referenced to the bore velocity), built from those parameters;
  * ``to_spec`` / ``from_spec`` -- (de)serialize the value object.

``cache_key`` carries the *structural* identity of the emitted coefficient
expression (which correlation) so the equation-template cache never mixes two
different laws; the numeric parameter values stay live `Parameter` symbols and
so never enter the key (retuning ``K`` / an area ratio needs no rebuild).

Correlations that ship to seed the interface (Idelchik, *Handbook of Hydraulic
Resistance*):

  * `FixedK`            -- a user-supplied constant ``K`` (the catalog default);
  * `SuddenExpansion`   -- Borda-Carnot ``K = (1 - beta)**2`` for an abrupt
    enlargement (``beta = A_bore / A_large`` <= 1, referenced to the smaller
    upstream = bore velocity);
  * `SuddenContraction` -- Idelchik ``K = 0.5 * (1 - beta)**0.75`` for an abrupt
    contraction (``beta = A_bore / A_large`` <= 1, referenced to the smaller
    downstream = bore velocity);
  * `LaminarTransitionK` -- a *Reynolds-dependent* example,
    ``K = K_turb + K_lam/Re``, showing how a velocity-dependent correlation
    reads `LossFlowState.Re`.

More correlations (bends, orifices, diffusers, tees, further Re-dependent forms)
drop in as further `LocalLossModel` subclasses registered in
`serialization.values._value_classes`; no consumer change is needed.

Reference section & area change.  A model returns the dimensionless ``K``; the
*consuming component* decides which velocity head it multiplies.  For the
default equal-area `flow.LocalResistance` (single bore ``A = pi*D**2/4``, inlet
velocity == outlet velocity) that is simply the bore velocity head.  A true
area-changing fitting (conical contraction / enlargement / diffuser, where the
two ends have different areas and velocities and the reversible Bernoulli
pressure recovery matters) is built by giving `flow.LocalResistance` distinct
``D_in`` / ``D_out``: it then rides the variable-area `flow.SegmentedChannel`,
so the momentum balance is the reversible area-change relation PLUS this
irreversible ``K`` loss, and ``reference='inlet'|'outlet'`` picks which end's
velocity head ``K`` is quoted against.

Velocity- / Reynolds-dependent ``K``.  A correlation may depend on the flow:
its `zeta` receives a `LossFlowState` (the local ``Re`` / velocity / density /
viscosity / hydraulic diameter / area), so an Idelchik laminar / transition
``K = f(Re)`` closes.  ``Re`` is floored (``+1``) so ``1/Re`` is finite through
zero flow; a velocity-dependent ``K`` just makes the momentum residual implicit
in the flow, which the DAE solver handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import sympy as sp

from ...model import Parameter
from ...paramspec import ParamSpec


@dataclass(frozen=True)
class LossFlowState:
    """The local (symbolic) flow state handed to `LocalLossModel.zeta`, so a
    correlation can be velocity- / Reynolds-dependent (many Idelchik local
    resistances have a laminar / transition ``K = f(Re)`` regime).

    All fields are sympy expressions in the consuming component's leaf symbols,
    evaluated at the point the coefficient is applied: the single cell's
    average state on the ``static`` level, or the individual staggered face on
    the dynamic levels.  ``Re`` is already floored (``+1``) so it is safe in a
    denominator through zero flow.

      * ``Re``  -- Reynolds number ``rho*|w|*Dh/mu + 1`` at the reference bore.
      * ``w``   -- reference (bore) velocity [m/s] (signed).
      * ``rho`` -- density [kg/m^3].
      * ``mu``  -- dynamic viscosity [Pa*s].
      * ``Dh``  -- hydraulic (bore) diameter [m].
      * ``A``   -- reference bore area [m^2].

    Constant-``K`` correlations simply ignore it.
    """

    Re: object
    w: object
    rho: object
    mu: object
    Dh: object
    A: object


class LocalLossModel:
    """Base class for a pluggable local (minor) pressure-loss correlation.

    A model supplies the dimensionless loss coefficient ``K`` (``zeta``)
    referenced to the consuming component's bore velocity
    (``v = m_dot/(rho*A)``, ``A = pi*D**2/4``), so ``Δp = K * rho * v**2 / 2``.
    Subclasses register their (live) parameters via `declare` and build ``K``
    from them in `zeta`.

    ``K`` may be constant (geometry only) or a function of the local flow: the
    `LossFlowState` passed to `zeta` carries the symbolic ``Re`` / velocity /
    density / viscosity so a Reynolds-dependent (laminar / transition)
    correlation can express ``K = f(Re)``.  A velocity-dependent ``K`` makes the
    momentum residual implicit in the flow (``K`` sits inside the sqrt
    inversion); the DAE solver already handles that -- keep any ``Re`` in a
    denominator (it is floored in `LossFlowState`).
    """

    #: Hashable identity of the emitted coefficient expression (which
    #: correlation + any baked-in structural exponents), keyed into the
    #: consuming component's `_cache_key_flags` so distinct correlations never
    #: share an equation-template cache entry.  It deliberately excludes the
    #: numeric parameter values -- those are live `Parameter` symbols.
    cache_key: tuple = ()

    #: The correlation a UI pre-fills when adding a local loss (an explicit
    #: opt-in so the default never depends on catalog ordering -- see
    #: `serialization.registry._concrete_value_type`).
    _catalog_default: bool = False

    def declare(self, comp):
        """Register any (live, tunable) `Parameter`s the correlation needs onto
        the consuming component ``comp``."""
        raise NotImplementedError

    def zeta(self, comp, flow: "LossFlowState | None" = None):
        """Return the symbolic dimensionless loss coefficient ``K`` (referenced
        to the bore velocity), built from the parameters `declare` registered
        and, for a velocity/Reynolds-dependent correlation, the local
        `LossFlowState` ``flow`` (``flow.Re`` etc.); constant models ignore it."""
        raise NotImplementedError

    def to_spec(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_spec(cls, d: dict) -> "LocalLossModel":
        raise NotImplementedError


class FixedK(LocalLossModel):
    """A user-supplied constant loss coefficient ``K`` (``zeta``), referenced to
    the component's bore velocity head::

        Δp = K * rho * v**2 / 2,     v = m_dot / (rho * A)

    Use this to enter a handbook ``K`` directly (a valve wide-open ``K``, a
    tabulated bend / tee value, an entrance / exit loss, ...) when a dedicated
    correlation is not needed.  ``K`` is a live `Parameter`, so it can be
    retuned without re-instantiating the model.
    """

    #: The default correlation a UI pre-fills (see `LocalLossModel`).
    _catalog_default = True

    def __init__(
        self,
        K: Annotated[float, ParamSpec(
            "Dimensionless local-loss coefficient (zeta) referenced to the "
            "bore velocity head: Δp = K * rho * v**2 / 2.", unit="1")] = 1.0,
    ):
        self.K = float(K)
        if self.K <= 0.0:
            raise ValueError(f"FixedK: K must be > 0, got {K!r}")
        self.cache_key = ("fixed_k",)

    def declare(self, comp):
        comp.add_component('zeta', Parameter(self.K, "1"))

    def zeta(self, comp, flow=None):
        return comp['zeta'].symbol

    def to_spec(self) -> dict:
        return {"__type__": "FixedK", "K": self.K}

    @classmethod
    def from_spec(cls, d: dict) -> "FixedK":
        return cls(d.get("K", 1.0))


class SuddenExpansion(LocalLossModel):
    """Abrupt enlargement (Borda-Carnot), from the smaller bore into a larger
    downstream section::

        K = (1 - beta)**2,     beta = A_bore / A_large  (<= 1)

    referenced to the *smaller* (upstream = bore) velocity, which is exactly the
    component's reference velocity, so ``K`` plugs straight in.  ``beta = 1`` is
    no area change (``K = 0``, a singular zero-loss limit -- keep ``beta < 1``).
    ``beta -> 0`` (discharge into a large plenum) recovers the full exit loss
    ``K = 1``.
    """

    def __init__(
        self,
        area_ratio: Annotated[float, ParamSpec(
            "Ratio of the (smaller) bore area to the larger downstream area, "
            "beta = A_bore / A_large (0 < beta <= 1); beta = (d_bore/D_large)**2 "
            "for round sections.", unit="1")] = 0.5,
    ):
        self.area_ratio = float(area_ratio)
        if not (0.0 < self.area_ratio <= 1.0):
            raise ValueError(
                f"SuddenExpansion: area_ratio must be in (0, 1], got "
                f"{area_ratio!r}")
        self.cache_key = ("sudden_expansion",)

    def declare(self, comp):
        comp.add_component('area_ratio', Parameter(self.area_ratio, "1"))

    def zeta(self, comp, flow=None):
        beta = comp['area_ratio'].symbol
        return (1 - beta) ** 2

    def to_spec(self) -> dict:
        return {"__type__": "SuddenExpansion", "area_ratio": self.area_ratio}

    @classmethod
    def from_spec(cls, d: dict) -> "SuddenExpansion":
        return cls(d.get("area_ratio", 0.5))


class SuddenContraction(LocalLossModel):
    """Abrupt contraction, from a larger upstream section into the smaller bore
    (Idelchik, turbulent regime)::

        K = 0.5 * (1 - beta)**0.75,     beta = A_bore / A_large  (<= 1)

    referenced to the *smaller* (downstream = bore) velocity, which is the
    component's reference velocity.  ``beta -> 0`` (entrance from a large
    plenum) gives the sharp-edged inlet loss ``K = 0.5``; ``beta = 1`` is no
    contraction (``K = 0``).
    """

    def __init__(
        self,
        area_ratio: Annotated[float, ParamSpec(
            "Ratio of the (smaller) bore area to the larger upstream area, "
            "beta = A_bore / A_large (0 < beta <= 1); beta = (d_bore/D_large)**2 "
            "for round sections.", unit="1")] = 0.5,
    ):
        self.area_ratio = float(area_ratio)
        if not (0.0 < self.area_ratio <= 1.0):
            raise ValueError(
                f"SuddenContraction: area_ratio must be in (0, 1], got "
                f"{area_ratio!r}")
        self.cache_key = ("sudden_contraction",)

    def declare(self, comp):
        comp.add_component('area_ratio', Parameter(self.area_ratio, "1"))

    def zeta(self, comp, flow=None):
        beta = comp['area_ratio'].symbol
        return 0.5 * (1 - beta) ** 0.75

    def to_spec(self) -> dict:
        return {"__type__": "SuddenContraction", "area_ratio": self.area_ratio}

    @classmethod
    def from_spec(cls, d: dict) -> "SuddenContraction":
        return cls(d.get("area_ratio", 0.5))


class LaminarTransitionK(LocalLossModel):
    """A Reynolds-dependent (laminar / transition) loss coefficient -- the
    two-term regime blend Idelchik uses for many local resistances::

        K(Re) = K_turb + K_lam / Re

    ``K_turb`` is the fully-turbulent (quadratic-regime) coefficient the loss
    approaches at high Re; ``K_lam`` is the laminar coefficient of the ``1/Re``
    term that dominates the low-Re / creeping regime (analogous to the ``A/Re``
    laminar friction term).  This is the *shape* of the interface for a
    velocity-dependent ``K`` -- read `flow.Re` and return an expression; supply
    the fitting-specific ``K_turb`` / ``K_lam`` from its Idelchik table.

    ``Re`` is the bore Reynolds number (floored at 1 in `LossFlowState`, so the
    ``1/Re`` term is finite through zero flow).  Both coefficients are live
    `Parameter`s.
    """

    def __init__(
        self,
        K_turb: Annotated[float, ParamSpec(
            "Fully-turbulent (quadratic-regime) loss coefficient the fitting "
            "approaches at high Re.", unit="1")] = 1.0,
        K_lam: Annotated[float, ParamSpec(
            "Laminar coefficient of the 1/Re term (K = K_turb + K_lam/Re) that "
            "dominates the low-Re regime.", unit="1")] = 0.0,
    ):
        self.K_turb = float(K_turb)
        self.K_lam = float(K_lam)
        if self.K_turb < 0 or self.K_lam < 0:
            raise ValueError(
                f"LaminarTransitionK: K_turb / K_lam must be >= 0, got "
                f"{K_turb!r} / {K_lam!r}")
        self.cache_key = ("laminar_transition",)

    def declare(self, comp):
        comp.add_component('K_turb', Parameter(self.K_turb, "1"))
        comp.add_component('K_lam', Parameter(self.K_lam, "1"))

    def zeta(self, comp, flow=None):
        if flow is None:
            raise ValueError(
                "LaminarTransitionK needs the local flow state (Re); it was "
                "applied in a context that did not supply one.")
        return comp['K_turb'].symbol + comp['K_lam'].symbol / flow.Re

    def to_spec(self) -> dict:
        return {"__type__": "LaminarTransitionK",
                "K_turb": self.K_turb, "K_lam": self.K_lam}

    @classmethod
    def from_spec(cls, d: dict) -> "LaminarTransitionK":
        return cls(d.get("K_turb", 1.0), d.get("K_lam", 0.0))


_LOSS_SPECS = {
    "FixedK": FixedK,
    "SuddenExpansion": SuddenExpansion,
    "SuddenContraction": SuddenContraction,
    "LaminarTransitionK": LaminarTransitionK,
}


def local_loss_model_from_spec(d: dict) -> LocalLossModel:
    """Rebuild a `LocalLossModel` from its value spec (see `to_spec`)."""
    t = d.get("__type__")
    cls = _LOSS_SPECS.get(t)
    if cls is None:
        raise ValueError(
            f"unknown local-loss model spec type {t!r}; "
            f"known: {sorted(_LOSS_SPECS)}")
    return cls.from_spec(d)


__all__ = [
    "LossFlowState",
    "LocalLossModel",
    "FixedK",
    "SuddenExpansion",
    "SuddenContraction",
    "LaminarTransitionK",
    "local_loss_model_from_spec",
]
