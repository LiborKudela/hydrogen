"""Typed connectors (`Port`) for the hydrogen DAE framework.

A `Port` is a small, declarative wrapper around a group of `Variable`s that
together describe a physical interface on a `Model`.  It exists for one
reason: to let users replace the brittle per-channel loops

    for io in ('p', 'h', 'm_dot'):
        self.add_connection(self['a'][f'{io}_out'], self['b'][f'{io}_in'])

with a single, type-checked `connect()` call

    self.connect(self['a'].ports['outlet'], self['b'].ports['inlet'])

that

  * generates one `Model.add_connection` per channel (so the whole port
    still rides the union-find fast path at instantiate time),
  * verifies port kind compatibility (you can't wire a fluid port to an
    electrical port),
  * verifies single-use multiplicity (a port can be wired exactly once;
    fan-out / fan-in must go through a `Splitter` / `MixingJunction`),
  * picks the per-channel `sign` automatically based on the two ports'
    `flow_orientation` -- opposite orientations are direct equality, same
    orientations are sum-to-zero (the Kirchhoff / Modelica convention).

The Port layer is **purely declarative**: it never owns Variables.  A port
binds itself to Variables that already live in the owner Model's
`components` dict, so existing code that talks to those Variables directly
keeps working byte-for-byte.  This is what lets the codebase migrate to
ports incrementally without an all-or-nothing rewrite.

See `hydrogen/components.py` for component-side port declarations and
`Model.connect` (in `hydrogen/model.py`) for the wiring entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from .model import Model, Variable


class PortError(RuntimeError):
    """Base class for port-level wiring errors (kind mismatch, etc.)."""


class PortAlreadyConnectedError(PortError):
    """Raised when `connect()` is called on a port that already has a wire.

    A port is single-multiplicity by design: if you need fan-out or fan-in,
    insert a dedicated junction component (`Splitter`, `MixingJunction`)
    that exposes one port per branch.  This error fires at `connect()`
    time, so wiring mistakes surface immediately instead of becoming
    silent over-constraints at solve time.
    """


class PortKindMismatchError(PortError):
    """Raised when two ports of different `kind` are wired together."""


class PortChannelMissingError(PortError):
    """A subclass declared a `required_channels` entry but the constructor
    binding dict didn't supply a Variable for it.  Or two connected ports
    declared the same `kind` but disagree on which channels they expose."""


class PortMediumMismatchError(PortError):
    """Two fluid ports declared distinct `medium` references.  Most useful
    catch in multi-medium systems where wiring an air port into a hydrogen
    network would otherwise produce a confusing CoolProp NameError much
    later (the lambdified residual references the wrong medium's symbolic
    property functions)."""


class Port:
    """A typed bundle of `Variable` references forming a single interface.

    A Port subclass declares:

      * `kind`                 - discriminator string; only ports with the
                                 same `kind` can be `connect()`-ed.
      * `required_channels`    - tuple of channel names that the subclass
                                 guarantees to be bound.  Subclasses may
                                 expose ADDITIONAL channels at construction
                                 time, but the required ones are the
                                 wire-up contract.
      * `flow_channels`        - subset of channels that behave as THROUGH
                                 variables (positive = into/out of me at
                                 this face).  Across variables (pressure,
                                 enthalpy, temperature, voltage, ...) are
                                 unified directly across the wire; flow
                                 variables get an automatic sign flip
                                 between same-orientation ports.

    A port instance holds:

      * `owner`               - the `Model` that hosts this port.
      * `name`                - this port's local name on the owner (set by
                                 `Model.add_port`).
      * `channels`            - dict mapping channel name -> backing
                                 `Variable` (already a member of
                                 `owner.components`).
      * `flow_orientation`    - "in"  (positive flow = INTO me at this face)
                                 or "out" (positive flow = OUT of me).
                                 Determines whether `connect()` emits a
                                 sign flip for flow channels.
      * `medium`              - optional reference (e.g. a
                                 `CoolPropMedium`); when both connected
                                 ports declare a non-None medium, they
                                 must match.
      * `_connected_to`       - back-reference to the other end after a
                                 successful `connect()`.  `None` until
                                 wired; raises on a second connect
                                 attempt.
    """

    kind: str = ""
    required_channels: Tuple[str, ...] = ()
    flow_channels: Tuple[str, ...] = ()

    def __init__(
        self,
        owner: "Model",
        channels: Dict[str, "Variable"],
        *,
        flow_orientation: str = "in",
        medium=None,
        name: str | None = None,
    ):
        if not self.kind:
            raise PortError(
                f"{type(self).__name__} must set a non-empty `kind` class attribute "
                "before it can be instantiated."
            )
        if flow_orientation not in ("in", "out"):
            raise PortError(
                f"flow_orientation must be 'in' or 'out', got {flow_orientation!r}"
            )
        missing = [ch for ch in self.required_channels if ch not in channels]
        if missing:
            raise PortChannelMissingError(
                f"{type(self).__name__} requires channels {self.required_channels}; "
                f"missing {missing} from binding {list(channels)}"
            )
        unknown_flow = [ch for ch in self.flow_channels if ch not in channels]
        if unknown_flow:
            raise PortChannelMissingError(
                f"{type(self).__name__} declares flow channels {self.flow_channels} "
                f"but {unknown_flow} are not in the binding {list(channels)}"
            )
        self.owner = owner
        self.name = name
        self.channels: Dict[str, "Variable"] = dict(channels)
        self.flow_orientation = flow_orientation
        self.medium = medium
        self._connected_to: "Port | None" = None

    # --- introspection ----------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected_to is not None

    def __repr__(self) -> str:
        owner_name = getattr(self.owner, "name", None) or type(self.owner).__name__
        return (
            f"<{type(self).__name__} kind={self.kind!r} "
            f"owner={owner_name} name={self.name!r} "
            f"orientation={self.flow_orientation!r}>"
        )

    # --- internal wiring hook (called by Model.connect) -------------------

    def _mark_connected(self, other: "Port") -> None:
        if self._connected_to is not None:
            raise PortAlreadyConnectedError(
                f"Port {self._path()} is already wired to "
                f"{self._connected_to._path()}; insert a Splitter or "
                "MixingJunction if you need fan-out / fan-in."
            )
        self._connected_to = other

    def _path(self) -> str:
        owner_name = getattr(self.owner, "name", None) or type(self.owner).__name__
        return f"{owner_name}.{self.name or '<unnamed>'}"


# ---------------------------------------------------------------------------
# Built-in port kinds
# ---------------------------------------------------------------------------


class FluidPort_phm(Port):
    """Compressible-fluid interface carrying `(p, h, m_dot)`.

    * `p`       - port pressure                   [Pa]  (across)
    * `h`       - port specific enthalpy          [J/kg] (across)
    * `m_dot`   - port mass flow rate             [kg/s] (THROUGH)
                  positive = "INTO me" for orientation="in";
                  positive = "OUT of me" for orientation="out".

    All standard fluid components in the package expose either an `outlet`
    (orientation="out") or `inlet` (orientation="in") port of this kind,
    and `Model.connect()` automatically picks the right per-channel sign
    based on those two orientations.
    """

    kind = "fluid_phm"
    required_channels = ("p", "h", "m_dot")
    flow_channels = ("m_dot",)


class ThermalPort_TQ(Port):
    """Heat-transfer interface: temperature (across) + heat-flow (through).

    Not used by any in-tree component today; declared here as a worked
    example of cross-domain extension.  Add a `T_wall` / `Q_dot_wall`
    port to a heated pipe segment and you can wire it directly to an
    insulating sleeve or a heat source via `connect()`.
    """

    kind = "thermal_TQ"
    required_channels = ("T", "Q_dot")
    flow_channels = ("Q_dot",)


class ElectricalPort_VI(Port):
    """Single-pin electrical interface: voltage (across) + current (through).

    Reserved for future electrolyser / fuel-cell components.  Same wiring
    rules: same `kind`, opposite orientation -> direct equality on `V` and
    on `I`; same orientation -> equality on `V` plus sum-to-zero on `I`
    (Kirchhoff's current law on a node).
    """

    kind = "electrical_VI"
    required_channels = ("V", "I")
    flow_channels = ("I",)
