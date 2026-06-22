"""Generic port machinery for the hydrogen DAE framework.

This module defines only the *domain-agnostic* core:

  * `Port` -- declarative base class binding a group of Variables to a
    typed connector with a `kind` discriminator, channel inventory,
    flow/across split, and single-use multiplicity.
  * `PortError` + subclasses -- the wiring-error hierarchy raised by
    `Model.connect()` and `Port._mark_connected()`.

Concrete port subclasses (`FluidPort_phm`, `ThermalPort_TQ`,
`PermeationPort_pN`, ...) live in their respective domain libraries next to
the components that use them, e.g. `hydrogen.components.thermofluid.ports`.
Keeping the library-specific ports inside the domain means each library is
self-contained: a user reading the thermofluid package sees the port contracts
and the component implementations side-by-side, without hopping into a global
"ports.py".

Wiring contract (enforced by `Model.connect()`):

  * matching `kind`,
  * matching channel sets,
  * matching `medium` when both ports declare one,
  * each side wired exactly once (fan-out/fan-in must go through a
    dedicated junction component such as `Splitter` or `MixingJunction`),
  * sign on every `add_connection` chosen automatically from the two
    ports' `flow_orientation`s -- opposite -> direct equality, same ->
    sum-to-zero on flow channels (Kirchhoff / Modelica connector rule).

The Port layer is **purely declarative**: it never owns Variables.  A
port binds itself to Variables that already live in the owner Model's
`components` dict, so existing code that talks to those Variables
directly keeps working byte-for-byte.  This is what lets the codebase
migrate to ports incrementally without an all-or-nothing rewrite.

See `Model.connect` in `hydrogen/model.py` for the wiring entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from .model import Model, Variable


class PortError(RuntimeError):
    """Base class for port-level wiring errors (kind mismatch, etc.)."""


class PortNotConnectedWarning(UserWarning):
    """Emitted at `instantiate()` for a port declared `require_connection=True`
    that ended up with no wire.

    Such a port leaves its backing across-variable unclosed, so the system is
    structurally singular and the Newton solve would otherwise fail with an
    opaque "Factor is exactly singular".  The warning names the offending port
    so the cause is obvious.  Used by, e.g., a heated fluid segment's `wall`
    port (`TwoPortSegment(heat_port=True)`), which only makes sense once a
    thermal boundary or wall is attached.
    """


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
      * `require_connection`  - when True, `Model.instantiate()` emits a
                                 `PortNotConnectedWarning` if this port was
                                 never wired (its across-variable would be
                                 left unclosed -> singular system).  Default
                                 False (a port may be intentionally open,
                                 e.g. a top-level boundary exposed for
                                 external wiring).
      * `allow_fanout`        - when True, this port may be wired more than
                                 once (one->many).  Only safe for ports with
                                 NO flow channels: each extra wire is a pure
                                 value-equality, so all partners collapse to
                                 one symbol and the constraint set stays
                                 consistent.  Used by signal OUTPUT ports
                                 (`RealSignal`) so one block output can drive
                                 several inputs, as in Modelica.  Default
                                 False -- ports are single-use, and fan-out of
                                 a FLOW channel must go through a dedicated
                                 junction (`Splitter` / `MixingJunction`).
      * `medium`              - optional reference (e.g. a
                                 `CoolPropMedium`); when both connected
                                 ports declare a non-None medium, they
                                 must match.
      * `_connected_to`       - back-reference to the FIRST partner after a
                                 successful `connect()`.  `None` until wired;
                                 a second connect raises unless
                                 `allow_fanout=True`.
      * `_connections`        - list of ALL partners (length 1 for a normal
                                 single-use port; longer for a fan-out output).
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
        require_connection: bool = False,
        allow_fanout: bool = False,
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
        if allow_fanout and self.flow_channels:
            raise PortError(
                f"{type(self).__name__} sets allow_fanout=True but declares flow "
                f"channels {self.flow_channels}; fan-out is only sound for ports "
                "with no THROUGH variables (otherwise Kirchhoff is violated). Use "
                "a Splitter/MixingJunction for flow fan-out."
            )
        self.owner = owner
        self.name = name
        self.channels: Dict[str, "Variable"] = dict(channels)
        self.flow_orientation = flow_orientation
        self.medium = medium
        self.require_connection = require_connection
        self.allow_fanout = allow_fanout
        # `_connected_to` keeps the FIRST partner (so `is_connected` and the
        # single-wire serialization walk keep working unchanged); `_connections`
        # records every partner for fan-out outputs.
        self._connected_to: "Port | None" = None
        self._connections: list["Port"] = []

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
        if self._connected_to is not None and not self.allow_fanout:
            raise PortAlreadyConnectedError(
                f"Port {self._path()} is already wired to "
                f"{self._connected_to._path()}; insert a Splitter or "
                "MixingJunction if you need fan-out / fan-in."
            )
        if self._connected_to is None:
            self._connected_to = other
        self._connections.append(other)

    def _path(self) -> str:
        owner_name = getattr(self.owner, "name", None) or type(self.owner).__name__
        return f"{owner_name}.{self.name or '<unnamed>'}"
