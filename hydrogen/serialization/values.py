"""(De)serialization of structured constructor-argument *values*.

Most leaf components take only JSON-scalar constructor params, which
:mod:`hydrogen.serialization.spec` round-trips reflectively.  A few take
structured value objects instead -- e.g. ``Pipe(layers=[WallLayer(...)])``,
where each :class:`WallLayer` nests a :class:`WallMaterial` and an optional
:class:`PermeationFlux` (itself carrying a :class:`TransportFit` /
:class:`Permeant`).

Those value objects opt in to serialization by implementing a tiny protocol:

  * ``to_spec(self) -> dict`` returning a JSON-able dict tagged with
    ``{"__type__": "<ClassName>", ...}`` (nested value objects encoded the
    same way), and
  * a ``from_spec(cls, d) -> obj`` classmethod that rebuilds it (decoding its
    own nested values directly).

This module is the codec's glue: it encodes a param value (scalar / list /
value object), decodes a spec value back, and validates that a param value
from a loaded spec is a scalar or a *known* value spec.  The value classes are
imported lazily so importing :mod:`hydrogen.serialization` never pulls in the
component tree eagerly.
"""

from __future__ import annotations

from .errors import SerializationError

_JSON_SCALAR = (int, float, str, bool, type(None))


def _value_classes() -> dict:
    """Lazy ``{__type__ name -> class}`` for every serializable value object."""
    from ..components.materials import WallMaterial
    from ..components.thermofluid.assemblies import WallLayer
    from ..components.thermofluid.permeation import (
        Permeant,
        SteadyRichardson,
        TransientDiffusion,
        TransportFit,
    )

    classes = (
        WallMaterial,
        Permeant,
        TransportFit,
        SteadyRichardson,
        TransientDiffusion,
        WallLayer,
    )
    return {c.__name__: c for c in classes}


def is_value_object(value) -> bool:
    """True if ``value`` is a (non-scalar) object implementing ``to_spec``."""
    return not isinstance(value, _JSON_SCALAR) and callable(
        getattr(value, "to_spec", None)
    )


def encode_value(value):
    """Encode a constructor-param value to a JSON-able form.

    Scalars pass through; lists/tuples recurse; value objects defer to their
    ``to_spec``.  Anything else is rejected (caller may still coerce numpy
    scalars via its own ``_jsonable``).
    """
    if isinstance(value, _JSON_SCALAR):
        return value
    if isinstance(value, (list, tuple)):
        return [encode_value(v) for v in value]
    if is_value_object(value):
        return value.to_spec()
    raise SerializationError(
        f"value {value!r} of type {type(value).__name__} is not serializable"
    )


def decode_value(value):
    """Rebuild a value from a spec produced by :func:`encode_value`."""
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    if isinstance(value, dict) and "__type__" in value:
        registry = _value_classes()
        name = value["__type__"]
        cls = registry.get(name)
        if cls is None:
            raise SerializationError(
                f"unknown value spec type {name!r}; known: {sorted(registry)}"
            )
        return cls.from_spec(value)
    return value


def value_spec_ok(value) -> bool:
    """Validate that a *loaded* param value is a scalar or known value spec."""
    if isinstance(value, _JSON_SCALAR):
        return True
    if isinstance(value, list):
        return all(value_spec_ok(v) for v in value)
    if isinstance(value, dict):
        return value.get("__type__") in _value_classes()
    return False
