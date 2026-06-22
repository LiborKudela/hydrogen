"""System (de)serialization for hydrogen.

Dump a built `Model` to a versioned, declarative spec (dict or JSON) and load
it back, validating that every component type resolves and that all parameter
names / values are well-formed.

Public API::

    from hydrogen.serialization import (
        to_dict, from_dict, to_json, from_json,
        register_component,
        SerializationError, SystemSpecError, SCHEMA_VERSION,
    )

The dict form is canonical; ``to_json`` / ``from_json`` are thin wrappers over
``to_dict`` / ``from_dict`` plus the stdlib ``json`` codec.

Scope (this phase): media, required component packages, components (registered
leaves + recursive inline ``type: "Model"`` composites), raw
``variables``/parameters, and port-level connections.  Symbolic ``equations``
and time-dependent ``inputs`` are reserved for a later phase and are rejected
with a clear error if present.
"""

from __future__ import annotations

import json

from ..model import Model
from .errors import SerializationError, SystemSpecError
from .registry import (
    SCHEMA_VERSION,
    available_domains,
    component_catalog,
    component_spec,
    format_component_catalog,
    register_component,
    spec_template,
    value_object_catalog,
    value_object_spec,
    value_template,
)
from .spec import from_dict, to_dict

__all__ = [
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "register_component",
    "component_catalog",
    "component_spec",
    "format_component_catalog",
    "available_domains",
    "value_object_catalog",
    "value_object_spec",
    "spec_template",
    "value_template",
    "SerializationError",
    "SystemSpecError",
    "SCHEMA_VERSION",
]


def to_json(model: Model, *, indent: int = 2, **json_kwargs) -> str:
    """Serialize a `Model` to a JSON string (see :func:`to_dict`)."""
    return json.dumps(to_dict(model), indent=indent, **json_kwargs)


def from_json(text: str, *, strict_version: bool = False) -> Model:
    """Build a `Model` from a JSON string (see :func:`from_dict`)."""
    return from_dict(json.loads(text), strict_version=strict_version)
