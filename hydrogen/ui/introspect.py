"""Component introspection helpers the canvas needs but the catalogue metadata
can't give statically.

Ports are created in a component's ``declare_components`` -- they only exist on
a *constructed* instance -- so the only way to know a node's connectors is to
build it once via ``from_dict`` and read them back.
"""

from __future__ import annotations

import hydrogen as hd
from hydrogen.serialization import SCHEMA_VERSION

__all__ = ["introspect_ports"]


def introspect_ports(type_name: str, medium: str | None,
                     params: dict | None) -> list[tuple[str, str]]:
    """``[(port_name, kind), ...]`` for a component, by building it once via
    ``from_dict``.  Returns ``[]`` if it can't be built (e.g. params are
    incomplete mid-edit)."""
    try:
        spec = hd.component_spec(type_name)
        template = dict(spec["template"])
        if params is not None:
            template["params"] = params
        media: dict[str, dict] = {}
        if spec["needs_medium"]:
            name = medium or "Hydrogen"
            template["medium"] = name
            media[name] = {"fluid": name}
        system = hd.from_dict({
            "hydrogen_version": hd.__version__, "schema_version": SCHEMA_VERSION,
            "media": media, "components": {"c": template}, "connections": [],
        })
        return [(n, p.kind) for n, p in system["c"].ports.items()]
    except Exception:
        return []
