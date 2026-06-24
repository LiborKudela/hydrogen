"""Component introspection helpers the canvas needs but the catalogue metadata
can't give statically.

Ports are created in a component's ``declare_components`` -- they only exist on
a *constructed* instance -- so the only way to know a node's connectors (and
whether it carries any time-integrated state) is to build it once via
``from_dict`` and read them back.
"""

from __future__ import annotations

import hydrogen as hd
from hydrogen.serialization import SCHEMA_VERSION

__all__ = ["introspect", "introspect_ports", "introspect_dynamics"]


def _build(type_name: str, medium: str | None, params: dict | None):
    """Construct the component once via ``from_dict`` and return the live
    `Model` instance, or ``None`` if it can't be built (e.g. params are
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
        return system["c"]
    except Exception:
        return None


def introspect(type_name: str, medium: str | None, params: dict | None
               ) -> tuple[bool, list[tuple[str, str]], bool, list[str]]:
    """``(ok, ports, is_dynamic, differential_vars)`` for a component, built ONCE.

    * ``ok``      -- ``True`` if the component could be constructed with the
      given params.  ``False`` means the configuration is invalid or incomplete
      (e.g. a `Pipe` with no wall layers, which the model rejects); callers
      should then *keep* whatever ports they already have rather than wiping
      them, so a mid-edit invalid state doesn't destroy a node's connectors and
      wires.
    * ``ports``   -- ``[(port_name, kind), ...]`` (empty when ``ok`` is False).
    * ``is_dynamic`` -- ``True`` if the component carries any differential
      state (dynamic); ``False`` if purely algebraic (quasi-static).
    * ``differential_vars`` -- dotted names of those differential states.

    Returns ``(False, [], False, [])`` if the component can't be built.
    """
    comp = _build(type_name, medium, params)
    if comp is None:
        return False, [], False, []
    try:
        ports = [(n, p.kind) for n, p in comp.ports.items()]
        diffs = comp.differential_variables()
        return True, ports, bool(diffs), diffs
    except Exception:
        return False, [], False, []


def introspect_ports(type_name: str, medium: str | None,
                     params: dict | None) -> list[tuple[str, str]]:
    """``[(port_name, kind), ...]`` for a component (see :func:`introspect`)."""
    return introspect(type_name, medium, params)[1]


def introspect_dynamics(type_name: str, medium: str | None,
                        params: dict | None) -> tuple[bool, list[str]]:
    """``(is_dynamic, differential_vars)`` for a component (see
    :func:`introspect`)."""
    _ok, _ports, is_dynamic, diffs = introspect(type_name, medium, params)
    return is_dynamic, diffs
