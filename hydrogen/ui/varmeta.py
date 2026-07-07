"""Offline variable introspection for a placed component.

The property editor works off the *catalogue* metadata (constructor params), but
the "Variables" window and the plot/table objects need the component's *model*
variables -- the time-integrated / algebraic quantities that evolve during a
simulation -- together with their units and descriptions, grouped the way the
component definition nests them (sub-models -> groups, value objects -> leaves).

Those only exist on a *constructed* instance (like the ports in
:mod:`hydrogen.ui.introspect`), so we build the component once via ``from_dict``
and walk its ``Model`` tree.  ``Variable`` / ``Parameter`` / ``Input`` are all
``Model`` subclasses that carry ``.unit`` / ``.description`` / ``.value``; a
node of one of those types is a selectable leaf, any other ``Model`` is a group
we recurse into.  The resulting tree mirrors the shape of
:meth:`hydrogen.service.client.SystemProxy.var_tree` so the same widget code can
render either source.
"""

from __future__ import annotations

from hydrogen.model import Input, Model, Parameter, Variable

from .introspect import _build

__all__ = ["variable_tree", "VARIABLE_KINDS"]

#: Model node types that are selectable leaves (not recursed into).  Order
#: matters only for the ``kind`` tag we attach for display.
VARIABLE_KINDS = (
    ("differential", "hydrogen.model.DifferentialVariable"),
    ("variable", "hydrogen.model.Variable"),
    ("input", "hydrogen.model.Input"),
    ("parameter", "hydrogen.model.Parameter"),
)


def _leaf_kind(obj) -> str | None:
    """Short tag for a value-object leaf, or ``None`` if ``obj`` is a group.

    ``DifferentialVariable`` extends ``Variable`` so it is checked first; ``Input``
    and ``Parameter`` are their own ``Model`` subclasses.
    """
    from hydrogen.model import DifferentialVariable

    if isinstance(obj, DifferentialVariable):
        return "differential"
    if isinstance(obj, Variable):
        return "variable"
    if isinstance(obj, Input):
        return "input"
    if isinstance(obj, Parameter):
        return "parameter"
    return None


def _scalar(value) -> float | None:
    """Best-effort float of a value object's current value (``None`` if it is a
    symbolic expression that has no concrete value yet)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _walk(model: Model, prefix: str) -> list[dict]:
    """Children of ``model`` as tree nodes (recursing into sub-models)."""
    nodes: list[dict] = []
    for name, child in getattr(model, "components", {}).items():
        path = f"{prefix}.{name}" if prefix else name
        kind = _leaf_kind(child)
        if kind is not None:
            nodes.append({
                "name": name,
                "path": path,
                "full": path,
                "leaf": True,
                "kind": kind,
                "unit": getattr(child, "unit", None) or "",
                "description": getattr(child, "description", None) or "",
                "value": _scalar(getattr(child, "value", None)),
            })
        elif isinstance(child, Model) and getattr(child, "components", None):
            children = _walk(child, path)
            if children:                     # skip empty sub-models
                nodes.append({
                    "name": name,
                    "path": path,
                    "leaf": False,
                    "children": children,
                    "count": sum(c.get("count", 1) for c in children),
                })
    return nodes


def variable_tree(type_name: str, medium: str | None,
                  params: dict | None) -> dict | None:
    """Structured variable tree for a component, or ``None`` if it can't build.

    Returns the root group ``{"name": "", "path": "", "leaf": False,
    "children": [...], "count": N}``.  Leaf nodes carry ``path`` / ``full`` (the
    dotted name *within the component*; prefix it with the instance ``comp_id``
    to match a recorded system variable), ``unit``, ``description``, ``value``
    and a ``kind`` tag.  Leaves (a level's own, outermost variables) sort before
    groups (nested sub-models); both are name-sorted.
    """
    comp = _build(type_name, medium, params)
    if comp is None:
        return None
    try:
        children = _walk(comp, "")
    except Exception:
        return None

    def _sort(nodes):
        for n in nodes:
            if not n["leaf"]:
                _sort(n["children"])
        # Leaves (the outermost, directly-owned variables at this level) sort
        # before groups (nested sub-models holding deeper/inner variables), so
        # a component's own variables sit at the top rather than below every
        # sub-model.  Both blocks are name-sorted.
        nodes.sort(key=lambda n: (not n["leaf"], n["name"].lower()))

    _sort(children)
    return {
        "name": "",
        "path": "",
        "leaf": False,
        "children": children,
        "count": sum(c.get("count", 1) for c in children),
    }
