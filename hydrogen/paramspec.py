"""Declarative, single-source parameter metadata for components & value objects.

A component (or serializable value object) describes its constructor arguments
*once*, on the class, in a ``PARAMS`` mapping keyed by the exact ``__init__``
argument name::

    class CylindricalWall(TwoNodeWall):
        PARAMS = {
            "r_in":   ParamSpec("Inner radius of the tube wall.", unit="m"),
            "r_out":  ParamSpec("Outer radius of the tube wall.", unit="m"),
            "length": ParamSpec("Axial length of the wall segment.", unit="m"),
        }

That single definition is consumed in two places, neither of which needs to
instantiate the component:

  * the **component catalog** (:mod:`hydrogen.serialization`) introspects the
    class ``__init__`` signature and overlays this metadata so a UI gets the
    unit / description / choices / conditional-relevance of every field, and
  * ``declare_components`` pulls ``unit`` / ``description`` straight onto the
    `Parameter` it builds (via :meth:`ParamSpec.param_kwargs`), so those strings
    live in exactly one place instead of being repeated at each ``Parameter(...)``
    call site.

``PARAMS`` is inherited: :func:`merged_param_specs` walks the MRO so a base
class (e.g. the shared ``rho`` / ``cp`` / ``k`` of ``TwoNodeWall``) contributes
to every subclass, with the subclass winning on conflicts.

This module imports nothing from the rest of ``hydrogen`` so it can be used
from ``model``, ``components`` and ``serialization`` alike without import cycles.
"""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass


class _Unset:
    """Sentinel: 'no default supplied' (distinct from a default of ``None``)."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "UNSET"


UNSET = _Unset()


@dataclass(frozen=True)
class ParamSpec:
    """Authoring-time metadata for one constructor argument.

    Attributes
    ----------
    description : str
        Human-readable, one-line description for a UI label / tooltip.
    unit : str | None
        Physical unit string (e.g. ``"Pa"`` / ``"K"`` / ``"m"``; ``"1"`` for a
        dimensionless quantity).  ``None`` means "no fixed unit" -- the catalog
        may still fall back to its shared unit table for common names.
    choices : tuple | None
        For a categorical (enum-like) argument, the closed set of valid values.
    structural : bool
        ``True`` marks an argument whose value changes the *structure* of the
        emitted equations (which variables/residuals exist), not just a numeric
        coefficient -- e.g. a ``dynamic`` / ``leaky`` / ``multiphase`` toggle.
        Such arguments are surfaced as catalog **literals** and contribute to
        the per-class equation-template cache key (see
        :func:`cache_key_flag_names`), so two instances with different values
        get distinct cache entries instead of replaying each other's template.
    relevant_when, required_when : dict | str | None
        Conditional-applicability hints for a UI.  Either a ``{sibling_param:
        value_or_list}`` predicate (the field applies only when the sibling
        equals one of the values) or a named predicate string the component
        documents (e.g. ``"any_layer_permeable"``).  ``relevant_when`` gates
        show/hide; ``required_when`` marks "must be filled under this condition".
        These are advisory: the constructor remains the hard validator.
    default : optional
        A suggested starting value for a UI / spec template, for an argument the
        *constructor* still requires (so it has no signature default).  This
        pre-fills the field without loosening the API: ``required`` keeps
        reflecting the constructor, the catalog just also reports a ``default``.
        Leave unset (``UNSET``) for no suggestion.
    """

    description: str = ""
    unit: str | None = None
    choices: tuple | None = None
    structural: bool = False
    relevant_when: "dict | str | None" = None
    required_when: "dict | str | None" = None
    default: object = UNSET

    @property
    def has_default(self) -> bool:
        return self.default is not UNSET

    def param_kwargs(self) -> dict:
        """``{unit, description}`` kwargs for building a `Parameter` from this
        spec (omitting the ones that are unset)."""
        kw: dict = {}
        if self.unit is not None:
            kw["unit"] = self.unit
        if self.description:
            kw["description"] = self.description
        return kw

    def catalog_extras(self) -> dict:
        """The descriptor keys this spec contributes to a catalog entry."""
        extras: dict = {}
        if self.description:
            extras["description"] = self.description
        if self.unit is not None:
            extras["unit"] = self.unit
        if self.choices is not None:
            extras["choices"] = list(self.choices)
        if self.relevant_when is not None:
            extras["relevant_when"] = self.relevant_when
        if self.required_when is not None:
            extras["required_when"] = self.required_when
        return extras


def _find_param_spec(ann) -> "ParamSpec | None":
    """Locate a `ParamSpec` in an annotation, searching through any
    ``Optional`` / ``Union`` wrappers.

    A parameter with a ``= None`` default makes `typing.get_type_hints` wrap the
    annotation in an outer ``Optional[...]`` (e.g.
    ``Optional[Annotated[list, ParamSpec(...)]]``), so the ``Annotated``
    metadata is nested rather than on the outermost type -- hence the recursion.
    """
    for meta in getattr(ann, "__metadata__", ()):
        if isinstance(meta, ParamSpec):
            return meta
    for arg in typing.get_args(ann):
        found = _find_param_spec(arg)
        if found is not None:
            return found
    return None


def _annotated_param_specs(func) -> dict:
    """``{arg name -> ParamSpec}`` harvested from a callable's ``Annotated``
    type hints, e.g. ``D: Annotated[float, ParamSpec("...", unit="m")]``.

    Returns an empty dict if the hints can't be resolved (e.g. an unresolvable
    forward reference under ``from __future__ import annotations``)."""
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:
        return {}
    out: dict = {}
    for name, ann in hints.items():
        if name == "return":
            continue
        spec = _find_param_spec(ann)
        if spec is not None:
            out[name] = spec
    return out


@functools.lru_cache(maxsize=None)
def merged_param_specs(cls: type) -> dict:
    """``{arg name -> ParamSpec}`` for ``cls``, base classes first so a subclass
    overrides an inherited entry of the same name.

    Two sources are merged (both walked across the MRO):

      1. ``Annotated`` metadata on each class's own ``__init__`` -- the
         preferred single source of truth (the default lives in the Python
         signature, the description / unit / choices / ``structural`` flag in
         the annotation), and
      2. the legacy ``PARAMS`` class attribute (kept working so the migration
         can proceed component-by-component).

    Within one class the ``PARAMS`` dict wins on conflicts (it is applied
    last), so a half-migrated class behaves predictably.  Result is cached per
    class; component ``__init__`` signatures are immutable at runtime."""
    out: dict = {}
    for klass in reversed(cls.__mro__):
        own_init = vars(klass).get("__init__")
        if own_init is not None:
            out.update(_annotated_param_specs(own_init))
        params = vars(klass).get("PARAMS")
        if params:
            out.update(params)
    return out


@functools.lru_cache(maxsize=None)
def structural_param_names(cls: type) -> tuple:
    """Names of ``cls``'s constructor args marked ``structural=True`` (in MRO
    order), i.e. the toggles that change the emitted equation structure."""
    return tuple(name for name, spec in merged_param_specs(cls).items()
                 if spec.structural)


@functools.lru_cache(maxsize=None)
def cache_key_flag_names(cls: type) -> tuple:
    """The effective equation-template cache-key flag names for ``cls``.

    The union (preserving order) of:

      * structural constructor args (``ParamSpec(structural=True)``), and
      * any names explicitly listed in the class's ``_cache_key_flags`` -- used
        for *computed* keys that are not constructor arguments (e.g. a private
        ``_perm_key`` summarising an injected model's structural identity).

    `Model.collect_equations` keys its per-class template cache on the live
    values of these names."""
    names = list(structural_param_names(cls))
    for extra in getattr(cls, "_cache_key_flags", ()) or ():
        if extra not in names:
            names.append(extra)
    return tuple(names)
