"""Component-type and medium registries for system (de)serialization.

The registry maps the ``"type"`` strings that appear in a spec onto concrete
classes:

  * **builtin hydrogen components** are registered under their bare class name
    (``"StraightPipe"``, ``"FlatWall"``, ...), discovered from
    ``hydrogen.components.__all__``.
  * **user-defined component libraries** are declared in the spec under
    ``requires.packages`` and registered under a *namespaced* name
    (``"acme_hydro.ControlValve"``) so a third-party type can never silently
    shadow a builtin.
  * **programmatic registration** via :func:`register_component` adds extra
    classes under a chosen name (handy for tests or ad-hoc classes).

Media are (de)serialised here too, since a `CoolPropMedium` is referenced by
key from many components and must be reconstructed once and shared.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import sys
import types
import typing

from ..medium import CoolPropMedium
from ..model import Model
from ..paramspec import cache_key_flag_names, merged_param_specs

# Bump when the on-disk spec *format* changes incompatibly (independent of the
# hydrogen library version, which is recorded separately in every dump).
SCHEMA_VERSION = 1

# Classes registered programmatically via `register_component`.
_EXTRA_REGISTRY: dict[str, type] = {}


def register_component(cls: type, name: str | None = None) -> type:
    """Register a component class so specs may reference it by ``name``.

    Returns ``cls`` so it can be used as a decorator.  ``name`` defaults to the
    class's ``__name__``.
    """
    if not (isinstance(cls, type) and issubclass(cls, Model)):
        raise TypeError(
            f"register_component expects a Model subclass, got {cls!r}"
        )
    _EXTRA_REGISTRY[name or cls.__name__] = cls
    return cls


def domain_of(cls: type) -> str | None:
    """The physics-domain folder a component lives in (``control``, ``fluid``,
    ``thermal``, ``power``, ...), derived from ``hydrogen.components.<domain>``;
    ``None`` for classes defined outside a domain subpackage."""
    parts = (cls.__module__ or "").split(".")
    if "components" in parts:
        i = parts.index("components")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def category_of(cls: type) -> str | None:
    """The component submodule inside its domain, e.g. ``assemblies`` or
    ``flow`` for ``hydrogen.components.thermofluid`` classes."""
    parts = (cls.__module__ or "").split(".")
    if "components" in parts:
        i = parts.index("components")
        if i + 2 < len(parts):
            return ".".join(parts[i + 2:])
    return None


def full_type_name(cls: type) -> str:
    """Canonical, collision-proof type name: ``hydrogen.<domain>.<ClassName>``.

    Two leaves in different physics domains may share a class name, so the
    fully-qualified, domain-namespaced name is what we register and dump.  A
    class outside a domain subpackage falls back to its bare ``__name__``.
    """
    dom = domain_of(cls)
    return f"hydrogen.{dom}.{cls.__name__}" if dom else cls.__name__


def _iter_builtin_components():
    """Yield every hydrogen-shipped component class.

    The component subpackages keep their natural module structure (no flat
    re-exports), so discovery walks every module under ``hydrogen.components``
    and collects the concrete `Model` subclasses *defined* in each one.  A class
    is skipped when its name is private (leading underscore) or it is flagged as
    an abstract base via ``_catalog_abstract = True``; classes are de-duplicated
    by identity so a class imported into a sibling module is only yielded once.
    """
    # Imported lazily: this module is imported from `hydrogen/__init__`, and we
    # want `hydrogen.components` to be importable first.
    from .. import components as comps

    seen: set[type] = set()
    for info in pkgutil.walk_packages(comps.__path__, comps.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (isinstance(obj, type) and issubclass(obj, Model)
                    and obj is not Model):
                continue
            if obj.__module__ != module.__name__:
                continue  # imported here, defined elsewhere -- yield at its home
            if obj.__dict__.get("_catalog_abstract", False):
                continue  # own flag only -- subclasses are still real components
            if obj in seen:
                continue
            seen.add(obj)
            yield obj


def builtin_registry() -> dict[str, type]:
    """The hydrogen-shipped components, keyed by canonical full name
    (``hydrogen.<domain>.<ClassName>``).

    Only fully-qualified names are registered: a spec must address a builtin by
    its ``hydrogen.<domain>.<ClassName>`` name so that leaves sharing a class
    name across domains never collide.
    """
    return {full_type_name(obj): obj for obj in _iter_builtin_components()}


def _module_version(mod) -> str | None:
    return getattr(mod, "__version__", None)


def _version_tuple(v: str):
    parts = []
    for chunk in str(v).split("."):
        num = "".join(c for c in chunk if c.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _check_pkg_version(mod, import_path: str, min_version, errors: list):
    if min_version is None:
        return
    have = _module_version(mod)
    if have is None:
        errors.append(
            f"required package {import_path!r} declares no __version__ but the "
            f"spec requires >= {min_version}"
        )
        return
    if _version_tuple(have) < _version_tuple(min_version):
        errors.append(
            f"required package {import_path!r} version {have} is older than the "
            f"spec's required >= {min_version}"
        )


def _package_components(mod, import_path: str) -> dict[str, type]:
    """Discover the `Model` subclasses a user package contributes.

    Preference order:
      1. an explicit ``register_hydrogen_components(registry: dict)`` hook on
         the module (the package controls exactly what it publishes), or
      2. a scan of the module's ``__all__`` for `Model` subclasses.

    Every contributed class is namespaced as ``"<import_path>.<ClassName>"``.
    """
    collected: dict[str, type] = {}
    hook = getattr(mod, "register_hydrogen_components", None)
    if callable(hook):
        local: dict[str, type] = {}
        hook(local)
        for name, cls in local.items():
            collected[name] = cls
    else:
        for name in getattr(mod, "__all__", dir(mod)):
            obj = getattr(mod, name, None)
            if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
                collected[name] = obj
    return {f"{import_path}.{name}": cls for name, cls in collected.items()}


def build_registry(spec: dict | None, errors: list) -> dict[str, type]:
    """Assemble the full type registry for a load: builtins + programmatic
    extras + every package declared under ``spec['requires']['packages']``.

    Import / version problems are appended to ``errors`` (never raised here) so
    they aggregate with the rest of the spec validation.
    """
    reg = builtin_registry()
    reg.update(_EXTRA_REGISTRY)

    requires = (spec or {}).get("requires", {}) or {}
    for pkg in requires.get("packages", []):
        import_path = pkg.get("import")
        if not import_path:
            errors.append("requires.packages entry is missing an 'import' field")
            continue
        try:
            mod = importlib.import_module(import_path)
        except ImportError as exc:
            errors.append(
                f"required package {import_path!r} could not be imported: {exc}"
            )
            continue
        _check_pkg_version(mod, import_path, pkg.get("min_version"), errors)
        reg.update(_package_components(mod, import_path))
    return reg


# --- media -----------------------------------------------------------------


def serialize_medium(medium: CoolPropMedium) -> dict:
    """Capture the constructor-relevant state of a `CoolPropMedium`."""
    return {
        "fluid": medium.medium,
        "backend": medium.backend,
        "disable_warnings": bool(getattr(medium, "disable_warnings", False)),
        "scalar_cache_maxsize": getattr(medium, "scalar_cache_maxsize", None),
    }


def make_medium(spec: dict) -> CoolPropMedium:
    """Rebuild a `CoolPropMedium` from a media-table entry."""
    if "fluid" not in spec:
        raise KeyError("medium spec is missing the 'fluid' field")
    kwargs = {}
    if "backend" in spec and spec["backend"] is not None:
        kwargs["backend"] = spec["backend"]
    if "disable_warnings" in spec:
        kwargs["disable_warnings"] = bool(spec["disable_warnings"])
    if spec.get("scalar_cache_maxsize") is not None:
        kwargs["scalar_cache_maxsize"] = spec["scalar_cache_maxsize"]
    return CoolPropMedium(spec["fluid"], **kwargs)


def package_for_class(cls: type) -> str | None:
    """The top-level import package of a class's defining module, or ``None``
    for hydrogen's own classes (which need no external requirement)."""
    top = (cls.__module__ or "").split(".")[0]
    if top in ("", "hydrogen"):
        return None
    return top


def package_requirement(import_path: str) -> dict:
    """A ``requires.packages`` entry for an installed package, pinning the
    currently-installed version as the minimum (best-effort)."""
    entry = {"import": import_path}
    ver = _module_version(sys.modules.get(import_path))
    if ver is not None:
        entry["min_version"] = ver
    return entry


# --- component catalog (for UIs that emit specs) ---------------------------


def _json_safe_default(value):
    """A spec-friendly representation of a constructor default."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def _first_doc_line(cls: type) -> str:
    doc = (cls.__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


# Map a Python type / value to a UI-friendly scalar label.
_TYPE_LABELS = {bool: "bool", int: "integer", float: "float", str: "string"}
_SCALAR_BY_NAME = {"bool": "bool", "int": "integer",
                   "float": "float", "str": "string"}


def _value_type_label(value) -> str:
    # bool is a subclass of int, so test it first.
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return "unknown"


def _value_class_map() -> dict:
    """`{name -> class}` of every serializable value object (lazy import)."""
    from .values import _value_classes

    return _value_classes()


def _resolved_hints(func) -> dict:
    """`typing.get_type_hints` for a callable, resolving stringized (PEP 563)
    annotations to real objects; an empty dict if they can't be evaluated.

    ``include_extras=True`` keeps any ``Annotated[...]`` metadata (e.g. a
    `ParamSpec`) intact; :func:`_describe_type` unwraps to the underlying type."""
    try:
        return typing.get_type_hints(func, include_extras=True)
    except Exception:
        return {}


def _object_descriptor(cls_or_name) -> dict:
    """Type descriptor for a value-object annotation.  When the annotation is an
    abstract base (e.g. ``PermeationFlux``), the concrete serializable subtypes
    are listed under ``value_types``."""
    vmap = _value_class_map()
    if isinstance(cls_or_name, type):
        name = cls_or_name.__name__
        concrete = sorted(n for n, c in vmap.items()
                          if c is cls_or_name or issubclass(c, cls_or_name))
    else:
        name = cls_or_name
        concrete = sorted(n for n, c in vmap.items()
                          if name in {b.__name__ for b in c.__mro__})
    desc = {"type": "object", "value_type": name}
    if name not in vmap and concrete:
        desc["value_types"] = concrete
    return desc


def _describe_type(annotation):
    """A rich type descriptor for a resolved annotation, or ``None`` when there
    is no usable annotation (the caller then infers from the default value).

    ``type`` is one of: ``bool`` / ``integer`` / ``float`` / ``string``
    (scalars), ``medium`` (a `CoolPropMedium` reference), ``object`` (a nested
    value object -- carries ``value_type`` and, for an abstract base,
    ``value_types``), ``list`` (carries ``item``), ``enum`` (carries
    ``choices``) or ``unknown``.  ``nullable`` flags an ``Optional`` argument.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return None
    if isinstance(annotation, str):
        return _describe_str_type(annotation)
    # Unwrap `Annotated[T, ...]` to its underlying type T (the metadata, e.g. a
    # ParamSpec, is consumed separately via `merged_param_specs`).
    if hasattr(annotation, "__metadata__"):
        annotation = typing.get_args(annotation)[0]

    origin = typing.get_origin(annotation)
    union_types = (typing.Union, getattr(types, "UnionType", typing.Union))
    if origin in union_types:
        args = typing.get_args(annotation)
        nullable = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _describe_type(non_none[0]) or {"type": "unknown"}
            if nullable:
                inner["nullable"] = True
            return inner
        return {"type": "unknown", "nullable": nullable}
    if origin is typing.Literal:
        return {"type": "enum", "choices": list(typing.get_args(annotation))}
    if origin in (list, tuple) or annotation in (list, tuple):
        args = typing.get_args(annotation)
        item = (_describe_type(args[0]) if args else None) or {"type": "unknown"}
        return {"type": "list", "item": item}
    if isinstance(annotation, type):
        if annotation in _TYPE_LABELS:
            return {"type": _TYPE_LABELS[annotation]}
        if issubclass(annotation, CoolPropMedium):
            return {"type": "medium"}
        vmap = _value_class_map()
        if annotation.__name__ in vmap or any(
                issubclass(c, annotation) for c in vmap.values()):
            return _object_descriptor(annotation)
        return {"type": annotation.__name__}
    return {"type": getattr(annotation, "__name__", "unknown")}


_OPTIONAL_RE = re.compile(r"^Optional\[(.+)\]$")
_LIST_RE = re.compile(r"^(?:list|List|Sequence|tuple|Tuple)\[(.+)\]$")


def _describe_str_type(s: str) -> dict:
    """Best-effort descriptor for a *stringized* annotation -- the fallback when
    `typing.get_type_hints` cannot resolve a `from __future__ import annotations`
    module."""
    s = s.strip()
    nullable = False
    m = _OPTIONAL_RE.match(s)
    if m:
        s, nullable = m.group(1).strip(), True
    elif s.endswith("| None"):
        s, nullable = s[: -len("| None")].strip(), True

    lm = _LIST_RE.match(s)
    if lm:
        desc = {"type": "list", "item": _describe_str_type(lm.group(1))}
    elif s in _SCALAR_BY_NAME:
        desc = {"type": _SCALAR_BY_NAME[s]}
    elif s == "CoolPropMedium":
        desc = {"type": "medium"}
    else:
        vmap = _value_class_map()
        base_names = {b.__name__ for c in vmap.values() for b in c.__mro__}
        if s in vmap or s in base_names:
            desc = _object_descriptor(s)
        else:
            desc = {"type": s}
    if nullable:
        desc["nullable"] = True
    return desc


# Descriptor keys (beyond name/type/required/default) carried into a catalog
# parameter entry when present.
_DESC_EXTRAS = ("nullable", "value_type", "value_types", "item", "choices")

# Physical units for the constructor params shared across many components,
# keyed by exact param name (these names are unit-consistent everywhere they
# appear).  ``"1"`` denotes a dimensionless quantity.  Component- or value-
# object-specific units (and any that would be ambiguous here, e.g. a wall's
# ``k`` vs. a control gain ``k``) live on the class as ``_spec_units`` and take
# precedence.
_COMMON_UNITS = {
    # pressures
    "p_init": "Pa", "p_source": "Pa", "p_ext": "Pa", "p_ambient": "Pa",
    "p_partial": "Pa", "p_in_init": "Pa", "p_out_init": "Pa", "p_eps": "Pa",
    "dp_eps": "Pa", "p_set": "Pa",
    # temperatures
    "T_init": "K", "T_source": "K", "T_set": "K", "T_ext": "K", "T_inf": "K",
    "T_outer": "K", "T_ambient": "K", "T_wall_init": "K",
    # lengths
    "D": "m", "L": "m", "epsilon": "m", "z_in": "m", "z_out": "m",
    "length": "m", "r_in": "m", "r_out": "m", "thickness": "m",
    # areas / flows / coefficients
    "A": "m^2", "m_flow": "kg/s", "m_dot_eps": "kg/s",
    "h": "W/(m^2*K)", "h_ext": "W/(m^2*K)", "Q_flow": "W", "G": "W/K",
    # dimensionless
    "opening": "1", "xT": "1", "gamma": "1", "n_segments": "1", "n_nodes": "1",
    "solubility_exponent": "1",
    # control: time / frequency / angle
    "duration": "s", "start_time": "s", "Tf": "s", "freq": "Hz", "phase": "rad",
}


def _param_units(cls: type) -> dict:
    """Unit map for ``cls``'s params: shared defaults overlaid with the class's
    own ``_spec_units`` (which wins)."""
    units = dict(_COMMON_UNITS)
    units.update(getattr(cls, "_spec_units", {}) or {})
    return units


def _param_entry(pname, p, hints, spec, choices, units) -> dict:
    """One ``{name, type, required, default, ...}`` parameter descriptor.

    ``spec`` is the class's :class:`~hydrogen.paramspec.ParamSpec` for this arg
    (the single source of truth) or ``None``; ``choices`` / ``units`` are the
    legacy ``_spec_choices`` / ``_COMMON_UNITS`` fallbacks used when the spec
    omits them, so not-yet-migrated components keep working.
    """
    ann = hints.get(pname, p.annotation)
    desc = _describe_type(ann)
    required = p.default is inspect.Parameter.empty
    if desc is None:
        desc = {"type": _value_type_label(p.default) if not required
                else "unknown"}
    entry = {
        "name": pname,
        "type": desc.get("type", "unknown"),
        "required": required,
        "default": None if required else _json_safe_default(p.default),
    }
    for extra in _DESC_EXTRAS:
        if extra in desc:
            entry[extra] = desc[extra]

    # A required arg may still carry a suggested UI/template default via its
    # ParamSpec (the constructor stays the validator, so `required` is unchanged).
    if spec is not None and spec.has_default:
        entry["default"] = _json_safe_default(spec.default)

    # Single source of truth (PARAMS) first, then legacy fallbacks.
    extras = spec.catalog_extras() if spec is not None else {}
    choice_vals = extras.get("choices") or (
        list(choices[pname]) if pname in choices else None)
    if choice_vals:
        entry["choices"] = choice_vals
        if entry["type"] in ("unknown", "string"):
            entry["type"] = "enum"
    unit = extras.get("unit") or units.get(pname)
    if unit:
        entry["unit"] = unit
    for key in ("description", "relevant_when", "required_when", "ui_label"):
        if key in extras:
            entry[key] = extras[key]
    return entry


def _constructor_info(cls: type):
    """(parameters, needs_medium) for a component's (or value object's)
    ``__init__``.

    ``parameters`` is an ordered list of descriptor dicts -- always
    ``{name, type, required, default}`` and, where applicable, the richer
    ``unit`` / ``description`` / ``choices`` / ``relevant_when`` /
    ``required_when`` / ``nullable`` / ``value_type`` / ``value_types`` /
    ``item`` keys.  Field metadata is sourced from the class's ``PARAMS``
    (:class:`~hydrogen.paramspec.ParamSpec`, the single source of truth), with
    :data:`_COMMON_UNITS` / ``_spec_choices`` as fallbacks.  The ``medium``
    argument is reported separately via ``needs_medium``.
    """
    params = []
    needs_medium = False
    sig = inspect.signature(cls.__init__)
    hints = _resolved_hints(cls.__init__)
    specs = merged_param_specs(cls)
    choices = dict(getattr(cls, "_spec_choices", {}) or {})
    units = _param_units(cls)
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if pname == "medium":
            needs_medium = True
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        params.append(
            _param_entry(pname, p, hints, specs.get(pname), choices, units))
    return params, needs_medium


def component_catalog(domain: str | None = None) -> list[dict]:
    """A machine-readable catalog of every shipped component, for tooling/UIs.

    Each entry describes one component leaf::

        {
          "type": "hydrogen.control.Sine",      # canonical full name to use in a spec
          "name": "Sine",
          "domain": "control",
          "summary": "Sine signal: ...",         # first docstring line
          "needs_medium": False,                 # requires a `medium` reference
          "parameters": [                         # constructor args -> spec "params"
            {"name": "amplitude", "type": "float", "required": False, "default": 1.0},
            ...
          ],
          "literals": [                           # _cache_key_flags: structural
            {"name": "dynamic", "type": "bool", "default": True},  # toggles that
          ],                                      #   change the equation set
        }

    The ``type`` on each parameter/literal tells a UI what kind of value to
    supply: ``bool`` / ``integer`` / ``float`` / ``string`` (scalars),
    ``medium`` (needs a `medium` reference), ``object`` (a nested value object;
    the entry also carries ``value_type`` and, for an abstract base,
    ``value_types`` -- see :func:`value_object_catalog`), ``list`` (carries an
    ``item`` sub-descriptor), ``enum`` (carries ``choices``) or ``unknown``.
    Optional ``object`` / list-element params also carry ``nullable``.  A
    physical param additionally carries a ``unit`` string (e.g. ``"Pa"`` /
    ``"K"`` / ``"m"``; ``"1"`` for dimensionless) so a UI can label the input.
    Use :func:`spec_template` to get a ready-to-fill spec entry for one
    component.

    Entries are sorted alphabetically by ``type``.  Pass ``domain`` (e.g.
    ``"thermal"``) to restrict the listing to one physics domain; omit it (or
    pass ``None``) to list everything available.
    """
    rows = [
        _component_entry(obj)
        for obj in _iter_builtin_components()
        if domain is None or (domain_of(obj) or "").lower() == domain.lower()
    ]
    rows.sort(key=lambda r: r["type"])
    return rows


def _component_entry(obj: type) -> dict:
    """The :func:`component_catalog` row for one component class."""
    params, needs_medium = _constructor_info(obj)
    by_name = {p["name"]: p for p in params}
    specs = merged_param_specs(obj)
    literals = []
    for lname in cache_key_flag_names(obj):
        src = by_name.get(lname)
        sp = specs.get(lname)
        lit = {
            "name": lname,
            "type": src["type"] if src else "unknown",
            "default": src["default"] if src else None,
        }
        # A structural flag inherits the description / choices of its matching
        # constructor param (or, for a flag that is not a direct __init__ arg,
        # straight from the class's ParamSpec).
        desc = (src or {}).get("description") or (sp.description if sp else "")
        if desc:
            lit["description"] = desc
        choices = (src or {}).get("choices") or (
            list(sp.choices) if sp and sp.choices else None)
        if choices:
            lit["choices"] = choices
        literals.append(lit)
    return {
        "type": full_type_name(obj),
        "name": obj.__name__,
        "domain": domain_of(obj),
        "module": obj.__module__,
        "category": category_of(obj),
        "summary": _first_doc_line(obj),
        "needs_medium": needs_medium,
        # UI symbol declared on the class as ``UI_ICON`` (filename in
        # ``hydrogen/components/icons/``); ``None`` -> generic box rendering.
        "icon": getattr(obj, "UI_ICON", None),
        "parameters": params,
        "literals": literals,
    }


def available_domains() -> list[str]:
    """Sorted list of physics-domain names that ship components."""
    return sorted({r["domain"] for r in component_catalog() if r["domain"]})


# --- value objects (structured params) & spec templates --------------------


def _value_object_entry(name: str, cls: type) -> dict:
    fields, _ = _constructor_info(cls)
    entry = {"value_type": name, "summary": _first_doc_line(cls), "fields": fields}
    # Optional ready-made instances a UI can offer as a choice list.  A class
    # advertises them via a ``PRESETS = {label: instance}`` mapping; each is
    # serialized to its value spec so a UI can fill the form from it.
    presets = getattr(cls, "PRESETS", None)
    if presets:
        entry["presets"] = [
            {"name": label,
             "spec": obj.to_spec() if hasattr(obj, "to_spec") else dict(obj)}
            for label, obj in presets.items()
        ]
    return entry


def value_object_catalog() -> list[dict]:
    """Catalog of the structured *value objects* some component params take.

    Most params are JSON scalars, but a few are nested objects -- e.g.
    ``Pipe(layers=[WallLayer(...)])`` where a `WallLayer` carries a
    `WallMaterial` and an optional `PermeationFlux` (`SteadyRichardson` /
    `TransientDiffusion`, itself carrying a `TransportFit` + `Permeant`).  These
    serialize as ``{"__type__": "<Name>", ...}`` dicts; this catalog describes
    their fields so a UI can render / validate them.  Each entry::

        {"value_type": "WallLayer",
         "summary": "...",                       # first docstring line
         "fields": [                              # like catalog "parameters"
           {"name": "material", "type": "object",
            "value_type": "WallMaterial", "required": True},
           {"name": "thickness", "type": "float",
            "required": True, "unit": "m"},
           {"name": "permeation", "type": "object",
            "value_type": "PermeationFlux", "nullable": True,
            "value_types": ["SteadyRichardson", "TransientDiffusion"],
            "required": False, "default": None},
           ...]}

    Fields use the same descriptor shape (incl. ``unit`` / ``choices``) as
    :func:`component_catalog` parameters.  Sorted alphabetically by
    ``value_type``.
    """
    return [_value_object_entry(name, cls)
            for name, cls in sorted(_value_class_map().items())]


def value_object_spec(value_type: str) -> dict:
    """The single :func:`value_object_catalog` entry for ``value_type``
    (e.g. ``"WallLayer"``).  Raises ``KeyError`` for an unknown type."""
    vmap = _value_class_map()
    cls = vmap.get(value_type)
    if cls is None:
        raise KeyError(
            f"unknown value type {value_type!r}; known: {sorted(vmap)}")
    return _value_object_entry(value_type, cls)


def _concrete_value_type(value_type: str) -> str:
    """Resolve an abstract value-object name to a concrete serializable one
    (the first registered subtype); concrete names pass through."""
    vmap = _value_class_map()
    if value_type in vmap:
        return value_type
    subs = sorted(n for n, c in vmap.items()
                  if value_type in {b.__name__ for b in c.__mro__})
    if not subs:
        raise KeyError(f"no serializable value type for {value_type!r}")
    return subs[0]


def _lookup_component(type_name: str) -> type:
    """Resolve a catalog ``type`` (canonical full name or bare class name)."""
    reg = builtin_registry()
    reg.update(_EXTRA_REGISTRY)
    cls = reg.get(type_name)
    if cls is None:
        cls = next((c for c in reg.values() if c.__name__ == type_name), None)
    if cls is None:
        raise KeyError(
            f"unknown component type {type_name!r}; "
            f"see component_catalog() for valid names")
    return cls


def _field_placeholder(field: dict):
    """A fill-in value for one parameter/field descriptor (for templates).

    Required scalars are ``None``; optional fields carry their default; object
    fields recurse into a nested :func:`value_template` (a required object) or
    ``None`` (an optional/nullable one); list-of-object fields seed one element.
    """
    ftype = field.get("type")
    required = field.get("required", False)
    if ftype == "list":
        item = field.get("item") or {}
        if item.get("type") == "object" and item.get("value_type"):
            return [value_template(item["value_type"])]
        return []
    if ftype == "object":
        if required and not field.get("nullable") and field.get("value_type"):
            return value_template(field["value_type"])
        return None
    if ftype == "enum":
        default = field.get("default")
        if default is not None:
            return default
        choices = field.get("choices") or []
        return choices[0] if choices else None
    # Scalars: use the (signature- or ParamSpec-supplied) default if any,
    # else None for a required field with no suggestion.
    return field.get("default")


def value_template(value_type: str) -> dict:
    """A fill-in ``{"__type__": ..., ...}`` skeleton for a value object.

    Required scalar fields are ``None`` placeholders; optional fields carry
    their default; nested value objects recurse (an abstract field such as
    ``PermeationFlux`` resolves to a concrete subtype).  Feed the result back,
    once filled, as a ``params`` value (or list element) in a system spec.
    """
    name = _concrete_value_type(value_type)
    spec = value_object_spec(name)
    # If the type ships presets, seed from the first one so the template comes
    # fully filled (the alternative is blank required placeholders).
    presets = spec.get("presets")
    if presets:
        return dict(presets[0]["spec"])
    out = {"__type__": name}
    for field in spec["fields"]:
        out[field["name"]] = _field_placeholder(field)
    return out


def spec_template(type_name: str) -> dict:
    """A fill-in component entry for a system spec's ``components`` map.

    ``type_name`` is a canonical full name from :func:`component_catalog`
    (e.g. ``"hydrogen.thermofluid.Pipe"``); a bare class name also resolves.
    Returns::

        {"type": <type_name>,
         "medium": None,          # only when the component needs a medium
         "params": {<name>: <placeholder-or-default>, ...}}

    Required scalars are ``None`` (the UI fills them in); optional params carry
    their default; value-object params get a nested :func:`value_template`
    skeleton.  Consult :func:`component_catalog` / :func:`value_object_catalog`
    for the per-field types, ``choices`` and ``value_types`` to render.
    """
    cls = _lookup_component(type_name)
    params, needs_medium = _constructor_info(cls)
    entry = {"type": type_name}
    if needs_medium:
        entry["medium"] = None
    entry["params"] = {p["name"]: _field_placeholder(p) for p in params}
    return entry


def _expand_value_type(value_type: str, seen: frozenset) -> dict | None:
    """Recursively resolve a value object's field descriptors (``None`` if the
    type is unknown or already on the current expansion path -- cycle guard)."""
    if value_type in seen:
        return None
    try:
        vspec = value_object_spec(value_type)
    except KeyError:
        return None
    seen = seen | {value_type}
    out = {
        "value_type": value_type,
        "summary": vspec["summary"],
        "fields": [_attach_value_specs(f, seen) for f in vspec["fields"]],
    }
    if "presets" in vspec:
        out["presets"] = vspec["presets"]
    return out


def _attach_value_specs(field: dict, seen: frozenset) -> dict:
    """Return a copy of a param/field descriptor with any nested value object
    expanded in place: a concrete ``object`` field gains a ``value_spec``; an
    abstract one gains ``options`` keyed by concrete type; a list recurses into
    its ``item``."""
    out = dict(field)
    if out.get("type") == "list" and isinstance(out.get("item"), dict):
        out["item"] = _attach_value_specs(out["item"], seen)
        return out
    vt = out.get("value_type")
    if not vt:
        return out
    options = out.get("value_types")
    if options:
        out["options"] = {n: _expand_value_type(n, seen) for n in options}
    else:
        out["value_spec"] = _expand_value_type(vt, seen)
    return out


def component_spec(type_name: str) -> dict:
    """The complete, self-contained spec for one component -- everything a UI
    needs from a single call.

    Like a :func:`component_catalog` entry (``type`` / ``name`` / ``domain`` /
    ``summary`` / ``needs_medium`` / ``parameters`` / ``literals``, each
    parameter carrying ``type`` / ``unit`` / ``description`` / ``required`` /
    ``default`` / ``choices`` / ``relevant_when`` ...), but additionally:

      * every nested value-object param is *expanded* in place -- a concrete
        ``object`` field gains a ``value_spec`` (its own fully-described
        fields), an abstract one gains ``options`` keyed by concrete type, and a
        list-of-objects recurses into ``item`` -- so the whole tree (e.g.
        ``Pipe -> WallLayer -> WallMaterial`` / ``PermeationFlux -> TransportFit
        -> Permeant``) comes back at once, with descriptions throughout; and
      * a ready-to-fill ``template`` (see :func:`spec_template`) is embedded,
        so the UI has both the rendering metadata and the editable skeleton.

    ``type_name`` is a canonical full name (or bare class name).  Raises
    ``KeyError`` for an unknown component.
    """
    entry = _component_entry(_lookup_component(type_name))
    entry["parameters"] = [
        _attach_value_specs(p, frozenset()) for p in entry["parameters"]
    ]
    entry["template"] = spec_template(entry["type"])
    return entry


def _format_params(params: list[dict]) -> str:
    if not params:
        return "-"
    out = []
    for p in params:
        base = f"{p['name']}:{p['type']}"
        if p["required"]:
            out.append(base + "*")
        else:
            out.append(f"{base}={p['default']!r}")
    return ", ".join(out)


def _format_literals(literals: list[dict]) -> str:
    if not literals:
        return "-"
    return ", ".join(f"{lit['name']}:{lit['type']}" for lit in literals)


def format_component_catalog(domain: str | None = None) -> str:
    """A human-readable text table of :func:`component_catalog`.

    ``*`` marks a required parameter; the ``LITERALS`` column lists the
    structural flags (``_cache_key_flags``) that change a component's equation
    set.  Pass ``domain`` to filter; omit to list all.
    """
    rows = component_catalog(domain)
    if not rows:
        return f"(no components found for domain {domain!r})"

    type_w = max(len("TYPE"), max(len(r["type"]) for r in rows))
    med_w = len("MEDIUM")
    lit_w = max(len("LITERALS"), max(len(_format_literals(r["literals"])) for r in rows))

    header = (f"{'TYPE':<{type_w}}  {'MEDIUM':<{med_w}}  "
              f"{'LITERALS':<{lit_w}}  PARAMETERS (name:type, * = required)")
    lines = [header, "-" * len(header)]
    current_domain = object()
    for r in rows:
        if r["domain"] != current_domain:
            current_domain = r["domain"]
            lines.append(f"# {current_domain}")
        med = "yes" if r["needs_medium"] else "-"
        lines.append(
            f"{r['type']:<{type_w}}  {med:<{med_w}}  "
            f"{_format_literals(r['literals']):<{lit_w}}  "
            f"{_format_params(r['parameters'])}"
        )
    return "\n".join(lines)
