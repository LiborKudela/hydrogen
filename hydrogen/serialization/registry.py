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
import sys

from ..medium import CoolPropMedium
from ..model import Model

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


def builtin_registry() -> dict[str, type]:
    """The hydrogen-shipped components, keyed by bare class name."""
    # Imported lazily: this module is imported from `hydrogen/__init__`, and we
    # want `hydrogen.components` to be fully initialised first (it is, by the
    # time the serialization subpackage is imported at the end of __init__).
    from .. import components as comps

    reg: dict[str, type] = {}
    for name in getattr(comps, "__all__", []):
        obj = getattr(comps, name, None)
        if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
            reg[name] = obj
    return reg


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
