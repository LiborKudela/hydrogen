"""Dict <-> `Model` codec for hydrogen systems (phased core).

This phase round-trips the *structure* of a system:

  * a top-level header (``hydrogen_version`` + ``schema_version``),
  * a ``media`` table (shared `CoolPropMedium` definitions, referenced by key),
  * an optional ``requires`` block (user component packages),
  * ``components`` -- each either a registered leaf
    (``{"type": "StraightPipe", "params": {...}}``) or an inline composite
    (``{"type": "Model", "components": {...}, "connections": [...],
    "exposed_ports": {...}}``) that nests recursively,
  * raw ``variables`` (``Variable`` / ``DifferentialVariable`` / ``Parameter``)
    declared directly on a composite, and
  * ``connections`` -- port-to-port wires (``"a.outlet" -> "b.inlet"``).

It does NOT yet handle ``equations`` or time-dependent ``inputs`` (those need
an expression parser and are a later phase); specs that use those keys are
rejected with a clear message rather than silently losing physics.

The dict form is canonical; JSON is a thin codec over it (see the package
``__init__``).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np

from ..model import DifferentialVariable, Model, Parameter, Variable
from .errors import SerializationError, SystemSpecError
from .registry import (
    SCHEMA_VERSION,
    build_registry,
    make_medium,
    package_for_class,
    package_requirement,
    serialize_medium,
)
from .values import decode_value, encode_value, value_spec_ok

# Variable "kind" discriminator <-> class.  Exact-type keyed (so a
# `ParameterAlias` or other subclass is never silently misclassified).
_VAR_CLASSES = {
    "Variable": Variable,
    "DifferentialVariable": DifferentialVariable,
    "Parameter": Parameter,
}
_KIND_OF_TYPE = {cls: kind for kind, cls in _VAR_CLASSES.items()}

_JSON_SCALAR = (int, float, str, bool, type(None))

# Composite keys that belong to a later phase; rejected on load for now.
_UNSUPPORTED_KEYS = ("equations", "inputs")

_EMPTY = inspect.Parameter.empty


def _hydrogen_version() -> str:
    from .. import __version__

    return __version__


def _jsonable(value):
    """Coerce numpy scalars to plain Python; raise on anything non-scalar."""
    if isinstance(value, _JSON_SCALAR):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise SerializationError(
        f"value {value!r} of type {type(value).__name__} is not a JSON scalar"
    )


# ---------------------------------------------------------------------------
# Constructor-parameter (de)serialization
# ---------------------------------------------------------------------------


def _param_spec(cls):
    """(allowed, required, has_medium) constructor-parameter info for ``cls``."""
    sig = inspect.signature(cls.__init__)
    allowed, required = set(), set()
    has_medium = False
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if pname == "medium":
            has_medium = True
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        allowed.add(pname)
        if p.default is _EMPTY:
            required.add(pname)
    return allowed, required, has_medium


def serialize_params(component) -> dict:
    """Recover a leaf component's constructor kwargs in JSON-able form.

    Relies on the package convention that each constructor argument is stored as
    a like-named attribute.  Scalars (and numpy scalars) are emitted directly;
    structured value objects (and lists of them) that implement ``to_spec`` --
    e.g. ``WallLayer`` / permeation flux models -- are encoded recursively (see
    :mod:`hydrogen.serialization.values`).  An optional kwarg that is not stored
    is simply omitted (the loader falls back to the constructor default); a
    *required* one that is missing is a hard error (needs a custom hook).
    """
    sig = inspect.signature(type(component).__init__)
    out = {}
    for pname, p in sig.parameters.items():
        if pname in ("self", "medium"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if not hasattr(component, pname):
            if p.default is not _EMPTY:
                continue
            raise SerializationError(
                f"{type(component).__name__}: required constructor argument "
                f"{pname!r} is not stored as an attribute, so it cannot be "
                f"serialized automatically (a custom hook would be needed)."
            )
        out[pname] = _encode_param(getattr(component, pname))
    return out


def _encode_param(value):
    """Encode one constructor-param value (scalar, numpy scalar, list, or a
    structured value object implementing ``to_spec``)."""
    if isinstance(value, _JSON_SCALAR):
        return value
    if isinstance(value, (list, tuple)):
        return [_encode_param(v) for v in value]
    if callable(getattr(value, "to_spec", None)):
        return encode_value(value)
    return _jsonable(value)  # numpy scalar -> python; else raises a clear error


def _validate_params(cls, params, where, errors):
    allowed, required, _ = _param_spec(cls)
    for key in set(params) - allowed:
        errors.append(
            f"{where}: unknown param {key!r} for {cls.__name__}; "
            f"allowed: {sorted(allowed)}"
        )
    for key in required - set(params):
        errors.append(f"{where}: missing required param {key!r} for {cls.__name__}")
    for key, val in params.items():
        if key in allowed and not value_spec_ok(val):
            errors.append(
                f"{where}: param {key!r} value {val!r} is not a JSON scalar "
                f"or a known value spec"
            )


# ---------------------------------------------------------------------------
# Variables / Parameters
# ---------------------------------------------------------------------------


def _validate_var(name, vspec, where, errors):
    if not isinstance(vspec, dict):
        errors.append(f"{where}.variables.{name}: entry must be an object")
        return
    kind = vspec.get("kind")
    if kind not in _VAR_CLASSES:
        errors.append(
            f"{where}.variables.{name}: unknown kind {kind!r}; "
            f"expected one of {sorted(_VAR_CLASSES)}"
        )
    if "value" not in vspec:
        errors.append(f"{where}.variables.{name}: missing 'value'")
    elif not isinstance(vspec["value"], _JSON_SCALAR):
        errors.append(f"{where}.variables.{name}: 'value' must be a JSON scalar")


def _make_var(vspec):
    cls = _VAR_CLASSES[vspec["kind"]]
    value = vspec["value"]
    unit = vspec.get("unit")
    if cls is Parameter:
        return Parameter(value, unit)
    return cls(value, unit, atol=vspec.get("atol"), scale=vspec.get("scale"))


def _var_to_dict(component) -> dict:
    out = {"kind": _KIND_OF_TYPE[type(component)], "value": _jsonable(component.value)}
    if component.unit is not None:
        out["unit"] = component.unit
    # atol / scale only exist on Variable (not Parameter).
    if isinstance(component, Variable):
        if getattr(component, "atol", None) is not None:
            out["atol"] = _jsonable(component.atol)
        if getattr(component, "scale", None) is not None:
            out["scale"] = _jsonable(component.scale)
    return out


# ---------------------------------------------------------------------------
# Load: validation
# ---------------------------------------------------------------------------


class _LoadCtx:
    def __init__(self, registry, media):
        self.registry = registry
        self.media = media


def _validate_connection(conn, where, errors):
    if not isinstance(conn, dict) or "from" not in conn or "to" not in conn:
        errors.append(f"{where}: connection must be an object with 'from' and 'to'")
        return
    for side in ("from", "to"):
        ref = conn[side]
        if not isinstance(ref, str) or "." not in ref:
            errors.append(
                f"{where}: connection {side} {ref!r} must be 'component.port'"
            )


def _validate_exposed(exposed, where, errors):
    if not isinstance(exposed, dict):
        errors.append(f"{where}.exposed_ports: must be an object")
        return
    for boundary, target in exposed.items():
        if not isinstance(target, str) or "." not in target:
            errors.append(
                f"{where}.exposed_ports.{boundary}: target {target!r} must be "
                f"'child.port'"
            )


def _validate_composite(cspec, ctx, where, errors):
    for key in _UNSUPPORTED_KEYS:
        if key in cspec:
            errors.append(
                f"{where}: '{key}' is not supported yet (planned for a later "
                f"serialization phase); remove it or define this composite in "
                f"Python instead"
            )
    if "medium" in cspec and cspec["medium"] not in ctx.media:
        errors.append(f"{where}: unknown medium {cspec['medium']!r}")
    for vname, vspec in cspec.get("variables", {}).items():
        _validate_var(vname, vspec, where, errors)
    for cname, child in cspec.get("components", {}).items():
        _validate_component(cname, child, ctx, where, errors)
    for conn in cspec.get("connections", []):
        _validate_connection(conn, where, errors)
    _validate_exposed(cspec.get("exposed_ports", {}), where, errors)


def _validate_component(name, cspec, ctx, parent_path, errors):
    where = f"{parent_path}.{name}"
    if not isinstance(cspec, dict):
        errors.append(f"{where}: component entry must be an object")
        return
    ctype = cspec.get("type")
    if ctype is None:
        errors.append(f"{where}: missing 'type'")
        return

    if ctype == "Model":
        if "params" in cspec:
            errors.append(f"{where}: an inline composite (type 'Model') cannot carry 'params'")
        _validate_composite(cspec, ctx, where, errors)
        return

    cls = ctx.registry.get(ctype)
    if cls is None:
        known = sorted(k for k in ctx.registry if k.startswith("hydrogen."))
        errors.append(
            f"{where}: unknown component type {ctype!r}. Builtin types: {known}. "
            f"(For a user library, declare it under requires.packages and use a "
            f"namespaced type like 'pkg.ClassName'.)"
        )
        return

    for bad in ("components", "connections", "exposed_ports", "variables"):
        if bad in cspec:
            errors.append(
                f"{where}: leaf component {ctype!r} cannot carry {bad!r} "
                f"(use type 'Model' for an inline composite)"
            )
    allowed, required, has_medium = _param_spec(cls)
    if has_medium and "medium" not in cspec:
        errors.append(f"{where}: component {ctype!r} requires a 'medium' reference")
    if "medium" in cspec and cspec["medium"] not in ctx.media:
        errors.append(f"{where}: unknown medium {cspec['medium']!r}")
    _validate_params(cls, cspec.get("params", {}), where, errors)


def _check_versions(spec, strict, errors):
    sv = spec.get("schema_version")
    if sv is None:
        errors.append("missing 'schema_version'")
    elif sv != SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {sv!r}; this build reads "
            f"schema_version {SCHEMA_VERSION}"
        )
    hv = spec.get("hydrogen_version")
    if hv is not None and hv != _hydrogen_version():
        msg = (
            f"spec hydrogen_version {hv!r} differs from the installed "
            f"{_hydrogen_version()!r}"
        )
        if strict:
            errors.append(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Load: build
# ---------------------------------------------------------------------------


def _build_component(cspec, ctx):
    ctype = cspec["type"]
    medium = ctx.media.get(cspec["medium"]) if "medium" in cspec else None
    if ctype == "Model":
        return _SpecComposite(cspec, ctx, medium=medium)
    cls = ctx.registry[ctype]
    params = {k: decode_value(v) for k, v in cspec.get("params", {}).items()}
    if medium is not None:
        return cls(medium, **params)
    return cls(**params)


class _SpecComposite(Model):
    """A generic `Model` container built from an inline-composite spec.

    Used both for the top-level system and for every ``type: "Model"`` node.
    Children, raw variables, port re-exposure and internal connections all come
    from the spec; the boundary ports are rebound from the named child ports
    exactly like the hand-written composites (e.g. `ConjugatePipe`).
    """

    def __init__(self, spec, ctx, medium=None):
        self._spec = spec
        self._ctx = ctx
        if medium is not None:
            self.medium = medium
        super().__init__()

    def declare_components(self):
        for vname, vspec in self._spec.get("variables", {}).items():
            self.add_component(vname, _make_var(vspec))
        for cname, cspec in self._spec.get("components", {}).items():
            self.add_component(cname, _build_component(cspec, self._ctx))
        for boundary, target in self._spec.get("exposed_ports", {}).items():
            child_name, _, port_name = target.partition(".")
            child = self.components.get(child_name)
            if child is None or port_name not in getattr(child, "ports", {}):
                raise SystemSpecError(
                    f"exposed_ports[{boundary!r}] -> {target!r}: no such child port"
                )
            p = child.ports[port_name]
            self.add_port(boundary, type(p)(
                self,
                channels=dict(p.channels),
                flow_orientation=p.flow_orientation,
                medium=p.medium,
            ))

    def declare_equations(self):
        for conn in self._spec.get("connections", []):
            self._wire(conn)
        return []

    def _wire(self, conn):
        a_name, _, a_port = conn["from"].partition(".")
        b_name, _, b_port = conn["to"].partition(".")
        for cname, pname in ((a_name, a_port), (b_name, b_port)):
            child = self.components.get(cname)
            if child is None:
                raise SystemSpecError(
                    f"connection {conn['from']!r} -> {conn['to']!r}: "
                    f"no component named {cname!r}"
                )
            if pname not in getattr(child, "ports", {}):
                raise SystemSpecError(
                    f"connection {conn['from']!r} -> {conn['to']!r}: component "
                    f"{cname!r} ({type(child).__name__}) has no port {pname!r}"
                )
        self.connect(self[a_name].ports[a_port], self[b_name].ports[b_port])


def from_dict(spec: dict, *, strict_version: bool = False) -> Model:
    """Build a `Model` from a system spec dict, validating it first.

    All structural problems are collected and raised together as a
    :class:`SystemSpecError`.  The returned model is *not* instantiated; call
    ``.instantiate(...)`` on it to assemble and validate the DAE.
    """
    if not isinstance(spec, dict):
        raise SystemSpecError("top-level spec must be an object/dict")

    errors: list[str] = []
    _check_versions(spec, strict_version, errors)

    registry = build_registry(spec, errors)
    media = {}
    for key, mspec in spec.get("media", {}).items():
        try:
            media[key] = make_medium(mspec)
        except Exception as exc:  # noqa: BLE001 - surface as a spec error
            errors.append(f"media[{key!r}]: {exc}")

    ctx = _LoadCtx(registry=registry, media=media)
    _validate_composite(spec, ctx, "<root>", errors)

    if errors:
        raise SystemSpecError(errors)

    return _SpecComposite(spec, ctx, medium=None)


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------


class _DumpCtx:
    def __init__(self):
        self.builtin = None  # class -> bare name (lazy)
        self.media = {}      # key -> serialized medium
        self._media_keys = {}  # id(medium) -> key
        self.packages = {}   # import_path -> requires entry

    def _builtin_by_class(self):
        if self.builtin is None:
            from .registry import builtin_registry
            self.builtin = {cls: name for name, cls in builtin_registry().items()}
        return self.builtin

    def type_name(self, cls) -> str:
        bare = self._builtin_by_class().get(cls)
        if bare is not None:
            return bare
        pkg = package_for_class(cls)
        if pkg is None:
            # hydrogen-internal but not a re-exported component; emit the bare
            # name and rely on it being registered/importable on load.
            return cls.__name__
        self.packages.setdefault(pkg, package_requirement(pkg))
        return f"{pkg}.{cls.__name__}"

    def add_medium(self, medium) -> str:
        existing = self._media_keys.get(id(medium))
        if existing is not None:
            return existing
        # Key by fluid name, disambiguating duplicates with a suffix.
        base = getattr(medium, "medium", "medium")
        key = base
        i = 1
        while key in self.media:
            i += 1
            key = f"{base}_{i}"
        self.media[key] = serialize_medium(medium)
        self._media_keys[id(medium)] = key
        return key


def _is_var_component(component) -> bool:
    return type(component) in _KIND_OF_TYPE


def _local_names(model) -> dict:
    return {id(c): n for n, c in model.components.items()}


def _connections_to_list(model) -> list:
    names = _local_names(model)
    conns, seen = [], set()
    for cname, comp in model.components.items():
        for pname, port in getattr(comp, "ports", {}).items():
            if not getattr(port, "is_connected", False):
                continue
            other = port._connected_to
            other_name = names.get(id(other.owner))
            if other_name is None:
                # Partner is not a direct child of this model (e.g. a boundary
                # wired one level up); that connection belongs to that level.
                continue
            a = f"{cname}.{pname}"
            b = f"{other_name}.{other.name}"
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            conns.append({"from": a, "to": b})
    return conns


def _same_channels(port_a, port_b) -> bool:
    a, b = port_a.channels, port_b.channels
    if set(a) != set(b):
        return False
    return all(a[k] is b[k] for k in a)


def _exposed_ports_to_dict(model) -> dict:
    out = {}
    for pname, port in getattr(model, "ports", {}).items():
        for cname, comp in model.components.items():
            matched = None
            for child_pn, child_port in getattr(comp, "ports", {}).items():
                if child_port is port:
                    continue
                if _same_channels(port, child_port):
                    matched = f"{cname}.{child_pn}"
                    break
            if matched is not None:
                out[pname] = matched
                break
    return out


def _leaf_to_dict(component, ctx) -> dict:
    entry = {"type": ctx.type_name(type(component))}
    medium = getattr(component, "medium", None)
    if medium is not None:
        entry["medium"] = ctx.add_medium(medium)
    params = serialize_params(component)
    if params:
        entry["params"] = params
    return entry


def _composite_to_dict(model, ctx) -> dict:
    variables, components = {}, {}
    diff_names = {
        n for n, c in model.components.items()
        if isinstance(c, DifferentialVariable)
    }
    for cname, comp in model.components.items():
        # Skip the auto-attached `der_<x>` companion of a DifferentialVariable.
        if cname.startswith("der_") and cname[4:] in diff_names:
            continue
        if _is_var_component(comp):
            variables[cname] = _var_to_dict(comp)
            continue
        # Skip shared parameter aliases and other non-independent leaves.
        if type(comp).__name__ == "ParameterAlias":
            continue
        if not isinstance(comp, Model) or not comp.is_composite():
            # A bare leaf Variable subclass / Input / unknown atomic -> not
            # representable as a structural component in this phase.
            if type(comp).__name__ == "Input":
                raise SerializationError(
                    f"component {cname!r}: Input serialization is a later phase"
                )
            continue
        if isinstance(comp, _SpecComposite):
            entry = {"type": "Model"}
            medium = getattr(comp, "medium", None)
            if medium is not None:
                entry["medium"] = ctx.add_medium(medium)
            entry.update(_composite_to_dict(comp, ctx))
            components[cname] = entry
        else:
            components[cname] = _leaf_to_dict(comp, ctx)

    out = {}
    if variables:
        out["variables"] = variables
    if components:
        out["components"] = components
    conns = _connections_to_list(model)
    if conns:
        out["connections"] = conns
    exposed = _exposed_ports_to_dict(model)
    if exposed:
        out["exposed_ports"] = exposed
    return out


_HEADER_KEYS = ("hydrogen_version", "schema_version")


def to_dict(model: Model) -> dict:
    """Serialize a live `Model` (treated as the root system) to a spec dict.

    Captures media, any required user packages, components (registered leaves
    kept opaque; inline composites recursed), raw variables/parameters, and
    port-level connections.  Equations and time-dependent inputs are out of
    scope for this phase.

    A model produced by :func:`from_dict` already carries its canonical spec,
    so it is echoed back verbatim (with a refreshed version header) -- this
    makes ``to_dict(from_dict(spec))`` exact without needing to instantiate.

    A hand-written `Model` is **wired on demand**: its `declare_equations()` is
    run once (via :meth:`Model.ensure_equations_declared`) so the port-level
    connections are present, unless it was already wired (by an earlier
    `instantiate()` / `declare_equations()` call), in which case nothing extra
    happens.  So a freshly built model can be dumped directly -- no need to
    `instantiate()` or call `declare_equations()` first.
    """
    if isinstance(model, _SpecComposite):
        out = dict(model._spec)
        out["hydrogen_version"] = _hydrogen_version()
        out["schema_version"] = SCHEMA_VERSION
        # Keep the header keys first for readability.
        ordered = {k: out[k] for k in _HEADER_KEYS}
        ordered.update({k: v for k, v in out.items() if k not in _HEADER_KEYS})
        return ordered

    # Wire the model on demand (idempotent) so its port connections are present
    # to dump -- callers no longer have to instantiate / declare_equations first.
    model.ensure_equations_declared()

    ctx = _DumpCtx()
    body = _composite_to_dict(model, ctx)
    out = {
        "hydrogen_version": _hydrogen_version(),
        "schema_version": SCHEMA_VERSION,
    }
    if ctx.packages:
        out["requires"] = {
            "packages": [ctx.packages[k] for k in sorted(ctx.packages)]
        }
    if ctx.media:
        out["media"] = ctx.media
    out.update(body)
    return out
