"""Tests for the phased system (de)serialization layer (`hydrogen.serialization`).

Covers the dict/JSON codec, validation/error aggregation, raw
variables/parameters, registered + inline (`type: "Model"`) composites,
exposed ports, shared media, programmatic `register_component`, and that a
loaded system actually instantiates and solves.
"""

from __future__ import annotations

import json

import pytest

from hydrogen import (
    CoolPropMedium,
    Model,
    ParamSpec,
    Parameter,
    Variable,
    from_dict,
    from_json,
    to_dict,
    to_json,
)
from hydrogen.components.thermofluid.flow import StraightPipe
from hydrogen.components.thermofluid.walls import (
    CylindricalWall,
    FixedTemperature,
    FlatWall,
    ThermalConductor,
)
from hydrogen.paramspec import merged_param_specs
from hydrogen.serialization import (
    SCHEMA_VERSION,
    SystemSpecError,
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


def _thermal_spec():
    return {
        "hydrogen_version": "0.1.0",
        "schema_version": 1,
        "components": {
            "hot": {"type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 400.0}},
            "cold": {"type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 300.0}},
            "cond": {"type": "hydrogen.thermofluid.ThermalConductor", "params": {"G": 5.0}},
        },
        "connections": [
            {"from": "hot.heat", "to": "cond.heat_a"},
            {"from": "cond.heat_b", "to": "cold.heat"},
        ],
    }


# --- dict / JSON round-trips -----------------------------------------------


def test_from_dict_to_dict_echo_is_stable():
    spec = _thermal_spec()
    model = from_dict(spec)
    # A from_dict model echoes its canonical spec (header refreshed).
    assert to_dict(model) == spec


def test_json_round_trip():
    spec = _thermal_spec()
    text = to_json(from_dict(spec))
    assert json.loads(text)["schema_version"] == 1
    model = from_json(text)
    assert sorted(model.components) == ["cold", "cond", "hot"]


def test_reflective_dump_of_handwritten_wired_model():
    class Net(Model):
        def declare_components(self):
            self.add_component("scale", Parameter(2.0, "W/K"))
            self.add_component("probe", Variable(293.15, "K", atol=1e-4))
            self.add_component("hot", FixedTemperature(T_set=350.0))
            self.add_component("cond", ThermalConductor(G=3.0))

        def declare_equations(self):
            self.connect(self["hot"].ports["heat"], self["cond"].ports["heat_a"])
            return []

    net = Net()
    net.declare_equations()  # wire so port connections are present
    d = to_dict(net)

    assert d["variables"]["scale"] == {"kind": "Parameter", "value": 2.0, "unit": "W/K"}
    assert d["variables"]["probe"]["kind"] == "Variable"
    assert d["variables"]["probe"]["atol"] == pytest.approx(1e-4)
    assert d["components"]["hot"] == {
        "type": "hydrogen.thermofluid.FixedTemperature", "params": {"T_set": 350.0}}
    assert {"from": "hot.heat", "to": "cond.heat_a"} in d["connections"]


# --- build / instantiate ----------------------------------------------------


def test_loaded_system_instantiates_and_solves():
    model = from_dict(_thermal_spec())
    model.instantiate(max_remove_trival_passes=5)
    model.initialise(n=1)
    model.solve_dae_step(1.0)  # must not raise


def test_inline_composite_with_exposed_ports_builds_and_shares_medium():
    spec = {
        "hydrogen_version": "0.1.0",
        "schema_version": 1,
        "media": {"air": {"fluid": "air", "backend": "HEOS", "disable_warnings": True}},
        "components": {
            "src": {"type": "hydrogen.thermofluid.AmbientInlet", "medium": "air",
                    "params": {"m_flow": 0.05}},
            "leg": {
                "type": "Model",
                "medium": "air",
                "components": {
                    "pipe": {
                        "type": "hydrogen.thermofluid.StraightPipe",
                        "medium": "air",
                        "params": {"D": 0.05, "L": 5, "epsilon": 1e-4,
                                   "z_in": 0, "z_out": 0, "n_segments": 2},
                    }
                },
                "exposed_ports": {"inlet": "pipe.inlet", "outlet": "pipe.outlet"},
            },
        },
        "connections": [{"from": "src.outlet", "to": "leg.inlet"}],
    }
    model = from_dict(spec)
    # The inline composite exposes a real `inlet` port wired to the source.
    assert "inlet" in model["leg"].ports
    # One shared CoolPropMedium instance (so connect()'s identity check passes).
    assert model["src"].medium is model["leg"]["pipe"].medium


def test_raw_variables_round_trip():
    spec = {
        "hydrogen_version": "0.1.0",
        "schema_version": 1,
        "variables": {
            "p_ref": {"kind": "Parameter", "value": 101325, "unit": "Pa"},
            "x": {"kind": "Variable", "value": 1.0, "unit": "m", "atol": 1e-6},
            "m": {"kind": "DifferentialVariable", "value": 0.0, "unit": "kg"},
        },
        "components": {},
    }
    model = from_dict(spec)
    assert isinstance(model["p_ref"], Parameter)
    assert model["x"].value == 1.0
    # DifferentialVariable auto-attaches its der_ companion.
    assert "der_m" in model.components
    assert to_dict(model)["variables"] == spec["variables"]


# --- programmatic registration ---------------------------------------------


def test_register_component_allows_custom_type():
    class _Gain(Model):
        def __init__(self, k=1.0):
            self.k = k
            super().__init__()

        def declare_components(self):
            self.add_component("y", Variable(0.0))

        def declare_equations(self):
            return [self["y"].symbol - self.k]

    register_component(_Gain, "Gain")
    spec = {
        "hydrogen_version": "0.1.0",
        "schema_version": 1,
        "components": {"g": {"type": "Gain", "params": {"k": 2.0}}},
    }
    model = from_dict(spec)
    assert model["g"].k == 2.0


# --- validation / error aggregation ----------------------------------------


def test_error_aggregation_reports_all_problems():
    bad = {
        "schema_version": 1,
        "media": {"air": {"fluid": "air", "backend": "HEOS", "disable_warnings": True}},
        "components": {
            "a": {"type": "NotAThing", "params": {}},
            "b": {"type": "hydrogen.thermofluid.StraightPipe", "medium": "air",
                  "params": {"D": 0.05, "L": 5, "epsilon": 1e-4, "z_in": 0,
                             "z_out": 0, "bogus": 1}},
            "c": {"type": "hydrogen.thermofluid.AmbientInlet", "medium": "nope", "params": {}},
            "d": {"type": "hydrogen.thermofluid.FixedTemperature", "components": {}},
            "e": {"type": "Model", "equations": ["x-1"]},
        },
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    errors = ei.value.errors
    assert len(errors) == 5
    joined = "\n".join(errors)
    assert "unknown component type 'NotAThing'" in joined
    assert "unknown param 'bogus'" in joined
    assert "unknown medium 'nope'" in joined
    assert "cannot carry 'components'" in joined
    assert "'equations' is not supported yet" in joined


def test_missing_required_param_is_reported():
    bad = {
        "schema_version": 1,
        "media": {"air": {"fluid": "air", "backend": "HEOS", "disable_warnings": True}},
        # StraightPipe needs D, L, epsilon, z_in, z_out.
        "components": {"p": {"type": "hydrogen.thermofluid.StraightPipe", "medium": "air",
                             "params": {"D": 0.05}}},
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    joined = "\n".join(ei.value.errors)
    assert "missing required param 'L'" in joined
    assert "missing required param 'epsilon'" in joined


def test_component_requiring_medium_without_one_is_reported():
    bad = {
        "schema_version": 1,
        "components": {"src": {"type": "hydrogen.thermofluid.AmbientInlet",
                               "params": {"m_flow": 0.1}}},
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    assert any("requires a 'medium'" in e for e in ei.value.errors)


def test_unknown_schema_version_is_rejected():
    with pytest.raises(SystemSpecError) as ei:
        from_dict({"schema_version": 999, "components": {}})
    assert any("unsupported schema_version" in e for e in ei.value.errors)


# --- full-name registration + component catalog ----------------------------


def test_full_namespaced_type_loads():
    spec = {
        "schema_version": 1,
        "components": {
            "hot": {"type": "hydrogen.thermofluid.FixedTemperature",
                    "params": {"T_set": 400.0}},
            "cold": {"type": "hydrogen.thermofluid.FixedTemperature",
                     "params": {"T_set": 300.0}},
            "cond": {"type": "hydrogen.thermofluid.ThermalConductor",
                     "params": {"G": 5.0}},
        },
        "connections": [
            {"from": "hot.heat", "to": "cond.heat_a"},
            {"from": "cond.heat_b", "to": "cold.heat"},
        ],
    }
    model = from_dict(spec)
    assert sorted(model.components) == ["cold", "cond", "hot"]


def test_bare_name_no_longer_resolves():
    # Only fully-qualified names are registered now; a bare builtin name is
    # rejected as an unknown type.
    bad = {
        "schema_version": 1,
        "components": {"hot": {"type": "FixedTemperature", "params": {"T_set": 400.0}}},
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    assert any("unknown component type 'FixedTemperature'" in e for e in ei.value.errors)


def test_reflective_dump_uses_full_names():
    class Net(Model):
        def declare_components(self):
            self.add_component("hot", FixedTemperature(T_set=350.0))
            self.add_component("cond", ThermalConductor(G=3.0))

        def declare_equations(self):
            self.connect(self["hot"].ports["heat"], self["cond"].ports["heat_a"])
            return []

    net = Net()
    net.declare_equations()
    d = to_dict(net)
    assert d["components"]["hot"]["type"] == "hydrogen.thermofluid.FixedTemperature"
    assert d["components"]["cond"]["type"] == "hydrogen.thermofluid.ThermalConductor"


def test_component_catalog_lists_params_and_literals():
    catalog = component_catalog()
    by_type = {row["type"]: row for row in catalog}

    # Canonical full names are the keys.
    assert "hydrogen.thermofluid.FlatWall" in by_type
    assert "hydrogen.control.Sine" in by_type

    # Sorted alphabetically by type.
    assert [r["type"] for r in catalog] == sorted(r["type"] for r in catalog)

    wall = by_type["hydrogen.thermofluid.FlatWall"]
    assert wall["domain"] == "thermofluid"
    # `dynamic` is a structural flag -> reported as a "literal" (_cache_key_flags),
    # carrying a UI type hint.
    dyn = next(lit for lit in wall["literals"] if lit["name"] == "dynamic")
    assert dyn["type"] == "bool"
    pnames = {p["name"] for p in wall["parameters"]}
    assert {"rho", "cp", "k", "A", "L"} <= pnames
    # rho is required (no default); T_init is optional.
    rho = next(p for p in wall["parameters"] if p["name"] == "rho")
    assert rho["required"]
    t_init = next(p for p in wall["parameters"] if p["name"] == "T_init")
    assert not t_init["required"] and t_init["type"] == "float"

    # A fluid component reports that it needs a medium.
    assert by_type["hydrogen.thermofluid.StraightPipe"]["needs_medium"] is True


def test_component_catalog_domain_filter():
    thermofluid = component_catalog(domain="thermofluid")
    assert thermofluid and all(r["domain"] == "thermofluid" for r in thermofluid)
    assert "thermofluid" in available_domains()
    assert "control" in available_domains()
    # The text table renders without error and includes the header.
    table = format_component_catalog(domain="thermofluid")
    assert "PARAMETERS" in table and "hydrogen.thermofluid.FlatWall" in table


def test_catalog_reports_rich_param_types():
    pipe = next(r for r in component_catalog(domain="thermofluid")
                if r["type"] == "hydrogen.thermofluid.Pipe")
    by_name = {p["name"]: p for p in pipe["parameters"]}

    # `layers` is a list of WallLayer value objects.
    layers = by_name["layers"]
    assert layers["type"] == "list" and layers["required"]
    assert layers["item"] == {"type": "object", "value_type": "WallLayer"}

    # `multiphase` / `outer_thermal` are enums with their allowed choices.
    multiphase = by_name["multiphase"]
    assert multiphase["type"] == "enum"
    assert multiphase["choices"] == ["single", "HEM"]
    assert by_name["outer_thermal"]["choices"] == [
        "adiabatic", "convective", "fixed", "expose"]

    # Plain scalars still carry their simple labels + defaults.
    assert by_name["h_ext"]["type"] == "float" and by_name["h_ext"]["default"] == 10.0

    # Physical params carry a unit string a UI can label.
    assert by_name["D"]["unit"] == "m"
    assert by_name["p_init"]["unit"] == "Pa"
    assert by_name["T_ext"]["unit"] == "K"
    assert by_name["h_ext"]["unit"] == "W/(m^2*K)"
    assert by_name["n_segments"]["unit"] == "1"


def test_catalog_units_for_value_objects_and_walls():
    # Wall thermal triple is unit-annotated wherever it appears.
    mat = {f["name"]: f for f in value_object_spec("WallMaterial")["fields"]}
    assert (mat["rho"]["unit"], mat["cp"]["unit"], mat["k"]["unit"]) == (
        "kg/m^3", "J/(kg*K)", "W/(m*K)")

    flat = next(r for r in component_catalog(domain="thermofluid")
                if r["type"] == "hydrogen.thermofluid.FlatWall")
    fw = {p["name"]: p for p in flat["parameters"]}
    assert fw["k"]["unit"] == "W/(m*K)" and fw["A"]["unit"] == "m^2"

    fit = {f["name"]: f for f in value_object_spec("TransportFit")["fields"]}
    assert fit["E_Phi"]["unit"] == "J/mol" and fit["D0"]["unit"] == "m^2/s"


def test_catalog_reports_descriptions_and_conditions():
    pipe = next(r for r in component_catalog(domain="thermofluid")
                if r["type"] == "hydrogen.thermofluid.Pipe")
    by_name = {p["name"]: p for p in pipe["parameters"]}

    # Descriptions come from the class's PARAMS (single source of truth).
    assert by_name["D"]["description"] == "Pipe bore (inner) diameter."

    # Conditional-relevance: enum-gated and named-predicate forms.
    assert by_name["h_ext"]["relevant_when"] == {"outer_thermal": "convective"}
    assert by_name["T_outer"]["relevant_when"] == {"outer_thermal": "fixed"}
    assert by_name["p_ext"]["relevant_when"] == "any_layer_permeable"

    # Value-object fields carry descriptions too.
    mat = {f["name"]: f for f in value_object_spec("WallMaterial")["fields"]}
    assert mat["rho"]["description"] == "Density."


def test_param_spec_single_source_fills_live_parameters():
    # The same PARAMS that feeds the catalog is what `declare_components` uses
    # to label the live Parameters -- unit AND description, authored once.
    wall = CylindricalWall(rho=7990.0, cp=500.0, k=15.0,
                           r_in=0.01, r_out=0.012, length=1.0)
    assert wall["rho"].unit == "kg/m^3"
    assert wall["rho"].description == "Density of the wall material."
    assert wall["r_in"].unit == "m"
    assert wall["r_in"].description == "Inner radius of the tube wall (bore side)."

    # The catalog and the live Parameter agree (no second source).
    cyl = next(r for r in component_catalog(domain="thermofluid")
               if r["type"] == "hydrogen.thermofluid.CylindricalWall")
    rho = next(p for p in cyl["parameters"] if p["name"] == "rho")
    assert rho["unit"] == wall["rho"].unit
    assert rho["description"] == wall["rho"].description


def test_merged_param_specs_inherits_base_then_overrides():
    # FlatWall inherits the shared material specs from TwoNodeWall and adds its
    # own geometry specs.
    specs = merged_param_specs(FlatWall)
    assert isinstance(specs["rho"], ParamSpec) and specs["rho"].unit == "kg/m^3"
    assert specs["A"].unit == "m^2"          # FlatWall-specific
    assert "L" in specs                       # FlatWall-specific (thickness)


def test_literal_inherits_param_description_and_choices():
    # A structural flag (`_cache_key_flags`) inherits the description/choices of
    # its matching constructor param, instead of only carrying type/default.
    cyl = next(r for r in component_catalog(domain="thermofluid")
               if r["type"] == "hydrogen.thermofluid.CylindricalWall")
    dyn = next(lit for lit in cyl["literals"] if lit["name"] == "dynamic")
    assert dyn["description"]                  # pulled from TwoNodeWall.PARAMS
    leaky = next(lit for lit in cyl["literals"] if lit["name"] == "leaky")
    assert leaky["description"]


# --- PARAMS coverage ratchet -----------------------------------------------
#
# Every catalog field (param + public literal) should carry a `description`
# sourced from the class's PARAMS spec.  This is a ratchet: components not yet
# migrated to PARAMS are listed below; migrate one, then delete it from here.
# The test fails if (a) a non-waived component is missing specs (a regression,
# e.g. a new component added without PARAMS), or (b) a waived component is now
# fully specced (a stale waiver to remove).  Private cache-key flags (names
# starting with "_", e.g. CylindricalWall._perm_key) are not constructor args,
# so they cannot carry a ParamSpec and are exempt.
_SPEC_WAIVERS = frozenset()


def _fields_missing_description(row: dict) -> dict:
    """{kind: [names]} of catalog fields on ``row`` lacking a description."""
    missing = {}
    miss_p = [p["name"] for p in row["parameters"] if not p.get("description")]
    if miss_p:
        missing["params"] = miss_p
    miss_l = [lit["name"] for lit in row["literals"]
              if not lit.get("description") and not lit["name"].startswith("_")]
    if miss_l:
        missing["literals"] = miss_l
    return missing


def test_value_objects_fully_specced():
    incomplete = {v["value_type"]: [f["name"] for f in v["fields"]
                                    if not f.get("description")]
                  for v in value_object_catalog()}
    incomplete = {k: v for k, v in incomplete.items() if v}
    assert not incomplete, f"value objects missing field descriptions: {incomplete}"


def test_component_paramspec_coverage_ratchet():
    catalog = component_catalog()
    missing = {r["name"]: _fields_missing_description(r)
               for r in catalog if _fields_missing_description(r)}

    # (a) No regressions: every non-waived component is fully specced.
    unexpected = {k: v for k, v in missing.items() if k not in _SPEC_WAIVERS}
    assert not unexpected, (
        "components missing PARAMS specs (add a ParamSpec for each field, or "
        "add to _SPEC_WAIVERS):\n" + json.dumps(unexpected, indent=2, sort_keys=True))

    # (b) No stale waivers: a waived component that is now complete must be
    # removed from the list (keeps the ratchet honest and shrinking).
    stale = sorted(_SPEC_WAIVERS - set(missing))
    assert not stale, (
        f"these components are now fully specced -- remove from "
        f"_SPEC_WAIVERS: {stale}")

    # Sanity: the already-migrated components are genuinely complete.
    assert {"hydrogen.thermofluid.Pipe", "hydrogen.thermofluid.FlatWall",
            "hydrogen.thermofluid.CylindricalWall"} <= {r["type"] for r in catalog}
    for r in catalog:
        if r["name"] in {"Pipe", "FlatWall", "CylindricalWall"}:
            assert not _fields_missing_description(r), r["name"]


def test_value_object_catalog_and_spec():
    cat = {v["value_type"]: v for v in value_object_catalog()}
    assert {"WallLayer", "WallMaterial", "TransportFit", "Permeant",
            "SteadyRichardson", "TransientDiffusion"} <= set(cat)

    layer = value_object_spec("WallLayer")
    fields = {f["name"]: f for f in layer["fields"]}

    # A required nested value object names its concrete type.
    assert fields["material"]["type"] == "object"
    assert fields["material"]["value_type"] == "WallMaterial"
    assert fields["material"]["required"]

    # An optional, abstract value object is nullable and lists concrete options.
    perm = fields["permeation"]
    assert perm["type"] == "object" and perm["nullable"] is True
    assert perm["value_type"] == "PermeationFlux"
    assert perm["value_types"] == ["SteadyRichardson", "TransientDiffusion"]

    # Scalars round-trip as before.
    assert fields["thickness"]["type"] == "float" and fields["thickness"]["required"]
    assert fields["dynamic"]["type"] == "bool" and fields["dynamic"]["default"] is True

    with pytest.raises(KeyError):
        value_object_spec("NotAThing")


def test_value_template_recurses_and_resolves_abstract():
    layer = value_template("WallLayer")
    assert layer["__type__"] == "WallLayer"
    # Required nested object is seeded; a type with presets is filled from its
    # first preset (so the template comes ready-to-load, not blank).
    assert layer["material"]["__type__"] == "WallMaterial"
    assert layer["material"]["rho"] is not None
    assert layer["permeation"] is None               # optional/nullable object
    assert layer["dynamic"] is True                  # optional scalar default

    # An abstract value type resolves to a concrete serializable subtype, and a
    # TransportFit (presets) comes fully filled, including its permeant.
    flux = value_template("PermeationFlux")
    assert flux["__type__"] in ("SteadyRichardson", "TransientDiffusion")
    assert flux["transport_fit"]["__type__"] == "TransportFit"
    assert flux["transport_fit"]["permeant"]["__type__"] == "Permeant"
    assert flux["transport_fit"]["Phi0"] is not None

    with pytest.raises(KeyError):
        value_template("NotAThing")


def test_spec_template_fills_and_loads():
    tmpl = spec_template("hydrogen.thermofluid.Pipe")
    assert tmpl["type"] == "hydrogen.thermofluid.Pipe"
    # A fluid component carries a `medium: None` placeholder.
    assert "medium" in tmpl
    p = tmpl["params"]
    # Optional params carry defaults; the list param is seeded with one element.
    assert p["multiphase"] == "single" and p["outer_thermal"] == "adiabatic"
    assert isinstance(p["layers"], list) and p["layers"][0]["__type__"] == "WallLayer"

    # Fill the blanks like a UI would, then load it -- the loop closes.
    tmpl["medium"] = "H2"
    p["D"], p["L"], p["epsilon"] = 0.01, 1.0, 1e-6
    p["z_in"], p["z_out"], p["n_segments"] = 0.0, 0.0, 1
    layer = p["layers"][0]
    layer["thickness"] = 0.002
    layer["material"].update(name="AISI 316", rho=7990.0, cp=500.0, k=15.0)

    spec = {
        "hydrogen_version": "0.1.0",
        "schema_version": SCHEMA_VERSION,
        "media": {"H2": {"fluid": "Hydrogen"}},
        "components": {"pipe": tmpl},
        "connections": [],
    }
    model = from_dict(spec)
    assert isinstance(model, Model)

    # A component with no medium omits the placeholder.
    sine = spec_template("hydrogen.control.Sine")
    assert "medium" not in sine

    with pytest.raises(KeyError):
        spec_template("hydrogen.nope.NotAComponent")


def test_component_spec_expands_whole_tree_with_descriptions():
    """`component_spec` is the single call a UI needs: catalog metadata plus a
    fully-expanded value-object tree (with descriptions) plus an embedded
    fill-in template."""
    spec = component_spec("hydrogen.thermofluid.Pipe")
    assert spec["type"] == "hydrogen.thermofluid.Pipe"
    # Carries the catalog metadata + an embedded ready-to-fill template.
    assert set(spec["template"]) == {"type", "medium", "params"}

    # The list-of-WallLayer param is expanded down its item -> value_spec.
    layers = next(p for p in spec["parameters"] if p["name"] == "layers")
    assert layers["type"] == "list"
    layer_fields = layers["item"]["value_spec"]["fields"]
    by_name = {f["name"]: f for f in layer_fields}

    # A concrete object field carries its own value_spec, with descriptions
    # and units flowing through from the nested class's PARAMS.
    mat = by_name["material"]["value_spec"]
    assert mat["value_type"] == "WallMaterial"
    rho = next(f for f in mat["fields"] if f["name"] == "rho")
    assert rho["unit"] == "kg/m^3" and rho["description"]

    # An abstract object field is expanded into `options` keyed by concrete
    # type, and recursion continues all the way down the tree.
    perm = by_name["permeation"]
    assert set(perm["options"]) == {"SteadyRichardson", "TransientDiffusion"}
    fit = perm["options"]["SteadyRichardson"]["fields"][0]
    assert fit["value_spec"]["value_type"] == "TransportFit"
    assert any(f["name"] == "permeant" for f in fit["value_spec"]["fields"])

    with pytest.raises(KeyError):
        component_spec("hydrogen.nope.NotAComponent")


def test_value_object_presets_surface_named_choices():
    """A value object with a `PRESETS` mapping exposes them (serialized) so a UI
    can offer a choice list; they also flow into the expanded component_spec."""
    presets = value_object_spec("Permeant").get("presets")
    assert presets is not None
    names = {p["name"] for p in presets}
    assert {"H2", "He", "N2"} <= names
    h2 = next(p for p in presets if p["name"] == "H2")
    assert h2["spec"]["__type__"] == "Permeant"
    assert h2["spec"]["solubility_exponent"] == 2.0
    assert h2["spec"]["M"] > 0

    # The same presets are embedded in the fully-expanded Pipe spec.
    spec = component_spec("hydrogen.thermofluid.Pipe")
    layer = next(p for p in spec["parameters"] if p["name"] == "layers")["item"]["value_spec"]
    perm = next(f for f in layer["fields"] if f["name"] == "permeation")
    fit = perm["options"]["SteadyRichardson"]["fields"][0]
    permeant = fit["value_spec"]["fields"][0]
    assert {p["name"] for p in permeant["value_spec"]["presets"]} == names

    # TransportFit also has presets; a fit fully defines the transport
    # (including its permeant), so there is no keep-on-preset carve-out.
    tf_spec = value_object_spec("TransportFit")
    assert {p["name"] for p in tf_spec["presets"]} >= {"H2 in AISI 304", "H2 in AISI 316"}
    h2_304 = next(p for p in tf_spec["presets"] if p["name"] == "H2 in AISI 304")
    assert h2_304["spec"]["permeant"]["__type__"] == "Permeant"
    assert "preset_keeps" not in tf_spec
    assert "preset_keeps" not in fit["value_spec"]

    # WallMaterial offers thermal-property presets too.
    mat_presets = value_object_spec("WallMaterial").get("presets")
    assert {p["name"] for p in mat_presets} >= {"AISI 316/316L", "AISI 304/304L"}
    assert all(p["spec"]["__type__"] == "WallMaterial" for p in mat_presets)


def test_missing_required_package_is_reported():
    bad = {
        "schema_version": 1,
        "requires": {"packages": [{"import": "no_such_pkg_xyz"}]},
        "components": {"v": {"type": "no_such_pkg_xyz.Thing", "params": {}}},
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    assert any("could not be imported" in e for e in ei.value.errors)


# --- structured value-object params (Pipe layers / permeation) -------------


def test_value_spec_roundtrip_wall_layer():
    """A `WallLayer` (material + permeation flux + transport fit + permeant)
    round-trips through the value-spec codec, value-for-value."""
    from hydrogen.components.materials import AISI_316
    from hydrogen.components.thermofluid.assemblies import WallLayer
    from hydrogen.components.thermofluid.permeation import (
        H2_IN_AUSTENITIC,
        SteadyRichardson,
    )
    from hydrogen.serialization.values import decode_value, encode_value

    layer = WallLayer(AISI_316, 1.5e-3,
                      permeation=SteadyRichardson(H2_IN_AUSTENITIC),
                      dynamic=False)
    spec = encode_value(layer)
    # The spec is plain JSON (tagged, nested).
    assert json.loads(json.dumps(spec)) == spec
    assert spec["__type__"] == "WallLayer"
    assert spec["material"]["__type__"] == "WallMaterial"
    assert spec["permeation"]["__type__"] == "SteadyRichardson"
    assert spec["permeation"]["transport_fit"]["permeant"]["__type__"] == "Permeant"

    back = decode_value(spec)
    assert isinstance(back, WallLayer)
    assert back.thickness == layer.thickness and back.dynamic is False
    assert back.material.to_spec() == layer.material.to_spec()
    fa, fb = layer.permeation.fit, back.permeation.fit
    assert (fb.Phi0, fb.E_Phi, fb.D0, fb.E_D) == (fa.Phi0, fa.E_Phi, fa.D0, fa.E_D)
    assert fb.permeant.to_spec() == fa.permeant.to_spec()


def _walled_pipe_net():
    from hydrogen.components.materials import AISI_316
    from hydrogen.components.thermofluid.assemblies import Pipe, WallLayer
    from hydrogen.components.thermofluid.flow import ClosedEnd, PressureSource
    from hydrogen.components.thermofluid.permeation import (
        H2_IN_AUSTENITIC,
        SteadyRichardson,
    )

    med = CoolPropMedium("Hydrogen", disable_warnings=True)

    class Net(Model):
        def declare_components(self):
            self.add_component("source", PressureSource(
                med, p_source=1e5, T_source=423.15, A=1e-5))
            self.add_component("pipe", Pipe(
                med, D=3e-3, L=0.1, epsilon=1e-6, z_in=0.0, z_out=0.0,
                n_segments=2,
                layers=[WallLayer(AISI_316, 1.5e-3,
                                  permeation=SteadyRichardson(H2_IN_AUSTENITIC),
                                  dynamic=False)],
                outer_thermal="fixed", T_outer=423.15, p_ext=1.0,
                T_wall_init=423.15, p_init=1e5))
            self.add_component("cap", ClosedEnd(med, p_init=1e5, T_init=423.15))

        def declare_equations(self):
            self.connect(self["source"].ports["outlet"], self["pipe"].ports["inlet"])
            self.connect(self["pipe"].ports["outlet"], self["cap"].ports["inlet"])
            return []

    return Net


def test_pipe_with_leaky_layer_roundtrips():
    """A `Pipe` carrying a `list[WallLayer]` param serializes as an opaque leaf
    whose structured params round-trip dict -> model -> dict exactly."""
    net = _walled_pipe_net()()
    net.declare_equations()  # wire so the inlet/outlet connections are dumped

    d = to_dict(net)
    pipe = d["components"]["pipe"]
    assert pipe["type"] == "hydrogen.thermofluid.Pipe"
    assert pipe["params"]["outer_thermal"] == "fixed"
    layers = pipe["params"]["layers"]
    assert isinstance(layers, list) and layers[0]["__type__"] == "WallLayer"
    assert layers[0]["permeation"]["transport_fit"]["__type__"] == "TransportFit"
    # JSON survives intact.
    assert json.loads(to_json(net)) == d

    # dict -> model -> dict is exact, and the rebuilt child is a real Pipe.
    model = from_dict(d)
    assert to_dict(model) == d
    rebuilt = model["pipe"]
    assert type(rebuilt).__name__ == "Pipe"
    assert len(rebuilt.layers) == 1
    assert rebuilt.layers[0].permeation.fit.permeant.M == \
        net["pipe"].layers[0].permeation.fit.permeant.M


def test_loaded_pipe_instantiates():
    """The reloaded leaky-`Pipe` system assembles its DAE (structure rebuilt)."""
    med = CoolPropMedium("Hydrogen", backend="BICUBIC&HEOS", disable_warnings=True)
    net = _walled_pipe_net()()
    net.declare_equations()
    model = from_dict(to_dict(net))
    model.instantiate(aditional_modules=med.modules)  # must not raise
    # The leaky wall structure was rebuilt from the decoded layers.
    assert any(name.startswith("wall_") for name in model["pipe"].components)
