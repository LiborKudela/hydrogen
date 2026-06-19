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
    FixedTemperature,
    Model,
    Parameter,
    StraightPipe,
    ThermalConductor,
    Variable,
    from_dict,
    from_json,
    to_dict,
    to_json,
)
from hydrogen.serialization import SystemSpecError, register_component


def _thermal_spec():
    return {
        "hydrogen_version": "0.1.0",
        "schema_version": 1,
        "components": {
            "hot": {"type": "FixedTemperature", "params": {"T_set": 400.0}},
            "cold": {"type": "FixedTemperature", "params": {"T_set": 300.0}},
            "cond": {"type": "ThermalConductor", "params": {"G": 5.0}},
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
    assert d["components"]["hot"] == {"type": "FixedTemperature", "params": {"T_set": 350.0}}
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
            "src": {"type": "AmbientInlet", "medium": "air", "params": {"m_flow": 0.05}},
            "leg": {
                "type": "Model",
                "medium": "air",
                "components": {
                    "pipe": {
                        "type": "StraightPipe",
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
            "b": {"type": "StraightPipe", "medium": "air",
                  "params": {"D": 0.05, "L": 5, "epsilon": 1e-4, "z_in": 0,
                             "z_out": 0, "bogus": 1}},
            "c": {"type": "AmbientInlet", "medium": "nope", "params": {}},
            "d": {"type": "FixedTemperature", "components": {}},
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
        "components": {"p": {"type": "StraightPipe", "medium": "air",
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
        "components": {"src": {"type": "AmbientInlet", "params": {"m_flow": 0.1}}},
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    assert any("requires a 'medium'" in e for e in ei.value.errors)


def test_unknown_schema_version_is_rejected():
    with pytest.raises(SystemSpecError) as ei:
        from_dict({"schema_version": 999, "components": {}})
    assert any("unsupported schema_version" in e for e in ei.value.errors)


def test_missing_required_package_is_reported():
    bad = {
        "schema_version": 1,
        "requires": {"packages": [{"import": "no_such_pkg_xyz"}]},
        "components": {"v": {"type": "no_such_pkg_xyz.Thing", "params": {}}},
    }
    with pytest.raises(SystemSpecError) as ei:
        from_dict(bad)
    assert any("could not be imported" in e for e in ei.value.errors)
