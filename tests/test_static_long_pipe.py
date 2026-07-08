"""Static long pipes fail on fast valve-opening transients (saved_system_2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import hydrogen as hd

_PROJECT = Path(__file__).resolve().parent.parent / "tutorials" / "saved_system_2.json"
_SIM = json.loads(_PROJECT.read_text())["sim_options"]


def _system(*, L: float, dynamic: str) -> str:
    data = json.loads(_PROJECT.read_text())
    components, media = {}, data.get("media") or {}
    for nd in data["canvas"]["nodes"]:
        spec = hd.component_spec(nd["type"])
        tmpl = dict(spec["template"])
        params = copy.deepcopy(nd.get("params") or tmpl.get("params") or {})
        if nd["comp_id"] == "pipe_2":
            params["L"] = L
            params["dynamic"] = dynamic
        tmpl["params"] = params
        if spec["needs_medium"]:
            key = nd.get("medium") or "Hydrogen"
            tmpl["medium"] = key
            media.setdefault(key, media[key])
        components[nd["comp_id"]] = tmpl
    return json.dumps({
        "hydrogen_version": hd.__version__,
        "schema_version": 1,
        "media": media,
        "components": components,
        "connections": data["canvas"]["connections"],
    })


def _run(spec_text: str) -> dict:
    inst, init, sim = _SIM["instantiate"], _SIM["initialise"], _SIM["simulate"]
    service = hd.start_host(workers=1)
    try:
        sysp = service.load_json(spec_text)
        sysp.instantiate(**inst)
        sysp.initialise(**init)
        sysp.run(
            stop_time=5.0, stream=False,
            strategy={"name": sim["strategy"], "tol_local": sim["tol_local"],
                      "atol": sim["atol"]},
            dt_start=sim["dt_start"], dt_min=sim["dt_min"],
            dt_max=sim["dt_max"], grow=sim["grow"], shrink=sim["shrink"],
            max_retries=sim["max_retries"], tol=sim["tol"],
            max_iter=sim["max_iter"], line_search=sim["line_search"],
        )
        return sysp.status()
    finally:
        service.shutdown()


def test_static_long_pipe_fails_at_valve_ramp():
    st = _run(_system(L=10.0, dynamic="static"))
    assert st["phase"] == "error"
    assert st["t"] == pytest.approx(1.0, abs=0.01)


def test_compressible_long_pipe_survives_valve_ramp():
    st = _run(_system(L=10.0, dynamic="compressible"))
    assert st["phase"] == "finished"
    assert st["t"] == pytest.approx(5.0, rel=1e-3)
