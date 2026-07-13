"""Solver diagnostics (`Model.diagnose` + the `diagnose` service command).

Uses a tiny CoolProp-free algebraic control system so the test is fast and
deterministic.  Checks that:

* `Model.diagnose()` returns a well-formed, JSON-serialisable report whose
  fields have the expected shape, and reports a healthy model as ``ok``;
* the report traces to component ids via variable full names;
* the `diagnose` service command round-trips the same report;
* a structurally singular system (a free, unconstrained variable) is detected
  and its verdict is *not* ``ok``.
"""

from __future__ import annotations

import json

import hydrogen

# Constant("src") -> Gain("amp"): purely algebraic, well-posed.
_SPEC = {
    "hydrogen_version": "0.1.0",
    "schema_version": 1,
    "components": {
        "src": {"type": "hydrogen.control.Constant", "params": {"k": 2.0}},
        "amp": {"type": "hydrogen.control.Gain", "params": {"k": 3.0}},
    },
    "connections": [{"from": "src.y", "to": "amp.u"}],
}

_REQUIRED_KEYS = {
    "ok", "n_v", "n_eq", "residual_finite", "jacobian_finite",
    "structurally_singular", "condition_estimate", "worst_residuals",
    "nonfinite_residuals", "nonfinite_vars", "singular_rows", "singular_cols",
    "near_singular_vars", "components", "summary", "likely_causes",
    "severity", "cause_codes", "conditioning",
}


def _iter_report_names(rep):
    """Yield every user-facing variable/equation name string in a report."""
    for key in ("near_singular_vars", "nonfinite_vars", "singular_cols"):
        for entry in rep.get(key) or []:
            if entry.get("name"):
                yield entry["name"]
    for key in ("worst_residuals", "nonfinite_residuals"):
        for entry in rep.get(key) or []:
            yield from entry.get("variables") or []


def test_diagnose_healthy_report_shape():
    m = hydrogen.from_dict(_SPEC)
    m.instantiate(max_remove_trival_passes=2)
    m.initialise(n=1)
    rep = m.diagnose()

    assert _REQUIRED_KEYS <= set(rep), _REQUIRED_KEYS - set(rep)
    assert rep["ok"] is True
    assert rep["residual_finite"] and rep["jacobian_finite"]
    assert not rep["structurally_singular"]
    assert not rep["nonfinite_vars"]
    assert isinstance(rep["summary"], str) and rep["summary"]
    # LLM-friendly labels: a healthy model is ``ok`` with no cause codes.
    assert rep["severity"] == "ok"
    assert rep["cause_codes"] == []
    assert isinstance(rep["conditioning"], dict)
    assert rep["conditioning"]["band"] in {"healthy", "ill_conditioned",
                                           "singular", "unknown"}
    # A brief summary is a single line.
    assert "\n" not in rep["summary"]
    # Component ids are recoverable from variable full names.
    comps = {c["component"] for c in rep["components"]}
    assert comps  # non-empty
    # Must survive strict JSON serialisation (no NaN/Inf leaking through).
    json.dumps(rep, allow_nan=False)


def test_diagnose_service_roundtrip():
    service = hydrogen.start_host(workers=1)
    try:
        system = service.load_dict(_SPEC)
        system.instantiate(max_remove_trival_passes=2)
        system.poll_events()
        system.initialise(n=1)
        rep = system.diagnose()
        assert _REQUIRED_KEYS <= set(rep)
        assert rep["ok"] is True
        json.dumps(rep, allow_nan=False)
    finally:
        service.shutdown()


def test_diagnose_detects_structural_singularity():
    # A `Constant` whose output feeds nothing leaves `amp` with an input that is
    # driven, but if we instead leave a Gain input unconnected the system is
    # under-determined.  Simplest reliable degenerate case: two Constants and a
    # Gain whose input is unconnected -> `amp.u` (and `amp.y`) are unconstrained.
    spec = {
        "hydrogen_version": "0.1.0",
        "schema_version": 1,
        "components": {
            "amp": {"type": "hydrogen.control.Gain", "params": {"k": 3.0}},
        },
        "connections": [],
    }
    m = hydrogen.from_dict(spec)
    m.instantiate(max_remove_trival_passes=2)
    try:
        m.initialise(n=1)
    except Exception:
        pass  # a singular initialise is fine; we diagnose the state regardless
    rep = m.diagnose()
    # Either a structural singularity or a numerically singular Jacobian must be
    # flagged, and the verdict must not be "ok".
    assert not rep["ok"]
    assert rep["severity"] == "error"
    assert rep["cause_codes"]      # at least one machine cause code
    assert (rep["structurally_singular"]
            or rep["singular_cols"]
            or (isinstance(rep["condition_estimate"], str))
            or (isinstance(rep["condition_estimate"], (int, float))
                and rep["condition_estimate"] is not None
                and rep["condition_estimate"] > 1e13))
    # Reported names are stripped of the root-composite prefix (no leaked
    # ``_SpecComposite.`` / class-name noise from JSON-loaded systems).
    for name in _iter_report_names(rep):
        assert not name.startswith("_SpecComposite"), name
        assert "_SpecComposite." not in name, name
