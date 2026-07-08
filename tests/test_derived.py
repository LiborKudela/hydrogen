"""Tests for client-side derived variable aggregation."""

import numpy as np

import pytest

from hydrogen.ui.derived import (
    compile_formula,
    compute_series,
    evaluate_formula,
    filter_leaves_regex,
    is_derived,
    make_derived_payload,
    resolve_regex_names,
    unit_for_agg,
)


class _Handle:
    def __init__(self, arr):
        self.array = np.asarray(arr, dtype=float)


def test_structural_sum():
    t = np.linspace(0, 3, 4)
    handles = {"a": _Handle([1, 2, 3, 4]), "b": _Handle([10, 20, 30, 40])}
    agg = {"op": "sum", "axis": "instances", "sources": ["a", "b"]}
    ts, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert np.allclose(ys, [11, 22, 33, 44])
    assert len(ts) == 4


def test_time_abs():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    handles = {"m": _Handle([-1.0, 2.0, -3.0, 4.0])}
    agg = {"op": "abs", "axis": "time", "sources": ["m"]}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert np.allclose(ys, [1.0, 2.0, 3.0, 4.0])


def test_time_integral():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    handles = {"m": _Handle([1.0, 1.0, 1.0, 1.0])}
    agg = {"op": "integral", "axis": "time", "sources": ["m"]}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert ys[0] == 0.0
    assert abs(ys[-1] - 3.0) < 1e-12


def test_series_values_pattern():
    t = np.array([0.0, 1.0])
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = {"leak": _Handle(mat)}
    agg = {"op": "sum", "axis": "instances", "pattern": "leak"}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles={}, values_handles=values)
    assert np.allclose(ys, [3.0, 7.0])


def test_derived_list_payloads_roundtrip():
    from hydrogen.ui.varwindow import _DerivedList
    from hydrogen.ui.qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    lst = _DerivedList()
    p = make_derived_payload(
        op="abs", axis="time", label="|pipe.p|",
        sources=["pipe.p"], unit="Pa")
    lst.add_payload(p, notify=False)
    got = lst.payloads()
    assert len(got) == 1
    assert got[0]["full"] == p["full"]
    assert got[0]["agg"]["op"] == "abs"
    assert got[0]["agg"]["sources"] == ["pipe.p"]
    lst.set_payloads(got, notify=False)
    assert len(lst.payloads()) == 1


def test_make_payload_and_filter():
    p = make_derived_payload(
        op="mean", axis="instances", label="x", sources=["a", "b"])
    assert is_derived(p["full"])
    assert p["agg"]["op"] == "mean"
    assert unit_for_agg("integral", "kg/s") == "kg"
    leaves = [{"full": "pipe.leak_1", "label": "l1", "description": "", "unit": ""},
              {"full": "pipe.p_out", "label": "p", "description": "", "unit": ""}]
    matched = filter_leaves_regex(leaves, r"leak")
    assert len(matched) == 1


def test_make_payload_regex_scope():
    p = make_derived_payload(
        op="sum", axis="instances", label="pipe.sum(wall)",
        regex=r"wall_\d+\.T", scope="pipe")
    assert p["agg"]["op"] == "sum"
    assert p["agg"]["regex"] == r"wall_\d+\.T"
    assert p["agg"]["scope"] == "pipe"
    # No frozen source list is stored for a regex aggregate.
    assert "sources" not in p["agg"]


def test_resolve_regex_names_scoped():
    names = [
        "System.pipe.wall_0.T", "System.pipe.wall_1.T",
        "System.pipe.wall_2.T", "System.pipe.p_out",
        "System.valve.wall_0.T",
    ]
    got = resolve_regex_names(names, r"wall_\d+\.T", scope="pipe")
    assert got == [
        "System.pipe.wall_0.T", "System.pipe.wall_1.T", "System.pipe.wall_2.T",
    ]


def test_resolve_regex_grows_with_instances():
    """The same regex picks up new instances (e.g. a higher n_segments)."""
    pattern = r"wall_\d+\.T"
    before = ["System.pipe.wall_0.T", "System.pipe.wall_1.T"]
    after = before + ["System.pipe.wall_2.T", "System.pipe.wall_3.T"]
    assert len(resolve_regex_names(before, pattern, scope="pipe")) == 2
    assert len(resolve_regex_names(after, pattern, scope="pipe")) == 4


def test_resolve_regex_blank_matches_scope():
    names = ["System.pipe.a", "System.pipe.b", "System.valve.a"]
    assert resolve_regex_names(names, "", scope="pipe") == [
        "System.pipe.a", "System.pipe.b"]


def test_resolve_regex_invalid_falls_back_to_substring():
    names = ["System.pipe.arr[0].T", "System.pipe.p_out"]
    # An unbalanced bracket is an invalid regex -> literal substring match.
    assert resolve_regex_names(names, "arr[", scope="pipe") == [
        "System.pipe.arr[0].T"]


def test_formula_elementwise():
    t = np.array([0.0, 1.0, 2.0])
    handles = {"a": _Handle([3.0, 4.0, 0.0]), "b": _Handle([4.0, 3.0, 5.0])}
    agg = {"op": "formula", "axis": "formula", "expr": "sqrt(v1**2 + v2**2)",
           "variables": {"v1": "a", "v2": "b"}}
    ts, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert np.allclose(ys, [5.0, 5.0, 5.0])
    assert len(ts) == 3


def test_formula_difference():
    t = np.array([0.0, 1.0])
    handles = {"hi": _Handle([10.0, 20.0]), "lo": _Handle([1.0, 2.0])}
    agg = {"op": "formula", "axis": "formula", "expr": "v1 - v2",
           "variables": {"v1": "hi", "v2": "lo"}}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert np.allclose(ys, [9.0, 18.0])


def test_formula_scalar_broadcast():
    t = np.array([0.0, 1.0, 2.0])
    handles = {"a": _Handle([1.0, 2.0, 3.0])}
    agg = {"op": "formula", "axis": "formula", "expr": "v1 * 2 + 1",
           "variables": {"v1": "a"}}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert np.allclose(ys, [3.0, 5.0, 7.0])


def test_formula_missing_source_is_empty():
    t = np.array([0.0, 1.0])
    handles = {"a": _Handle([1.0, 2.0])}
    agg = {"op": "formula", "axis": "formula", "expr": "v1 + v2",
           "variables": {"v1": "a", "v2": "missing"}}
    ts, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert len(ts) == 0 and len(ys) == 0


def test_make_formula_payload():
    p = make_derived_payload(
        op="formula", axis="formula", label="pipe.dp",
        expr="v1 - v2", variables={"v1": "pipe.p_in", "v2": "pipe.p_out"},
        unit="Pa")
    assert p["agg"]["axis"] == "formula"
    assert p["agg"]["expr"] == "v1 - v2"
    assert p["agg"]["variables"] == {"v1": "pipe.p_in", "v2": "pipe.p_out"}


def test_compile_formula_rejects_unsafe():
    # Attribute access, dunders, non-whitelisted calls and imports are rejected.
    for bad in ("v1.__class__", "__import__('os')", "open('x')",
                "v1[0]", "eval('1')"):
        with pytest.raises(ValueError):
            compile_formula(bad)


def test_compile_formula_allows_whitelisted():
    code = compile_formula("where(v1 > 0, sqrt(v1), 0.0)")
    out = evaluate_formula(code, {"v1": np.array([4.0, -1.0, 9.0])})
    assert np.allclose(out, [2.0, 0.0, 3.0])


def test_formula_group_reduction():
    t = np.array([0.0, 1.0, 2.0])
    mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    agg = {"op": "formula", "axis": "formula", "expr": "sum(g1)",
           "groups": {"g1": {"regex": "leak", "scope": "pipe"}}}
    ts, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles={}, values_handles={}, group_arrays={"g1": mat})
    assert np.allclose(ys, [3.0, 7.0, 11.0])
    assert len(ts) == 3


def test_formula_group_and_scalar_mix():
    t = np.array([0.0, 1.0])
    mat = np.array([[1.0, 1.0], [2.0, 2.0]])       # sum -> [2, 4]
    handles = {"p": _Handle([10.0, 20.0])}
    agg = {"op": "formula", "axis": "formula", "expr": "p1 - mean(g1)",
           "variables": {"p1": "p"},
           "groups": {"g1": {"regex": "x", "scope": "c"}}}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={}, group_arrays={"g1": mat})
    assert np.allclose(ys, [9.0, 18.0])            # 10-1, 20-2


def test_formula_temporal_integral():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    handles = {"m": _Handle([1.0, 1.0, 1.0, 1.0])}
    agg = {"op": "formula", "axis": "formula", "expr": "integral(v1)",
           "variables": {"v1": "m"}}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles=handles, values_handles={})
    assert ys[0] == 0.0
    assert abs(ys[-1] - 3.0) < 1e-12


def test_formula_integral_of_group_sum():
    t = np.array([0.0, 1.0, 2.0])
    mat = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])   # sum -> all 2.0
    agg = {"op": "formula", "axis": "formula", "expr": "integral(sum(g1))",
           "groups": {"g1": {"regex": "x", "scope": "c"}}}
    _, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles={}, values_handles={}, group_arrays={"g1": mat})
    # cumulative trapezoid of a constant 2 over [0,1,2] -> 0, 2, 4.
    assert np.allclose(ys, [0.0, 2.0, 4.0])


def test_formula_missing_group_is_empty():
    t = np.array([0.0, 1.0])
    agg = {"op": "formula", "axis": "formula", "expr": "sum(g1)",
           "groups": {"g1": {"regex": "x", "scope": "c"}}}
    ts, ys = compute_series(
        agg, time_array=t, start_index=0,
        series_handles={}, values_handles={}, group_arrays={})
    assert len(ts) == 0 and len(ys) == 0


def test_make_formula_payload_with_groups_and_full():
    p = make_derived_payload(
        op="formula", axis="formula", label="pipe.leak",
        expr="integral(sum(g1))",
        groups={"g1": {"regex": "m_dot_leak", "scope": "pipe"}},
        full="__derived__:keepme")
    assert p["full"] == "__derived__:keepme"
    assert p["agg"]["groups"]["g1"]["regex"] == "m_dot_leak"


def test_editor_roundtrip_preserves_id():
    from hydrogen.ui.varwindow import _DerivedEditor
    from hydrogen.ui.qt import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    leaves = [{"full": "pipe.p_in", "label": "pipe.p_in", "unit": "Pa"},
              {"full": "pipe.p_out", "label": "pipe.p_out", "unit": "Pa"}]
    orig = make_derived_payload(
        op="formula", axis="formula", label="pipe.dp", unit="Pa",
        expr="v1 - v2",
        variables={"v1": "pipe.p_in", "v2": "pipe.p_out"})
    ed = _DerivedEditor("pipe", leaves, payload=orig)
    # Add another variable and change the expression, then save.
    ed._add_variable_input(leaves[0])
    ed._expr.setText("v1 - v2 + v3*0")
    out = ed.payload()
    assert out is not None
    assert out["full"] == orig["full"]            # id preserved for edit-in-place
    assert out["agg"]["variables"]["v1"] == "pipe.p_in"
    assert "v3" in out["agg"]["variables"]


def test_editor_converts_legacy_regex():
    from hydrogen.ui.varwindow import _DerivedEditor
    from hydrogen.ui.qt import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    legacy = make_derived_payload(
        op="sum", axis="instances", label="pipe.sum(leak)",
        regex="m_dot_leak", scope="pipe")
    ed = _DerivedEditor("pipe", [], payload=legacy)
    out = ed.payload()
    assert out["agg"]["axis"] == "formula"
    assert out["agg"]["expr"] == "sum(g1)"
    assert out["agg"]["groups"]["g1"]["regex"] == "m_dot_leak"
