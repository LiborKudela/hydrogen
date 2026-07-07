"""Tests for client-side derived variable aggregation."""

import numpy as np

from hydrogen.ui.derived import (
    compute_series,
    filter_leaves_regex,
    is_derived,
    make_derived_payload,
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
