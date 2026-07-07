"""The per-component *Variables* window.

Opened from a placed component's context menu, it shows that component's model
variables in a tree that mirrors the component definition's structure
(sub-models -> groups, value objects -> leaves), each leaf annotated with its
unit and description.  Leaves are a drag source carrying the
:data:`~hydrogen.ui.plots.VARIABLE_MIME` payload, so they can be dropped onto a
Table or Timeseries object on the canvas.

Selected (or regex-filtered) variables can be aggregated client-side — sum /
mean / time integral etc. — without a model rebuild.  Derived results appear in
the list at the bottom and are draggable like ordinary variables.
"""

from __future__ import annotations

import re
import copy

from .derived import (
    STRUCTURAL_OPS,
    make_derived_payload,
    unit_for_agg,
)
from .plots import encode_variables
from .qt import QtCore, QtGui, QtWidgets, Signal
from .varmeta import variable_tree

__all__ = ["VariablesWindow"]

#: Leaf-kind -> swatch colour (matches the four value-object classes).
_KIND_COLOR = {
    "differential": "#c62828",
    "variable": "#1565c0",
    "input": "#6a1b9a",
    "parameter": "#2e7d32",
    "derived": "#e65100",
}


class _VarTree(QtWidgets.QTreeWidget):
    """Tree whose leaves drag a variable payload; groups are not draggable."""

    PAYLOAD_ROLE = QtCore.Qt.UserRole

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Variable", "Unit"])
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().setStretchLastSection(False)
        self.header().resizeSection(1, 80)

    def mimeData(self, items):  # noqa: N802 (Qt override)
        payloads = [it.data(0, self.PAYLOAD_ROLE) for it in items]
        payloads = [p for p in payloads if p]
        return encode_variables(payloads)

    def visible_leaf_items(self) -> list[QtWidgets.QTreeWidgetItem]:
        out: list[QtWidgets.QTreeWidgetItem] = []

        def walk(item):
            if item.data(0, self.PAYLOAD_ROLE) and not item.isHidden():
                out.append(item)
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.topLevelItemCount()):
            walk(self.topLevelItem(i))
        return out

    def selected_leaf_payloads(self) -> list[dict]:
        return [
            p for item in self.selectedItems()
            if (p := item.data(0, self.PAYLOAD_ROLE)) and not item.isHidden()
        ]


class _DerivedList(QtWidgets.QListWidget):
    """Draggable list of client-side derived variables."""

    changed = Signal()
    PAYLOAD_ROLE = QtCore.Qt.UserRole

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def payloads(self) -> list[dict]:
        out: list[dict] = []
        for i in range(self.count()):
            p = self.item(i).data(self.PAYLOAD_ROLE)
            if p:
                out.append(copy.deepcopy(p))
        return out

    def set_payloads(self, payloads: list[dict], *, notify: bool = True):
        self.clear()
        for p in payloads:
            self.add_payload(p, notify=False)
        if notify:
            self.changed.emit()

    def add_payload(self, payload: dict, *, notify: bool = True):
        item = QtWidgets.QListWidgetItem(payload.get("label", payload["full"]))
        item.setData(self.PAYLOAD_ROLE, copy.deepcopy(payload))
        item.setToolTip(payload.get("description", ""))
        item.setForeground(QtGui.QBrush(QtGui.QColor(_KIND_COLOR["derived"])))
        self.addItem(item)
        if notify:
            self.changed.emit()

    def mimeData(self, items):  # noqa: N802
        payloads = [it.data(self.PAYLOAD_ROLE) for it in items]
        payloads = [p for p in payloads if p]
        return encode_variables(payloads)


class VariablesWindow(QtWidgets.QDialog):
    """Non-modal browser of one component's variables (a drag source)."""

    def __init__(self, comp_id: str, type_name: str, medium: str | None,
                 params: dict | None, *,
                 derived: list[dict] | None = None,
                 on_derived_changed=None,
                 parent=None):
        super().__init__(parent)
        self._comp_id = comp_id
        self._on_derived_changed = on_derived_changed
        self._leaf_total = 0
        self.setWindowTitle(f"Variables — {comp_id}")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.resize(460, 640)

        lay = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"<b>{comp_id}</b> <span style='color:#777'>{type_name}</span><br>"
            "<span style='color:#777; font-size:11px'>Drag variables (or "
            "derived aggregates below) onto a Table, Timeseries, Bar chart, or "
            "Pie chart on the canvas.</span>")
        header.setWordWrap(True)
        lay.addWidget(header)

        tool = QtWidgets.QHBoxLayout()
        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter (regex)…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        tool.addWidget(self._filter, 1)

        sel_btn = QtWidgets.QPushButton("Select all")
        sel_btn.setToolTip("Select every variable currently visible in the filter.")
        sel_btn.clicked.connect(self._select_all_visible)
        tool.addWidget(sel_btn)

        agg_btn = QtWidgets.QToolButton()
        agg_btn.setText("Aggregate ▾")
        agg_btn.setToolTip(
            "Build a derived variable from the current selection.")
        agg_menu = QtWidgets.QMenu(agg_btn)
        for op in STRUCTURAL_OPS:
            act = agg_menu.addAction(op)
            act.triggered.connect(lambda _=False, o=op: self._aggregate_selection(o))
        agg_btn.setMenu(agg_menu)
        agg_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tool.addWidget(agg_btn)
        lay.addLayout(tool)

        self._tree = _VarTree()
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._tree_context_menu)
        lay.addWidget(self._tree, 1)

        lay.addWidget(QtWidgets.QLabel(
            "<b>Derived</b> <span style='color:#777; font-size:11px'>"
            "— drag onto a plot</span>"))
        self._derived = _DerivedList()
        self._derived.setMaximumHeight(120)
        self._derived.changed.connect(self._publish_derived)
        lay.addWidget(self._derived)

        self._status = QtWidgets.QLabel()
        self._status.setStyleSheet("color:#777; font-size:11px;")
        lay.addWidget(self._status)

        self._build(type_name, medium, params)
        if derived:
            self._derived.set_payloads(derived, notify=False)
            if derived:
                self._status.setText(
                    f"{self._leaf_total} variable(s); "
                    f"{len(derived)} derived")

    def _publish_derived(self):
        if self._on_derived_changed is not None:
            self._on_derived_changed(self._derived.payloads())

    def _build(self, type_name, medium, params):
        tree = variable_tree(type_name, medium, params)
        if tree is None:
            self._status.setText(
                "Could not build this component — check its parameters.")
            return
        for node in tree.get("children", []):
            self._add_node(node, self._tree.invisibleRootItem())
        self._tree.expandToDepth(0)
        self._leaf_total = self._count_leaves(self._tree.invisibleRootItem())
        self._status.setText(f"{self._leaf_total} variable(s)")

    def _count_leaves(self, item) -> int:
        if item.data(0, _VarTree.PAYLOAD_ROLE):
            return 1
        return sum(self._count_leaves(item.child(i))
                   for i in range(item.childCount()))

    def _add_node(self, node: dict, parent):
        if node["leaf"]:
            item = QtWidgets.QTreeWidgetItem(parent, [node["name"], node["unit"]])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsDragEnabled)
            full = f"{self._comp_id}.{node['full']}"
            payload = {
                "full": full,
                "label": f"{self._comp_id}.{node['path']}",
                "name": node["name"],
                "unit": node["unit"],
                "description": node["description"],
                "kind": node["kind"],
                "value": node["value"],
            }
            item.setData(0, _VarTree.PAYLOAD_ROLE, payload)
            color = _KIND_COLOR.get(node["kind"], "#455a64")
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(color)))
            tip = f"<b>{full}</b>"
            if node["unit"]:
                tip += f" [{node['unit']}]"
            tip += f"<br><i>{node['kind']}</i>"
            if node["description"]:
                tip += f"<br>{node['description']}"
            item.setToolTip(0, tip)
        else:
            item = QtWidgets.QTreeWidgetItem(
                parent, [node["name"], f"({node.get('count', 0)})"])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDragEnabled)
            f = item.font(0)
            f.setBold(True)
            item.setFont(0, f)
            for child in node.get("children", []):
                self._add_node(child, item)

    def _apply_filter(self, text: str):
        needle = text.strip()
        rx = None
        if needle:
            try:
                rx = re.compile(needle, re.IGNORECASE)
            except re.error:
                rx = None
        visible = self._filter_item(self._tree.invisibleRootItem(), needle, rx)
        if needle:
            self._tree.expandAll()
            self._status.setText(
                f"{visible} / {self._leaf_total} variable(s) visible")
        else:
            self._status.setText(f"{self._leaf_total} variable(s)")

    def _filter_item(self, item, needle: str, rx) -> int:
        """Hide non-matching leaves; return count of visible leaves in subtree."""
        payload = item.data(0, _VarTree.PAYLOAD_ROLE)
        if payload:
            hay = f"{payload['full']} {payload['unit']} " \
                  f"{payload['description']}".lower()
            if rx is not None:
                match = rx.search(hay) is not None
            else:
                match = (needle.lower() in hay) if needle else True
            item.setHidden(not match)
            return 1 if match else 0
        visible = 0
        for i in range(item.childCount()):
            visible += self._filter_item(item.child(i), needle, rx)
        if item is not self._tree.invisibleRootItem():
            item.setHidden(needle != "" and visible == 0)
        return visible

    def _select_all_visible(self):
        self._tree.clearSelection()
        for item in self._tree.visible_leaf_items():
            item.setSelected(True)

    def _aggregate_selection(self, op: str):
        payloads = self._tree.selected_leaf_payloads()
        if not payloads:
            self._status.setText("Select one or more variables to aggregate.")
            return
        self._add_structural_derived(op, payloads)

    def _add_structural_derived(self, op: str, sources: list[dict]):
        fulls = [p["full"] for p in sources]
        unit = sources[0].get("unit", "")
        short = sources[0]["name"] if len(sources) == 1 else f"{op}({len(sources)})"
        label = f"{self._comp_id}.{short}"
        desc = f"{op} of " + ", ".join(p["name"] for p in sources[:6])
        if len(sources) > 6:
            desc += f" … (+{len(sources) - 6} more)"
        payload = make_derived_payload(
            op=op,
            axis="instances",
            label=label,
            unit=unit_for_agg(op, unit),
            description=desc,
            sources=fulls,
        )
        self._derived.add_payload(payload)
        self._status.setText(f"Derived {op} → drag from the list below.")

    def _add_temporal_derived(self, op: str, source: dict):
        if op == "abs":
            label = f"|{self._comp_id}.{source['name']}|"
            description = f"|{source['full']}|"
        else:
            label = f"{self._comp_id}.{source['name']} ({op})"
            description = f"{op} of {source['full']}"
        payload = make_derived_payload(
            op=op,
            axis="time",
            label=label,
            unit=unit_for_agg(op, source.get("unit", "")),
            description=description,
            sources=[source["full"]],
        )
        self._derived.add_payload(payload)
        self._status.setText(f"Derived {op} → drag from the list below.")

    def _tree_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(0, _VarTree.PAYLOAD_ROLE)
        if not payload:
            return
        menu = QtWidgets.QMenu(self)
        transform = menu.addMenu("Transform")
        for op, label in (("integral", "Time integral (∫)"),
                          ("cumsum", "Cumulative sum"),
                          ("abs", "Absolute value (|x|)")):
            act = transform.addAction(label)
            act.triggered.connect(
                lambda _=False, o=op, p=payload: self._add_temporal_derived(o, p))
        if len(self._tree.selected_leaf_payloads()) > 1:
            menu.addSeparator()
            sub = menu.addMenu("Aggregate selection")
            for op in STRUCTURAL_OPS:
                act = sub.addAction(op)
                act.triggered.connect(
                    lambda _=False, o=op: self._aggregate_selection(o))
        menu.exec(self._tree.viewport().mapToGlobal(pos))
