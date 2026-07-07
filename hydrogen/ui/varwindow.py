"""The per-component *Variables* window.

Opened from a placed component's context menu, it shows that component's model
variables in a tree that mirrors the component definition's structure
(sub-models -> groups, value objects -> leaves), each leaf annotated with its
unit and description.  Leaves are a drag source carrying the
:data:`~hydrogen.ui.plots.VARIABLE_MIME` payload, so they can be dropped onto a
Table or Timeseries object on the canvas.
"""

from __future__ import annotations

from .plots import encode_variables
from .qt import QtCore, QtGui, QtWidgets
from .varmeta import variable_tree

__all__ = ["VariablesWindow"]

#: Leaf-kind -> swatch colour (matches the four value-object classes).
_KIND_COLOR = {
    "differential": "#c62828",   # time-integrated state
    "variable": "#1565c0",       # algebraic variable
    "input": "#6a1b9a",          # external input signal
    "parameter": "#2e7d32",      # compile-time parameter
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


class VariablesWindow(QtWidgets.QDialog):
    """Non-modal browser of one component's variables (a drag source)."""

    def __init__(self, comp_id: str, type_name: str, medium: str | None,
                 params: dict | None, parent=None):
        super().__init__(parent)
        self._comp_id = comp_id
        self.setWindowTitle(f"Variables — {comp_id}")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.resize(420, 560)

        lay = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"<b>{comp_id}</b> <span style='color:#777'>{type_name}</span><br>"
            "<span style='color:#777; font-size:11px'>Drag a variable onto a "
            "Table or Timeseries object on the canvas.</span>")
        header.setWordWrap(True)
        lay.addWidget(header)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter variables…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self._filter)

        self._tree = _VarTree()
        lay.addWidget(self._tree, 1)

        self._status = QtWidgets.QLabel()
        self._status.setStyleSheet("color:#777; font-size:11px;")
        lay.addWidget(self._status)

        self._build(type_name, medium, params)

    def _build(self, type_name, medium, params):
        tree = variable_tree(type_name, medium, params)
        if tree is None:
            self._status.setText(
                "Could not build this component — check its parameters.")
            return
        for node in tree.get("children", []):
            self._add_node(node, self._tree.invisibleRootItem())
        self._tree.expandToDepth(0)
        self._status.setText(f"{tree.get('count', 0)} variable(s)")

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
        needle = text.strip().lower()
        self._filter_item(self._tree.invisibleRootItem(), needle)
        if needle:
            self._tree.expandAll()

    def _filter_item(self, item, needle: str) -> bool:
        payload = item.data(0, _VarTree.PAYLOAD_ROLE)
        child_count = item.childCount()
        if payload and child_count == 0:
            hay = f"{payload['full']} {payload['unit']} " \
                  f"{payload['description']}".lower()
            match = needle in hay if needle else True
            item.setHidden(not match)
            return match
        visible = 0
        for i in range(child_count):
            if self._filter_item(item.child(i), needle):
                visible += 1
        # invisibleRootItem has no setHidden of consequence, guard for it.
        if item is not self._tree.invisibleRootItem():
            item.setHidden(needle != "" and visible == 0)
        return visible > 0
