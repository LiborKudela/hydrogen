"""Left pane: the component catalogue tree (a drag source).

Groups every shipped component by physics *domain* (from
``hd.component_catalog()``); each leaf is a drag source carrying the
component's canonical ``type`` string on :data:`MIME_TYPE`.
"""

from __future__ import annotations

from hydrogen.components.icons import icon_path

from .qt import QtCore, QtGui, QtWidgets
from .style import domain_leaf_color

__all__ = ["MIME_TYPE", "ComponentTree"]

#: Custom drag payload: the component's canonical ``type`` string, UTF-8 encoded.
MIME_TYPE = "application/x-hydrogen-component"


class ComponentTree(QtWidgets.QTreeWidget):
    """Catalogue grouped by domain; leaves drag their component ``type``."""

    #: Role under which a leaf stores the full catalogue entry dict.
    ENTRY_ROLE = QtCore.Qt.UserRole

    def __init__(self, catalog: list[dict]):
        super().__init__()
        self.setHeaderHidden(True)
        self.setIconSize(QtCore.QSize(22, 22))
        font = self.font()
        font.setPointSize(font.pointSize() + 1)
        self.setFont(font)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._catalog: list[dict] = []
        self.set_catalog(catalog)

    def set_catalog(self, catalog: list[dict]):
        self._catalog = list(catalog)
        self.clear()
        by_domain: dict[str, dict[str, list[dict]]] = {}
        for entry in catalog:
            domain = entry["domain"]
            category = entry.get("category") or "components"
            by_domain.setdefault(domain, {}).setdefault(category, []).append(entry)

        for domain in sorted(by_domain):
            parent = QtWidgets.QTreeWidgetItem(self, [domain])
            # Domain headers are not draggable; only their component leaves are.
            self._style_group(parent)
            for category in sorted(by_domain[domain]):
                cat_item = QtWidgets.QTreeWidgetItem(parent, [category])
                self._style_group(cat_item)
                for entry in sorted(by_domain[domain][category], key=lambda e: e["name"]):
                    leaf = QtWidgets.QTreeWidgetItem(cat_item, [entry["name"]])
                    leaf.setData(0, self.ENTRY_ROLE, entry)
                    tip = entry.get("summary") or entry["type"]
                    if entry.get("needs_medium"):
                        tip += "\n(needs a medium)"
                    leaf.setToolTip(0, f"{entry['type']}\n\n{tip}")
                    leaf.setForeground(0, QtGui.QBrush(domain_leaf_color(domain)))
                    path = icon_path(entry.get("icon"))
                    if path:
                        leaf.setIcon(0, QtGui.QIcon(path))
        self.expandAll()

    def restyle(self):
        """Recolour leaf text for the active theme (called after a theme swap)
        without rebuilding the tree."""
        def walk(item):
            for i in range(item.childCount()):
                child = item.child(i)
                entry = child.data(0, self.ENTRY_ROLE)
                if entry:
                    child.setForeground(
                        0, QtGui.QBrush(domain_leaf_color(entry["domain"])))
                walk(child)
        root = self.invisibleRootItem()
        walk(root)

    @staticmethod
    def _style_group(item):
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDragEnabled)
        f = item.font(0)
        f.setBold(True)
        item.setFont(0, f)

    # QAbstractItemView calls this to build the drag payload from the selection.
    def mimeData(self, items):  # noqa: N802 (Qt camelCase override)
        md = QtCore.QMimeData()
        entry = items[0].data(0, self.ENTRY_ROLE) if items else None
        if entry:
            md.setData(MIME_TYPE, QtCore.QByteArray(entry["type"].encode("utf-8")))
            md.setText(entry["type"])
        return md
