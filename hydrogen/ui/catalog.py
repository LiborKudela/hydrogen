"""Left pane: the component catalogue tree (a drag source).

Groups every shipped component by physics *domain* (from
``hd.component_catalog()``); each leaf is a drag source carrying the
component's canonical ``type`` string on :data:`MIME_TYPE`.
"""

from __future__ import annotations

from .qt import QtCore, QtGui, QtWidgets
from .style import domain_color

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
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        by_domain: dict[str, list[dict]] = {}
        for entry in catalog:
            by_domain.setdefault(entry["domain"], []).append(entry)

        for domain in sorted(by_domain):
            parent = QtWidgets.QTreeWidgetItem(self, [domain])
            # Domain headers are not draggable; only their component leaves are.
            parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsDragEnabled)
            f = parent.font(0)
            f.setBold(True)
            parent.setFont(0, f)
            for entry in sorted(by_domain[domain], key=lambda e: e["name"]):
                leaf = QtWidgets.QTreeWidgetItem(parent, [entry["name"]])
                leaf.setData(0, self.ENTRY_ROLE, entry)
                tip = entry.get("summary") or entry["type"]
                if entry.get("needs_medium"):
                    tip += "\n(needs a medium)"
                leaf.setToolTip(0, f"{entry['type']}\n\n{tip}")
                leaf.setForeground(0, QtGui.QBrush(domain_color(domain).darker(170)))
        self.expandAll()

    # QAbstractItemView calls this to build the drag payload from the selection.
    def mimeData(self, items):  # noqa: N802 (Qt camelCase override)
        md = QtCore.QMimeData()
        entry = items[0].data(0, self.ENTRY_ROLE) if items else None
        if entry:
            md.setData(MIME_TYPE, QtCore.QByteArray(entry["type"].encode("utf-8")))
            md.setText(entry["type"])
        return md
