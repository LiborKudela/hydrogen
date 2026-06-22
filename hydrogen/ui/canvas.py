"""Right pane: the :class:`Canvas` graphics view.

Accepts drops from the catalogue tree, owns the placed :class:`NodeItem`s and
the :class:`ConnectionItem` wires between their ports, and drives the
interaction loop (drag-to-wire, zoom/pan, per-node context menu, selection).
"""

from __future__ import annotations

import re

from .catalog import MIME_TYPE
from .items import ConnectionItem, NodeItem, PortItem
from .properties import PropertiesDialog
from .qt import QtCore, QtGui, QtWidgets, drop_point, exec_
from .style import kind_color

__all__ = ["Canvas"]


def _transform_icon(kind: str) -> QtGui.QIcon:
    """Small theme-independent icons for the transform context menu."""
    pix = QtGui.QPixmap(18, 18)
    pix.fill(QtCore.Qt.transparent)

    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor("#4b5563"), 1.8)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(QtGui.QColor("#4b5563"))

    if kind in {"rotate-cw", "rotate-ccw", "rotate-180", "transforms"}:
        rect = QtCore.QRectF(3.5, 3.5, 11.0, 11.0)
        if kind == "rotate-ccw":
            p.drawArc(rect, 35 * 16, 280 * 16)
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(4.3, 6.0),
                QtCore.QPointF(4.1, 2.6),
                QtCore.QPointF(7.0, 4.3),
            ]))
        elif kind == "rotate-180":
            p.drawArc(rect, 20 * 16, 320 * 16)
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(13.7, 12.0),
                QtCore.QPointF(13.9, 15.4),
                QtCore.QPointF(11.0, 13.7),
            ]))
            p.drawText(QtCore.QRectF(5.0, 5.0, 8.0, 8.0),
                       QtCore.Qt.AlignCenter, "2")
        else:
            p.drawArc(rect, -315 * 16, 280 * 16)
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(13.7, 6.0),
                QtCore.QPointF(13.9, 2.6),
                QtCore.QPointF(11.0, 4.3),
            ]))
    elif kind == "mirror-h":
        p.drawLine(QtCore.QPointF(9, 3), QtCore.QPointF(9, 15))
        p.drawLine(QtCore.QPointF(3, 6), QtCore.QPointF(7, 9))
        p.drawLine(QtCore.QPointF(3, 12), QtCore.QPointF(7, 9))
        p.drawLine(QtCore.QPointF(15, 6), QtCore.QPointF(11, 9))
        p.drawLine(QtCore.QPointF(15, 12), QtCore.QPointF(11, 9))
    elif kind == "mirror-v":
        p.drawLine(QtCore.QPointF(3, 9), QtCore.QPointF(15, 9))
        p.drawLine(QtCore.QPointF(6, 3), QtCore.QPointF(9, 7))
        p.drawLine(QtCore.QPointF(12, 3), QtCore.QPointF(9, 7))
        p.drawLine(QtCore.QPointF(6, 15), QtCore.QPointF(9, 11))
        p.drawLine(QtCore.QPointF(12, 15), QtCore.QPointF(9, 11))
    elif kind == "reset":
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(QtCore.QRectF(4, 5, 10, 8), 1.8, 1.8)
        p.setBrush(QtGui.QColor("#4b5563"))
        p.drawLine(QtCore.QPointF(5.0, 3.0), QtCore.QPointF(2.5, 3.0))
        p.drawLine(QtCore.QPointF(2.5, 3.0), QtCore.QPointF(2.5, 5.5))
        p.drawLine(QtCore.QPointF(2.5, 3.0), QtCore.QPointF(5.0, 5.5))

    p.end()
    return QtGui.QIcon(pix)


class Canvas(QtWidgets.QGraphicsView):
    """Graphics canvas that accepts drops from the catalogue tree."""

    def __init__(self, by_type: dict[str, dict], on_status):
        super().__init__()
        self._by_type = by_type
        self._on_status = on_status
        self._counter = 0
        self._connections: list[ConnectionItem] = []
        self._pending: PortItem | None = None   # source port of an in-flight wire
        self._temp: QtWidgets.QGraphicsPathItem | None = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QtCore.QPoint()
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setSceneRect(-2000, -2000, 6800, 5200)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        # Zoom centred on the cursor; pan with the middle mouse button.
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._scene.selectionChanged.connect(self._report_selection)

    # --- zoom & pan --------------------------------------------------------- #
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.0015 ** delta
        new_zoom = self._zoom * factor
        if new_zoom < 0.2 or new_zoom > 6.0:
            return
        self._zoom = new_zoom
        self.scale(factor, factor)

    def reset_zoom(self):
        self.resetTransform()
        self._zoom = 1.0

    def fit_view(self):
        """Zoom/pan so every placed component fits in the viewport."""
        nodes = self.nodes()
        if not nodes:
            self._on_status("Nothing to fit — canvas is empty.")
            return
        rect = QtCore.QRectF()
        for node in nodes:
            box = node.sceneBoundingRect()
            rect = box if rect.isNull() else rect.united(box)
        rect.adjust(-40, -40, 40, 40)   # breathing room around the bounds
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()  # keep wheel-zoom clamping in sync
        self._on_status(f"Fit {len(nodes)} component(s) in view")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # --- canvas grid backdrop ---------------------------------------------- #
    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        step = 24
        pen = QtGui.QPen(QtGui.QColor("#ececec"))
        pen.setWidth(0)
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % step)
        top = int(rect.top()) - (int(rect.top()) % step)
        lines = []
        x = left
        while x < rect.right():
            lines.append(QtCore.QLineF(x, rect.top(), x, rect.bottom()))
            x += step
        y = top
        while y < rect.bottom():
            lines.append(QtCore.QLineF(rect.left(), y, rect.right(), y))
            y += step
        painter.drawLines(lines)

    # --- drop handling ------------------------------------------------------ #
    def _accepts(self, event) -> bool:
        md = event.mimeData()
        return md.hasFormat(MIME_TYPE) or md.hasText()

    def dragEnterEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasFormat(MIME_TYPE):
            type_name = bytes(md.data(MIME_TYPE)).decode("utf-8")
        elif md.hasText():
            type_name = md.text()
        else:
            event.ignore()
            return
        entry = self._by_type.get(type_name)
        if entry is None:
            event.ignore()
            return
        self.add_node(entry, self.mapToScene(drop_point(event)))
        event.acceptProposedAction()

    # --- node lifecycle ----------------------------------------------------- #
    def _next_id(self, name: str) -> str:
        base = re.sub(r"\W+", "_", name).strip("_").lower() or "comp"
        self._counter += 1
        return f"{base}_{self._counter}"

    def add_node(self, entry: dict, scene_pos: QtCore.QPointF) -> NodeItem:
        node = NodeItem(entry, self._next_id(entry["name"]), self)
        r = node.rect()
        node.setPos(scene_pos.x() - r.width() / 2, scene_pos.y() - r.height() / 2)
        self._scene.addItem(node)
        self._scene.clearSelection()
        node.setSelected(True)
        self._on_status(f"Added '{node.comp_id}'  ({self.node_count()} on canvas)")
        return node

    def add_node_at_center(self, entry: dict) -> NodeItem:
        return self.add_node(entry, self.mapToScene(self.viewport().rect().center()))

    # --- project (canvas) save / load --------------------------------------- #
    def to_project(self) -> dict:
        """Full canvas state: every node (placement, transform, medium, params)
        plus the wiring between them."""
        nodes = []
        for n in self.nodes():
            nodes.append({
                "comp_id": n.comp_id,
                "type": n.type_name,
                "x": n.pos().x(),
                "y": n.pos().y(),
                "rot": n._rot,
                "mirror_x": n._mirror_x,
                "mirror_y": n._mirror_y,
                "medium": n.medium,
                "params": n.params,
                "ports": {name: list(layout)
                          for name, layout in n._port_layout.items()},
            })
        connections = [
            {"from": f"{c.src.node.comp_id}.{c.src.pname}",
             "to": f"{c.dst.node.comp_id}.{c.dst.pname}"}
            for c in self.connections()
        ]
        return {"nodes": nodes, "connections": connections}

    def load_project(self, data: dict):
        """Rebuild the canvas from :meth:`to_project` output (replacing it)."""
        self.clear_nodes()
        by_id: dict[str, NodeItem] = {}
        max_suffix = 0
        for nd in data.get("nodes", []):
            entry = self._by_type.get(nd["type"])
            if entry is None:
                self._on_status(f"Skipped unknown component type {nd['type']!r}")
                continue
            node = NodeItem(entry, nd["comp_id"], self)
            node.medium = nd.get("medium")
            node.params = nd.get("params")
            node._port_layout = {name: tuple(layout)
                                 for name, layout in nd.get("ports", {}).items()}
            node.rebuild_ports()                    # ports match params + layout
            node.setPos(nd.get("x", 0.0), nd.get("y", 0.0))
            node._rot = int(nd.get("rot", 0))
            node._mirror_x = bool(nd.get("mirror_x", False))
            node._mirror_y = bool(nd.get("mirror_y", False))
            node.apply_transform()
            self._scene.addItem(node)
            by_id[node.comp_id] = node
            m = re.search(r"_(\d+)$", node.comp_id)
            if m:
                max_suffix = max(max_suffix, int(m.group(1)))
        self._counter = max(self._counter, max_suffix)  # avoid future id clashes

        for cn in data.get("connections", []):
            src = self._port_by_path(by_id, cn.get("from", ""))
            dst = self._port_by_path(by_id, cn.get("to", ""))
            if src is None or dst is None:
                self._on_status(f"Skipped connection {cn}")
                continue
            conn = ConnectionItem(src, dst, self)
            self._scene.addItem(conn)
            self._connections.append(conn)
        self._on_status(
            f"Loaded {len(by_id)} component(s), {len(self._connections)} wire(s)")

    @staticmethod
    def _port_by_path(by_id: dict, path: str) -> "PortItem | None":
        comp_id, _, pname = path.rpartition(".")
        node = by_id.get(comp_id)
        if node is None:
            return None
        return next((p for p in node.port_items if p.pname == pname), None)

    def nodes(self) -> list[NodeItem]:
        return [i for i in self._scene.items() if isinstance(i, NodeItem)]

    def node_count(self) -> int:
        return len(self.nodes())

    def remove_node(self, node: NodeItem):
        for pi in node.port_items:
            for conn in list(pi.connections):
                conn.remove()
        self._scene.removeItem(node)

    def clear_nodes(self):
        for node in self.nodes():
            self.remove_node(node)
        self._connections.clear()
        self._on_status("Canvas cleared")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape and self._temp is not None:
            self._clear_temp()
            self._on_status("Connection cancelled")
            return
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            removed = [i for i in self._scene.selectedItems() if isinstance(i, NodeItem)]
            for node in removed:
                self.remove_node(node)
            if removed:
                self._on_status(f"Removed {len(removed)} node(s)  "
                                f"({self.node_count()} on canvas)")
                return
        super().keyPressEvent(event)

    # --- connection lifecycle (driven by PortItem mouse events) ------------- #
    def connections(self) -> list[ConnectionItem]:
        return list(self._connections)

    def forget_connection(self, conn: ConnectionItem):
        if conn in self._connections:
            self._connections.remove(conn)

    def refresh_connections(self, node: NodeItem):
        for pi in node.port_items:
            for conn in pi.connections:
                conn.update_path()

    def begin_connection(self, port: PortItem, scene_pos: QtCore.QPointF):
        self._pending = port
        self._temp = QtWidgets.QGraphicsPathItem()
        pen = QtGui.QPen(kind_color(port.kind), 2)
        pen.setStyle(QtCore.Qt.DashLine)
        self._temp.setPen(pen)
        self._temp.setZValue(-1)
        self._scene.addItem(self._temp)
        self.update_connection(scene_pos)

    def update_connection(self, scene_pos: QtCore.QPointF):
        if self._temp is not None and self._pending is not None:
            p1 = self._pending.scene_center()
            # Cursor end has no port: let it "enter" from whichever side faces
            # the source so the live wire stays smooth in any drag direction.
            d2 = QtCore.QPointF(-1.0, 0.0) if scene_pos.x() >= p1.x() \
                else QtCore.QPointF(1.0, 0.0)
            self._temp.setPath(ConnectionItem._spline(
                p1, scene_pos, self._pending.out_dir(), d2))

    def finish_connection(self, scene_pos: QtCore.QPointF):
        src = self._pending
        target = self._port_at(scene_pos)
        self._clear_temp()
        if src is None:
            return
        ok, msg = self._can_connect(src, target)
        if not ok:
            self._on_status(msg)
            return
        conn = ConnectionItem(src, target, self)
        self._scene.addItem(conn)
        self._connections.append(conn)
        self._on_status(
            f"connected {src.node.comp_id}.{src.pname} ↔ "
            f"{target.node.comp_id}.{target.pname}")

    def _clear_temp(self):
        if self._temp is not None:
            self._scene.removeItem(self._temp)
            self._temp = None
        self._pending = None

    def _port_at(self, scene_pos: QtCore.QPointF) -> PortItem | None:
        for it in self._scene.items(scene_pos):
            if isinstance(it, PortItem):
                return it
        return None

    def _can_connect(self, a: PortItem, b: PortItem | None):
        if b is None:
            return False, "Connection cancelled (released on empty canvas)."
        if a is b or a.node is b.node:
            return False, "Can't wire a component to itself."
        if a.kind != b.kind:
            return False, f"Port kind mismatch: {a.kind} ↔ {b.kind}."
        for conn in a.connections:
            if {conn.src, conn.dst} == {a, b}:
                return False, "Those ports are already connected."
        return True, ""

    # --- per-node context menu ---------------------------------------------- #
    def _node_at(self, view_pos: QtCore.QPoint) -> NodeItem | None:
        item = self.itemAt(view_pos)
        while item is not None and not isinstance(item, NodeItem):
            item = item.parentItem()
        return item

    def contextMenuEvent(self, event):
        items_at = self.items(event.pos())
        top = items_at[0] if items_at else None

        # Right-click on a wire -> just a Delete action.
        if isinstance(top, ConnectionItem):
            menu = QtWidgets.QMenu(self)
            act_del = menu.addAction("Delete connection")
            if exec_(menu, event.globalPos()) == act_del:
                label = f"{top.src.node.comp_id}.{top.src.pname} ↔ " \
                        f"{top.dst.node.comp_id}.{top.dst.pname}"
                top.remove()
                self._on_status(f"Removed connection {label}")
            return

        node = self._node_at(event.pos())
        if node is None:
            return
        menu = QtWidgets.QMenu(self)
        act_props = menu.addAction("Properties…")

        tmenu = menu.addMenu("Transforms")
        tmenu.setIcon(_transform_icon("transforms"))
        act_cw = tmenu.addAction(_transform_icon("rotate-cw"), "Rotate 90° CW")
        act_ccw = tmenu.addAction(_transform_icon("rotate-ccw"), "Rotate 90° CCW")
        act_180 = tmenu.addAction(_transform_icon("rotate-180"), "Rotate 180°")
        tmenu.addSeparator()
        act_mh = tmenu.addAction(_transform_icon("mirror-h"), "Mirror horizontal")
        act_mv = tmenu.addAction(_transform_icon("mirror-v"), "Mirror vertical")
        tmenu.addSeparator()
        act_reset = tmenu.addAction(_transform_icon("reset"), "Reset transform")

        menu.addSeparator()
        act_del = menu.addAction("Delete")
        chosen = exec_(menu, event.globalPos())
        if chosen == act_props:
            self.open_properties(node)
        elif chosen == act_cw:
            node.rotate_by(90)
            self._on_status(f"Rotated '{node.comp_id}' to {node._rot}°")
        elif chosen == act_ccw:
            node.rotate_by(-90)
            self._on_status(f"Rotated '{node.comp_id}' to {node._rot}°")
        elif chosen == act_180:
            node.rotate_by(180)
            self._on_status(f"Rotated '{node.comp_id}' to {node._rot}°")
        elif chosen == act_mh:
            node.mirror("x")
            self._on_status(f"Mirrored '{node.comp_id}' horizontally")
        elif chosen == act_mv:
            node.mirror("y")
            self._on_status(f"Mirrored '{node.comp_id}' vertically")
        elif chosen == act_reset:
            node.reset_transform()
            self._on_status(f"Reset transform on '{node.comp_id}'")
        elif chosen == act_del:
            self.remove_node(node)
            self._on_status(f"Removed '{node.comp_id}'  "
                            f"({self.node_count()} on canvas)")

    def open_properties(self, node: NodeItem):
        dlg = PropertiesDialog(node, parent=self)
        if exec_(dlg):
            changed = node.sync_ports()
            self._on_status(
                f"Updated '{node.comp_id}'"
                + ("  (ports changed -> wires reset)" if changed else ""))
        node.update()

    def mouseDoubleClickEvent(self, event):
        node = self._node_at(event.pos())
        if node is not None:
            self.open_properties(node)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _report_selection(self):
        sel = [i for i in self._scene.selectedItems() if isinstance(i, NodeItem)]
        if len(sel) == 1:
            self._on_status(f"{sel[0].entry['type']} — "
                            f"{sel[0].entry.get('summary', '')}")
