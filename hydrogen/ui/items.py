"""Canvas scene items: the connector dots, the component boxes, and the wires.

  * :class:`PortItem`       -- a connector dot on a node's edge.
  * :class:`NodeItem`       -- a movable box standing in for one component.
  * :class:`ConnectionItem` -- a spline wire between two compatible ports.

These three only ever talk *back* to their owning :class:`~hydrogen.ui.canvas.Canvas`
through duck-typed calls (``node._canvas.<method>``), so this module stays free
of an import cycle with the view.
"""

from __future__ import annotations

import math

from hydrogen.components.icons import icon_path

from .introspect import introspect_ports
from .qt import QtCore, QtGui, QtSvg, QtWidgets
from .style import domain_color, kind_color, port_on_right

__all__ = ["PortItem", "NodeItem", "ConnectionItem", "nearest_on_rect"]

#: One parsed :class:`QSvgRenderer` per icon path, shared across all nodes.
_ICON_RENDERERS: dict[str, "QtSvg.QSvgRenderer"] = {}


def icon_renderer(icon: str | None):
    """A cached SVG renderer for a component's ``UI_ICON`` filename (as surfaced
    by the catalog's ``"icon"`` field), or ``None`` if there's no icon, the file
    is missing, or the Qt binding lacks ``QtSvg`` -- in which case the node falls
    back to the generic box drawing."""
    if QtSvg is None:
        return None
    path = icon_path(icon)
    if not path:
        return None
    renderer = _ICON_RENDERERS.get(path)
    if renderer is None:
        renderer = QtSvg.QSvgRenderer(path)
        if not renderer.isValid():
            return None
        _ICON_RENDERERS[path] = renderer
    return renderer


def nearest_on_rect(rect: QtCore.QRectF, p: QtCore.QPointF) -> QtCore.QPointF:
    """Closest point on a rectangle's *perimeter* to ``p`` (snaps a port to
    whichever of the four edges is nearest the cursor)."""
    x = min(max(p.x(), rect.left()), rect.right())
    y = min(max(p.y(), rect.top()), rect.bottom())
    d = {"l": abs(p.x() - rect.left()), "r": abs(p.x() - rect.right()),
         "t": abs(p.y() - rect.top()), "b": abs(p.y() - rect.bottom())}
    edge = min(d, key=d.get)
    if edge == "l":
        return QtCore.QPointF(rect.left(), y)
    if edge == "r":
        return QtCore.QPointF(rect.right(), y)
    if edge == "t":
        return QtCore.QPointF(x, rect.top())
    return QtCore.QPointF(x, rect.bottom())


class PortItem(QtWidgets.QGraphicsEllipseItem):
    """A connector dot on a node's edge.  Press-drag from it to wire a spline
    to a compatible port (same ``kind``).  Coloured by kind; grows on hover."""

    R = 6.0

    def __init__(self, node: "NodeItem", name: str, kind: str,
                 local_pos: QtCore.QPointF, side: int = 1):
        # Bounding box leaves room for the hover-enlarged radius.
        super().__init__(-(self.R + 2), -(self.R + 2),
                         2 * (self.R + 2), 2 * (self.R + 2), node)
        self.node = node
        self.pname = name
        self.kind = kind
        self.side = side  # +1 = right edge (faces right), -1 = left edge
        self.connections: list["ConnectionItem"] = []
        self._hovered = False
        self._moving = False
        self.setPos(local_pos)
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setToolTip(f"{name}  ·  {kind}  (Ctrl+drag to reposition)")

        self._label = QtWidgets.QGraphicsSimpleTextItem(name, self)
        lf = self._label.font()
        lf.setPointSize(7)
        self._label.setFont(lf)
        self._label.setBrush(QtGui.QColor("#333"))
        self._reposition_label()

    def _reposition_label(self):
        """Anchor the label outside whichever edge the port sits on, so it never
        overlaps the block: above on top, below on bottom, beside on sides."""
        rect = self.node.rect()
        p = self.pos()
        br = self._label.boundingRect()
        eps = 0.5
        if abs(p.y() - rect.top()) < eps:          # top edge -> above the dot
            self._label.setPos(-br.width() / 2, -br.height() - 8)
        elif abs(p.y() - rect.bottom()) < eps:     # bottom edge -> below the dot
            self._label.setPos(-br.width() / 2, 8)
        elif abs(p.x() - rect.right()) < eps or self.side == 1:
            self._label.setPos(-10 - br.width(), -br.height() / 2)  # right -> left
        else:                                      # left edge -> right of the dot
            self._label.setPos(10, -br.height() / 2)

    def move_to_cursor(self, scene_pos: QtCore.QPointF):
        """Drag the port along the node's edge, following the cursor."""
        node = self.node
        rect = node.rect()
        local = node.mapFromScene(scene_pos)      # undo the node's transform
        pt = nearest_on_rect(rect, local)
        self.setPos(pt)
        self.side = 1 if pt.x() >= rect.center().x() else -1
        self._reposition_label()
        node._keep_text_upright()                 # keep label upright if rotated
        node._canvas.refresh_connections(node)

    def scene_center(self) -> QtCore.QPointF:
        return self.scenePos()  # item origin == the dot's centre

    def _edge_dir_local(self) -> QtCore.QPointF:
        """Unit vector (in the node's local frame) pointing straight out of the
        edge the port sits on -- so a wire can leave perpendicular to the block:
        left/right -> horizontal, top/bottom -> vertical."""
        rect = self.node.rect()
        p = self.pos()
        eps = 0.5
        if abs(p.y() - rect.top()) < eps:
            return QtCore.QPointF(0.0, -1.0)
        if abs(p.y() - rect.bottom()) < eps:
            return QtCore.QPointF(0.0, 1.0)
        if abs(p.x() - rect.left()) < eps:
            return QtCore.QPointF(-1.0, 0.0)
        if abs(p.x() - rect.right()) < eps:
            return QtCore.QPointF(1.0, 0.0)
        return QtCore.QPointF(float(self.side), 0.0)  # fallback: face left/right

    def out_dir(self) -> QtCore.QPointF:
        """The port's outward direction in *scene* coordinates (the node's
        rotation / mirroring is folded in), as a unit vector."""
        t = self.node.sceneTransform()
        origin = t.map(QtCore.QPointF(0.0, 0.0))
        tip = t.map(self._edge_dir_local())
        vec = tip - origin                         # map() carries translation; drop it
        length = math.hypot(vec.x(), vec.y()) or 1.0
        return QtCore.QPointF(vec.x() / length, vec.y() / length)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.R + (2 if self._hovered else 0)
        pen = QtGui.QPen(QtGui.QColor("#1b1b1b") if self._hovered
                         else QtGui.QColor("#3a3a3a"))
        pen.setWidth(2 if self._hovered else 1)
        painter.setPen(pen)
        painter.setBrush(kind_color(self.kind))
        painter.drawEllipse(QtCore.QPointF(0, 0), r, r)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                self._moving = True            # Ctrl+drag -> reposition the port
                self.setCursor(QtCore.Qt.SizeAllCursor)
            else:
                self.node._canvas.begin_connection(self, event.scenePos())
            event.accept()  # grab the mouse; keep the node from moving
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._moving:
            self.move_to_cursor(event.scenePos())
        else:
            self.node._canvas.update_connection(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._moving:
            self._moving = False
            self.setCursor(QtCore.Qt.CrossCursor)
            self.node.store_port_layout(self)
            self.node._canvas._on_status(
                f"Moved port '{self.node.comp_id}.{self.pname}'")
        else:
            self.node._canvas.finish_connection(event.scenePos())
        event.accept()


class NodeItem(QtWidgets.QGraphicsRectItem):
    """A movable box on the canvas standing in for one component instance.

    Holds the per-instance spec the canvas serialises: ``comp_id`` (its key in
    the system's ``components`` map), an optional ``medium`` name, and a
    ``params`` dict (``None`` until the user edits it -> falls back to the
    component template's defaults).  Its connector :class:`PortItem` children
    are derived from the component's actual ports.
    """

    HEADER_H = 36.0
    ROW_H = 22.0
    ICON_BODY_H = 70.0     # min body height when a node renders an SVG symbol

    def __init__(self, entry: dict, comp_id: str, canvas: "Canvas"):
        super().__init__(0, 0, 190, 58)
        self.entry = entry
        self.type_name = entry["type"]
        self.comp_id = comp_id
        self._canvas = canvas
        self.params: dict | None = None
        self.medium: str | None = "Hydrogen" if entry.get("needs_medium") else None
        self._fill = domain_color(entry["domain"])
        self._icon = icon_renderer(entry.get("icon"))   # None -> generic box render
        self._hovered = False
        self.port_items: list[PortItem] = []
        # Custom port placements: pname -> (x, y, side); empty = use defaults.
        self._port_layout: dict[str, tuple] = {}
        # Visual transform state (applied around the node's centre).
        self._rot = 0           # degrees, multiples of 90
        self._mirror_x = False
        self._mirror_y = False
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self.setToolTip(f"{entry['type']}\n\n{entry.get('summary', '')}")

        self._title = QtWidgets.QGraphicsSimpleTextItem(entry["name"], self)
        tf = self._title.font()
        tf.setBold(True)
        tf.setPointSize(10)
        self._title.setFont(tf)
        self._title.setPos(12, 7)
        self._title.setZValue(6)

        sub_text = comp_id + ("  *" if entry.get("needs_medium") else "")
        self._sub = QtWidgets.QGraphicsSimpleTextItem(sub_text, self)
        sf = self._sub.font()
        sf.setPointSize(8)
        self._sub.setFont(sf)
        self._sub.setBrush(QtGui.QColor("#555"))
        self._sub.setPos(12, 21)
        self._sub.setZValue(6)

        self.rebuild_ports()

    # --- ports -------------------------------------------------------------- #
    def rebuild_ports(self, specs: list[tuple[str, str]] | None = None):
        scene = self.scene()
        for pi in self.port_items:
            for conn in list(pi.connections):
                conn.remove()
            if scene is not None:
                scene.removeItem(pi)      # takes its label child with it
        self.port_items = []

        if specs is None:
            specs = introspect_ports(self.type_name, self.medium, self.params)
        left = [s for s in specs if not port_on_right(s[0])]
        right = [s for s in specs if port_on_right(s[0])]

        rows = max(len(left), len(right), 1)
        width = max(190.0, self._title.boundingRect().width() + 28)
        height = self.HEADER_H + rows * self.ROW_H + 6
        if self._icon is not None:           # give the symbol room to breathe
            height = max(height, self.HEADER_H + self.ICON_BODY_H)
        self.setRect(0, 0, width, height)
        self._place_side(left, x=0.0, align_left=True)
        self._place_side(right, x=width, align_left=False)
        self.apply_transform()   # rect changed -> recompute centre-anchored xform

    def _place_side(self, specs, x: float, align_left: bool):
        # With an icon the ports are centred on the body (so they line up with
        # the symbol's inlet/outlet); the generic box keeps them top-aligned.
        n = len(specs)
        if self._icon is not None and n:
            body = self.rect().height() - self.HEADER_H
            top = self.HEADER_H + (body - n * self.ROW_H) / 2
        else:
            top = self.HEADER_H
        for i, (name, kind) in enumerate(specs):
            y = top + self.ROW_H / 2 + i * self.ROW_H
            side = -1 if align_left else 1
            lx, ly = x, y
            if name in self._port_layout:         # restore a custom placement
                lx, ly, side = self._port_layout[name]
            port = PortItem(self, name, kind, QtCore.QPointF(lx, ly), side)
            self.port_items.append(port)

    def store_port_layout(self, port: "PortItem"):
        self._port_layout[port.pname] = (port.pos().x(), port.pos().y(), port.side)

    def sync_ports(self) -> bool:
        """Re-derive ports after a parameter edit; rebuild only if the port set
        actually changed (so ordinary edits keep existing wires)."""
        new = introspect_ports(self.type_name, self.medium, self.params)
        if {(p.pname, p.kind) for p in self.port_items} == set(new):
            return False
        self.rebuild_ports(new)
        return True

    # --- transforms (rotate / mirror, around the node centre) --------------- #
    def apply_transform(self):
        rect = self.rect()
        cx, cy = rect.center().x(), rect.center().y()
        t = QtGui.QTransform()
        t.translate(cx, cy)
        t.rotate(self._rot)
        t.scale(-1 if self._mirror_x else 1, -1 if self._mirror_y else 1)
        t.translate(-cx, -cy)
        self.setTransform(t)
        self._keep_text_upright()
        if self._canvas is not None:
            self._canvas.refresh_connections(self)

    def _keep_text_upright(self):
        """Counter-transform the text so labels stay readable (upright, not
        mirrored) while the box and ports rotate/flip around them."""
        texts = [self._title, self._sub]
        for pi in self.port_items:
            texts.extend(pi.childItems())
        for it in texts:
            tc = it.boundingRect().center()
            tt = QtGui.QTransform()
            tt.translate(tc.x(), tc.y())
            tt.scale(-1 if self._mirror_x else 1, -1 if self._mirror_y else 1)
            tt.rotate(-self._rot)
            tt.translate(-tc.x(), -tc.y())
            it.setTransform(tt)
        self._place_title_text()

    def _place_title_text(self):
        """Pin the block title to the node's visual top edge.

        The node itself rotates around its centre, so the original local header
        may end up at the bottom or side.  Work in scene coordinates here: place
        the two title lines at the top-left of the transformed node bounds, then
        map those points back into the node's local coordinates.
        """
        bbox = self.sceneTransform().mapRect(self.rect())
        title_at = QtCore.QPointF(bbox.left() + 12.0, bbox.top() + 7.0)
        sub_at = QtCore.QPointF(bbox.left() + 12.0, bbox.top() + 21.0)
        self._set_text_scene_origin(self._title, title_at)
        self._set_text_scene_origin(self._sub, sub_at)

    def _set_text_scene_origin(self, item, scene_pos: QtCore.QPointF):
        inv, ok = self.sceneTransform().inverted()
        if not ok:
            return
        local_pos = inv.map(scene_pos)
        # Counter-rotation around the text's own centre moves its local origin;
        # compensate so the rendered text starts at the requested scene point.
        offset = item.transform().map(QtCore.QPointF(0.0, 0.0))
        item.setPos(local_pos - offset)

    def _visual_header_line(self) -> tuple[QtCore.QPointF, QtCore.QPointF] | None:
        """Line under the visual title, kept with the text after transforms."""
        inv, ok = self.sceneTransform().inverted()
        if not ok:
            return None
        bbox = self.sceneTransform().mapRect(self.rect())
        y = bbox.top() + self.HEADER_H
        return (
            inv.map(QtCore.QPointF(bbox.left() + 6.0, y)),
            inv.map(QtCore.QPointF(bbox.right() - 6.0, y)),
        )

    def _visual_icon_rect(self, icon_size) -> QtCore.QRectF | None:
        """Icon slot centred below the visual title/header line."""
        inv, ok = self.sceneTransform().inverted()
        if not ok:
            return None
        bbox = self.sceneTransform().mapRect(self.rect())
        body = QtCore.QRectF(
            bbox.left() + 12.0,
            bbox.top() + self.HEADER_H + 8.0,
            max(1.0, bbox.width() - 24.0),
            max(1.0, bbox.height() - self.HEADER_H - 18.0),
        )
        # The icon still rotates with the block. For quarter-turns, local width
        # becomes visual height and local height becomes visual width.
        if self._rot % 180:
            available = QtCore.QRectF(0.0, 0.0, body.height(), body.width())
        else:
            available = QtCore.QRectF(0.0, 0.0, body.width(), body.height())
        fitted = self._fit(icon_size, available)
        center = inv.map(body.center())
        return QtCore.QRectF(
            center.x() - fitted.width() / 2.0,
            center.y() - fitted.height() / 2.0,
            fitted.width(),
            fitted.height(),
        )

    def rotate_by(self, degrees: int):
        self._rot = (self._rot + degrees) % 360
        self.apply_transform()

    def mirror(self, axis: str):
        if axis == "x":
            self._mirror_x = not self._mirror_x
        else:
            self._mirror_y = not self._mirror_y
        self.apply_transform()

    def reset_transform(self):
        self._rot = 0
        self._mirror_x = self._mirror_y = False
        self.apply_transform()

    # --- geometry / hover --------------------------------------------------- #
    def itemChange(self, change, value):
        if (change == QtWidgets.QGraphicsItem.ItemPositionHasChanged
                and self._canvas is not None):
            self._canvas.refresh_connections(self)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()

    @staticmethod
    def _fit(size, into: QtCore.QRectF) -> QtCore.QRectF:
        """Aspect-preserving rect for ``size`` centred inside ``into``."""
        w, h = size.width(), size.height()
        if w <= 0 or h <= 0:
            return into
        s = min(into.width() / w, into.height() / h)
        fw, fh = w * s, h * s
        return QtCore.QRectF(into.left() + (into.width() - fw) / 2,
                             into.top() + (into.height() - fh) / 2, fw, fh)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        selected = self.isSelected()
        if selected:
            color, w = QtGui.QColor("#1b1b1b"), 2
        elif self._hovered:
            color, w = QtGui.QColor("#1b6fb3"), 2
        else:
            color, w = QtGui.QColor("#5a5a5a"), 1
        rect = self.rect()

        if self._icon is not None:
            # P&ID symbol on a clean (white) card: a domain-coloured header strip
            # carries the labels, the SVG fills the body below the separator.
            painter.setPen(QtGui.QPen(color, w))
            painter.setBrush(QtGui.QColor("#fdfdfd") if not self._hovered
                             else QtGui.QColor("#ffffff"))
            painter.drawRoundedRect(rect, 9, 9)
            painter.setPen(QtGui.QPen(self._fill.darker(135), 1.5))
            header_line = self._visual_header_line()
            if header_line is not None:
                painter.drawLine(*header_line)
            body = self._visual_icon_rect(self._icon.defaultSize())
            if body is not None:
                self._icon.render(painter, body)
            return

        painter.setPen(QtGui.QPen(color, w))
        painter.setBrush(self._fill.lighter(106) if self._hovered else self._fill)
        painter.drawRoundedRect(rect, 9, 9)


class ConnectionItem(QtWidgets.QGraphicsPathItem):
    """A spline wire between two ports, coloured by their (shared) ``kind``."""

    def __init__(self, src: PortItem, dst: PortItem, canvas: "Canvas"):
        super().__init__()
        self.src = src
        self.dst = dst
        self._canvas = canvas
        self.kind = src.kind
        self._hovered = False
        self.setZValue(-1)               # wires sit behind the nodes
        self.setAcceptHoverEvents(True)
        self.setPen(self._make_pen())
        src.connections.append(self)
        dst.connections.append(self)
        self.update_path()

    def _make_pen(self) -> QtGui.QPen:
        col = kind_color(self.kind)
        if self._hovered:
            col = col.darker(135)
        pen = QtGui.QPen(col, 4 if self._hovered else 2.2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        return pen

    @staticmethod
    def _spline(p1: QtCore.QPointF, p2: QtCore.QPointF,
                d1: QtCore.QPointF, d2: QtCore.QPointF) -> QtGui.QPainterPath:
        """Cubic between two points; each control handle is pushed along the
        endpoint's outward direction ``d`` (a unit vector), so a wire leaves a
        port *perpendicular* to whichever edge it sits on (horizontal for
        left/right ports, vertical for top/bottom ports)."""
        reach = max(40.0, (abs(p2.x() - p1.x()) + abs(p2.y() - p1.y())) * 0.4)
        c1 = QtCore.QPointF(p1.x() + d1.x() * reach, p1.y() + d1.y() * reach)
        c2 = QtCore.QPointF(p2.x() + d2.x() * reach, p2.y() + d2.y() * reach)
        path = QtGui.QPainterPath(p1)
        path.cubicTo(c1, c2, p2)
        return path

    def update_path(self):
        self.setPath(self._spline(self.src.scene_center(), self.dst.scene_center(),
                                  self.src.out_dir(), self.dst.out_dir()))

    def shape(self):  # fat hit area so hover / right-click are easy
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.setPen(self._make_pen())

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.setPen(self._make_pen())

    def remove(self):
        for pi in (self.src, self.dst):
            if self in pi.connections:
                pi.connections.remove(self)
        if self.scene() is not None:
            self.scene().removeItem(self)
        self._canvas.forget_connection(self)
