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
from typing import TYPE_CHECKING

from hydrogen.components.icons import icon_path

from . import theme
from .introspect import introspect, introspect_ports
from .qt import QtCore, QtGui, QtSvg, QtWidgets
from .style import domain_color, kind_color, port_on_right

if TYPE_CHECKING:
    from .canvas import Canvas

__all__ = ["PortItem", "NodeItem", "ConnectionItem", "nearest_on_rect",
           "display_type_name"]


def display_type_name(entry: dict) -> str:
    """The type as shown to the user, including the component's submodule.

    The *serialized* type is domain-namespaced only (``hydrogen.thermofluid.Pipe``)
    so names never collide across domains, but that hides which submodule a leaf
    lives in.  For display we splice the catalog's ``category`` (the submodule
    inside the domain, e.g. ``assemblies`` / ``flow``) back in, giving the fully
    qualified ``hydrogen.thermofluid.assemblies.Pipe``.  Falls back to the bare
    type when there is no category (or the name isn't a ``hydrogen.<domain>.``
    one)."""
    type_name = entry.get("type", "")
    category = entry.get("category")
    if category and type_name.startswith("hydrogen."):
        head, _, cls = type_name.rpartition(".")
        return f"{head}.{category}.{cls}"
    return type_name

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
        self._update_tooltip()

        self._label = QtWidgets.QGraphicsSimpleTextItem(name, self)
        lf = self._label.font()
        lf.setPointSize(7)
        self._label.setFont(lf)
        self._label.setBrush(QtGui.QColor(theme.current().port_label))
        self._reposition_label()

    def restyle(self):
        """Re-apply the port label colour for the active theme."""
        self._label.setBrush(QtGui.QColor(theme.current().port_label))
        self.update()

    def _update_tooltip(self):
        if self.node.ports_locked:
            hint = "port position locked"
        else:
            hint = "Ctrl+drag to reposition"
        self.setToolTip(f"{self.pname}  ·  {self.kind}  ({hint})")

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
        elif self.node._icon_only and (abs(p.x() - rect.right()) < eps or self.side == 1):
            self._label.setPos(10, -br.height() / 2)  # right edge -> outside right
        elif self.node._icon_only:
            self._label.setPos(-10 - br.width(), -br.height() / 2)  # left -> outside left
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
        c = theme.current()
        r = self.R + (2 if self._hovered else 0)
        pen = QtGui.QPen(QtGui.QColor(c.port_border_hover) if self._hovered
                         else QtGui.QColor(c.port_border))
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
                if self.node.ports_locked:
                    self.node._canvas._on_status(
                        f"Port positions are locked on '{self.node.comp_id}'")
                    event.accept()
                    return
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


class LabelTextItem(QtWidgets.QGraphicsSimpleTextItem):
    """Movable node label text; Ctrl-hover/drag moves the whole label group."""

    def __init__(self, text: str, node: "NodeItem", group: str):
        super().__init__(text, node)
        self.node = node
        self.group = group
        self.setAcceptHoverEvents(True)

    def hoverMoveEvent(self, event):
        if event.modifiers() & QtCore.Qt.ControlModifier:
            self.node._set_label_hover_group(self.group)
            self.setCursor(QtCore.Qt.SizeAllCursor)
            event.accept()
            return
        self.node._set_label_hover_group(None)
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        if self.node._label_hover_group == self.group:
            self.node._set_label_hover_group(None)
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if (event.button() == QtCore.Qt.LeftButton
                and event.modifiers() & QtCore.Qt.ControlModifier):
            self.node._begin_label_drag(self.group, event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.node._label_drag_group == self.group:
            self.node._update_label_drag(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.node._label_drag_group == self.group:
            self.node._finish_label_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)


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
    MIN_W = 48.0
    MIN_H = 36.0
    RESIZE_HIT = 10.0
    # Padding added around the rect for the bounding box, so the resize-handle
    # dots (drawn centred on the corners, extending outside the rect) are inside
    # the invalidated region -- otherwise shrinking leaves uncleared corner
    # trails (especially visible over X-forwarding).
    BOUND_MARGIN = 8.0
    DEFAULT_ROTATIONS = {
        "hydrogen.thermofluid.Tank": -90,
    }

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
        self._is_control = entry.get("domain") == "control"
        # ``icon_only`` is declared on the component class as ``UI_ICON_ONLY``
        # and surfaced through the catalog; control blocks are icon-only by
        # convention whenever they ship a symbol.
        self._icon_only = self._icon is not None and (
            bool(entry.get("icon_only")) or self._is_control
        )
        self._ports_locked = self._icon_only
        self._hovered = False
        self._hover_corner: str | None = None
        self._resize_corner: str | None = None
        self._resize_anchor_scene: QtCore.QPointF | None = None
        self._custom_size: tuple[float, float] | None = None
        self._label_offsets = {
            "title": QtCore.QPointF(0.0, 0.0),
            "params": QtCore.QPointF(0.0, 0.0),
        }
        self._label_hover_group: str | None = None
        self._label_drag_group: str | None = None
        self._label_drag_start_scene: QtCore.QPointF | None = None
        self._label_drag_start_offset: QtCore.QPointF | None = None
        self.port_items: list[PortItem] = []
        # Quasi-static vs dynamic categorisation (recomputed whenever the params
        # change): a component carrying any DifferentialVariable is *dynamic*.
        self._is_dynamic: bool = False
        self._diff_vars: list[str] = []
        # Modelling warnings the host raised for this component (build/run), each
        # a formatted message; drives the amber warning badge + its dialog.
        self._warnings: list[str] = []
        # Client-side derived variables created in the Variables window for this
        # component (persisted in the project file).
        self.derived_variables: list[dict] = []
        # Custom port placements: pname -> (x, y, side); empty = use defaults.
        self._port_layout: dict[str, tuple] = {}
        # Visual transform state (applied around the node's centre).
        self._default_rot = self.DEFAULT_ROTATIONS.get(self.type_name, 0)
        self._rot = self._default_rot  # degrees, multiples of 90
        self._mirror_x = False
        self._mirror_y = False
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self.setToolTip(
            f"{display_type_name(entry)}\n\n{entry.get('summary', '')}")

        self._title = LabelTextItem(comp_id, self, "title")
        tf = self._title.font()
        tf.setBold(True)
        tf.setPointSize(10)
        self._title.setFont(tf)
        self._title.setBrush(QtGui.QColor(theme.current().node_title))
        self._title.setPos(12, 7)
        self._title.setZValue(6)

        sub_text = display_type_name(entry) + (
            "  *" if entry.get("needs_medium") else "")
        self._sub = LabelTextItem(sub_text, self, "title")
        sf = self._sub.font()
        sf.setPointSize(8)
        self._sub.setFont(sf)
        self._sub.setBrush(QtGui.QColor(theme.current().node_sub))
        self._sub.setPos(12, 21)
        self._sub.setZValue(6)

        # Each ui_label parameter renders as two right-aligned columns: a name
        # item and a value item, for a compact technical/tabular look.
        self._param_labels: list[tuple[dict, LabelTextItem, LabelTextItem]] = []
        for field in entry.get("parameters", []):
            if not field.get("ui_label"):
                continue
            name_item = LabelTextItem("", self, "params")
            value_item = LabelTextItem("", self, "params")
            for item, color in ((name_item, theme.current().node_name),
                                (value_item, theme.current().node_value)):
                fnt = item.font()
                fnt.setPointSize(8)
                if item is value_item:
                    fnt.setBold(True)
                item.setFont(fnt)
                item.setBrush(QtGui.QColor(color))
                item.setZValue(6)
            self._param_labels.append((field, name_item, value_item))
        self._name_visible = True
        self._type_visible = True
        self._params_visible = True
        self._port_labels_visible = True
        self.refresh_param_labels()

        self.rebuild_ports()

    @property
    def ports_locked(self) -> bool:
        return self._ports_locked

    def set_ports_locked(self, locked: bool):
        self._ports_locked = bool(locked)
        for port in self.port_items:
            port._update_tooltip()

    # --- ports -------------------------------------------------------------- #
    def rebuild_ports(self, specs: list[tuple[str, str]] | None = None):
        self.refresh_param_labels()
        scene = self.scene()
        for pi in self.port_items:
            for conn in list(pi.connections):
                conn.remove()
            if scene is not None:
                scene.removeItem(pi)      # takes its label child with it
            else:
                pi.setParentItem(None)    # not in a scene yet -> detach from node
        self.port_items = []

        if specs is None:
            # One build gives both the ports and the dynamic/quasi-static
            # categorisation, so node creation / load / param edits don't pay
            # for two introspection builds.
            _ok, specs, is_dyn, diffs = introspect(self.type_name, self.medium, self.params)
            self._set_dynamics(is_dyn, diffs)
        left = [s for s in specs if not port_on_right(s[0])]
        right = [s for s in specs if port_on_right(s[0])]

        rows = max(len(left), len(right), 1)
        if self._icon_only:
            width = 150.0
            height = 78.0
        else:
            width = max(190.0, self._title.boundingRect().width() + 28)
            height = self.HEADER_H + rows * self.ROW_H + 6
            if self._icon is not None:           # give the symbol room to breathe
                height = max(height, self.HEADER_H + self.ICON_BODY_H)
        if self._custom_size is not None:
            width = max(width, self._custom_size[0], self.MIN_W)
            height = max(height, self._custom_size[1], self.MIN_H)
        self.setRect(0, 0, width, height)
        if self._icon_only:
            self._place_icon_ports(specs)
        else:
            self._place_side(left, x=0.0, align_left=True)
            self._place_side(right, x=width, align_left=False)
        # Freshly built ports default to a visible label; honour the toggle.
        if not self._port_labels_visible:
            for port in self.port_items:
                port._label.setVisible(False)
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

    def _place_icon_ports(self, specs):
        if self._is_control:
            self._place_control_icon_ports(specs)
            return

        rect = self.rect()
        extras_top = []
        extras_bottom = []
        for name, kind in specs:
            if name in ("inlet", "p_set", "m_set"):
                pt = QtCore.QPointF(rect.left(), rect.center().y())
                side = -1
            elif name == "outlet":
                pt = QtCore.QPointF(rect.right(), rect.center().y())
                side = 1
            elif name == "opening" or kind == "thermal":
                extras_top.append((name, kind))
                continue
            else:
                extras_bottom.append((name, kind))
                continue
            if name in self._port_layout:
                x, y, side = self._port_layout[name]
                pt = QtCore.QPointF(x, y)
            port = PortItem(self, name, kind, pt, side)
            self.port_items.append(port)

        for i, (name, kind) in enumerate(extras_top):
            x = rect.left() + rect.width() * (i + 1) / (len(extras_top) + 1)
            pt = QtCore.QPointF(x, rect.top())
            side = 1
            if name in self._port_layout:
                x, y, side = self._port_layout[name]
                pt = QtCore.QPointF(x, y)
            port = PortItem(self, name, kind, pt, side)
            self.port_items.append(port)

        for i, (name, kind) in enumerate(extras_bottom):
            x = rect.left() + rect.width() * (i + 1) / (len(extras_bottom) + 1)
            pt = QtCore.QPointF(x, rect.bottom())
            side = 1
            if name in self._port_layout:
                x, y, side = self._port_layout[name]
                pt = QtCore.QPointF(x, y)
            port = PortItem(self, name, kind, pt, side)
            self.port_items.append(port)

    def _place_control_icon_ports(self, specs):
        rect = self.rect()
        left = [s for s in specs if not port_on_right(s[0])]
        right = [s for s in specs if port_on_right(s[0])]

        for i, (name, kind) in enumerate(left):
            y = rect.top() + rect.height() * (i + 1) / (len(left) + 1)
            pt = QtCore.QPointF(rect.left(), y)
            side = -1
            if name in self._port_layout:
                x, y, side = self._port_layout[name]
                pt = QtCore.QPointF(x, y)
            self.port_items.append(PortItem(self, name, kind, pt, side))

        for i, (name, kind) in enumerate(right):
            y = rect.top() + rect.height() * (i + 1) / (len(right) + 1)
            pt = QtCore.QPointF(rect.right(), y)
            side = 1
            if name in self._port_layout:
                x, y, side = self._port_layout[name]
                pt = QtCore.QPointF(x, y)
            self.port_items.append(PortItem(self, name, kind, pt, side))

    def store_port_layout(self, port: "PortItem"):
        self._port_layout[port.pname] = (port.pos().x(), port.pos().y(), port.side)

    def set_custom_size(self, width: float, height: float):
        self._custom_size = (max(self.MIN_W, float(width)),
                             max(self.MIN_H, float(height)))

    def sync_ports(self) -> bool:
        """Re-derive ports after a parameter edit; rebuild only if the port set
        actually changed (so ordinary edits keep existing wires).  The
        dynamic/quasi-static badge is refreshed either way -- a param can flip
        a component dynamic (e.g. a wall's `dynamic` toggle) without changing
        its ports."""
        self.refresh_param_labels()
        ok, new, is_dyn, diffs = introspect(self.type_name, self.medium, self.params)
        if not ok:
            # The new params yield a component the model refuses to build (e.g. a
            # Pipe stripped of every wall layer).  Keep the existing ports + wires
            # and the previous dynamic badge rather than wiping them -- the edit
            # is simply not valid yet.
            return False
        self._set_dynamics(is_dyn, diffs)
        if {(p.pname, p.kind) for p in self.port_items} == set(new):
            return False
        self.rebuild_ports(new)
        return True

    # --- dynamic / quasi-static categorisation ----------------------------- #
    @property
    def is_dynamic(self) -> bool:
        """Whether this component carries any differential state."""
        return self._is_dynamic

    def _set_dynamics(self, is_dynamic: bool, diff_vars: list[str]):
        """Record the dynamic categorisation, refresh the tooltip and repaint
        the badge."""
        self._is_dynamic = bool(is_dynamic)
        self._diff_vars = list(diff_vars)
        self._update_dynamic_tooltip()
        self.update()

    # --- modelling warnings ------------------------------------------------ #
    @property
    def has_warnings(self) -> bool:
        return bool(self._warnings)

    def warning_messages(self) -> list[str]:
        return list(self._warnings)

    def set_warnings(self, messages: list[str]):
        """Record the host's modelling warnings for this component; refresh the
        badge + tooltip.  Empty clears them."""
        new = list(messages or [])
        if new == self._warnings:
            return
        self._warnings = new
        self._update_dynamic_tooltip()
        self.update()

    def _update_dynamic_tooltip(self):
        base = (f"{display_type_name(self.entry)}\n\n"
                f"{self.entry.get('summary', '')}").rstrip()
        if self._is_dynamic:
            n = len(self._diff_vars)
            shown = self._diff_vars[:12]
            lines = "\n".join(f"  \u2022 {d}" for d in shown)
            more = "" if n <= len(shown) else f"\n  \u2026 (+{n - len(shown)} more)"
            base += (f"\n\n\u25cf Dynamic \u2014 {n} differential "
                     f"state{'s' if n != 1 else ''}:\n{lines}{more}")
        else:
            base += "\n\n\u25cb Quasi-static (no differential states)"
        if self._warnings:
            n = len(self._warnings)
            shown = self._warnings[:6]
            lines = "\n".join(f"  \u2022 {w}" for w in shown)
            more = "" if n <= len(shown) else f"\n  \u2026 (+{n - len(shown)} more)"
            base += (f"\n\n\u26a0 {n} warning{'s' if n != 1 else ''}:"
                     f"\n{lines}{more}")
        self.setToolTip(base)

    def set_name_visible(self, visible: bool):
        self._name_visible = visible
        self._title.setVisible(visible)

    def set_type_visible(self, visible: bool):
        self._type_visible = visible
        self._sub.setVisible(visible)

    def set_param_labels_visible(self, visible: bool):
        self._params_visible = visible
        for _field, name_item, value_item in self._param_labels:
            name_item.setVisible(visible)
            value_item.setVisible(visible)

    def set_port_labels_visible(self, visible: bool):
        self._port_labels_visible = visible
        for port in self.port_items:
            port._label.setVisible(visible)

    def restyle(self):
        """Re-apply every themed brush after a theme change and repaint."""
        self._sync_label_hover_brushes()      # title / sub / param label colours
        for port in self.port_items:
            port.restyle()
        self.update()

    def refresh_param_labels(self):
        params = self.params or {}
        for field, name_item, value_item in self._param_labels:
            name = field["name"]
            value = params.get(name, field.get("default"))
            if value is None:
                value_text = "?"
            elif isinstance(value, float):
                value_text = f"{value:g}"
            else:
                value_text = f"{value}"
            unit = field.get("unit")
            if unit:
                value_text += f" {unit}"
            name_item.setText(f"{name} =")
            value_item.setText(value_text)

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
        for _field, name_item, value_item in self._param_labels:
            texts.append(name_item)
            texts.append(value_item)
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
        self._place_param_labels()
        self._sync_label_hover_brushes()

    def _place_title_text(self):
        """Pin the instance/type label group to the node's visual top edge.

        The node itself rotates around its centre, so the original local header
        may end up at the bottom or side.  Work in scene coordinates here: place
        the two title lines at the top-left of the transformed node bounds, then
        map those points back into the node's local coordinates.
        """
        bbox = self.sceneTransform().mapRect(self.rect())
        offset = self._label_offsets["title"]
        gap = 4.0
        title_h = self._title.boundingRect().height()
        sub_h = self._sub.boundingRect().height()
        top = bbox.top() - gap - (title_h + sub_h)
        title_at = QtCore.QPointF(bbox.left() + 12.0, top) + offset
        sub_at = QtCore.QPointF(bbox.left() + 12.0, top + title_h) + offset
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

    def _place_param_labels(self):
        if not self._param_labels:
            return
        bbox = self.sceneTransform().mapRect(self.rect())
        gap = 4.0          # below the node
        line_gap = 2.0     # between rows
        col_gap = 6.0      # between the name and value columns
        offset = self._label_offsets["params"]

        name_w = max(n.boundingRect().width() for _f, n, _v in self._param_labels)
        val_w = max(v.boundingRect().width() for _f, _n, v in self._param_labels)
        total_w = name_w + col_gap + val_w
        left = bbox.center().x() - total_w / 2.0 + offset.x()
        name_right = left + name_w        # names right-aligned to here
        val_left = name_right + col_gap   # values left-aligned from here

        y = bbox.bottom() + gap + offset.y()
        for _field, name_item, value_item in self._param_labels:
            nbr = name_item.boundingRect()
            vbr = value_item.boundingRect()
            row_h = max(nbr.height(), vbr.height())
            self._set_text_scene_origin(
                name_item, QtCore.QPointF(name_right - nbr.width(), y))
            self._set_text_scene_origin(
                value_item, QtCore.QPointF(val_left, y))
            y += row_h + line_gap

    def _label_items(self, group: str):
        if group == "title":
            return [self._title, self._sub]
        if group == "params":
            items = []
            for _field, name_item, value_item in self._param_labels:
                items.append(name_item)
                items.append(value_item)
            return items
        return []

    def _label_group_scene_rect(self, group: str) -> QtCore.QRectF | None:
        rect: QtCore.QRectF | None = None
        for item in self._label_items(group):
            if not item.isVisible():
                continue
            item_rect = item.sceneBoundingRect()
            rect = item_rect if rect is None else rect.united(item_rect)
        return rect

    def _label_group_at(self, scene_pos: QtCore.QPointF) -> str | None:
        for group in ("title", "params"):
            rect = self._label_group_scene_rect(group)
            if rect is not None and rect.adjusted(-4, -4, 4, 4).contains(scene_pos):
                return group
        return None

    def _set_label_offset(self, group: str, offset: QtCore.QPointF):
        self._label_offsets[group] = offset
        self._keep_text_upright()
        self.update()

    def _set_label_hover_group(self, group: str | None):
        self._label_hover_group = group
        self._sync_label_hover_brushes()
        self.update()

    def _begin_label_drag(self, group: str, scene_pos: QtCore.QPointF):
        self._label_drag_group = group
        self._label_drag_start_scene = scene_pos
        self._label_drag_start_offset = QtCore.QPointF(self._label_offsets[group])
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setCursor(QtCore.Qt.SizeAllCursor)
        if self._canvas is not None:
            self._canvas.begin_live_update()
        self._sync_label_hover_brushes()

    def _update_label_drag(self, scene_pos: QtCore.QPointF):
        if self._label_drag_group is None:
            return
        delta = scene_pos - self._label_drag_start_scene
        self._set_label_offset(
            self._label_drag_group,
            self._label_drag_start_offset + delta,
        )

    def _finish_label_drag(self):
        if self._label_drag_group is None:
            return
        group = self._label_drag_group
        self._label_drag_group = None
        self._label_drag_start_scene = None
        self._label_drag_start_offset = None
        self._label_hover_group = None
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setCursor(QtCore.Qt.ArrowCursor)
        if self._canvas is not None:
            self._canvas.end_live_update()
        self._sync_label_hover_brushes()
        self._canvas._on_status(f"Moved {group} label on '{self.comp_id}'")

    def _sync_label_hover_brushes(self):
        c = theme.current()
        primary = {self._title}
        muted = {self._sub}
        for _field, name_item, value_item in self._param_labels:
            muted.add(name_item)
            primary.add(value_item)
        for group in ("title", "params"):
            active = group == self._label_hover_group or group == self._label_drag_group
            for item in self._label_items(group):
                if active:
                    item.setBrush(QtGui.QColor(c.node_label_active))
                else:
                    item.setBrush(QtGui.QColor(
                        c.node_value if item in primary else c.node_sub))

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
        if self._icon_only:
            body = bbox.adjusted(4.0, 4.0, -4.0, -4.0)
        else:
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

    #: Source symbols whose control signal enters the reservoir circle on the
    #: left: ``{type_name: control-port name}``.  Both icons share the same
    #: viewBox (0 0 120 60) with a circle at x=40, r=20 -> left tangent x=20.
    _CONTROL_GUIDE_SOURCES = {
        "hydrogen.thermofluid.PressureSource": "p_set",
        "hydrogen.thermofluid.MassSource": "m_set",
    }

    def _draw_source_control_guide(self, painter, icon_rect: QtCore.QRectF):
        port_name = self._CONTROL_GUIDE_SOURCES.get(self.type_name)
        if port_name is None:
            return
        port = next((p for p in self.port_items if p.pname == port_name), None)
        if port is None:
            return
        pen = QtGui.QPen(kind_color("signal_real"), 1.6)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        # The icon uses viewBox 0 0 120 60 and a circle at x=40, r=20; the left
        # tangent is x=20, on the vertical centreline.
        target = QtCore.QPointF(
            icon_rect.left() + icon_rect.width() * (20.0 / 120.0),
            icon_rect.center().y(),
        )
        painter.drawLine(port.pos(), target)

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
        self._rot = self._default_rot
        self._mirror_x = self._mirror_y = False
        self.apply_transform()

    # --- geometry / hover --------------------------------------------------- #
    def _resize_handles(self) -> dict[str, QtCore.QPointF]:
        r = self.rect()
        return {
            "tl": r.topLeft(),
            "tr": r.topRight(),
            "bl": r.bottomLeft(),
            "br": r.bottomRight(),
        }

    def _resize_corner_at(self, pos: QtCore.QPointF) -> str | None:
        for name, corner in self._resize_handles().items():
            if (abs(pos.x() - corner.x()) <= self.RESIZE_HIT
                    and abs(pos.y() - corner.y()) <= self.RESIZE_HIT):
                return name
        return None

    def _resize_cursor(self, corner: str):
        if corner in ("tl", "br"):
            return QtCore.Qt.SizeFDiagCursor
        return QtCore.Qt.SizeBDiagCursor

    def _resize_to(self, width: float, height: float,
                   fixed_scene: QtCore.QPointF | None = None,
                   fixed_local: QtCore.QPointF | None = None):
        old = self.rect()
        # Scene region the node occupied before resizing -- including the resize
        # handles (boundingRect margin) AND every child (ports + their labels,
        # which stick out past the body).  Forced-updated below so a minimal /
        # X-forwarded viewport repaint can't leave a stale port or corner ghost.
        old_scene_rect = self.sceneTransform().mapRect(
            self.boundingRect().united(self.childrenBoundingRect()))
        width = max(self.MIN_W, width)
        height = max(self.MIN_H, height)
        sx = width / old.width() if old.width() else 1.0
        sy = height / old.height() if old.height() else 1.0
        self._custom_size = (width, height)
        self.setRect(0, 0, width, height)

        # Keep ports on their same relative spot after a resize. Defaults such
        # as inlet/outlet therefore remain on the icon stubs.
        for port in self.port_items:
            port.setPos(QtCore.QPointF(port.pos().x() * sx, port.pos().y() * sy))
            port._reposition_label()
            if port.pname in self._port_layout:
                self._port_layout[port.pname] = (
                    port.pos().x(), port.pos().y(), port.side)

        self.apply_transform()
        if fixed_scene is not None and fixed_local is not None:
            transformed_fixed = self.transform().map(fixed_local)
            self.setPos(fixed_scene - transformed_fixed)
        self._keep_text_upright()
        self.update()
        # Repaint the union of the old and new footprints so no corner pixels
        # are left behind when the node gets smaller.
        scene = self.scene()
        if scene is not None:
            new_scene_rect = self.sceneTransform().mapRect(
                self.boundingRect().united(self.childrenBoundingRect()))
            scene.update(old_scene_rect.united(new_scene_rect))

    def itemChange(self, change, value):
        if (change == QtWidgets.QGraphicsItem.ItemPositionHasChanged
                and self._canvas is not None):
            self._canvas.refresh_connections(self)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverMoveEvent(self, event):
        ctrl = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        self._label_hover_group = self._label_group_at(event.scenePos()) if ctrl else None
        if self._label_hover_group is not None:
            self.setCursor(QtCore.Qt.SizeAllCursor)
            self._hover_corner = None
            self._set_label_hover_group(self._label_hover_group)
            self.update()
            return

        self._hover_corner = self._resize_corner_at(event.pos())
        if self._hover_corner is not None:
            self.setCursor(self._resize_cursor(self._hover_corner))
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)
        self._set_label_hover_group(None)
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._hover_corner = None
        self._set_label_hover_group(None)
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                group = self._label_group_at(event.scenePos())
                if group is not None:
                    self._begin_label_drag(group, event.scenePos())
                    event.accept()
                    return
            corner = self._resize_corner_at(event.pos())
            if corner is not None:
                self._resize_corner = corner
                fixed = {
                    "tl": self.rect().bottomRight(),
                    "tr": self.rect().bottomLeft(),
                    "bl": self.rect().topRight(),
                    "br": self.rect().topLeft(),
                }[corner]
                self._resize_anchor_scene = self.mapToScene(fixed)
                self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
                if self._canvas is not None:
                    self._canvas.begin_live_update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._label_drag_group is not None:
            self._update_label_drag(event.scenePos())
            event.accept()
            return
        if self._resize_corner is not None:
            r = self.rect()
            p = event.pos()
            if self._resize_corner == "br":
                width, height = p.x(), p.y()
                fixed_local = QtCore.QPointF(0.0, 0.0)
            elif self._resize_corner == "tr":
                width, height = p.x(), r.bottom() - p.y()
                fixed_local = QtCore.QPointF(0.0, max(self.MIN_H, height))
            elif self._resize_corner == "bl":
                width, height = r.right() - p.x(), p.y()
                fixed_local = QtCore.QPointF(max(self.MIN_W, width), 0.0)
            else:  # tl
                width, height = r.right() - p.x(), r.bottom() - p.y()
                fixed_local = QtCore.QPointF(max(self.MIN_W, width),
                                             max(self.MIN_H, height))
            self._resize_to(width, height, self._resize_anchor_scene, fixed_local)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._label_drag_group is not None:
            self._finish_label_drag()
            event.accept()
            return
        if self._resize_corner is not None:
            self._resize_corner = None
            self._resize_anchor_scene = None
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
            if self._canvas is not None:
                self._canvas.end_live_update()
            self._canvas.refresh_connections(self)
            self._canvas._on_status(
                f"Resized '{self.comp_id}' to "
                f"{self.rect().width():.0f}×{self.rect().height():.0f}")
            event.accept()
            return
        super().mouseReleaseEvent(event)

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

    def _draw_dynamic_badge(self, painter, rect):
        """A tiny teal dot in the top-right corner marking a *dynamic*
        component (one carrying differential state); quasi-static nodes get no
        mark.  A bare dot is orientation-agnostic, so it reads correctly even
        on rotated/mirrored nodes.  Hover the node for the state list."""
        if not self._is_dynamic:
            return
        c = theme.current()
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        center = QtCore.QPointF(rect.right() - 11.0, rect.top() + 11.0)
        painter.setPen(QtGui.QPen(QtGui.QColor(c.badge_outline), 1.4))
        painter.setBrush(QtGui.QColor(c.badge_dynamic))
        painter.drawEllipse(center, 4.5, 4.5)
        painter.restore()

    def _draw_warning_badge(self, painter, rect):
        """An amber warning triangle with a ``!`` in the top-left corner when
        the host raised a modelling warning for this component.  Right-click ->
        'Show warnings…' (or hover) for the messages."""
        if not self._warnings:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        cx, cy, r = rect.left() + 12.0, rect.top() + 11.0, 8.0
        tri = QtGui.QPolygonF([
            QtCore.QPointF(cx, cy - r),
            QtCore.QPointF(cx - r * 0.92, cy + r * 0.72),
            QtCore.QPointF(cx + r * 0.92, cy + r * 0.72),
        ])
        c = theme.current()
        painter.setPen(QtGui.QPen(QtGui.QColor(c.warn_border), 1.2))
        painter.setBrush(QtGui.QColor(c.warn_fill))
        painter.drawPolygon(tri)
        # Exclamation mark.
        painter.setPen(QtGui.QPen(QtGui.QColor(c.warn_mark), 1.5))
        painter.drawLine(QtCore.QPointF(cx, cy - r * 0.30),
                         QtCore.QPointF(cx, cy + r * 0.28))
        painter.setBrush(QtGui.QColor(c.warn_mark))
        painter.drawEllipse(QtCore.QPointF(cx, cy + r * 0.52), 0.9, 0.9)
        painter.restore()

    def _draw_resize_handles(self, painter):
        if self._hover_corner is None and self._resize_corner is None:
            return
        c = theme.current()
        painter.save()
        for name, corner in self._resize_handles().items():
            active = name in (self._hover_corner, self._resize_corner)
            painter.setPen(QtGui.QPen(
                QtGui.QColor(c.resize_handle_active if active else c.resize_handle),
                1.2))
            painter.setBrush(QtGui.QColor(c.resize_handle_fill))
            painter.drawEllipse(corner, 3.5 if active else 3.0, 3.5 if active else 3.0)
        painter.restore()

    def boundingRect(self) -> QtCore.QRectF:
        # Expand past the rect so the corner resize handles (and the selection
        # outline) are inside the region Qt repaints/clears; without this the
        # handles leave trails when the node is resized smaller.
        m = self.BOUND_MARGIN
        return self.rect().adjusted(-m, -m, m, m)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        c = theme.current()
        selected = self.isSelected()
        if selected:
            color, w = QtGui.QColor(c.node_border_selected), 2
        elif self._hovered:
            color, w = QtGui.QColor(c.node_border_hover), 2
        else:
            color, w = QtGui.QColor(c.node_border), 1
        rect = self.rect()

        if self._icon is not None:
            if self._icon_only:
                if self._is_control or selected or self._hovered:
                    border = color if selected else QtGui.QColor(c.node_icon_border)
                    painter.setPen(QtGui.QPen(border, w))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 8, 8)
                body = self._visual_icon_rect(self._icon.defaultSize())
                if body is not None:
                    self._icon.render(painter, body)
                    self._draw_source_control_guide(painter, body)
                self._draw_dynamic_badge(painter, rect)
                self._draw_warning_badge(painter, rect)
                self._draw_resize_handles(painter)
                return

            # P&ID symbol on a clean card: a domain-coloured header strip
            # carries the labels, the SVG fills the body below the separator.
            painter.setPen(QtGui.QPen(color, w))
            painter.setBrush(QtGui.QColor(c.node_card) if not self._hovered
                             else QtGui.QColor(c.node_card_hover))
            painter.drawRoundedRect(rect, 9, 9)
            painter.setPen(QtGui.QPen(self._fill.darker(135), 1.5))
            header_line = self._visual_header_line()
            if header_line is not None:
                painter.drawLine(*header_line)
            body = self._visual_icon_rect(self._icon.defaultSize())
            if body is not None:
                self._icon.render(painter, body)
            self._draw_dynamic_badge(painter, rect)
            self._draw_warning_badge(painter, rect)
            self._draw_resize_handles(painter)
            return

        painter.setPen(QtGui.QPen(color, w))
        painter.setBrush(self._fill.lighter(106) if self._hovered else self._fill)
        painter.drawRoundedRect(rect, 9, 9)
        self._draw_dynamic_badge(painter, rect)
        self._draw_warning_badge(painter, rect)
        self._draw_resize_handles(painter)


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
