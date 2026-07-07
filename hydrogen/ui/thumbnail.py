"""Render a small preview ("thumbnail") of a saved project's canvas.

Instead of re-implementing the drawing, this builds a real (off-screen)
:class:`~hydrogen.ui.canvas.Canvas` from the persisted layout and renders its
scene scaled down -- so a thumbnail is pixel-for-pixel the same factory the
editor uses (same symbols, ports, wire splines, transforms), just smaller and
with the text labels hidden. The grid is drawn at the view level, so the
rendered scene shows symbols + wires on a clean background.
"""

from __future__ import annotations

from .qt import QtCore, QtGui

__all__ = ["render_canvas_thumbnail"]

_PAD = 8.0           # inner padding inside the pixmap, in device px


def render_canvas_thumbnail(canvas_state: dict, size: QtCore.QSize,
                            by_type: dict | None = None,
                            bg: str = "#ffffff") -> QtGui.QPixmap:
    """A ``size`` pixmap previewing ``canvas_state`` (``{"nodes", "connections"}``)
    as a scaled-down, text-free copy of the editor canvas. An empty layout (or
    one whose component types aren't in ``by_type``) yields a blank pixmap."""
    pm = QtGui.QPixmap(size)
    pm.fill(QtGui.QColor(bg))

    nodes = canvas_state.get("nodes", []) if isinstance(canvas_state, dict) else []
    if not nodes:
        return pm

    # Local import: keeps this module light and avoids any import cycle.
    from .canvas import Canvas

    canvas = Canvas(by_type or {}, lambda *_: None)
    try:
        # Hide every text marker *before* loading so nodes are built label-free;
        # we want the symbols/wires only.
        canvas._show_names = False
        canvas._show_types = False
        canvas._show_params = False
        canvas._show_port_names = False
        canvas.load_project(canvas_state)

        scene = canvas._scene
        src = scene.itemsBoundingRect()
        if src.isEmpty():
            return pm
        margin = max(src.width(), src.height()) * 0.04 + 6.0
        src = src.adjusted(-margin, -margin, margin, margin)

        target = QtCore.QRectF(_PAD, _PAD,
                               size.width() - 2 * _PAD, size.height() - 2 * _PAD)
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        scene.render(painter, target, src, QtCore.Qt.KeepAspectRatio)
        painter.end()
    finally:
        # Plot/table objects keep their content widget off-screen (a top-level
        # widget); clear them so this throwaway canvas doesn't leak them (they
        # would otherwise linger as "open windows" and keep the app running).
        canvas.clear_nodes()
        canvas.deleteLater()
    return pm
