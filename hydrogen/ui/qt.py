"""Qt binding shim + tiny cross-binding event helpers.

Every other module in :mod:`hydrogen.ui` imports Qt *through here*, so the rest
of the package never has to care whether PySide6 (preferred) or PyQt5 is the
installed binding.  Two small helpers paper over the remaining API differences
between the two (dialog ``exec`` and drag/drop event coordinates).
"""

from __future__ import annotations

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    try:                                    # QtSvg is a separate module/wheel
        from PySide6 import QtSvg
    except ImportError:
        QtSvg = None  # type: ignore
except ImportError:  # fall back to PyQt5
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    try:
        from PyQt5 import QtSvg  # type: ignore
    except ImportError:
        QtSvg = None  # type: ignore

#: ``Signal`` is ``Signal`` on PySide6 and ``pyqtSignal`` on PyQt5.
Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal

__all__ = ["QtCore", "QtGui", "QtWidgets", "QtSvg", "Signal", "exec_", "drop_point"]


def exec_(widget, *args):
    """Run a modal ``exec`` across bindings (PySide6 ``exec`` / PyQt5 ``exec_``)."""
    fn = getattr(widget, "exec", None) or widget.exec_
    return fn(*args)


def drop_point(event) -> "QtCore.QPoint":
    """Viewport-local point of a drag/drop event (Qt6 ``position()`` -> QPointF,
    Qt5 ``pos()`` -> QPoint)."""
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()
