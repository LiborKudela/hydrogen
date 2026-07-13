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
    try:                                    # QtCharts too (bundled with PySide6)
        from PySide6 import QtCharts
    except ImportError:
        QtCharts = None  # type: ignore
except ImportError:  # fall back to PyQt5
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    try:
        from PyQt5 import QtSvg  # type: ignore
    except ImportError:
        QtSvg = None  # type: ignore
    try:                                    # PyQt5 names the module QtChart
        from PyQt5 import QtChart as QtCharts  # type: ignore
    except ImportError:
        QtCharts = None  # type: ignore

#: ``Signal`` is ``Signal`` on PySide6 and ``pyqtSignal`` on PyQt5.
Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal

__all__ = ["QtCore", "QtGui", "QtWidgets", "QtSvg", "QtCharts", "Signal",
           "exec_", "drop_point", "install_wheel_guard"]


class _WheelGuard(QtCore.QObject):
    """App-wide event filter that stops the mouse wheel from changing the
    value of combo boxes, spin boxes and sliders.

    Scrolling *over* one of these controls used to flip its selection by
    accident while the user only meant to scroll the surrounding panel.  We
    swallow the wheel event on the control and forward it to the enclosing
    scroll area instead, so the panel keeps scrolling but the value never
    changes on a stray scroll.  The controls stay fully editable by click,
    keyboard and drop-down as usual.
    """

    _GUARDED = (
        QtWidgets.QComboBox,
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QSlider,
    )

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if event.type() == QtCore.QEvent.Type.Wheel and isinstance(obj, self._GUARDED):
            area = self._scroll_area(obj)
            if area is not None:
                QtWidgets.QApplication.sendEvent(area.viewport(), event)
            return True                 # never let the control consume the wheel
        return super().eventFilter(obj, event)

    @staticmethod
    def _scroll_area(widget):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractScrollArea):
                return parent
            parent = parent.parentWidget()
        return None


def install_wheel_guard(app):
    """Install the shared :class:`_WheelGuard` on ``app`` (idempotent).

    Returns the guard, which is also stashed on ``app`` so it survives garbage
    collection for the lifetime of the application.
    """
    guard = getattr(app, "_hydrogen_wheel_guard", None)
    if guard is None:
        guard = _WheelGuard(app)
        app.installEventFilter(guard)
        app._hydrogen_wheel_guard = guard
    return guard


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
