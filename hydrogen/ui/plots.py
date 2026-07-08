"""Canvas *plot objects* -- the Table and Timeseries viewers that float over the
canvas next to the components, consume dragged variables, and update live off
the simulation stream.

Three layers:

* :class:`PlotItem` -- a movable / resizable titled ``QGraphicsItem`` that hosts
  one *content* widget on the canvas.  The view transform scales it (frame,
  header, and text) with zoom, just like the placed components.  The content
  widget is **not** embedded in the scene (``QGraphicsProxyWidget`` / a
  ``QChart`` graphics item blank out after interactions on some platforms);
  instead it is kept off-screen and rasterised to a pixmap the item paints,
  re-rendered at the current zoom so text stays crisp.
* :class:`VariableTable` -- a droppable, editable, reorderable table whose rows
  auto-fill unit + description from the dragged variable and whose value column
  updates live.
* :class:`TimeseriesChart` -- a ``QtCharts`` line chart; each dragged variable
  becomes a trace with a fully editable pen (colour / width / dash / legend
  name), plus chart-level title / legend / font settings.
* :class:`BarChart` -- a bar chart of each variable's *latest* value (live
  snapshot comparison).
* :class:`PieChart` -- a pie chart of the latest values (proportional slices
  with optional percentage labels on each slice).

Content widgets accept variable drops directly (normal widget drag-and-drop) and
announce changes via a ``changed`` signal; live values arrive via
:meth:`set_live_source` + :meth:`refresh_live` driven by
:class:`~hydrogen.ui.live.LiveController`.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .qt import QtCharts, QtCore, QtGui, QtWidgets, Signal, exec_

__all__ = [
    "VARIABLE_MIME", "TIME_KEY", "STEP_KEY", "PROGRESS_KEY", "STEPSIZE_KEY",
    "WALLTIME_KEY", "SPECIAL_KEYS", "encode_variables", "decode_variables",
    "PlotItem", "VariableTable", "TimeseriesChart", "BarChart", "PieChart",
    "make_content", "content_from_dict",
]

#: Drag payload for one or more variables dragged out of the Variables window.
#: A JSON list of ``{full, label, name, unit, description, kind}`` dicts.
VARIABLE_MIME = "application/x-hydrogen-variable"

#: Reserved ``full`` names for a table's *run-status* rows (simulation time,
#: solver step index, run progress).  These are not recorded variables, so they
#: are excluded from :meth:`VariableTable.variable_names` (the live pump never
#: registers them as stream series) and are resolved by the live source's
#: ``latest`` to the current run status.  Tables only -- a chart already has
#: time on its X axis.
TIME_KEY = "__sim_time__"
STEP_KEY = "__sim_step__"
PROGRESS_KEY = "__sim_progress__"
STEPSIZE_KEY = "__sim_dt__"
WALLTIME_KEY = "__run_wall__"

#: All reserved run-status keys (never watched as stream series / dragged out).
SPECIAL_KEYS = frozenset(
    {TIME_KEY, STEP_KEY, PROGRESS_KEY, STEPSIZE_KEY, WALLTIME_KEY})

#: Pen dash styles offered in the trace editor (label -> Qt.PenStyle name).
_DASH_STYLES = ["solid", "dash", "dot", "dashdot", "dashdotdot"]
_DASH_TO_QT = {
    "solid": QtCore.Qt.SolidLine,
    "dash": QtCore.Qt.DashLine,
    "dot": QtCore.Qt.DotLine,
    "dashdot": QtCore.Qt.DashDotLine,
    "dashdotdot": QtCore.Qt.DashDotDotLine,
}

#: A small palette cycled through for fresh traces.
_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


def encode_variables(payloads: list[dict]) -> QtCore.QMimeData:
    md = QtCore.QMimeData()
    md.setData(VARIABLE_MIME,
               QtCore.QByteArray(json.dumps(payloads).encode("utf-8")))
    md.setText(", ".join(p.get("label", p.get("full", "")) for p in payloads))
    return md


def decode_variables(md: QtCore.QMimeData) -> list[dict]:
    if not md.hasFormat(VARIABLE_MIME):
        return []
    try:
        data = json.loads(bytes(md.data(VARIABLE_MIME)).decode("utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _exec_beside(dlg: "QtWidgets.QDialog", anchor: "QtCore.QPoint | None"):
    """Show ``dlg`` at ``anchor`` (a global point at the plot's right edge),
    clamped to the screen, then run it modally."""
    if anchor is not None:
        dlg.adjustSize()
        pos = QtCore.QPoint(anchor)
        screen = QtWidgets.QApplication.primaryScreen()
        try:                                # keep it fully on-screen
            avail = (dlg.screen() or screen).availableGeometry()
        except Exception:
            avail = screen.availableGeometry() if screen else None
        if avail is not None:
            pos.setX(min(pos.x(), avail.right() - dlg.width()))
            pos.setX(max(pos.x(), avail.left()))
            pos.setY(min(pos.y(), avail.bottom() - dlg.height()))
            pos.setY(max(pos.y(), avail.top()))
        dlg.move(pos)
    return exec_(dlg)


# --------------------------------------------------------------------------- #
# Content: the variable table.
# --------------------------------------------------------------------------- #
class _RowDragTable(QtWidgets.QTableWidget):
    """Table whose selected rows are a variable drag source.

    Dragging rows out emits the same :data:`VARIABLE_MIME` payload the Variables
    window produces, so a row can be dropped onto a Timeseries (or another
    Table) object exactly like a variable dragged from the tree.
    """

    def __init__(self, payload_for_row, rows, cols):
        super().__init__(rows, cols)
        self._payload_for_row = payload_for_row
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)

    def mimeData(self, items):  # noqa: N802 (Qt override)
        rows = sorted({it.row() for it in items})
        payloads = [p for p in (self._payload_for_row(r) for r in rows) if p]
        return encode_variables(payloads)


class VariableTable(QtWidgets.QWidget):
    """A live table of variables: name / value / unit / description.

    Rows are added by dropping variables (unit + description auto-filled), can be
    reordered (drag the row, or the Up / Down buttons), edited in place, and
    removed.  The value column updates from the live source (or shows the
    static introspected value when idle).
    """

    KIND = "table"
    DEFAULT_TITLE = "Variables"

    _COLS = ["Variable", "Value", "Unit", "Description"]

    #: Emitted whenever the set of variables changes (so the live pump re-syncs).
    changed = Signal()

    def __init__(self):
        super().__init__()
        self._title = self.DEFAULT_TITLE
        self._rows: list[dict] = []          # [{full, label, unit, description, value}]
        self._source = None                  # LiveController-provided source or None
        self._render_hook = None             # PlotItem re-render callback
        self.setAcceptDrops(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(3)
        self._toolbar_buttons: list[tuple[QtWidgets.QToolButton, object]] = []
        for text, tip, slot in (
            ("▲", "Move the selected row up", self._move_up),
            ("▼", "Move the selected row down", self._move_down),
            ("✕", "Remove the selected row", self._remove_selected),
        ):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            bar.addWidget(b)
            self._toolbar_buttons.append((b, slot))
        bar.addStretch(1)
        hint = QtWidgets.QLabel("drag rows to a plot")
        hint.setStyleSheet("color:#999; font-size:10px;")
        bar.addWidget(hint)
        lay.addLayout(bar)

        self._table = _RowDragTable(self._row_payload, 0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(self._COLS)
        self._table.verticalHeader().setSectionsMovable(True)
        self._table.verticalHeader().sectionMoved.connect(self._on_section_moved)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self._table, 1)

    def toolbar_action_at(self, pos: QtCore.QPoint):
        """Return the toolbar callback under ``pos`` (this widget's coords)."""
        for btn, slot in self._toolbar_buttons:
            if btn.geometry().contains(pos):
                return slot
        return None

    def remove_selected_row(self) -> bool:
        """Remove the currently selected row; return True if one was removed."""
        before = len(self._rows)
        self._remove_selected()
        return len(self._rows) < before

    # --- content contract -------------------------------------------------- #
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title or self.DEFAULT_TITLE
        self._notify_render()

    def consume_payload(self, payloads: list[dict]):
        added = False
        for p in payloads:
            full = p.get("full")
            if not full or any(r["full"] == full for r in self._rows):
                continue
            row = {
                "full": full,
                "label": p.get("label", full),
                "unit": p.get("unit", ""),
                "description": p.get("description", ""),
                "value": p.get("value"),
            }
            if p.get("agg"):
                row["agg"] = dict(p["agg"])
            self._rows.append(row)
            added = True
        self._rebuild()
        if added:
            self.changed.emit()

    def variable_names(self) -> list[str]:
        # Run-status rows are resolved by the live source, not watched as series.
        return [r["full"] for r in self._rows if r["full"] not in SPECIAL_KEYS]

    def derived_specs(self) -> dict[str, dict]:
        """``{derived_full: agg}`` for client-side computed rows."""
        return {r["full"]: dict(r["agg"])
                for r in self._rows
                if r.get("agg") and r["full"] not in SPECIAL_KEYS}

    def _add_special_row(self, full, label, unit, description):
        if any(r["full"] == full for r in self._rows):
            return
        self._rows.append({
            "full": full, "label": label, "unit": unit,
            "description": description, "value": None,
        })
        self._rebuild()
        self.changed.emit()

    def add_time_row(self):
        """Add a row that shows the live simulation time (once)."""
        self._add_special_row(TIME_KEY, "time", "s", "simulation time")

    def add_step_row(self):
        """Add a row that shows the live solver step index (once)."""
        self._add_special_row(STEP_KEY, "step", "", "solver step index")

    def add_progress_row(self):
        """Add a row that shows the live run progress (once)."""
        self._add_special_row(PROGRESS_KEY, "progress", "%", "run progress")

    def add_step_size_row(self):
        """Add a row that shows the live solver step size dt (once)."""
        self._add_special_row(STEPSIZE_KEY, "step size", "s",
                              "last time step size (dt)")

    def add_wall_time_row(self):
        """Add a row that shows the run's wall-clock time (once)."""
        self._add_special_row(WALLTIME_KEY, "wall time", "s",
                              "run wall-clock time")

    def set_render_hook(self, fn):
        """Register a callback invoked whenever the visible content changes, so
        the hosting scene item can re-render its pixmap."""
        self._render_hook = fn

    def _notify_render(self):
        if self._render_hook is not None:
            self._render_hook()

    # --- drop target ------------------------------------------------------- #
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(VARIABLE_MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(VARIABLE_MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):
        payloads = decode_variables(event.mimeData())
        if payloads:
            self.consume_payload(payloads)
            event.acceptProposedAction()

    def _row_payload(self, row: int) -> dict | None:
        """Variable payload for a row, used as the drag-out MIME data."""
        if not (0 <= row < len(self._rows)):
            return None
        self._commit_edits()
        r = self._rows[row]
        payload = {
            "full": r["full"],
            "label": r.get("label", r["full"]),
            "name": r.get("label", r["full"]).rsplit(".", 1)[-1],
            "unit": r.get("unit", ""),
            "description": r.get("description", ""),
            "value": r.get("value"),
        }
        if r.get("agg"):
            payload["agg"] = dict(r["agg"])
        return payload

    def drag_mime_at(self, pos: "QtCore.QPoint"):
        """MIME payload for a drag that starts at ``pos`` (this widget's own
        coordinates), or ``None`` if that point isn't on a data row.

        The widget is rendered off-screen inside a :class:`PlotItem`, so the
        item -- not the table -- starts the drag; this maps a press point to the
        row under it and hands back the same payload a live row drag would.
        """
        vp = self._table.viewport()
        p = vp.mapFrom(self, pos)
        if not vp.rect().contains(p):
            return None
        idx = self._table.indexAt(p)
        if not idx.isValid():
            return None
        row = idx.row()
        # Run-status rows live in tables only -- never drag them out to a chart.
        if 0 <= row < len(self._rows) and self._rows[row]["full"] in SPECIAL_KEYS:
            return None
        self._table.selectRow(row)
        payload = self._row_payload(row)
        if not payload:
            return None
        return encode_variables([payload])

    def set_live_source(self, source):
        self._source = source
        self.refresh_live()

    def refresh_live(self):
        for i, r in enumerate(self._rows):
            val = None
            if self._source is not None:
                val = self._source.latest(r["full"])
            if val is None:
                val = r.get("value")
            item = self._table.item(i, 1)
            if item is not None:
                item.setText(self._fmt(val))
        self._notify_render()

    # --- editing ----------------------------------------------------------- #
    def _rebuild(self):
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._rows))
        for i, r in enumerate(self._rows):
            self._set(i, 0, r["label"], editable=True)
            self._set(i, 1, self._fmt(r.get("value")), editable=False)
            self._set(i, 2, r["unit"], editable=True)
            self._set(i, 3, r["description"], editable=True)
        self._table.blockSignals(False)
        self._table.resizeRowsToContents()
        self.refresh_live()

    def _set(self, row, col, text, *, editable):
        item = QtWidgets.QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if col == 1:
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._table.setItem(row, col, item)

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return "—"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return str(val)
        if f == 0 or 1e-3 <= abs(f) < 1e5:
            return f"{f:.4g}"
        return f"{f:.4e}"

    def _selected_row(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _move_up(self):
        i = self._selected_row()
        if i > 0:
            self._rows[i - 1], self._rows[i] = self._rows[i], self._rows[i - 1]
            self._rebuild()
            self._table.selectRow(i - 1)

    def _move_down(self):
        i = self._selected_row()
        if 0 <= i < len(self._rows) - 1:
            self._rows[i + 1], self._rows[i] = self._rows[i], self._rows[i + 1]
            self._rebuild()
            self._table.selectRow(i + 1)

    def _remove_selected(self):
        i = self._selected_row()
        if 0 <= i < len(self._rows):
            del self._rows[i]
            self._rebuild()
            self.changed.emit()

    def _on_section_moved(self, _logical, old_visual, new_visual):
        # Reflect a header drag-reorder into the backing list, then normalise the
        # visual order back to identity so indices and the model stay in step.
        self._rows.insert(new_visual, self._rows.pop(old_visual))
        vh = self._table.verticalHeader()
        vh.blockSignals(True)
        for logical in range(vh.count()):
            vh.moveSection(vh.visualIndex(logical), logical)
        vh.blockSignals(False)
        self._rebuild()

    def _commit_edits(self):
        """Pull any in-place edits (label / unit / description) back into the
        backing rows before serialising."""
        for i, r in enumerate(self._rows):
            for col, key in ((0, "label"), (2, "unit"), (3, "description")):
                item = self._table.item(i, col)
                if item is not None:
                    r[key] = item.text()

    # --- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict:
        self._commit_edits()
        return {"title": self._title, "rows": self._rows}

    @classmethod
    def from_dict(cls, d: dict) -> "VariableTable":
        w = cls()
        w.set_title(d.get("title", cls.DEFAULT_TITLE))
        w._rows = [dict(r) for r in d.get("rows", [])]
        w._rebuild()
        return w

    def populate_menu(self, menu: "QtWidgets.QMenu"):
        rem = menu.addAction("Remove selected row")
        rem.triggered.connect(self._remove_selected)
        menu.addSeparator()
        present = {r["full"] for r in self._rows}
        for label, key, adder in (
            ("Add time column", TIME_KEY, self.add_time_row),
            ("Add step column", STEP_KEY, self.add_step_row),
            ("Add step size column", STEPSIZE_KEY, self.add_step_size_row),
            ("Add wall time column", WALLTIME_KEY, self.add_wall_time_row),
            ("Add progress column", PROGRESS_KEY, self.add_progress_row),
        ):
            act = menu.addAction(label)
            act.setEnabled(key not in present)
            act.triggered.connect(adder)
        clear = menu.addAction("Clear all rows")
        clear.triggered.connect(self._clear)

    def _clear(self):
        self._rows.clear()
        self._rebuild()
        self.changed.emit()


# --------------------------------------------------------------------------- #
# Content: the timeseries chart.
# --------------------------------------------------------------------------- #
class TraceStyleDialog(QtWidgets.QDialog):
    """Edit one trace's pen + legend name."""

    def __init__(self, trace: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trace style")
        self._trace = dict(trace)
        form = QtWidgets.QFormLayout(self)

        self._legend = QtWidgets.QLineEdit(trace.get("label", ""))
        form.addRow("Legend name", self._legend)

        self._color = trace.get("color", "#1f77b4")
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.clicked.connect(self._pick_color)
        self._sync_color_btn()
        form.addRow("Colour", self._color_btn)

        self._width = QtWidgets.QDoubleSpinBox()
        self._width.setRange(0.5, 12.0)
        self._width.setSingleStep(0.5)
        self._width.setValue(float(trace.get("width", 2.0)))
        form.addRow("Line width", self._width)

        self._dash = QtWidgets.QComboBox()
        self._dash.addItems(_DASH_STYLES)
        self._dash.setCurrentText(trace.get("dash", "solid"))
        form.addRow("Line style", self._dash)

        self._visible = QtWidgets.QCheckBox("Visible")
        self._visible.setChecked(bool(trace.get("visible", True)))
        form.addRow("", self._visible)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _sync_color_btn(self):
        self._color_btn.setText(self._color)
        self._color_btn.setStyleSheet(
            f"background:{self._color}; color:white; padding:4px;")

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._color), self,
                                            "Trace colour")
        if c.isValid():
            self._color = c.name()
            self._sync_color_btn()

    def result_trace(self) -> dict:
        self._trace.update({
            "label": self._legend.text(),
            "color": self._color,
            "width": self._width.value(),
            "dash": self._dash.currentText(),
            "visible": self._visible.isChecked(),
        })
        return self._trace


class ChartSettingsDialog(QtWidgets.QDialog):
    """Chart-level title / legend / font settings."""

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chart settings")
        self._s = dict(settings)
        form = QtWidgets.QFormLayout(self)

        self._title = QtWidgets.QLineEdit(settings.get("title", ""))
        form.addRow("Title", self._title)

        self._legend = QtWidgets.QCheckBox("Show legend")
        self._legend.setChecked(bool(settings.get("legend", True)))
        form.addRow("", self._legend)

        self._legend_pos = QtWidgets.QComboBox()
        self._legend_pos.addItems(["bottom", "top", "left", "right"])
        self._legend_pos.setCurrentText(settings.get("legend_pos", "bottom"))
        form.addRow("Legend position", self._legend_pos)

        self._title_size = QtWidgets.QSpinBox()
        self._title_size.setRange(6, 48)
        self._title_size.setValue(int(settings.get("title_size", 12)))
        form.addRow("Title font size", self._title_size)

        self._axis_size = QtWidgets.QSpinBox()
        self._axis_size.setRange(6, 32)
        self._axis_size.setValue(int(settings.get("axis_size", 9)))
        form.addRow("Axis font size", self._axis_size)

        self._x_title = QtWidgets.QLineEdit(settings.get("x_title", "time [s]"))
        form.addRow("X axis label", self._x_title)
        self._y_title = QtWidgets.QLineEdit(settings.get("y_title", ""))
        form.addRow("Y axis label", self._y_title)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_settings(self) -> dict:
        self._s.update({
            "title": self._title.text(),
            "legend": self._legend.isChecked(),
            "legend_pos": self._legend_pos.currentText(),
            "title_size": self._title_size.value(),
            "axis_size": self._axis_size.value(),
            "x_title": self._x_title.text(),
            "y_title": self._y_title.text(),
        })
        return self._s


class TimeseriesChart(QtWidgets.QWidget):
    """A live line chart; each dragged variable becomes an editable trace.

    A normal widget wrapping a ``QChartView`` -- rendered off-screen to a pixmap
    by :class:`PlotItem`, never embedded in the scene, so it renders reliably.
    Degrades to a small placeholder if QtCharts is unavailable.
    """

    KIND = "timeseries"
    DEFAULT_TITLE = "Timeseries"

    #: Cap on points drawn per trace; a long run is decimated to this for display.
    MAX_POINTS = 2000

    #: Emitted whenever the set of traces changes (so the live pump re-syncs).
    changed = Signal()

    _LEGEND_ALIGN = {
        "bottom": QtCore.Qt.AlignBottom, "top": QtCore.Qt.AlignTop,
        "left": QtCore.Qt.AlignLeft, "right": QtCore.Qt.AlignRight,
    }

    def __init__(self):
        super().__init__()
        self._title = self.DEFAULT_TITLE
        self._traces: list[dict] = []        # [{full, label, color, width, dash, visible}]
        self._series: dict[str, object] = {}  # full -> QLineSeries
        self._source = None
        self._render_hook = None             # PlotItem re-render callback
        # Where edit dialogs parent + pop up (set by the PlotItem).
        self._dialog_parent = None
        self._dialog_anchor = None
        self._settings = {
            "title": self._title, "legend": True, "legend_pos": "bottom",
            "title_size": 12, "axis_size": 9, "x_title": "time [s]", "y_title": "",
        }
        self.setAcceptDrops(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if QtCharts is None:                 # graceful placeholder if no charts
            self._chart = None
            self._view = None
            lay.addWidget(QtWidgets.QLabel(
                "QtCharts is not available in this Qt build."))
            return

        self._chart = QtCharts.QChart()
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(QtCore.Qt.AlignBottom)
        self._axis_x = QtCharts.QValueAxis()
        self._axis_y = QtCharts.QValueAxis()
        self._chart.addAxis(self._axis_x, QtCore.Qt.AlignBottom)
        self._chart.addAxis(self._axis_y, QtCore.Qt.AlignLeft)
        self._view = QtCharts.QChartView(self._chart)
        self._view.setRenderHint(QtGui.QPainter.Antialiasing)
        self._view.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        # A QChartView is a QGraphicsView, which accepts drops by default; let
        # them fall through to this widget's drop handler instead.
        self._view.setAcceptDrops(False)
        lay.addWidget(self._view, 1)
        self._apply_settings()

    # --- content contract -------------------------------------------------- #
    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title or self.DEFAULT_TITLE
        self._settings["title"] = self._title
        self._apply_settings()

    def consume_payload(self, payloads: list[dict]):
        added = False
        for p in payloads:
            full = p.get("full")
            if not full or any(t["full"] == full for t in self._traces):
                continue
            trace = {
                "full": full,
                "label": p.get("label", full),
                "color": _PALETTE[len(self._traces) % len(_PALETTE)],
                "width": 2.0,
                "dash": "solid",
                "visible": True,
            }
            if p.get("agg"):
                trace["agg"] = dict(p["agg"])
            self._traces.append(trace)
            self._add_series(trace)
            added = True
        self.refresh_live()
        if added:
            self.changed.emit()

    def variable_names(self) -> list[str]:
        return [t["full"] for t in self._traces]

    def derived_specs(self) -> dict[str, dict]:
        return {t["full"]: dict(t["agg"])
                for t in self._traces if t.get("agg")}

    # --- drop target ------------------------------------------------------- #
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(VARIABLE_MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(VARIABLE_MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):
        payloads = decode_variables(event.mimeData())
        if payloads:
            self.consume_payload(payloads)
            event.acceptProposedAction()

    def set_live_source(self, source):
        self._source = source
        self.refresh_live()

    def refresh_live(self):
        if self._chart is None:
            return
        xmin = xmax = ymin = ymax = None
        for t in self._traces:
            series = self._series.get(t["full"])
            if series is None:
                continue
            if self._source is not None:
                xs, ys = self._source.series(t["full"])
            else:
                xs, ys = (), ()
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            n = int(min(xs.size, ys.size))
            if n == 0:
                series.replace([])
                continue
            xs, ys = xs[:n], ys[:n]
            xd, yd = self._decimate(xs, ys, n)
            # A single C++-side vector swap; avoids per-point Python overhead
            # for the (bounded) display set.
            series.replace([QtCore.QPointF(float(x), float(y))
                            for x, y in zip(xd, yd)])
            if t.get("visible", True):
                xmn, xmx = float(xs.min()), float(xs.max())
                ymn, ymx = float(ys.min()), float(ys.max())
                xmin = xmn if xmin is None else min(xmin, xmn)
                xmax = xmx if xmax is None else max(xmax, xmx)
                ymin = ymn if ymin is None else min(ymin, ymn)
                ymax = ymx if ymax is None else max(ymax, ymx)
        if xmin is not None and xmax > xmin:
            self._axis_x.setRange(xmin, xmax)
            self._axis_x.setLabelFormat(self._axis_label_format(xmin, xmax))
        if ymin is not None:
            pad = (ymax - ymin) * 0.05 or (abs(ymax) * 0.05 or 1.0)
            lo, hi = ymin - pad, ymax + pad
            self._axis_y.setRange(lo, hi)
            self._axis_y.setLabelFormat(self._axis_label_format(lo, hi))
        self._notify_render()

    @staticmethod
    def _axis_label_format(lo: float, hi: float) -> str:
        """A printf label format for a value axis over ``[lo, hi]``.

        QtCharts' default (``%f``) renders large / tiny magnitudes as long or
        rounded-flat strings (e.g. ``100000.00``, or a near-constant range as
        repeated ``0.00`` labels that look normalised).  Switch to proper
        scientific notation for big / small magnitudes and to a compact general
        format otherwise, choosing the precision from the span so close labels
        stay distinct."""
        mag = max(abs(lo), abs(hi))
        span = abs(hi - lo)
        if mag != 0 and (mag >= 1e4 or mag < 1e-2):
            # Enough mantissa digits that adjacent ticks differ.
            digits = 2
            if span > 0:
                digits = max(2, min(6, int(math.log10(mag / span)) + 2))
            return f"%.{digits}e"
        # General format: pick decimals from the span (so a tight window like
        # [1.2340, 1.2343] still shows differing labels), capped for sanity.
        if span > 0:
            decimals = max(0, min(6, 2 - int(math.floor(math.log10(span)))))
            return f"%.{decimals}f"
        return "%.4g"

    @classmethod
    def _decimate(cls, xs, ys, n):
        """Down-sample a (possibly huge, ever-growing) run history to at most
        ``MAX_POINTS`` samples for display, so a long run doesn't rebuild tens of
        thousands of points on every live tick (the cause of GUI stalls).  Keeps
        the first/last sample and spreads the rest evenly."""
        if n <= cls.MAX_POINTS:
            return xs, ys
        idx = np.linspace(0, n - 1, cls.MAX_POINTS).astype(np.intp)
        return xs[idx], ys[idx]

    # --- chart plumbing ---------------------------------------------------- #
    def _add_series(self, trace: dict):
        if self._chart is None:
            return
        series = QtCharts.QLineSeries()
        self._chart.addSeries(series)
        series.attachAxis(self._axis_x)
        series.attachAxis(self._axis_y)
        self._series[trace["full"]] = series
        self._style_series(trace)

    def _style_series(self, trace: dict):
        series = self._series.get(trace["full"])
        if series is None:
            return
        series.setName(trace.get("label", trace["full"]))
        pen = QtGui.QPen(QtGui.QColor(trace.get("color", "#1f77b4")))
        pen.setWidthF(float(trace.get("width", 2.0)))
        pen.setStyle(_DASH_TO_QT.get(trace.get("dash", "solid"),
                                     QtCore.Qt.SolidLine))
        series.setPen(pen)
        series.setVisible(bool(trace.get("visible", True)))
        self._notify_render()

    def set_render_hook(self, fn):
        """Register a callback invoked whenever the visible content changes, so
        the hosting scene item can re-render its pixmap."""
        self._render_hook = fn

    def _notify_render(self):
        if self._render_hook is not None:
            self._render_hook()

    def _apply_settings(self):
        if self._chart is None:
            return
        s = self._settings
        self._chart.setTitle(s.get("title", ""))
        tf = self._chart.titleFont()
        tf.setPointSize(int(s.get("title_size", 12)))
        tf.setBold(True)
        self._chart.setTitleFont(tf)
        legend = self._chart.legend()
        legend.setVisible(bool(s.get("legend", True)))
        legend.setAlignment(
            self._LEGEND_ALIGN.get(s.get("legend_pos", "bottom"),
                                   QtCore.Qt.AlignBottom))
        af = QtGui.QFont()
        af.setPointSize(int(s.get("axis_size", 9)))
        for axis, key in ((self._axis_x, "x_title"), (self._axis_y, "y_title")):
            axis.setLabelsFont(af)
            axis.setTitleText(s.get(key, ""))
            axis.setTitleVisible(bool(s.get(key)))
        self._notify_render()

    # --- context-menu actions ---------------------------------------------- #
    def set_dialog_context(self, parent, anchor):
        """Where edit dialogs should parent + pop up (right side of the plot)."""
        self._dialog_parent = parent
        self._dialog_anchor = anchor

    def populate_menu(self, menu: "QtWidgets.QMenu"):
        if self._chart is None:
            return
        settings_act = menu.addAction("Chart settings…")
        settings_act.triggered.connect(self._edit_settings)
        if self._traces:
            traces_menu = menu.addMenu("Traces")
            for t in list(self._traces):
                sub = traces_menu.addMenu(t.get("label", t["full"]))
                edit = sub.addAction("Edit style…")
                edit.triggered.connect(lambda _=False, tr=t: self._edit_trace(tr))
                rem = sub.addAction("Remove")
                rem.triggered.connect(lambda _=False, tr=t: self._remove_trace(tr))

    def _edit_settings(self):
        dlg = ChartSettingsDialog(self._settings, self._dialog_parent)
        if _exec_beside(dlg, self._dialog_anchor):
            self._settings = dlg.result_settings()
            self._title = self._settings.get("title") or self.DEFAULT_TITLE
            self._apply_settings()

    def _edit_trace(self, trace: dict):
        dlg = TraceStyleDialog(trace, self._dialog_parent)
        if _exec_beside(dlg, self._dialog_anchor):
            trace.update(dlg.result_trace())
            self._style_series(trace)

    def _remove_trace(self, trace: dict):
        series = self._series.pop(trace["full"], None)
        if series is not None and self._chart is not None:
            self._chart.removeSeries(series)
        if trace in self._traces:
            self._traces.remove(trace)
            self.changed.emit()
        self._notify_render()

    # --- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"title": self._title, "traces": self._traces,
                "settings": self._settings}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeseriesChart":
        w = cls()
        w._settings.update(d.get("settings", {}))
        w._title = d.get("title", cls.DEFAULT_TITLE)
        for tr in d.get("traces", []):
            trace = dict(tr)
            w._traces.append(trace)
            w._add_series(trace)
        w._apply_settings()
        return w


# --------------------------------------------------------------------------- #
# Shared helpers / dialogs for snapshot charts (bar / pie).
# --------------------------------------------------------------------------- #
class EntryStyleDialog(QtWidgets.QDialog):
    """Edit one bar/pie entry's label, colour and visibility."""

    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Entry style")
        self._entry = dict(entry)
        form = QtWidgets.QFormLayout(self)

        self._legend = QtWidgets.QLineEdit(entry.get("label", ""))
        form.addRow("Label", self._legend)

        self._color = entry.get("color", "#1f77b4")
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.clicked.connect(self._pick_color)
        self._sync_color_btn()
        form.addRow("Colour", self._color_btn)

        self._visible = QtWidgets.QCheckBox("Visible")
        self._visible.setChecked(bool(entry.get("visible", True)))
        form.addRow("", self._visible)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _sync_color_btn(self):
        self._color_btn.setText(self._color)
        self._color_btn.setStyleSheet(
            f"background-color: {self._color}; color: #fff;")

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._color), self, "Pick colour")
        if c.isValid():
            self._color = c.name()
            self._sync_color_btn()

    def result_entry(self) -> dict:
        out = dict(self._entry)
        out["label"] = self._legend.text()
        out["color"] = self._color
        out["visible"] = self._visible.isChecked()
        return out


class _SnapshotChartSettingsDialog(QtWidgets.QDialog):
    """Title + legend settings shared by bar and pie charts."""

    def __init__(self, settings: dict, parent=None, *, y_label=False,
                 pie_options=False, hide_legend=False):
        super().__init__(parent)
        self.setWindowTitle("Chart settings")
        self._s = dict(settings)
        form = QtWidgets.QFormLayout(self)

        self._title = QtWidgets.QLineEdit(settings.get("title", ""))
        form.addRow("Title", self._title)

        self._title_size = QtWidgets.QSpinBox()
        self._title_size.setRange(8, 24)
        self._title_size.setValue(int(settings.get("title_size", 12)))
        form.addRow("Title size", self._title_size)

        self._legend = None
        self._legend_pos = None
        if not hide_legend:
            self._legend = QtWidgets.QCheckBox("Show legend")
            self._legend.setChecked(bool(settings.get("legend", True)))
            form.addRow("", self._legend)

            self._legend_pos = QtWidgets.QComboBox()
            self._legend_pos.addItems(["bottom", "top", "left", "right"])
            self._legend_pos.setCurrentText(settings.get("legend_pos", "bottom"))
            form.addRow("Legend position", self._legend_pos)

        self._y_title = None
        if y_label:
            self._y_title = QtWidgets.QLineEdit(settings.get("y_title", ""))
            form.addRow("Y axis title", self._y_title)

        self._show_percent = None
        if pie_options:
            self._show_percent = QtWidgets.QCheckBox(
                "Show percentages on slices")
            self._show_percent.setChecked(
                bool(settings.get("show_percent", True)))
            form.addRow("", self._show_percent)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_settings(self) -> dict:
        self._s.update({
            "title": self._title.text(),
            "title_size": self._title_size.value(),
        })
        if self._legend is not None:
            self._s["legend"] = self._legend.isChecked()
        if self._legend_pos is not None:
            self._s["legend_pos"] = self._legend_pos.currentText()
        if self._y_title is not None:
            self._s["y_title"] = self._y_title.text()
        if self._show_percent is not None:
            self._s["show_percent"] = self._show_percent.isChecked()
        return self._s


def _pie_slice_label(label: str, pct: float, *, show_percent: bool) -> str:
    if show_percent:
        return f"{label} ({pct:.1f}%)"
    return label


def _style_pie_slice(sl, *, text: str, show_label: bool, inside: bool):
    """Apply readable slice labels for the canvas pixmap size."""
    sl.setLabel(text)
    sl.setLabelVisible(show_label)
    if not show_label:
        return
    pos = (QtCharts.QPieSlice.LabelPosition.LabelInsideHorizontal if inside
           else QtCharts.QPieSlice.LabelPosition.LabelOutside)
    sl.setLabelPosition(pos)
    if inside:
        sl.setLabelColor(QtGui.QColor("#ffffff"))
    else:
        sl.setLabelArmLengthFactor(0.12)
    font = sl.labelFont()
    font.setPointSize(9 if inside else 8)
    sl.setLabelFont(font)


def _snapshot_value(source, full: str) -> float | None:
    if source is None:
        return None
    val = source.latest(full)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _entry_from_payload(p: dict, index: int) -> dict:
    entry = {
        "full": p["full"],
        "label": p.get("label", p["full"]),
        "color": _PALETTE[index % len(_PALETTE)],
        "visible": True,
    }
    if p.get("agg"):
        entry["agg"] = dict(p["agg"])
    return entry


class _SnapshotChartBase(QtWidgets.QWidget):
    """Shared drag/drop + live-source wiring for bar and pie charts."""

    changed = Signal()

    def __init__(self, default_title: str):
        super().__init__()
        self._title = default_title
        self._entries: list[dict] = []
        self._source = None
        self._render_hook = None
        self._dialog_parent = None
        self._dialog_anchor = None
        self.setAcceptDrops(True)

    def title(self) -> str:
        return self._title

    def set_title(self, title: str):
        self._title = title or self.DEFAULT_TITLE
        self._settings["title"] = self._title
        self._apply_settings()

    def consume_payload(self, payloads: list[dict]):
        added = False
        for p in payloads:
            full = p.get("full")
            if not full or any(e["full"] == full for e in self._entries):
                continue
            self._entries.append(_entry_from_payload(p, len(self._entries)))
            added = True
        if added:
            self._rebuild_entries()
            self.changed.emit()

    def variable_names(self) -> list[str]:
        return [e["full"] for e in self._entries]

    def derived_specs(self) -> dict[str, dict]:
        return {e["full"]: dict(e["agg"])
                for e in self._entries if e.get("agg")}

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(VARIABLE_MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(VARIABLE_MIME):
            event.acceptProposedAction()

    def dropEvent(self, event):
        payloads = decode_variables(event.mimeData())
        if payloads:
            self.consume_payload(payloads)
            event.acceptProposedAction()

    def set_live_source(self, source):
        self._source = source
        self.refresh_live()

    def set_render_hook(self, fn):
        self._render_hook = fn

    def _notify_render(self):
        if self._render_hook is not None:
            self._render_hook()

    def set_dialog_context(self, parent, anchor):
        self._dialog_parent = parent
        self._dialog_anchor = anchor

    def _visible_entries(self) -> list[dict]:
        return [e for e in self._entries if e.get("visible", True)]

    def _edit_entry(self, entry: dict):
        dlg = EntryStyleDialog(entry, self._dialog_parent)
        if _exec_beside(dlg, self._dialog_anchor):
            entry.update(dlg.result_entry())
            self._rebuild_entries()

    def _remove_entry(self, entry: dict):
        if entry in self._entries:
            self._entries.remove(entry)
            self.changed.emit()
            self._rebuild_entries()

    def populate_menu(self, menu: "QtWidgets.QMenu"):
        if self._chart is None:
            return
        settings_act = menu.addAction("Chart settings…")
        settings_act.triggered.connect(self._edit_settings)
        if self._entries:
            entries_menu = menu.addMenu("Entries")
            for e in list(self._entries):
                sub = entries_menu.addMenu(e.get("label", e["full"]))
                sub.addAction("Edit style…").triggered.connect(
                    lambda _=False, ent=e: self._edit_entry(ent))
                sub.addAction("Remove").triggered.connect(
                    lambda _=False, ent=e: self._remove_entry(ent))

    def _rebuild_entries(self):
        raise NotImplementedError

    def _apply_settings(self):
        raise NotImplementedError

    def refresh_live(self):
        raise NotImplementedError

    def _edit_settings(self):
        raise NotImplementedError


class BarChart(_SnapshotChartBase):
    """Bar chart of each variable's latest (live) value."""

    KIND = "bar"
    DEFAULT_TITLE = "Bar chart"

    def __init__(self):
        super().__init__(self.DEFAULT_TITLE)
        self._settings = {
            "title": self._title, "legend": True, "legend_pos": "bottom",
            "title_size": 12, "y_title": "",
        }
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if QtCharts is None:
            self._chart = None
            self._view = None
            self._bar_series = None
            lay.addWidget(QtWidgets.QLabel(
                "QtCharts is not available in this Qt build."))
            return

        self._chart = QtCharts.QChart()
        self._bar_series: QtCharts.QBarSeries | None = None
        self._axis_x = QtCharts.QBarCategoryAxis()
        self._axis_y = QtCharts.QValueAxis()
        self._chart.addAxis(self._axis_x, QtCore.Qt.AlignBottom)
        self._chart.addAxis(self._axis_y, QtCore.Qt.AlignLeft)
        self._view = QtCharts.QChartView(self._chart)
        self._view.setRenderHint(QtGui.QPainter.Antialiasing)
        self._view.setAcceptDrops(False)
        lay.addWidget(self._view, 1)
        self._apply_settings()

    def _apply_settings(self):
        if self._chart is None:
            return
        s = self._settings
        self._chart.setTitle(s.get("title", ""))
        tf = self._chart.titleFont()
        tf.setPointSize(int(s.get("title_size", 12)))
        tf.setBold(True)
        self._chart.setTitleFont(tf)
        legend = self._chart.legend()
        legend.setVisible(bool(s.get("legend", True)))
        legend.setAlignment(
            TimeseriesChart._LEGEND_ALIGN.get(s.get("legend_pos", "bottom"),
                                            QtCore.Qt.AlignBottom))
        self._axis_y.setTitleText(s.get("y_title", ""))
        self._axis_y.setTitleVisible(bool(s.get("y_title")))
        self._notify_render()

    def _edit_settings(self):
        dlg = _SnapshotChartSettingsDialog(
            self._settings, self._dialog_parent, y_label=True)
        if _exec_beside(dlg, self._dialog_anchor):
            self._settings = dlg.result_settings()
            self._title = self._settings.get("title") or self.DEFAULT_TITLE
            self._apply_settings()

    def _rebuild_entries(self):
        self.refresh_live()

    def refresh_live(self):
        if self._chart is None:
            return
        entries = self._visible_entries()
        labels: list[str] = []
        values: list[float] = []
        for e in entries:
            val = _snapshot_value(self._source, e["full"])
            if val is None:
                val = 0.0
            labels.append(e.get("label", e["full"]))
            values.append(val)

        self._chart.removeAllSeries()
        self._bar_series = QtCharts.QBarSeries()
        if len(values) == 1:
            bs = QtCharts.QBarSet(labels[0])
            bs.append(values[0])
            bs.setColor(QtGui.QColor(entries[0].get("color", "#1f77b4")))
            self._bar_series.append(bs)
            self._axis_x.clear()
            self._axis_x.append(labels)
        elif values:
            for e, label, val in zip(entries, labels, values):
                bs = QtCharts.QBarSet(label)
                bs.append(val)
                bs.setColor(QtGui.QColor(e.get("color", "#1f77b4")))
                self._bar_series.append(bs)
            self._axis_x.clear()
            self._axis_x.append([""])
        else:
            self._axis_x.clear()
        self._chart.addSeries(self._bar_series)
        self._bar_series.attachAxis(self._axis_x)
        self._bar_series.attachAxis(self._axis_y)
        if values:
            lo = min(0.0, min(values))
            hi = max(values)
            pad = (hi - lo) * 0.08 or (abs(hi) * 0.08 or 1.0)
            self._axis_y.setRange(lo - pad * 0.2, hi + pad)
            self._axis_y.setLabelFormat(
                TimeseriesChart._axis_label_format(lo, hi))
        self._notify_render()

    def to_dict(self) -> dict:
        return {"title": self._title, "entries": self._entries,
                "settings": self._settings}

    @classmethod
    def from_dict(cls, d: dict) -> "BarChart":
        w = cls()
        w._settings.update(d.get("settings", {}))
        w._title = d.get("title", cls.DEFAULT_TITLE)
        w._entries = [dict(e) for e in d.get("entries", [])]
        w._apply_settings()
        w.refresh_live()
        return w


class PieChart(_SnapshotChartBase):
    """Pie chart of each variable's latest value (proportional slices)."""

    KIND = "pie"
    DEFAULT_TITLE = "Pie chart"

    def __init__(self):
        super().__init__(self.DEFAULT_TITLE)
        self._settings = {
            "title": self._title, "legend": False,
            "title_size": 12, "show_percent": True,
        }
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if QtCharts is None:
            self._chart = None
            self._view = None
            self._pie_series = None
            lay.addWidget(QtWidgets.QLabel(
                "QtCharts is not available in this Qt build."))
            return

        self._chart = QtCharts.QChart()
        self._pie_series = QtCharts.QPieSeries()
        self._pie_series.setPieSize(0.68)
        self._chart.addSeries(self._pie_series)
        self._chart.setMargins(QtCore.QMargins(12, 12, 12, 12))
        self._view = QtCharts.QChartView(self._chart)
        self._view.setRenderHint(QtGui.QPainter.Antialiasing)
        self._view.setAcceptDrops(False)
        lay.addWidget(self._view, 1)
        self._apply_settings()

    def _apply_settings(self):
        if self._chart is None:
            return
        s = self._settings
        self._chart.setTitle(s.get("title", ""))
        tf = self._chart.titleFont()
        tf.setPointSize(int(s.get("title_size", 12)))
        tf.setBold(True)
        self._chart.setTitleFont(tf)
        legend = self._chart.legend()
        legend.setVisible(False)
        show_pct = bool(s.get("show_percent", True))
        if self._pie_series is not None:
            self._pie_series.setLabelsVisible(show_pct)
            # Leave room for outside callout labels when percentages are on.
            self._pie_series.setPieSize(0.62 if show_pct else 0.78)
        self._notify_render()

    def _edit_settings(self):
        dlg = _SnapshotChartSettingsDialog(
            self._settings, self._dialog_parent, pie_options=True,
            hide_legend=True)
        if _exec_beside(dlg, self._dialog_anchor):
            self._settings = dlg.result_settings()
            self._title = self._settings.get("title") or self.DEFAULT_TITLE
            self._apply_settings()
            self.refresh_live()

    def _rebuild_entries(self):
        self.refresh_live()

    def refresh_live(self):
        if self._chart is None:
            return
        self._pie_series.clear()
        show_pct = bool(self._settings.get("show_percent", True))
        slices: list[tuple[str, float, str]] = []
        for e in self._visible_entries():
            val = _snapshot_value(self._source, e["full"])
            if val is None:
                val = 0.0
            display = max(0.0, float(val))
            label = e.get("label", e["full"])
            slices.append((label, display, e.get("color", "#1f77b4")))
        total = sum(v for _, v, _ in slices)
        use_inside = show_pct and len(slices) > 3
        for label, display, color in slices:
            pct = 100.0 * display / total if total > 0 else 0.0
            sl = self._pie_series.append(label, display)
            sl.setColor(QtGui.QColor(color))
            if show_pct and total > 0:
                text = (f"{pct:.1f}%" if use_inside
                        else _pie_slice_label(label, pct, show_percent=True))
                _style_pie_slice(sl, text=text, show_label=True, inside=use_inside)
            else:
                _style_pie_slice(sl, text=label, show_label=True, inside=False)
        self._notify_render()

    def to_dict(self) -> dict:
        return {"title": self._title, "entries": self._entries,
                "settings": self._settings}

    @classmethod
    def from_dict(cls, d: dict) -> "PieChart":
        w = cls()
        w._settings.update(d.get("settings", {}))
        w._title = d.get("title", cls.DEFAULT_TITLE)
        w._entries = [dict(e) for e in d.get("entries", [])]
        w._apply_settings()
        w.refresh_live()
        return w


# --------------------------------------------------------------------------- #
# Content factory.
# --------------------------------------------------------------------------- #
_CONTENT_TYPES = {
    VariableTable.KIND: VariableTable,
    TimeseriesChart.KIND: TimeseriesChart,
    BarChart.KIND: BarChart,
    PieChart.KIND: PieChart,
}


def make_content(kind: str):
    """Fresh content widget for a kind (``"table"`` / ``"timeseries"`` / …)."""
    return _CONTENT_TYPES[kind]()


def content_from_dict(kind: str, d: dict):
    return _CONTENT_TYPES[kind].from_dict(d)


# --------------------------------------------------------------------------- #
# The scene item that hosts one content widget on the canvas.
#
# Unlike a floating widget over the viewport, this is a real ``QGraphicsItem``
# living in the scene, so the view transform scales *everything* -- frame,
# header, and content text -- as you zoom, exactly like the placed components.
# The content widget is never embedded in the scene (that path blanks out
# after interactions on some platforms); instead it is kept off-screen and
# rendered to a pixmap that the item paints.  The pixmap is re-rendered at the
# current zoom's device-pixel-ratio so text stays crisp.
# --------------------------------------------------------------------------- #
class PlotItem(QtWidgets.QGraphicsObject):
    """A titled, movable, resizable scene item hosting one content widget.

    The content (:class:`VariableTable`, :class:`TimeseriesChart`,
    :class:`BarChart`, or :class:`PieChart`) is kept
    off-screen and painted as a pixmap.  Structural edits happen through the
    right-click menu (and an ``Edit…`` pop-out that re-attaches the live widget
    for full interaction); drops of variables are consumed directly.
    """

    MIN_W = 120.0
    MIN_H = 90.0
    HEADER_H = 22.0
    GRIP = 14.0
    MARGIN = 2.0

    def __init__(self, content, canvas, w: float = 320.0, h: float = 230.0):
        super().__init__()
        self._canvas = canvas
        self.content = content
        self.kind = content.KIND
        self._w = max(self.MIN_W, float(w))
        self._h = max(self.MIN_H, float(h))
        self._pixmap: QtGui.QPixmap | None = None
        self._rendered_scale = 0.0
        self._resizing = False
        self._resize_start = None
        self._editing = False
        self._dialog = None
        # Content-area mouse interaction (forwarded to the off-screen widget).
        self._press_item_pos: QtCore.QPointF | None = None
        self._press_mouse_target = None
        self._press_toolbar_action = None
        self._dragging = False

        # Live data can request a re-render many times a second; coalesce those
        # into at most one raster per interval so a fast run can't stall the GUI
        # thread with back-to-back widget renders.
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(60)
        self._render_timer.timeout.connect(self._render)

        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(50)                    # float above wires / nodes

        # Keep the content off-screen (WA_DontShowOnScreen lays it out without a
        # visible window) so render() produces a correct pixmap.
        content.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
        content.show()
        content.set_render_hook(self._schedule_render)
        try:
            content.changed.connect(self._on_content_changed)
        except (AttributeError, TypeError):
            pass
        self._render()

    # --- geometry ---------------------------------------------------------- #
    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0.0, 0.0, self._w, self._h)

    def _content_size(self) -> tuple[int, int]:
        return (int(self._w - 2 * self.MARGIN),
                int(self._h - self.HEADER_H - self.MARGIN))

    def _target_scale(self) -> float:
        """Render resolution that matches on-screen device pixels: the view zoom
        times the screen's device-pixel-ratio (so it's crisp on HiDPI too),
        capped to bound the pixmap size."""
        zoom = self._canvas.transform().m11() if self._canvas is not None else 1.0
        if zoom <= 1e-9:
            zoom = 1.0
        try:
            dpr = self._canvas.viewport().devicePixelRatioF()
        except (AttributeError, TypeError):
            dpr = 1.0
        return max(1.0, min(5.0, zoom * (dpr if dpr > 0 else 1.0)))

    def on_view_scaled(self):
        """Re-render at the current zoom's resolution (called by the canvas after
        a zoom) so the pixmap resolution tracks the on-screen size and stays
        crisp instead of being stretched."""
        target = self._target_scale()
        if self._rendered_scale <= 0.0 or not (
                0.85 <= target / self._rendered_scale <= 1.18):
            self._render()

    def _schedule_render(self):
        """Coalesce frequent re-render requests (e.g. live data) into one raster
        per timer interval, keeping the GUI thread responsive during a run."""
        if self._editing or self.content is None:
            return
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _render(self):
        """Rasterise the (off-screen) content widget into a pixmap."""
        self._render_timer.stop()             # fold any pending coalesced render
        if self._editing or self.content is None:
            return
        cw, ch = self._content_size()
        if cw <= 2 or ch <= 2:
            self._pixmap = None
            self.update()
            return
        scale = self._target_scale()
        self._rendered_scale = scale
        if self.content.size() != QtCore.QSize(cw, ch):
            self.content.resize(cw, ch)
        pm = QtGui.QPixmap(max(1, int(cw * scale)), max(1, int(ch * scale)))
        pm.setDevicePixelRatio(scale)
        pm.fill(QtCore.Qt.white)
        self.content.render(pm)
        self._pixmap = pm
        self.update()

    # --- painting ---------------------------------------------------------- #
    def paint(self, p, _option, _widget=None):
        if self.content is None:                  # disposed; nothing to draw
            return
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.fillPath(path, QtGui.QColor("#ffffff"))

        if self._pixmap is not None:
            p.save()
            p.setClipPath(path)
            p.drawPixmap(QtCore.QPointF(self.MARGIN, self.HEADER_H), self._pixmap)
            p.restore()

        p.save()
        p.setClipPath(path)
        p.fillRect(QtCore.QRectF(0.0, 0.0, self._w, self.HEADER_H),
                   QtGui.QColor("#546e7a"))
        p.restore()
        p.setPen(QtGui.QColor("#ffffff"))
        f = p.font()
        f.setPointSizeF(10.0)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QtCore.QRectF(8.0, 0.0, self._w - 16.0, self.HEADER_H),
                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                   f"{self.content.title()}  ·  {self.kind}")

        selected = self.isSelected()
        p.setPen(QtGui.QPen(QtGui.QColor("#1976d2" if selected else "#90a4ae"),
                            1.5 if selected else 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(path)

        p.setPen(QtGui.QPen(QtGui.QColor("#607d8b"), 1.2))
        for d in (4.0, 8.0, 12.0):
            p.drawLine(QtCore.QPointF(self._w - d, self._h - 2.0),
                       QtCore.QPointF(self._w - 2.0, self._h - d))

    # --- interaction ------------------------------------------------------- #
    # The content widget is rendered off-screen (a pixmap), so mouse events on
    # the item don't reach its buttons / rows on their own.  We forward events
    # in the *content area* to the off-screen widget (so its buttons, selection
    # and cell edits work), reserve the *header* as the move handle, and the
    # bottom-right *grip* as the resize handle.  A press-drag on a table row
    # starts a native variable drag the canvas can drop onto another object.
    def _in_grip(self, pos: QtCore.QPointF) -> bool:
        return (pos.x() >= self._w - self.GRIP
                and pos.y() >= self._h - self.GRIP)

    def _in_content(self, pos: QtCore.QPointF) -> bool:
        return (self.content is not None
                and pos.y() >= self.HEADER_H and not self._in_grip(pos))

    def _content_point(self, item_pos: QtCore.QPointF) -> QtCore.QPoint:
        """Map an item-local point into the off-screen content widget's
        coordinates (the pixmap is drawn 1:1 below the header)."""
        return QtCore.QPoint(int(item_pos.x() - self.MARGIN),
                             int(item_pos.y() - self.HEADER_H))

    def _content_widget_at(self, item_pos: QtCore.QPointF):
        """Map an item-local point to (child widget, local point) in content."""
        cpos = self._content_point(item_pos)
        cpos.setX(max(0, min(cpos.x(), self.content.width() - 1)))
        cpos.setY(max(0, min(cpos.y(), self.content.height() - 1)))
        target = self.content.childAt(cpos) or self.content
        local = target.mapFrom(self.content, cpos)
        return target, local, cpos

    def _forward_mouse(self, event, etype, target=None):
        """Deliver a synthetic mouse event to the child widget under the cursor
        so the off-screen content stays interactive."""
        if self.content is None:
            return None
        if target is None:
            target, local, _cpos = self._content_widget_at(event.pos())
        else:
            _t, local, _cpos = self._content_widget_at(event.pos())
            local = target.mapFrom(self.content, _cpos)
        gp = event.screenPos()
        me = QtGui.QMouseEvent(
            etype, QtCore.QPointF(local), QtCore.QPointF(gp),
            event.button(), event.buttons(), event.modifiers())
        QtWidgets.QApplication.sendEvent(target, me)
        return target

    def remove_selected_row(self) -> bool:
        """Remove a selected table row (keyboard / canvas shortcut)."""
        fn = getattr(self.content, "remove_selected_row", None)
        if fn is None or not fn():
            return False
        self._render()
        return True

    def _forward_wheel(self, item_pos: QtCore.QPointF, event) -> bool:
        """Scroll table content when the wheel is over the content area."""
        if (self.content is None
                or getattr(self.content, "KIND", None) != "table"
                or not self._in_content(item_pos)):
            return False
        table = getattr(self.content, "_table", None)
        if table is None:
            return False
        sb = table.verticalScrollBar()
        if sb is None or sb.maximum() <= 0:
            return False
        pixel = event.pixelDelta().y()
        angle = event.angleDelta().y()
        if not pixel and not angle:
            return False
        before = sb.value()
        if pixel:
            sb.setValue(before - pixel)
        else:
            step = max(1, table.verticalHeader().defaultSectionSize())
            sb.setValue(before - int(angle / 120.0 * step * 3))
        if sb.value() == before:
            return False
        self._render()
        return True

    def hoverMoveEvent(self, event):
        if self._in_grip(event.pos()):
            self.setCursor(QtCore.Qt.SizeFDiagCursor)
        elif self._in_content(event.pos()):
            self.setCursor(QtCore.Qt.ArrowCursor)
        else:
            self.setCursor(QtCore.Qt.SizeAllCursor)   # header = move handle
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if (event.button() == QtCore.Qt.LeftButton
                and self._in_grip(event.pos())):
            self._resizing = True
            self._resize_start = (event.scenePos(), self._w, self._h)
            event.accept()
            return
        if (event.button() == QtCore.Qt.LeftButton
                and self._in_content(event.pos())):
            self.setSelected(True)
            self._dragging = False
            self._press_mouse_target = None
            self._press_toolbar_action = None
            _target, _local, cpos = self._content_widget_at(event.pos())
            toolbar_at = getattr(self.content, "toolbar_action_at", None)
            if toolbar_at is not None and toolbar_at(cpos) is not None:
                # Toolbar clicks on the off-screen pixmap are handled directly
                # on release -- synthetic events often miss QToolButton targets.
                self._press_toolbar_action = toolbar_at(cpos)
                self._press_item_pos = None
            else:
                self._press_item_pos = event.pos()
                self._press_mouse_target = self._forward_mouse(
                    event, QtCore.QEvent.MouseButtonPress)
            event.accept()
            return
        super().mousePressEvent(event)        # header -> move / select

    def mouseMoveEvent(self, event):
        if self._resizing:
            start, w0, h0 = self._resize_start
            d = event.scenePos() - start
            self.prepareGeometryChange()
            self._w = max(self.MIN_W, w0 + d.x())
            self._h = max(self.MIN_H, h0 + d.y())
            self._render()
            event.accept()
            return
        if self._press_item_pos is not None:
            moved = (event.pos() - self._press_item_pos).manhattanLength()
            if (not self._dragging
                    and moved >= QtWidgets.QApplication.startDragDistance()):
                self._maybe_start_drag()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_start = None
            self._notify_changed()
            event.accept()
            return
        if self._press_toolbar_action is not None:
            if event.button() == QtCore.Qt.LeftButton:
                self._press_toolbar_action()
            self._press_toolbar_action = None
            self._render()
            event.accept()
            return
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        if self._press_item_pos is not None:
            self._forward_mouse(
                event, QtCore.QEvent.MouseButtonRelease,
                self._press_mouse_target)
            self._press_item_pos = None
            self._press_mouse_target = None
            self._render()                    # reflect selection / button state
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._notify_changed()            # position may have changed

    def _maybe_start_drag(self):
        """Start a native variable drag if the pressed content point is a table
        row, so it can be dropped onto another plot object via the canvas."""
        getter = getattr(self.content, "drag_mime_at", None)
        if getter is None:
            return
        mime = getter(self._content_point(self._press_item_pos))
        if mime is None:
            return
        self._dragging = True
        self._press_item_pos = None
        self._render()                        # show the row selection
        drag = QtGui.QDrag(self._canvas.viewport())
        drag.setMimeData(mime)
        exec_(drag, QtCore.Qt.CopyAction)

    def mouseDoubleClickEvent(self, event):
        if self._in_content(event.pos()):
            self._forward_mouse(event, QtCore.QEvent.MouseButtonDblClick)
            event.accept()
            return
        self.rename()                         # double-click the header to rename
        event.accept()

    def contextMenuEvent(self, event):
        anchor = self._right_edge_global()
        if hasattr(self.content, "set_dialog_context"):
            self.content.set_dialog_context(self._canvas.window(), anchor)
        menu = QtWidgets.QMenu()
        menu.addAction("Rename…").triggered.connect(self.rename)
        menu.addAction("Edit…").triggered.connect(self._open_editor)
        menu.addSeparator()
        self.content.populate_menu(menu)
        menu.addSeparator()
        menu.addAction("Delete object").triggered.connect(self._delete)
        exec_(menu, event.screenPos())
        event.accept()

    # --- helpers ----------------------------------------------------------- #
    def _right_edge_global(self) -> QtCore.QPoint:
        scene_pt = self.mapToScene(QtCore.QPointF(self._w + 8.0, 0.0))
        view_pt = self._canvas.mapFromScene(scene_pt)
        return self._canvas.viewport().mapToGlobal(view_pt)

    def _on_content_changed(self):
        self._render()
        self._notify_changed()

    def _notify_changed(self):
        if self._canvas is not None:
            self._canvas._objects_changed()

    def consume(self, payloads: list[dict]):
        """Feed dropped variables to the content widget."""
        self.content.consume_payload(payloads)

    def rename(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self._canvas.window(), "Rename", "Title:", text=self.content.title())
        if ok:
            self.content.set_title(text)
            self._render()

    def _delete(self):
        if self._canvas is not None:
            self._canvas.remove_overlay(self)

    def dispose(self):
        """Tear down the off-screen content widget (and any open editor).

        The content widget is a *top-level* off-screen widget (shown with
        ``WA_DontShowOnScreen`` so it lays out for rendering); if left around it
        both leaks and -- being counted as a visible window -- keeps the Qt
        event loop alive after the main window closes.  Call this when the item
        is removed or the app shuts down."""
        self._render_timer.stop()
        if self._dialog is not None:
            self._dialog.close()
            self._dialog = None
        if self.content is not None:
            try:
                self.content.set_render_hook(None)
            except AttributeError:
                pass
            try:
                self.content.changed.disconnect(self._on_content_changed)
            except (RuntimeError, TypeError, AttributeError):
                pass
            self.content.close()             # hide -> no longer a "visible" window
            self.content.deleteLater()
            self.content = None
        self._pixmap = None

    def _open_editor(self):
        """Pop the live content widget out into a dialog for full interaction
        (row edits, reordering, in-place value edits), re-rendering on close."""
        if self._dialog is not None:
            self._dialog.raise_()
            self._dialog.activateWindow()
            return
        self._editing = True
        self.content.hide()
        self.content.setParent(None)
        self.content.setAttribute(QtCore.Qt.WA_DontShowOnScreen, False)

        dlg = QtWidgets.QDialog(self._canvas.window())
        dlg.setWindowTitle(f"Edit — {self.content.title()}")
        dlg.resize(max(440, int(self._w) + 40), max(320, int(self._h) + 60))
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(self.content, 1)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        self.content.show()
        self._dialog = dlg
        dlg.finished.connect(self._close_editor)
        dlg.show()
        dlg.move(self._right_edge_global())

    def _close_editor(self, *_):
        self.content.setParent(None)
        self.content.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
        self.content.show()                   # back off-screen
        self._dialog = None
        self._editing = False
        self._render()
        self._notify_changed()

    # --- persistence ------------------------------------------------------- #
    def scene_rect(self) -> QtCore.QRectF:
        return self.sceneBoundingRect()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x": self.pos().x(), "y": self.pos().y(),
            "w": self._w, "h": self._h,
            "content": self.content.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict, canvas) -> "PlotItem":
        content = content_from_dict(d["kind"], d.get("content", {}))
        obj = cls(content, canvas,
                  w=float(d.get("w", 320.0)), h=float(d.get("h", 230.0)))
        obj.setPos(float(d.get("x", 0.0)), float(d.get("y", 0.0)))
        return obj
