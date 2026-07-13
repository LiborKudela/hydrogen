"""The solver diagnostics window: a human-readable view of `Model.diagnose`.

When a run fails (or stalls), the host produces a structured post-mortem of the
Newton/Jacobian state -- non-finite residuals, a singular Jacobian, the
near-null-space variables, the worst residual rows, and a per-component
ranking.  This window renders that report so the user can see *why* the solve
failed and *which component* to fix, instead of a bare ``last_err=inf``.

The window is passive: it renders whatever report dict it is given via
:meth:`set_report` and asks its owner to re-fetch by emitting
:attr:`refreshRequested` (the blocking host call must run off the GUI thread).
"""

from __future__ import annotations

from .qt import QtCore, QtWidgets, Signal
from . import theme


def _fmt_num(x, unit: str = "") -> str:
    """Format a report number that may be ``None`` (non-finite) or the string
    ``"inf"`` from the host."""
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:                                   # NaN
        return "n/a"
    if abs(f) != 0 and (abs(f) >= 1e5 or abs(f) < 1e-3):
        s = f"{f:.4e}"
    else:
        s = f"{f:.6g}"
    return f"{s}{unit}"


class DiagnosticWindow(QtWidgets.QDialog):
    """Non-modal viewer for a :func:`hydrogen.Model.diagnose` report."""

    refreshRequested = Signal()

    def __init__(self, report: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solver diagnostics")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.resize(720, 720)
        self._report: dict = {}

        lay = QtWidgets.QVBoxLayout(self)

        # --- verdict banner ------------------------------------------------
        self._banner = QtWidgets.QLabel()
        self._banner.setWordWrap(True)
        self._banner.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._banner.setMargin(10)
        lay.addWidget(self._banner)

        # --- stat strip ----------------------------------------------------
        self._stats = QtWidgets.QLabel()
        self._stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._stats.setWordWrap(True)
        lay.addWidget(self._stats)

        # --- causes --------------------------------------------------------
        self._causes = QtWidgets.QLabel()
        self._causes.setWordWrap(True)
        self._causes.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self._causes)

        # --- detail tabs ---------------------------------------------------
        self._tabs = QtWidgets.QTabWidget()
        self._t_components = self._make_table(
            ["Component", "Score", "Max residual",
             "NaN/Inf res", "NaN/Inf var", "Singular", "Vars"])
        self._t_singular = self._make_table(
            ["Variable", "Weight", "Component"])
        self._t_nonfinite = self._make_table(
            ["Kind", "Location", "Component", "Detail"])
        self._t_residuals = self._make_table(
            ["Equation", "Residual", "Component", "Variables"])
        self._tabs.addTab(self._t_components, "Components")
        self._tabs.addTab(self._t_singular, "Near-singular")
        self._tabs.addTab(self._t_nonfinite, "Non-finite")
        self._tabs.addTab(self._t_residuals, "Worst residuals")
        lay.addWidget(self._tabs, 1)

        # --- buttons -------------------------------------------------------
        btns = QtWidgets.QHBoxLayout()
        self._refresh_btn = QtWidgets.QPushButton("Re-run diagnosis")
        self._refresh_btn.setToolTip(
            "Recompute the post-mortem against the current model state.")
        self._refresh_btn.clicked.connect(self.refreshRequested.emit)
        copy_btn = QtWidgets.QPushButton("Copy report")
        copy_btn.setToolTip("Copy the full report (JSON) to the clipboard.")
        copy_btn.clicked.connect(self._copy_report)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self._refresh_btn)
        btns.addWidget(copy_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        if report is not None:
            self.set_report(report)

    def _make_table(self, headers: list[str]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    # --- rendering --------------------------------------------------------

    def set_pending(self, message: str = "Running diagnosis…"):
        self._refresh_btn.setEnabled(False)
        self._banner.setText(f"<i>{message}</i>")

    def set_report(self, report: dict | None):
        self._refresh_btn.setEnabled(True)
        if not report:
            self._banner.setText("<b>No diagnostic available.</b> Run the model "
                                 "(and let it fail) or press "
                                 "<b>Re-run diagnosis</b>.")
            self._stats.clear()
            self._causes.clear()
            for t in (self._t_components, self._t_singular, self._t_nonfinite,
                      self._t_residuals):
                t.setRowCount(0)
            return
        self._report = dict(report)
        pal = theme.current()
        ok = bool(report.get("ok"))
        # Severity drives the banner colour/label: error (red) > warning
        # (amber) > ok (green).  Fall back to the boolean `ok` for old reports.
        severity = report.get("severity") or ("ok" if ok else "error")
        colour, label = {
            "ok": (getattr(pal, "ok", "#2e7d32"), "OK"),
            "warning": ("#f9a825", "ADVISORY"),
            "error": ("#c62828", "PROBLEM"),
        }.get(severity, ("#c62828", "PROBLEM"))
        summary = report.get("summary") or ("Healthy." if ok else "Solve failed.")
        codes = report.get("cause_codes") or []
        code_str = (" &nbsp; " + " ".join(
            f"<span style='background:rgba(255,255,255,0.25); "
            f"border-radius:4px; padding:1px 5px;'>{c}</span>" for c in codes)
        ) if codes else ""
        self._banner.setStyleSheet(
            f"background:{colour}; color:white; border-radius:6px;")
        self._banner.setText(f"<b>{label}:</b> {summary}{code_str}")

        cond_info = report.get("conditioning") or {}
        cond = cond_info.get("value", report.get("condition_estimate"))
        band = cond_info.get("band")
        band_str = f" ({band})" if band and band != "unknown" else ""
        self._stats.setText(
            f"<span style='color:{getattr(pal, 'muted', '#888')}'>"
            f"t = {_fmt_num(report.get('t'))} s &nbsp;|&nbsp; "
            f"dt = {_fmt_num(report.get('dt'))} s &nbsp;|&nbsp; "
            f"vars = {report.get('n_v', '?')} &nbsp;|&nbsp; "
            f"||F|| = {_fmt_num(report.get('residual_norm'))} &nbsp;|&nbsp; "
            f"cond(J) = {_fmt_num(cond)}{band_str} &nbsp;|&nbsp; "
            f"min σ = {_fmt_num(report.get('min_singular_value'))} &nbsp;|&nbsp; "
            f"last: {report.get('last_iters', 0)} iters, "
            f"res {_fmt_num(report.get('last_residual'))}</span>")

        causes = report.get("likely_causes") or []
        if causes:
            items = "".join(f"<li>{c}</li>" for c in causes)
            self._causes.setText(f"<b>Likely cause(s):</b><ul>{items}</ul>")
        else:
            self._causes.clear()

        self._fill_components(report.get("components") or [])
        self._fill_singular(report.get("near_singular_vars") or [])
        self._fill_nonfinite(report)
        self._fill_residuals(report.get("worst_residuals") or [])

        # Nudge the user toward the most useful tab.
        if report.get("near_singular_vars"):
            self._tabs.setCurrentWidget(self._t_singular)
        elif report.get("nonfinite_residuals") or report.get("nonfinite_vars"):
            self._tabs.setCurrentWidget(self._t_nonfinite)
        else:
            self._tabs.setCurrentWidget(self._t_components)

    def _set_row(self, table, row, cells):
        for col, val in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(str(val))
            table.setItem(row, col, item)

    def _fill_components(self, comps):
        self._t_components.setRowCount(len(comps))
        for i, c in enumerate(comps):
            self._set_row(self._t_components, i, [
                c.get("component"), _fmt_num(c.get("score")),
                _fmt_num(c.get("max_residual")),
                c.get("n_nonfinite_res", 0), c.get("n_nonfinite_vars", 0),
                c.get("n_singular", 0), c.get("n_vars", 0)])
        self._t_components.resizeColumnsToContents()

    def _fill_singular(self, nsv):
        self._t_singular.setRowCount(len(nsv))
        for i, v in enumerate(nsv):
            self._set_row(self._t_singular, i, [
                v.get("name"), _fmt_num(v.get("weight")), v.get("component")])
        self._t_singular.resizeColumnsToContents()

    def _fill_nonfinite(self, report):
        rows = []
        for v in report.get("nonfinite_vars") or []:
            rows.append(("variable", v.get("name"), v.get("component"),
                         "value is NaN/Inf"))
        for e in report.get("nonfinite_residuals") or []:
            rows.append(("residual", f"eq {e.get('eq')}", e.get("component"),
                         ", ".join(e.get("variables") or [])[:80]))
        self._t_nonfinite.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self._set_row(self._t_nonfinite, i, r)
        self._t_nonfinite.resizeColumnsToContents()

    def _fill_residuals(self, worst):
        self._t_residuals.setRowCount(len(worst))
        for i, w in enumerate(worst):
            self._set_row(self._t_residuals, i, [
                w.get("eq"), _fmt_num(w.get("residual")), w.get("component"),
                ", ".join(w.get("variables") or [])])
        self._t_residuals.resizeColumnsToContents()

    def _copy_report(self):
        import json
        try:
            text = json.dumps(self._report, indent=2)
        except Exception:  # noqa: BLE001
            text = str(self._report)
        QtWidgets.QApplication.clipboard().setText(text)
