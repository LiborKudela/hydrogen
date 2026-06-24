"""Per-node parameter editor (reuses the dynamic spec form from
:mod:`hydrogen.ui.forms`)."""

from __future__ import annotations

import json

import hydrogen as hd

from .forms import FieldsForm
from .qt import QtCore, QtWidgets
from .session import structural_param_names

__all__ = ["PropertiesDialog"]


class PropertiesDialog(QtWidgets.QDialog):
    """Edit one node's medium + parameters.

    The form is built from ``hd.component_spec(type)`` (same metadata the
    standalone spec window renders).  On *OK* the values are written back onto
    the node.  The assembled component JSON is tucked behind a toggle button
    instead of being shown inline.
    """

    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self._spec = hd.component_spec(node.type_name)
        self.setWindowTitle(f"Properties — {node.comp_id}")

        outer = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"<b>{self._spec['name']}</b> &mdash; {self._spec['summary']}")
        header.setWordWrap(True)
        outer.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)

        self._medium = None
        if self._spec["needs_medium"]:
            mrow = QtWidgets.QWidget()
            mform = QtWidgets.QFormLayout(mrow)
            self._medium = QtWidgets.QLineEdit(node.medium or "Hydrogen")
            self._medium.setToolTip("CoolProp fluid name (also the media key).")
            self._medium.textChanged.connect(self._refresh_preview)
            mform.addRow("medium *", self._medium)
            body_layout.addWidget(mrow)

        self._params = FieldsForm(self._spec["parameters"])
        if node.params:
            self._params.set_values(node.params)
        self._params.mark_structural(structural_param_names(node.type_name))
        self._params.changed.connect(self._refresh_preview)
        body_layout.addWidget(self._params)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        legend = QtWidgets.QLabel(
            "<span style='color:#1b7a31'>&#9632; pure</span> "
            "(updated live on the running model) &nbsp;&nbsp; "
            "<span style='color:#b00020'>&#9632; structural</span> "
            "(changing it re-instantiates the model)")
        legend.setWordWrap(True)
        outer.addWidget(legend)

        # JSON preview: hidden behind a toggle (not shown inline).
        self._toggle = QtWidgets.QPushButton("Show JSON")
        self._toggle.setCheckable(True)
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle, alignment=QtCore.Qt.AlignLeft)

        self._preview = QtWidgets.QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setVisible(False)
        outer.addWidget(self._preview, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.resize(560, 720)

    def component_template(self) -> dict:
        """The node's entry for a system spec's ``components`` map."""
        template = dict(self._spec["template"])
        template["params"] = self._params.value()
        if self._medium is not None:
            template["medium"] = self._medium.text().strip() or None
        return template

    def _on_toggle(self, checked: bool):
        self._toggle.setText("Hide JSON" if checked else "Show JSON")
        self._preview.setVisible(checked)
        self._refresh_preview()

    def _refresh_preview(self, *_):
        if self._preview.isVisible():
            self._preview.setPlainText(json.dumps(self.component_template(), indent=2))

    def _accept(self):
        self.node.params = self._params.value()
        if self._medium is not None:
            self.node.medium = self._medium.text().strip() or None
        self.accept()
