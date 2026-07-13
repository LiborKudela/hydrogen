"""Standalone single-component form: render one ``component_spec`` into a window
and round-trip it through ``from_dict`` to confirm it loads.

A focused harness for the dynamic form in :mod:`hydrogen.ui.forms` (handy when
authoring a new component's :class:`~hydrogen.paramspec.ParamSpec` metadata).

Run::

    python3 -m hydrogen.ui.spec_window [hydrogen.thermofluid.Pipe]
"""

from __future__ import annotations

import json
import sys

import hydrogen as hd
from hydrogen.serialization import SCHEMA_VERSION

from .forms import FieldsForm
from .qt import QtWidgets, exec_, install_wheel_guard
from .theme import apply_theme

__all__ = ["SpecWindow", "main"]

DEFAULT_COMPONENT = "hydrogen.thermofluid.Pipe"


class SpecWindow(QtWidgets.QMainWindow):
    def __init__(self, type_name: str):
        super().__init__()
        self._spec = hd.component_spec(type_name)
        self.setWindowTitle(f"component_spec: {type_name}")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

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
            self._medium = QtWidgets.QLineEdit("H2")
            mform.addRow("medium *", self._medium)
            body_layout.addWidget(mrow)

        self._params = FieldsForm(self._spec["parameters"])
        body_layout.addWidget(self._params)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        build = QtWidgets.QPushButton("Build spec")
        build.clicked.connect(self._build)
        outer.addWidget(build)

        self._out = QtWidgets.QPlainTextEdit()
        self._out.setReadOnly(True)
        outer.addWidget(self._out, 1)

    def _build(self):
        template = dict(self._spec["template"])
        template["params"] = self._params.value()
        medium_name = None
        if self._medium is not None:
            medium_name = self._medium.text().strip() or None
            template["medium"] = medium_name

        system = {
            "hydrogen_version": hd.__version__,
            "schema_version": SCHEMA_VERSION,
            "media": {medium_name: {"fluid": "Hydrogen"}} if medium_name else {},
            "components": {"comp": template},
            "connections": [],
        }
        try:
            hd.from_dict(system)
            status = "OK -- from_dict() accepted the spec."
        except Exception as exc:  # surface validation feedback in the UI
            status = f"from_dict() rejected the spec:\n{exc}"
        self._out.setPlainText(status + "\n\n" + json.dumps(template, indent=2))


def main(argv: list[str] | None = None):
    argv = list(sys.argv if argv is None else argv)
    type_name = argv[1] if len(argv) > 1 else DEFAULT_COMPONENT
    app = QtWidgets.QApplication(argv[:1])
    install_wheel_guard(app)
    apply_theme(app)
    win = SpecWindow(type_name)
    win.resize(680, 860)
    win.show()
    sys.exit(exec_(app))


if __name__ == "__main__":
    main()
