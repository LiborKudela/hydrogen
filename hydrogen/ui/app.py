"""The editor shell: a two-pane main window (catalogue tree | canvas) with a
File menu for project persistence and a Simulate action.

This module is the glue -- it wires the catalogue, canvas, dialogs and project
format together; the functionality lives in the sibling modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydrogen as hd
from hydrogen.serialization import SCHEMA_VERSION

from .canvas import Canvas
from .catalog import ComponentTree
from .items import NodeItem
from .project import is_project, make_project
from .qt import QtCore, QtGui, QtWidgets, exec_
from .simulate import SimulateDialog, default_sim_options

__all__ = ["MainWindow", "main"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        catalog = hd.component_catalog()
        self._by_type = {e["type"]: e for e in catalog}
        self._path: str | None = None             # current project file
        self._sim_options = default_sim_options()  # persisted run settings
        self.setWindowTitle("hydrogen — component palette")
        self._build_menu()
        self._build_window_controls()

        # Left: filter + tree.
        self._tree = ComponentTree(catalog)
        self._tree.itemDoubleClicked.connect(self._on_double_click)

        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter components…")
        self._filter.textChanged.connect(self._apply_filter)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.addWidget(QtWidgets.QLabel(
            f"<b>Catalogue</b> &mdash; {len(catalog)} components, "
            f"{len(hd.available_domains())} domains"))
        ll.addWidget(self._filter)
        ll.addWidget(self._tree, 1)

        # Right: canvas + toolbar (Simulate / Clear).
        self._canvas = Canvas(self._by_type, self._set_status)
        simulate = QtWidgets.QPushButton("▶ Simulate")
        simulate.setToolTip("Assemble all placed nodes into a system spec and "
                            "run it on a hydrogen host (via JSON load).")
        simulate.clicked.connect(self._simulate)
        clear = QtWidgets.QPushButton("Clear canvas")
        clear.clicked.connect(self._canvas.clear_nodes)

        fit = QtWidgets.QPushButton("Fit view")
        fit.setToolTip("Zoom/pan so all placed components fit the view.")
        fit.clicked.connect(self._canvas.fit_view)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("<b>Canvas</b>"))
        bar.addWidget(fit)
        bar.addStretch(1)
        bar.addWidget(simulate)
        bar.addWidget(clear)
        rl.addLayout(bar)
        rl.addWidget(self._canvas, 1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 720])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Ready")

    # --- File menu / project persistence ----------------------------------- #
    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        new_act = file_menu.addAction("&New")
        new_act.setShortcut(QtGui.QKeySequence.New)
        new_act.triggered.connect(self._new_project)

        open_act = file_menu.addAction("&Open…")
        open_act.setShortcut(QtGui.QKeySequence.Open)
        open_act.triggered.connect(self._open_project)

        file_menu.addSeparator()
        save_act = file_menu.addAction("&Save")
        save_act.setShortcut(QtGui.QKeySequence.Save)
        save_act.triggered.connect(self._save_project)

        save_as_act = file_menu.addAction("Save &As…")
        save_as_act.setShortcut(QtGui.QKeySequence.SaveAs)
        save_as_act.triggered.connect(self._save_project_as)

        file_menu.addSeparator()
        quit_act = file_menu.addAction("&Quit")
        quit_act.setShortcut(QtGui.QKeySequence.Quit)
        quit_act.triggered.connect(self.close)

    def _build_window_controls(self):
        controls = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(2)

        self._minimize_btn = QtWidgets.QToolButton(controls)
        self._minimize_btn.setText("−")
        self._minimize_btn.setToolTip("Minimize")
        self._minimize_btn.clicked.connect(self.showMinimized)

        self._fullscreen_btn = QtWidgets.QToolButton(controls)
        self._fullscreen_btn.setToolTip("Restore")
        self._fullscreen_btn.clicked.connect(self._toggle_fullscreen)

        self._close_btn = QtWidgets.QToolButton(controls)
        self._close_btn.setText("×")
        self._close_btn.setToolTip("Close")
        self._close_btn.clicked.connect(self.close)

        for btn in (self._minimize_btn, self._fullscreen_btn, self._close_btn):
            btn.setFixedSize(30, 24)
            btn.setAutoRaise(True)

        layout.addWidget(self._minimize_btn)
        layout.addWidget(self._fullscreen_btn)
        layout.addWidget(self._close_btn)
        self.menuBar().setCornerWidget(controls, QtCore.Qt.TopRightCorner)
        self._sync_window_controls()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._sync_window_controls()

    def _sync_window_controls(self):
        if not hasattr(self, "_fullscreen_btn"):
            return
        if self.isFullScreen():
            self._fullscreen_btn.setText("□")
            self._fullscreen_btn.setToolTip("Restore")
        else:
            self._fullscreen_btn.setText("▣")
            self._fullscreen_btn.setToolTip("Full screen")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange:
            self._sync_window_controls()

    def _new_project(self):
        self._canvas.clear_nodes()
        self._sim_options = default_sim_options()
        self._path = None
        self._update_title()
        self._set_status("New project")

    def _open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "Hydrogen UI project (*.json);;All files (*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            return
        if not is_project(data):
            QtWidgets.QMessageBox.warning(
                self, "Open", "This file isn't a hydrogen UI project.")
            return
        self._canvas.load_project(data.get("canvas", {}))
        self._sim_options = data.get("sim_options") or default_sim_options()
        self._path = path
        self._update_title()

    def _save_project(self):
        if self._path is None:
            return self._save_project_as()
        return self._write_project(self._path)

    def _save_project_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", self._path or "system.json",
            "Hydrogen UI project (*.json);;All files (*)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        self._path = path
        self._write_project(path)

    def _write_project(self, path: str):
        project = make_project(self._canvas.to_project(), self._sim_options)
        try:
            Path(path).write_text(json.dumps(project, indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._update_title()
        self._set_status(f"Saved project to {path}")

    def _update_title(self):
        name = Path(self._path).name if self._path else "untitled"
        self.setWindowTitle(f"hydrogen — component palette — {name}")

    def _set_status(self, text: str):
        self.statusBar().showMessage(text)

    def _on_double_click(self, item, _column):
        entry = item.data(0, ComponentTree.ENTRY_ROLE)
        if entry:
            self._canvas.add_node_at_center(entry)

    def _apply_filter(self, text: str):
        needle = text.strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            domain_item = self._tree.topLevelItem(i)
            visible_children = 0
            for j in range(domain_item.childCount()):
                leaf = domain_item.child(j)
                entry = leaf.data(0, ComponentTree.ENTRY_ROLE) or {}
                hay = (entry.get("name", "") + " " + entry.get("type", "")).lower()
                match = needle in hay
                leaf.setHidden(not match)
                visible_children += match
            domain_item.setHidden(needle != "" and visible_children == 0)

    def _build_system(self, nodes: list[NodeItem]) -> dict:
        components: dict[str, dict] = {}
        media: dict[str, dict] = {}
        for node in nodes:
            spec = hd.component_spec(node.type_name)
            template = dict(spec["template"])
            if node.params is not None:
                template["params"] = node.params
            if spec["needs_medium"]:
                name = node.medium or "Hydrogen"
                template["medium"] = name
                media.setdefault(name, {"fluid": name})
            components[node.comp_id] = template
        connections = [
            {"from": f"{c.src.node.comp_id}.{c.src.pname}",
             "to": f"{c.dst.node.comp_id}.{c.dst.pname}"}
            for c in self._canvas.connections()
        ]
        return {
            "hydrogen_version": hd.__version__,
            "schema_version": SCHEMA_VERSION,
            "media": media,
            "components": components,
            "connections": connections,
        }

    def _simulate(self):
        nodes = self._canvas.nodes()
        if not nodes:
            self._set_status("Place at least one component before simulating.")
            return
        system = self._build_system(nodes)
        dlg = SimulateDialog(system, options=self._sim_options, parent=self)
        exec_(dlg)
        self._sim_options = dlg.options()  # remember settings for save / reuse


def main(argv: list[str] | None = None):
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    win = MainWindow()
    win.resize(1040, 720)            # sensible size if restored from fullscreen
    win.showFullScreen()
    sys.exit(exec_(app))


if __name__ == "__main__":
    main()
