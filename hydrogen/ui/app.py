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
from .home import HomeScreen
from .items import NodeItem
from .project import is_project, make_project
from .qt import QtCore, QtGui, QtSvg, QtWidgets, exec_
from .recent import add_recent_file, recent_files, remove_recent_file
from .session import SimulationSession
from .simulate import SimSettingsDialog, SimulateDialog, default_sim_options

__all__ = ["MainWindow", "main"]

# Floppy-disk ("save") glyphs, rendered ourselves so the toolbar shows a real
# diskette regardless of the desktop icon theme. Save As = the same diskette
# with a small pencil, to read as "save to a different file".
_FLOPPY_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>
  <path d='M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z'
        fill='#37474f'/>
  <rect x='7.5' y='3.4' width='7' height='5' rx='0.4' fill='#eceff1'/>
  <rect x='11.6' y='4.1' width='1.9' height='3.4' rx='0.3' fill='#90a4ae'/>
  <rect x='6.4' y='12.4' width='11.2' height='6.6' rx='0.6' fill='#eceff1'/>
  <line x1='8.2' y1='14.5' x2='15.8' y2='14.5' stroke='#90a4ae' stroke-width='1'/>
  <line x1='8.2' y1='16.6' x2='15.8' y2='16.6' stroke='#b0bec5' stroke-width='1'/>
</svg>
"""

_FLOPPY_AS_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>
  <path d='M4 3h9l3 3v8.5l-7.2 7.2A2 2 0 0 1 8 22H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z'
        fill='#37474f'/>
  <rect x='6' y='3.4' width='6' height='4.6' rx='0.4' fill='#eceff1'/>
  <rect x='9.5' y='4.0' width='1.8' height='3.2' rx='0.3' fill='#90a4ae'/>
  <rect x='5' y='11.6' width='9' height='5.4' rx='0.6' fill='#eceff1'/>
  <line x1='6.6' y1='13.3' x2='12.4' y2='13.3' stroke='#90a4ae' stroke-width='0.9'/>
  <g transform='rotate(45 17.8 17.8)'>
    <rect x='13.2' y='16.5' width='8' height='2.7' rx='0.3' fill='#fbc02d'/>
    <rect x='13.2' y='16.5' width='1.5' height='2.7' fill='#cfd8dc'/>
    <path d='M21.2 16.5 L22.8 17.85 L21.2 19.2 Z' fill='#4e342e'/>
  </g>
</svg>
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        catalog = hd.component_catalog()
        self._by_type = {e["type"]: e for e in catalog}
        self._path: str | None = None             # current project file
        self._sim_options = default_sim_options()  # persisted run settings
        self._session = SimulationSession()        # long-lived host + model
        self.setWindowTitle("hydrogen — component palette")
        self._build_menu()
        self._build_toolbar()
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
        self._catalog_label = QtWidgets.QLabel()
        self._update_catalog_label(catalog)
        refresh = QtWidgets.QPushButton("Refresh data")
        refresh.setToolTip("Reload the component catalogue and rebuild placed "
                           "canvas components from the latest metadata.")
        refresh.clicked.connect(self._refresh_data)

        catalog_bar = QtWidgets.QHBoxLayout()
        catalog_bar.addWidget(self._catalog_label, 1)
        catalog_bar.addWidget(refresh)
        ll.addLayout(catalog_bar)
        ll.addWidget(self._filter)
        ll.addWidget(self._tree, 1)

        # Right: canvas + toolbar (Settings / Simulate / Clear).
        self._canvas = Canvas(self._by_type, self._set_status)
        sim_settings = QtWidgets.QPushButton("⚙ Settings")
        sim_settings.setToolTip("Edit the instantiate / initialise / simulate "
                                "options used by every run.")
        sim_settings.clicked.connect(self._edit_sim_settings)
        simulate = QtWidgets.QPushButton("▶ Simulate")
        simulate.setToolTip("Open the run window. The model is kept alive after "
                            "building and only re-instantiated when its "
                            "structure changes.")
        simulate.clicked.connect(self._simulate)
        clear = QtWidgets.QPushButton("Clear canvas")
        clear.clicked.connect(self._canvas.clear_nodes)

        fit = QtWidgets.QPushButton("Fit view")
        fit.setToolTip("Zoom/pan so all placed components fit the view.")
        fit.clicked.connect(self._canvas.fit_view)

        markers = QtWidgets.QToolButton()
        markers.setText("Markers ▾")
        markers.setToolTip("Choose which on-canvas markers are shown.")
        markers.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        markers_menu = QtWidgets.QMenu(markers)
        for label, tip, slot in (
            ("Names", "Instance-name labels",
             self._canvas.set_names_visible),
            ("Types", "Component-type labels",
             self._canvas.set_types_visible),
            ("Labels", "ui_label parameter labels",
             self._canvas.set_params_visible),
            ("Port names", "Per-port name labels",
             self._canvas.set_port_names_visible),
        ):
            act = markers_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(True)
            act.setToolTip(tip)
            act.toggled.connect(slot)
        markers_menu.setToolTipsVisible(True)
        markers.setMenu(markers_menu)
        self._markers_btn = markers

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("<b>Canvas</b>"))
        bar.addWidget(fit)
        bar.addWidget(self._markers_btn)
        bar.addStretch(1)
        bar.addWidget(sim_settings)
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
        self._editor = splitter

        self._home = HomeScreen(self._by_type)
        self._home.newRequested.connect(self._new_project)
        self._home.openRequested.connect(self._open_project)
        self._home.openRecentRequested.connect(self._open_path)

        # Central area: bottom tabs to switch between the start screen, the
        # model canvas and (later) simulation-result viewers. Home + Model are
        # permanent; result tabs are added dynamically and are closable.
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setTabPosition(QtWidgets.QTabWidget.South)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self._tabs.addTab(self._home, "⌂  Home")
        self._tabs.addTab(self._editor, "Model")
        tab_bar = self._tabs.tabBar()
        for i in (0, 1):                      # strip close buttons off the permanent tabs
            for side in (QtWidgets.QTabBar.RightSide, QtWidgets.QTabBar.LeftSide):
                tab_bar.setTabButton(i, side, None)
        self.setCentralWidget(self._tabs)

        self.statusBar().showMessage("Ready")
        self._show_home()

    # --- File menu / project persistence ----------------------------------- #
    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        new_act = file_menu.addAction("&New")
        new_act.setShortcut(QtGui.QKeySequence.New)
        new_act.triggered.connect(self._new_project)

        open_act = file_menu.addAction("&Open…")
        open_act.setShortcut(QtGui.QKeySequence.Open)
        open_act.triggered.connect(self._open_project)

        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._recent_menu.aboutToShow.connect(self._rebuild_recent_menu)

        home_act = file_menu.addAction("&Home")
        home_act.triggered.connect(self._show_home)

        file_menu.addSeparator()
        save_act = file_menu.addAction("&Save")
        save_act.setIcon(self._svg_icon(_FLOPPY_SVG, QtWidgets.QStyle.SP_DialogSaveButton))
        save_act.setShortcut(QtGui.QKeySequence.Save)
        save_act.triggered.connect(self._save_project)
        self._save_act = save_act

        save_as_act = file_menu.addAction("Save &As…")
        save_as_act.setIcon(self._svg_icon(_FLOPPY_AS_SVG, QtWidgets.QStyle.SP_DriveFDIcon))
        save_as_act.setShortcut(QtGui.QKeySequence.SaveAs)
        save_as_act.triggered.connect(self._save_project_as)
        self._save_as_act = save_as_act

        file_menu.addSeparator()
        quit_act = file_menu.addAction("&Quit")
        quit_act.setShortcut(QtGui.QKeySequence.Quit)
        quit_act.triggered.connect(self.close)

    def _svg_icon(self, svg: str, fallback) -> QtGui.QIcon:
        """Build a crisp `QIcon` from an inline SVG string, falling back to a
        Qt standard pixmap if the SVG module isn't available."""
        if QtSvg is None:
            return self.style().standardIcon(fallback)
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.strip().encode("utf-8")))
        pm = QtGui.QPixmap(64, 64)
        pm.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pm)
        renderer.render(painter)
        painter.end()
        return QtGui.QIcon(pm)

    def _build_toolbar(self):
        """The top tool bar. Currently holds the project Save / Save As
        actions (shared with the File menu); more actions land here later."""
        tb = QtWidgets.QToolBar("Main", self)
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QtCore.QSize(20, 20))
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)
        tb.addAction(self._save_act)
        tb.addAction(self._save_as_act)
        self._toolbar = tb

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

    def _show_home(self):
        self._home.refresh()
        self._tabs.setCurrentWidget(self._home)
        self.setWindowTitle("hydrogen — start")
        self._set_status("Ready")

    def _show_editor(self):
        self._tabs.setCurrentWidget(self._editor)

    def add_result_tab(self, widget: QtWidgets.QWidget, title: str) -> int:
        """Add a (closable) simulation-result viewer tab and switch to it.
        Returns the new tab index."""
        index = self._tabs.addTab(widget, title)
        self._tabs.setCurrentIndex(index)
        return index

    def _close_tab(self, index: int):
        widget = self._tabs.widget(index)
        if widget in (self._home, self._editor):
            return                            # Home + Model are permanent
        self._tabs.removeTab(index)
        widget.deleteLater()

    def _new_project(self):
        self._canvas.clear_nodes()
        self._session.reset()
        self._sim_options = default_sim_options()
        self._path = None
        self._show_editor()
        self._update_title()
        self._set_status("New project")

    def _open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "Hydrogen UI project (*.json);;All files (*)")
        if path:
            self._open_path(path)

    def _open_path(self, path: str):
        """Load a project from ``path`` (shared by the dialog + recent list)."""
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            remove_recent_file(path)
            self._home.refresh()
            return
        if not is_project(data):
            QtWidgets.QMessageBox.warning(
                self, "Open", "This file isn't a hydrogen UI project.")
            return
        self._canvas.load_project(data.get("canvas", {}))
        self._session.reset()
        self._sim_options = data.get("sim_options") or default_sim_options()
        self._path = path
        add_recent_file(path)
        self._show_editor()
        self._update_title()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        files = recent_files()
        if not files:
            empty = self._recent_menu.addAction("(no recent projects)")
            empty.setEnabled(False)
            return
        for path in files:
            act = self._recent_menu.addAction(path)
            act.triggered.connect(lambda _=False, p=path: self._open_path(p))
        self._recent_menu.addSeparator()
        clear_act = self._recent_menu.addAction("Clear recent")
        clear_act.triggered.connect(self._clear_recent)

    def _clear_recent(self):
        from .recent import clear_recent_files
        clear_recent_files()
        self._home.refresh()

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
        add_recent_file(path)
        self._update_title()
        self._set_status(f"Saved project to {path}")

    def _update_title(self):
        name = Path(self._path).name if self._path else "untitled"
        self.setWindowTitle(f"hydrogen — component palette — {name}")
        if hasattr(self, "_tabs"):
            self._tabs.setTabText(self._tabs.indexOf(self._editor), name)

    def _set_status(self, text: str):
        self.statusBar().showMessage(text)

    def _update_catalog_label(self, catalog: list[dict]):
        domains = {entry["domain"] for entry in catalog}
        self._catalog_label.setText(
            f"<b>Catalogue</b> &mdash; {len(catalog)} components, "
            f"{len(domains)} domains")

    def _refresh_data(self):
        catalog = hd.component_catalog()
        self._by_type = {e["type"]: e for e in catalog}
        self._tree.set_catalog(catalog)
        self._canvas.refresh_catalog(self._by_type)
        self._home.set_catalog(self._by_type)
        self._session.reset()
        self._apply_filter(self._filter.text())
        self._update_catalog_label(catalog)
        self._set_status(
            f"Refreshed catalogue and {self._canvas.node_count()} canvas component(s)")

    def _on_double_click(self, item, _column):
        entry = item.data(0, ComponentTree.ENTRY_ROLE)
        if entry:
            self._canvas.add_node_at_center(entry)

    def _apply_filter(self, text: str):
        needle = text.strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            domain_item = self._tree.topLevelItem(i)
            domain_visible = self._filter_item(domain_item, needle)
            domain_item.setHidden(needle != "" and not domain_visible)

    def _filter_item(self, item, needle: str) -> bool:
        entry = item.data(0, ComponentTree.ENTRY_ROLE)
        if entry:
            hay = " ".join(
                str(entry.get(key, ""))
                for key in ("name", "type", "domain", "category")
            ).lower()
            match = needle in hay
            item.setHidden(not match)
            return match

        visible_children = 0
        for i in range(item.childCount()):
            child = item.child(i)
            if self._filter_item(child, needle):
                visible_children += 1
        item.setHidden(needle != "" and visible_children == 0)
        return visible_children > 0

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

    def _edit_sim_settings(self):
        nodes = self._canvas.nodes()
        system = self._build_system(nodes) if nodes else None
        dlg = SimSettingsDialog(self._sim_options, system=system, parent=self)
        if exec_(dlg):
            self._sim_options = dlg.options()  # remember for save / reuse
            self._set_status("Simulation settings updated.")

    def _simulate(self):
        if not self._canvas.nodes():
            self._set_status("Place at least one component before simulating.")
            return
        dlg = SimulateDialog(
            self._session,
            build_system=lambda: self._build_system(self._canvas.nodes()),
            get_options=lambda: self._sim_options,
            parent=self,
        )
        exec_(dlg)

    def closeEvent(self, event):
        self._session.shutdown()
        super().closeEvent(event)


def main(argv: list[str] | None = None):
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    app.setOrganizationName("hydrogen")
    app.setApplicationName("hydrogen-ui")
    win = MainWindow()
    win.resize(1040, 720)            # sensible size if restored from fullscreen
    win.showFullScreen()
    sys.exit(exec_(app))


if __name__ == "__main__":
    main()
