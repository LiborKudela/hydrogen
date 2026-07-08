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
from .live import LiveController
from .media import MediaManagerDialog, default_media_spec
from .project import is_project, make_project
from .qt import QtCore, QtGui, QtSvg, QtWidgets, exec_
from .recent import add_recent_file, recent_files, remove_recent_file
from .session import SimulationSession
from .simulate import (
    SimSettingsDialog,
    SimulateDialog,
    _SessionWorker,
    default_sim_options,
    initialise_kwargs,
    instantiate_kwargs,
    run_config_patch,
    run_kwargs,
)

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
        self._media = {"Hydrogen": default_media_spec("Hydrogen")}  # shared fluids
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

        # Right: canvas + toolbar (Media / Settings / Simulate / Clear).
        self._canvas = Canvas(self._by_type, self._set_status)
        self._canvas.set_media_provider(lambda: list(self._media))
        # Live pump: refresh canvas plot/table objects off the simulation stream.
        self._live = LiveController(self._session)
        self._canvas.set_objects_changed_hook(self._sync_live_objects)
        self._canvas.set_model_changed_hook(self._on_canvas_model_changed)
        # Persistent run log: every build/run log line is recorded here (even
        # for runs launched from the toolbar with no window open), so the
        # Simulate window shows the full output + summary whenever it is opened.
        self._run_log: list[tuple[str, str]] = []
        self._log_view = None               # the open Simulate window's panel
        self._live.set_log_sink(self._record_log)
        self._live.start()
        media_btn = QtWidgets.QPushButton("Media")
        media_btn.setToolTip("Manage the shared CoolProp fluids (backend, cache) "
                             "the components reference.")
        media_btn.clicked.connect(self._edit_media)
        sim_settings = QtWidgets.QPushButton("⚙ Settings")
        sim_settings.setToolTip("Edit the instantiate / initialise / simulate "
                                "options used by every run.")
        sim_settings.clicked.connect(self._edit_sim_settings)
        self._sim_settings_btn = sim_settings
        simulate = QtWidgets.QPushButton("▶ Simulate")
        simulate.setToolTip("Open the run window. The model is kept alive after "
                            "building and only re-instantiated when its "
                            "structure changes.")
        simulate.clicked.connect(self._simulate)

        # Live run controls next to Simulate. One button doubles as Run / Pause
        # / Resume: it starts a run straight from the toolbar (build + initialise
        # + streaming run, no window needed), then pauses/resumes the live run.
        # A streaming run keeps advancing on the host even with every window
        # closed, so these stay meaningful. Driven by the pump's phaseChanged.
        self._starting = False              # a toolbar-launched run is building
        self._run_worker = None             # keep the launch worker alive
        self._runctl_btn = QtWidgets.QPushButton("▶ Run")
        self._runctl_btn.setToolTip(
            "Run the model (build if needed, then simulate) — no window needed. "
            "While running this pauses/resumes the live run.")
        self._runctl_btn.clicked.connect(self._on_runctl)
        self._stop_btn = QtWidgets.QPushButton("⏹ Stop")
        self._stop_btn.setToolTip("Stop the running simulation (the built model "
                                  "stays alive for another run).")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_run)
        self._live.phaseChanged.connect(self._on_run_phase)
        self._live.warningsChanged.connect(self._refresh_component_warnings)

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
        bar.addWidget(media_btn)
        bar.addWidget(sim_settings)
        bar.addWidget(simulate)
        bar.addWidget(self._runctl_btn)
        bar.addWidget(self._stop_btn)
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

        plots_menu = self.menuBar().addMenu("&Plots")
        add_table = plots_menu.addAction("Add &Table")
        add_table.setToolTip("Add a live variable table to the canvas.")
        add_table.triggered.connect(lambda: self._add_object("table"))
        add_ts = plots_menu.addAction("Add T&imeseries")
        add_ts.setToolTip("Add a live timeseries chart to the canvas.")
        add_ts.triggered.connect(lambda: self._add_object("timeseries"))
        add_bar = plots_menu.addAction("Add &Bar chart")
        add_bar.setToolTip("Add a bar chart of each variable's latest value.")
        add_bar.triggered.connect(lambda: self._add_object("bar"))
        add_pie = plots_menu.addAction("Add &Pie chart")
        add_pie.setToolTip("Add a pie chart of each variable's latest value.")
        add_pie.triggered.connect(lambda: self._add_object("pie"))

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
        self._media = {"Hydrogen": default_media_spec("Hydrogen")}
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
        # v2 persists the media table; v1 files don't, so synthesise one from the
        # loaded components' medium names (fast defaults) for backward compat.
        self._media = data.get("media") or self._media_from_nodes()
        self._path = path
        add_recent_file(path)
        self._show_editor()
        self._update_title()
        # Frame the loaded contents. Deferred so the editor view is shown and
        # laid out first (fitInView needs the final viewport size).
        QtCore.QTimer.singleShot(0, self._canvas.fit_view)

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
        project = make_project(self._canvas.to_project(), self._sim_options,
                               self._media)
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

    # --- plot / table objects + live pump ---------------------------------- #
    def _add_object(self, kind: str):
        self._show_editor()
        self._canvas.add_object(kind)

    def _sync_live_objects(self):
        """Re-register the canvas objects with the live pump (called whenever the
        set of objects or their watched variables changes)."""
        self._live.set_contents([o.content for o in self._canvas.overlays()])

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
                key = node.medium or "Hydrogen"
                template["medium"] = key
                # Pull the shared definition from the project media table; a key
                # the table doesn't know yet (e.g. typed straight into a node)
                # gets the fast defaults.
                media.setdefault(
                    key, dict(self._media.get(key) or default_media_spec(key)))
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

    # --- media table ------------------------------------------------------- #
    def _referenced_media_keys(self) -> set[str]:
        """Medium keys currently referenced by canvas components."""
        keys: set[str] = set()
        for node in self._canvas.nodes():
            if hd.component_spec(node.type_name)["needs_medium"]:
                keys.add(node.medium or "Hydrogen")
        return keys

    def _media_from_nodes(self) -> dict:
        """Synthesise a media table from the components' medium names (used to
        migrate v1 projects that predate the persisted table)."""
        keys = self._referenced_media_keys()
        if not keys:
            return {"Hydrogen": default_media_spec("Hydrogen")}
        return {k: default_media_spec(k) for k in sorted(keys)}

    def _edit_media(self):
        # Seed entries for any referenced-but-undefined key so the manager shows
        # everything actually in use.
        used = self._referenced_media_keys()
        for key in used:
            self._media.setdefault(key, default_media_spec(key))
        if not self._media:
            self._media = {"Hydrogen": default_media_spec("Hydrogen")}
        dlg = MediaManagerDialog(self._media, used_keys=used, parent=self)
        if exec_(dlg):
            self._media = dlg.media()
            self._on_canvas_model_changed()
            self._set_status("Media updated.")

    def _on_canvas_model_changed(self):
        """Canvas topology/params changed — resume is no longer valid."""
        if self._session._run_checkpoint is not None or self._session._run_stale:
            self._session.mark_model_stale()
            self._update_runctl()
            if self._session.run_phase in ("paused", "finished", "stopped"):
                self._set_status(
                    "Model changed — use Run to re-initialise (cannot resume).")

    def _edit_sim_settings(self):
        phase = self._session.run_phase
        if phase == "running":
            self._set_status("Pause the simulation before changing settings.")
            return
        nodes = self._canvas.nodes()
        system = self._build_system(nodes) if nodes else None
        live = (phase in ("paused", "finished", "stopped")
                and self._session.built)
        dlg = SimSettingsDialog(
            self._sim_options, system=system, parent=self,
            live_sim_only=live)
        if exec_(dlg):
            self._sim_options = dlg.options()
            if live:
                patch = run_config_patch(self._sim_options)
                self._session.push_run_config(
                    patch, log=lambda m, level="status": self._record_log(m, level))
            self._update_runctl()
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
            log_sink=self._record_log,
        )
        # Show everything recorded so far (e.g. a run started from the toolbar),
        # then mirror new lines into this panel while it is open.
        dlg.prime_log(self._run_log)
        self._log_view = dlg._logs
        try:
            exec_(dlg)
        finally:
            self._log_view = None

    def _record_log(self, message: str, level: str = "status"):
        """Append a build/run log line to the persistent store and, if the
        Simulate window is open, mirror it into that panel.  Always called on
        the GUI thread (worker/live signals are queued)."""
        self._run_log.append((message, level))
        if len(self._run_log) > 5000:        # cap memory on long sessions
            del self._run_log[:len(self._run_log) - 5000]
        if self._log_view is not None:
            self._log_view.add(message, level)

    # --- live run controls ------------------------------------------------- #
    def _update_runctl(self):
        """Sync the Run/Pause/Resume + Stop buttons to the current run phase."""
        phase = self._session.run_phase
        nodes = self._canvas.nodes()
        inst_kw = instantiate_kwargs(self._sim_options)
        system = self._build_system(nodes) if nodes else {}
        can_resume = (bool(nodes)
                      and self._session.can_steering_resume(system, inst_kw))
        # Once the run is actually live, drop any 'starting' state so the button
        # flips straight to Pause (the launch worker's done signal may lag).
        if phase in ("running", "paused"):
            self._starting = False
        # Settings: editable when idle, paused, or finished/stopped (continuable
        # or not — extending stop_time may enable resume).
        if phase == "running":
            self._sim_settings_btn.setEnabled(False)
            self._sim_settings_btn.setToolTip(
                "Pause the simulation to edit live run settings.")
        else:
            self._sim_settings_btn.setEnabled(True)
            self._sim_settings_btn.setToolTip(
                "Edit instantiate / initialise / simulate options.")
        if phase == "running":
            self._runctl_btn.setText("⏸ Pause")
            self._runctl_btn.setEnabled(True)
            self._stop_btn.setEnabled(True)
        elif can_resume and phase == "paused":
            self._runctl_btn.setText("▶ Resume")
            self._runctl_btn.setEnabled(True)
            self._stop_btn.setEnabled(True)
        elif can_resume:
            self._runctl_btn.setText("▶ Resume")
            self._runctl_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
        elif self._starting:
            self._runctl_btn.setText("… Starting")
            self._runctl_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
        else:                                # idle / stale / finished / stopped
            self._runctl_btn.setText("▶ Run")
            self._runctl_btn.setEnabled(True)
            self._stop_btn.setEnabled(phase in ("running", "paused"))

    def _on_run_phase(self, phase: str):
        """React to a streaming run's phase change (driven by the live pump)."""
        self._update_runctl()
        if phase:
            self._set_status(f"Simulation {phase}.")

    def _refresh_component_warnings(self):
        """Push the session's per-component modelling warnings onto the canvas
        nodes (driven by the live pump's ``warningsChanged`` signal), so an
        offending component shows a warning badge and lists its messages."""
        self._canvas.set_component_warnings(self._session.component_warnings())

    def _on_runctl(self):
        """Run when idle/stale, pause when running, resume when checkpoint matches."""
        phase = self._session.run_phase
        nodes = self._canvas.nodes()
        inst_kw = instantiate_kwargs(self._sim_options)
        system = self._build_system(nodes) if nodes else {}
        can_resume = (bool(nodes)
                      and self._session.can_steering_resume(system, inst_kw))
        if phase == "running":
            self._session.pause_run()
        elif can_resume and phase == "paused":
            self._session.resume_run()
        elif can_resume:
            self._continue_run_from_toolbar()
        else:
            self._start_run_from_toolbar()

    def _stop_run(self):
        self._session.stop_run()

    def _start_run_from_toolbar(self):
        """Build (if needed) and launch a streaming run straight from the
        toolbar -- no Simulate window. The heavy build/initialise runs on a
        worker thread so the UI stays responsive."""
        if self._starting:
            return
        if self._session.run_phase == "running":
            return
        nodes = self._canvas.nodes()
        if not nodes:
            self._set_status("Place at least one component before simulating.")
            return
        options = self._sim_options
        run_kw = run_kwargs(options)
        inst_kw = instantiate_kwargs(options)
        try:
            system = self._build_system(nodes)
        except Exception as exc:
            self._set_status(f"Build failed: {type(exc).__name__}: {exc}")
            return
        if (self._session.run_phase == "paused"
                and self._session.can_steering_resume(system, inst_kw)):
            return
        if run_kw.get("stop_time") is None:
            self._set_status("Set a stop_time in ⚙ Settings first — the run is "
                             "driven by it.")
            return
        if (run_kw.get("strategy", {}).get("name") == "fixed"
                and run_kw.get("dt") is None):
            self._set_status("strategy='fixed' needs a dt (set it in ⚙ Settings).")
            return
        init_kw = initialise_kwargs(options)

        def task(log):
            if self._session.run_phase in ("running", "paused"):
                self._session.stop_and_drain(log)
            self._session.ensure_built(system, inst_kw, log)
            self._session.set_run_checkpoint(system, inst_kw)
            return self._session.start_run(init_kw, run_kw, log)

        self._starting = True
        self._update_runctl()
        self._set_status("Building / starting simulation…")
        self._record_log("\n— run launched from the toolbar —", "status")
        worker = _SessionWorker(task)
        worker.logged.connect(self._on_toolbar_log)
        worker.done.connect(self._on_toolbar_run_started)
        worker.failed.connect(self._on_toolbar_run_failed)
        worker.finished.connect(worker.deleteLater)
        self._run_worker = worker
        worker.start()

    def _on_toolbar_log(self, message: str, level: str = "status"):
        self._record_log(message, level)
        self._set_status(message.strip() or level)

    def _on_toolbar_run_started(self, _ack):
        self._starting = False
        self._run_worker = None
        self._set_status("Simulation running — use ⏸/⏹ to steer it.")
        self._update_runctl()

    def _on_toolbar_run_failed(self, kind: str, message: str):
        self._starting = False
        self._run_worker = None
        self._set_status(f"Run failed: {kind}: {message}")
        self._update_runctl()

    def _continue_run_from_toolbar(self):
        """Continue a finished/stopped run without re-initialising."""
        if self._session.run_active or self._starting:
            return
        if not self._session.can_continue:
            return
        self._starting = True
        self._update_runctl()
        self._set_status("Continuing simulation…")
        self._record_log("\n— run continued from toolbar —", "status")

        def task(log):
            self._session.continue_run()
            return "continued"

        # continue_run is fire-and-forget on the host; poll handles phase.
        def done(_):
            self._starting = False
            self._set_status("Simulation running — use ⏸/⏹ to steer it.")
            self._update_runctl()

        worker = _SessionWorker(task)
        worker.logged.connect(self._on_toolbar_log)
        worker.done.connect(done)
        worker.failed.connect(self._on_toolbar_run_failed)
        worker.finished.connect(worker.deleteLater)
        self._run_worker = worker
        worker.start()

    def closeEvent(self, event):
        self._live.shutdown()
        self._session.shutdown()
        # Release the plot/table objects' off-screen render widgets and any open
        # Variables windows -- these are top-level widgets that would otherwise
        # keep the Qt event loop alive after this window closes.
        for overlay in self._canvas.overlays():
            overlay.dispose()
        for win in list(getattr(self._canvas, "_var_windows", {}).values()):
            win.close()
        super().closeEvent(event)
        # This is the application's main window: closing it ends the session.
        # Quit explicitly rather than relying on quitOnLastWindowClosed, which
        # off-screen helper widgets (plot render surfaces) can otherwise defeat.
        QtWidgets.QApplication.instance().quit()


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
