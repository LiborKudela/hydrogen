"""Run a placed system on a hydrogen host (JSON load), plus the option
descriptors that drive the host's ``instantiate`` / ``initialise`` / ``run``
kwargs.

  * the ``_*_OPTS`` descriptor lists + :func:`default_sim_options` define the
    knobs (and their defaults) the UI persists and replays,
  * :class:`OptionsForm` renders one descriptor list as a typed form,
  * :class:`LogPanel` is the filterable host/run log,
  * :class:`SimSettingsDialog` edits the three option groups (+ spec JSON
    preview) -- the *settings*, split out from the run action,
  * :class:`SimulateDialog` is the *run window*: it drives a long-lived
    :class:`~hydrogen.ui.session.SimulationSession`, building the model only
    when its structure changed and reusing it otherwise.
"""

from __future__ import annotations

import json

from . import theme
from .qt import QtCore, QtGui, QtWidgets, Signal
from .session import SimulationSession

__all__ = [
    "OptionsForm", "LogPanel", "SimSettingsDialog", "SimulateDialog",
    "default_sim_options", "instantiate_kwargs", "initialise_kwargs",
    "run_kwargs", "run_config_patch", "LIVE_SIM_FIELDS",
]

#: Field descriptors for the Simulate window's option tabs.  Each entry is
#: ``{name, type, default, tip}`` (type in bool / int / float / choice); a blank
#: numeric field is omitted from the call so the host's own default applies.
#: These map 1:1 onto the host-side `instantiate` / `initialise` / `run` kwargs.
_INSTANTIATE_OPTS = [
    {"name": "cse", "type": "bool", "default": True,
     "tip": "Common-subexpression elimination during lambdify."},
    {"name": "enable_blt", "type": "bool", "default": True,
     "tip": "Block-lower-triangular reordering of the equation system."},
    {"name": "enable_var_scaling", "type": "bool", "default": True,
     "tip": "Scale variables before the Newton solve."},
    {"name": "max_remove_trival_passes", "type": "int", "default": 1,
     "tip": "Passes of trivial-equation elimination."},
    {"name": "max_remove_duplicate_passes", "type": "int", "default": 5,
     "tip": "Passes of duplicate-equation elimination."},
    {"name": "max_remove_linear_block_passes", "type": "int", "default": 3,
     "tip": "Passes of multi-var linear-block elimination."},
]
_INITIALISE_OPTS = [
    {"name": "n", "type": "int", "default": 1,
     "tip": "Number of initial solves (continuation steps to t=0)."},
    {"name": "relaxation", "type": "float", "default": 1.0,
     "tip": "Newton step damping factor (<= 1 for stiff start-ups)."},
    {"name": "tol", "type": "float", "default": 1e-6,
     "tip": "Newton convergence tolerance."},
    {"name": "max_iter", "type": "int", "default": 200,
     "tip": "Max Newton iterations."},
    {"name": "line_search", "type": "bool", "default": False,
     "tip": "Feasibility-guarded backtracking line search (damps Newton "
            "overshoots into infeasible thermodynamic states)."},
]
#: The Simulation tab mixes the stepping strategy, the `stop_time` that drives
#: the run, and the adaptive controller knobs.  The run length is ALWAYS set by
#: `stop_time` (model time), never a step count -- the adaptive strategies pick
#: their own dt and `fixed` just marches `dt` until `stop_time` is reached.
#: `tol_local` / `atol` only apply to richardson.
_SIMULATE_OPTS = [
    {"name": "strategy", "type": "choice", "default": "richardson",
     "choices": ["richardson", "tr_bdf2", "predictor_corrector",
                 "derivative_limit", "fixed"],
     "tip": "Time-stepping rule. 'tr_bdf2' is L-stable (robust on stiff "
            "transients); 'fixed' needs dt; the others self-adapt."},
    {"name": "stop_time", "type": "float", "default": 1.0,
     "tip": "Integrate until model time reaches this [s]. Drives the run."},
    {"name": "dt", "type": "float", "default": None,
     "tip": "Fixed step (strategy=fixed) or first adaptive target [s]."},
    {"name": "dt_start", "type": "float", "default": 1e-4,
     "tip": "First adaptive dt target [s]."},
    {"name": "dt_min", "type": "float", "default": 1e-9,
     "tip": "Hard floor on adaptive dt [s]."},
    {"name": "dt_max", "type": "float", "default": 1.0,
     "tip": "Hard ceiling on adaptive dt [s]."},
    {"name": "grow", "type": "float", "default": 1.5,
     "tip": "dt growth factor on easy steps."},
    {"name": "shrink", "type": "float", "default": 0.5,
     "tip": "dt shrink factor on a rejected step."},
    {"name": "max_retries", "type": "int", "default": 20,
     "tip": "Max rejection/retry iterations within one step."},
    {"name": "relaxation", "type": "float", "default": 1.0,
     "tip": "Newton step damping per internal solve."},
    {"name": "tol", "type": "float", "default": 1e-6,
     "tip": "Newton tolerance per internal solve."},
    {"name": "max_iter", "type": "int", "default": 200,
     "tip": "Max Newton iterations per internal solve."},
    {"name": "line_search", "type": "bool", "default": False,
     "tip": "Feasibility-guarded backtracking line search per internal solve "
            "(damps Newton overshoots into infeasible thermodynamic states)."},
    {"name": "tol_local", "type": "float", "default": 1e-3,
     "tip": "richardson / tr_bdf2: local error tolerance (relative)."},
    {"name": "atol", "type": "float", "default": 1.0,
     "tip": "richardson / tr_bdf2: absolute error floor."},
]
#: run() top-level kwargs vs. the `adaptive=` controller dict (everything else).
_RUN_TOPLEVEL = {"stop_time", "dt"}
_STRATEGY_PARAMS = {"tol_local", "atol"}


def _opts_defaults(fields: list[dict]) -> dict:
    """Default values for an option descriptor list (skipping blank numerics)."""
    out: dict = {}
    for f in fields:
        d = f.get("default")
        if f["type"] == "bool":
            out[f["name"]] = bool(d)
        elif f["type"] == "choice":
            out[f["name"]] = str(d)
        elif d is not None:
            out[f["name"]] = d
    return out


#: Simulate options that can be pushed mid-run / after finish (step boundaries).
LIVE_SIM_FIELDS = frozenset({
    "stop_time", "dt_max", "dt_min", "grow", "shrink", "max_retries",
    "relaxation", "tol", "max_iter", "line_search", "tol_local", "atol",
})


def run_config_patch(options: dict) -> dict:
    """Build :meth:`SystemProxy.update_run_config` kwargs from simulate options."""
    sim = dict(options.get("simulate", {}))
    patch: dict = {}
    if "stop_time" in sim and sim["stop_time"] is not None:
        patch["stop_time"] = float(sim["stop_time"])
    adaptive = {}
    for key in (
        "dt_min", "dt_max", "grow", "shrink", "max_retries",
        "relaxation", "tol", "max_iter", "line_search",
    ):
        if key in sim and sim[key] is not None:
            adaptive[key] = sim[key]
    if adaptive:
        patch["adaptive"] = adaptive
    strat_name = sim.get("strategy")
    strat_extra = {}
    for key in ("tol_local", "atol"):
        if key in sim and sim[key] is not None:
            strat_extra[key] = sim[key]
    if strat_name in ("richardson", "tr_bdf2") and strat_extra:
        patch["strategy"] = {"name": strat_name, **strat_extra}
    elif strat_extra:
        patch.update(strat_extra)
    return patch


def default_sim_options() -> dict:
    return {
        "instantiate": _opts_defaults(_INSTANTIATE_OPTS),
        "initialise": _opts_defaults(_INITIALISE_OPTS),
        "simulate": _opts_defaults(_SIMULATE_OPTS),
    }


# --------------------------------------------------------------------------- #
# Option dicts (as persisted) -> host call kwargs.
# --------------------------------------------------------------------------- #
def instantiate_kwargs(options: dict) -> dict:
    return dict(options.get("instantiate", {}))


def initialise_kwargs(options: dict) -> dict:
    return dict(options.get("initialise", {}))


def run_kwargs(options: dict) -> dict:
    """Translate the ``simulate`` option group into :meth:`SystemProxy.run`
    kwargs: split the stepping strategy + its params from the run-length
    (``stop_time`` / ``dt``) and the adaptive controller knobs."""
    sim = dict(options.get("simulate", {}))
    strat = {"name": sim.pop("strategy", "richardson")}
    for key in _STRATEGY_PARAMS:
        val = sim.pop(key, None)
        if val is not None and strat["name"] in ("richardson", "tr_bdf2"):
            strat[key] = val
    top = {key: sim.pop(key) for key in list(sim) if key in _RUN_TOPLEVEL}
    # Whatever remains are the adaptive controller knobs.
    return {"strategy": strat, "adaptive": sim, "stream": False, **top}


class OptionsForm(QtWidgets.QWidget):
    """A small typed form over a list of option descriptors.

    ``values()`` returns a dict of name -> value; a blank numeric field is
    omitted so the host's own default for that kwarg applies.
    """

    def __init__(self, fields: list[dict]):
        super().__init__()
        self._editors: dict[str, tuple] = {}
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        for f in fields:
            name, ftype, default = f["name"], f["type"], f.get("default")
            if ftype == "bool":
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(default))
            elif ftype == "choice":
                w = QtWidgets.QComboBox()
                w.addItems([str(c) for c in f["choices"]])
                if default is not None:
                    w.setCurrentText(str(default))
            else:
                w = QtWidgets.QLineEdit("" if default is None else repr(default)
                                       if isinstance(default, float) else str(default))
                w.setPlaceholderText(ftype)
            w.setToolTip(f.get("tip", ""))
            form.addRow(name, w)
            self._editors[name] = (ftype, w)

    def field(self, name: str):
        return self._editors[name][1]

    def set_values(self, values: dict):
        for name, (ftype, w) in self._editors.items():
            if name not in values:
                continue
            val = values[name]
            if ftype == "bool":
                w.setChecked(bool(val))
            elif ftype == "choice":
                if val is not None:
                    w.setCurrentText(str(val))
            else:
                w.setText("" if val is None else str(val))

    def values(self) -> dict:
        out: dict = {}
        for name, (ftype, w) in self._editors.items():
            if ftype == "bool":
                out[name] = w.isChecked()
            elif ftype == "choice":
                out[name] = w.currentText()
            else:
                txt = w.text().strip()
                if not txt:
                    continue  # blank -> use the host default
                try:
                    out[name] = int(txt) if ftype == "int" else float(txt)
                except ValueError:
                    continue
        return out


class LogPanel(QtWidgets.QWidget):
    """Scrollable run/host log with per-level toggles + a free-text filter.

    Every line is tagged with a level (``"status"`` for the dialog's own
    progress notes, ``"host"`` for messages streamed from the hydrogen host,
    ``"warning"`` for host warnings, ``"error"`` for host errors / failures).
    Each level has its own checkbox so any combination can be shown or hidden
    independently -- e.g. keep progress + host output but mute warnings -- and
    a case-insensitive substring narrows things further.
    """

    #: (checkbox label, stored level key), in display order.
    _LEVELS = [
        ("Progress", "status"),
        ("Host", "host"),
        ("Warnings", "warning"),
        ("Errors", "error"),
    ]

    def __init__(self):
        super().__init__()
        self._entries: list[tuple[str, str]] = []   # (level, text)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("show:"))
        self._checks: dict[str, QtWidgets.QCheckBox] = {}
        for label, key in self._LEVELS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._render)
            bar.addWidget(cb)
            self._checks[key] = cb
        self._text = QtWidgets.QLineEdit()
        self._text.setPlaceholderText("filter text…")
        self._text.setClearButtonEnabled(True)
        self._text.textChanged.connect(self._render)
        bar.addWidget(self._text, 1)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.clear)
        bar.addWidget(clear)
        v.addLayout(bar)

        self._view = QtWidgets.QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(10000)   # cap memory on long runs
        # Host output includes ASCII-aligned tables (e.g. the instantiation
        # timing breakdown), which only line up in a fixed-width font.
        self._view.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self._view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        v.addWidget(self._view, 1)

    def _passes(self, level: str, text: str) -> bool:
        cb = self._checks.get(level)
        if cb is not None and not cb.isChecked():
            return False
        needle = self._text.text().strip().lower()
        return needle in text.lower() if needle else True

    def add(self, text: str, level: str = "status"):
        self._entries.append((level, text))
        if self._passes(level, text):
            self._view.appendPlainText(text)
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def clear(self):
        self._entries.clear()
        self._view.clear()

    def _render(self, *_):
        self._view.clear()
        for level, text in self._entries:
            if self._passes(level, text):
                self._view.appendPlainText(text)


def _scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


class SimSettingsDialog(QtWidgets.QDialog):
    """Edit the simulation *settings* -- the instantiate / initialise / simulate
    option groups -- separately from launching a run.

    Each group gets its own tab (mapping straight onto the host's
    `instantiate` / `initialise` / `run` kwargs); a final tab previews the spec
    JSON that would be shipped.  ``options()`` returns the edited groups for the
    caller to persist.
    """

    def __init__(self, options: dict | None = None, system: dict | None = None,
                 parent=None, *, live_sim_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Simulation settings")
        self._live_sim_only = live_sim_only

        outer = QtWidgets.QVBoxLayout(self)
        if live_sim_only:
            header = QtWidgets.QLabel(
                "<b>Live simulation controls</b> &mdash; apply at the next step "
                "(or before resuming). Instantiation / initialisation options "
                "are locked while a run is in progress.")
        else:
            header = QtWidgets.QLabel(
                "<b>Simulation settings</b> &mdash; these drive the host's "
                "instantiate / initialise / run calls. They are saved with the "
                "project and reused on every run.")
        header.setWordWrap(True)
        outer.addWidget(header)

        self._inst_form = OptionsForm(_INSTANTIATE_OPTS)
        self._init_form = OptionsForm(_INITIALISE_OPTS)
        self._sim_form = OptionsForm(_SIMULATE_OPTS)
        if options:
            self._inst_form.set_values(options.get("instantiate", {}))
            self._init_form.set_values(options.get("initialise", {}))
            self._sim_form.set_values(options.get("simulate", {}))

        tabs = QtWidgets.QTabWidget()
        self._tabs = tabs
        tabs.addTab(_scroll(self._inst_form), "Instantiation")
        tabs.addTab(_scroll(self._init_form), "Initialisation")
        tabs.addTab(_scroll(self._sim_form), "Simulation")
        if system is not None:
            spec = QtWidgets.QPlainTextEdit()
            spec.setReadOnly(True)
            spec.setPlainText(json.dumps(system, indent=2))
            tabs.addTab(spec, "Spec JSON")
        outer.addWidget(tabs, 1)

        if live_sim_only:
            tabs.setTabEnabled(0, False)
            tabs.setTabEnabled(1, False)
            if tabs.count() > 3:
                tabs.setTabEnabled(3, False)
            tabs.setCurrentIndex(2)
            for name, (ftype, w) in self._sim_form._editors.items():
                ro = name not in LIVE_SIM_FIELDS
                w.setEnabled(not ro)
                if ro:
                    w.setToolTip(w.toolTip() + " (locked during a live run)")

        note = QtWidgets.QLabel(
            "Changing an <b>instantiation</b> option re-compiles the model on "
            "the next run; initialise / simulate options apply without a "
            "rebuild.")
        if live_sim_only:
            note.setText(
                "Only the highlighted <b>simulation</b> knobs are pushed to the "
                "host; stop the run to edit instantiation / strategy.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.current().muted};")
        outer.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.resize(620, 720)

    def options(self) -> dict:
        """The three option groups as plain dicts (for persistence / reuse)."""
        return {
            "instantiate": self._inst_form.values(),
            "initialise": self._init_form.values(),
            "simulate": self._sim_form.values(),
        }


class _SessionWorker(QtCore.QThread):
    """Runs one blocking session operation off the GUI thread.

    The hydrogen host does the heavy compute in its *own* process, but the
    client calls (``load_json`` / ``instantiate`` / ``run``) block the calling
    thread until the host replies. Calling them on the GUI thread freezes the
    event loop -- so this thread makes the call (mostly parked in ``recv``, which
    releases the GIL) and marshals log lines, the result and any error back to
    the GUI thread via queued signals.
    """

    logged = Signal(str, str)     # (message, level) -> LogPanel
    done = Signal(object)         # the operation's return value
    failed = Signal(str, str)     # (exception type name, message)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn             # fn(log) -> result, where log(message, level)

    def run(self):
        try:
            result = self._fn(lambda m, level="status": self.logged.emit(m, level))
        except Exception as exc:                       # report, don't crash the thread
            self.failed.emit(type(exc).__name__, str(exc))
        else:
            self.done.emit(result)


class SimulateDialog(QtWidgets.QDialog):
    """The *run window*: build / run the canvas against a long-lived host model.

    It does not own the settings (those live in :class:`SimSettingsDialog`); it
    pulls a fresh system spec + the saved options through callables each time,
    and drives a :class:`~hydrogen.ui.session.SimulationSession` that keeps the
    instantiated model alive between runs -- re-instantiating only when the
    structure changed.
    """

    def __init__(self, session: SimulationSession, build_system, get_options,
                 parent=None, log_sink=None, on_run_started=None):
        super().__init__(parent)
        self._session = session
        self._build_system = build_system   # () -> system spec dict
        self._get_options = get_options     # () -> persisted options dict
        self._on_run_started = on_run_started
        # Optional shared recorder: when set, every log line is routed through
        # it (so the run output is persisted and visible even for runs launched
        # from the toolbar, not this window).  It mirrors back into this panel.
        self._log_sink = log_sink
        self._worker: _SessionWorker | None = None
        self._op_label = "build"            # current operation, for the cancel label
        self._cancelling = False            # an abort is in flight
        self._close_after = False           # close the window once the worker stops
        # Polls the host's streamed log events (`instantiate` / `run` progress)
        # onto the GUI thread while a worker thread is blocked in a host call,
        # so the log fills continuously instead of in one burst at the end.
        self._log_pump = QtCore.QTimer(self)
        self._log_pump.setInterval(120)
        self._log_pump.timeout.connect(self._pump_logs)
        self.setWindowTitle("Simulate via hydrogen service")

        outer = QtWidgets.QVBoxLayout(self)
        self._status = QtWidgets.QLabel()
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._logs = LogPanel()
        outer.addWidget(self._logs, 1)

        bar = QtWidgets.QHBoxLayout()
        self._build_label = "Build / update model"
        self._build_btn = QtWidgets.QPushButton(self._build_label)
        self._build_btn.setToolTip(
            "Instantiate the model on the host (only re-compiles if the "
            "structure changed; pushes changed numeric parameters live). While "
            "building this becomes 'Cancel build'.")
        self._build_btn.clicked.connect(self._on_build_clicked)
        bar.addWidget(self._build_btn)

        # Force a full re-instantiate / drop the built model, for when the
        # automatic reuse isn't wanted (e.g. an external code change the
        # structural signature can't see, or to free the host model).
        self._more_btn = QtWidgets.QToolButton()
        self._more_btn.setText("Rebuild ▾")
        self._more_btn.setToolTip(
            "Force a full re-instantiation, or reset (drop) the built model.")
        self._more_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        more_menu = QtWidgets.QMenu(self._more_btn)
        act_force = more_menu.addAction("Force rebuild (re-instantiate)")
        act_force.setToolTip(
            "Re-instantiate the model even if its structure is unchanged.")
        act_force.triggered.connect(self._on_force_build)
        act_init = more_menu.addAction("Force initialise (solve to t=0)")
        act_init.setToolTip(
            "Re-solve the built model to a consistent state at t=0 "
            "(no run).")
        act_init.triggered.connect(self._on_force_init)
        act_reset = more_menu.addAction("Reset model (drop from host)")
        act_reset.setToolTip(
            "Free the built model on the host; the next run instantiates fresh.")
        act_reset.triggered.connect(self._on_reset_model)
        self._more_btn.setMenu(more_menu)
        bar.addWidget(self._more_btn)

        self._run_btn = QtWidgets.QPushButton("Run")
        self._run_btn.setToolTip(
            "Build the model if needed, then initialise and run it.")
        self._run_btn.clicked.connect(self._on_run)
        bar.addWidget(self._run_btn)

        bar.addStretch(1)
        self._close_btn = QtWidgets.QPushButton("Close")
        self._close_btn.setToolTip(
            "Close this window. A streaming run keeps advancing on the host "
            "(steer it with the Pause/Stop controls next to Simulate); only a build / "
            "initialise still in progress is cancelled first. The built model "
            "stays alive on the host.")
        self._close_btn.clicked.connect(self._on_close_clicked)
        bar.addWidget(self._close_btn)
        outer.addLayout(bar)
        self.resize(760, 640)
        self._refresh_status()

    # --- helpers ----------------------------------------------------------- #
    def _refresh_status(self):
        if self._session.built:
            self._status.setText(
                "<b>Model status:</b> built and kept alive on the host. A run "
                "reuses it; structural edits trigger a rebuild automatically.")
        else:
            self._status.setText(
                "<b>Model status:</b> not built yet. Build or run to "
                "instantiate it on the host.")

    def _log(self, message: str, level: str = "status"):
        # Route through the shared recorder when present (it mirrors back into
        # this panel); otherwise write straight to the panel.
        if self._log_sink is not None:
            self._log_sink(message, level)
        else:
            self._logs.add(message, level)

    def prime_log(self, entries):
        """Fill the panel with previously-recorded log lines (on open), so a run
        started from the toolbar is already visible here.  Written straight to
        the panel to avoid re-recording them in the shared store."""
        for message, level in entries:
            self._logs.add(message, level)

    def _busy(self, busy: bool):
        # While busy the Build button turns into a Cancel control; Run and the
        # rebuild/reset menu are disabled (only one operation at a time).
        self._run_btn.setEnabled(not busy)
        self._more_btn.setEnabled(not busy)
        if busy:
            self._build_btn.setText(f"■ Cancel {self._op_label}")
            self._build_btn.setToolTip(
                f"Cancel the {self._op_label} in progress (tears down the host).")
        else:
            self._build_btn.setText(self._build_label)
            self._build_btn.setEnabled(True)
            self._build_btn.setToolTip(
                "Instantiate the model on the host (only re-compiles if the "
                "structure changed; pushes changed numeric parameters live). "
                "While building this becomes 'Cancel build'.")

    def _is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # --- worker plumbing --------------------------------------------------- #
    def _start_worker(self, task, on_done, op_label: str):
        """Run ``task(log)`` on a background thread (so the host calls don't
        freeze the UI); ``on_done(result)`` runs on the GUI thread on success."""
        self._op_label = op_label
        self._cancelling = False
        self._busy(True)
        worker = _SessionWorker(task)
        worker.logged.connect(self._log)                       # queued: GUI thread
        worker.done.connect(lambda result: self._worker_finished(on_done, result))
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()
        self._log_pump.start()             # stream host logs while the worker blocks

    def _pump_logs(self):
        """Forward any host log events queued so far onto the GUI thread."""
        self._session.poll_logs(self._log)

    def _on_build_clicked(self):
        """Build, or -- if a build/run is in flight -- cancel it."""
        if self._is_busy():
            self._cancel()
        else:
            self._on_build()

    def _cancel(self):
        """Abort the in-flight operation by tearing down the host; the worker
        then unblocks and finalises via :meth:`_worker_failed`."""
        if not self._is_busy() or self._cancelling:
            return
        self._cancelling = True
        self._build_btn.setEnabled(False)   # avoid a double-cancel while tearing down
        self._log(f"\ncancelling {self._op_label} — tearing down the host …",
                  "status")
        self._session.abort()

    def _after_run_started(self, message: str):
        hook = self._on_run_started
        if hook is not None:
            hook()
        self._log(message)

    def _worker_finished(self, on_done, result):
        if self._cancelling:
            self._log(f"\n{self._op_label} cancelled.", "status")
        else:
            try:
                on_done(result)
            except Exception as exc:
                self._log(f"\nFAILED: {type(exc).__name__}: {exc}", "error")
        self._finalize_worker()

    def _worker_failed(self, kind: str, message: str):
        if self._cancelling:
            self._log(f"\n{self._op_label} cancelled.", "status")
        else:
            self._log(f"\nFAILED: {kind}: {message}", "error")
        self._finalize_worker()

    def _finalize_worker(self):
        self._log_pump.stop()
        self._pump_logs()                  # flush any stragglers since the last tick
        self._worker = None
        self._cancelling = False
        self._refresh_status()
        self._busy(False)
        if self._close_after:
            self._close_after = False
            QtWidgets.QDialog.accept(self)

    # --- actions ----------------------------------------------------------- #
    def _on_build(self):
        try:
            system = self._build_system()
        except Exception as exc:
            self._log(f"\nFAILED building spec: {type(exc).__name__}: {exc}",
                      "error")
            return
        inst_kw = instantiate_kwargs(self._get_options())

        def task(log):
            return self._session.ensure_built(system, inst_kw, log)

        self._start_worker(
            task, lambda outcome: self._log(f"\nready ({outcome})."),
            op_label="build")

    def _on_force_build(self):
        """Re-instantiate the model unconditionally (no reuse)."""
        try:
            system = self._build_system()
        except Exception as exc:
            self._log(f"\nFAILED building spec: {type(exc).__name__}: {exc}",
                      "error")
            return
        inst_kw = instantiate_kwargs(self._get_options())

        def task(log):
            return self._session.force_build(system, inst_kw, log)

        self._start_worker(
            task, lambda outcome: self._log(f"\nready ({outcome})."),
            op_label="rebuild")

    def _on_force_init(self):
        """Re-solve the built model to t=0 without launching a run."""
        if self._session.run_active:
            self._log("Stop the run before re-initialising (use the Stop "
                      "control).", "status")
            return
        if not self._session.built:
            self._log("Build the model first.", "status")
            return
        init_kw = initialise_kwargs(self._get_options())

        def task(log):
            return self._session.initialise(init_kw, log)

        self._start_worker(
            task, lambda outcome: self._log(f"\nready ({outcome})."),
            op_label="initialise")

    def _on_reset_model(self):
        """Drop the built model from the host (next run re-instantiates)."""
        if self._session.run_active:
            self._log("Stop the run before resetting the model "
                      "(use the Stop control).", "status")
            return
        if not self._session.built:
            self._log("No built model to reset.", "status")
            return

        def task(log):
            self._session.reset()
            log("model reset — dropped from the host; the next run "
                "re-instantiates it.", "status")
            return "reset"

        self._start_worker(
            task, lambda _outcome: self._refresh_status(), op_label="reset")

    def _on_run(self):
        if self._session.run_active:
            self._log("A run is already in progress — pause or stop it first "
                      "(the Pause/Stop controls next to Simulate).", "status")
            return
        options = self._get_options()
        run_kw = run_kwargs(options)
        if run_kw.get("stop_time") is None:
            self._log("Set a stop_time first — the run is driven by it.")
            return
        if (run_kw.get("strategy", {}).get("name") == "fixed"
                and run_kw.get("dt") is None):
            self._log("strategy='fixed' needs a dt (it marches dt until "
                      "stop_time).")
            return
        try:
            system = self._build_system()
        except Exception as exc:
            self._log(f"\nFAILED building spec: {type(exc).__name__}: {exc}",
                      "error")
            return
        inst_kw = instantiate_kwargs(options)
        init_kw = initialise_kwargs(options)

        def task(log):
            outcome = self._session.ensure_built(system, inst_kw, log)
            log(f"model {outcome}; starting run …", "status")
            self._session.set_run_checkpoint(system, inst_kw)
            return self._session.start_run(init_kw, run_kw, log)

        self._start_worker(
            task,
            lambda _ack: self._after_run_started(
                "\nrun started — it streams live and keeps running in the "
                "background. You can close this window; use the Pause/Stop controls "
                "next to Simulate to pause or stop it."),
            op_label="run")

    # --- lifecycle --------------------------------------------------------- #
    def _on_close_clicked(self):
        self._request_close()

    def _request_close(self):
        """Close now if idle; otherwise cancel the in-flight operation and close
        once the worker has actually stopped (so its slots aren't torn down
        mid-flight)."""
        if self._is_busy():
            self._close_after = True
            self._cancel()
            return
        QtWidgets.QDialog.accept(self)

    def closeEvent(self, event):
        if self._is_busy() or self._close_after:
            event.ignore()           # defer until the worker stops, then close
            self._request_close()
            return
        super().closeEvent(event)

    def reject(self):
        self._request_close()        # Esc / window close -> same as the Close button

    def accept(self):
        self._request_close()
