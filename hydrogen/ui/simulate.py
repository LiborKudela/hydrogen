"""Run a placed system on a hydrogen host (JSON load), plus the option
descriptors that drive the host's ``instantiate`` / ``initialise`` / ``run``
kwargs.

  * the ``_*_OPTS`` descriptor lists + :func:`default_sim_options` define the
    knobs (and their defaults) the UI persists and replays,
  * :class:`OptionsForm` renders one descriptor list as a typed form,
  * :class:`LogPanel` is the filterable host/run log,
  * :class:`SimulateDialog` ties it together: tabs of options + spec JSON +
    logs, and the run loop that streams host events back.
"""

from __future__ import annotations

import json

import hydrogen as hd

from .qt import QtCore, QtWidgets

__all__ = ["OptionsForm", "LogPanel", "SimulateDialog", "default_sim_options"]

#: Field descriptors for the Simulate window's option tabs.  Each entry is
#: ``{name, type, default, tip}`` (type in bool / int / float / choice); a blank
#: numeric field is omitted from the call so the host's own default applies.
#: These map 1:1 onto the host-side `instantiate` / `initialise` / `run` kwargs.
_INSTANTIATE_OPTS = [
    {"name": "cse", "type": "bool", "default": True,
     "tip": "Common-subexpression elimination during lambdify."},
    {"name": "enable_blt", "type": "bool", "default": True,
     "tip": "Block-lower-triangular reordering of the equation system."},
    {"name": "enable_var_scaling", "type": "bool", "default": False,
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
]
#: The Simulation tab mixes the stepping strategy, the `stop_time` that drives
#: the run, and the adaptive controller knobs.  The run length is ALWAYS set by
#: `stop_time` (model time), never a step count -- the adaptive strategies pick
#: their own dt and `fixed` just marches `dt` until `stop_time` is reached.
#: `tol_local` / `atol` only apply to richardson.
_SIMULATE_OPTS = [
    {"name": "strategy", "type": "choice", "default": "richardson",
     "choices": ["richardson", "predictor_corrector", "derivative_limit", "fixed"],
     "tip": "Time-stepping rule. 'fixed' needs dt; the others self-adapt."},
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
    {"name": "tol_local", "type": "float", "default": 1e-3,
     "tip": "richardson: local error tolerance (relative)."},
    {"name": "atol", "type": "float", "default": 1.0,
     "tip": "richardson: absolute error floor."},
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


def default_sim_options() -> dict:
    return {
        "instantiate": _opts_defaults(_INSTANTIATE_OPTS),
        "initialise": _opts_defaults(_INITIALISE_OPTS),
        "simulate": _opts_defaults(_SIMULATE_OPTS),
    }


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
    """Scrollable run/host log with a level + free-text filter.

    Every line is tagged with a level (``"status"`` for the dialog's own
    progress notes, ``"host"`` for messages streamed from the hydrogen host,
    ``"error"`` for host errors / failures).  The toolbar filters by level
    (combo) and a case-insensitive substring, so host chatter can be isolated
    from the dialog's bookkeeping.
    """

    #: Combo label -> stored level key (None = show every level).
    _LEVELS = {"All": None, "Host": "host", "Errors": "error", "Status": "status"}

    def __init__(self):
        super().__init__()
        self._entries: list[tuple[str, str]] = []   # (level, text)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("show:"))
        self._level = QtWidgets.QComboBox()
        self._level.addItems(list(self._LEVELS))
        self._level.currentTextChanged.connect(self._render)
        bar.addWidget(self._level)
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
        v.addWidget(self._view, 1)

    def _passes(self, level: str, text: str) -> bool:
        want = self._LEVELS.get(self._level.currentText())
        if want is not None and level != want:
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


class SimulateDialog(QtWidgets.QDialog):
    """Assemble the canvas into a system spec and run it on a hydrogen host.

    Instantiation / initialisation / simulation options each get their own tab
    (mapping straight onto the host's `instantiate` / `initialise` / `run`
    kwargs); further tabs show the spec JSON and the filterable host log.
    """

    def __init__(self, system: dict, options: dict | None = None, parent=None):
        super().__init__(parent)
        self._system = system
        self.setWindowTitle("Simulate via hydrogen service")

        outer = QtWidgets.QVBoxLayout(self)
        n = len(system["components"])
        c = len(system["connections"])
        header = QtWidgets.QLabel(
            f"<b>System spec</b> &mdash; {n} component(s), {c} connection(s) "
            f"shipped to a hydrogen host as JSON.")
        header.setWordWrap(True)
        outer.addWidget(header)

        self._inst_form = OptionsForm(_INSTANTIATE_OPTS)
        self._init_form = OptionsForm(_INITIALISE_OPTS)
        self._sim_form = OptionsForm(_SIMULATE_OPTS)
        if options:
            self._inst_form.set_values(options.get("instantiate", {}))
            self._init_form.set_values(options.get("initialise", {}))
            self._sim_form.set_values(options.get("simulate", {}))

        self._json = QtWidgets.QPlainTextEdit()
        self._json.setReadOnly(True)
        self._json.setPlainText(json.dumps(system, indent=2))

        self._logs = LogPanel()

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._scroll(self._inst_form), "Instantiation")
        self._tabs.addTab(self._scroll(self._init_form), "Initialisation")
        self._tabs.addTab(self._scroll(self._sim_form), "Simulation")
        self._tabs.addTab(self._json, "Spec JSON")
        self._tabs.addTab(self._logs, "Logs")
        outer.addWidget(self._tabs, 3)

        bar = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Run on hydrogen service")
        self._run_btn.clicked.connect(self._run)
        bar.addWidget(self._run_btn)
        bar.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        bar.addWidget(close)
        outer.addLayout(bar)
        self.resize(760, 820)

    @staticmethod
    def _scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def options(self) -> dict:
        """The three option groups as plain dicts (for persistence / reuse)."""
        return {
            "instantiate": self._inst_form.values(),
            "initialise": self._init_form.values(),
            "simulate": self._sim_form.values(),
        }

    # --- option assembly ---------------------------------------------------- #
    def _instantiate_kwargs(self) -> dict:
        return self._inst_form.values()

    def _initialise_kwargs(self) -> dict:
        return self._init_form.values()

    def _run_kwargs(self) -> dict:
        sim = self._sim_form.values()
        strat = {"name": sim.pop("strategy", "richardson")}
        for key in _STRATEGY_PARAMS:
            val = sim.pop(key, None)
            if val is not None and strat["name"] == "richardson":
                strat[key] = val
        top = {key: sim.pop(key) for key in list(sim) if key in _RUN_TOPLEVEL}
        # Whatever remains are the adaptive controller knobs.
        kwargs = {"strategy": strat, "adaptive": sim, "stream": False, **top}
        return kwargs

    # --- run loop ----------------------------------------------------------- #
    def _append(self, line: str, level: str = "status"):
        self._logs.add(line, level)
        QtWidgets.QApplication.processEvents()

    def _drain(self, sysp):
        for ev in sysp.poll_events():
            if ev.get("type") == "log":
                self._append(f"  [host] {ev['message']}", "host")
            elif ev.get("type") == "error":
                self._append(f"  [host error] {ev.get('kind')}: {ev.get('message')}",
                             "error")

    def _run(self):
        self._run_btn.setEnabled(False)
        self._logs.clear()
        self._tabs.setCurrentWidget(self._logs)   # surface the log while running
        try:
            self._do_run()
        except Exception as exc:  # surface any host/validation failure in the UI
            self._append(f"\nFAILED: {type(exc).__name__}: {exc}", "error")
        finally:
            self._run_btn.setEnabled(True)

    def _do_run(self):
        inst_kw = self._instantiate_kwargs()
        init_kw = self._initialise_kwargs()
        run_kw = self._run_kwargs()
        if run_kw.get("stop_time") is None:
            self._append("Set a stop_time first — the run is driven by it.")
            return
        if run_kw.get("strategy", {}).get("name") == "fixed" and run_kw.get("dt") is None:
            self._append("strategy='fixed' needs a dt (it marches dt until stop_time).")
            return

        text = json.dumps(self._system)
        self._append("starting hydrogen host…")
        service = hd.start_host(workers=1)
        try:
            sysp = service.load_json(text)
            self._append(f"loaded JSON; instantiating with {inst_kw} …")
            sysp.instantiate(**inst_kw)
            self._drain(sysp)
            self._append(f"initialising with {init_kw} …")
            sysp.initialise(**init_kw)
            self._drain(sysp)
            self._append(f"running with {run_kw} …")
            summary = sysp.run(**run_kw)
            self._drain(sysp)
            self._append(f"\nOK — run summary: {summary}")
            sysp.close()
        finally:
            service.shutdown()
            self._append("host shut down.")
