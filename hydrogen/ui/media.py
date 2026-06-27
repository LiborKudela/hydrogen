"""Project-level media manager: edit the shared CoolProp media table.

A *medium* is a shared, keyed resource -- components reference it by KEY and
several components can share one definition.  That is why a medium's backend /
cache / warnings live HERE (one place, keyed once) rather than on the per-node
properties form: two components referencing the same key must not be able to
disagree on its backend.

Each entry is ``{key: {"fluid", "backend", "scalar_cache_maxsize",
"disable_warnings"}}`` -- exactly what
:func:`hydrogen.serialization.registry.make_medium` consumes.  The ``key`` is
the name a node stores in ``NodeItem.medium``; ``fluid`` is the CoolProp fluid
name, so two media can share a fluid but differ in backend (e.g. a fast tabular
medium for the bulk network and a reference-quality HEOS one for a probe).
"""

from __future__ import annotations

from functools import lru_cache

from .qt import QtCore, QtGui, QtWidgets

__all__ = ["MediaManagerDialog", "default_media_spec", "BACKENDS",
           "PROVIDERS", "fluid_status"]

#: CoolProp backends offered in the manager, fastest engineering-grade first.
#: ``BICUBIC&HEOS`` / ``TTSE&HEOS`` are tabular (≈50-60x faster per property,
#: ~1e-4 accuracy); ``HEOS`` is the full reference equation-of-state solver.
BACKENDS = ["BICUBIC&HEOS", "TTSE&HEOS", "HEOS"]

#: Thermophysical backends.  ``coolprop`` is the reference default; ``feos``
#: uses the feos equation of state (Peng-Robinson built from CoolProp critical
#: constants) for thermodynamics and delegates transport to CoolProp.  The
#: ``Backend`` column only applies to ``coolprop``.
PROVIDERS = ["coolprop", "feos"]


def default_media_spec(fluid: str) -> dict:
    """Media-table entry for a fluid, with the fast tabular defaults.

    Defaults to CoolProp's tabular ``BICUBIC&HEOS`` backend (~50-60x faster per
    property evaluation than full ``HEOS``, engineering-grade ~1e-4 accuracy)
    with a cache big enough for many-segment pipes -- otherwise the default
    100-entry cache thrashes on a >=100-segment pipe and HEOS scales
    super-linearly, making long runs crawl.  ``make_medium`` falls back to plain
    HEOS when the tabular backend can't be built for the fluid.
    """
    return {
        "fluid": fluid,
        "backend": "BICUBIC&HEOS",
        "scalar_cache_maxsize": 1000,
        "disable_warnings": True,
    }


#: Per-state glyph + colour + default tooltip for the row status indicator.
_STATUS_STYLE = {
    "ok":      ("\u2713", "#1b7a31"),   # check
    "bad":     ("\u2717", "#b00020"),   # cross
    "unknown": ("?",      "#8a6d00"),   # amber
}


@lru_cache(maxsize=512)
def fluid_status(fluid: str) -> tuple[str, str]:
    """Validate a CoolProp fluid name -> ``(state, message)``.

    ``state`` is ``"ok"`` (a backend can be built for it), ``"bad"`` (CoolProp
    rejects it), or ``"unknown"`` (CoolProp itself isn't importable, so we can't
    check).  Uses the cheap ``HEOS`` backend to validate -- it's the same name
    CoolProp resolves for the tabular backends, and ``make_medium`` falls back
    to HEOS anyway.  Cached so retyping doesn't rebuild states repeatedly.
    """
    fluid = (fluid or "").strip()
    if not fluid:
        return "bad", "Fluid name is empty."
    try:
        import CoolProp.CoolProp as CP
    except Exception as exc:                       # CoolProp not installed
        return "unknown", f"CoolProp unavailable — can't validate ({exc})."
    try:
        CP.AbstractState("HEOS", fluid)
    except Exception as exc:
        return "bad", f"Not a recognised CoolProp fluid:\n{exc}"
    return "ok", f"'{fluid}' is a valid CoolProp fluid."


class MediaManagerDialog(QtWidgets.QDialog):
    """Edit a project's media table.

    ``media`` is the current table (``{key: spec}``); ``used_keys`` are the keys
    currently referenced by canvas components -- their key cell is locked (so a
    rename can't orphan a node) and they can't be removed.  Call :meth:`media`
    after an accepted exec to get the edited table.
    """

    _COLS = ("Key", "Fluid", "Provider", "Backend", "Cache size",
             "Disable warnings", "")
    _PROVIDER_COL = 2
    _BACKEND_COL = 3
    _CACHE_COL = 4
    _WARN_COL = 5
    _STATUS_COL = 6

    def __init__(self, media: dict | None, used_keys=None, parent=None):
        super().__init__(parent)
        self._used = set(used_keys or ())
        self._result: dict = dict(media or {})
        self._suppress = False        # guard re-entrant itemChanged while we edit
        self.setWindowTitle("Media")

        outer = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Shared fluids referenced by components. <b>Provider</b> picks the "
            "engine: <code>coolprop</code> (reference) or <code>feos</code> "
            "(equation of state; transport via CoolProp). <b>Backend</b> "
            "(CoolProp only) trades speed for accuracy: tabular "
            "<code>BICUBIC&amp;HEOS</code> is ~50-60x faster per property "
            "(≈1e-4 error); <code>HEOS</code> is the full reference solver. "
            "<b>Cache size</b> should exceed a pipe's segment count to avoid "
            "re-computing states every Newton iteration. Editing a medium "
            "re-instantiates the model on the next run.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._table = QtWidgets.QTableWidget(0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(list(self._COLS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for c in range(2, len(self._COLS)):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        status_hdr = self._table.horizontalHeaderItem(self._STATUS_COL)
        if status_hdr is not None:
            status_hdr.setToolTip("CoolProp validity of the Fluid (hover the "
                                  "row's status icon for details).")
        outer.addWidget(self._table, 1)

        for key, spec in (media or {}).items():
            self._append_row(key, spec)
        if self._table.rowCount() == 0:
            self._append_row("Hydrogen", default_media_spec("Hydrogen"))

        bar = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add")
        add.setToolTip("Add a new medium definition.")
        add.clicked.connect(self._on_add)
        self._remove_btn = QtWidgets.QPushButton("Remove")
        self._remove_btn.setToolTip("Remove the selected medium (only if no "
                                    "component uses it).")
        self._remove_btn.clicked.connect(self._on_remove)
        bar.addWidget(add)
        bar.addWidget(self._remove_btn)
        bar.addStretch(1)
        outer.addLayout(bar)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._table.itemSelectionChanged.connect(self._sync_remove_enabled)
        self._table.itemChanged.connect(self._on_item_changed)
        self._sync_remove_enabled()
        self.resize(680, 380)

    # --- rows -------------------------------------------------------------- #
    def _append_row(self, key: str, spec: dict):
        r = self._table.rowCount()
        self._table.insertRow(r)

        key_item = QtWidgets.QTableWidgetItem(key)
        if key in self._used:
            key_item.setFlags(key_item.flags() & ~QtCore.Qt.ItemIsEditable)
            key_item.setToolTip("In use by a component — can't be renamed or "
                                "removed here.")
        self._table.setItem(r, 0, key_item)
        self._table.setItem(r, 1, QtWidgets.QTableWidgetItem(
            spec.get("fluid") or key))

        provider = QtWidgets.QComboBox()
        provider.addItems(PROVIDERS)
        prov = (spec.get("provider") or "coolprop").lower()
        if provider.findText(prov) < 0:
            provider.addItem(prov)
        provider.setCurrentText(prov)
        self._table.setCellWidget(r, self._PROVIDER_COL, provider)

        backend = QtWidgets.QComboBox()
        backend.addItems(BACKENDS)
        b = spec.get("backend") or "HEOS"
        if backend.findText(b) < 0:
            backend.addItem(b)
        backend.setCurrentText(b)
        self._table.setCellWidget(r, self._BACKEND_COL, backend)
        # Backend is CoolProp-only; feos ignores it.
        provider.currentTextChanged.connect(
            lambda _t, bx=backend: bx.setEnabled(_t == "coolprop"))
        backend.setEnabled(prov == "coolprop")

        cache = QtWidgets.QSpinBox()
        cache.setRange(1, 10_000_000)
        cache.setSingleStep(100)
        cache.setValue(int(spec.get("scalar_cache_maxsize") or 100))
        self._table.setCellWidget(r, self._CACHE_COL, cache)

        warn = QtWidgets.QCheckBox()
        warn.setChecked(bool(spec.get("disable_warnings", True)))
        wrap = QtWidgets.QWidget()
        wl = QtWidgets.QHBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setAlignment(QtCore.Qt.AlignCenter)
        wl.addWidget(warn)
        self._table.setCellWidget(r, self._WARN_COL, wrap)

        status = QtWidgets.QTableWidgetItem()
        status.setFlags(QtCore.Qt.ItemIsEnabled)   # read-only, non-selectable
        status.setTextAlignment(QtCore.Qt.AlignCenter)
        font = status.font()
        font.setBold(True)
        status.setFont(font)
        self._table.setItem(r, self._STATUS_COL, status)
        self._validate_row(r)

    # --- fluid validation -------------------------------------------------- #
    def _on_item_changed(self, item):
        # Re-validate a row whenever its Fluid cell is edited (ignore our own
        # programmatic updates to the status cell).
        if not self._suppress and item.column() == 1:
            self._validate_row(item.row())

    def _validate_row(self, r: int):
        fluid_item = self._table.item(r, 1)
        status_item = self._table.item(r, self._STATUS_COL)
        if fluid_item is None or status_item is None:
            return
        state, message = fluid_status(fluid_item.text().strip())
        glyph, color = _STATUS_STYLE[state]
        self._suppress = True
        try:
            status_item.setText(glyph)
            status_item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            status_item.setToolTip(message)
        finally:
            self._suppress = False

    def _on_add(self):
        self._append_row("", default_media_spec("Hydrogen"))
        self._table.editItem(self._table.item(self._table.rowCount() - 1, 0))

    def _on_remove(self):
        r = self._table.currentRow()
        if r < 0:
            return
        key = self._table.item(r, 0).text().strip()
        if key in self._used:
            return
        self._table.removeRow(r)

    def _sync_remove_enabled(self):
        r = self._table.currentRow()
        item = self._table.item(r, 0) if r >= 0 else None
        key = item.text().strip() if item else ""
        self._remove_btn.setEnabled(item is not None and key not in self._used)

    # --- result ------------------------------------------------------------ #
    def _checkbox(self, r: int) -> QtWidgets.QCheckBox:
        return self._table.cellWidget(r, self._WARN_COL).findChild(
            QtWidgets.QCheckBox)

    def _accept(self):
        media: dict = {}
        for r in range(self._table.rowCount()):
            key = self._table.item(r, 0).text().strip()
            if not key:
                self._warn("Every medium needs a non-empty key.")
                return
            if key in media:
                self._warn(f"Duplicate medium key {key!r}.")
                return
            fluid = self._table.item(r, 1).text().strip() or key
            provider = self._table.cellWidget(
                r, self._PROVIDER_COL).currentText().strip() or "coolprop"
            spec = {
                "fluid": fluid,
                "scalar_cache_maxsize": int(
                    self._table.cellWidget(r, self._CACHE_COL).value()),
                "disable_warnings": bool(self._checkbox(r).isChecked()),
            }
            if provider == "feos":
                # feos: thermodynamics from the equation of state, no CoolProp
                # tabular backend to choose.
                spec["provider"] = "feos"
            else:
                spec["backend"] = self._table.cellWidget(
                    r, self._BACKEND_COL).currentText().strip() or "HEOS"
            media[key] = spec
        missing = self._used - set(media)
        if missing:
            self._warn("These media are in use and can't be removed: "
                       + ", ".join(sorted(missing)))
            return
        self._result = media
        self.accept()

    def _warn(self, text: str):
        QtWidgets.QMessageBox.warning(self, "Media", text)

    def media(self) -> dict:
        """The edited media table (valid after an accepted exec)."""
        return self._result
