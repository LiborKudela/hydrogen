"""The per-component *Variables* window.

Opened from a placed component's context menu, it shows that component's model
variables in a tree that mirrors the component definition's structure
(sub-models -> groups, value objects -> leaves), each leaf annotated with its
unit and description.  Leaves are a drag source carrying the
:data:`~hydrogen.ui.plots.VARIABLE_MIME` payload, so they can be dropped onto a
Table or Timeseries object on the canvas.

Regex-filtered variables can be aggregated client-side — sum / mean / time
integral etc. — without a model rebuild.  A structural aggregate stores the
filter *regex* (not a frozen variable list), so it re-resolves against the live
model on every run: change a pipe's ``n_segments`` and the aggregate grows or
shrinks to match automatically.  Derived results appear in the list at the
bottom and are draggable like ordinary variables.
"""

from __future__ import annotations

import re
import copy

import numpy as np

from .derived import (
    FORMULA_FUNCS,
    FORMULA_REDUCERS,
    FORMULA_TEMPORAL,
    STRUCTURAL_OPS,
    compile_formula,
    evaluate_formula,
    make_derived_payload,
    resolve_regex_names,
    unit_for_agg,
)
from . import theme
from .plots import encode_variables
from .qt import QtCore, QtGui, QtWidgets, Signal
from .varmeta import variable_tree

__all__ = ["VariablesWindow"]

#: Leaf-kind -> swatch colour (matches the four value-object classes).
_KIND_COLOR = {
    "differential": "#c62828",
    "variable": "#1565c0",
    "input": "#6a1b9a",
    "parameter": "#2e7d32",
    "derived": "#e65100",
}


class _VarTree(QtWidgets.QTreeWidget):
    """Tree whose leaves drag a variable payload; groups are not draggable."""

    PAYLOAD_ROLE = QtCore.Qt.UserRole

    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Variable", "Unit"])
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.header().setStretchLastSection(False)
        self.header().resizeSection(1, 80)

    def mimeData(self, items):  # noqa: N802 (Qt override)
        payloads = [it.data(0, self.PAYLOAD_ROLE) for it in items]
        payloads = [p for p in payloads if p]
        return encode_variables(payloads)


class _DerivedList(QtWidgets.QListWidget):
    """Draggable list of client-side derived variables."""

    changed = Signal()
    PAYLOAD_ROLE = QtCore.Qt.UserRole

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.edit_callback = None            # set by the window: fn(item)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.setToolTip("Derived variables — double-click to edit, drag onto a "
                        "plot. Right-click or press Delete to remove.")

    def _on_double_clicked(self, item):
        if self.edit_callback is not None:
            self.edit_callback(item)

    def _context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        has_sel = bool(self.selectedItems())
        has_any = self.count() > 0
        item = self.itemAt(pos)
        edit = menu.addAction("Edit…")
        edit.setEnabled(item is not None and self.edit_callback is not None)
        edit.triggered.connect(
            lambda _=False, it=item: self.edit_callback(it))
        menu.addSeparator()
        remove = menu.addAction("Remove")
        remove.setEnabled(has_sel)
        remove.triggered.connect(self.remove_selected)
        clear = menu.addAction("Remove all")
        clear.setEnabled(has_any)
        clear.triggered.connect(self.remove_all)
        menu.exec(self.mapToGlobal(pos))

    def replace_payload(self, item, payload: dict):
        item.setText(payload.get("label", payload["full"]))
        item.setData(self.PAYLOAD_ROLE, copy.deepcopy(payload))
        item.setToolTip(payload.get("description", ""))
        self.changed.emit()

    def keyPressEvent(self, event):  # noqa: N802 (Qt override)
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self.selectedItems():
                self.remove_selected()
                return
        super().keyPressEvent(event)

    def remove_selected(self):
        rows = sorted((self.row(it) for it in self.selectedItems()), reverse=True)
        if not rows:
            return
        for r in rows:
            self.takeItem(r)
        self.changed.emit()

    def remove_all(self):
        if self.count() == 0:
            return
        self.clear()
        self.changed.emit()

    def payloads(self) -> list[dict]:
        out: list[dict] = []
        for i in range(self.count()):
            p = self.item(i).data(self.PAYLOAD_ROLE)
            if p:
                out.append(copy.deepcopy(p))
        return out

    def set_payloads(self, payloads: list[dict], *, notify: bool = True):
        self.clear()
        for p in payloads:
            self.add_payload(p, notify=False)
        if notify:
            self.changed.emit()

    def add_payload(self, payload: dict, *, notify: bool = True):
        item = QtWidgets.QListWidgetItem(payload.get("label", payload["full"]))
        item.setData(self.PAYLOAD_ROLE, copy.deepcopy(payload))
        item.setToolTip(payload.get("description", ""))
        item.setForeground(QtGui.QBrush(QtGui.QColor(_KIND_COLOR["derived"])))
        self.addItem(item)
        if notify:
            self.changed.emit()

    def mimeData(self, items):  # noqa: N802
        payloads = [it.data(self.PAYLOAD_ROLE) for it in items]
        payloads = [p for p in payloads if p]
        return encode_variables(payloads)


class _AddVariableDialog(QtWidgets.QDialog):
    """Pick one variable, with a live preview of the highlighted variable's
    full name, unit, kind and description."""

    def __init__(self, comp_id: str, leaves: list[dict], parent=None):
        super().__init__(parent)
        self._leaves = list(leaves)
        self._selected: dict | None = None
        self.setWindowTitle("Add variable")
        self.resize(460, 500)

        lay = QtWidgets.QVBoxLayout(self)
        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter (regex)…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self._filter)

        self._list = QtWidgets.QListWidget()
        self._list.currentItemChanged.connect(self._update_preview)
        self._list.itemDoubleClicked.connect(lambda _it: self.accept())
        lay.addWidget(self._list, 1)

        box = QtWidgets.QGroupBox("Preview")
        bl = QtWidgets.QVBoxLayout(box)
        self._preview = QtWidgets.QLabel()
        self._preview.setWordWrap(True)
        self._preview.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        self._preview.setMinimumHeight(70)
        self._preview.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        bl.addWidget(self._preview)
        lay.addWidget(box)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        lay.addWidget(self._buttons)

        self._populate(self._leaves)

    def _populate(self, leaves: list[dict]):
        self._list.clear()
        for p in leaves:
            item = QtWidgets.QListWidgetItem(p.get("label", p["full"]))
            item.setData(QtCore.Qt.UserRole, p)
            color = _KIND_COLOR.get(p.get("kind", ""), "#455a64")
            item.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)
        else:
            self._update_preview(None, None)

    def _apply_filter(self, text: str):
        needle = text.strip()
        rx = None
        if needle:
            try:
                rx = re.compile(needle, re.IGNORECASE)
            except re.error:
                rx = None
        out = []
        for p in self._leaves:
            hay = f"{p['full']} {p.get('unit', '')} {p.get('description', '')}"
            if not needle:
                out.append(p)
            elif rx is not None and rx.search(hay) is not None:
                out.append(p)
            elif rx is None and needle.lower() in hay.lower():
                out.append(p)
        self._populate(out)

    def _update_preview(self, current, _previous=None):
        p = current.data(QtCore.Qt.UserRole) if current is not None else None
        self._selected = p
        ok = self._buttons.button(QtWidgets.QDialogButtonBox.Ok)
        if not p:
            self._preview.setText("<i>No variable selected.</i>")
            ok.setEnabled(False)
            return
        ok.setEnabled(True)
        unit = f" [{p['unit']}]" if p.get("unit") else ""
        txt = f"<b>{p['full']}</b>{unit}<br><i>{p.get('kind', '')}</i>"
        val = p.get("value")
        if val is not None:
            txt += f"<br>current: {val}"
        if p.get("description"):
            txt += f"<br>{p['description']}"
        self._preview.setText(txt)

    def selected_payload(self) -> dict | None:
        return self._selected


class _AddGroupDialog(QtWidgets.QDialog):
    """Define a regex group, with a live list of the variables it matches."""

    def __init__(self, comp_id: str, leaves: list[dict], *,
                 regex: str = "", scope: str | None = None, parent=None):
        super().__init__(parent)
        self._comp_id = comp_id
        self._leaves = list(leaves)
        self.setWindowTitle("Add group")
        self.resize(460, 500)

        muted = theme.current().muted
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "A group expands to every variable matching this regex and is "
            "reduced with sum(gN), mean(gN), … The match preview updates live."))
        form = QtWidgets.QFormLayout()
        self._regex = QtWidgets.QLineEdit(regex)
        self._regex.setPlaceholderText("regex, e.g. m_dot_leak or wall_\\d+\\.T")
        self._regex.textChanged.connect(self._refresh)
        form.addRow("Regex", self._regex)
        self._scope = QtWidgets.QLineEdit(scope or comp_id)
        self._scope.setToolTip("Only match variables under this component id "
                               "(a dotted path segment). Blank matches all.")
        self._scope.textChanged.connect(self._refresh)
        form.addRow("Scope", self._scope)
        lay.addLayout(form)

        self._count = QtWidgets.QLabel()
        self._count.setStyleSheet(f"color:{muted}; font-size:11px;")
        lay.addWidget(self._count)
        self._list = QtWidgets.QListWidget()
        lay.addWidget(self._list, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._refresh()

    def _refresh(self):
        pattern = self._regex.text().strip()
        scope = self._scope.text().strip() or None
        matched = resolve_regex_names(
            [p["full"] for p in self._leaves], pattern, scope)
        self._list.clear()
        self._list.addItems(matched)
        n = len(matched)
        if n:
            self._count.setText(f"{n} matching variable(s)")
        else:
            self._count.setText(
                "0 matches here — a group can still pick up runtime instances "
                "(e.g. per-segment columns) not shown in this static list.")

    def result(self) -> tuple[str, str]:
        return (self._regex.text().strip(),
                self._scope.text().strip() or self._comp_id)


class _DerivedEditor(QtWidgets.QDialog):
    """Unified, editable builder for a derived variable.

    Every derived variable is a *formula* over aliased inputs:

      * a **variable** (``v1``, ``v2`` ...) — one recorded series, and
      * a **group** (``g1``, ``g2`` ...) — a regex over the component's
        variables that expands to every matching column (re-resolved each run).

    The expression combines them with element-wise math (``sqrt``, ``v1 - v2``),
    instance reducers that collapse a group per time step
    (``sum``/``mean``/``max``/``min``/``std``) and temporal transforms along the
    run's time axis (``integral``/``cumsum``).  Inputs can be added or removed,
    and opening the editor on an existing derived variable pre-fills it so it can
    be edited in place (keeping its id, so plots already using it stay linked).
    """

    def __init__(self, comp_id: str, all_leaves: list[dict] | None = None, *,
                 inputs: list[dict] | None = None, payload: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self._comp_id = comp_id
        self._leaves = list(all_leaves or [])
        self._leaf_by_full = {p["full"]: p for p in self._leaves}
        self._orig_full: str | None = None
        self._inputs: list[dict] = []       # {kind, alias, ...}
        self._vcount = 0
        self._gcount = 0
        self.setWindowTitle("Derived variable")
        self.resize(540, 560)

        muted = theme.current().muted
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Build a derived variable from the inputs below. Double-click a row "
            "to insert its alias into the expression."))

        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Alias", "Kind", "Reference", "Unit"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch)
        self._table.cellDoubleClicked.connect(self._insert_alias)
        self._table.setMaximumHeight(180)
        lay.addWidget(self._table)

        row = QtWidgets.QHBoxLayout()
        add_var = QtWidgets.QPushButton("Add variable…")
        add_var.clicked.connect(self._on_add_variable)
        add_grp = QtWidgets.QPushButton("Add group…")
        add_grp.setToolTip("Add a regex group whose matching columns you reduce "
                           "with sum(g1), mean(g1), max(g1), …")
        add_grp.clicked.connect(self._on_add_group)
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(self._on_remove)
        row.addWidget(add_var)
        row.addWidget(add_grp)
        row.addWidget(rm)
        row.addStretch(1)
        lay.addLayout(row)

        form = QtWidgets.QFormLayout()
        self._expr = QtWidgets.QLineEdit()
        self._expr.setPlaceholderText("e.g. v1 - v2, sum(g1), integral(sum(g1))")
        self._expr.textChanged.connect(self._validate)
        form.addRow("Expression", self._expr)
        self._label = QtWidgets.QLineEdit()
        self._label.setPlaceholderText(f"{comp_id}.derived")
        form.addRow("Label", self._label)
        self._unit = QtWidgets.QLineEdit()
        form.addRow("Unit", self._unit)
        lay.addLayout(form)

        hint = QtWidgets.QLabel(
            "<b>Math:</b> " + ", ".join(sorted(FORMULA_FUNCS)) + "<br>"
            "<b>Reduce a group:</b> " + ", ".join(FORMULA_REDUCERS) + "<br>"
            "<b>Over time:</b> " + ", ".join(FORMULA_TEMPORAL))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{muted}; font-size:11px;")
        lay.addWidget(hint)

        self._msg = QtWidgets.QLabel()
        self._msg.setWordWrap(True)
        self._msg.setStyleSheet("font-size:11px;")
        lay.addWidget(self._msg)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        lay.addWidget(self._buttons)

        if payload is not None:
            self._load_payload(payload)
        else:
            for p in inputs or []:
                self._add_variable_input(p)
        self._validate()

    # --- input model ------------------------------------------------------- #
    def _leaf_for(self, full: str) -> dict:
        return self._leaf_by_full.get(
            full, {"full": full, "label": full, "unit": ""})

    def _add_variable_input(self, payload: dict, alias: str | None = None) -> str:
        if alias is None:
            self._vcount += 1
            alias = f"v{self._vcount}"
        else:
            try:
                self._vcount = max(self._vcount, int(alias[1:]))
            except (ValueError, IndexError):
                pass
        self._inputs.append({
            "kind": "var", "alias": alias,
            "full": payload["full"],
            "label": payload.get("label", payload["full"]),
            "unit": payload.get("unit", ""),
        })
        if not self._unit.text().strip() and payload.get("unit"):
            self._unit.setText(payload["unit"])
        self._refresh_table()
        return alias

    def _add_group_input(self, regex: str, scope: str,
                         alias: str | None = None) -> str:
        if alias is None:
            self._gcount += 1
            alias = f"g{self._gcount}"
        else:
            try:
                self._gcount = max(self._gcount, int(alias[1:]))
            except (ValueError, IndexError):
                pass
        self._inputs.append({
            "kind": "group", "alias": alias,
            "regex": regex, "scope": scope or self._comp_id,
        })
        self._refresh_table()
        return alias

    def _refresh_table(self):
        self._table.setRowCount(len(self._inputs))
        for r, inp in enumerate(self._inputs):
            if inp["kind"] == "var":
                kind, ref, unit = "variable", inp["label"], inp.get("unit", "")
            else:
                kind = "group"
                ref = f"/{inp['regex'] or '.*'}/ in {inp['scope']}"
                unit = ""
            for c, txt in enumerate((inp["alias"], kind, ref, unit)):
                self._table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

    def _insert_alias(self, row: int, _col: int):
        if 0 <= row < len(self._inputs):
            self._expr.insert(self._inputs[row]["alias"])
            self._expr.setFocus()

    def _on_add_variable(self):
        if not self._leaves:
            self._msg.setText("No variables available to add.")
            return
        dlg = _AddVariableDialog(self._comp_id, self._leaves, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        payload = dlg.selected_payload()
        if payload is not None:
            self._add_variable_input(payload)
            self._validate()

    def _on_add_group(self):
        dlg = _AddGroupDialog(self._comp_id, self._leaves, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        regex, scope = dlg.result()
        self._add_group_input(regex, scope)
        self._validate()

    def _on_remove(self):
        rows = sorted((idx.row() for idx in
                       self._table.selectionModel().selectedRows()),
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._inputs):
                self._inputs.pop(r)
        self._refresh_table()
        self._validate()

    # --- validation / result ---------------------------------------------- #
    def _validate(self) -> bool:
        text = self._expr.text().strip()
        ok_btn = self._buttons.button(QtWidgets.QDialogButtonBox.Ok)
        if not self._inputs:
            self._msg.setText("Add at least one input.")
            ok_btn.setEnabled(False)
            return False
        if not text:
            self._msg.setText("")
            ok_btn.setEnabled(False)
            return False
        try:
            code = compile_formula(text)
            env = {inp["alias"]: (np.ones((3, 2)) if inp["kind"] == "group"
                                  else np.ones(3))
                   for inp in self._inputs}
            y = np.asarray(evaluate_formula(code, env, time=np.arange(3.0)),
                           dtype=float)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self._msg.setText(f"<span style='color:#c62828'>{exc}</span>")
            ok_btn.setEnabled(False)
            return False
        if y.ndim > 1:
            self._msg.setText(
                "<span style='color:#c62828'>expression must reduce to one "
                "series — reduce groups with sum(g1), mean(g1), …</span>")
            ok_btn.setEnabled(False)
            return False
        self._msg.setText("<span style='color:#2e7d32'>OK</span>")
        ok_btn.setEnabled(True)
        return True

    def _load_payload(self, payload: dict):
        self._orig_full = payload.get("full")
        self._label.setText(payload.get("label", ""))
        self._unit.setText(payload.get("unit", ""))
        agg = payload.get("agg") or {}
        axis = agg.get("axis", "instances")
        op = agg.get("op", "sum")
        if axis == "formula":
            for alias, full in (agg.get("variables") or {}).items():
                self._add_variable_input(self._leaf_for(full), alias=alias)
            for alias, g in (agg.get("groups") or {}).items():
                self._add_group_input(g.get("regex", ""),
                                      g.get("scope") or self._comp_id,
                                      alias=alias)
            self._expr.setText(agg.get("expr", ""))
        elif axis == "time":
            srcs = agg.get("sources") or []
            alias = self._add_variable_input(
                self._leaf_for(srcs[0] if srcs else ""))
            self._expr.setText(f"{op}({alias})")
        else:  # legacy structural
            regex = agg.get("regex")
            if regex is not None:
                alias = self._add_group_input(
                    regex, agg.get("scope") or self._comp_id)
                self._expr.setText(f"{op}({alias})")
            else:
                aliases = [self._add_variable_input(self._leaf_for(s))
                           for s in agg.get("sources") or []]
                if op == "mean" and aliases:
                    self._expr.setText(
                        "(" + " + ".join(aliases) + f") / {len(aliases)}")
                else:
                    self._expr.setText(" + ".join(aliases))

    def payload(self) -> dict | None:
        if not self._validate():
            return None
        expr = self._expr.text().strip()
        variables = {inp["alias"]: inp["full"]
                     for inp in self._inputs if inp["kind"] == "var"}
        groups = {inp["alias"]: {"regex": inp["regex"], "scope": inp["scope"]}
                  for inp in self._inputs if inp["kind"] == "group"}
        default_label = (f"{self._comp_id}.({expr})" if self._comp_id
                         else f"({expr})")
        label = self._label.text().strip() or default_label
        parts = [f"{inp['alias']}=" + (inp["full"] if inp["kind"] == "var"
                 else f"/{inp['regex']}/") for inp in self._inputs]
        desc = f"formula {expr} over " + ", ".join(parts)
        return make_derived_payload(
            op="formula",
            axis="formula",
            label=label,
            unit=self._unit.text().strip(),
            description=desc,
            expr=expr,
            variables=variables or None,
            groups=groups or None,
            full=self._orig_full,
        )


class VariablesWindow(QtWidgets.QDialog):
    """Non-modal browser of component variables (a drag source).

    Takes a list of component descriptors (``{comp_id, type_name, medium,
    params}``); a single-component window shows that component's tree flat, while
    a whole-system window nests each component's tree under a bold group.  Use
    :meth:`for_component` for the common single-component case.
    """

    def __init__(self, components: list[dict], *,
                 scope: str | None = None, title: str | None = None,
                 derived: list[dict] | None = None,
                 on_derived_changed=None, on_derived_edited=None,
                 parent=None):
        super().__init__(parent)
        self._components = list(components)
        # Default scope for aggregates: a component id, or None = whole system.
        self._scope = scope
        self._scope_id = scope or ""
        self._on_derived_changed = on_derived_changed
        # Called with a single updated payload when an existing derived variable
        # is edited in place, so referencing plot objects can update live.
        self._on_derived_edited = on_derived_edited
        self._leaf_total = 0
        single = len(self._components) == 1
        if title is None:
            title = (f"Variables — {self._components[0]['comp_id']}" if single
                     else "Variables — whole system")
        self.setWindowTitle(title)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.resize(460, 640)

        muted = theme.current().muted
        lay = QtWidgets.QVBoxLayout(self)
        if single:
            c = self._components[0]
            head = (f"<b>{c['comp_id']}</b> "
                    f"<span style='color:{muted}'>{c['type_name']}</span><br>")
        else:
            head = (f"<b>Whole system</b> <span style='color:{muted}'>"
                    f"{len(self._components)} component(s)</span><br>")
        header = QtWidgets.QLabel(
            head +
            f"<span style='color:{muted}; font-size:11px'>Drag variables (or "
            "derived aggregates below) onto a Table, Timeseries, Bar chart, or "
            "Pie chart on the canvas.</span>")
        header.setWordWrap(True)
        lay.addWidget(header)

        tool = QtWidgets.QHBoxLayout()
        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText("filter (regex)…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        tool.addWidget(self._filter, 1)

        agg_btn = QtWidgets.QToolButton()
        agg_btn.setText("Aggregate ▾")
        agg_btn.setToolTip(
            "Build a derived variable that reduces every variable matching the "
            "current filter regex. It stores the regex, not the matched list, so "
            "it re-resolves against the model on each run (e.g. tracks n_segments).")
        agg_menu = QtWidgets.QMenu(agg_btn)
        for op in STRUCTURAL_OPS:
            act = agg_menu.addAction(op)
            act.triggered.connect(lambda _=False, o=op: self._aggregate_regex(o))
        agg_btn.setMenu(agg_menu)
        agg_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tool.addWidget(agg_btn)

        formula_btn = QtWidgets.QToolButton()
        formula_btn.setText("New derived…")
        formula_btn.setToolTip(
            "Open the derived-variable editor. Combine variables and regex "
            "groups with math, reducers (sum/mean/…) and time transforms "
            "(integral/cumsum). Any selected variables are pre-added.")
        formula_btn.clicked.connect(self._open_editor)
        tool.addWidget(formula_btn)
        lay.addLayout(tool)

        self._tree = _VarTree()
        self._tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._tree_context_menu)
        lay.addWidget(self._tree, 1)

        lay.addWidget(QtWidgets.QLabel(
            f"<b>Derived</b> <span style='color:{muted}; font-size:11px'>"
            "— drag onto a plot</span>"))
        self._derived = _DerivedList()
        self._derived.setMaximumHeight(120)
        self._derived.changed.connect(self._publish_derived)
        self._derived.edit_callback = self._edit_derived
        lay.addWidget(self._derived)

        self._status = QtWidgets.QLabel()
        self._status.setStyleSheet(f"color:{muted}; font-size:11px;")
        lay.addWidget(self._status)

        self._build()
        if derived:
            self._derived.set_payloads(derived, notify=False)
            self._status.setText(
                f"{self._leaf_total} variable(s); {len(derived)} derived")

    @classmethod
    def for_component(cls, comp_id: str, type_name: str, medium: str | None,
                     params: dict | None, **kwargs) -> "VariablesWindow":
        """Convenience builder for a single-component window."""
        return cls([{"comp_id": comp_id, "type_name": type_name,
                     "medium": medium, "params": params}],
                   scope=comp_id, **kwargs)

    def _publish_derived(self):
        if self._on_derived_changed is not None:
            self._on_derived_changed(self._derived.payloads())

    def _build(self):
        if not self._components:
            self._status.setText("No components in the system.")
            return
        single = len(self._components) == 1
        root = self._tree.invisibleRootItem()
        built = 0
        for c in self._components:
            tree = variable_tree(c["type_name"], c.get("medium"),
                                 c.get("params"))
            if tree is None:
                continue
            built += 1
            if single:
                parent = root
            else:
                parent = QtWidgets.QTreeWidgetItem(root, [c["comp_id"], ""])
                parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsDragEnabled)
                f = parent.font(0)
                f.setBold(True)
                parent.setFont(0, f)
            for node in tree.get("children", []):
                self._add_node(node, parent, c["comp_id"])
        if not built:
            self._status.setText(
                "Could not build the component(s) — check their parameters.")
            return
        self._tree.expandToDepth(0)
        self._leaf_total = self._count_leaves(root)
        self._status.setText(f"{self._leaf_total} variable(s)")

    def _count_leaves(self, item) -> int:
        if item.data(0, _VarTree.PAYLOAD_ROLE):
            return 1
        return sum(self._count_leaves(item.child(i))
                   for i in range(item.childCount()))

    def _add_node(self, node: dict, parent, comp_id: str):
        if node["leaf"]:
            item = QtWidgets.QTreeWidgetItem(parent, [node["name"], node["unit"]])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsDragEnabled)
            full = f"{comp_id}.{node['full']}"
            payload = {
                "full": full,
                "label": f"{comp_id}.{node['path']}",
                "name": node["name"],
                "unit": node["unit"],
                "description": node["description"],
                "kind": node["kind"],
                "value": node["value"],
            }
            item.setData(0, _VarTree.PAYLOAD_ROLE, payload)
            color = _KIND_COLOR.get(node["kind"], "#455a64")
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(color)))
            tip = f"<b>{full}</b>"
            if node["unit"]:
                tip += f" [{node['unit']}]"
            tip += f"<br><i>{node['kind']}</i>"
            if node["description"]:
                tip += f"<br>{node['description']}"
            item.setToolTip(0, tip)
        else:
            item = QtWidgets.QTreeWidgetItem(
                parent, [node["name"], f"({node.get('count', 0)})"])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsDragEnabled)
            f = item.font(0)
            f.setBold(True)
            item.setFont(0, f)
            for child in node.get("children", []):
                self._add_node(child, item, comp_id)

    def _apply_filter(self, text: str):
        needle = text.strip()
        rx = None
        if needle:
            try:
                rx = re.compile(needle, re.IGNORECASE)
            except re.error:
                rx = None
        visible = self._filter_item(self._tree.invisibleRootItem(), needle, rx)
        if needle:
            self._tree.expandAll()
            self._status.setText(
                f"{visible} / {self._leaf_total} variable(s) visible")
        else:
            self._status.setText(f"{self._leaf_total} variable(s)")

    def _filter_item(self, item, needle: str, rx) -> int:
        """Hide non-matching leaves; return count of visible leaves in subtree."""
        payload = item.data(0, _VarTree.PAYLOAD_ROLE)
        if payload:
            hay = f"{payload['full']} {payload['unit']} " \
                  f"{payload['description']}".lower()
            if rx is not None:
                match = rx.search(hay) is not None
            else:
                match = (needle.lower() in hay) if needle else True
            item.setHidden(not match)
            return 1 if match else 0
        visible = 0
        for i in range(item.childCount()):
            visible += self._filter_item(item.child(i), needle, rx)
        if item is not self._tree.invisibleRootItem():
            item.setHidden(needle != "" and visible == 0)
        return visible

    def _all_leaf_payloads(self) -> list[dict]:
        """Every leaf's payload in the tree, regardless of the current filter."""
        out: list[dict] = []

        def walk(item):
            payload = item.data(0, _VarTree.PAYLOAD_ROLE)
            if payload:
                out.append(payload)
            for i in range(item.childCount()):
                walk(item.child(i))

        walk(self._tree.invisibleRootItem())
        return out

    def _aggregate_regex(self, op: str):
        """Aggregate every variable matching the current filter regex.

        Produces a formula whose single group ``g1`` stores the regex (not the
        matched list), reduced by ``op`` -- so it re-resolves against the running
        model each run and can be extended later in the editor.
        """
        pattern = self._filter.text().strip()
        leaves = self._all_leaf_payloads()
        matched_names = set(resolve_regex_names([p["full"] for p in leaves],
                                                pattern, scope=None))
        matched = [p for p in leaves if p["full"] in matched_names]
        if not matched:
            self._status.setText(
                "No variables match the current filter to aggregate.")
            return
        unit = matched[0].get("unit", "")
        shown = pattern or "all"
        prefix = f"{self._scope_id}." if self._scope_id else "system."
        where = self._scope_id or "the system"
        payload = make_derived_payload(
            op="formula",
            axis="formula",
            label=f"{prefix}{op}({shown})",
            unit=unit_for_agg(op, unit),
            description=(f"{op} of {len(matched)} variable(s) matching /{shown}/ "
                        f"in {where} (re-resolved each run)"),
            expr=f"{op}(g1)",
            groups={"g1": {"regex": pattern, "scope": self._scope_id}},
        )
        self._derived.add_payload(payload)
        self._status.setText(
            f"Derived {op} of {len(matched)} variable(s) → double-click to edit, "
            "or drag from the list below.")

    def _selected_leaf_payloads(self) -> list[dict]:
        """Payloads of the currently-selected draggable leaves, in row order."""
        out: list[dict] = []
        for item in self._tree.selectedItems():
            payload = item.data(0, _VarTree.PAYLOAD_ROLE)
            if payload:
                out.append(payload)
        return out

    def _open_editor(self, _checked: bool = False,
                     variables: list[dict] | None = None):
        if variables is None:
            variables = self._selected_leaf_payloads()
        dlg = _DerivedEditor(self._scope_id, self._all_leaf_payloads(),
                             inputs=variables, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        payload = dlg.payload()
        if payload is None:
            return
        self._derived.add_payload(payload)
        self._status.setText("Derived variable added → drag from the list below.")

    def _edit_derived(self, item):
        if item is None:
            return
        current = item.data(_DerivedList.PAYLOAD_ROLE)
        if not current:
            return
        dlg = _DerivedEditor(self._scope_id, self._all_leaf_payloads(),
                             payload=current, parent=self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        payload = dlg.payload()
        if payload is not None:
            self._derived.replace_payload(item, payload)
            if self._on_derived_edited is not None:
                self._on_derived_edited(copy.deepcopy(payload))
            self._status.setText(
                "Derived variable updated (canvas objects refreshed).")

    def _add_temporal_derived(self, op: str, source: dict):
        """Quick temporal transform of one variable, stored as a formula."""
        base = source.get("label") or source["full"]
        if op == "abs":
            label = f"|{base}|"
            expr = "abs(v1)"
        else:
            label = f"{base} ({op})"
            expr = f"{op}(v1)"
        payload = make_derived_payload(
            op="formula",
            axis="formula",
            label=label,
            unit=unit_for_agg(op, source.get("unit", "")),
            description=f"{op} of {source['full']}",
            expr=expr,
            variables={"v1": source["full"]},
        )
        self._derived.add_payload(payload)
        self._status.setText(f"Derived {op} → double-click to edit, or drag "
                             "from the list below.")

    def _tree_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        payload = item.data(0, _VarTree.PAYLOAD_ROLE)
        if not payload:
            return
        menu = QtWidgets.QMenu(self)
        transform = menu.addMenu("Transform")
        for op, label in (("integral", "Time integral (∫)"),
                          ("cumsum", "Cumulative sum"),
                          ("abs", "Absolute value (|x|)")):
            act = transform.addAction(label)
            act.triggered.connect(
                lambda _=False, o=op, p=payload: self._add_temporal_derived(o, p))
        # Open the full editor pre-seeded with the current selection (including
        # the clicked leaf).
        selected = self._selected_leaf_payloads()
        if payload not in selected:
            selected.append(payload)
        editor_act = menu.addAction("New derived from selection…")
        editor_act.triggered.connect(
            lambda _=False, v=selected: self._open_editor(variables=v))
        menu.exec(self._tree.viewport().mapToGlobal(pos))
