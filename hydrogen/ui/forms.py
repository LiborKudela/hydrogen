"""Dynamic Qt form widgets driven by a hydrogen ``component_spec``.

Everything here is metadata-driven -- give a field descriptor (as returned by
``hd.component_spec()`` / ``hd.value_object_spec()``) and you get the right
editor:

  * scalars         -> line edits / checkboxes (unit in the label, description tip)
  * enums           -> combo boxes seeded from ``choices``
  * nested objects  -> recursive sub-forms (:class:`ObjectEditor`)
  * abstract object -> a type chooser over ``options``, swapping the sub-form
  * nullable object -> a "(none)" choice
  * lists           -> add / remove element rows (:class:`ListEditor`)
  * relevant_when   -> a row shows/hides as its sibling changes

:class:`FieldsForm` is the entry point; ``.value()`` reads the whole tree back
into the plain dict shape a spec's ``params`` expects.
"""

from __future__ import annotations

from . import theme
from .qt import QtCore, QtWidgets, Signal

__all__ = [
    "field_label",
    "field_tooltip",
    "build_scalar",
    "build_field",
    "set_widget_value",
    "FieldsForm",
    "ObjectEditor",
    "ListEditor",
]


# --------------------------------------------------------------------------- #
# Small helpers for labels / tooltips built from the descriptor.
# --------------------------------------------------------------------------- #
def field_label(field: dict) -> str:
    name = field.get("name", "item")
    unit = field.get("unit")
    star = " *" if field.get("required") else ""
    return f"{name} [{unit}]{star}" if unit else f"{name}{star}"


def field_tooltip(field: dict) -> str:
    bits = []
    if field.get("description"):
        bits.append(field["description"])
    if field.get("choices"):
        bits.append("choices: " + ", ".join(map(str, field["choices"])))
    if field.get("relevant_when"):
        bits.append(f"relevant when: {field['relevant_when']}")
    if not field.get("required", False):
        bits.append("(optional)")
    return "\n".join(bits)


# --------------------------------------------------------------------------- #
# Scalar editors: return (widget, getter, changed_signal_or_None).
# --------------------------------------------------------------------------- #
def build_scalar(field: dict):
    ftype = field.get("type")
    default = field.get("default")

    if ftype == "bool":
        w = QtWidgets.QCheckBox()
        w.setChecked(bool(default))
        return w, w.isChecked, w.stateChanged

    if ftype == "enum":
        w = QtWidgets.QComboBox()
        choices = [str(c) for c in field.get("choices", [])]
        w.addItems(choices)
        if default is not None and str(default) in choices:
            w.setCurrentText(str(default))
        return w, w.currentText, w.currentTextChanged

    # float / integer / string / medium / unknown -> a line edit.  A param with
    # no type annotation comes through as "unknown"; if it carries a unit we
    # treat it as numeric, otherwise as free text.
    numeric = ftype in ("float", "integer") or (
        ftype == "unknown" and field.get("unit") not in (None, "1"))
    w = QtWidgets.QLineEdit("" if default is None else str(default))
    if numeric:
        w.setPlaceholderText(ftype if ftype != "unknown" else "number")

        def get(_int=ftype == "integer"):
            txt = w.text().strip()
            if not txt:
                return None
            try:
                return int(txt) if _int else float(txt)
            except ValueError:
                return None
    else:
        def get():
            txt = w.text().strip()
            return txt or None
    return w, get, w.textChanged


def set_widget_value(editor, val):
    """Set an editor's value (used when applying a preset / restoring a node)."""
    if isinstance(editor, QtWidgets.QCheckBox):
        editor.setChecked(bool(val))
    elif isinstance(editor, QtWidgets.QComboBox):
        if val is not None:
            editor.setCurrentText(str(val))
    elif isinstance(editor, QtWidgets.QLineEdit):
        editor.setText("" if val is None else str(val))
    elif isinstance(editor, ObjectEditor):
        editor.set_value(val)
    elif isinstance(editor, ListEditor):
        editor.set_value(val)


# --------------------------------------------------------------------------- #
# Dispatch: scalar vs nested object vs list.
# --------------------------------------------------------------------------- #
def build_field(field: dict):
    ftype = field.get("type")
    if ftype == "object":
        ed = ObjectEditor(field)
        return ed, ed.value, ed.changed
    if ftype == "list":
        ed = ListEditor(field)
        return ed, ed.value, ed.changed
    return build_scalar(field)


class FieldsForm(QtWidgets.QWidget):
    """A form over a list of field descriptors; ``.value()`` -> dict of values.

    Owns ``relevant_when`` visibility for its own (sibling) scope.
    """

    changed = Signal()

    def __init__(self, fields: list[dict]):
        super().__init__()
        # Each row: (field, [widgets to show/hide], editor widget, getter).
        self._rows: list[tuple] = []
        layout = QtWidgets.QFormLayout(self)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        for f in fields:
            editor, getter, changed = build_field(f)
            tip = field_tooltip(f)
            if isinstance(editor, (ObjectEditor, ListEditor)):
                # Nested objects / lists span the full width inside a titled
                # box so deep nesting stays readable (no compounding indent).
                box = QtWidgets.QGroupBox(field_label(f))
                box.setToolTip(tip)
                bl = QtWidgets.QVBoxLayout(box)
                bl.setContentsMargins(8, 4, 8, 8)
                bl.addWidget(editor)
                layout.addRow(box)
                toggles = [box]
            else:
                label = QtWidgets.QLabel(field_label(f))
                label.setToolTip(tip)
                editor.setToolTip(tip)
                layout.addRow(label, editor)
                toggles = [label, editor]
            self._rows.append((f, toggles, editor, getter))
            if changed is not None:
                changed.connect(self._on_changed)
        self._apply_conditions()

    def _on_changed(self, *_args):
        self._apply_conditions()
        self.changed.emit()

    def _sibling_value(self, name: str):
        for f, _, _, getter in self._rows:
            if f["name"] == name:
                return getter()
        return None

    def _apply_conditions(self):
        for f, toggles, _, _ in self._rows:
            cond = f.get("relevant_when")
            visible = True
            if isinstance(cond, dict):  # {sibling: value | [values]}
                visible = all(
                    self._sibling_value(k)
                    in (v if isinstance(v, (list, tuple)) else [v])
                    for k, v in cond.items()
                )
            # named-predicate conditions (str) are advisory -> always shown.
            for w in toggles:
                w.setVisible(visible)

    def value(self) -> dict:
        return {f["name"]: getter() for f, _, _, getter in self._rows}

    def set_values(self, values: dict):
        """Push values (e.g. from a preset spec) into the editors by name,
        recursing into nested objects (a preset fully defines the object)."""
        for f, _, editor, _ in self._rows:
            if f["name"] in values:
                set_widget_value(editor, values[f["name"]])

    def set_editable(self, enabled: bool):
        for _, _, editor, _ in self._rows:
            editor.setEnabled(enabled)

    def mark_structural(self, names):
        """Colour each row's label by parameter class: red for a *structural*
        parameter (its value changes the equation structure -> needs a model
        rebuild) and green for a *pure* one (a live-updatable numeric knob).

        ``names`` is the set of structural parameter names for this scope.
        """
        c = theme.current()
        for f, toggles, _, _ in self._rows:
            color = c.param_structural if f.get("name") in names else c.param_pure
            head = toggles[0] if toggles else None
            if isinstance(head, QtWidgets.QGroupBox):
                head.setStyleSheet(f"QGroupBox::title {{ color: {color}; }}")
            elif head is not None:
                head.setStyleSheet(f"color: {color};")


class ObjectEditor(QtWidgets.QWidget):
    """Editor for an ``object`` field: concrete ``value_spec``, abstract
    ``options``, and/or nullable (adds a "(none)" choice)."""

    changed = Signal()

    CUSTOM = "Custom (enter manually)"

    def __init__(self, field: dict):
        super().__init__()
        self._field = field
        self._options = field.get("options")  # {concrete_type: {fields,...}} or None
        self._nullable = bool(field.get("nullable")) or not field.get("required", False)
        self._form = None
        self._type = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        self._chooser = QtWidgets.QComboBox()
        names: list[str] = []
        if self._nullable:
            names.append("(none)")
        if self._options:
            names += list(self._options.keys())
        elif field.get("value_spec"):
            names.append(field["value_spec"]["value_type"])
        self._chooser.addItems(names)
        self._chooser.setVisible(self._chooser.count() > 1)
        v.addWidget(self._chooser)

        # A concrete object that advertises `presets` is filled from a choice
        # list ("Custom" lets the user type values in by hand).
        vspec = field.get("value_spec") or {}
        presets = vspec.get("presets") if not self._options else None
        self._preset_map = {p["name"]: p["spec"] for p in (presets or [])}
        self._preset_combo = None
        if self._preset_map:
            self._preset_combo = QtWidgets.QComboBox()
            self._preset_combo.addItems(list(self._preset_map) + [self.CUSTOM])
            v.addWidget(self._preset_combo)

        self._holder = QtWidgets.QWidget()
        self._holder_layout = QtWidgets.QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._holder)

        self._chooser.currentTextChanged.connect(self._rebuild)
        self._rebuild()
        if self._preset_combo is not None:
            self._preset_combo.currentTextChanged.connect(self._apply_preset)
            self._apply_preset(self._preset_combo.currentText())

    def _rebuild(self, *_args):
        if self._form is not None:
            self._form.setParent(None)
            self._form.deleteLater()
            self._form = None
        sel = self._chooser.currentText()
        if sel and sel != "(none)":
            if self._options:
                self._type = sel
                fields = self._options[sel]["fields"]
            else:
                self._type = self._field["value_spec"]["value_type"]
                fields = self._field["value_spec"]["fields"]
            self._form = FieldsForm(fields)
            self._form.changed.connect(self.changed)
            self._holder_layout.addWidget(self._form)
        else:
            self._type = None
        self._relayout()
        self.changed.emit()

    def _apply_preset(self, name: str):
        """Fill (and lock) the form from a named preset; "Custom" frees it."""
        if self._form is None:
            return
        if name == self.CUSTOM or name not in self._preset_map:
            self._form.set_editable(True)
        else:
            self._form.set_values(self._preset_map[name])
            self._form.set_editable(False)
        self.changed.emit()

    def set_value(self, spec):
        """Select the type/preset matching a serialized value spec and fill it."""
        if not isinstance(spec, dict):
            return
        vtype = spec.get("__type__")
        if self._options and vtype in self._options:
            self._chooser.setCurrentText(vtype)
        if self._preset_combo is not None:
            match = next((n for n, s in self._preset_map.items() if s == spec), None)
            self._preset_combo.setCurrentText(match or self.CUSTOM)
            if match is None and self._form is not None:
                self._form.set_values(spec)
                self._form.set_editable(True)
        elif self._form is not None:
            self._form.set_values(spec)

    def _relayout(self):
        """Recompute sizes after swapping the form.

        A nullable object starts on "(none)" with an empty holder, which caches
        a zero size hint; without this the freshly-added sub-form would render
        with zero height.  Invalidate the new subtree's layouts, then activate
        up the ancestor chain so the height reaches the scroll area.
        """
        for child in self._holder.findChildren(QtWidgets.QWidget):
            child_layout = child.layout()
            if child_layout is not None:
                child_layout.invalidate()
        w = self._holder
        while w is not None:
            lay = w.layout()
            if lay is not None:
                lay.invalidate()
                lay.activate()
            w.updateGeometry()
            w = w.parentWidget()

    def value(self):
        if self._type is None or self._form is None:
            return None
        return {"__type__": self._type, **self._form.value()}


class ListEditor(QtWidgets.QWidget):
    """Editor for a ``list`` field: add / remove element rows."""

    changed = Signal()

    def __init__(self, field: dict):
        super().__init__()
        self._item = field.get("item") or {}
        self._entries: list[tuple] = []

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self._rows = QtWidgets.QVBoxLayout()
        v.addLayout(self._rows)
        add = QtWidgets.QPushButton("+ add")
        add.clicked.connect(lambda: self._add())
        v.addWidget(add, alignment=QtCore.Qt.AlignLeft)

        self._add()  # seed one element, like spec_template does

    def _add(self, value=None):
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        item_field = {**self._item, "name": "item", "required": True}
        editor, getter, changed = build_field(item_field)
        box = QtWidgets.QGroupBox(self._item.get("value_type") or self._item.get("type"))
        QtWidgets.QVBoxLayout(box).addWidget(editor)
        h.addWidget(box, 1)

        rm = QtWidgets.QPushButton("\u2715")           # ✕ multiplication x
        rm.setToolTip("Remove this entry")
        rm.setFixedSize(26, 26)
        # The global QPushButton style adds 12px horizontal padding, which on a
        # 26px-wide button clips the glyph out of view; zero it so the ✕ shows.
        rm.setStyleSheet("QPushButton { padding: 0px; }")
        h.addWidget(rm, alignment=QtCore.Qt.AlignTop)
        self._rows.addWidget(row)

        entry = (row, getter)
        self._entries.append(entry)

        def remove():
            self._entries.remove(entry)
            row.setParent(None)
            row.deleteLater()
            self.changed.emit()

        rm.clicked.connect(remove)
        if changed is not None:
            changed.connect(lambda *_a: self.changed.emit())
        if value is not None:
            set_widget_value(editor, value)
        self.changed.emit()

    def set_value(self, values):
        """Rebuild the rows from a serialized list so existing values (e.g. a
        Pipe's ``layers`` with their permeation models) round-trip back into the
        editor instead of falling back to the default seeded element."""
        if not isinstance(values, list):
            return
        while self._entries:                       # drop the current rows
            row, _ = self._entries.pop()
            row.setParent(None)
            row.deleteLater()
        for v in values:
            self._add(v)
        self.changed.emit()

    def value(self):
        return [getter() for _, getter in self._entries]
