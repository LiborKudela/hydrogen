"""Inspecting hydrogen's component catalogue from a UI / tooling perspective.

A UI that lets a user assemble a system needs three machine-readable things,
all of which hydrogen exposes without instantiating anything:

  1. the full *catalogue* of available component types (what can I drop on the
     canvas?),
  2. the same, filtered to one physics *domain* / submodule (e.g. ``control``),
  3. a per-component *spec template*: a fill-in-the-blanks skeleton, plus the
     per-field type info (scalar / enum / list / nested object / medium,
     required-vs-optional, choices, nested value-object structure) a form
     renderer needs.

The catalogue ``type`` strings and the template shape are exactly what go into
a system spec's ``components`` map (see ``to_dict`` / ``from_dict`` and the
``tutorials/h2_permeation_pressurize/system.json`` dump), so a UI can build a
valid spec purely from this metadata and hand it back to hydrogen to load.

Run::

    python3 tutorials/data_structures_info.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running the script directly (no `pip install -e .` required).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydrogen as hd  # noqa: E402


def show(title: str, payload) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(json.dumps(payload, indent=2) if not isinstance(payload, str) else payload)


# ---------------------------------------------------------------------------
# 1. The full catalogue.
# ---------------------------------------------------------------------------
# `component_catalog()` returns one dict per shipped component leaf, keyed for
# a spec by its canonical full `type` (e.g. "hydrogen.control.Ramp").  Every
# entry is plain JSON: type / name / domain / summary / needs_medium /
# parameters / literals.

catalog = hd.component_catalog()
show("1. FULL CATALOGUE -- overview (type, domain, needs_medium)",
     [{"type": c["type"], "domain": c["domain"], "needs_medium": c["needs_medium"]}
      for c in catalog])

print(f"\n{len(catalog)} components across domains: {hd.available_domains()}")

pipe_template = hd.spec_template("hydrogen.thermofluid.Pipe")
show("5a. spec_template('hydrogen.thermofluid.Pipe') -- fill the nulls in", pipe_template)


# ---------------------------------------------------------------------------
# 6. Everything for one component in ONE call.
# ---------------------------------------------------------------------------
# `component_spec(type)` is the single command a form renderer wants: it is a
# catalogue entry (type / domain / summary / needs_medium / parameters /
# literals -- each param with type / unit / description / required / choices /
# relevant_when) *plus* every nested value object expanded in place (a concrete
# object field gains a `value_spec`, an abstract one `options` keyed by concrete
# type, lists recurse into `item`) *plus* an embedded fill-in `template`.  No
# follow-up calls to value_object_spec() needed -- the whole tree, with
# descriptions, comes back at once.

full = hd.component_spec("hydrogen.thermofluid.Pipe")
show("6. component_spec('hydrogen.thermofluid.Pipe') -- full annotated tree", full)

