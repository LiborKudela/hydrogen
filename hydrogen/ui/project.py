"""On-disk format for the editor's own project files (canvas layout + per-node
properties + simulation settings).

The file is a thin envelope around :meth:`Canvas.to_project` output plus the
persisted simulate-dialog options; these helpers keep the format constants and
(de)serialisation in one place so :mod:`hydrogen.ui.app` only does the file IO
and dialogs.
"""

from __future__ import annotations

import hydrogen as hd

__all__ = ["PROJECT_FORMAT", "PROJECT_VERSION", "make_project", "is_project"]

#: Marker for the editor's own save files (canvas + properties + sim settings).
PROJECT_FORMAT = "hydrogen-ui-project"
#: v2 adds the project-level ``media`` table (shared CoolProp fluid definitions).
#: v1 files have no table; the editor synthesises one from the components' medium
#: names on load, so older projects open unchanged.
PROJECT_VERSION = 2


def make_project(canvas_state: dict, sim_options: dict,
                 media: dict | None = None) -> dict:
    """Wrap a canvas state + simulation options + media table into the on-disk
    envelope."""
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "hydrogen_version": hd.__version__,
        "canvas": canvas_state,
        "sim_options": sim_options,
        "media": media or {},
    }


def is_project(data: object) -> bool:
    """True if ``data`` looks like one of our project files."""
    return isinstance(data, dict) and data.get("format") == PROJECT_FORMAT
