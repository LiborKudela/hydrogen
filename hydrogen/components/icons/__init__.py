"""Vector (SVG) icons for components, drawn in a P&ID-diagram style.

These are used by the optional Qt authoring UI (:mod:`hydrogen.ui`) to render a
recognisable symbol on the canvas for each component.  A component *declares*
its icon directly on the class as a ``UI_ICON`` static attribute (a filename in
this directory); the catalog surfaces it as the ``"icon"`` field, and the UI
resolves it to a path with :func:`icon_path`.

:func:`icon_path` returns ``None`` when the file is missing; the UI then falls
back to its generic labelled-box rendering.  This module is pure stdlib (no Qt,
no hydrogen imports) so it is cheap to import from anywhere.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["icon_path", "available_icons"]

_DIR = Path(__file__).resolve().parent


def icon_path(filename: str | None) -> str | None:
    """Absolute path to ``filename`` in this directory, or ``None`` if it isn't
    a shipped icon (``filename`` itself being ``None`` also yields ``None``)."""
    if not filename:
        return None
    path = _DIR / filename
    return str(path) if path.is_file() else None


def available_icons() -> dict[str, str]:
    """All shipped icons as ``{filename: absolute_path}`` (for tooling)."""
    return {p.name: str(p) for p in sorted(_DIR.glob("*.svg"))}
