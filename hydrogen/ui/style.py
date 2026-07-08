"""Visual conventions for the canvas: per-domain node fills, per-kind wire
colours, and the cosmetic rule for which side of a node a port sits on.

These are purely presentational -- wiring *validity* never depends on them
(that is decided by port ``kind`` in :mod:`hydrogen.ui.items`).
"""

from __future__ import annotations

from . import theme
from .qt import QtGui

__all__ = ["domain_color", "kind_color", "port_on_right", "domain_leaf_color"]

#: Pastel fills cycled per domain so canvas nodes are visually grouped.
_PALETTE = ["#cfe8ff", "#d8f5d0", "#ffe3c2", "#f3d4ff",
            "#fff3b0", "#c2f0ef", "#ffd6d6", "#e0e0ff"]
_DOMAIN_COLORS: dict[str, QtGui.QColor] = {}


def domain_color(domain: str) -> QtGui.QColor:
    """A stable pastel fill for a physics domain (assigned on first sight)."""
    if domain not in _DOMAIN_COLORS:
        _DOMAIN_COLORS[domain] = QtGui.QColor(
            _PALETTE[len(_DOMAIN_COLORS) % len(_PALETTE)])
    return _DOMAIN_COLORS[domain]


def domain_leaf_color(domain: str) -> QtGui.QColor:
    """The catalogue-tree text colour for a domain: a pastel derivative kept
    legible against the active theme's backdrop."""
    return theme.catalog_leaf_color(domain_color(domain))


#: One colour per port *kind* (the ``connect()`` discriminator).  Only same-kind
#: ports can be wired, so the wire colour reads off as "what flows here".
_KIND_COLORS = {
    "fluid_phm": "#1f77b4",      # blue   -- fluid (p/h/m_dot)
    "thermal_tq": "#d62728",     # red    -- thermal (T/Q)
    "permeation_pn": "#9467bd",  # purple -- permeation (p_partial/N)
    "signal_real": "#2ca02c",    # green  -- scalar control signal
}
_KIND_FALLBACK = ["#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f"]
_KIND_EXTRA: dict[str, str] = {}


def kind_color(kind: str) -> QtGui.QColor:
    """A stable colour for a port kind (known kinds fixed, others cycled)."""
    if kind in _KIND_COLORS:
        return QtGui.QColor(_KIND_COLORS[kind])
    if kind not in _KIND_EXTRA:
        _KIND_EXTRA[kind] = _KIND_FALLBACK[len(_KIND_EXTRA) % len(_KIND_FALLBACK)]
    return QtGui.QColor(_KIND_EXTRA[kind])


def port_on_right(name: str) -> bool:
    """Cosmetic side-of-the-box rule (wiring validity doesn't depend on it):
    source-ish ports (fluid outlets, signal outputs, leaks) sit on the right.

    Matches ``outlet`` / ``outlet_k`` / ``*_out`` / ``y`` / ``leak`` specifically
    so a substring like the thermal ``wall_outer_k`` port is NOT mistaken for a
    fluid outlet (it stays on the left with the other non-fluid connectors).
    """
    n = name.lower()
    return (n == "y" or n == "outlet" or n.startswith("outlet")
            or n.endswith("_out") or "leak" in n)
