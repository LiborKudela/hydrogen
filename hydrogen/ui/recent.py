"""Persistent "recently opened" project list for the editor's home screen.

Backed by :class:`QtCore.QSettings` so the list survives across sessions on
every platform without us owning a config file path.  Entries are absolute,
de-duplicated, most-recent-first, and capped at :data:`MAX_RECENT`.
"""

from __future__ import annotations

import os

from .qt import QtCore

__all__ = [
    "MAX_RECENT",
    "recent_files",
    "add_recent_file",
    "remove_recent_file",
    "clear_recent_files",
]

#: Organisation / application keys for the backing :class:`QSettings` store.
_ORG = "hydrogen"
_APP = "hydrogen-ui"
_KEY = "recent_files"

#: How many entries to keep.
MAX_RECENT = 10


def _settings() -> "QtCore.QSettings":
    return QtCore.QSettings(_ORG, _APP)


def _normalise(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def recent_files() -> list[str]:
    """The stored recent-project paths, most-recent first (may be stale)."""
    value = _settings().value(_KEY, [])
    if isinstance(value, str):          # a single entry round-trips as a str
        value = [value]
    return [str(p) for p in (value or []) if p]


def _store(paths: list[str]) -> None:
    _settings().setValue(_KEY, paths[:MAX_RECENT])


def add_recent_file(path: str) -> None:
    """Promote ``path`` to the front of the recent list (de-duplicated)."""
    if not path:
        return
    norm = _normalise(path)
    files = [f for f in recent_files() if _normalise(f) != norm]
    files.insert(0, norm)
    _store(files)


def remove_recent_file(path: str) -> None:
    """Drop ``path`` from the recent list (e.g. when it no longer exists)."""
    norm = _normalise(path)
    _store([f for f in recent_files() if _normalise(f) != norm])


def clear_recent_files() -> None:
    """Forget every recent entry."""
    _settings().remove(_KEY)
