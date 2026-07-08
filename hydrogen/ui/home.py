"""The editor's landing page: start a new project, open an existing one, or
pick up a recently opened project.

A self-contained, modern-looking widget that only emits intent;
:class:`hydrogen.ui.app` decides what to do with it (load a file, swap to the
editor, ...).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .qt import QtCore, QtGui, QtWidgets, Signal
from .recent import recent_files
from .theme import home_stylesheet
from .thumbnail import render_canvas_thumbnail

__all__ = ["HomeScreen"]

#: Recent-card thumbnail size (logical px); rendered at 2x for crispness.
_THUMB_W = 132
_THUMB_H = 76


class _RecentCard(QtWidgets.QFrame):
    """A clickable, hover-highlighted row for one recent project."""

    clicked = Signal(str)

    def __init__(self, path: str, exists: bool, thumb: QtGui.QPixmap | None = None,
                 parent=None):
        super().__init__(parent)
        self._path = path
        self.setObjectName("recentCard")
        self.setProperty("missing", not exists)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        name = os.path.basename(path) or path
        folder = os.path.dirname(path) or "—"

        thumb_label = QtWidgets.QLabel()
        thumb_label.setObjectName("recentThumb")
        thumb_label.setProperty("missing", not exists)
        thumb_label.setFixedSize(_THUMB_W, _THUMB_H)
        thumb_label.setAlignment(QtCore.Qt.AlignCenter)
        if thumb is not None and not thumb.isNull():
            thumb_label.setPixmap(thumb)
        else:
            thumb_label.setText("missing" if not exists else "empty")

        title = QtWidgets.QLabel(name if exists else f"{name}  ·  missing")
        title.setObjectName("recentTitle")
        sub = QtWidgets.QLabel(folder)
        sub.setObjectName("recentSub")
        sub.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        sub.setWordWrap(True)

        text = QtWidgets.QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        text.addWidget(title)
        text.addWidget(sub)
        text.addStretch(1)

        chevron = QtWidgets.QLabel(">")
        chevron.setObjectName("recentChevron")

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 9, 14, 9)
        row.setSpacing(12)
        row.addWidget(thumb_label)
        row.addLayout(text, 1)
        row.addWidget(chevron)

        self.setToolTip(path)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.rect().contains(
                event.position().toPoint() if hasattr(event, "position")
                else event.pos()):
            self.clicked.emit(self._path)
        super().mouseReleaseEvent(event)


class HomeScreen(QtWidgets.QWidget):
    """Start screen with New / Open actions and a recent-projects list."""

    #: Start a blank project.
    newRequested = Signal()
    #: Open the file-picker dialog.
    openRequested = Signal()
    #: Open a specific recent project (carries its absolute path).
    openRecentRequested = Signal(str)

    def __init__(self, by_type: dict | None = None, parent=None):
        super().__init__(parent)
        self._by_type = by_type or {}
        # path -> (mtime, pixmap); avoids rebuilding the scene on every visit.
        self._thumb_cache: dict[str, tuple[float, QtGui.QPixmap]] = {}
        self.setObjectName("homeScreen")
        self.setStyleSheet(home_stylesheet())

        # A centred fixed-width "card" so the page looks deliberate at any size.
        card = QtWidgets.QWidget()
        card.setObjectName("homeCard")
        card.setMinimumWidth(440)
        card.setMaximumWidth(620)
        cl = QtWidgets.QVBoxLayout(card)
        cl.setContentsMargins(36, 34, 36, 30)
        cl.setSpacing(6)

        brand = QtWidgets.QLabel("hydrogen")
        brand.setObjectName("brand")
        tagline = QtWidgets.QLabel("System editor")
        tagline.setObjectName("tagline")
        subtitle = QtWidgets.QLabel(
            "Build, wire and simulate component networks.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        cl.addWidget(brand)
        cl.addWidget(tagline)
        cl.addWidget(subtitle)
        cl.addSpacing(18)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(10)
        new_btn = QtWidgets.QPushButton("New project")
        new_btn.setObjectName("primaryBtn")
        new_btn.setCursor(QtCore.Qt.PointingHandCursor)
        new_btn.clicked.connect(self.newRequested)
        open_btn = QtWidgets.QPushButton("Open project…")
        open_btn.setObjectName("secondaryBtn")
        open_btn.setCursor(QtCore.Qt.PointingHandCursor)
        open_btn.clicked.connect(self.openRequested)
        actions.addWidget(new_btn)
        actions.addWidget(open_btn)
        actions.addStretch(1)
        cl.addLayout(actions)
        cl.addSpacing(20)

        recent_label = QtWidgets.QLabel("RECENT PROJECTS")
        recent_label.setObjectName("sectionLabel")
        cl.addWidget(recent_label)

        # Scrollable column of recent-project cards.
        self._list_host = QtWidgets.QWidget()
        self._list_layout = QtWidgets.QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 4, 0, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("recentScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._list_host)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        cl.addWidget(scroll, 1)

        self._empty = QtWidgets.QLabel(
            "No recent projects yet — create or open one to get started.")
        self._empty.setObjectName("emptyHint")
        self._empty.setWordWrap(True)
        cl.addWidget(self._empty)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row, 0)
        outer.addStretch(2)

        self.refresh()

    def set_catalog(self, by_type: dict):
        """Update the type→entry map used to render thumbnails (called when the
        component catalogue is reloaded)."""
        self._by_type = by_type or {}
        self._thumb_cache.clear()      # symbols/colours may have changed

    def restyle(self):
        """Re-apply the theme stylesheet and re-render thumbnails against the
        new backdrop (called after a light/dark swap)."""
        self.setStyleSheet(home_stylesheet())
        self._thumb_cache.clear()      # thumbnails baked the old canvas backdrop
        self.refresh()

    def refresh(self):
        """Repopulate the recent-projects list from the persistent store."""
        # Drop the existing cards (keep the trailing stretch at the end).
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        files = recent_files()
        for path in files:
            exists = os.path.exists(path)
            thumb = self._thumbnail_for(path) if exists else None
            card = _RecentCard(path, exists, thumb=thumb)
            card.clicked.connect(self.openRecentRequested)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

        has_items = bool(files)
        self._list_host.setVisible(has_items)
        self._empty.setVisible(not has_items)

    def _thumbnail_for(self, path: str) -> QtGui.QPixmap | None:
        """Render a layout preview of the project at ``path`` (``None`` on any
        read/parse error), memoised by file mtime."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._thumb_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        canvas_state = data.get("canvas", {})
        dpr = 2.0          # render crisp; the label shows it at logical size
        pm = render_canvas_thumbnail(
            canvas_state,
            QtCore.QSize(int(_THUMB_W * dpr), int(_THUMB_H * dpr)),
            self._by_type)
        pm.setDevicePixelRatio(dpr)
        self._thumb_cache[path] = (mtime, pm)
        return pm

