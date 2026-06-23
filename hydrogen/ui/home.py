"""The editor's landing page: start a new project, open an existing one, or
pick up a recently opened project.

A self-contained, modern-looking widget that only emits intent;
:class:`hydrogen.ui.app` decides what to do with it (load a file, swap to the
editor, ...).
"""

from __future__ import annotations

import os

from .qt import QtCore, QtGui, QtWidgets, Signal
from .recent import recent_files

__all__ = ["HomeScreen"]


# Palette shared across the home screen's stylesheet.
_ACCENT = "#2563eb"
_ACCENT_HOVER = "#1d4ed8"
_INK = "#0f172a"
_MUTED = "#64748b"
_BORDER = "#e2e8f0"
_CARD_BG = "#ffffff"
_CARD_HOVER = "#f1f5f9"


class _RecentCard(QtWidgets.QFrame):
    """A clickable, hover-highlighted row for one recent project."""

    clicked = Signal(str)

    def __init__(self, path: str, exists: bool, parent=None):
        super().__init__(parent)
        self._path = path
        self.setObjectName("recentCard")
        self.setProperty("missing", not exists)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        name = os.path.basename(path) or path
        folder = os.path.dirname(path) or "—"

        icon = QtWidgets.QLabel("●")
        icon.setObjectName("recentIcon")
        icon.setFixedWidth(20)
        icon.setAlignment(QtCore.Qt.AlignCenter)
        icon.setStyleSheet(
            f"color: {_ACCENT if exists else '#f59e0b'}; font-size: 12px;")

        title = QtWidgets.QLabel(name if exists else f"{name}  ·  missing")
        title.setObjectName("recentTitle")
        sub = QtWidgets.QLabel(folder)
        sub.setObjectName("recentSub")
        sub.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)

        text = QtWidgets.QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(1)
        text.addWidget(title)
        text.addWidget(sub)

        chevron = QtWidgets.QLabel("›")
        chevron.setObjectName("recentChevron")

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(12, 9, 14, 9)
        row.setSpacing(10)
        row.addWidget(icon)
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("homeScreen")
        self.setStyleSheet(self._stylesheet())

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
        new_btn = QtWidgets.QPushButton("＋  New project")
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
            card = _RecentCard(path, os.path.exists(path))
            card.clicked.connect(self.openRecentRequested)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

        has_items = bool(files)
        self._list_host.setVisible(has_items)
        self._empty.setVisible(not has_items)

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        #homeScreen {{
            background: #eef2f7;
        }}
        #homeCard {{
            background: {_CARD_BG};
            border: 1px solid {_BORDER};
            border-radius: 16px;
        }}
        #brand {{
            color: {_ACCENT};
            font-size: 34px;
            font-weight: 800;
        }}
        #tagline {{
            color: {_INK};
            font-size: 16px;
            font-weight: 600;
        }}
        #subtitle {{
            color: {_MUTED};
            font-size: 13px;
        }}
        #sectionLabel {{
            color: {_MUTED};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
        }}
        #primaryBtn {{
            background: {_ACCENT};
            color: white;
            border: none;
            border-radius: 9px;
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 600;
        }}
        #primaryBtn:hover {{ background: {_ACCENT_HOVER}; }}
        #primaryBtn:pressed {{ background: {_ACCENT_HOVER}; }}
        #secondaryBtn {{
            background: {_CARD_BG};
            color: {_INK};
            border: 1px solid {_BORDER};
            border-radius: 9px;
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 600;
        }}
        #secondaryBtn:hover {{ background: {_CARD_HOVER}; }}
        #recentScroll {{ background: transparent; }}
        #recentScroll > QWidget > QWidget {{ background: transparent; }}
        #recentCard {{
            background: {_CARD_BG};
            border: 1px solid {_BORDER};
            border-radius: 10px;
        }}
        #recentCard:hover {{
            background: {_CARD_HOVER};
            border: 1px solid {_ACCENT};
        }}
        #recentCard[missing="true"] {{ background: #fbfbfc; }}
        #recentTitle {{ color: {_INK}; font-size: 13px; font-weight: 600; }}
        #recentSub {{ color: {_MUTED}; font-size: 11px; }}
        #recentIcon {{ font-size: 15px; }}
        #recentChevron {{ color: {_MUTED}; font-size: 20px; font-weight: 700; }}
        #emptyHint {{ color: #94a3b8; font-size: 12px; padding: 18px 2px; }}
        """
