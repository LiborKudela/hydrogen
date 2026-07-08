"""Cross-platform Qt chrome: one Fusion-based look on Linux and Windows, with a
switchable light / dark palette.

Hydrogen's home screen already uses a custom palette; the editor shell used to
rely on each OS's native Qt style (Vista/11 on Windows, Fusion/GTK on Linux),
which made spacing, fonts, and control chrome diverge.  Call :func:`apply_theme`
once right after creating ``QApplication`` and, later, :func:`set_mode` to flip
between light and dark at runtime (widgets, the canvas and the plot objects all
listen on :func:`manager` for the change).
"""

from __future__ import annotations

from dataclasses import dataclass

from .qt import QtCore, QtGui, QtWidgets

__all__ = [
    "Colors",
    "LIGHT",
    "DARK",
    "current",
    "mode",
    "set_mode",
    "manager",
    "is_dark",
    "apply_theme",
    "home_stylesheet",
    "catalog_leaf_color",
]

#: QSettings coordinates (shared with the recent-files store).
_ORG = "hydrogen"
_APP = "hydrogen-ui"
_KEY = "theme_mode"


@dataclass(frozen=True)
class Colors:
    """A complete set of the semantic colours the UI paints with.

    Every hardcoded colour in the editor reads off the *active* instance
    (:func:`current`) so a single swap re-themes the whole application.
    """

    # --- window chrome (widgets, menus, dialogs) --------------------------- #
    accent: str
    accent_hover: str
    ink: str            # primary text
    muted: str          # secondary text / captions
    border: str
    card_bg: str        # base surface (inputs, menus, cards)
    card_hover: str
    page_bg: str        # the home screen backdrop
    window_bg: str      # main window / dialog backdrop
    selection: str      # list/tree selection wash
    disabled_fg: str
    disabled_bg: str
    scroll_handle: str
    scroll_handle_hover: str
    close_hover_bg: str
    close_hover_fg: str
    highlighted_text: str

    # --- canvas ------------------------------------------------------------ #
    canvas_bg: str
    grid: str
    node_title: str
    node_sub: str
    node_value: str
    node_name: str
    node_label_active: str
    node_border: str
    node_border_hover: str
    node_border_selected: str
    node_icon_border: str
    node_card: str
    node_card_hover: str
    port_label: str
    port_border: str
    port_border_hover: str
    resize_handle: str
    resize_handle_active: str
    resize_handle_fill: str
    badge_outline: str
    badge_dynamic: str
    warn_fill: str
    warn_border: str
    warn_mark: str
    menu_icon: str

    # --- plot / table objects --------------------------------------------- #
    plot_bg: str
    plot_header: str
    plot_header_text: str
    plot_border: str
    plot_border_selected: str
    plot_grip: str
    dark_charts: bool

    # --- semantic accents (kept legible on either backdrop) ---------------- #
    param_structural: str
    param_pure: str


LIGHT = Colors(
    accent="#2563eb",
    accent_hover="#1d4ed8",
    ink="#0f172a",
    muted="#64748b",
    border="#e2e8f0",
    card_bg="#ffffff",
    card_hover="#f1f5f9",
    page_bg="#eef2f7",
    window_bg="#f8fafc",
    selection="#dbeafe",
    disabled_fg="#94a3b8",
    disabled_bg="#f8fafc",
    scroll_handle="#cbd5e1",
    scroll_handle_hover="#94a3b8",
    close_hover_bg="#fee2e2",
    close_hover_fg="#b91c1c",
    highlighted_text="#ffffff",
    canvas_bg="#ffffff",
    grid="#ececec",
    node_title="#222222",
    node_sub="#555555",
    node_value="#222222",
    node_name="#555555",
    node_label_active="#d62828",
    node_border="#5a5a5a",
    node_border_hover="#1b6fb3",
    node_border_selected="#d62828",
    node_icon_border="#111111",
    node_card="#fdfdfd",
    node_card_hover="#ffffff",
    port_label="#333333",
    port_border="#3a3a3a",
    port_border_hover="#1b1b1b",
    resize_handle="#333333",
    resize_handle_active="#d62828",
    resize_handle_fill="#ffffff",
    badge_outline="#ffffff",
    badge_dynamic="#0a9396",
    warn_fill="#ffcc33",
    warn_border="#8a5a00",
    warn_mark="#5a3d00",
    menu_icon="#4b5563",
    plot_bg="#ffffff",
    plot_header="#546e7a",
    plot_header_text="#ffffff",
    plot_border="#90a4ae",
    plot_border_selected="#1976d2",
    plot_grip="#607d8b",
    dark_charts=False,
    param_structural="#b00020",
    param_pure="#1b7a31",
)


DARK = Colors(
    accent="#3b82f6",
    accent_hover="#60a5fa",
    ink="#e6edf3",
    muted="#8b98a9",
    border="#2c333d",
    card_bg="#20262e",
    card_hover="#2a313b",
    page_bg="#15191f",
    window_bg="#181c22",
    selection="#1e3a5f",
    disabled_fg="#5b6672",
    disabled_bg="#1c2128",
    scroll_handle="#3a434f",
    scroll_handle_hover="#4c5766",
    close_hover_bg="#5b1d1d",
    close_hover_fg="#fca5a5",
    highlighted_text="#ffffff",
    # The canvas stays deliberately *not* pure black so the light-bodied nodes
    # and P&ID symbol cards (whose SVG strokes are dark) still read clearly.
    canvas_bg="#171b21",
    grid="#262c34",
    node_title="#e6edf3",
    node_sub="#9aa7b6",
    node_value="#e6edf3",
    node_name="#9aa7b6",
    node_label_active="#ff6b6b",
    node_border="#8b98a9",
    node_border_hover="#4c9be8",
    node_border_selected="#ff6b6b",
    node_icon_border="#c9d4e0",
    # Symbol/box cards stay light so the dark-stroked SVG icons remain visible.
    node_card="#f4f6f8",
    node_card_hover="#ffffff",
    port_label="#c2ccd8",
    port_border="#b4bfcc",
    port_border_hover="#ffffff",
    resize_handle="#c2ccd8",
    resize_handle_active="#ff6b6b",
    resize_handle_fill="#20262e",
    badge_outline="#20262e",
    badge_dynamic="#2dd4bf",
    warn_fill="#f5b731",
    warn_border="#6b4700",
    warn_mark="#3a2900",
    menu_icon="#aab6c4",
    plot_bg="#20262e",
    plot_header="#3a4a5a",
    plot_header_text="#e6edf3",
    plot_border="#4a5766",
    plot_border_selected="#4c9be8",
    plot_grip="#7c8a99",
    dark_charts=True,
    param_structural="#ff6b6b",
    param_pure="#4ade80",
)


# ``Signal`` differs between bindings; resolve it the same way qt.py does.
_Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal


class _Manager(QtCore.QObject):
    """Carries a single ``changed`` signal every themed surface can subscribe
    to so a runtime mode swap repaints the whole app."""

    changed = _Signal()


_MANAGER = _Manager()
_MODE = "light"
_ACTIVE = LIGHT


def manager() -> "_Manager":
    """The process-wide theme manager; connect to ``manager().changed``."""
    return _MANAGER


def current() -> Colors:
    """The colours the UI should paint with right now."""
    return _ACTIVE


def mode() -> str:
    """The active mode as chosen by the user: ``"light"``, ``"dark"`` or
    ``"system"``."""
    return _MODE


def is_dark() -> bool:
    return _ACTIVE is DARK


def _system_is_dark() -> bool:
    """Best-effort OS dark-mode probe (Qt 6.5+ exposes it; older builds fall
    back to light)."""
    app = QtWidgets.QApplication.instance()
    hints = app.styleHints() if app is not None else None
    scheme = getattr(hints, "colorScheme", None)
    if scheme is not None:
        try:
            return scheme() == QtCore.Qt.ColorScheme.Dark
        except (AttributeError, TypeError):
            pass
    # Fall back to reading the window text vs. window brightness.
    if app is not None:
        pal = app.palette()
        win = pal.color(QtGui.QPalette.Window)
        return win.lightness() < 128
    return False


def _resolve(mode_name: str) -> Colors:
    if mode_name == "dark":
        return DARK
    if mode_name == "system":
        return DARK if _system_is_dark() else LIGHT
    return LIGHT


def _settings() -> "QtCore.QSettings":
    return QtCore.QSettings(_ORG, _APP)


def load_saved_mode() -> str:
    """The persisted mode (defaults to ``"light"``)."""
    value = _settings().value(_KEY, "light")
    value = str(value).lower()
    return value if value in ("light", "dark", "system") else "light"


def _app_font() -> QtGui.QFont:
    """A cross-platform UI font stack with stable metrics."""
    font = QtGui.QFont()
    font.setFamilies([
        "Segoe UI",
        "Ubuntu",
        "Noto Sans",
        "Roboto",
        "Helvetica Neue",
        "Liberation Sans",
        "sans-serif",
    ])
    font.setPointSize(10)
    return font


def _palette(c: Colors) -> QtGui.QPalette:
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(c.window_bg))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(c.ink))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(c.card_bg))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(c.card_hover))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(c.ink))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(c.card_bg))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(c.ink))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(c.card_bg))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(c.ink))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(c.accent))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(c.highlighted_text))
    pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(c.muted))
    pal.setColor(QtGui.QPalette.Mid, QtGui.QColor(c.border))
    pal.setColor(QtGui.QPalette.Link, QtGui.QColor(c.accent))
    disabled = QtGui.QPalette.Disabled
    pal.setColor(disabled, QtGui.QPalette.WindowText, QtGui.QColor(c.disabled_fg))
    pal.setColor(disabled, QtGui.QPalette.Text, QtGui.QColor(c.disabled_fg))
    pal.setColor(disabled, QtGui.QPalette.ButtonText, QtGui.QColor(c.disabled_fg))
    return pal


def _widget_stylesheet(c: Colors) -> str:
    return f"""
    QWidget {{
        color: {c.ink};
    }}
    QMainWindow, QDialog {{
        background: {c.window_bg};
    }}
    QToolTip {{
        color: {c.ink};
        background: {c.card_bg};
        border: 1px solid {c.border};
    }}
    QMenuBar {{
        background: {c.card_bg};
        border-bottom: 1px solid {c.border};
        padding: 2px 0;
    }}
    QMenuBar::item {{
        padding: 4px 10px;
        background: transparent;
    }}
    QMenuBar::item:selected {{
        background: {c.card_hover};
        border-radius: 4px;
    }}
    QMenu {{
        background: {c.card_bg};
        border: 1px solid {c.border};
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 6px 28px 6px 16px;
    }}
    QMenu::item:selected {{
        background: {c.selection};
    }}
    QToolBar {{
        background: {c.card_bg};
        border-bottom: 1px solid {c.border};
        spacing: 6px;
        padding: 4px 8px;
    }}
    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 4px 8px;
    }}
    QToolButton:hover {{
        background: {c.card_hover};
        border-color: {c.border};
    }}
    QPushButton {{
        background: {c.card_bg};
        color: {c.ink};
        border: 1px solid {c.border};
        border-radius: 6px;
        padding: 6px 12px;
        min-height: 1.2em;
    }}
    QPushButton:hover {{
        background: {c.card_hover};
        border-color: {c.accent};
    }}
    QPushButton:pressed {{
        background: {c.card_hover};
    }}
    QPushButton:disabled {{
        color: {c.disabled_fg};
        background: {c.disabled_bg};
    }}
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {c.card_bg};
        border: 1px solid {c.border};
        border-radius: 6px;
        padding: 4px 8px;
        selection-background-color: {c.accent};
        selection-color: {c.highlighted_text};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {c.accent};
    }}
    QComboBox QAbstractItemView {{
        background: {c.card_bg};
        border: 1px solid {c.border};
        selection-background-color: {c.selection};
    }}
    QTabWidget::pane {{
        border: 1px solid {c.border};
        background: {c.window_bg};
        border-radius: 6px;
    }}
    QTabBar::tab {{
        background: transparent;
        border: 1px solid transparent;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 8px 16px;
        margin-right: 2px;
        color: {c.muted};
    }}
    QTabBar::tab:selected {{
        background: {c.card_bg};
        border-color: {c.border};
        color: {c.ink};
        font-weight: 600;
    }}
    QTabBar::tab:hover:!selected {{
        background: {c.card_hover};
        color: {c.ink};
    }}
    QTreeWidget, QListWidget, QTableWidget, QTreeView, QListView, QTableView {{
        background: {c.card_bg};
        border: 1px solid {c.border};
        border-radius: 6px;
        outline: none;
        alternate-background-color: {c.card_hover};
    }}
    QTreeWidget::item, QListWidget::item {{
        padding: 3px 2px;
    }}
    QTreeWidget::item:selected, QListWidget::item:selected,
    QTableWidget::item:selected {{
        background: {c.selection};
        color: {c.ink};
    }}
    QHeaderView::section {{
        background: {c.card_hover};
        color: {c.muted};
        border: none;
        border-bottom: 1px solid {c.border};
        padding: 6px 8px;
    }}
    QSplitter::handle {{
        background: {c.border};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    QStatusBar {{
        background: {c.card_bg};
        border-top: 1px solid {c.border};
        color: {c.muted};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c.scroll_handle};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.scroll_handle_hover};
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c.scroll_handle};
        border-radius: 5px;
        min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c.scroll_handle_hover};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0;
        height: 0;
    }}
    QGroupBox {{
        border: 1px solid {c.border};
        border-radius: 6px;
        margin-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}
    #winCtlBtn, #winCloseBtn {{
        color: {c.muted};
        font-size: 14px;
        font-weight: 600;
        border-radius: 4px;
        padding: 0;
    }}
    #winCtlBtn:hover, #winCloseBtn:hover {{
        background: {c.card_hover};
        color: {c.ink};
    }}
    #winCloseBtn:hover {{
        background: {c.close_hover_bg};
        color: {c.close_hover_fg};
    }}
    """


def home_stylesheet() -> str:
    """Stylesheet for :class:`~hydrogen.ui.home.HomeScreen` widgets."""
    c = _ACTIVE
    return f"""
    #homeScreen {{
        background: {c.page_bg};
    }}
    #homeCard {{
        background: {c.card_bg};
        border: 1px solid {c.border};
        border-radius: 16px;
    }}
    #brand {{
        color: {c.accent};
        font-size: 34px;
        font-weight: 800;
    }}
    #tagline {{
        color: {c.ink};
        font-size: 16px;
        font-weight: 600;
    }}
    #subtitle {{
        color: {c.muted};
        font-size: 13px;
    }}
    #sectionLabel {{
        color: {c.muted};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    #primaryBtn {{
        background: {c.accent};
        color: {c.highlighted_text};
        border: none;
        border-radius: 9px;
        padding: 9px 18px;
        font-size: 13px;
        font-weight: 600;
    }}
    #primaryBtn:hover {{ background: {c.accent_hover}; }}
    #primaryBtn:pressed {{ background: {c.accent_hover}; }}
    #secondaryBtn {{
        background: {c.card_bg};
        color: {c.ink};
        border: 1px solid {c.border};
        border-radius: 9px;
        padding: 9px 18px;
        font-size: 13px;
        font-weight: 600;
    }}
    #secondaryBtn:hover {{ background: {c.card_hover}; }}
    #recentScroll {{ background: transparent; }}
    #recentScroll > QWidget > QWidget {{ background: transparent; }}
    #recentCard {{
        background: {c.card_bg};
        border: 1px solid {c.border};
        border-radius: 10px;
    }}
    #recentCard:hover {{
        background: {c.card_hover};
        border: 1px solid {c.accent};
    }}
    #recentCard[missing="true"] {{ background: {c.disabled_bg}; }}
    #recentThumb {{
        background: {c.canvas_bg};
        border: 1px solid {c.border};
        border-radius: 7px;
        color: {c.muted};
        font-size: 10px;
    }}
    #recentThumb[missing="true"] {{ color: #f59e0b; }}
    #recentTitle {{ color: {c.ink}; font-size: 13px; font-weight: 600; }}
    #recentSub {{ color: {c.muted}; font-size: 11px; }}
    #recentIcon {{ font-size: 15px; }}
    #recentChevron {{ color: {c.muted}; font-size: 20px; font-weight: 700; }}
    #emptyHint {{ color: {c.disabled_fg}; font-size: 12px; padding: 18px 2px; }}
    """


def catalog_leaf_color(base: QtGui.QColor) -> QtGui.QColor:
    """Readable tree-leaf text derived from a domain's pastel fill: darkened on
    a light backdrop, lightened on a dark one."""
    if is_dark():
        return base.lighter(135)
    return base.darker(170)


def apply_theme(app: QtWidgets.QApplication, mode_name: str | None = None) -> None:
    """Apply hydrogen's cross-platform Fusion look to *app* in the requested
    mode (or the persisted one when ``mode_name`` is ``None``)."""
    global _MODE, _ACTIVE
    _MODE = (mode_name or load_saved_mode()).lower()
    if _MODE not in ("light", "dark", "system"):
        _MODE = "light"
    _ACTIVE = _resolve(_MODE)
    app.setStyle("Fusion")
    app.setFont(_app_font())
    app.setPalette(_palette(_ACTIVE))
    app.setStyleSheet(_widget_stylesheet(_ACTIVE))


def set_mode(mode_name: str, app: QtWidgets.QApplication | None = None) -> None:
    """Switch the active theme, persist the choice, restyle *app* and notify
    every subscribed surface via :func:`manager`."""
    app = app or QtWidgets.QApplication.instance()
    if app is None:
        return
    _settings().setValue(_KEY, mode_name.lower())
    apply_theme(app, mode_name)
    _MANAGER.changed.emit()
