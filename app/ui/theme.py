"""
Dark theme stylesheet for AInimotion.
"""

VIDEO_EXTENSIONS = ('.mkv', '.mp4', '.avi', '.mov', '.webm', '.wmv', '.m4v', '.ts')

DARK_THEME = """
/* ---- Base ---- */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}

/* ---- Menu Bar ---- */
QMenuBar {
    background-color: #252538;
    border-bottom: 1px solid #3d3d5c;
    padding: 2px 0;
}
QMenuBar::item {
    padding: 5px 12px;
    border-radius: 4px;
    margin: 1px 2px;
}
QMenuBar::item:selected {
    background-color: #3d3d5c;
}
QMenu {
    background-color: #2d2d44;
    border: 1px solid #3d3d5c;
    border-radius: 6px;
    padding: 4px 0;
}
QMenu::item {
    padding: 6px 30px 6px 20px;
}
QMenu::item:selected {
    background-color: #7c3aed;
    border-radius: 4px;
    margin: 0 4px;
}
QMenu::separator {
    height: 1px;
    background-color: #3d3d5c;
    margin: 4px 10px;
}

/* ---- Group Boxes ---- */
QGroupBox {
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    margin-top: 8px;
    padding-top: 14px;
    font-size: 13px;
    font-weight: bold;
    color: #888;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
}

/* ---- Queue Items: purple gradient row ---- */
QFrame#queueItem {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2e2845, stop:0.5 #352d52, stop:1 #2e2845);
    border: none;
    border-radius: 6px;
    padding: 0;
    margin: 1px 0;
}
QFrame#queueItem QLabel,
QFrame#queueItem QProgressBar {
    background: transparent;
}

/* ---- Progress Bar: outlined, fills in ---- */
QProgressBar {
    background-color: transparent;
    border: 1px solid #5d5d7c;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #7c3aed;
    border-radius: 3px;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: #3d3d5c;
    color: #e0e0e0;
    border: 1px solid #4d4d6c;
    border-radius: 5px;
    padding: 4px 14px;
}
QPushButton:hover {
    background-color: #4d4d6c;
}
QPushButton:pressed {
    background-color: #5d5d7c;
}
QPushButton#addBtn {
    background-color: #7c3aed;
    border: 1px solid #8b4fef;
    font-weight: bold;
}
QPushButton#addBtn:hover {
    background-color: #8b4fef;
}
QPushButton#smallBtn {
    padding: 0;
    border-radius: 4px;
}
QPushButton#retryBtn {
    background-color: #FF9800;
    border: 1px solid #FFB74D;
    color: #1e1e2e;
    font-weight: bold;
    padding: 2px 10px;
}

/* ---- Combo Box ---- */
QComboBox {
    background-color: #2d2d44;
    border: 1px solid #3d3d5c;
    border-radius: 6px;
    padding: 5px 10px;
    color: #e0e0e0;
}
QComboBox:hover {
    border: 1px solid #5d5d7c;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #888;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d44;
    border: 1px solid #3d3d5c;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #7c3aed;
    color: #e0e0e0;
    outline: none;
}

/* ---- Scroll Area ---- */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: transparent;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #4d4d6c;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #5d5d7c;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* ---- Tooltips ---- */
QToolTip {
    background-color: #3d3d5c;
    color: #e0e0e0;
    border: 1px solid #5d5d7c;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}

/* ---- Message Box ---- */
QMessageBox {
    background-color: #2d2d44;
}
QMessageBox QLabel {
    color: #e0e0e0;
}
"""

STATUS_COLORS = {
    'default': '#888',
    'processing': '#e0e0e0',
    'paused': '#FF9800',
    'complete': '#4CAF50',
    'error': '#F44336',
    'cancelled': '#FF9800',
    'analyzing': '#7c3aed',
}
