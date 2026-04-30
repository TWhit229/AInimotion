"""
Compact queue item widget.
"""

from __future__ import annotations

import subprocess
import sys
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QMimeData, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QDrag
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .theme import STATUS_COLORS


class ItemState(Enum):
    QUEUED = 'queued'
    PROCESSING = 'processing'
    PAUSED = 'paused'
    COMPLETE = 'complete'
    ERROR = 'error'
    CANCELLED = 'cancelled'


def fmt_time(s: float) -> str:
    if s < 60: return f"{s:.0f}s"
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def fmt_dur(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class QueueItemWidget(QFrame):
    """
    Queue item with two rows:
      Row 1: [Name]  [Status]  [===progress %===]  [Pause] [X]
      Row 2: (only during processing) 1.8 frames/sec  Elapsed: 3m  Remaining: ~8m
    """

    remove_requested = pyqtSignal(object)
    pause_resume_requested = pyqtSignal(object)
    retry_requested = pyqtSignal(object)
    start_now_requested = pyqtSignal(object)

    def __init__(self, input_path, output_path, file_info=None, parent=None):
        super().__init__(parent)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.file_info = file_info
        self.state = ItemState.QUEUED
        self._elapsed_at_complete = 0.0
        self.setObjectName("queueItem")
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 6, 8, 6)
        outer.setSpacing(2)

        # === Row 1 ===
        r1 = QHBoxLayout()
        r1.setSpacing(8)

        # Name (truncates with ellipsis, full name in tooltip)
        stem = self.input_path.stem
        if len(stem) > 45:
            stem = stem[:42] + "..."
        name = stem
        if self.file_info and self.file_info.get('duration', 0) > 0:
            name += f"  ({fmt_dur(self.file_info['duration'])})"
        self._name = QLabel(name)
        self._name.setStyleSheet("font-weight: bold; background: transparent;")
        self._name.setToolTip(str(self.input_path))
        self._name.setMaximumWidth(400)
        self._name.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        r1.addWidget(self._name)

        # Status (fixed)
        self._status = QLabel("Queued")
        self._status.setFixedWidth(80)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("background: transparent;")
        r1.addWidget(self._status)

        # Progress bar (fixed max width so it doesn't stretch on wide windows)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        self._bar.setFixedHeight(18)
        self._bar.setMinimumWidth(120)
        self._bar.setMaximumWidth(250)
        r1.addWidget(self._bar)

        # Spacer pushes buttons to right
        r1.addStretch()

        # Buttons
        self._start_btn = self._mkbtn("Start", "Start now", lambda: self.start_now_requested.emit(self), 42)
        r1.addWidget(self._start_btn)

        self._pause_btn = self._mkbtn("Pause", "Pause", lambda: self.pause_resume_requested.emit(self), 50)
        r1.addWidget(self._pause_btn)

        self._retry_btn = self._mkbtn("Retry", "Retry", lambda: self.retry_requested.emit(self), 44)
        self._retry_btn.setStyleSheet("background-color: #FF9800; color: #1e1e2e; font-weight: bold; border-radius: 4px;")
        r1.addWidget(self._retry_btn)

        self._open_btn = self._mkbtn("Play", "Play output", self._on_open, 40)
        r1.addWidget(self._open_btn)

        self._folder_btn = self._mkbtn("\U0001F4C2", "Show in folder", self._on_folder, 28)
        r1.addWidget(self._folder_btn)

        self._remove_btn = self._mkbtn("\u2715", "Remove", lambda: self.remove_requested.emit(self), 28)
        r1.addWidget(self._remove_btn)

        outer.addLayout(r1)

        # === Row 2: details (fps, elapsed, remaining) ===
        self._details = QLabel("")
        self._details.setStyleSheet("color: #999; font-size: 11px; background: transparent; padding-left: 2px;")
        outer.addWidget(self._details)

    def _mkbtn(self, text, tip, cb, w=28):
        b = QPushButton(text)
        b.setObjectName("smallBtn")
        b.setFixedSize(w, 24)
        b.setToolTip(tip)
        b.clicked.connect(cb)
        return b

    def _refresh(self):
        q = self.state == ItemState.QUEUED
        active = self.state in (ItemState.PROCESSING, ItemState.PAUSED)
        paused = self.state == ItemState.PAUSED
        done = self.state == ItemState.COMPLETE
        err = self.state == ItemState.ERROR

        # Buttons
        self._start_btn.setVisible(q)
        self._pause_btn.setVisible(active)
        self._retry_btn.setVisible(err)
        self._open_btn.setVisible(done)
        self._folder_btn.setVisible(done)

        # Progress bar only when active
        self._bar.setVisible(active)

        # Pause/Resume text
        if paused:
            self._pause_btn.setText("Resume")
            self._pause_btn.setFixedWidth(55)
        else:
            self._pause_btn.setText("Pause")
            self._pause_btn.setFixedWidth(50)

        # Status
        smap = {
            ItemState.QUEUED:     ("Queued",     'default'),
            ItemState.PROCESSING: ("Processing", 'processing'),
            ItemState.PAUSED:     ("Paused",     'paused'),
            ItemState.COMPLETE:   ("Complete",   'complete'),
            ItemState.ERROR:      ("Error",      'error'),
            ItemState.CANCELLED:  ("Cancelled",  'cancelled'),
        }
        text, ckey = smap[self.state]
        self._status.setStyleSheet(f"color: {STATUS_COLORS.get(ckey, '#888')}; background: transparent;")
        self._status.setText(text)

        # Details row
        if q:
            self._bar.setValue(0)
            self._details.setVisible(False)
        elif done:
            self._details.setText(f"Completed in {fmt_time(self._elapsed_at_complete)}")
            self._details.setVisible(True)
        elif err:
            self._details.setVisible(True)
        else:
            self._details.setVisible(active)

    def set_state(self, state):
        self.state = state
        self._refresh()

    def update_progress(self, p):
        if p.phase == 'analyzing':
            self._bar.setVisible(True)
            self._pause_btn.setVisible(True)
            self._start_btn.setVisible(False)
            self._status.setText("Analyzing")
            self._status.setStyleSheet(f"color: {STATUS_COLORS['analyzing']}; background: transparent;")
            if p.pairs_total > 0:
                self._bar.setValue(int(p.pairs_done / p.pairs_total * 1000))
            self._details.setText(f"Scanning frames: {p.pairs_done}/{p.pairs_total}")
            self._details.setVisible(True)

        elif p.phase == 'processing':
            self._bar.setVisible(True)
            self._pause_btn.setVisible(True)
            self._start_btn.setVisible(False)
            self._status.setText("Processing")
            self._status.setStyleSheet(f"color: {STATUS_COLORS['processing']}; background: transparent;")
            if p.inferences_total > 0:
                self._bar.setValue(int(p.inferences_done / p.inferences_total * 1000))
            parts = []
            if p.fps_rate > 0: parts.append(f"{p.fps_rate:.1f} frames/sec")
            if p.elapsed_seconds > 0: parts.append(f"Elapsed: {fmt_time(p.elapsed_seconds)}")
            if p.eta_seconds is not None: parts.append(f"Remaining: ~{fmt_time(p.eta_seconds)}")
            self._details.setText("    ".join(parts))
            self._details.setVisible(True)

        elif p.phase == 'complete':
            self._elapsed_at_complete = p.elapsed_seconds
            self.set_state(ItemState.COMPLETE)
        elif p.phase == 'cancelled':
            self.set_state(ItemState.CANCELLED)
        elif p.phase == 'error':
            self.set_state(ItemState.ERROR)
            self._details.setText(str(p.error_message or 'Unknown error')[:80])
            self._details.setVisible(True)

    # Actions
    def _on_open(self):
        if not self.output_path.exists(): return
        w = self.window()
        player = w.get_player_path() if hasattr(w, 'get_player_path') else None
        if player:
            try:
                subprocess.Popen([player, str(self.output_path)])
            except FileNotFoundError:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path)))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path)))

    def _on_folder(self):
        if not self.output_path.exists(): return
        if sys.platform == 'win32':
            subprocess.Popen(['explorer', '/select,', str(self.output_path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path.parent)))

    # Drag reorder
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.state == ItemState.QUEUED:
            self._drag_start = e.pos()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.MouseButton.LeftButton) or self.state != ItemState.QUEUED:
            return
        if not hasattr(self, '_drag_start'): return
        if (e.pos() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-ainimotion-queue", b"d")
        drag.setMimeData(mime)
        pixmap = self.grab()
        drag.setPixmap(pixmap.scaledToWidth(min(pixmap.width(), 500)))
        drag.setHotSpot(e.pos())
        drag.exec(Qt.DropAction.MoveAction)
