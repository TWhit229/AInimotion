"""
Main application window — menu bar, queue/completed split, processing control.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PyQt6.QtCore import QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.analysis_cache import AnalysisCache
from app.core.processor import ProcessingProgress, VideoProcessor
from app.core.video_io import probe_video
from .queue_item_widget import ItemState, QueueItemWidget
from .theme import VIDEO_EXTENSIONS

VIDEO_FILTER_STR = (
    "Video Files (" + " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS) + ");;All Files (*)"
)

SETTINGS_KEY_OUTPUT_DIR = "output_directory"
SETTINGS_KEY_WINDOW_GEOMETRY = "window_geometry"


def make_output_path(
    input_path: Path,
    output_dir: Path | None,
    fps_multiplier: int = 2,
    input_fps: float | None = None,
) -> Path:
    """Generate output path with actual output FPS in the name."""
    if input_fps:
        out_fps = int(input_fps * fps_multiplier)
        suffix = f"_{out_fps}fps"
    else:
        suffix = f"_{fps_multiplier}x"
    name = f"{input_path.stem}{suffix}{input_path.suffix}"
    if output_dir:
        return output_dir / name
    return input_path.parent / name


def find_vlc() -> str | None:
    """Auto-detect VLC installation on Windows."""
    for path in [
        Path("C:/Program Files/VideoLAN/VLC/vlc.exe"),
        Path("C:/Program Files (x86)/VideoLAN/VLC/vlc.exe"),
    ]:
        if path.exists():
            return str(path)
    return None


class ProcessingThread(QThread):
    """Processing thread with its own pause/cancel events for independent control."""
    progress_updated = pyqtSignal(object)
    finished_processing = pyqtSignal(bool)

    def __init__(self, processor: VideoProcessor, input_path: str, output_path: str,
                 precomputed_analysis=None):
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.output_path = output_path
        self.precomputed_analysis = precomputed_analysis
        self._cancel = __import__('threading').Event()
        self._pause = __import__('threading').Event()
        self._pause.set()

    def run(self):
        try:
            success = self.processor.process(
                self.input_path, self.output_path,
                progress_callback=lambda p: self.progress_updated.emit(p),
                cancel_event=self._cancel,
                pause_event=self._pause,
                precomputed_analysis=self.precomputed_analysis,
            )
            self.finished_processing.emit(success)
        except Exception as e:
            self.progress_updated.emit(
                ProcessingProgress(phase='error', error_message=str(e))
            )
            self.finished_processing.emit(False)

    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    def cancel(self):
        self._cancel.set()
        self._pause.set()  # unblock so thread can exit


class MainWindow(QMainWindow):

    def __init__(self, processor: VideoProcessor):
        super().__init__()
        self.processor = processor
        self.queue_items: list[QueueItemWidget] = []
        self.completed_items: list[QueueItemWidget] = []
        self._current_thread: ProcessingThread | None = None
        self._current_item: QueueItemWidget | None = None
        self._suspended_thread: ProcessingThread | None = None
        self._suspended_item: QueueItemWidget | None = None
        self._settings = QSettings("AInimotion", "AInimotion")
        self._output_dir: Path | None = None
        self._auto_start = False

        saved_dir = self._settings.value(SETTINGS_KEY_OUTPUT_DIR, "")
        if saved_dir and Path(saved_dir).is_dir():
            self._output_dir = Path(saved_dir)

        # Restore settings (type= prevents ValueError from corrupted QSettings)
        saved_max_h = self._settings.value("max_height", 720, type=int)
        self.processor.max_height = saved_max_h if saved_max_h > 0 else None
        self.processor.codec = self._settings.value("codec", "libx264", type=str)
        self.processor.crf = self._settings.value("crf", 18, type=int)
        self.processor.fps_multiplier = self._settings.value("fps_multiplier", 2, type=int)

        # Background analysis cache (pre-analyzes next queued videos on CPU)
        max_h = self.processor.max_height
        self._analysis_cache = AnalysisCache(max_height=max_h if max_h and max_h > 0 else None)

        self.setWindowTitle("AInimotion")
        self.setMinimumSize(750, 650)
        self.resize(900, 750)
        self.setAcceptDrops(True)

        self._build_ui()
        self._build_menu_bar()
        self._rebuild_output_combo()
        self._update_status()

        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_info)
        self._gpu_timer.start(3000)
        self._update_gpu_info()

        geometry = self._settings.value(SETTINGS_KEY_WINDOW_GEOMETRY)
        if geometry:
            self.restoreGeometry(geometry)

    # ================================================================
    # MENU BAR
    # ================================================================

    def _build_menu_bar(self):
        menubar = self.menuBar()
        menubar.setObjectName("menuBar")

        # File
        file_menu = menubar.addMenu("&File")

        add_action = QAction("&Add Files...", self)
        add_action.setShortcut(QKeySequence("Ctrl+O"))
        add_action.triggered.connect(self._on_add_files)
        file_menu.addAction(add_action)

        add_folder_action = QAction("Add &Folder...", self)
        add_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        add_folder_action.triggered.connect(self._on_add_folder)
        file_menu.addAction(add_folder_action)

        file_menu.addSeparator()

        output_action = QAction("Set &Output Folder...", self)
        output_action.triggered.connect(self._on_pick_output_dir)
        file_menu.addAction(output_action)

        reset_output_action = QAction("&Reset Output to Default", self)
        reset_output_action.triggered.connect(self._on_reset_output_dir)
        file_menu.addAction(reset_output_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Alt+F4"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Queue
        queue_menu = menubar.addMenu("&Queue")

        start_action = QAction("&Start Queue", self)
        start_action.setShortcut(QKeySequence("Ctrl+Return"))
        start_action.triggered.connect(self._on_start_queue)
        queue_menu.addAction(start_action)

        stop_action = QAction("S&top All", self)
        stop_action.triggered.connect(self._on_stop_all)
        queue_menu.addAction(stop_action)

        pause_action = QAction("&Pause / Resume", self)
        pause_action.setShortcut(QKeySequence("Space"))
        pause_action.triggered.connect(self._on_toggle_pause)
        queue_menu.addAction(pause_action)

        queue_menu.addSeparator()

        self._auto_start_action = QAction("&Auto-Start on Add", self, checkable=True)
        self._auto_start_action.setChecked(False)
        self._auto_start_action.triggered.connect(lambda c: setattr(self, '_auto_start', c))
        queue_menu.addAction(self._auto_start_action)

        queue_menu.addSeparator()

        clear_all_action = QAction("Clear &All", self)
        clear_all_action.triggered.connect(self._on_clear_all)
        queue_menu.addAction(clear_all_action)

        # Settings
        settings_menu = menubar.addMenu("&Settings")

        res_menu = settings_menu.addMenu("&Resolution")
        self._res_actions = {}
        for label, value in [("720p (recommended)", 720), ("480p (faster)", 480), ("Native (no downscale)", 0)]:
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _, v=value: self._on_set_resolution(v))
            res_menu.addAction(action)
            self._res_actions[value] = action
        saved_res = int(self._settings.value("max_height", 720))
        self._res_actions.get(saved_res, self._res_actions[720]).setChecked(True)

        fps_menu = settings_menu.addMenu("&FPS Multiplier")
        self._fps_actions = {}
        for label, value in [("2x", 2), ("3x", 3), ("4x", 4)]:
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _, v=value: self._on_set_fps_mult(v))
            fps_menu.addAction(action)
            self._fps_actions[value] = action
        saved_fps = int(self._settings.value("fps_multiplier", 2))
        self._fps_actions.get(saved_fps, self._fps_actions[2]).setChecked(True)

        settings_menu.addSeparator()

        codec_menu = settings_menu.addMenu("&Codec")
        self._codec_actions = {}
        for label, value in [("H.264 (compatible)", "libx264"), ("H.265 / HEVC (smaller)", "libx265")]:
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _, v=value: self._on_set_codec(v))
            codec_menu.addAction(action)
            self._codec_actions[value] = action
        saved_codec = self._settings.value("codec", "libx264")
        self._codec_actions.get(saved_codec, self._codec_actions["libx264"]).setChecked(True)

        quality_menu = settings_menu.addMenu("&Quality")
        self._quality_actions = {}
        for label, value in [("Lossless (CRF 0)", 0), ("Very High (CRF 14)", 14), ("High (CRF 18)", 18), ("Medium (CRF 23)", 23), ("Low (CRF 28)", 28)]:
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _, v=value: self._on_set_quality(v))
            quality_menu.addAction(action)
            self._quality_actions[value] = action
        saved_quality = int(self._settings.value("crf", 18))
        self._quality_actions.get(saved_quality, self._quality_actions[18]).setChecked(True)

        fmt_menu = settings_menu.addMenu("Output &Format")
        self._fmt_actions = {}
        for label, value in [("Same as input", "auto"), ("MKV (.mkv)", ".mkv"), ("MP4 (.mp4)", ".mp4")]:
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _, v=value: self._on_set_format(v))
            fmt_menu.addAction(action)
            self._fmt_actions[value] = action
        saved_fmt = self._settings.value("output_format", "auto")
        self._fmt_actions.get(saved_fmt, self._fmt_actions["auto"]).setChecked(True)

        settings_menu.addSeparator()

        upscale_menu = settings_menu.addMenu("&Upscaling")
        for label in ["None", "2x (coming soon)", "4x (coming soon)"]:
            action = QAction(label, self, checkable=True)
            action.setEnabled("coming soon" not in label)
            if label == "None":
                action.setChecked(True)
            upscale_menu.addAction(action)

        settings_menu.addSeparator()

        # Video player
        self._vlc_path = self._settings.value("video_player", "") or find_vlc() or ""
        player_action = QAction("Set Video &Player...", self)
        player_action.triggered.connect(self._on_set_player)
        settings_menu.addAction(player_action)

        if self._vlc_path:
            self._log(f"Video player: {Path(self._vlc_path).name}")
        else:
            self._log("Tip: Install VLC (videolan.org) for best playback, or set a player in Settings")

        # Help
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About AInimotion", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ================================================================
    # MAIN UI
    # ================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 8, 16, 8)
        main_layout.setSpacing(8)

        # ---- Queue Group ----
        queue_group = QGroupBox("Queue")
        queue_group.setObjectName("queueGroup")
        queue_inner = QVBoxLayout(queue_group)
        queue_inner.setContentsMargins(8, 12, 8, 8)
        queue_inner.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._queue_container = QWidget()
        self._queue_container.setAcceptDrops(True)
        self._queue_container.dragEnterEvent = self._queue_drag_enter
        self._queue_container.dragMoveEvent = self._queue_drag_move
        self._queue_container.dropEvent = self._queue_drop

        self._queue_layout = QVBoxLayout(self._queue_container)
        self._queue_layout.setContentsMargins(0, 0, 0, 0)
        self._queue_layout.setSpacing(8)

        self._drop_hint = QLabel("Drop video files here\nor use File \u2192 Add Files")
        self._drop_hint.setObjectName("dropHint")
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setMinimumHeight(100)
        self._queue_layout.addWidget(self._drop_hint)
        self._queue_layout.addStretch()

        scroll.setWidget(self._queue_container)
        queue_inner.addWidget(scroll)

        q_btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Files...")
        add_btn.setObjectName("addBtn")
        add_btn.setFixedHeight(34)
        add_btn.setMinimumWidth(130)
        add_btn.clicked.connect(self._on_add_files)
        q_btn_row.addWidget(add_btn)

        self._start_btn = QPushButton("\u25B6  Start")
        self._start_btn.setObjectName("addBtn")
        self._start_btn.setFixedHeight(34)
        self._start_btn.clicked.connect(self._on_start_queue)
        q_btn_row.addWidget(self._start_btn)

        q_btn_row.addStretch()
        queue_inner.addLayout(q_btn_row)
        main_layout.addWidget(queue_group, stretch=3)

        # ---- Completed Group ----
        completed_group = QGroupBox("Completed")
        completed_group.setObjectName("queueGroup")
        completed_inner = QVBoxLayout(completed_group)
        completed_inner.setContentsMargins(8, 12, 8, 8)
        completed_inner.setSpacing(8)

        completed_scroll = QScrollArea()
        completed_scroll.setWidgetResizable(True)
        completed_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        completed_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._completed_container = QWidget()
        self._completed_layout = QVBoxLayout(self._completed_container)
        self._completed_layout.setContentsMargins(0, 0, 0, 0)
        self._completed_layout.setSpacing(8)

        self._completed_empty = QLabel("Completed items will appear here")
        self._completed_empty.setObjectName("dropHint")
        self._completed_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._completed_empty.setMinimumHeight(40)
        self._completed_layout.addWidget(self._completed_empty)
        self._completed_layout.addStretch()

        completed_scroll.setWidget(self._completed_container)
        completed_inner.addWidget(completed_scroll)

        c_btn_row = QHBoxLayout()
        c_btn_row.addStretch()
        clear_complete_btn = QPushButton("Clear Completed")
        clear_complete_btn.setFixedHeight(30)
        clear_complete_btn.clicked.connect(self._on_clear_complete)
        c_btn_row.addWidget(clear_complete_btn)
        completed_inner.addLayout(c_btn_row)

        main_layout.addWidget(completed_group, stretch=2)

        # ---- Output Destination ----
        output_group = QGroupBox("Output Destination")
        output_group.setObjectName("outputGroup")
        output_layout = QHBoxLayout(output_group)
        output_layout.setContentsMargins(10, 8, 10, 8)
        output_layout.setSpacing(8)

        self._output_combo = QComboBox()
        self._output_combo.setObjectName("outputCombo")
        self._output_combo.setMinimumHeight(30)
        self._output_combo.currentIndexChanged.connect(self._on_output_combo_changed)
        output_layout.addWidget(self._output_combo, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedHeight(30)
        browse_btn.clicked.connect(self._on_pick_output_dir)
        output_layout.addWidget(browse_btn)

        main_layout.addWidget(output_group)

        # ---- Log ----
        self._log_group = QGroupBox("Log")
        self._log_group.setObjectName("outputGroup")
        self._log_group.setMaximumHeight(110)
        log_layout = QVBoxLayout(self._log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(
            "background-color: #1a1a28; color: #aaa; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; border: none;"
        )
        log_layout.addWidget(self._log_text)
        main_layout.addWidget(self._log_group)

        # ---- Status Bar ----
        status_row = QHBoxLayout()
        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("statusBar")
        status_row.addWidget(self._status_label, stretch=1)

        self._gpu_label = QLabel("")
        self._gpu_label.setObjectName("gpuInfo")
        status_row.addWidget(self._gpu_label)
        main_layout.addLayout(status_row)

    def _log(self, msg: str):
        self._log_text.append(msg)
        # Cap log at 500 lines to prevent memory growth
        doc = self._log_text.document()
        if doc.blockCount() > 500:
            cursor = self._log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, doc.blockCount() - 400)
            cursor.removeSelectedText()
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ================================================================
    # DRAG AND DROP
    # ================================================================

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(VIDEO_EXTENSIONS):
                    event.acceptProposedAction()
                    return

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [
            Path(url.toLocalFile()) for url in event.mimeData().urls()
            if url.toLocalFile().lower().endswith(VIDEO_EXTENSIONS)
        ]
        added = self._add_files_to_queue(sorted(paths))
        if added and self._auto_start:
            self._try_start_next()

    # ================================================================
    # FILE / FOLDER
    # ================================================================

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", VIDEO_FILTER_STR)
        if not files:
            return
        added = self._add_files_to_queue([Path(f) for f in files])
        if added and self._auto_start:
            self._try_start_next()

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Videos")
        if not folder:
            return
        video_files = sorted(
            f for f in Path(folder).iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )
        added = self._add_files_to_queue(video_files)
        if added and self._auto_start:
            self._try_start_next()

    def _add_files_to_queue(self, paths: list[Path]) -> int:
        existing = {str(item.input_path) for item in self.queue_items}
        added = 0
        fmt = self._settings.value("output_format", "auto")
        mult = self.processor.fps_multiplier

        for input_path in paths:
            if str(input_path) in existing:
                self._log(f"Skipped (already in queue): {input_path.name}")
                continue
            try:
                info = probe_video(input_path)
                file_info = {
                    'width': info.width, 'height': info.height,
                    'fps': info.fps, 'duration': info.duration,
                    'codec': info.codec, 'frame_count': info.frame_count,
                }
            except Exception as e:
                self._log(f"Error probing {input_path.name}: {e}")
                continue

            output_path = make_output_path(input_path, self._output_dir, mult, info.fps)
            if fmt != "auto":
                output_path = output_path.with_suffix(fmt)

            self._add_queue_item(input_path, output_path, file_info)
            self._analysis_cache.submit(input_path)  # pre-analyze in background
            added += 1

        if added:
            self._log(f"Added {added} file(s)")
        return added

    # ================================================================
    # OUTPUT DESTINATION
    # ================================================================

    def _on_pick_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            str(self._output_dir) if self._output_dir else "",
        )
        if not dir_path:
            return
        self._output_dir = Path(dir_path)
        self._settings.setValue(SETTINGS_KEY_OUTPUT_DIR, dir_path)
        self._add_recent_dir(dir_path)
        self._rebuild_output_combo()

    def _on_reset_output_dir(self):
        self._output_dir = None
        self._settings.remove(SETTINGS_KEY_OUTPUT_DIR)
        self._rebuild_output_combo()

    def _on_output_combo_changed(self, index: int):
        if index < 0:
            return
        data = self._output_combo.currentData()
        if data == "__same__":
            self._output_dir = None
            self._settings.remove(SETTINGS_KEY_OUTPUT_DIR)
        elif data == "__browse__":
            self._on_pick_output_dir()
        elif data:
            self._output_dir = Path(data)
            self._settings.setValue(SETTINGS_KEY_OUTPUT_DIR, data)

    def _rebuild_output_combo(self):
        self._output_combo.blockSignals(True)
        self._output_combo.clear()
        self._output_combo.addItem("Same folder as input file", "__same__")
        recents = self._get_recent_dirs()
        if recents:
            self._output_combo.insertSeparator(self._output_combo.count())
            for d in recents:
                p = Path(d)
                label = f"{p.parent.name}/{p.name}" if p.parent.name else str(p)
                self._output_combo.addItem(f"\U0001F4C1  {label}", str(d))
        self._output_combo.insertSeparator(self._output_combo.count())
        self._output_combo.addItem("\U0001F50D  Browse for folder...", "__browse__")
        if self._output_dir:
            for i in range(self._output_combo.count()):
                if self._output_combo.itemData(i) == str(self._output_dir):
                    self._output_combo.setCurrentIndex(i)
                    break
        else:
            self._output_combo.setCurrentIndex(0)
        self._output_combo.blockSignals(False)

    def _get_recent_dirs(self) -> list[str]:
        raw = self._settings.value("recent_output_dirs", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        return [d for d in raw if Path(d).is_dir()]

    def _add_recent_dir(self, dir_path: str):
        recents = self._get_recent_dirs()
        if dir_path in recents:
            recents.remove(dir_path)
        recents.insert(0, dir_path)
        self._settings.setValue("recent_output_dirs", recents[:8])

    # ================================================================
    # SETTINGS
    # ================================================================

    def _on_set_resolution(self, value):
        for v, a in self._res_actions.items(): a.setChecked(v == value)
        self._settings.setValue("max_height", value)
        self.processor.max_height = value if value > 0 else None
        self._analysis_cache.update_max_height(value if value > 0 else None)

    def _on_set_fps_mult(self, value):
        for v, a in self._fps_actions.items(): a.setChecked(v == value)
        self._settings.setValue("fps_multiplier", value)
        self.processor.fps_multiplier = value

    def _on_set_codec(self, value):
        for v, a in self._codec_actions.items(): a.setChecked(v == value)
        self._settings.setValue("codec", value)
        self.processor.codec = value

    def _on_set_quality(self, value):
        for v, a in self._quality_actions.items(): a.setChecked(v == value)
        self._settings.setValue("crf", value)
        self.processor.crf = value

    def _on_set_format(self, value):
        for v, a in self._fmt_actions.items(): a.setChecked(v == value)
        self._settings.setValue("output_format", value)

    def _on_set_player(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video Player",
            str(Path(self._vlc_path).parent) if self._vlc_path else "",
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self._vlc_path = path
            self._settings.setValue("video_player", path)
            self._log(f"Video player set: {Path(path).name}")

    def get_player_path(self) -> str | None:
        """Return configured video player path, or None for system default."""
        return self._vlc_path if self._vlc_path else None

    def _on_about(self):
        from app.main import VERSION
        QMessageBox.about(
            self, "About AInimotion",
            f"<h3>AInimotion v{VERSION}</h3>"
            "<p>Anime frame interpolation powered by deep learning.</p>"
            "<p>Multi-frame temporal attention model trained specifically for anime.</p>"
            "<p style='color: #888;'>github.com/TWhit229/AInimotion</p>"
        )

    # ================================================================
    # QUEUE ITEM MANAGEMENT
    # ================================================================

    # ---- Queue Drag-and-Drop Reorder ----

    def _queue_drag_enter(self, event):
        if event.mimeData().hasFormat("application/x-ainimotion-queue"):
            event.acceptProposedAction()

    def _queue_drag_move(self, event):
        if event.mimeData().hasFormat("application/x-ainimotion-queue"):
            event.acceptProposedAction()

    def _queue_drop(self, event):
        if not event.mimeData().hasFormat("application/x-ainimotion-queue"):
            return

        # Find the dragged item (the drag source)
        source = event.source()
        if not isinstance(source, QueueItemWidget) or source not in self.queue_items:
            return

        # Find drop position by Y coordinate
        drop_y = event.position().y()
        target_idx = len(self.queue_items)  # default: end
        for i, item in enumerate(self.queue_items):
            widget_y = item.pos().y() + item.height() / 2
            if drop_y < widget_y:
                target_idx = i
                break

        # Move the item
        old_idx = self.queue_items.index(source)
        if old_idx == target_idx or old_idx + 1 == target_idx:
            return  # no change

        self.queue_items.remove(source)
        if target_idx > old_idx:
            target_idx -= 1
        self.queue_items.insert(target_idx, source)

        # Rebuild layout order (offset by 1 for hidden drop_hint at index 0)
        self._queue_layout.removeWidget(source)
        self._queue_layout.insertWidget(target_idx + 1, source)

        event.acceptProposedAction()
        self._log(f"Moved {source.input_path.name} to position {target_idx + 1}")

    # ---- Queue Item Management ----

    def _add_queue_item(self, input_path, output_path, file_info=None):
        item = QueueItemWidget(input_path, output_path, file_info=file_info)
        item.remove_requested.connect(self._on_remove_item)
        item.pause_resume_requested.connect(self._on_pause_resume)
        item.retry_requested.connect(self._on_retry)
        item.start_now_requested.connect(self._on_start_now)

        self._queue_layout.insertWidget(self._queue_layout.count() - 1, item)
        self.queue_items.append(item)
        self._drop_hint.setVisible(False)
        self._update_status()

    def _move_to_completed(self, item: QueueItemWidget):
        """Move an item from Queue to Completed section."""
        self._analysis_cache.evict(item.input_path)
        if item in self.queue_items:
            self.queue_items.remove(item)
        self._queue_layout.removeWidget(item)
        self._drop_hint.setVisible(len(self.queue_items) == 0)

        self._completed_layout.insertWidget(0, item)  # newest at top
        self.completed_items.append(item)
        self._completed_empty.setVisible(False)
        self._update_status()

    def _on_remove_item(self, item: QueueItemWidget):
        self._analysis_cache.evict(item.input_path)
        self._analysis_cache.cancel(item.input_path)
        if item.state in (ItemState.PROCESSING, ItemState.PAUSED):
            if self._current_item == item and self._current_thread:
                # Disconnect signals BEFORE cancelling to prevent use-after-free
                try:
                    self._current_thread.progress_updated.disconnect()
                    self._current_thread.finished_processing.disconnect()
                except TypeError:
                    pass
                self._current_thread.cancel()
                self._current_thread = None
                self._current_item = None
            elif self._suspended_item == item and self._suspended_thread:
                try:
                    self._suspended_thread.progress_updated.disconnect()
                    self._suspended_thread.finished_processing.disconnect()
                except TypeError:
                    pass
                self._suspended_thread.cancel()
                self._suspended_thread = None
                self._suspended_item = None
        self._remove_item(item)

    def _remove_item(self, item: QueueItemWidget):
        if item in self.queue_items:
            self.queue_items.remove(item)
            self._drop_hint.setVisible(len(self.queue_items) == 0)
        if item in self.completed_items:
            self.completed_items.remove(item)
            self._completed_empty.setVisible(len(self.completed_items) == 0)
        self._queue_layout.removeWidget(item)
        self._completed_layout.removeWidget(item)
        item.deleteLater()
        self._update_status()

    def _on_start_now(self, item: QueueItemWidget):
        """Preempt: pause current job, start this one immediately."""
        if item.state != ItemState.QUEUED:
            return

        # Move item to top of queue
        if item in self.queue_items:
            self.queue_items.remove(item)
            self._queue_layout.removeWidget(item)
            self.queue_items.insert(0, item)
            self._queue_layout.insertWidget(1, item)  # 0 = drop_hint

        # Cancel any existing suspended thread first (prevent orphaning)
        if self._suspended_thread and self._suspended_thread.isRunning():
            try:
                self._suspended_thread.progress_updated.disconnect()
                self._suspended_thread.finished_processing.disconnect()
            except TypeError:
                pass
            self._suspended_thread.cancel()
            if self._suspended_item:
                self._suspended_item.set_state(ItemState.QUEUED)
            self._suspended_thread = None
            self._suspended_item = None

        # If something is currently processing, suspend it
        if self._current_thread and self._current_thread.isRunning() and self._current_item:
            self._current_thread.pause()
            self._current_item.set_state(ItemState.PAUSED)
            self._suspended_thread = self._current_thread
            self._suspended_item = self._current_item
            self._log(f"Suspended: {self._current_item.input_path.name}")
            self._current_thread = None
            self._current_item = None

        self.processor.reset_stop()
        self._try_start_next()

    def _on_pause_resume(self, item):
        if item.state == ItemState.PROCESSING and self._current_thread:
            self._current_thread.pause()
            item.set_state(ItemState.PAUSED)
            self._log("Paused")
        elif item.state == ItemState.PAUSED:
            if self._current_item == item and self._current_thread:
                self._current_thread.resume()
                item.set_state(ItemState.PROCESSING)
                self._log("Resumed")
            # Don't resume suspended thread while current is running (GPU contention)
        self._update_status()

    def _on_retry(self, item):
        # Move back from completed to queue
        if item in self.completed_items:
            self.completed_items.remove(item)
            self._completed_layout.removeWidget(item)
            self._completed_empty.setVisible(len(self.completed_items) == 0)
            self._queue_layout.insertWidget(self._queue_layout.count() - 1, item)
            self.queue_items.append(item)
            self._drop_hint.setVisible(False)
        item.set_state(ItemState.QUEUED)
        self._log(f"Retrying: {item.input_path.name}")
        self._update_status()
        self._try_start_next()

    def _on_toggle_pause(self):
        for item in self.queue_items:
            if item.state in (ItemState.PROCESSING, ItemState.PAUSED):
                self._on_pause_resume(item)
                return

    def _on_clear_complete(self):
        for item in list(self.completed_items):
            self._remove_item(item)

    def _on_clear_all(self):
        self.processor.stop_all()
        # Cancel threads properly (disconnect signals to prevent crashes)
        if self._suspended_thread and self._suspended_thread.isRunning():
            try:
                self._suspended_thread.progress_updated.disconnect()
                self._suspended_thread.finished_processing.disconnect()
            except TypeError:
                pass
            self._suspended_thread.cancel()
            self._suspended_thread = None
            self._suspended_item = None
        if self._current_thread and self._current_thread.isRunning():
            try:
                self._current_thread.progress_updated.disconnect()
                self._current_thread.finished_processing.disconnect()
            except TypeError:
                pass
            self._current_thread.cancel()
            self._current_thread = None
            self._current_item = None
        for item in list(self.queue_items):
            self._remove_item(item)
        for item in list(self.completed_items):
            self._remove_item(item)

    def _on_start_queue(self):
        queued = [i for i in self.queue_items if i.state == ItemState.QUEUED]
        if not queued:
            self._log("No videos in queue to process")
            return
        self.processor.reset_stop()
        self._try_start_next()

    def _on_stop_all(self):
        self.processor.stop_all()
        if self._suspended_thread and self._suspended_thread.isRunning():
            try:
                self._suspended_thread.progress_updated.disconnect()
                self._suspended_thread.finished_processing.disconnect()
            except TypeError:
                pass
            self._suspended_thread.cancel()
            if self._suspended_item:
                self._suspended_item.set_state(ItemState.CANCELLED)
            self._suspended_thread = None
            self._suspended_item = None
        if self._current_thread and self._current_thread.isRunning():
            try:
                self._current_thread.progress_updated.disconnect()
                self._current_thread.finished_processing.disconnect()
            except TypeError:
                pass
            self._current_thread.cancel()
            if self._current_item:
                self._current_item.set_state(ItemState.CANCELLED)
            self._current_thread = None
            self._current_item = None
        self._log("Queue stopped")

    # ================================================================
    # PROCESSING
    # ================================================================

    def _try_start_next(self):
        if self.processor.is_stopped:
            return
        if self._current_thread is not None and self._current_thread.isRunning():
            return

        next_item = None
        for item in self.queue_items:
            if item.state == ItemState.QUEUED:
                next_item = item
                break

        if next_item is None:
            self._update_status()
            return

        # Overwrite check (non-blocking: log a warning but proceed)
        if next_item.output_path.exists():
            self._log(f"Overwriting: {next_item.output_path.name}")

        next_item.set_state(ItemState.PROCESSING)

        # Check for pre-analyzed result
        cached = self._analysis_cache.get(next_item.input_path)
        if cached:
            self._log(f"Processing: {next_item.input_path.name} (pre-analyzed)")
        else:
            self._log(f"Processing: {next_item.input_path.name}")

        thread = ProcessingThread(
            self.processor, str(next_item.input_path), str(next_item.output_path),
            precomputed_analysis=cached,
        )
        thread.progress_updated.connect(lambda p, i=next_item: self._on_progress(i, p))
        thread.finished_processing.connect(lambda s, i=next_item: self._on_finished(i, s))
        self._current_thread = thread
        self._current_item = next_item
        thread.start()
        self._update_status()

    def _on_progress(self, item, progress):
        item.update_progress(progress)

    def _on_finished(self, item, success):
        # Disconnect progress signal to prevent stale updates
        if self._current_thread:
            try:
                self._current_thread.progress_updated.disconnect()
            except TypeError:
                pass

        if success and item.state not in (ItemState.COMPLETE, ItemState.CANCELLED):
            item.set_state(ItemState.COMPLETE)
            self._log(f"Complete: {item.input_path.name}")

        # Only move to completed if actually done (not re-queued by preemption)
        if item.state in (ItemState.COMPLETE, ItemState.ERROR, ItemState.CANCELLED):
            self._move_to_completed(item)

        self._current_thread = None
        self._current_item = None
        self._update_status()

        # Resume suspended job if there is one
        if self._suspended_thread and self._suspended_item:
            self._log(f"Resuming: {self._suspended_item.input_path.name}")
            self._current_thread = self._suspended_thread
            self._current_item = self._suspended_item
            self._suspended_thread = None
            self._suspended_item = None
            self._current_thread.resume()
            self._current_item.set_state(ItemState.PROCESSING)
            self._update_status()
            return

        if not self.processor.is_stopped:
            queued_remaining = sum(1 for i in self.queue_items if i.state == ItemState.QUEUED)
            if queued_remaining > 0:
                self._try_start_next()
            else:
                self._log("Queue complete!")
                try:
                    QApplication.alert(self, 0)
                except Exception:
                    pass

    # ================================================================
    # STATUS BAR
    # ================================================================

    def _update_status(self):
        processing = sum(1 for i in self.queue_items if i.state in (ItemState.PROCESSING, ItemState.PAUSED))
        queued = sum(1 for i in self.queue_items if i.state == ItemState.QUEUED)
        complete = len(self.completed_items)
        total = len(self.queue_items) + complete

        if processing > 0:
            paused = sum(1 for i in self.queue_items if i.state == ItemState.PAUSED)
            s = "Paused" if paused else "Processing"
            self._status_label.setText(f"{s}  |  {queued} queued  |  {complete} complete  |  {total} total")
        elif queued > 0:
            self._status_label.setText(f"{queued} queued  |  {complete} complete")
        elif total > 0:
            self._status_label.setText(f"All done  |  {complete} complete")
        else:
            self._status_label.setText("Ready \u2014 add video files to begin")

    def _update_gpu_info(self):
        if not torch.cuda.is_available():
            self._gpu_label.setText("GPU: not available")
            return
        try:
            name = torch.cuda.get_device_name(0)
            name = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self._gpu_label.setText(f"GPU: {name} ({allocated:.1f} / {total:.0f} GB)")
        except Exception:
            self._gpu_label.setText("GPU: unknown")

    def closeEvent(self, event):
        self._settings.setValue(SETTINGS_KEY_WINDOW_GEOMETRY, self.saveGeometry())
        self._gpu_timer.stop()
        self._analysis_cache.shutdown()

        # Disconnect all signals to prevent callbacks during destruction
        for thread in [self._suspended_thread, self._current_thread]:
            if thread and thread.isRunning():
                try:
                    thread.progress_updated.disconnect()
                    thread.finished_processing.disconnect()
                except TypeError:
                    pass
                thread.cancel()

        if self._suspended_thread and self._suspended_thread.isRunning():
            self._suspended_thread.wait(3000)
        if self._current_thread and self._current_thread.isRunning():
            self._current_thread.wait(5000)
        event.accept()
