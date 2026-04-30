"""
AInimotion — Anime Frame Interpolation GUI.

Entry point: python -m app
"""

import sys
from pathlib import Path

VERSION = "1.0.0"


def _find_base_dir() -> Path:
    """Base directory: next to the exe (frozen) or project root (dev)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


def main():
    # Windows taskbar icon fix — must be before QApplication
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'AInimotion.AInimotion.1'
        )
    except Exception:
        pass

    base = _find_base_dir()
    default_model = base / "model" / "ainimotion.pt"
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(default_model)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Looked in: {model_path}")
        print(f"Place ainimotion.pt in: {base / 'model'}")
        sys.exit(1)

    print("Loading model...")
    from app.core.processor import VideoProcessor
    processor = VideoProcessor(model_path)
    print("Model ready.\n")

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow
    from app.ui.theme import DARK_THEME

    app = QApplication(sys.argv)
    app.setApplicationName("AInimotion")
    app.setOrganizationName("AInimotion")
    app.setApplicationVersion(VERSION)
    app.setStyleSheet(DARK_THEME)

    # Set icon on app AND as fallback for all windows
    icon_path = base / "assets" / "ainimotion.ico"
    if not icon_path.exists():
        icon_path = base / "assets" / "logo_256.png"
    if icon_path.exists():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

    window = MainWindow(processor)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
