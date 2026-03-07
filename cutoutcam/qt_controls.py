from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .app_controls import (
    ACTION_BUTTONS,
    BG_MODE_LABELS,
    CONTROL_WINDOW,
    LayoutControls,
    LayoutValue,
    PAGE_LABELS,
    TRACKBARS,
    TRACKBAR_LABELS,
    TRACKBAR_PAGES,
)

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except ImportError:
    QApplication = None
    QFileDialog = None
    QLabel = None
    QPushButton = None
    QSlider = None
    QTabWidget = None
    QVBoxLayout = None
    QHBoxLayout = None
    QFormLayout = None
    QWidget = object
    Qt = None
    QT_AVAILABLE = False


class _QtControlWindow(QWidget):  # type: ignore[misc]
    def __init__(self, on_close: Callable[[], None]) -> None:
        super().__init__()
        self._on_close = on_close

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._on_close()
        super().closeEvent(event)


class QtLayoutControls:
    uses_opencv_window = False

    def __init__(self) -> None:
        if not QT_AVAILABLE:
            raise RuntimeError("PySide6 is not installed.")

        self._app = QApplication.instance() or QApplication([])
        self._action_events = {key: False for key in ACTION_BUTTONS}
        self._values: dict[str, LayoutValue] = {}
        self._window: _QtControlWindow | None = None
        self._tabs: QTabWidget | None = None
        self._value_labels: dict[str, QLabel] = {}

    @staticmethod
    def load_settings(path: Path) -> dict[str, LayoutValue]:
        return LayoutControls.load_settings(path)

    @staticmethod
    def save_settings(path: Path, values: dict[str, LayoutValue]) -> None:
        LayoutControls.save_settings(path, values)

    def _set_action(self, name: str) -> None:
        self._action_events[name] = True

    def _build_slider_row(self, key: str, initial_value: int):
        slider = QSlider(Qt.Horizontal)
        _default, low, high = TRACKBARS[key]
        slider.setRange(low, high)
        slider.setValue(initial_value)

        value_label = QLabel(str(initial_value))
        value_label.setMinimumWidth(52)
        self._value_labels[key] = value_label

        def on_change(value: int, *, key_name: str = key) -> None:
            self._values[key_name] = value
            self._value_labels[key_name].setText(str(value))

        slider.valueChanged.connect(on_change)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)
        return row

    def _build_tab(self, page_idx: int) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        for key in TRACKBAR_PAGES[page_idx]:
            value = int(self._values.get(key, TRACKBARS[key][0]))
            form.addRow(TRACKBAR_LABELS.get(key, key.replace("_", " ")), self._build_slider_row(key, value))
        return tab

    def create_window(self, initial_values: dict[str, LayoutValue]) -> dict[str, tuple[int, int]]:
        self._values = initial_values.copy()
        self._window = _QtControlWindow(lambda: self._set_action("quit"))
        self._window.setWindowTitle(CONTROL_WINDOW)
        self._window.resize(520, 760)

        root = QVBoxLayout(self._window)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self._tabs = QTabWidget()
        for page_idx, _page in enumerate(TRACKBAR_PAGES):
            self._tabs.addTab(self._build_tab(page_idx), PAGE_LABELS.get(page_idx, f"Page {page_idx + 1}"))
        root.addWidget(self._tabs, stretch=1)

        button_rows = (
            ("page_prev", "page_next", "save", "quit"),
            ("pick_bg", "clear_bg"),
        )
        for row_keys in button_rows:
            row = QHBoxLayout()
            row.setSpacing(8)
            for key in row_keys:
                button = QPushButton(ACTION_BUTTONS[key][2])
                button.setMinimumHeight(36)
                if key in ("page_prev", "page_next"):
                    if key == "page_prev":
                        button.clicked.connect(lambda _checked=False: self._goto_relative_page(-1))
                    else:
                        button.clicked.connect(lambda _checked=False: self._goto_relative_page(1))
                else:
                    button.clicked.connect(lambda _checked=False, name=key: self._set_action(name))
                row.addWidget(button)
            root.addLayout(row)

        footer = QLabel(self._footer_text())
        footer.setObjectName("footerLabel")
        root.addWidget(footer)
        self._window.show()
        return {key: (low, high) for key, (_default, low, high) in TRACKBARS.items()}

    def _goto_relative_page(self, delta: int) -> None:
        if self._tabs is None:
            return
        count = self._tabs.count()
        if count <= 0:
            return
        self._tabs.setCurrentIndex((self._tabs.currentIndex() + delta) % count)

    def _footer_text(self) -> str:
        page_idx = self._tabs.currentIndex() if self._tabs is not None else 0
        page_name = PAGE_LABELS.get(page_idx, f"Page {page_idx + 1}")
        mode = int(self._values.get("bg_mode", 0))
        return f"Background: {BG_MODE_LABELS.get(mode, f'Mode {mode}')}    {page_name}"

    def read_values(self, _limits: dict[str, tuple[int, int]]) -> dict[str, int]:
        return {key: int(value) for key, value in self._values.items() if key in TRACKBARS}

    def draw_overlay(self, current_layout: dict[str, LayoutValue]) -> np.ndarray:
        self._values.update(current_layout)
        if self._window is not None:
            footer = self._window.findChild(QLabel, "footerLabel")
            if footer is not None:
                footer.setText(self._footer_text())
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def pop_action(self, name: str) -> bool:
        value = self._action_events.get(name, False)
        self._action_events[name] = False
        return value

    def choose_background_file(self) -> str | None:
        if QFileDialog is None or self._window is None:
            return None
        path, _selected_filter = QFileDialog.getOpenFileName(
            self._window,
            "Choose background image",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        return path or None

    def process_events(self) -> None:
        self._app.processEvents()

    def close(self) -> None:
        if self._window is not None:
            self._window.close()
            self._window = None
