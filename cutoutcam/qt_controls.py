from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import (
        QImage as _QImage,
        QIntValidator as _QIntValidator,
        QPixmap as _QPixmap,
    )
    from PySide6.QtWidgets import (
        QApplication as _QApplication,
        QCheckBox as _QCheckBox,
        QFileDialog as _QFileDialog,
        QFormLayout as _QFormLayout,
        QGroupBox as _QGroupBox,
        QHBoxLayout as _QHBoxLayout,
        QLabel as _QLabel,
        QLineEdit as _QLineEdit,
        QPushButton as _QPushButton,
        QSlider as _QSlider,
        QTabWidget as _QTabWidget,
        QVBoxLayout as _QVBoxLayout,
        QWidget as _QWidget,
    )

    Qt: Any = _Qt
    QImage: Any = _QImage
    QIntValidator: Any = _QIntValidator
    QPixmap: Any = _QPixmap
    QApplication: Any = _QApplication
    QCheckBox: Any = _QCheckBox
    QFileDialog: Any = _QFileDialog
    QFormLayout: Any = _QFormLayout
    QGroupBox: Any = _QGroupBox
    QHBoxLayout: Any = _QHBoxLayout
    QLabel: Any = _QLabel
    QLineEdit: Any = _QLineEdit
    QPushButton: Any = _QPushButton
    QSlider: Any = _QSlider
    QTabWidget: Any = _QTabWidget
    QVBoxLayout: Any = _QVBoxLayout
    QWidget: Any = _QWidget
    QT_AVAILABLE = True
except ImportError:
    Qt = None
    QImage = None
    QIntValidator = None
    QPixmap = None
    QApplication = None
    QCheckBox = None
    QFileDialog = None
    QFormLayout = None
    QGroupBox = None
    QHBoxLayout = None
    QLabel = None
    QLineEdit = None
    QPushButton = None
    QSlider = None
    QTabWidget = None
    QVBoxLayout = None
    QWidget = object
    QT_AVAILABLE = False


class _QtControlWindow(QWidget):  # type: ignore[misc]
    def __init__(self, on_close: Callable[[], None]) -> None:
        super().__init__()
        self._on_close = on_close

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._on_close()
        super().closeEvent(event)


class _UndoLineEdit(QLineEdit):  # type: ignore[misc]
    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        modifiers = event.modifiers()
        key = event.key()

        if modifiers & Qt.ControlModifier:
            if key == Qt.Key_Z and modifiers & Qt.ShiftModifier:
                self.redo()
                event.accept()
                return
            if key == Qt.Key_Z:
                self.undo()
                event.accept()
                return
            if key == Qt.Key_Y:
                self.redo()
                event.accept()
                return

        super().keyPressEvent(event)


class QtLayoutControls:
    uses_opencv_window = False

    def __init__(self) -> None:
        if not QT_AVAILABLE:
            raise RuntimeError("PySide6 is not installed.")

        self._app = QApplication.instance() or QApplication([])
        self._action_events = {key: False for key in ACTION_BUTTONS}
        self._values: dict[str, LayoutValue] = {}
        self._window: _QtControlWindow | None = None
        self._tabs: Any | None = None
        self._value_labels: dict[str, Any] = {}
        self._value_inputs: dict[str, Any] = {}
        self._sliders: dict[str, Any] = {}
        self._footer_label: Any | None = None
        self._points_toggle: Any | None = None
        self._tracking_toggle: Any | None = None
        self._hands_toggle: Any | None = None
        self._mic_toggle: Any | None = None
        self._character_label: Any | None = None
        self._points_label: Any | None = None
        self._points_group: Any | None = None

    @staticmethod
    def _sync_input_text(input_widget: Any, value: int) -> None:
        if input_widget is None:
            return
        value_text = str(value)
        if input_widget.hasFocus():
            return
        if input_widget.text() != value_text:
            input_widget.setText(value_text)

    @staticmethod
    def load_settings(path: Path) -> dict[str, LayoutValue]:
        return LayoutControls.load_settings(path)

    @staticmethod
    def save_settings(path: Path, values: dict[str, LayoutValue]) -> None:
        LayoutControls.save_settings(path, values)

    def _set_action(self, name: str) -> None:
        self._action_events[name] = True

    def _build_slider_row(self, key: str, initial_value: int) -> Any:
        slider = QSlider(Qt.Horizontal)
        _default, low, high = TRACKBARS[key]
        slider.setRange(low, high)
        slider.setValue(initial_value)
        self._sliders[key] = slider

        value_label = QLabel(str(initial_value))
        value_label.setMinimumWidth(48)
        self._value_labels[key] = value_label

        value_input = _UndoLineEdit(str(initial_value))
        value_input.setMinimumWidth(60)
        value_input.setMaximumWidth(72)
        value_input.setAlignment(Qt.AlignRight)
        value_input.setValidator(QIntValidator(low, high, value_input))
        self._value_inputs[key] = value_input

        def on_change(value: int, *, key_name: str = key) -> None:
            self._values[key_name] = value
            self._value_labels[key_name].setText(str(value))
            input_widget = self._value_inputs.get(key_name)
            self._sync_input_text(input_widget, value)
            self._refresh_footer()

        slider.valueChanged.connect(on_change)

        def apply_input(*_args: object, key_name: str = key, min_value: int = low, max_value: int = high) -> None:
            input_widget = self._value_inputs.get(key_name)
            slider_widget = self._sliders.get(key_name)
            if input_widget is None or slider_widget is None:
                return
            try:
                value = int(input_widget.text())
            except ValueError:
                value = int(self._values.get(key_name, initial_value))
            value = max(min_value, min(max_value, value))
            input_widget.setText(str(value))
            slider_widget.setValue(value)

        value_input.editingFinished.connect(apply_input)
        value_input.returnPressed.connect(apply_input)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)
        layout.addWidget(value_input)
        return row

    def _build_tab(self, page_idx: int) -> Any:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)
        for key in TRACKBAR_PAGES[page_idx]:
            value = int(self._values.get(key, TRACKBARS[key][0]))
            form.addRow(TRACKBAR_LABELS.get(key, key.replace("_", " ")), self._build_slider_row(key, value))
        return tab

    def _build_preview_panel(self, title: str) -> tuple[Any, Any]:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        preview = QLabel("No frame yet")
        preview.setMinimumSize(480, 270)
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet("background:#20252b; color:#d7dee9; border:1px solid #3a4048;")
        layout.addWidget(preview)
        return group, preview

    def create_window(self, initial_values: dict[str, LayoutValue]) -> dict[str, tuple[int, int]]:
        self._values = initial_values.copy()
        self._window = _QtControlWindow(lambda: self._set_action("quit"))
        self._window.setWindowTitle(CONTROL_WINDOW)
        self._window.resize(1440, 900)

        root = QHBoxLayout(self._window)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_panel.setMinimumWidth(420)
        left_panel.setMaximumWidth(520)

        self._tabs = QTabWidget()
        for page_idx, _page in enumerate(TRACKBAR_PAGES):
            self._tabs.addTab(self._build_tab(page_idx), PAGE_LABELS.get(page_idx, f"Page {page_idx + 1}"))
        self._tabs.currentChanged.connect(lambda _index: self._refresh_footer())
        left_layout.addWidget(self._tabs, stretch=1)

        buttons_grid = (
            ("page_prev", "page_next"),
            ("save", "quit"),
            ("pick_bg", "clear_bg"),
        )
        for row_keys in buttons_grid:
            row = QHBoxLayout()
            row.setSpacing(8)
            for key in row_keys:
                button = QPushButton(ACTION_BUTTONS[key][2])
                button.setMinimumHeight(34)
                if key == "page_prev":
                    button.clicked.connect(lambda _checked=False: self._goto_relative_page(-1))
                elif key == "page_next":
                    button.clicked.connect(lambda _checked=False: self._goto_relative_page(1))
                else:
                    button.clicked.connect(lambda _checked=False, name=key: self._set_action(name))
                row.addWidget(button)
            left_layout.addLayout(row)

        self._points_toggle = QCheckBox("Show points section")
        self._points_toggle.setChecked(int(self._values.get("ui_show_points", 1)) > 0)
        self._points_toggle.toggled.connect(self._toggle_points_section)
        left_layout.addWidget(self._points_toggle)

        toggles_row = QHBoxLayout()
        toggles_row.setSpacing(8)
        self._tracking_toggle = QCheckBox("Tracking")
        self._tracking_toggle.setChecked(int(self._values.get("ui_tracking_enabled", 1)) > 0)
        self._hands_toggle = QCheckBox("Hands")
        self._hands_toggle.setChecked(int(self._values.get("ui_hands_enabled", 1)) > 0)
        self._mic_toggle = QCheckBox("Mic")
        self._mic_toggle.setChecked(int(self._values.get("ui_mic_enabled", 1)) > 0)
        toggles_row.addWidget(self._tracking_toggle)
        toggles_row.addWidget(self._hands_toggle)
        toggles_row.addWidget(self._mic_toggle)
        left_layout.addLayout(toggles_row)

        self._footer_label = QLabel("")
        left_layout.addWidget(self._footer_label)
        self._refresh_footer()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        character_group, self._character_label = self._build_preview_panel("Character Output")
        points_group, self._points_label = self._build_preview_panel("Points Preview")
        self._points_group = points_group
        right_layout.addWidget(character_group, stretch=1)
        right_layout.addWidget(points_group, stretch=1)

        root.addWidget(left_panel, stretch=1)
        root.addWidget(right_panel, stretch=2)

        self._window.show()
        return {key: (low, high) for key, (_default, low, high) in TRACKBARS.items()}

    def _toggle_points_section(self, checked: bool) -> None:
        if self._points_group is not None:
            self._points_group.setVisible(bool(checked))

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

    def _refresh_footer(self) -> None:
        if self._footer_label is not None:
            self._footer_label.setText(self._footer_text())

    @staticmethod
    def _to_pixmap(frame_bgr) -> Any:
        rgb = frame_bgr[:, :, ::-1].copy()
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def read_values(self, _limits: dict[str, tuple[int, int]]) -> dict[str, int]:
        values = {key: int(value) for key, value in self._values.items() if key in TRACKBARS}
        values["ui_show_points"] = 1 if self.is_points_preview_enabled() else 0
        values["ui_tracking_enabled"] = 1 if self.is_tracking_enabled() else 0
        values["ui_hands_enabled"] = 1 if self.is_hands_enabled() else 0
        values["ui_mic_enabled"] = 1 if self.is_mic_enabled() else 0
        return values

    def draw_overlay(self, current_layout: dict[str, LayoutValue]):
        self._values.update(current_layout)
        for key, slider in self._sliders.items():
            value = int(self._values.get(key, TRACKBARS[key][0]))
            if slider.value() != value:
                slider.setValue(value)
            input_widget = self._value_inputs.get(key)
            self._sync_input_text(input_widget, value)
        self._refresh_footer()
        return None

    def update_previews(self, points_frame, character_frame) -> None:
        if self._character_label is not None:
            self._character_label.setPixmap(
                self._to_pixmap(character_frame).scaled(
                    self._character_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        if (
            points_frame is not None
            and self._points_label is not None
            and self._points_toggle is not None
            and self._points_toggle.isChecked()
        ):
            self._points_label.setPixmap(
                self._to_pixmap(points_frame).scaled(
                    self._points_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        elif self._points_label is not None:
            self._points_label.setText("Points preview disabled")
            self._points_label.setPixmap(QPixmap())

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

    def is_tracking_enabled(self) -> bool:
        return bool(self._tracking_toggle.isChecked()) if self._tracking_toggle is not None else True

    def is_hands_enabled(self) -> bool:
        return bool(self._hands_toggle.isChecked()) if self._hands_toggle is not None else True

    def is_mic_enabled(self) -> bool:
        return bool(self._mic_toggle.isChecked()) if self._mic_toggle is not None else True

    def is_points_preview_enabled(self) -> bool:
        return bool(self._points_toggle.isChecked()) if self._points_toggle is not None else True
