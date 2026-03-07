from pathlib import Path
import json
from typing import TypeAlias

import cv2
import numpy as np


CONTROL_WINDOW = "Layout Controls"
LayoutValue: TypeAlias = int | str
UI_STATE_DEFAULTS = {
    "ui_show_points": 1,
    "ui_tracking_enabled": 1,
    "ui_hands_enabled": 1,
    "ui_mic_enabled": 1,
}

TRACKBARS = {
    "body_scale": (90, 20, 160),
    "body_x": (0, -200, 200),
    "body_y": (0, -200, 200),
    "hand_enabled": (1, 0, 1),
    "hand_mirror_x": (1, 0, 1),
    "hand_size": (7, 2, 20),
    "hand_x": (0, -200, 200),
    "hand_y": (0, -200, 200),
    "hand_range_x": (70, 20, 150),
    "hand_range_y": (75, 20, 150),
    "hand_shoulder_y": (0, -120, 120),
    "head_scale": (62, 20, 120),
    "head_x": (0, -200, 200),
    "head_y": (0, -200, 200),
    "head_left_scale": (100, 40, 180),
    "head_left_x": (0, -200, 200),
    "head_left_y": (0, -200, 200),
    "head_right_scale": (100, 40, 180),
    "head_right_x": (0, -200, 200),
    "head_right_y": (0, -200, 200),
    "hair_scale": (100, 20, 200),
    "hair_x": (0, -200, 200),
    "hair_y": (0, -200, 200),
    "hair_left_scale": (100, 40, 180),
    "hair_left_x": (0, -200, 200),
    "hair_left_y": (0, -200, 200),
    "hair_right_scale": (100, 40, 180),
    "hair_right_x": (0, -200, 200),
    "hair_right_y": (0, -200, 200),
    "eyes_scale": (55, 20, 200),
    "eyes_x": (0, -150, 150),
    "eyes_y": (0, -150, 150),
    "eyes_left_x": (0, -150, 150),
    "eyes_left_y": (0, -150, 150),
    "eyes_right_x": (0, -150, 150),
    "eyes_right_y": (0, -150, 150),
    "pupil_scale": (4, 1, 20),
    "pupil_x": (0, -100, 100),
    "pupil_y": (0, -100, 100),
    "turn_amp_x": (14, 0, 40),
    "turn_amp_y": (10, 0, 40),
    "closed_eyes_scale": (100, 20, 250),
    "closed_eyes_x": (0, -120, 120),
    "closed_eyes_y": (0, -120, 120),
    "brow_scale": (100, 20, 250),
    "brow_x": (0, -120, 120),
    "brow_y": (0, -120, 120),
    "brow_left_x": (0, -120, 120),
    "brow_left_y": (0, -120, 120),
    "brow_right_x": (0, -120, 120),
    "brow_right_y": (0, -120, 120),
    "mouth_scale": (100, 20, 250),
    "mouth_x": (0, -120, 120),
    "mouth_y": (0, -120, 120),
    "mouth_left_x": (0, -120, 120),
    "mouth_left_y": (0, -120, 120),
    "mouth_right_x": (0, -120, 120),
    "mouth_right_y": (0, -120, 120),
    "speech_active": (0, 0, 1),
    "speech_energy": (0, 0, 100),
    "mic_enabled": (1, 0, 1),
    "mic_threshold": (8, 1, 100),
    "mic_gain": (200, 10, 500),
    "bg_mode": (0, 0, 4),
    "bg_r1": (245, 0, 255),
    "bg_g1": (245, 0, 255),
    "bg_b1": (245, 0, 255),
    "bg_r2": (220, 0, 255),
    "bg_g2": (220, 0, 255),
    "bg_b2": (220, 0, 255),
    "bg_center_x": (0, -100, 100),
    "bg_center_y": (0, -100, 100),
    "bg_scale": (100, 30, 300),
    "bg_x": (0, -600, 600),
    "bg_y": (0, -600, 600),
    "mask_left": (0, 0, 90),
    "mask_right": (0, 0, 90),
    "mask_top": (0, 0, 90),
    "mask_bottom": (0, 0, 90),
}

TRACKBAR_LABELS = {
    "body_scale": "body scale",
    "body_x": "body x",
    "body_y": "body y",
    "hand_enabled": "hands on",
    "hand_mirror_x": "mirror hands x",
    "hand_size": "hand size",
    "hand_x": "hand x",
    "hand_y": "hand y",
    "hand_range_x": "hand range x",
    "hand_range_y": "hand range y",
    "hand_shoulder_y": "shoulder y",
    "head_scale": "head scale",
    "head_x": "head x",
    "head_y": "head y",
    "head_left_scale": "left head scale",
    "head_left_x": "left head x",
    "head_left_y": "left head y",
    "head_right_scale": "right head scale",
    "head_right_x": "right head x",
    "head_right_y": "right head y",
    "hair_scale": "hair scale",
    "hair_x": "hair x",
    "hair_y": "hair y",
    "hair_left_scale": "left hair scale",
    "hair_left_x": "left hair x",
    "hair_left_y": "left hair y",
    "hair_right_scale": "right hair scale",
    "hair_right_x": "right hair x",
    "hair_right_y": "right hair y",
    "eyes_scale": "eyes scale",
    "eyes_x": "eyes x",
    "eyes_y": "eyes y",
    "eyes_left_x": "left eyes x",
    "eyes_left_y": "left eyes y",
    "eyes_right_x": "right eyes x",
    "eyes_right_y": "right eyes y",
    "pupil_scale": "pupil scale",
    "pupil_x": "pupil x",
    "pupil_y": "pupil y",
    "turn_amp_x": "turn amp x",
    "turn_amp_y": "turn amp y",
    "closed_eyes_scale": "closed eyes scale",
    "closed_eyes_x": "closed eyes x",
    "closed_eyes_y": "closed eyes y",
    "brow_scale": "brow scale",
    "brow_x": "brow x",
    "brow_y": "brow y",
    "brow_left_x": "left brow x",
    "brow_left_y": "left brow y",
    "brow_right_x": "right brow x",
    "brow_right_y": "right brow y",
    "mouth_scale": "mouth scale",
    "mouth_x": "mouth x",
    "mouth_y": "mouth y",
    "mouth_left_x": "left mouth x",
    "mouth_left_y": "left mouth y",
    "mouth_right_x": "right mouth x",
    "mouth_right_y": "right mouth y",
    "speech_active": "speech active",
    "speech_energy": "speech energy",
    "mic_enabled": "mic enabled",
    "mic_threshold": "mic threshold",
    "mic_gain": "mic gain",
    "bg_mode": "bg mode",
    "bg_r1": "bg r1",
    "bg_g1": "bg g1",
    "bg_b1": "bg b1",
    "bg_r2": "bg r2",
    "bg_g2": "bg g2",
    "bg_b2": "bg b2",
    "bg_center_x": "bg center x",
    "bg_center_y": "bg center y",
    "bg_scale": "bg scale",
    "bg_x": "bg x",
    "bg_y": "bg y",
    "mask_left": "mask left",
    "mask_right": "mask right",
    "mask_top": "mask top",
    "mask_bottom": "mask bottom",
}

TRACKBAR_PAGES = [
    [
        "body_scale",
        "body_x",
        "body_y",
        "hand_enabled",
        "hand_mirror_x",
        "hand_size",
        "hand_x",
        "hand_y",
        "hand_range_x",
        "hand_range_y",
        "hand_shoulder_y",
        "head_scale",
        "head_x",
        "head_y",
        "hair_scale",
        "hair_x",
        "hair_y",
    ],
    [
        "head_left_scale",
        "head_left_x",
        "head_left_y",
        "head_right_scale",
        "head_right_x",
        "head_right_y",
        "hair_left_scale",
        "hair_left_x",
        "hair_left_y",
        "hair_right_scale",
        "hair_right_x",
        "hair_right_y",
    ],
    [
        "eyes_scale",
        "eyes_x",
        "eyes_y",
        "eyes_left_x",
        "eyes_left_y",
        "eyes_right_x",
        "eyes_right_y",
        "pupil_scale",
        "pupil_x",
        "pupil_y",
        "turn_amp_x",
        "turn_amp_y",
    ],
    [
        "closed_eyes_scale",
        "closed_eyes_x",
        "closed_eyes_y",
        "brow_scale",
        "brow_x",
        "brow_y",
        "brow_left_x",
        "brow_left_y",
        "brow_right_x",
        "brow_right_y",
        "mouth_scale",
        "mouth_x",
        "mouth_y",
        "mouth_left_x",
        "mouth_left_y",
        "mouth_right_x",
        "mouth_right_y",
        "speech_active",
        "speech_energy",
        "mic_enabled",
        "mic_threshold",
        "mic_gain",
    ],
    [
        "bg_mode",
        "bg_r1",
        "bg_g1",
        "bg_b1",
        "bg_r2",
        "bg_g2",
        "bg_b2",
        "bg_center_x",
        "bg_center_y",
        "bg_scale",
        "bg_x",
        "bg_y",
    ],
    ["mask_left", "mask_right", "mask_top", "mask_bottom"],
]

ACTION_BUTTONS = {
    "page_prev": ((20, 16), (145, 56), "Prev"),
    "page_next": ((165, 16), (290, 56), "Next"),
    "save": ((310, 16), (435, 56), "Save"),
    "quit": ((455, 16), (580, 56), "Quit"),
    "pick_bg": ((20, 66), (290, 106), "Pick BG"),
    "clear_bg": ((310, 66), (580, 106), "Clear BG"),
}

BG_MODE_LABELS = {
    0: "Solid",
    1: "Vertical Gradient",
    2: "Horizontal Gradient",
    3: "Radial Gradient",
    4: "Custom Image",
}

PAGE_LABELS = {
    0: "Page 1: Body and Hands",
    1: "Page 2: Side Head",
    2: "Page 3: Eyes",
    3: "Page 4: Face Features",
    4: "Page 5: Background",
    5: "Page 6: Mask",
}


class LayoutControls:
    def __init__(self):
        self.uses_opencv_window = True
        self.current_page = 0
        self._action_events = {key: False for key in ACTION_BUTTONS}
        self._trackbar_names: dict[str, str] = {}

    @staticmethod
    def load_settings(path: Path) -> dict[str, LayoutValue]:
        defaults: dict[str, LayoutValue] = {key: default for key, (default, _low, _high) in TRACKBARS.items()}
        defaults.update(UI_STATE_DEFAULTS)
        if not path.exists():
            return defaults

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults

        if not isinstance(data, dict):
            return defaults

        result = defaults.copy()
        bg_image_path = data.get("bg_image_path")
        if isinstance(bg_image_path, str) and bg_image_path:
            result["bg_image_path"] = bg_image_path

        for key, (_default, low, high) in TRACKBARS.items():
            value = data.get(key, result[key])
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = result[key]
            result[key] = max(low, min(high, value))
        for key, default in UI_STATE_DEFAULTS.items():
            value = data.get(key, default)
            try:
                result[key] = 1 if int(value) > 0 else 0
            except (TypeError, ValueError):
                result[key] = default
        return result

    @staticmethod
    def save_settings(path: Path, values: dict[str, LayoutValue]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable: dict[str, LayoutValue] = {key: int(values[key]) for key in TRACKBARS if key in values}
        for key in UI_STATE_DEFAULTS:
            if key in values:
                serializable[key] = 1 if int(values[key]) > 0 else 0
        bg_image_path = values.get("bg_image_path")
        if isinstance(bg_image_path, str) and bg_image_path:
            serializable["bg_image_path"] = bg_image_path
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    @staticmethod
    def choose_background_file() -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            print("Tkinter is not available; cannot open file picker.")
            return None

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose background image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")],
        )
        root.destroy()
        return path or None

    @staticmethod
    def _noop(_value: int) -> None:
        pass

    def _on_controls_click(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for key, ((x0, y0), (x1, y1), _label) in ACTION_BUTTONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._action_events[key] = True

    def create_window(self, initial_values: dict[str, LayoutValue]) -> dict[str, tuple[int, int]]:
        try:
            cv2.destroyWindow(CONTROL_WINDOW)
        except cv2.error:
            pass
        cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROL_WINDOW, 760, 740)
        cv2.setMouseCallback(CONTROL_WINDOW, self._on_controls_click)

        limits: dict[str, tuple[int, int]] = {}
        self._trackbar_names = {}
        for key in TRACKBAR_PAGES[self.current_page]:
            default, low, high = TRACKBARS[key]
            initial = initial_values.get(key, default)
            initial_int = int(initial)
            label = TRACKBAR_LABELS.get(key, key.replace("_", " "))
            self._trackbar_names[key] = label
            cv2.createTrackbar(label, CONTROL_WINDOW, initial_int - low, high - low, self._noop)
            limits[key] = (low, high)
        return limits

    def read_values(self, limits: dict[str, tuple[int, int]]) -> dict[str, int]:
        values: dict[str, int] = {}
        for key, (low, _high) in limits.items():
            label = self._trackbar_names.get(key, TRACKBAR_LABELS.get(key, key.replace("_", " ")))
            values[key] = cv2.getTrackbarPos(label, CONTROL_WINDOW) + low
        return values

    def draw_overlay(self, current_layout: dict[str, LayoutValue]) -> np.ndarray:
        image = np.full((136, 620, 3), 238, dtype=np.uint8)
        mode = int(current_layout.get("bg_mode", 0))
        mode_label = BG_MODE_LABELS.get(mode, f"Mode {mode}")
        cv2.putText(
            image,
            f"Background: {mode_label}",
            (12, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        page_name = PAGE_LABELS.get(self.current_page, f"Page {self.current_page + 1}")
        cv2.putText(
            image,
            page_name,
            (200, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        for key, ((x0, y0), (x1, y1), label) in ACTION_BUTTONS.items():
            if key == "quit":
                fill = (85, 90, 220)
            elif key in ("page_prev", "page_next"):
                fill = (90, 140, 210)
            elif key == "clear_bg":
                fill = (170, 130, 90)
            elif key == "pick_bg":
                fill = (85, 155, 95)
            else:
                fill = (70, 160, 90)
            cv2.rectangle(image, (x0, y0), (x1, y1), fill, thickness=-1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (40, 40, 40), thickness=1)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            text_x = x0 + max(0, ((x1 - x0) - text_w) // 2)
            text_y = y0 + max(text_h, ((y1 - y0) + text_h) // 2) - max(0, baseline // 2)
            cv2.putText(
                image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return image

    def pop_action(self, name: str) -> bool:
        value = self._action_events.get(name, False)
        self._action_events[name] = False
        return value

    def goto_prev_page(self) -> None:
        self.current_page = (self.current_page - 1) % len(TRACKBAR_PAGES)

    def goto_next_page(self) -> None:
        self.current_page = (self.current_page + 1) % len(TRACKBAR_PAGES)

    def process_events(self) -> None:
        return

    def close(self) -> None:
        try:
            cv2.destroyWindow(CONTROL_WINDOW)
        except cv2.error:
            pass

    def update_previews(self, _points_frame, _character_frame) -> None:
        return

    def is_tracking_enabled(self) -> bool:
        return True

    def is_hands_enabled(self) -> bool:
        return True

    def is_mic_enabled(self) -> bool:
        return True

    def is_points_preview_enabled(self) -> bool:
        return True
