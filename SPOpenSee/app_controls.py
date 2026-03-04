from pathlib import Path
import json

import cv2
import numpy as np


CONTROL_WINDOW = "Layout Controls"

TRACKBARS = {
    "body_scale": (90, 20, 160),
    "body_x": (0, -200, 200),
    "body_y": (0, -200, 200),
    "head_scale": (62, 20, 120),
    "head_x": (0, -200, 200),
    "head_y": (0, -200, 200),
    "hair_scale": (100, 20, 200),
    "hair_x": (0, -200, 200),
    "hair_y": (0, -200, 200),
    "eyes_scale": (55, 20, 200),
    "eyes_x": (0, -150, 150),
    "eyes_y": (0, -150, 150),
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
    "mouth_scale": (100, 20, 250),
    "mouth_x": (0, -120, 120),
    "mouth_y": (0, -120, 120),
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

TRACKBAR_PAGES = [
    [
        "body_scale",
        "body_x",
        "body_y",
        "head_scale",
        "head_x",
        "head_y",
        "hair_scale",
        "hair_x",
        "hair_y",
        "eyes_scale",
        "eyes_x",
        "eyes_y",
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
        "mouth_scale",
        "mouth_x",
        "mouth_y",
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
    "page_prev": ((20, 20), (100, 70), "Prev"),
    "page_next": ((110, 20), (190, 70), "Next"),
    "save": ((210, 20), (290, 70), "Save"),
    "quit": ((300, 20), (380, 70), "Quit"),
    "pick_bg": ((20, 75), (200, 120), "Pick BG"),
    "clear_bg": ((210, 75), (380, 120), "Clear BG"),
}

BG_MODE_LABELS = {
    0: "Solid",
    1: "Vertical Gradient",
    2: "Horizontal Gradient",
    3: "Radial Gradient",
    4: "Custom Image",
}

PAGE_LABELS = {
    0: "Page 1: Character",
    1: "Page 2: Face Features",
    2: "Page 3: Background",
    3: "Page 4: Mask",
}


class LayoutControls:
    def __init__(self):
        self.current_page = 0
        self._action_events = {key: False for key in ACTION_BUTTONS}

    @staticmethod
    def load_settings(path: Path) -> dict[str, int]:
        defaults = {key: default for key, (default, _low, _high) in TRACKBARS.items()}
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
        return result

    @staticmethod
    def save_settings(path: Path, values: dict[str, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: int(values[key]) for key in TRACKBARS if key in values}
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

    def create_window(self, initial_values: dict[str, int]) -> dict[str, tuple[int, int]]:
        try:
            cv2.destroyWindow(CONTROL_WINDOW)
        except cv2.error:
            pass
        cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROL_WINDOW, 560, 690)
        cv2.setMouseCallback(CONTROL_WINDOW, self._on_controls_click)

        limits: dict[str, tuple[int, int]] = {}
        for key in TRACKBAR_PAGES[self.current_page]:
            default, low, high = TRACKBARS[key]
            initial = initial_values.get(key, default)
            cv2.createTrackbar(key, CONTROL_WINDOW, initial - low, high - low, self._noop)
            limits[key] = (low, high)
        return limits

    @staticmethod
    def read_values(limits: dict[str, tuple[int, int]]) -> dict[str, int]:
        values: dict[str, int] = {}
        for key, (low, _high) in limits.items():
            values[key] = cv2.getTrackbarPos(key, CONTROL_WINDOW) + low
        return values

    def draw_overlay(self, current_layout: dict[str, int]) -> np.ndarray:
        image = np.full((145, 400, 3), 238, dtype=np.uint8)
        mode = int(current_layout.get("bg_mode", 0))
        mode_label = BG_MODE_LABELS.get(mode, f"Mode {mode}")
        cv2.putText(
            image,
            f"Background: {mode_label}",
            (12, 137),
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
            (210, 137),
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
            cv2.putText(
                image,
                label,
                (x0 + 14, y0 + 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
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
