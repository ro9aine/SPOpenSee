from collections import Counter, deque
from pathlib import Path
import json

import cv2
import numpy as np

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None
from opensee.tracker import Tracker
from spopensee.analyzers import FaceAnalyzer
from spopensee.generators import CharacterGenerator
from spopensee.state import FaceState


SMOOTHING_WINDOW = 4

CONTROL_WINDOW = "Layout Controls"
WEBCAM_WINDOW = "Webcam"
CHARACTER_WINDOW = "Character"
CHARACTER_WIDTH = 1280
CHARACTER_HEIGHT = 720

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
    "closed_eyes_scale": (100, 20, 250),
    "closed_eyes_x": (0, -120, 120),
    "closed_eyes_y": (0, -120, 120),
    "brow_scale": (100, 20, 250),
    "brow_x": (0, -120, 120),
    "brow_y": (0, -120, 120),
    "mouth_scale": (100, 20, 250),
    "mouth_x": (0, -120, 120),
    "mouth_y": (0, -120, 120),
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
    [
        "mask_left",
        "mask_right",
        "mask_top",
        "mask_bottom",
    ],
]

ACTION_BUTTONS = {
    "page_prev": ((20, 20), (100, 70), "Prev"),
    "page_next": ((110, 20), (190, 70), "Next"),
    "save": ((210, 20), (290, 70), "Save"),
    "quit": ((300, 20), (380, 70), "Quit"),
    "pick_bg": ((20, 75), (200, 120), "Pick BG"),
    "clear_bg": ((210, 75), (380, 120), "Clear BG"),
}
ACTION_EVENTS = {key: False for key in ACTION_BUTTONS}
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


def load_layout_settings(path: Path) -> dict[str, int]:
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


def save_layout_settings(path: Path, values: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: int(values[key]) for key in TRACKBARS if key in values}
    bg_image_path = values.get("bg_image_path")
    if isinstance(bg_image_path, str) and bg_image_path:
        serializable["bg_image_path"] = bg_image_path
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _noop(_value: int) -> None:
    pass


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


def _on_controls_click(event: int, x: int, y: int, _flags: int, _param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for key, ((x0, y0), (x1, y1), _label) in ACTION_BUTTONS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            ACTION_EVENTS[key] = True


def draw_controls_overlay(current_layout: dict[str, int]) -> np.ndarray:
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
    page_name = PAGE_LABELS.get(current_page, f"Page {current_page + 1}")
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


def pop_action_event(name: str) -> bool:
    value = ACTION_EVENTS.get(name, False)
    ACTION_EVENTS[name] = False
    return value


def create_layout_controls(initial_values: dict[str, int], page: int) -> dict[str, tuple[int, int]]:
    try:
        cv2.destroyWindow(CONTROL_WINDOW)
    except cv2.error:
        pass
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 560, 690)
    cv2.setMouseCallback(CONTROL_WINDOW, _on_controls_click)

    limits: dict[str, tuple[int, int]] = {}
    for key in TRACKBAR_PAGES[page]:
        default, low, high = TRACKBARS[key]
        initial = initial_values.get(key, default)
        cv2.createTrackbar(key, CONTROL_WINDOW, initial - low, high - low, _noop)
        limits[key] = (low, high)
    return limits


def read_layout_controls(limits: dict[str, tuple[int, int]]) -> dict[str, int]:
    values: dict[str, int] = {}
    for key, (low, _high) in limits.items():
        values[key] = cv2.getTrackbarPos(key, CONTROL_WINDOW) + low
    return values


def open_virtual_camera(width: int, height: int, fps: int = 30):
    if pyvirtualcam is None:
        print("pyvirtualcam is not installed. Virtual webcam is disabled.")
        return None

    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
        print(f"Virtual camera started: {cam.device}")
        return cam
    except Exception as exc:
        print(f"Cannot start virtual camera: {exc}")
        return None


def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def open_camera():
    # Prefer DirectShow on Windows to avoid common MSMF read failures.
    attempts = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
    ]

    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def majority_with_recent_tiebreak(history: deque[FaceState], attr: str):
    values = [getattr(state, attr) for state in history]
    counts = Counter(values)
    max_count = max(counts.values())
    leaders = {value for value, count in counts.items() if count == max_count}

    # If tie, prefer the latest seen value for responsiveness.
    for state in reversed(history):
        value = getattr(state, attr)
        if value in leaders:
            return value
    return values[-1]


def smooth_face_state(history: deque[FaceState]) -> FaceState:
    smoothed = FaceState()
    smoothed.turn = majority_with_recent_tiebreak(history, "turn")
    smoothed.left_eye = majority_with_recent_tiebreak(history, "left_eye")
    smoothed.right_eye = majority_with_recent_tiebreak(history, "right_eye")
    smoothed.mouth = majority_with_recent_tiebreak(history, "mouth")
    smoothed.emotion = majority_with_recent_tiebreak(history, "emotion")
    smoothed.left_brow = majority_with_recent_tiebreak(history, "left_brow")
    smoothed.right_brow = majority_with_recent_tiebreak(history, "right_brow")
    smoothed.rotation = majority_with_recent_tiebreak(history, "rotation")
    return smoothed


cap = open_camera()
if cap is None:
    print("Cannot open camera with available backends/devices")
    raise SystemExit(1)

tracker = Tracker(480, 640, silent=True)
anl = FaceAnalyzer()
char_id = "default"
layout_path = Path("configs") / "characters" / f"{char_id}_layout.json"
char = CharacterGenerator(char_id, width=CHARACTER_WIDTH, height=CHARACTER_HEIGHT)
initial_layout = load_layout_settings(layout_path)
current_layout = initial_layout.copy()
current_page = 0
control_limits = create_layout_controls(current_layout, current_page)
char.set_layout(initial_layout)
char.set_background_image(initial_layout.get("bg_image_path") if isinstance(initial_layout.get("bg_image_path"), str) else None)
state_history: deque[FaceState] = deque(maxlen=SMOOTHING_WINDOW)
virtual_cam = open_virtual_camera(char.width, char.height, fps=30)
last_state = FaceState()
last_character_frame, _last_mask_frame = char.generate_with_mask(last_state)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Can't receive frame from camera")
        continue

    faces = tracker.predict(frame)
    for face in faces:
        for pt_num, (x, y, c) in enumerate(face.lms):
            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0))

    current_layout.update(read_layout_controls(control_limits))
    char.set_layout(current_layout)

    if len(faces) == 1:
        current_state = anl.find_all(faces[0])
        state_history.append(current_state)
        last_state = smooth_face_state(state_history)

    last_character_frame, _last_mask_frame = char.generate_with_mask(last_state)

    cv2.imshow(WEBCAM_WINDOW, frame)
    cv2.imshow(CHARACTER_WINDOW, last_character_frame)
    cv2.imshow(CONTROL_WINDOW, draw_controls_overlay(current_layout))

    if virtual_cam is not None:
        virtual_cam.send(to_rgb(last_character_frame))
        virtual_cam.sleep_until_next_frame()

    key = cv2.waitKey(1) & 0xFF
    if pop_action_event("save"):
        save_layout_settings(layout_path, current_layout)
        print(f"Layout saved to {layout_path}")
    if pop_action_event("page_prev"):
        current_page = (current_page - 1) % len(TRACKBAR_PAGES)
        control_limits = create_layout_controls(current_layout, current_page)
    if pop_action_event("page_next"):
        current_page = (current_page + 1) % len(TRACKBAR_PAGES)
        control_limits = create_layout_controls(current_layout, current_page)
    if pop_action_event("pick_bg"):
        bg_path = choose_background_file()
        if bg_path:
            current_layout["bg_image_path"] = bg_path
            current_layout["bg_mode"] = 4
            char.set_background_image(bg_path)
            control_limits = create_layout_controls(current_layout, current_page)
    if pop_action_event("clear_bg"):
        current_layout.pop("bg_image_path", None)
        if int(current_layout.get("bg_mode", 0)) == 4:
            current_layout["bg_mode"] = 0
        char.set_background_image(None)
        control_limits = create_layout_controls(current_layout, current_page)
    if pop_action_event("quit"):
        save_layout_settings(layout_path, current_layout)
        break
    if key == 27:
        save_layout_settings(layout_path, current_layout)
        break

cap.release()
if virtual_cam is not None:
    virtual_cam.close()
cv2.destroyAllWindows()
