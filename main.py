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
CHARACTER_WIDTH = 800
CHARACTER_HEIGHT = 800

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
}

ACTION_BUTTONS = {
    "save": ((20, 20), (180, 70), "Save"),
    "quit": ((220, 20), (380, 70), "Quit"),
}
ACTION_EVENTS = {"save": False, "quit": False}


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
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _noop(_value: int) -> None:
    pass


def _on_controls_click(event: int, x: int, y: int, _flags: int, _param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for key, ((x0, y0), (x1, y1), _label) in ACTION_BUTTONS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            ACTION_EVENTS[key] = True


def draw_controls_overlay() -> np.ndarray:
    image = np.full((95, 400, 3), 238, dtype=np.uint8)
    for key, ((x0, y0), (x1, y1), label) in ACTION_BUTTONS.items():
        fill = (85, 90, 220) if key == "quit" else (70, 160, 90)
        cv2.rectangle(image, (x0, y0), (x1, y1), fill, thickness=-1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (40, 40, 40), thickness=1)
        cv2.putText(
            image,
            label,
            (x0 + 52, y0 + 33),
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


def create_layout_controls(initial_values: dict[str, int]) -> dict[str, tuple[int, int]]:
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 500, 700)
    cv2.setMouseCallback(CONTROL_WINDOW, _on_controls_click)

    limits: dict[str, tuple[int, int]] = {}
    for key, (default, low, high) in TRACKBARS.items():
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
control_limits = create_layout_controls(initial_layout)
char.set_layout(initial_layout)
state_history: deque[FaceState] = deque(maxlen=SMOOTHING_WINDOW)
virtual_cam = open_virtual_camera(char.width, char.height, fps=30)
last_character_frame = char.generate(FaceState())

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

    current_layout = read_layout_controls(control_limits)
    char.set_layout(current_layout)

    if len(faces) == 1:
        current_state = anl.find_all(faces[0])
        state_history.append(current_state)
        smoothed_state = smooth_face_state(state_history)
        last_character_frame = char.generate(smoothed_state)

    cv2.imshow(WEBCAM_WINDOW, frame)
    cv2.imshow(CHARACTER_WINDOW, last_character_frame)
    cv2.imshow(CONTROL_WINDOW, draw_controls_overlay())

    if virtual_cam is not None:
        virtual_cam.send(to_rgb(last_character_frame))
        virtual_cam.sleep_until_next_frame()

    key = cv2.waitKey(1) & 0xFF
    if pop_action_event("save"):
        save_layout_settings(layout_path, current_layout)
        print(f"Layout saved to {layout_path}")
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
