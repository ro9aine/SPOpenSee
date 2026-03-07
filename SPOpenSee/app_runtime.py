from collections import Counter, deque

import cv2
import numpy as np

from .state import FaceState

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


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


def average_attr(history: deque[FaceState], attr: str) -> float:
    values = [float(getattr(state, attr)) for state in history]
    return float(sum(values) / max(1, len(values)))


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
    smoothed.left_arm_visible = majority_with_recent_tiebreak(history, "left_arm_visible")
    smoothed.right_arm_visible = majority_with_recent_tiebreak(history, "right_arm_visible")
    smoothed.left_shoulder_x = average_attr(history, "left_shoulder_x")
    smoothed.left_shoulder_y = average_attr(history, "left_shoulder_y")
    smoothed.right_shoulder_x = average_attr(history, "right_shoulder_x")
    smoothed.right_shoulder_y = average_attr(history, "right_shoulder_y")
    smoothed.left_elbow_x = average_attr(history, "left_elbow_x")
    smoothed.left_elbow_y = average_attr(history, "left_elbow_y")
    smoothed.right_elbow_x = average_attr(history, "right_elbow_x")
    smoothed.right_elbow_y = average_attr(history, "right_elbow_y")
    smoothed.left_hand_x = average_attr(history, "left_hand_x")
    smoothed.left_hand_y = average_attr(history, "left_hand_y")
    smoothed.right_hand_x = average_attr(history, "right_hand_x")
    smoothed.right_hand_y = average_attr(history, "right_hand_y")
    return smoothed
