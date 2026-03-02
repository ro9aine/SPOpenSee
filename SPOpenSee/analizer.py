import enum
import numpy as np
from OpenSeeFace.tracker import FaceInfo


class NoseState(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    CENTER = 4


class Analizer:
    def __init__(self):
        self._nose = NoseState.CENTER

    def analyze(self, face: FaceInfo):
        self._nose = self._get_nose_state(face)

    def _get_state(self, face: FaceInfo):
        lms = face.lms

        # Convert to numpy array for easier math
        pts = np.array([[p[1], p[0]] for p in lms])
        # now pts[i] = [x, y]

        # ---- Key landmark indices (for 68-point model) ----
        NOSE_TIP = 30
        LEFT_FACE = 0
        RIGHT_FACE = 16
        LEFT_EYE = 36
        RIGHT_EYE = 45
        MOUTH_TOP = 62
        MOUTH_BOTTOM = 66

        # ---- Important points ----
        nose = pts[NOSE_TIP]
        left_face = pts[LEFT_FACE]
        right_face = pts[RIGHT_FACE]

        left_eye = pts[LEFT_EYE]
        right_eye = pts[RIGHT_EYE]

        mouth_top = pts[MOUTH_TOP]
        mouth_bottom = pts[MOUTH_BOTTOM]

        # ---- Face dimensions ----
        face_width = np.linalg.norm(right_face - left_face)
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (mouth_top + mouth_bottom) / 2
        vertical_center = (eye_center + mouth_center) / 2
        horizontal_center = (left_face + right_face) / 2

        # ---- Horizontal ratio ----
        horizontal_offset = (nose[0] - horizontal_center[0]) / face_width

        # ---- Vertical ratio ----
        face_height = np.linalg.norm(mouth_center - eye_center)
        vertical_offset = (nose[1] - vertical_center[1]) / face_height

        # ---- Thresholds ----
        H_THRESH = 0.08
        V_THRESH = 0.15

        # Determine horizontal state
        if horizontal_offset > H_THRESH:
            horizontal = "right"
        elif horizontal_offset < -H_THRESH:
            horizontal = "left"
        else:
            horizontal = "center"

        # Determine vertical state
        if vertical_offset > V_THRESH:
            vertical = "bottom"
        elif vertical_offset < -V_THRESH:
            vertical = "top"
        else:
            vertical = "center"

        # ---- Combine states ----
        if horizontal == "center" and vertical == "center":
            return "center"

        if vertical == "center":
            return horizontal

        if horizontal == "center":
            return vertical

        return f"{vertical}-{horizontal}"

    def _get_nose_state(self, face: FaceInfo):
        nose_x = face.lms[30][1]
        left_x = face.lms[2][1]
        right_x = face.lms[14][1]

        # Distance from nose to each side
        dist_left = nose_x - left_x
        dist_right = right_x - nose_x

        if abs(dist_left - dist_right) < 60:
            return NoseState.CENTER
        elif dist_left > dist_right:
            return NoseState.RIGHT
        else:
            return NoseState.LEFT

    NoseState = NoseState
