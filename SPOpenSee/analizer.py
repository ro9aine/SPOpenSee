import math
import enum
import numpy as np
from typing import NewType
from opensee.tracker import FaceInfo
from .state import FaceState, FState, EState, BState, MState, EmotionState


LeftEyeState = NewType('LeftEyeState', EState)
RightEyeState = NewType('RightEyeState', EState)
LeftBrowState = NewType('LeftBrowState', BState)
RightBrowState = NewType('RightBrowState', BState)


class NoseState(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    CENTER = 4
    UP_LEFT = 5
    UP_RIGHT = 6
    DOWN_LEFT = 7
    DOWN_RIGHT = 8

    def __add__(self, other):
        if self == NoseState.UP and other == NoseState.LEFT:
            return NoseState.UP_LEFT
        elif self == NoseState.UP and other == NoseState.RIGHT:
            return NoseState.UP_RIGHT
        elif self == NoseState.DOWN and other == NoseState.LEFT:
            return NoseState.DOWN_LEFT
        elif self == NoseState.DOWN and other == NoseState.RIGHT:
            return NoseState.DOWN_RIGHT
        elif self == NoseState.LEFT and other == NoseState.UP:
            return NoseState.UP_LEFT
        elif self == NoseState.LEFT and other == NoseState.DOWN:
            return NoseState.DOWN_LEFT
        elif self == NoseState.RIGHT and other == NoseState.UP:
            return NoseState.UP_RIGHT
        elif self == NoseState.RIGHT and other == NoseState.DOWN:
            return NoseState.DOWN_RIGHT
        else:
            return self


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_aspect_ratio(eye):
    """
    eye = list of 6 (x,y) points
    """
    A = _dist(eye[1], eye[5])
    B = _dist(eye[2], eye[4])
    C = _dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)


class Analizer:
    def __init__(self):
        self._nose = NoseState.CENTER

    def analyze(self, face: FaceInfo):
        self._nose = self._get_nose_state(face)

    def find_all(self, face: FaceInfo) -> FaceState:
        state = FaceState()
        state.turn = self.find_turn_state(face)
        left, right = self.find_eyes_state(face)
        state.left_eye = left
        state.right_eye = right
        state.mouth = self.find_mouth_state(face)
        # state.emotion = self.find_emotion_state(face)
        left, right = self.find_brows_state(face)
        state.left_brow = left
        state.right_brow = right
        return state

    def find_turn_state(self, face: FaceInfo) -> FState:
        lm = face.lms

        # compute face bounding box
        xs = [p[1] for p in lm]
        ys = [p[0] for p in lm]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        face_center_x = (min_x + max_x) / 2
        face_center_y = (min_y + max_y) / 2

        nose = lm[30]  # nose tip

        dx = nose[1] - face_center_x
        dy = nose[0] - face_center_y

        # IMPORTANT: make threshold relative to face size
        face_width = max_x - min_x
        face_height = max_y - min_y

        horizontal_thresh = face_width * 0.10
        vertical_thresh = face_height * 0.12

        h_state = None
        v_state = None

        if dx > horizontal_thresh:
            h_state = FState.RIGHT
        elif dx < -horizontal_thresh:
            h_state = FState.LEFT

        if dy > vertical_thresh:
            v_state = FState.DOWN
        elif dy < -vertical_thresh:
            v_state = FState.UP

        if h_state and v_state:
            return h_state + v_state
        elif h_state:
            return h_state
        elif v_state:
            return v_state
        else:
            return FState.CENTER

    def find_eyes_state(self, face: FaceInfo) -> tuple[LeftEyeState, RightEyeState]:
        lm = face.lms

        # remember: x = p[1], y = p[0]
        left_eye_pts = [lm[i] for i in range(36, 42)]
        right_eye_pts = [lm[i] for i in range(42, 48)]

        def ear(eye):
            def dist(p1, p2):
                return ((p1[1] - p2[1])**2 + (p1[0] - p2[0])**2) ** 0.5

            A = dist(eye[1], eye[5])
            B = dist(eye[2], eye[4])
            C = dist(eye[0], eye[3])

            return (A + B) / (2.0 * C)

        left_ear = ear(left_eye_pts)
        right_ear = ear(right_eye_pts)

        THRESH = 0.12   # tune this

        left_state = EState.CLOSED if left_ear < THRESH else EState.OPEN
        right_state = EState.CLOSED if right_ear < THRESH else EState.OPEN

        return left_state, right_state

    def find_mouth_state(self, face: FaceInfo) -> MState:
        lm = face.lms

        def dist(p1, p2):
            return ((p1[1] - p2[1])**2 + (p1[0] - p2[0])**2) ** 0.5

        # Inner mouth landmarks
        top_inner = lm[60]      # bottom of top lip
        bottom_inner = lm[64]   # top of bottom lip
        left_inner = lm[58]     # left inner corner
        right_inner = lm[62]    # right inner corner

        vertical = dist(top_inner, bottom_inner)
        horizontal = dist(left_inner, right_inner)

        if horizontal < 5:   # safety check
            return MState.CLOSED

        ratio = vertical / horizontal

        # --- thresholds ---
        OPEN_THRESH = 0.17
        WIDE_THRESH = 0.52

        if ratio > WIDE_THRESH:
            return MState.WIDE_OPEN
        elif ratio > OPEN_THRESH:
            return MState.OPEN
        else:
            return MState.CLOSED

    def find_emotion_state(self, face: FaceInfo) -> EmotionState:
        ...

    def find_brows_state(self, face: FaceInfo) -> tuple[LeftBrowState, RightBrowState]:
        lm = face.lms

        def avg_y(indices):
            return sum(lm[i][0] for i in indices) / len(indices)

        # Correct indices
        left_brow_indices = range(17, 22)      # 17-21
        right_brow_indices = range(22, 27)     # 22-26

        left_eye_top_indices = [37, 38]
        right_eye_top_indices = [43, 44]

        # Compute average Y positions
        left_brow_y = avg_y(left_brow_indices)
        right_brow_y = avg_y(right_brow_indices)

        left_eye_y = avg_y(left_eye_top_indices)
        right_eye_y = avg_y(right_eye_top_indices)

        # Distance between eye and brow
        left_distance = left_eye_y - left_brow_y
        right_distance = right_eye_y - right_brow_y

        # Normalize by face height (important!)
        ys = [p[0] for p in lm]
        face_height = max(ys) - min(ys)

        left_ratio = left_distance / face_height
        right_ratio = right_distance / face_height

        # Tune these based on your webcam
        UP_THRESH = 0.155
        DOWN_THRESH = 0.105

        def classify(ratio):
            if ratio > UP_THRESH:
                return BState.UP
            elif ratio < DOWN_THRESH:
                return BState.DOWN
            else:
                return BState.MIDDLE

        return classify(left_ratio), classify(right_ratio)

    def _get_nose_state(self, face: FaceInfo):
        nose_x = face.lms[30][1]
        left_x = face.lms[2][1]
        right_x = face.lms[14][1]

        nose_y = face.lms[30][0]
        left_y = face.lms[2][0]
        right_y = face.lms[14][0]
        face_len = right_x - left_x

        # Distance from nose to each side
        dist_left = nose_x - left_x
        dist_right = right_x - nose_x

        h_state = NoseState.CENTER
        v_state = NoseState.CENTER

        if abs(dist_left - dist_right) < face_len / 2:
            if max(left_y, right_y) - nose_y > face_len / 4:
                v_state = NoseState.UP
            elif max(left_y, right_y) - nose_y < -(face_len / 4):
                v_state = NoseState.DOWN
            else:
                v_state = NoseState.CENTER

        if dist_left > face_len / 1.5 or dist_right > face_len / 1.5:
            if dist_left > dist_right:
                h_state = NoseState.RIGHT
            elif dist_left < dist_right:
                h_state = NoseState.LEFT
            else:
                h_state = NoseState.CENTER

        if h_state == v_state:
            return NoseState.CENTER
        elif h_state == NoseState.CENTER:
            return v_state
        elif v_state == NoseState.CENTER:
            return h_state
        else:
            return h_state + v_state

    NoseState = NoseState
