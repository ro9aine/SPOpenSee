import math
from typing import NewType

from opensee.tracker import FaceInfo

from ..state import BState, EState, EmotionState, FState, FaceState, MState, RState


LeftEyeState = NewType("LeftEyeState", EState)
RightEyeState = NewType("RightEyeState", EState)
LeftBrowState = NewType("LeftBrowState", BState)
RightBrowState = NewType("RightBrowState", BState)


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_aspect_ratio(eye):
    """
    eye = list of 6 (x,y) points
    """
    a = _dist(eye[1], eye[5])
    b = _dist(eye[2], eye[4])
    c = _dist(eye[0], eye[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


class Analizer:
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
        state.rotation = self.find_rotation_state(face)
        return state

    def find_rotation_state(self, face: FaceInfo) -> RState:
        lm = face.lms

        if len(lm) < 48:
            return RState.NORMAL

        left_eye_pts = [lm[i] for i in range(36, 42)]
        right_eye_pts = [lm[i] for i in range(42, 48)]

        left_eye_y = sum(p[0] for p in left_eye_pts) / len(left_eye_pts)
        left_eye_x = sum(p[1] for p in left_eye_pts) / len(left_eye_pts)
        right_eye_y = sum(p[0] for p in right_eye_pts) / len(right_eye_pts)
        right_eye_x = sum(p[1] for p in right_eye_pts) / len(right_eye_pts)

        dx = right_eye_x - left_eye_x
        if abs(dx) < 1e-6:
            return RState.NORMAL

        dy = right_eye_y - left_eye_y
        roll_deg = math.degrees(math.atan2(dy, dx))

        slight_thresh = 4.0
        strong_thresh = 10.0

        if roll_deg >= strong_thresh:
            return RState.RIGHT
        if roll_deg >= slight_thresh:
            return RState.SLIGHTLY_RIGHT
        if roll_deg <= -strong_thresh:
            return RState.LEFT
        if roll_deg <= -slight_thresh:
            return RState.SLIGHTLY_LEFT

        return RState.NORMAL

    def find_turn_state(self, face: FaceInfo) -> FState:
        lm = face.lms

        xs = [p[1] for p in lm]
        ys = [p[0] for p in lm]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        face_center_x = (min_x + max_x) / 2
        face_center_y = (min_y + max_y) / 2

        nose = lm[30]

        dx = nose[1] - face_center_x
        dy = nose[0] - face_center_y

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
        if h_state:
            return h_state
        if v_state:
            return v_state
        return FState.CENTER

    def find_eyes_state(self, face: FaceInfo) -> tuple[LeftEyeState, RightEyeState]:
        lm = face.lms

        left_eye_pts = [lm[i] for i in range(36, 42)]
        right_eye_pts = [lm[i] for i in range(42, 48)]

        left_ear = _eye_aspect_ratio(left_eye_pts)
        right_ear = _eye_aspect_ratio(right_eye_pts)

        thresh = 0.12

        left_state = EState.CLOSED if left_ear < thresh else EState.OPEN
        right_state = EState.CLOSED if right_ear < thresh else EState.OPEN

        return left_state, right_state

    def find_mouth_state(self, face: FaceInfo) -> MState:
        lm = face.lms

        def dist(p1, p2):
            return ((p1[1] - p2[1]) ** 2 + (p1[0] - p2[0]) ** 2) ** 0.5

        top_outer = lm[51]
        bottom_outer = lm[55]
        left_corner = lm[58]
        right_corner = lm[62]

        if len(lm) > 66:
            top_inner = lm[60]
            bottom_inner = lm[64]
        else:
            top_inner = top_outer
            bottom_inner = bottom_outer

        outer_vertical = dist(top_outer, bottom_outer)
        inner_vertical = dist(top_inner, bottom_inner)
        horizontal = dist(left_corner, right_corner)

        if horizontal < 5:
            return MState.CLOSED

        ys = [p[0] for p in lm]
        face_height = max(ys) - min(ys)
        if face_height <= 1:
            return MState.CLOSED

        outer_ratio = outer_vertical / horizontal
        inner_ratio = inner_vertical / horizontal
        face_open_ratio = outer_vertical / face_height

        open_outer_thresh = 0.22
        open_inner_thresh = 0.075
        wide_outer_thresh = 0.40
        wide_inner_thresh = 0.20
        wide_face_thresh = 0.10

        if (
            outer_ratio > wide_outer_thresh
            and inner_ratio > wide_inner_thresh
            and face_open_ratio > wide_face_thresh
        ):
            return MState.WIDE_OPEN

        if outer_ratio > open_outer_thresh and inner_ratio > open_inner_thresh:
            return MState.OPEN

        return MState.CLOSED

    def find_emotion_state(self, face: FaceInfo) -> EmotionState:
        ...

    def find_brows_state(self, face: FaceInfo) -> tuple[LeftBrowState, RightBrowState]:
        lm = face.lms

        def avg_y(indices):
            return sum(lm[i][0] for i in indices) / len(indices)

        left_brow_indices = range(17, 22)
        right_brow_indices = range(22, 27)

        left_eye_top_indices = [37, 38]
        right_eye_top_indices = [43, 44]

        left_brow_y = avg_y(left_brow_indices)
        right_brow_y = avg_y(right_brow_indices)

        left_eye_y = avg_y(left_eye_top_indices)
        right_eye_y = avg_y(right_eye_top_indices)

        left_distance = left_eye_y - left_brow_y
        right_distance = right_eye_y - right_brow_y

        ys = [p[0] for p in lm]
        face_height = max(ys) - min(ys)
        if face_height == 0:
            return BState.MIDDLE, BState.MIDDLE

        left_ratio = left_distance / face_height
        right_ratio = right_distance / face_height

        up_thresh = 0.155
        down_thresh = 0.105

        def classify(ratio):
            if ratio > up_thresh:
                return BState.UP
            if ratio < down_thresh:
                return BState.DOWN
            return BState.MIDDLE

        return classify(left_ratio), classify(right_ratio)


FaceAnalyzer = Analizer

__all__ = [
    "Analizer",
    "FaceAnalyzer",
    "LeftEyeState",
    "RightEyeState",
    "LeftBrowState",
    "RightBrowState",
]
