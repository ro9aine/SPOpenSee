import math
from typing import NewType
from opensee.tracker import FaceInfo
from .state import FaceState, FState, EState, BState, MState, EmotionState


LeftEyeState = NewType('LeftEyeState', EState)
RightEyeState = NewType('RightEyeState', EState)
LeftBrowState = NewType('LeftBrowState', BState)
RightBrowState = NewType('RightBrowState', BState)


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _eye_aspect_ratio(eye):
    """
    eye = list of 6 (x,y) points
    """
    A = _dist(eye[1], eye[5])
    B = _dist(eye[2], eye[4])
    C = _dist(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


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

        left_ear = _eye_aspect_ratio(left_eye_pts)
        right_ear = _eye_aspect_ratio(right_eye_pts)

        THRESH = 0.12   # tune this

        left_state = EState.CLOSED if left_ear < THRESH else EState.OPEN
        right_state = EState.CLOSED if right_ear < THRESH else EState.OPEN

        return left_state, right_state

    def find_mouth_state(self, face: FaceInfo) -> MState:
        lm = face.lms

        def dist(p1, p2):
            return ((p1[1] - p2[1])**2 + (p1[0] - p2[0])**2) ** 0.5

        # Outer mouth points
        top_outer = lm[51]
        bottom_outer = lm[55]
        left_corner = lm[58]
        right_corner = lm[62]

        # Inner mouth points (68-point model). Fallback to outer points if unavailable.
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

        # OPEN and WIDE_OPEN are intentionally separated to avoid soft-open false positives.
        OPEN_OUTER_THRESH = 0.22
        OPEN_INNER_THRESH = 0.075
        WIDE_OUTER_THRESH = 0.40
        WIDE_INNER_THRESH = 0.20
        WIDE_FACE_THRESH = 0.10

        if (
            outer_ratio > WIDE_OUTER_THRESH
            and inner_ratio > WIDE_INNER_THRESH
            and face_open_ratio > WIDE_FACE_THRESH
        ):
            return MState.WIDE_OPEN

        if outer_ratio > OPEN_OUTER_THRESH and inner_ratio > OPEN_INNER_THRESH:
            return MState.OPEN

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
        if face_height == 0:
            return BState.MIDDLE, BState.MIDDLE

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

