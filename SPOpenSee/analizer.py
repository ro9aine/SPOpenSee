import enum
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
