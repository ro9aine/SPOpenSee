import enum
import numpy as np
from openseeface.tracker import FaceInfo


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


class Analizer:
    def __init__(self):
        self._nose = NoseState.CENTER

    def analyze(self, face: FaceInfo):
        self._nose = self._get_nose_state(face)

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
