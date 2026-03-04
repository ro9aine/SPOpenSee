import enum
from dataclasses import dataclass


class FState(enum.Enum):
    # Face turn state
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
        if self == FState.UP and other == FState.LEFT:
            return FState.UP_LEFT
        elif self == FState.UP and other == FState.RIGHT:
            return FState.UP_RIGHT
        elif self == FState.DOWN and other == FState.LEFT:
            return FState.DOWN_LEFT
        elif self == FState.DOWN and other == FState.RIGHT:
            return FState.DOWN_RIGHT
        elif self == FState.LEFT and other == FState.UP:
            return FState.UP_LEFT
        elif self == FState.LEFT and other == FState.DOWN:
            return FState.DOWN_LEFT
        elif self == FState.RIGHT and other == FState.UP:
            return FState.UP_RIGHT
        elif self == FState.RIGHT and other == FState.DOWN:
            return FState.DOWN_RIGHT
        else:
            raise ValueError(f"Invalid state: {self} + {other}")


class EState(enum.Enum):
    # Eye state
    OPEN = 0
    CLOSED = 1
    HALF = 2


class MState(enum.Enum):
    # Mouth state
    OPEN = 0
    CLOSED = 1
    WIDE_OPEN = 2


class EmotionState(enum.Enum):
    # Emotion state
    HAPPY = 0
    SAD = 1
    NEUTRAL = 2


class BState(enum.Enum):
    # Brows state
    MIDDLE = 0
    UP = 1
    DOWN = 2


class RState(enum.Enum):
    # Rotation state
    NORMAL = 0
    SLIGHTLY_LEFT = 1
    LEFT = 2
    SLIGHTLY_RIGHT = 3
    RIGHT = 4


@dataclass
class FaceState:
    turn: FState = FState.CENTER
    # Eyes state
    left_eye: EState = EState.OPEN
    right_eye: EState = EState.OPEN

    # Mouth state
    mouth: MState = MState.OPEN

    # Emotion state
    emotion: EmotionState = EmotionState.NEUTRAL

    # Brows
    left_brow: BState = BState.MIDDLE
    right_brow: BState = BState.MIDDLE
