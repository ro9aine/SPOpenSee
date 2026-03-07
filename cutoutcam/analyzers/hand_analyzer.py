from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..state import FaceState

cv2: Any = None
mp: Any = None
mp_hands_module: Any = None

try:
    import cv2 as _cv2
    import mediapipe as _mp
except ImportError:
    pass
else:
    cv2 = _cv2
    mp = _mp

try:
    from mediapipe.python.solutions import hands as _mp_hands_module
except ImportError:
    pass
else:
    mp_hands_module = _mp_hands_module


class HandAnalyzer:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.available = mp is not None and cv2 is not None
        self._hands: Any | None = None
        self._mp_hands: Any | None = None

        if not self.available:
            return

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._mp_hands = mp.solutions.hands
        elif mp_hands_module is not None:
            self._mp_hands = mp_hands_module
        else:
            self.available = False
            return

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def enrich_state(self, frame_bgr: Any, state: FaceState) -> FaceState:
        if not self.available or self._hands is None:
            return state

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        hands = results.multi_hand_landmarks or []
        handedness = results.multi_handedness or []

        left_x = state.left_hand_x
        left_y = state.left_hand_y
        right_x = state.right_hand_x
        right_y = state.right_hand_y
        left_visible = False
        right_visible = False

        unlabeled_points: list[tuple[float, float]] = []
        for idx, hand_landmarks in enumerate(hands):
            if self._mp_hands is None:
                continue
            wrist = hand_landmarks.landmark[self._mp_hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[self._mp_hands.HandLandmark.PINKY_MCP]
            palm_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
            palm_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
            point = (float(min(1.0, max(0.0, palm_x))), float(min(1.0, max(0.0, palm_y))))

            label = None
            if idx < len(handedness) and handedness[idx].classification:
                label = handedness[idx].classification[0].label.lower()

            if label == "left":
                left_x, left_y = point
                left_visible = True
            elif label == "right":
                right_x, right_y = point
                right_visible = True
            else:
                unlabeled_points.append(point)

        if unlabeled_points:
            unlabeled_points.sort(key=lambda pt: pt[0])
            if not left_visible:
                left_x, left_y = unlabeled_points[0]
                left_visible = True
                unlabeled_points = unlabeled_points[1:]
            if unlabeled_points and not right_visible:
                right_x, right_y = unlabeled_points[-1]
                right_visible = True

        return replace(
            state,
            left_arm_visible=left_visible,
            right_arm_visible=right_visible,
            left_hand_x=left_x,
            left_hand_y=left_y,
            right_hand_x=right_x,
            right_hand_y=right_y,
        )
