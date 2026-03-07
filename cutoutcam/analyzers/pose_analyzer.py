from __future__ import annotations

from dataclasses import replace

from ..state import FaceState

try:
    import cv2
    import mediapipe as mp
    _POSE_IMPORT_ERROR = None
except ImportError:
    cv2 = None
    mp = None
    _POSE_IMPORT_ERROR = "Failed to import cv2 or mediapipe."

try:
    from mediapipe.python.solutions import pose as mp_pose_module
except ImportError:
    mp_pose_module = None
    if _POSE_IMPORT_ERROR is None and mp is not None:
        _POSE_IMPORT_ERROR = "mediapipe is present, but mediapipe.python.solutions.pose is unavailable."


class PoseAnalyzer:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visibility_threshold: float = 0.45,
    ):
        self.available = mp is not None and cv2 is not None
        self._pose = None
        self._mp_pose = None
        self._visibility_threshold = visibility_threshold
        self._last_points: dict[str, tuple[float, float, float]] = {}
        self.unavailable_reason: str | None = _POSE_IMPORT_ERROR

        if not self.available:
            return

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._mp_pose = mp.solutions.pose
            self.unavailable_reason = None
        elif mp_pose_module is not None:
            self._mp_pose = mp_pose_module
            self.unavailable_reason = None
        else:
            self.available = False
            if self.unavailable_reason is None:
                self.unavailable_reason = "mediapipe was imported, but no supported pose module layout was found."
            return

        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def draw_debug(self, frame_bgr) -> None:
        if not self._last_points:
            return

        h, w = frame_bgr.shape[:2]
        connections = [
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
        ]
        for start_key, end_key in connections:
            if start_key not in self._last_points or end_key not in self._last_points:
                continue
            sx, sy, sv = self._last_points[start_key]
            ex, ey, ev = self._last_points[end_key]
            if min(sv, ev) < 0.15:
                continue
            cv2.line(
                frame_bgr,
                (int(sx * w), int(sy * h)),
                (int(ex * w), int(ey * h)),
                (80, 220, 80),
                2,
                cv2.LINE_AA,
            )

        for key, (x, y, visibility) in self._last_points.items():
            color = (0, 220, 0) if visibility >= self._visibility_threshold else (0, 140, 255)
            cv2.circle(frame_bgr, (int(x * w), int(y * h)), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                frame_bgr,
                key.split("_")[1][0].upper(),
                (int(x * w) + 6, int(y * h) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    def _landmark_xy(self, landmarks, landmark_id) -> tuple[float, float, float]:
        point = landmarks[landmark_id]
        return (
            float(min(1.0, max(0.0, point.x))),
            float(min(1.0, max(0.0, point.y))),
            float(getattr(point, "visibility", 1.0)),
        )

    @staticmethod
    def _blend_joint(
        shoulder: tuple[float, float, float],
        elbow: tuple[float, float, float],
        wrist: tuple[float, float, float],
    ) -> tuple[float, float]:
        shoulder_x, shoulder_y, _ = shoulder
        elbow_x, elbow_y, elbow_v = elbow
        wrist_x, wrist_y, wrist_v = wrist

        if elbow_v >= 0.25:
            return elbow_x, elbow_y

        mid_x = (shoulder_x + wrist_x) * 0.5
        mid_y = (shoulder_y + wrist_y) * 0.5 + 0.05
        if wrist_v >= 0.25:
            return mid_x, mid_y
        return elbow_x, elbow_y

    def enrich_state(self, frame_bgr, state: FaceState) -> FaceState:
        if not self.available or self._pose is None:
            return state

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)
        if results.pose_landmarks is None:
            self._last_points = {}
            return replace(state, left_arm_visible=False, right_arm_visible=False)

        lm = results.pose_landmarks.landmark
        pl = self._mp_pose.PoseLandmark

        left_shoulder_x, left_shoulder_y, left_shoulder_v = self._landmark_xy(lm, pl.LEFT_SHOULDER)
        right_shoulder_x, right_shoulder_y, right_shoulder_v = self._landmark_xy(lm, pl.RIGHT_SHOULDER)
        left_elbow_x, left_elbow_y, left_elbow_v = self._landmark_xy(lm, pl.LEFT_ELBOW)
        right_elbow_x, right_elbow_y, right_elbow_v = self._landmark_xy(lm, pl.RIGHT_ELBOW)
        left_wrist_x, left_wrist_y, left_wrist_v = self._landmark_xy(lm, pl.LEFT_WRIST)
        right_wrist_x, right_wrist_y, right_wrist_v = self._landmark_xy(lm, pl.RIGHT_WRIST)

        left_elbow_x, left_elbow_y = self._blend_joint(
            (left_shoulder_x, left_shoulder_y, left_shoulder_v),
            (left_elbow_x, left_elbow_y, left_elbow_v),
            (left_wrist_x, left_wrist_y, left_wrist_v),
        )
        right_elbow_x, right_elbow_y = self._blend_joint(
            (right_shoulder_x, right_shoulder_y, right_shoulder_v),
            (right_elbow_x, right_elbow_y, right_elbow_v),
            (right_wrist_x, right_wrist_y, right_wrist_v),
        )

        self._last_points = {
            "left_shoulder": (left_shoulder_x, left_shoulder_y, left_shoulder_v),
            "left_elbow": (left_elbow_x, left_elbow_y, left_elbow_v),
            "left_wrist": (left_wrist_x, left_wrist_y, left_wrist_v),
            "right_shoulder": (right_shoulder_x, right_shoulder_y, right_shoulder_v),
            "right_elbow": (right_elbow_x, right_elbow_y, right_elbow_v),
            "right_wrist": (right_wrist_x, right_wrist_y, right_wrist_v),
        }

        left_arm_visible = min(left_shoulder_v, max(left_elbow_v, left_wrist_v)) >= self._visibility_threshold
        right_arm_visible = min(right_shoulder_v, max(right_elbow_v, right_wrist_v)) >= self._visibility_threshold

        return replace(
            state,
            left_arm_visible=left_arm_visible,
            right_arm_visible=right_arm_visible,
            left_shoulder_x=left_shoulder_x,
            left_shoulder_y=left_shoulder_y,
            right_shoulder_x=right_shoulder_x,
            right_shoulder_y=right_shoulder_y,
            left_elbow_x=left_elbow_x,
            left_elbow_y=left_elbow_y,
            right_elbow_x=right_elbow_x,
            right_elbow_y=right_elbow_y,
            left_hand_x=left_wrist_x,
            left_hand_y=left_wrist_y,
            right_hand_x=right_wrist_x,
            right_hand_y=right_wrist_y,
        )
