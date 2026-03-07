import cv2
import numpy as np

from ..state import BState, EState, EmotionState, FState, FaceState, MState, RState


class SquareGenerator:
    def __init__(
        self,
        width: int = 320,
        height: int = 320,
        bg_color: tuple[int, int, int] = (245, 245, 245),
        fg_color: tuple[int, int, int] = (30, 30, 30),
    ):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.fg_color = fg_color

    def _turn_offset(self, turn: FState, step_x: int, step_y: int) -> tuple[int, int]:
        offsets = {
            FState.CENTER: (0, 0),
            FState.LEFT: (-step_x, 0),
            FState.RIGHT: (step_x, 0),
            FState.UP: (0, -step_y),
            FState.DOWN: (0, step_y),
            FState.UP_LEFT: (-step_x, -step_y),
            FState.UP_RIGHT: (step_x, -step_y),
            FState.DOWN_LEFT: (-step_x, step_y),
            FState.DOWN_RIGHT: (step_x, step_y),
        }
        return offsets.get(turn, (0, 0))

    def _rotation_angle(self, rotation: RState) -> float:
        mapping = {
            RState.NORMAL: 0.0,
            RState.SLIGHTLY_LEFT: 7.0,
            RState.LEFT: 14.0,
            RState.SLIGHTLY_RIGHT: -7.0,
            RState.RIGHT: -14.0,
        }
        return mapping.get(rotation, 0.0)

    def _draw_eye(self, canvas: np.ndarray, center: tuple[int, int], state: EState):
        cx, cy = center

        if state == EState.CLOSED:
            cv2.line(canvas, (cx - 18, cy), (cx + 18, cy), self.fg_color, 3)
            return

        if state == EState.HALF:
            cv2.rectangle(canvas, (cx - 18, cy - 6), (cx + 18, cy + 3), (255, 255, 255), -1)
            cv2.rectangle(canvas, (cx - 18, cy - 6), (cx + 18, cy + 3), self.fg_color, 2)
            cv2.circle(canvas, (cx, cy), 4, self.fg_color, -1)
            return

        cv2.circle(canvas, (cx, cy), 14, (255, 255, 255), -1)
        cv2.circle(canvas, (cx, cy), 14, self.fg_color, 2)
        cv2.circle(canvas, (cx, cy), 5, self.fg_color, -1)

    def _draw_brow(self, canvas: np.ndarray, center: tuple[int, int], state: BState, side: str):
        cx, cy = center

        y_shift = 0
        if state == BState.UP:
            y_shift = -8
        elif state == BState.DOWN:
            y_shift = 7

        if side == "left":
            p1 = (cx - 20, cy + y_shift + 2)
            p2 = (cx + 20, cy + y_shift - 2)
        else:
            p1 = (cx - 20, cy + y_shift - 2)
            p2 = (cx + 20, cy + y_shift + 2)

        if state == BState.MIDDLE:
            p1 = (cx - 20, cy + y_shift)
            p2 = (cx + 20, cy + y_shift)

        cv2.line(canvas, p1, p2, self.fg_color, 3)

    def _draw_mouth(self, canvas: np.ndarray, center: tuple[int, int], mouth: MState, emotion: EmotionState):
        cx, cy = center

        if mouth == MState.WIDE_OPEN:
            cv2.ellipse(canvas, (cx, cy), (26, 18), 0, 0, 360, self.fg_color, 2)
            cv2.ellipse(canvas, (cx, cy), (24, 16), 0, 0, 360, (0, 0, 180), -1)
            return

        if mouth == MState.OPEN:
            cv2.ellipse(canvas, (cx, cy), (22, 10), 0, 0, 360, self.fg_color, 2)
            return

        if emotion == EmotionState.HAPPY:
            cv2.ellipse(canvas, (cx, cy - 2), (24, 12), 0, 20, 160, self.fg_color, 3)
        elif emotion == EmotionState.SAD:
            cv2.ellipse(canvas, (cx, cy + 10), (24, 12), 0, 200, 340, self.fg_color, 3)
        else:
            cv2.line(canvas, (cx - 20, cy), (cx + 20, cy), self.fg_color, 3)

    def generate(self, state: FaceState) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        cx = self.width // 2
        cy = self.height // 2

        head_half_w = int(self.width * 0.33)
        head_half_h = int(self.height * 0.36)
        cv2.rectangle(
            canvas,
            (cx - head_half_w, cy - head_half_h),
            (cx + head_half_w, cy + head_half_h),
            self.fg_color,
            3,
        )

        turn_dx, turn_dy = self._turn_offset(state.turn, step_x=24, step_y=18)

        left_eye = (cx - 50 + turn_dx * 2 // 3, cy - 40 + turn_dy * 2 // 3)
        right_eye = (cx + 50 + turn_dx * 2 // 3, cy - 40 + turn_dy * 2 // 3)

        self._draw_brow(canvas, (left_eye[0], left_eye[1] - 28), state.left_brow, "left")
        self._draw_brow(canvas, (right_eye[0], right_eye[1] - 28), state.right_brow, "right")

        self._draw_eye(canvas, left_eye, state.left_eye)
        self._draw_eye(canvas, right_eye, state.right_eye)

        nose_center = (cx + turn_dx, cy + turn_dy)
        cv2.circle(canvas, nose_center, 5, self.fg_color, -1)

        mouth_center = (cx + turn_dx // 2, cy + 55 + turn_dy * 2 // 3)
        self._draw_mouth(canvas, mouth_center, state.mouth, state.emotion)

        angle = self._rotation_angle(state.rotation)
        if abs(angle) > 1e-6:
            center = (self.width // 2, self.height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            canvas = cv2.warpAffine(
                canvas,
                matrix,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.bg_color,
            )

        return canvas

    def draw_on(self, frame: np.ndarray, state: FaceState, x: int = 10, y: int = 10) -> np.ndarray:
        face_img = self.generate(state)
        h, w = face_img.shape[:2]

        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame

        frame[y:y + h, x:x + w] = face_img
        return frame
