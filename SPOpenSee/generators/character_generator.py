from pathlib import Path

import cv2
import numpy as np

from ..state import FState, FaceState, RState


class CharacterGenerator:
    def __init__(self, char_id: str = "default", width: int = 512, height: int = 512):
        self.char_id = char_id
        self.width = width
        self.height = height
        self._assets_dir = Path("assets") / "characters" / char_id / "parts"

        self._body = self._load_part("body.png")
        self._head = self._load_part("head.png")
        self._haircut = self._load_part("haircut.png")
        self._eyes_white = self._load_part("white-of-the-eyes.png")
        self._pupil = self._load_part("pupil.png")

    def _load_part(self, filename: str) -> np.ndarray:
        path = self._assets_dir / filename
        part = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if part is None:
            raise FileNotFoundError(f"Missing asset: {path}")
        return part

    @staticmethod
    def _overlay_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
        h, w = src.shape[:2]
        if x >= dst.shape[1] or y >= dst.shape[0] or x + w <= 0 or y + h <= 0:
            return

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(dst.shape[1], x + w)
        y1 = min(dst.shape[0], y + h)

        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        src_crop = src[sy0:sy1, sx0:sx1]
        dst_crop = dst[y0:y1, x0:x1]

        alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
        dst_crop[:, :, :3] = (src_crop[:, :, :3] * alpha + dst_crop[:, :, :3] * (1.0 - alpha)).astype(np.uint8)
        dst_crop[:, :, 3:4] = np.maximum(dst_crop[:, :, 3:4], src_crop[:, :, 3:4])

    @staticmethod
    def _rotation_angle(rotation: RState) -> float:
        mapping = {
            RState.NORMAL: 0.0,
            RState.SLIGHTLY_LEFT: 7.0,
            RState.LEFT: 14.0,
            RState.SLIGHTLY_RIGHT: -7.0,
            RState.RIGHT: -14.0,
        }
        return mapping.get(rotation, 0.0)

    @staticmethod
    def _turn_offset(turn: FState, step_x: int, step_y: int) -> tuple[int, int]:
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

    @staticmethod
    def _resized(img: np.ndarray, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        target_h = max(1, int(h * (target_w / w)))
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def generate(self, state: FaceState) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        canvas[:, :, :3] = 245
        canvas[:, :, 3] = 255

        body = self._resized(self._body, int(self.width * 0.90))
        head = self._resized(self._head, int(self.width * 0.62))
        hair = self._resized(self._haircut, head.shape[1])
        eyes_white = self._resized(self._eyes_white, int(head.shape[1] * 0.55))
        pupil = self._resized(self._pupil, max(10, int(head.shape[1] * 0.035)))

        body_x = (self.width - body.shape[1]) // 2
        body_y = self.height - body.shape[0]
        self._overlay_rgba(canvas, body, body_x, body_y)

        head_layer = np.zeros_like(canvas)
        head_x = (self.width - head.shape[1]) // 2
        head_y = int(self.height * 0.12)
        self._overlay_rgba(head_layer, head, head_x, head_y)

        eyes_x = head_x + int(head.shape[1] * 0.23)
        eyes_y = head_y + int(head.shape[0] * 0.38)
        self._overlay_rgba(head_layer, eyes_white, eyes_x, eyes_y)

        turn_dx, turn_dy = self._turn_offset(state.turn, step_x=8, step_y=6)

        left_eye_cx = eyes_x + int(eyes_white.shape[1] * 0.27)
        right_eye_cx = eyes_x + int(eyes_white.shape[1] * 0.73)
        eye_cy = eyes_y + int(eyes_white.shape[0] * 0.48)

        pupil_w = pupil.shape[1]
        pupil_h = pupil.shape[0]
        self._overlay_rgba(
            head_layer,
            pupil,
            left_eye_cx - pupil_w // 2 + turn_dx // 3,
            eye_cy - pupil_h // 2 + turn_dy // 3,
        )
        self._overlay_rgba(
            head_layer,
            pupil,
            right_eye_cx - pupil_w // 2 + turn_dx // 3,
            eye_cy - pupil_h // 2 + turn_dy // 3,
        )

        hair_x = head_x + (head.shape[1] - hair.shape[1]) // 2
        hair_y = head_y - int(head.shape[0] * 0.06)
        self._overlay_rgba(head_layer, hair, hair_x, hair_y)

        angle = self._rotation_angle(state.rotation)
        if abs(angle) > 1e-6:
            center = (head_x + head.shape[1] // 2, head_y + head.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            head_layer = cv2.warpAffine(
                head_layer,
                matrix,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

        self._overlay_rgba(canvas, head_layer, 0, 0)
        return cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
