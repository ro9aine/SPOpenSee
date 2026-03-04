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

        self._bg_image_path: str | None = None
        self._bg_image: np.ndarray | None = None

        self.layout = {
            "body_scale": 90,
            "body_x": 0,
            "body_y": 0,
            "head_scale": 62,
            "head_x": 0,
            "head_y": 0,
            "hair_scale": 100,
            "hair_x": 0,
            "hair_y": 0,
            "eyes_scale": 55,
            "eyes_x": 0,
            "eyes_y": 0,
            "pupil_scale": 4,
            "pupil_x": 0,
            "pupil_y": 0,
            "bg_mode": 0,
            "bg_r1": 245,
            "bg_g1": 245,
            "bg_b1": 245,
            "bg_r2": 220,
            "bg_g2": 220,
            "bg_b2": 220,
            "bg_center_x": 0,
            "bg_center_y": 0,
            "bg_scale": 100,
            "bg_x": 0,
            "bg_y": 0,
            "mask_left": 0,
            "mask_right": 0,
            "mask_top": 0,
            "mask_bottom": 0,
        }

    def set_layout(self, values: dict[str, int]) -> None:
        for key, value in values.items():
            if key in self.layout:
                self.layout[key] = int(value)

    def set_background_image(self, image_path: str | None) -> None:
        if not image_path:
            self._bg_image_path = None
            self._bg_image = None
            return

        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self._bg_image_path = None
            self._bg_image = None
            return

        self._bg_image_path = str(path)
        self._bg_image = image

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
    def _overlay_alpha(dst_alpha: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
        h, w = src.shape[:2]
        if x >= dst_alpha.shape[1] or y >= dst_alpha.shape[0] or x + w <= 0 or y + h <= 0:
            return

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(dst_alpha.shape[1], x + w)
        y1 = min(dst_alpha.shape[0], y + h)

        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        src_alpha = src[sy0:sy1, sx0:sx1, 3]
        dst_alpha[y0:y1, x0:x1] = np.maximum(dst_alpha[y0:y1, x0:x1], src_alpha)

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

    def _fit_background_image(self, image: np.ndarray, fill_color: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        base_scale = max(self.width / max(1, w), self.height / max(1, h))
        scale_factor = max(0.1, float(self.layout.get("bg_scale", 100)) / 100.0)
        scale = base_scale * scale_factor
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((self.height, self.width, 3), fill_color, dtype=np.uint8)
        pos_x = (self.width - new_w) // 2 + int(self.layout.get("bg_x", 0))
        pos_y = (self.height - new_h) // 2 + int(self.layout.get("bg_y", 0))

        x0 = max(0, pos_x)
        y0 = max(0, pos_y)
        x1 = min(self.width, pos_x + new_w)
        y1 = min(self.height, pos_y + new_h)
        if x1 <= x0 or y1 <= y0:
            return canvas

        sx0 = x0 - pos_x
        sy0 = y0 - pos_y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        canvas[y0:y1, x0:x1] = resized[sy0:sy1, sx0:sx1]
        return canvas

    def _build_background(self) -> np.ndarray:
        mode = int(self.layout.get("bg_mode", 0))
        c1 = np.array(
            [
                self.layout.get("bg_b1", 245),
                self.layout.get("bg_g1", 245),
                self.layout.get("bg_r1", 245),
            ],
            dtype=np.float32,
        )
        c2 = np.array(
            [
                self.layout.get("bg_b2", 220),
                self.layout.get("bg_g2", 220),
                self.layout.get("bg_r2", 220),
            ],
            dtype=np.float32,
        )

        if mode <= 0:
            bg = np.full((self.height, self.width, 3), c1, dtype=np.float32)
        elif mode == 1:
            t = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
            row = c1 * (1.0 - t) + c2 * t
            bg = np.repeat(row, self.width, axis=1)
        elif mode == 2:
            t = np.linspace(0.0, 1.0, self.width, dtype=np.float32)[None, :, None]
            col = c1 * (1.0 - t) + c2 * t
            bg = np.repeat(col, self.height, axis=0)
        elif mode == 3:
            xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
            cx = int(self.width * (0.5 + self.layout.get("bg_center_x", 0) / 200.0))
            cy = int(self.height * (0.5 + self.layout.get("bg_center_y", 0) / 200.0))
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
            max_dist = max(1.0, np.sqrt(max(cx, self.width - cx) ** 2 + max(cy, self.height - cy) ** 2))
            t = np.clip(dist / max_dist, 0.0, 1.0)[:, :, None]
            bg = c1 * (1.0 - t) + c2 * t
        elif self._bg_image is not None:
            return self._fit_background_image(self._bg_image, c1.astype(np.uint8))
        else:
            bg = np.full((self.height, self.width, 3), c1, dtype=np.float32)

        return bg.astype(np.uint8)

    def _apply_mask_cuts(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        cut_left = int(w * max(0, self.layout.get("mask_left", 0)) / 100.0)
        cut_right = int(w * max(0, self.layout.get("mask_right", 0)) / 100.0)
        cut_top = int(h * max(0, self.layout.get("mask_top", 0)) / 100.0)
        cut_bottom = int(h * max(0, self.layout.get("mask_bottom", 0)) / 100.0)

        if cut_left > 0:
            mask[:, :cut_left] = 0
        if cut_right > 0:
            mask[:, max(0, w - cut_right):] = 0
        if cut_top > 0:
            mask[:cut_top, :] = 0
        if cut_bottom > 0:
            mask[max(0, h - cut_bottom):, :] = 0
        return mask

    def generate_with_mask(self, state: FaceState) -> tuple[np.ndarray, np.ndarray]:
        canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        background = self._build_background()
        canvas[:, :, :3] = background
        canvas[:, :, 3] = 255
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        body = self._resized(self._body, max(40, int(self.width * self.layout["body_scale"] / 100)))
        head = self._resized(self._head, max(40, int(self.width * self.layout["head_scale"] / 100)))
        hair = self._resized(self._haircut, max(10, int(head.shape[1] * self.layout["hair_scale"] / 100)))
        eyes_white = self._resized(self._eyes_white, max(10, int(head.shape[1] * self.layout["eyes_scale"] / 100)))
        pupil = self._resized(self._pupil, max(4, int(head.shape[1] * self.layout["pupil_scale"] / 100)))

        body_x = (self.width - body.shape[1]) // 2 + self.layout["body_x"]
        body_y = self.height - body.shape[0] + self.layout["body_y"]
        self._overlay_rgba(canvas, body, body_x, body_y)
        self._overlay_alpha(mask, body, body_x, body_y)

        head_layer = np.zeros_like(canvas)
        head_x = (self.width - head.shape[1]) // 2 + self.layout["head_x"]
        head_y = int(self.height * 0.12) + self.layout["head_y"]
        self._overlay_rgba(head_layer, head, head_x, head_y)

        eyes_x = head_x + int(head.shape[1] * 0.23) + self.layout["eyes_x"]
        eyes_y = head_y + int(head.shape[0] * 0.38) + self.layout["eyes_y"]
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
            left_eye_cx - pupil_w // 2 + turn_dx // 3 + self.layout["pupil_x"],
            eye_cy - pupil_h // 2 + turn_dy // 3 + self.layout["pupil_y"],
        )
        self._overlay_rgba(
            head_layer,
            pupil,
            right_eye_cx - pupil_w // 2 + turn_dx // 3 + self.layout["pupil_x"],
            eye_cy - pupil_h // 2 + turn_dy // 3 + self.layout["pupil_y"],
        )

        hair_x = head_x + (head.shape[1] - hair.shape[1]) // 2 + self.layout["hair_x"]
        hair_y = head_y - int(head.shape[0] * 0.06) + self.layout["hair_y"]
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
        mask = np.maximum(mask, head_layer[:, :, 3])
        mask = self._apply_mask_cuts(mask)
        full_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        out_bgr = background.copy()
        out_bgr[mask > 0] = full_bgr[mask > 0]
        return out_bgr, mask

    def generate(self, state: FaceState) -> np.ndarray:
        img, _mask = self.generate_with_mask(state)
        return img
