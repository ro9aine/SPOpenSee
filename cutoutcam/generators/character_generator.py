from pathlib import Path
from collections import deque
from typing import Mapping, cast

import cv2
import numpy as np

from ..state import BState, EState, FState, FaceState, MState, RState


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
        self._closed_eyes = self._load_part("closed-eyes.png")
        self._pupil = self._load_part("pupil.png")
        self._left_brow = self._load_part("left-brow.png")
        self._shoulder_part = self._load_part("shoulder.png")
        self._arm_part = self._load_part("arm.png")
        self._hand_part = self._load_part("hand.png")
        self._mouth_closed = self._load_part("mouth-1.png")
        self._mouth_open = self._load_part("mouth-2.png")
        self._mouth_wide_open = self._load_part("mouth-3.png")
        self._mouth_talk_4 = self._load_part("mouth-4.png")
        self._mouth_talk_5 = self._load_part("mouth-5.png")
        self._mouth_talk_6 = self._load_part("mouth-6.png")
        self._mouth_talk_7 = self._load_part("mouth-7.png")

        self._bg_image_path: str | None = None
        self._bg_image: np.ndarray | None = None
        self._rng = np.random.default_rng()
        self._speech_energy = 0.0
        self._speech_history: deque[float] = deque(maxlen=8)
        self._last_talking_mouth = 2
        self._mouth_hold_frames = 0

        self.layout = {
            "body_scale": 90,
            "body_x": 0,
            "body_y": 0,
            "hand_enabled": 1,
            "hand_mirror_x": 1,
            "hand_size": 7,
            "hand_x": 0,
            "hand_y": 0,
            "hand_range_x": 70,
            "hand_range_y": 75,
            "hand_shoulder_y": 0,
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
            "turn_amp_x": 14,
            "turn_amp_y": 10,
            "closed_eyes_scale": 100,
            "closed_eyes_x": 0,
            "closed_eyes_y": 0,
            "brow_scale": 100,
            "brow_x": 0,
            "brow_y": 0,
            "mouth_scale": 100,
            "mouth_x": 0,
            "mouth_y": 0,
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

    def set_layout(self, values: Mapping[str, int | str]) -> None:
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

    @staticmethod
    def _brow_shift(state: BState) -> int:
        if state == BState.UP:
            return -8
        if state == BState.DOWN:
            return 8
        return 0

    @staticmethod
    def _merged_brow_state(left: BState, right: BState) -> BState:
        # Simplified rule: both brows share a single state.
        if left == BState.DOWN or right == BState.DOWN:
            return BState.DOWN
        if left == BState.UP or right == BState.UP:
            return BState.UP
        return BState.MIDDLE

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

    def _select_talking_mouth(self, is_speaking: bool, speech_energy: float) -> int:
        energy = max(0.0, min(1.0, float(speech_energy)))

        if not is_speaking:
            self._speech_energy = 0.0
            self._speech_history.clear()
            self._mouth_hold_frames = 0
            self._last_talking_mouth = 2
            return 2

        # React to speech changes faster so mouth shapes update more frequently.
        self._speech_energy = self._speech_energy * 0.45 + energy * 0.55
        self._speech_history.append(self._speech_energy)
        anchor = 2 if self._speech_energy < 0.55 else 3

        # Stable tone (e.g. repeated "a-a-a") should not shuffle mouth shapes.
        stable_voice = False
        if len(self._speech_history) >= 5:
            stable_voice = max(self._speech_history) - min(self._speech_history) < 0.08

        if stable_voice:
            self._mouth_hold_frames = 0
            self._last_talking_mouth = anchor
            return anchor

        if self._mouth_hold_frames > 0:
            self._mouth_hold_frames -= 1
            return self._last_talking_mouth

        weighted_pool = [anchor, anchor, anchor]
        if anchor == 2:
            if self._speech_energy > 0.22:
                weighted_pool.extend([4, 4])
            if self._speech_energy > 0.34:
                weighted_pool.append(5)
        else:
            weighted_pool.extend([6, 6])
            if self._speech_energy > 0.74:
                weighted_pool.append(7)
            if self._speech_energy < 0.65:
                weighted_pool.append(5)

        next_mouth = int(self._rng.choice(np.array(weighted_pool)))
        self._last_talking_mouth = next_mouth
        # Keep at most 1 frame hold to increase mouth shape change rate.
        self._mouth_hold_frames = 0 if self._speech_energy > 0.7 else 1
        return next_mouth

    @staticmethod
    def _draw_hand(canvas: np.ndarray, elbow: tuple[int, int], hand: tuple[int, int], palm_radius: int) -> None:
        arm_color = (205, 190, 175, 255)
        outline_color = (120, 105, 95, 255)
        arm_thickness = max(6, palm_radius)
        outline_thickness = arm_thickness + 2

        cv2.line(canvas, elbow, hand, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.line(canvas, elbow, hand, arm_color, arm_thickness, cv2.LINE_AA)
        cv2.circle(canvas, hand, palm_radius + 2, outline_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, hand, palm_radius, arm_color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_arm_segment(
        canvas: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        thickness: int,
    ) -> None:
        arm_color = (180, 195, 215, 255)
        outline_color = (110, 125, 145, 255)
        cv2.line(canvas, start, end, outline_color, thickness + 4, cv2.LINE_AA)
        cv2.line(canvas, start, end, arm_color, thickness, cv2.LINE_AA)

    @staticmethod
    def _segment_angle(start: tuple[int, int], end: tuple[int, int]) -> float:
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        return float(np.degrees(np.arctan2(dy, dx)) + 90.0)

    @staticmethod
    def _segment_length(start: tuple[int, int], end: tuple[int, int]) -> float:
        return float(np.hypot(end[0] - start[0], end[1] - start[1]))

    def _transform_part(
        self,
        part: np.ndarray,
        anchor: tuple[int, int],
        angle: float,
        scale_x: float,
        scale_y: float,
        mirror_x: bool = False,
    ) -> tuple[np.ndarray, int, int]:
        src = cv2.flip(part, 1) if mirror_x else part
        scaled_w = max(1, int(round(src.shape[1] * max(0.05, scale_x))))
        scaled_h = max(1, int(round(src.shape[0] * max(0.05, scale_y))))
        resized = cv2.resize(src, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        center = (scaled_w / 2.0, scaled_h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_v = abs(matrix[0, 0])
        sin_v = abs(matrix[0, 1])
        bound_w = max(1, int(np.ceil((scaled_h * sin_v) + (scaled_w * cos_v))))
        bound_h = max(1, int(np.ceil((scaled_h * cos_v) + (scaled_w * sin_v))))
        matrix[0, 2] += bound_w / 2.0 - center[0]
        matrix[1, 2] += bound_h / 2.0 - center[1]

        rotated = cv2.warpAffine(
            resized,
            matrix,
            (bound_w, bound_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        original_anchor = np.array([scaled_w / 2.0, 0.0, 1.0], dtype=np.float32)
        rotated_anchor = matrix @ original_anchor
        pos_x = int(round(anchor[0] - rotated_anchor[0]))
        pos_y = int(round(anchor[1] - rotated_anchor[1]))
        return rotated, pos_x, pos_y

    def _overlay_segment_part(
        self,
        canvas: np.ndarray,
        part: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        width_scale: float,
        mirror_x: bool = False,
    ) -> None:
        length = max(8.0, self._segment_length(start, end))
        angle = self._segment_angle(start, end)
        scale_x = width_scale
        scale_y = length / max(1, part.shape[0])
        transformed, pos_x, pos_y = self._transform_part(
            part,
            start,
            angle,
            scale_x=scale_x,
            scale_y=scale_y,
            mirror_x=mirror_x,
        )
        self._overlay_rgba(canvas, transformed, pos_x, pos_y)

    def _overlay_joint_part(
        self,
        canvas: np.ndarray,
        part: np.ndarray,
        center: tuple[int, int],
        size_scale: float,
        mirror_x: bool = False,
    ) -> None:
        src = cv2.flip(part, 1) if mirror_x else part
        scaled_w = max(1, int(round(src.shape[1] * max(0.05, size_scale))))
        scaled_h = max(1, int(round(src.shape[0] * max(0.05, size_scale))))
        resized = cv2.resize(src, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        pos_x = int(round(center[0] - scaled_w / 2.0))
        pos_y = int(round(center[1] - scaled_h / 2.0))
        self._overlay_rgba(canvas, resized, pos_x, pos_y)

    def _arm_points(
        self,
        body_x: int,
        body_y: int,
        body: np.ndarray,
        shoulder_x_ratio: float,
        state_shoulder: tuple[float, float],
        state_elbow: tuple[float, float],
        state_wrist: tuple[float, float],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        shoulder = (
            body_x + int(body.shape[1] * shoulder_x_ratio) + int(self.layout.get("hand_x", 0)),
            body_y + int(body.shape[0] * 0.22) + int(self.layout.get("hand_shoulder_y", 0)),
        )
        motion_scale_x = body.shape[1] * self.layout.get("hand_range_x", 70) / 100.0
        motion_scale_y = body.shape[0] * self.layout.get("hand_range_y", 75) / 100.0
        mirror_x = -1.0 if int(self.layout.get("hand_mirror_x", 1)) > 0 else 1.0

        elbow = (
            int(shoulder[0] + (state_elbow[0] - state_shoulder[0]) * motion_scale_x * mirror_x),
            int(shoulder[1] + (state_elbow[1] - state_shoulder[1]) * motion_scale_y + self.layout.get("hand_y", 0)),
        )
        wrist = (
            int(elbow[0] + (state_wrist[0] - state_elbow[0]) * motion_scale_x * mirror_x),
            int(elbow[1] + (state_wrist[1] - state_elbow[1]) * motion_scale_y),
        )
        return shoulder, elbow, wrist

    @staticmethod
    def _fallback_elbow(shoulder: tuple[int, int], wrist: tuple[int, int], bend_dir: int) -> tuple[int, int]:
        mid_x = (shoulder[0] + wrist[0]) // 2
        mid_y = (shoulder[1] + wrist[1]) // 2
        dx = wrist[0] - shoulder[0]
        dy = wrist[1] - shoulder[1]
        length = max(1.0, float(np.hypot(dx, dy)))
        normal_x = -dy / length
        normal_y = dx / length
        bend = max(18, int(length * 0.18))
        return (
            int(mid_x + normal_x * bend * bend_dir),
            int(mid_y + normal_y * bend * bend_dir),
        )

    def _draw_hands(
        self,
        canvas: np.ndarray,
        body_x: int,
        body_y: int,
        body: np.ndarray,
        state: FaceState,
    ) -> None:
        if int(self.layout.get("hand_enabled", 1)) <= 0:
            return

        size_scale = max(0.25, float(self.layout.get("hand_size", 7)) / 7.0)
        arm_width_scale = max(0.25, size_scale)
        joint_scale = max(0.25, size_scale)

        left_shoulder, left_elbow, left_hand = self._arm_points(
            body_x,
            body_y,
            body,
            0.34,
            (state.left_shoulder_x, state.left_shoulder_y),
            (state.left_elbow_x, state.left_elbow_y),
            (state.left_hand_x, state.left_hand_y),
        )
        right_shoulder, right_elbow, right_hand = self._arm_points(
            body_x,
            body_y,
            body,
            0.66,
            (state.right_shoulder_x, state.right_shoulder_y),
            (state.right_elbow_x, state.right_elbow_y),
            (state.right_hand_x, state.right_hand_y),
        )

        if state.left_arm_visible:
            if abs(left_elbow[0] - left_shoulder[0]) + abs(left_elbow[1] - left_shoulder[1]) < 8:
                left_elbow = self._fallback_elbow(left_shoulder, left_hand, -1)
            self._overlay_segment_part(
                canvas, self._arm_part, left_shoulder, left_elbow, arm_width_scale, mirror_x=False
            )
            self._overlay_segment_part(
                canvas, self._arm_part, left_elbow, left_hand, arm_width_scale, mirror_x=False
            )
            self._overlay_joint_part(canvas, self._shoulder_part, left_shoulder, joint_scale, mirror_x=False)
            self._overlay_joint_part(canvas, self._hand_part, left_hand, joint_scale, mirror_x=False)

        if state.right_arm_visible:
            if abs(right_elbow[0] - right_shoulder[0]) + abs(right_elbow[1] - right_shoulder[1]) < 8:
                right_elbow = self._fallback_elbow(right_shoulder, right_hand, 1)
            self._overlay_segment_part(
                canvas, self._arm_part, right_shoulder, right_elbow, arm_width_scale, mirror_x=True
            )
            self._overlay_segment_part(
                canvas, self._arm_part, right_elbow, right_hand, arm_width_scale, mirror_x=True
            )
            self._overlay_joint_part(canvas, self._shoulder_part, right_shoulder, joint_scale, mirror_x=True)
            self._overlay_joint_part(canvas, self._hand_part, right_hand, joint_scale, mirror_x=True)

    def generate_with_mask(
        self,
        state: FaceState,
        is_speaking: bool = False,
        speech_energy: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        background = self._build_background()
        canvas[:, :, :3] = background
        canvas[:, :, 3] = 255
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        body = self._resized(self._body, max(40, int(self.width * self.layout["body_scale"] / 100)))
        head = self._resized(self._head, max(40, int(self.width * self.layout["head_scale"] / 100)))
        hair = self._resized(self._haircut, max(10, int(head.shape[1] * self.layout["hair_scale"] / 100)))
        eyes_white = self._resized(self._eyes_white, max(10, int(head.shape[1] * self.layout["eyes_scale"] / 100)))
        closed_eyes = self._resized(
            self._closed_eyes,
            max(
                10,
                int(
                    eyes_white.shape[1]
                    * (self._closed_eyes.shape[1] / self._eyes_white.shape[1])
                    * self.layout["closed_eyes_scale"]
                    / 100
                ),
            ),
        )
        pupil = self._resized(self._pupil, max(4, int(head.shape[1] * self.layout["pupil_scale"] / 100)))
        brow = self._resized(
            self._left_brow,
            max(10, int(head.shape[1] * 0.24 * self.layout["brow_scale"] / 100)),
        )
        right_brow = cv2.flip(brow, 1)
        mouth_closed = self._resized(
            self._mouth_closed,
            max(6, int(head.shape[1] * 0.16 * self.layout["mouth_scale"] / 100)),
        )
        mouth_open = self._resized(
            self._mouth_open,
            max(6, int(mouth_closed.shape[1] * (self._mouth_open.shape[1] / self._mouth_closed.shape[1]))),
        )
        mouth_wide_open = self._resized(
            self._mouth_wide_open,
            max(6, int(mouth_closed.shape[1] * (self._mouth_wide_open.shape[1] / self._mouth_closed.shape[1]))),
        )
        mouth_talk_4 = self._resized(
            self._mouth_talk_4,
            max(6, int(mouth_closed.shape[1] * (self._mouth_talk_4.shape[1] / self._mouth_closed.shape[1]))),
        )
        mouth_talk_5 = self._resized(
            self._mouth_talk_5,
            max(6, int(mouth_closed.shape[1] * (self._mouth_talk_5.shape[1] / self._mouth_closed.shape[1]))),
        )
        mouth_talk_6 = self._resized(
            self._mouth_talk_6,
            max(6, int(mouth_closed.shape[1] * (self._mouth_talk_6.shape[1] / self._mouth_closed.shape[1]))),
        )
        mouth_talk_7 = self._resized(
            self._mouth_talk_7,
            max(6, int(mouth_closed.shape[1] * (self._mouth_talk_7.shape[1] / self._mouth_closed.shape[1]))),
        )

        body_x = (self.width - body.shape[1]) // 2 + self.layout["body_x"]
        body_y = self.height - body.shape[0] + self.layout["body_y"]
        body_layer = np.zeros_like(canvas)
        self._overlay_rgba(body_layer, body, body_x, body_y)
        self._overlay_rgba(canvas, body_layer, 0, 0)
        mask = np.maximum(mask, body_layer[:, :, 3])

        head_layer = np.zeros_like(canvas)
        head_x = (self.width - head.shape[1]) // 2 + self.layout["head_x"]
        head_y = int(self.height * 0.12) + self.layout["head_y"]
        self._overlay_rgba(head_layer, head, head_x, head_y)

        eyes_x = head_x + int(head.shape[1] * 0.23) + self.layout["eyes_x"]
        eyes_y = head_y + int(head.shape[0] * 0.38) + self.layout["eyes_y"]
        closed_eyes_x = eyes_x + (eyes_white.shape[1] - closed_eyes.shape[1]) // 2 + self.layout["closed_eyes_x"]
        closed_eyes_y = eyes_y + (eyes_white.shape[0] - closed_eyes.shape[0]) // 2 + self.layout["closed_eyes_y"]
        unified_brow_state = self._merged_brow_state(state.left_brow, state.right_brow)
        brow_y_base = eyes_y - int(brow.shape[0] * 1.10) + self.layout["brow_y"]
        left_brow_x = eyes_x + int(eyes_white.shape[1] * 0.02) + self.layout["brow_x"]
        right_brow_x = (
            eyes_x
            + eyes_white.shape[1]
            - right_brow.shape[1]
            - int(eyes_white.shape[1] * 0.02)
            - self.layout["brow_x"]
        )
        self._overlay_rgba(
            head_layer,
            brow,
            left_brow_x,
            brow_y_base + self._brow_shift(unified_brow_state),
        )
        self._overlay_rgba(
            head_layer,
            right_brow,
            right_brow_x,
            brow_y_base + self._brow_shift(unified_brow_state),
        )

        turn_dx, turn_dy = self._turn_offset(
            state.turn,
            step_x=max(0, int(self.layout["turn_amp_x"])),
            step_y=max(0, int(self.layout["turn_amp_y"])),
        )

        left_eye_cx = eyes_x + int(eyes_white.shape[1] * 0.27)
        right_eye_cx = eyes_x + int(eyes_white.shape[1] * 0.73)
        eye_cy = eyes_y + int(eyes_white.shape[0] * 0.48)

        if state.left_eye == EState.CLOSED or state.right_eye == EState.CLOSED:
            self._overlay_rgba(head_layer, closed_eyes, closed_eyes_x, closed_eyes_y)
        else:
            self._overlay_rgba(head_layer, eyes_white, eyes_x, eyes_y)

            pupil_w = pupil.shape[1]
            pupil_h = pupil.shape[0]
            self._overlay_rgba(
                head_layer,
                pupil,
                left_eye_cx - pupil_w // 2 + turn_dx // 2 + self.layout["pupil_x"],
                eye_cy - pupil_h // 2 + turn_dy // 2 + self.layout["pupil_y"],
            )
            self._overlay_rgba(
                head_layer,
                pupil,
                right_eye_cx - pupil_w // 2 + turn_dx // 2 + self.layout["pupil_x"],
                eye_cy - pupil_h // 2 + turn_dy // 2 + self.layout["pupil_y"],
            )

        if state.mouth == MState.CLOSED:
            mouth = mouth_closed
        elif not is_speaking:
            mouth = mouth_open
            self._select_talking_mouth(False, 0.0)
        else:
            mouth_idx = self._select_talking_mouth(True, speech_energy)
            mouth_map = {
                2: mouth_open,
                3: mouth_wide_open,
                4: mouth_talk_4,
                5: mouth_talk_5,
                6: mouth_talk_6,
                7: mouth_talk_7,
            }
            mouth = mouth_map.get(
                mouth_idx,
                mouth_open,
            )

        mouth_x = head_x + (head.shape[1] - mouth.shape[1]) // 2 + self.layout["mouth_x"] + turn_dx
        mouth_y = head_y + int(head.shape[0] * 0.73) + self.layout["mouth_y"] + turn_dy
        self._overlay_rgba(head_layer, mouth, mouth_x, mouth_y)

        hair_x = head_x + (head.shape[1] - hair.shape[1]) // 2 + self.layout["hair_x"]
        hair_y = head_y - int(head.shape[0] * 0.06) + self.layout["hair_y"]
        self._overlay_rgba(head_layer, hair, hair_x, hair_y)

        angle = self._rotation_angle(state.rotation)
        if abs(angle) > 1e-6:
            center = (head_x + head.shape[1] // 2, head_y + head.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            head_layer = cast(np.ndarray, cv2.warpAffine(
                head_layer,
                matrix,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            ))

        self._overlay_rgba(canvas, head_layer, 0, 0)
        mask = np.maximum(mask, head_layer[:, :, 3])

        hands_layer = np.zeros_like(canvas)
        self._draw_hands(hands_layer, body_x, body_y, body, state)
        self._overlay_rgba(canvas, hands_layer, 0, 0)
        mask = np.maximum(mask, hands_layer[:, :, 3])

        mask = self._apply_mask_cuts(mask)
        full_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        out_bgr = background.copy()
        out_bgr[mask > 0] = full_bgr[mask > 0]
        return out_bgr, mask

    def generate(self, state: FaceState, is_speaking: bool = False, speech_energy: float = 0.0) -> np.ndarray:
        img, _mask = self.generate_with_mask(state, is_speaking=is_speaking, speech_energy=speech_energy)
        return img
