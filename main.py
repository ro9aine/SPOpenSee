import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from opensee.tracker import Tracker
from cutoutcam.analyzers import FaceAnalyzer, PoseAnalyzer
from cutoutcam.app_controls import CONTROL_WINDOW, LayoutControls, LayoutValue
from cutoutcam.app_runtime import open_camera, open_virtual_camera, smooth_face_state, to_rgb
from cutoutcam.audio_input import MicSpeechInput
from cutoutcam.generators import CharacterGenerator
from cutoutcam.state import FaceState


SMOOTHING_WINDOW = 2
WEBCAM_WINDOW = "Webcam"
CHARACTER_WINDOW = "Character"
CHARACTER_WIDTH = 1280
CHARACTER_HEIGHT = 720
SHOW_LANDMARK_LABELS = False
POSE_EVERY_N_FRAMES = 2


@dataclass
class QtPreviewState:
    layout: dict[str, LayoutValue]
    points_frame: Any | None = None
    character_frame: Any | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot_layout(self) -> dict[str, LayoutValue]:
        with self.lock:
            return self.layout.copy()

    def update_layout(self, layout: dict[str, LayoutValue]) -> None:
        with self.lock:
            self.layout = layout.copy()

    def publish_frames(self, points_frame: Any | None, character_frame: Any) -> None:
        with self.lock:
            self.points_frame = points_frame
            self.character_frame = character_frame

    def snapshot_frames(self) -> tuple[Any | None, Any | None]:
        with self.lock:
            return self.points_frame, self.character_frame

    def set_error(self, message: str) -> None:
        with self.lock:
            self.error = message

    def get_error(self) -> str | None:
        with self.lock:
            return self.error


def run_qt_worker(args: argparse.Namespace, preview_state: QtPreviewState, stop_event: threading.Event) -> None:
    cap = open_camera()
    if cap is None:
        preview_state.set_error("Cannot open camera with available backends/devices")
        return

    tracker = Tracker(480, 640, silent=True)
    analyzer = FaceAnalyzer()
    pose_analyzer = PoseAnalyzer() if not args.disable_hands else None
    character = CharacterGenerator("default", width=CHARACTER_WIDTH, height=CHARACTER_HEIGHT)
    mic_input = MicSpeechInput() if not args.disable_mic else None
    virtual_cam = None
    state_history: deque[FaceState] = deque(maxlen=SMOOTHING_WINDOW)
    last_state = FaceState()
    frame_index = 0
    bg_image_path: str | None = None

    try:
        layout = preview_state.snapshot_layout()
        character.set_layout(layout)
        initial_bg_path = layout.get("bg_image_path")
        if isinstance(initial_bg_path, str):
            bg_image_path = initial_bg_path
            character.set_background_image(bg_image_path)
        virtual_cam = open_virtual_camera(character.width, character.height, fps=30)
        if mic_input is not None:
            mic_input.start()

        while not stop_event.is_set():
            layout = preview_state.snapshot_layout()
            character.set_layout(layout)
            bg_value = layout.get("bg_image_path")
            next_bg_path: str | None = bg_value if isinstance(bg_value, str) else None
            if next_bg_path != bg_image_path:
                bg_image_path = next_bg_path
                character.set_background_image(bg_image_path)

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            tracking_enabled = int(layout.get("ui_tracking_enabled", 1)) > 0
            hands_enabled = int(layout.get("ui_hands_enabled", 1)) > 0
            mic_runtime_enabled = int(layout.get("ui_mic_enabled", 1)) > 0
            points_preview_enabled = int(layout.get("ui_show_points", 1)) > 0

            if tracking_enabled:
                faces = tracker.predict(frame)
                if points_preview_enabled:
                    for face in faces:
                        for pt_num, (x, y, c) in enumerate(face.lms):
                            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
                            if SHOW_LANDMARK_LABELS:
                                frame = cv2.putText(
                                    frame,
                                    str(pt_num),
                                    (int(y), int(x)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.25,
                                    (255, 255, 0),
                                )
            else:
                faces = []

            if tracking_enabled and len(faces) == 1:
                state_history.append(analyzer.find_all(faces[0]))
                last_state = smooth_face_state(state_history)

            if hands_enabled and pose_analyzer is not None and frame_index % POSE_EVERY_N_FRAMES == 0:
                last_state = pose_analyzer.enrich_state(frame, last_state)
            if points_preview_enabled and hands_enabled and pose_analyzer is not None:
                pose_analyzer.draw_debug(frame)

            mic_enabled = int(layout.get("mic_enabled", 1)) > 0
            if mic_runtime_enabled and mic_enabled and mic_input is not None and mic_input.running:
                mic_threshold = max(0.01, min(1.0, int(layout.get("mic_threshold", 8)) / 100.0))
                mic_gain = max(0.1, int(layout.get("mic_gain", 200)) / 10.0)
                is_speaking, speech_energy = mic_input.read_state(threshold=mic_threshold, gain=mic_gain)
            else:
                is_speaking = int(layout.get("speech_active", 0)) > 0
                speech_energy = max(0.0, min(1.0, int(layout.get("speech_energy", 0)) / 100.0))

            character_frame, _mask_frame = character.generate_with_mask(
                last_state,
                is_speaking=is_speaking,
                speech_energy=speech_energy,
            )
            preview_state.publish_frames(frame if points_preview_enabled else None, character_frame)

            if virtual_cam is not None:
                virtual_cam.send(to_rgb(character_frame))
                virtual_cam.sleep_until_next_frame()

            frame_index += 1
    except Exception as exc:
        preview_state.set_error(str(exc))
    finally:
        cap.release()
        if virtual_cam is not None:
            virtual_cam.close()
        if mic_input is not None:
            mic_input.close()
        if pose_analyzer is not None:
            pose_analyzer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-hands", action="store_true", help="Disable pose-based arm and hand tracking.")
    parser.add_argument("--disable-mic", action="store_true", help="Disable microphone speech input handling.")
    parser.add_argument("--ui", choices=("opencv", "qt"), default="qt", help="Choose the controls UI backend.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    controls: Any = LayoutControls()
    if args.ui == "qt":
        try:
            from cutoutcam.qt_controls import QT_AVAILABLE, QtLayoutControls

            if QT_AVAILABLE:
                controls = QtLayoutControls()
            else:
                print("PySide6 is not installed. Falling back to OpenCV controls.")
        except Exception as exc:
            print(f"Cannot start Qt controls ({exc}). Falling back to OpenCV controls.")

    char_id = "default"
    layout_path = Path("configs") / "characters" / f"{char_id}_layout.json"
    current_layout: dict[str, LayoutValue] = controls.load_settings(layout_path)
    control_limits = controls.create_window(current_layout)

    if controls.uses_opencv_window:
        cap = open_camera()
        if cap is None:
            print("Cannot open camera with available backends/devices")
            raise SystemExit(1)

        tracker = Tracker(480, 640, silent=True)
        analyzer = FaceAnalyzer()
        pose_analyzer = PoseAnalyzer() if not args.disable_hands else None
        character = CharacterGenerator(char_id, width=CHARACTER_WIDTH, height=CHARACTER_HEIGHT)
        character.set_layout(current_layout)
        bg_image_path = current_layout.get("bg_image_path")
        if isinstance(bg_image_path, str):
            character.set_background_image(bg_image_path)

        state_history: deque[FaceState] = deque(maxlen=SMOOTHING_WINDOW)
        virtual_cam = open_virtual_camera(character.width, character.height, fps=30)
        mic_input = MicSpeechInput() if not args.disable_mic else None
        if mic_input is not None:
            mic_input.start()
        last_state = FaceState()
        frame_index = 0
        if args.disable_hands:
            print("Pose-based arm tracking is disabled by argument.")
        elif pose_analyzer is not None and not pose_analyzer.available:
            reason = pose_analyzer.unavailable_reason or "unknown reason"
            print(f"Pose-based arm tracking is disabled: {reason}")
        if args.disable_mic:
            print("Microphone speech input is disabled by argument.")

        while True:
            controls.process_events()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Can't receive frame from camera")
                continue

            tracking_enabled = controls.is_tracking_enabled()
            hands_enabled = controls.is_hands_enabled()
            mic_runtime_enabled = controls.is_mic_enabled()
            points_preview_enabled = controls.is_points_preview_enabled()

            if tracking_enabled:
                faces = tracker.predict(frame)
                if points_preview_enabled:
                    for face in faces:
                        for pt_num, (x, y, c) in enumerate(face.lms):
                            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
                            if SHOW_LANDMARK_LABELS:
                                frame = cv2.putText(
                                    frame,
                                    str(pt_num),
                                    (int(y), int(x)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.25,
                                    (255, 255, 0),
                                )
            else:
                faces = []

            current_layout.update(controls.read_values(control_limits))
            character.set_layout(current_layout)

            if tracking_enabled and len(faces) == 1:
                state_history.append(analyzer.find_all(faces[0]))
                last_state = smooth_face_state(state_history)

            if hands_enabled and pose_analyzer is not None and frame_index % POSE_EVERY_N_FRAMES == 0:
                last_state = pose_analyzer.enrich_state(frame, last_state)
            if points_preview_enabled and hands_enabled and pose_analyzer is not None:
                pose_analyzer.draw_debug(frame)

            mic_enabled = int(current_layout.get("mic_enabled", 1)) > 0
            if mic_runtime_enabled and mic_enabled and mic_input is not None and mic_input.running:
                mic_threshold = max(0.01, min(1.0, int(current_layout.get("mic_threshold", 8)) / 100.0))
                mic_gain = max(0.1, int(current_layout.get("mic_gain", 200)) / 10.0)
                is_speaking, speech_energy = mic_input.read_state(threshold=mic_threshold, gain=mic_gain)
                current_layout["speech_active"] = int(is_speaking)
                current_layout["speech_energy"] = int(max(0, min(100, round(speech_energy * 100.0))))
            else:
                is_speaking = int(current_layout.get("speech_active", 0)) > 0
                speech_energy = max(0.0, min(1.0, int(current_layout.get("speech_energy", 0)) / 100.0))
            character_frame, _mask_frame = character.generate_with_mask(
                last_state,
                is_speaking=is_speaking,
                speech_energy=speech_energy,
            )

            cv2.imshow(WEBCAM_WINDOW, frame)
            cv2.imshow(CHARACTER_WINDOW, character_frame)
            cv2.imshow(CONTROL_WINDOW, controls.draw_overlay(current_layout))

            if virtual_cam is not None:
                virtual_cam.send(to_rgb(character_frame))
                virtual_cam.sleep_until_next_frame()

            key = cv2.waitKey(1) & 0xFF
            if controls.pop_action("save"):
                controls.save_settings(layout_path, current_layout)
                print(f"Layout saved to {layout_path}")
            if controls.pop_action("page_prev"):
                controls.goto_prev_page()
                control_limits = controls.create_window(current_layout)
            if controls.pop_action("page_next"):
                controls.goto_next_page()
                control_limits = controls.create_window(current_layout)
            if controls.pop_action("pick_bg"):
                bg_path = controls.choose_background_file()
                if bg_path:
                    current_layout["bg_image_path"] = bg_path
                    current_layout["bg_mode"] = 4
                    character.set_background_image(bg_path)
                    control_limits = controls.create_window(current_layout)
            if controls.pop_action("clear_bg"):
                current_layout.pop("bg_image_path", None)
                if int(current_layout.get("bg_mode", 0)) == 4:
                    current_layout["bg_mode"] = 0
                character.set_background_image(None)
                control_limits = controls.create_window(current_layout)
            if controls.pop_action("quit") or key == 27:
                controls.save_settings(layout_path, current_layout)
                break
            frame_index += 1

        cap.release()
        if virtual_cam is not None:
            virtual_cam.close()
        if mic_input is not None:
            mic_input.close()
        if pose_analyzer is not None:
            pose_analyzer.close()
    else:
        preview_state = QtPreviewState(current_layout.copy())
        stop_event = threading.Event()
        worker = threading.Thread(target=run_qt_worker, args=(args, preview_state, stop_event), daemon=True)
        worker.start()

        while True:
            controls.process_events()
            current_layout.update(controls.read_values(control_limits))
            preview_state.update_layout(current_layout)
            controls.draw_overlay(current_layout)

            points_frame, qt_character_frame = preview_state.snapshot_frames()
            if qt_character_frame is not None:
                if controls.is_points_preview_enabled():
                    controls.update_previews(points_frame, qt_character_frame)
                else:
                    controls.update_previews(None, qt_character_frame)

            if controls.pop_action("save"):
                controls.save_settings(layout_path, current_layout)
                print(f"Layout saved to {layout_path}")
            if controls.pop_action("pick_bg"):
                bg_path = controls.choose_background_file()
                if bg_path:
                    current_layout["bg_image_path"] = bg_path
                    current_layout["bg_mode"] = 4
                    preview_state.update_layout(current_layout)
            if controls.pop_action("clear_bg"):
                current_layout.pop("bg_image_path", None)
                if int(current_layout.get("bg_mode", 0)) == 4:
                    current_layout["bg_mode"] = 0
                preview_state.update_layout(current_layout)

            worker_error = preview_state.get_error()
            if worker_error is not None:
                print(worker_error)
                break
            if controls.pop_action("quit"):
                controls.save_settings(layout_path, current_layout)
                break
            time.sleep(0.005)

        stop_event.set()
        worker.join(timeout=2.0)

    controls.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
