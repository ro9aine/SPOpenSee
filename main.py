from collections import deque
from pathlib import Path

import cv2

from opensee.tracker import Tracker
from spopensee.analyzers import FaceAnalyzer
from spopensee.app_controls import CONTROL_WINDOW, LayoutControls
from spopensee.app_runtime import open_camera, open_virtual_camera, smooth_face_state, to_rgb
from spopensee.generators import CharacterGenerator
from spopensee.state import FaceState


SMOOTHING_WINDOW = 4
WEBCAM_WINDOW = "Webcam"
CHARACTER_WINDOW = "Character"
CHARACTER_WIDTH = 1280
CHARACTER_HEIGHT = 720


def main() -> None:
    cap = open_camera()
    if cap is None:
        print("Cannot open camera with available backends/devices")
        raise SystemExit(1)

    tracker = Tracker(480, 640, silent=True)
    analyzer = FaceAnalyzer()
    controls = LayoutControls()

    char_id = "default"
    layout_path = Path("configs") / "characters" / f"{char_id}_layout.json"
    character = CharacterGenerator(char_id, width=CHARACTER_WIDTH, height=CHARACTER_HEIGHT)

    current_layout = controls.load_settings(layout_path)
    control_limits = controls.create_window(current_layout)
    character.set_layout(current_layout)
    if isinstance(current_layout.get("bg_image_path"), str):
        character.set_background_image(current_layout.get("bg_image_path"))

    state_history: deque[FaceState] = deque(maxlen=SMOOTHING_WINDOW)
    virtual_cam = open_virtual_camera(character.width, character.height, fps=30)
    last_state = FaceState()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Can't receive frame from camera")
            continue

        faces = tracker.predict(frame)
        for face in faces:
            for pt_num, (x, y, c) in enumerate(face.lms):
                cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
                frame = cv2.putText(
                    frame,
                    str(pt_num),
                    (int(y), int(x)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (255, 255, 0),
                )

        current_layout.update(controls.read_values(control_limits))
        character.set_layout(current_layout)

        if len(faces) == 1:
            state_history.append(analyzer.find_all(faces[0]))
            last_state = smooth_face_state(state_history)

        character_frame, _mask_frame = character.generate_with_mask(last_state)

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

    cap.release()
    if virtual_cam is not None:
        virtual_cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
