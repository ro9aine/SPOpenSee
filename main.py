from collections import Counter, deque

import cv2
from opensee.tracker import Tracker
from spopensee.analyzers import FaceAnalyzer
from spopensee.generators import CharacterGenerator
from spopensee.state import FaceState


SMOOTHING_WINDOW = 4


def open_camera():
    # Prefer DirectShow on Windows to avoid common MSMF read failures.
    attempts = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
    ]

    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()

    return None


def majority_with_recent_tiebreak(history: deque[FaceState], attr: str):
    values = [getattr(state, attr) for state in history]
    counts = Counter(values)
    max_count = max(counts.values())
    leaders = {value for value, count in counts.items() if count == max_count}

    # If tie, prefer the latest seen value for responsiveness.
    for state in reversed(history):
        value = getattr(state, attr)
        if value in leaders:
            return value

    return values[-1]


def smooth_face_state(history: deque[FaceState]) -> FaceState:
    smoothed = FaceState()
    smoothed.turn = majority_with_recent_tiebreak(history, "turn")
    smoothed.left_eye = majority_with_recent_tiebreak(history, "left_eye")
    smoothed.right_eye = majority_with_recent_tiebreak(history, "right_eye")
    smoothed.mouth = majority_with_recent_tiebreak(history, "mouth")
    smoothed.emotion = majority_with_recent_tiebreak(history, "emotion")
    smoothed.left_brow = majority_with_recent_tiebreak(history, "left_brow")
    smoothed.right_brow = majority_with_recent_tiebreak(history, "right_brow")
    smoothed.rotation = majority_with_recent_tiebreak(history, "rotation")
    return smoothed


cap = open_camera()

if cap is None:
    print("Cannot open camera with available backends/devices")
    raise SystemExit(1)

tracker = Tracker(480, 640, silent=True)
anl = FaceAnalyzer()
char = CharacterGenerator("default")
state_history: deque[FaceState] = deque(maxlen=SMOOTHING_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Can't receive frame from camera")
        continue

    faces = tracker.predict(frame)
    for face in faces:
        for pt_num, (x, y, c) in enumerate(face.lms):
            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0))

    if len(faces) == 1:
        current_state = anl.find_all(faces[0])
        state_history.append(current_state)

        smoothed_state = smooth_face_state(state_history)
        img = char.generate(smoothed_state)
        cv2.imshow("Character", img)

    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


