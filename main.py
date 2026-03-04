import cv2
from opensee.tracker import Tracker
from spopensee.analizer import Analizer
from spopensee.charactergenerator import ConsoleGenerator
from spopensee.charactergenerator import SquareGenerator
from spopensee.state import FaceState


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


cap = open_camera()

if cap is None:
    print("Cannot open camera with available backends/devices")
    raise SystemExit(1)

tracker = Tracker(480, 640, silent=True)
anl = Analizer()
generator = ConsoleGenerator(anl)


def getimg(path):
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # разделяем каналы
    b, g, r, a = cv2.split(img)

    # создаем белый фон
    background = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255

    # нормализуем альфу
    alpha = a / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)

    # смешиваем
    return (img[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)


sq = SquareGenerator()


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Can't receive frame from camera")
        continue

    height, width, channels = frame.shape
    faces = tracker.predict(frame)
    for face in faces:
        for pt_num, (x, y, c) in enumerate(face.lms):
            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0))

    if len(faces) == 1:
        face_state = anl.find_all(faces[0])
        print(face_state.mouth)

        img = sq.generate(face_state)
        cv2.imshow("SquareFace", img)

    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
