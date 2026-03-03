import cv2
from opensee.tracker import Tracker
from spopensee.analizer import Analizer
from spopensee.charactergenerator import ConsoleGenerator

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

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
    alpha = np.stack([alpha]*3, axis=-1)

    # смешиваем
    return (img[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)


while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape
    faces = tracker.predict(frame)
    for face in faces:
        for pt_num, (x, y, c) in enumerate(face.lms):
            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)
            frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0))

    if len(faces) == 1:
        state = anl.find_brows_state(faces[0])
        print(state)
        # if anl._nose == Analizer.NoseState.CENTER:
        #     cv2.imshow("sp", getimg('./characters/parts/front.png'))
        # elif anl._nose == Analizer.NoseState.LEFT:
        #     cv2.imshow("sp", getimg('./characters/parts/left.png'))
        # elif anl._nose == Analizer.NoseState.RIGHT:
        #     cv2.imshow("sp", getimg(
        #         './characters/parts/right.png'))
        # elif anl._nose == Analizer.NoseState.UP:
        #     cv2.imshow("sp", getimg(
        #         './characters/parts/top.png'))
        # elif anl._nose == Analizer.NoseState.DOWN:
        #     cv2.imshow("sp", getimg(
        #         './characters/parts/bottom.png'))
        # generator.generate()
        # print(anl._get_state(faces[0]))
    if not ret:
        print("Can't receive frame")
        break
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
