import cv2
from OpenSeeFace.tracker import Tracker
from SPOpenSee.analizer import Analizer

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracker = Tracker(480, 640, silent=True)
anl = Analizer()

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape
    faces = tracker.predict(frame)
    for face in faces:
        for (x, y, c) in face.lms:
            cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)

    if len(faces) == 1:
        anl.analyze(faces[0])
        print(anl._nose)
    if not ret:
        print("Can't receive frame")
        break

    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
