import cv2
from OpenSeeFace.tracker import Tracker
import numpy as np


img1 = cv2.imread("characters/parts/head.png")


cv2.imshow("Two Images", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

# tracker = Tracker(480, 640)

# while True:
#     ret, frame = cap.read()
#     height, width, channels = frame.shape
#     faces = tracker.predict(frame)
#     for face in faces:
#         for (x, y, c) in face.lms:
#             cv2.circle(frame, (int(y), int(x)), 1, (0, 0, 255), -1)

#     if not ret:
#         print("Can't receive frame")
#         break

#     cv2.imshow("Webcam", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
