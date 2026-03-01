import cv2
from OpenSeeFace.tracker import Tracker

tracker = Tracker(640, 640)
results = tracker.predict(cv2.imread('./test.jpg'))
for f in results:
    print(f.success)
    print(f.eye_blink[0], f.eye_blink[1])
    print(f.lms)
