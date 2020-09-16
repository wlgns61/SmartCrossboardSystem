import cv2
import numpy as np
from main import Signal

cross_upper = [[1,1], [50, 80], [250, 126]]
cross_lower = [[1,1], [120, 20], [250, 126]]
people = [(80, 60)]

capture = cv2.VideoCapture("testvideo.mp4")

point1 = np.array([[10,10], [170,10], [200,230], [70,70], [50,150]])
signal = Signal(wait=0)
while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("Image/Star.mp4")

    ret, frame = capture.read()
    #cv2.polylines(frame, [point1], True, (255, 0, 0), 3
    signal(people, cross_upper, cross_lower, cross_upper, cross_lower)

    #cv2.line(frame, people[0], people[0], (0, 255, 0), 3)
    #cv2.line(frame, people[1], people[1], (0, 255, 0), 3)
    cv2.line(frame, people[0], people[0], (0,255,0), 5)
    cv2.polylines(frame, [np.array(cross_upper + cross_lower[1:-1])], True, (255, 0, 0), 3)
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(33) > 0: break

capture.release()
cv2.destroyAllWindows()