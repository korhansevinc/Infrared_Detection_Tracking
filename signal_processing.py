import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("small_target_test2.mp4")
ret, frame1 = cap.read()
prvs_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while(cap.isOpened()):
    ret, frame2 = cap.read()
    if not ret:
        break
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs_gray, next_gray, None, 0.4,3,30,3,7,2.4,0)
    magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    moving_regions = magnitude > 2 # Hareket esigi : 2 pixel per sec.

    moving_objects = np.zeros_like(next_gray)
    moving_objects[moving_regions] = 255
    cv2.imshow("Moving Objects", moving_objects)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    prvs_gray = next_gray
cap.release()
cv2.destroyAllWindows()
