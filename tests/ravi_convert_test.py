#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:51:56 2020

@author: benjamin
"""


import cv2

cap = cv2.VideoCapture("/home/benjamin/Met_ParametersTST/test_ravi/test.ravi")




ret, frame = cap.read()


frame

cv2

frame.shape

cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_Y422)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

