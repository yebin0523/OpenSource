import numpy as np
import cv2

cap = cv2.VideoCapture(0) # 노트북 웹캠을 카메라로 사용
cap.set(3,640) # 너비
cap.set(4,480) # 높이

ret, frame = cap.read() # 사진 촬영
frame = cv2.flip(frame, 1) # 좌우 대칭

cv2.imwrite('self camera.jpg', frame) # 사진 저장
<<<<<<< HEAD

cap.release()
cv2.destroyAllWindows()

#https://github.com/kimjinho1/Real-time-face-recognition-and-mosaic-using-deep-learning/blob/master/0.%20opencv_self_camera.ipynb
cap.release()
cv2.destroyAllWindows()
