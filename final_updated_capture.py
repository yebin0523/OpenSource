import cv2
import numpy as np

# 얼굴 검출을 위한 Haar Cascade Classifier 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # 노트북 웹캠을 카메라로 사용
cap.set(3, 640)  # 너비
cap.set(4, 480)  # 높이

while True:
    ret, frame = cap.read()  # 웹캠으로부터 프레임 읽기
    frame = cv2.flip(frame, 1)  # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)

    if len(faces) > 0:
        # 얼굴 중심 좌표 구하기
        face_centers = [(x + w // 2, y + h // 2) for x, y, w, h in faces]
        # 가운데 얼굴 찾기
        center_face_idx = np.argmin(np.sum(np.abs(np.array(face_centers) - np.mean(face_centers, axis=0)), axis=1))

        for i, (x, y, w, h) in enumerate(faces):
            if i == center_face_idx:  # 가운데 얼굴은 그대로 나오도록
                continue
            else:
                # 나머지 얼굴은 모자이크 처리
                face_img = frame[y:y + h, x:x + w]  # 탐지된 얼굴 이미지 crop
                face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)  # 축소
                face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)  # 확대
                frame[y:y + h, x:x + w] = face_img  # 탐지된 얼굴 영역 모자이크 처리

    cv2.imshow('result', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's' 키를 누르면 현재 프레임을 저장
        cv2.imwrite('self_camera_photo.jpg', frame)
        print("Photo saved!")

    elif key == 27:  # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
