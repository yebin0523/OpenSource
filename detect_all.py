import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(200, 200))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_gray = gray[y: y+h, x:x+w]
            
            smiles = smile_cascade.detectMultiScale(face_gray, 1.8, 20, minSize=(50, 20))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)
                cv2.putText(frame, 'Smile', (x+sx, y+sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(30,30))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
                cv2.putText(frame, 'Eye', (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()