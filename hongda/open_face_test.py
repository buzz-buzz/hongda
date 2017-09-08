import sys
import dlib
from skimage import io
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_detector = dlib.get_frontal_face_detector()

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    detected_faces = face_detector(frame, 1)
    for i, face_rect in enumerate(detected_faces):
        print("found")
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 3)

    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()