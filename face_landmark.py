# -*- coding: utf-8 -*-
import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

my_image = face_recognition.load_image_file("./ibuki.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

known_face_encodings = [
    my_face_encoding,
]
known_face_names = [
    "ibuki yoshinaga",
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # 自分の顔の輪郭を取得
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame)

    process_this_frame = not process_this_frame

    for face_landmark in face_landmarks:
        for face_landmark_key in face_landmark.keys():
            for face_position in face_landmark[face_landmark_key]:
                print(face_position)
                cv2.circle(
                    frame, (face_position[0] * 2, face_position[1] * 2), 2, (255, 255, 255), -1)
                cv2.circle(
                    frame, (face_position[0] * 4, face_position[1] * 4), 2, (255, 255, 255), -1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
