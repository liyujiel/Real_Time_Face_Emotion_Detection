import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from constants import *


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (FACE_SIZE, FACE_SIZE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    cv2.imshow("test", image)
    cv2.waitKey(0)
    return image


face_cascade = cv2.CascadeClassifier(CASC_PATH)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # result = format_image(frame)

    result = None

    if result is not None:
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                          int(result[0][index] * 100), (index + 1) * 20 + 4),
                          (255, 0, 0), -1)

        face_image = feelings_faces[np.argmax(result[0])]

        # Ugly transparent fix
        for c in range(0, 3):
            frame[200:320, 10:130, c] = face_image[:, :, c] * \
                                        (face_image[:, :, 3] / 255.0) + frame[200:320,
                                                                        10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
