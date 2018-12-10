import cv2
import sys
import logging as log
from PIL import Image
from time import sleep
import numpy as np
import keras
from constants import *
from utils import *


face_cascade = cv2.CascadeClassifier(CASC_PATH)
log.basicConfig(filename='webcam.log', level=log.INFO)

# load yolo pre-trained model
# yolo_model = keras.models.load_model(FACE_MODEL_FILE)
# yolo_model.load_weights(FACE_MODEL_FILE)


# load emotion model

fer_model = keras.models.load_model(FER_MODEL_FILE)
fer_model.load_weights(FER_MODEL_FILE)

# load gender model
# gender_model = keras.models.load_model(GENDER_MODEL_FILE)
# gender_model.load_weights(GENDER_MODEL_FILE)


video_capture = cv2.VideoCapture(0)

emotion_path = './emoji/'
emotion_face = ['angry.jpg', 'disgust.jpg', 'fear.jpg', 'happy.jpg', 'sad.jpg', 'surprise.jpg', 'neutral.jpg']

emoji_size = (50, 50)
emoji_img = []
for i in range(len(emotion_face)):
    emoji_img.append(cv2.imread(emotion_path + emotion_face[i]))
    emoji_img[i] = cv2.resize(emoji_img[i], emoji_size)

# emoji_img[-1] = cv2.resize(emoji_img[-1], (200, 200))

# print(emoji_img.shape)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with network
    image_faces, faces = format_image(frame, max_face=10)

    if image_faces is None:
        continue
    # try:
    #     cv2.imshow('face1', image_faces[0])
    # except Exception:
    #     print("no face 1")
    #
    # try:
    #     cv2.imshow('face2', image_faces[1])
    # except Exception:
    #     print("no face 2")

    results = []
    print(len(image_faces), "<- image_faces")
    for i, image_face in enumerate(image_faces):
        face = image_face.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        results.append(fer_model.predict(face))

    for i, face in enumerate(faces):
        cv2.rectangle(frame, \
                      (face[0], face[1]), \
                      (face[0] + face[2], face[1] + face[3]), \
                      (255, 0, 0), \
                      2)

        half_width = int(face[2] / 2)
        half_height = int(face[3] / 2)

        emotion_index = np.argmax(results[i])

        for x in range(emoji_size[0]):
            for y in range(emoji_size[1]):
                try:
                    frame[y + face[1] - emoji_size[0], x + face[0] + half_width - int(emoji_size[0] / 2)] = emoji_img[emotion_index][y, x]
                except Exception:
                    print("out of range")

    # if found is not None:
    #     image = found.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
    #     result = fer_model.predict(image)
    # else:
    #     result = None
    #
    # # Write results in frame
    # if result is not None:
    #     for index, emotion in enumerate(EMOTIONS):
    #         # cv2.putText(frame, emotion, (10, index * 20 + 20),
    #         #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    #         #
    #         # cv2.rectangle(frame, \
    #         #               (130, index * 20 + 10), \
    #         #               (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), \
    #         #               (255, 0, 0), \
    #         #               -1)
    #
    #         for face in faces:
    #             cv2.rectangle(frame, \
    #                           (face[0], face[1]), \
    #                           (face[0] + face[2], face[1] + face[3]), \
    #                           (255, 0, 0), \
    #                           2)
    #
    #     # print(result)
    #     face = find_max_area_face(faces)
    #
    #     half_width = int(face[2] / 2)
    #     half_height = int(face[3] / 2)
    #
    #     # frame[height, width]
    #     for x in range(emoji_size[0]):
    #         for y in range(emoji_size[1]):
    #             frame[y + face[1] - emoji_size[0], x + face[0] + half_width - int(emoji_size[0] / 2)] = emoji_img[np.argmax(result)][y, x]
    #
    # # else:
    # #     for x in range(200):
    # #         for y in range(200):
    # #             frame[x, y] = emoji_img[-1][x, y]
    #
    #             # print(frame.shape)
    #     # print(emoji_img.shape)
    #
    #
    #
    # Display the resulting frame
    try:
        cv2.imshow('Video', frame)
    except Exception:
        print("Out of range")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
