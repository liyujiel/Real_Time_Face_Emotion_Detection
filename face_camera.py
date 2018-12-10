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

    if image_faces is not None:

        results = []
        for i, image_face in enumerate(image_faces):
            face = image_face.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
            results.append(fer_model.predict(face))

        frame = draw_rectangle(frame, faces)

        # half_width = int(face[2] / 2)
        # half_height = int(face[3] / 2)
        #
        # emotion_index = np.argmax(results[i])
        #
        # for x in range(emoji_size[0]):
        #     for y in range(emoji_size[1]):
        #         try:
        #             frame[y + face[1] - emoji_size[0], x + face[0] + half_width - int(emoji_size[0] / 2)] = emoji_img[emotion_index][y, x]
        #         except Exception:
        #             print("out of range")

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
