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

# load emotion model

fer_model = keras.models.load_model(FER_MODEL_FILE)
fer_model.load_weights(FER_MODEL_FILE)

# load gender model
# gender_model = keras.models.load_model(GENDER_MODEL_FILE)
# gender_model.load_weights(GENDER_MODEL_FILE)

# TODO: load mix face model

video_capture = cv2.VideoCapture(0)

feelings_faces = []
# for index, emotion in enumerate(EMOTIONS):
#     feelings_faces.append(cv2.imread('./emoji/' + emotion + '.png' ))

emotion_path = './emoji/'
emotion_face = ['angry.jpg', 'disgust.jpg', 'fear.jpg', 'happy.jpg', 'sad.jpg', 'surprise.jpg', 'neutral.jpg', 'none.jpg']

emoji_size = (120, 120)
emoji_img = []
for i in range(len(emotion_face)):
    emoji_img.append(cv2.imread(emotion_path + emotion_face[i]))
    emoji_img[i] = cv2.resize(emoji_img[i], emoji_size)

# print(emoji_img.shape)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # print(frame.shape)
    #get the size of frame
    frame_width = frame.shape[0]
    frame_height = frame.shape[1]

    # Predict result with network
    found = format_image(frame)

    if found is not None:
        image = found.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        # print(image)
        result = fer_model.predict(image)
    else:
        result = None

    # Write results in frame
    if result is not None:
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.rectangle(frame, \
                          (130, index * 20 + 10), \
                          (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), \
                          (255, 0, 0), \
                          -1)

        # print(result)

        for x in range(emoji_size[0]):
            for y in range(emoji_size[1]):
                frame[x, y] = emoji_img[np.argmax(result)][x, y]

    else:
        for x in range(emoji_size[0]):
            for y in range(emoji_size[1]):
                frame[x, y] = emoji_img[7][x, y]

                # print(frame.shape)
        # print(emoji_img.shape)



    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
