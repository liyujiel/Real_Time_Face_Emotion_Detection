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

feelings_faces = []
# for index, emotion in enumerate(EMOTIONS):
#     feelings_faces.append(cv2.imread('./emoji/' + emotion + '.png' ))

emotion_path = './emoji/'
emotion_face = ['angry.jpg', 'disgust.jpg', 'fear.jpg', 'happy.jpg', 'sad.jpg', 'surprise.jpg', 'neutral.jpg', 'none.jpg']

emoji_size = (50, 50)
emoji_img = []
for i in range(len(emotion_face)):
    emoji_img.append(cv2.imread(emotion_path + emotion_face[i]))
    emoji_img[i] = cv2.resize(emoji_img[i], emoji_size)

# print(emoji_img.shape)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with network
    found, faces = format_image(frame)

    if found is not None:
        image = found.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
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

            for face in faces:
                cv2.rectangle(frame, \
                              (face[0], face[1]), \
                              (face[0] + face[2], face[1] + face[3]), \
                              (255, 0, 0), \
                              2)

        # print(result)
        face = find_max_area_face(faces)

        print(face)
        half_width = int(face[2] / 2)
        half_height = int(face[3] / 2)

        print(half_width, half_height)

        for x in range(emoji_size[0]):
            for y in range(emoji_size[1]):
                frame[x + face[0] + half_width, y] = emoji_img[np.argmax(result)][x, y]

    else:
        for x in range(emoji_size[0]):
            for y in range(emoji_size[1]):
                frame[x, y] = emoji_img[-1][x, y]

                # print(frame.shape)
        # print(emoji_img.shape)



    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
