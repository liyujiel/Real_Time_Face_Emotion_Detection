import cv2
import sys
import logging as log
from PIL import Image
from time import sleep
import numpy as np
import keras
from constants import *
from utils import *
import time
from math import sqrt

face_cascade = cv2.CascadeClassifier(CASC_PATH)
log.basicConfig(filename='webcam.log', level=log.INFO)

# load yolo pre-trained model
# yolo_model = keras.models.load_model(FACE_MODEL_FILE)
# yolo_model.load_weights(FACE_MODEL_FILE)


# load emotion model

fer_model = keras.models.load_model(ENSEMBLE_MODEL_FILE)
# fer_model.load_weights(FER_MODEL_FILE)

# load gender model
# gender_model = keras.models.load_model(GENDER_MODEL_FILE)
# gender_model.load_weights(GENDER_MODEL_FILE)

video_capture = cv2.VideoCapture(0)

emotion_path = './emoji/'
emotion_face = ['angry.jpg', 'disgust.jpg', 'fear.jpg', 'happy.jpg', 'sad.jpg', 'surprise.jpg', 'neutral.jpg']

emoji_size = EMOTION_SIZE
emoji_img = []
for i in range(len(emotion_face)):
    emoji_img.append(cv2.imread(emotion_path + emotion_face[i]))
    emoji_img[i] = cv2.resize(emoji_img[i], (emoji_size, emoji_size))

emoji_codes = []
emoji_center = EMOTION_SIZE / 2
threshold = EMOTION_THRESHOLD
color_threshold = COLOR_THRESHOLD

for i, emoji in enumerate(emoji_img):
    emoji_code = []
    for x in range(EMOTION_SIZE):
        x_list = []
        for y in range(EMOTION_SIZE):
            dis = sqrt((float(x) - emoji_center) ** 2 + (float(y) - emoji_center) ** 2)
            if emoji[x,y][0] >= color_threshold and \
                emoji[x,y][1] >= color_threshold and \
                emoji[x,y][2] >= color_threshold and \
                dis > threshold:

                x_list.append(1)
            else:
                x_list.append(0)

        emoji_code.append(x_list)
    emoji_codes.append(emoji_code)


while True:

    start = time.time()

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with network
    image_faces, faces = format_image(frame, max_face=10)

    if image_faces is not None:
        results = predict_emotions(image_faces, fer_model)

        frame = draw_rectangle(frame, faces)

        try:
            frame = draw_emotions(frame, emoji_img, results, faces, emoji_codes)
        except Exception:
            print("drawing failed")

    end = time.time()
    fps = round(1.0 / (end - start))

    cv2.putText(frame, "fps: " + str(fps), (0, 15),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

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
