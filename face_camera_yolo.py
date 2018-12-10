import cv2
import sys
import logging as log
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
# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHT_FILE)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)


# load emotion model
fer_model = keras.models.load_model(FER_MODEL_FILE)
fer_model.load_weights(FER_MODEL_FILE)

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
            if emoji[x,y][0] >= COLOR_THRESHOLD and \
                emoji[x,y][1] >= COLOR_THRESHOLD and \
                emoji[x,y][2] >= COLOR_THRESHOLD and \
                dis > threshold:

                x_list.append(1)
            else:
                x_list.append(0)

        emoji_code.append(x_list)
    emoji_codes.append(emoji_code)




while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    start = time.time()

    # Create a 4D blob from a frame.
    blob = cv2.UMat(cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False))

    # Predict result with network
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    faces, resized_faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    # predict the emotion with face by emotion model
    results = []
    if resized_faces is not None:
        for resized_face in resized_faces:
            if resized_face is None:
                continue
            else:
                image = resized_face.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
                results.append(fer_model.predict(image))
    else:
        results = None

    # Write results in frame
    if results is not None:

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
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
