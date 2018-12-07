import cv2
import sys
import logging as log
from time import sleep
import numpy as np
import keras
from constants import *
from utils import *


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

emoji_size = (50, 50)
emoji_img = []

for i in range(len(emotion_face)):
    emoji_img.append(cv2.imread(emotion_path + emotion_face[i]))
    emoji_img[i] = cv2.resize(emoji_img[i], emoji_size)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Create a 4D blob from a frame.
    blob = cv2.UMat(cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False))

    # Predict result with network
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    face, resized_face = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    if resized_face is not None:
        image = resized_face.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        result = fer_model.predict(image)
    else:
        result = None

    # Write results in frame
    if result is not None:
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                          int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

        # print(found)

        half_width = int(face[2] / 2)
        half_height = int(face[3] / 2)

        # frame[height, width]
        for x in range(emoji_size[0]):
            for y in range(emoji_size[1]):
                frame[y + face[1] - emoji_size[0], x + face[0] + half_width - int(emoji_size[0] / 2)] = \
                emoji_img[np.argmax(result)][y, x]



    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
