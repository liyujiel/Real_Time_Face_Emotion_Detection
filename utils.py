import cv2
import numpy as np
from constants import *

face_cascade = cv2.CascadeClassifier(CASC_PATH)


def format_image(image, max_face = 1):
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
        return None, None

    image_faces = []
    count = 0
    for face in faces:
        image_face = image[face[1] : face[1] + face[2], face[0] : face[0] + face[3]]
        image_face = _resize_face_img(image_face)
        image_faces.append(image_face)
        count += 1
        if count == max_face:
            break

    try:
        cv2.imshow("face1", image_faces[0])
    except Exception:
        print("no face 1")

    try:
        cv2.imshow("face2", image_faces[1])
    except Exception:
        print("no face 2")

    return image_faces, faces

    # ============= Old ==================

    cv2.imshow("gray", image)

    image = _parsing_max_area_face(faces, image)

    # Resize image to network size
    image = _resize_face_img(image)

    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)
    return image, faces

    # ====================================


def find_max_area_face(faces):
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    return max_area_face


def _parsing_max_area_face(faces, image):
    # Chop image to face
    face = find_max_area_face(faces)

    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    return image


def _resize_face_img(image):
    try:
        image = cv2.resize(image, (FACE_SIZE, FACE_SIZE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)

    return image


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    if not len(indices) > 0:
        return None, None

    # for i in indices:
    #     i = i[0]
    #     box = boxes[i]
    #     left = box[0]
    #     top = box[1]
    #     width = box[2]
    #     height = box[3]
    #     final_boxes.append(box)
    #     draw_predict(frame, confidences[i], left, top, left + width,
    #                  top + height)

    box = boxes[indices[0][0]]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    draw_predict(frame, confidences[indices[0][0]], left, top, left + width,
                     top + height)

    if len(frame.shape) > 2 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = cv2.imdecode(frame, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    image = _parsing_max_area_yolo(box, frame)

    try:
        cv2.imshow("yolo_camera", image)
    except Exception:
        print("face not in frame")
    # Resize image to network size
    resized_face = _resize_face_img(image)

    return final_boxes, resized_face

def _parsing_max_area_yolo(box, frame):
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    # print(left, top, width, height)
    image = frame[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]

    return image

def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)

def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]