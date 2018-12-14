import scipy.misc as sm
import cv2
import pandas as pd
import numpy as np

NAEMS = ['emotion', 'pixels', 'Usage']
NEW_IMG_SIZE = 96
OLD_IMG_SIZE = 48

def get_img_name(index):
    max_size = 10000

    res = str(index)
    if index < int(max_size):
        res = "0" + res
        max_size /= 10

    res += ".png"
    return res

def main():
    faces_data = pd.read_csv("fer2013.csv", names=NAEMS)
    # print(df.head())

    for index in range(len(faces_data)):
        file_name = get_img_name(index)
        ints = sm.imread(file_name)
        new_pixels = ints.reshape(1, NEW_IMG_SIZE**2)
        new_strings = map(str, new_pixels[0])
        new_string = " ".join(new_strings)
        faces_data['pixels'][index] = new_string
    
    return True

# ints = sm.imread("test.png")
# new_pixels = ints.reshape(1, OLD_IMG_SIZE**2)
# new_strings = map(str, new_pixels[0])
# new_string = " ".join(new_strings)
# print(new_string)

    # image_data = faces_data['pixels'][index]
    # data_array = list(map(float, image_data.split()))
    # data_array = np.asarray(data_array)
    # data_img = data_array.reshape(48, 48)

    # sm.toimage(data_img).save("test.png")
