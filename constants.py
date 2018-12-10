FACE_SIZE = 48
IMG_WIDTH = 416
IMG_HEIGHT = 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

EMOTION_SIZE = 50
EMOTIONS = ['angry', 'disgusted', 'fear',
            'happy', 'sad', 'surprised', 'neutral']

CASC_PATH = "./haarcascade/haarcascade_frontalface_default.xml"
DATA_DIR = "./data"
MODEL_CFG = "./cfg/yolov3-face.cfg"
MODEL_WEIGHT_FILE = "./models/yolov3-wider_16000.weights"
FER_MODEL_FILE = "./models/alexnet_epoch10_batch256.h5"
GENDER_MODEL_FILE = "./models/gender_models/simple_CNN.81-0.96.hdf5"
