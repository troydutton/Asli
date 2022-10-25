import tensorflow as tf
import cv2
import os, string

from main import ALPHABET

EXIT_KEY = 27
IMG_WIDTH = 200
IMG_HEIGHT = 200
ALPHABET = list(string.ascii_lowercase)
DATA_DIR = "data\\asl_alphabet_test\\asl_alphabet_test"

model = tf.keras.models.load_model("new_model.h5")


for filename in os.listdir(DATA_DIR):
    img = cv2.imread(os.path.join(DATA_DIR, filename))
    #Resize the image
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    print(ALPHABET[model.predict(img.reshape(-1, 200, 200, 3)).argmax()])

