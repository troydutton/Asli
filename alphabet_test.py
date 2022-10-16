import tensorflow as tf
import cv2
import string
import numpy as np

from main import ALPHABET

EXIT_KEY = 27
ALPHABET = list(string.ascii_lowercase)

model = tf.keras.models.load_model("model.h5")


img = cv2.imread("data\\amer_sign3.png", cv2.IMREAD_GRAYSCALE)

# Runs through the alphabet in sign language   
for i in range(4):
    for j in range(6):
        sub_img = img[(i * 83):(i*83)+83, (j * 83):(j*83)+83]
        print(ALPHABET[model.predict(np.array(cv2.resize(sub_img, (28, 28))).reshape((-1, 28, 28, 1)).astype(np.uint8)).argmax()])
        cv2.imshow("Image", sub_img)
        if (cv2.waitKey(1000) & 0xFF == EXIT_KEY):
            cv2.destroyAllWindows()
            break

