import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CATEGORIES = 26


def main():
    pass


def load_data(data_dir):
    """Load image data from directory 'data_dir'.
        
    Return tuple '(images, labels)'. 'images' should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT. 'labels' should
    be a list of integer labels, representing the categories for each of the
    corresponding 'images'.
    """
    images = []
    labels = []
    for i in range(NUM_CATEGORIES):
        #sub_dir Ex: gtsrb\21
        os.system('cls')
        print(f"Accessing subdirectory {i+1}/{NUM_CATEGORIES}")
        sub_dir = os.path.join(data_dir, str(i))
        for filename in os.listdir(sub_dir):
            #Load the image as a 'numpy.ndarry'
            img = cv2.imread(os.path.join(sub_dir, filename))
            #Resize the image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            #Add the image and its label to the data lists
            images.append(img)
            labels.append(i)
    return (images, labels)


if __name__ == "__main__":
    main()


