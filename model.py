import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os, sys

EPOCHS = 8
IMG_WIDTH = 200
IMG_HEIGHT = 200
NUM_CATEGORIES = 26
DATA_DIR = "data\\asl_alphabet_train"

def main():
    # Get image arrays and labels
    images, labels = load_data(DATA_DIR)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    training_images, testing_images, training_labels, testing_labels = train_test_split(
        np.array(images), np.array(labels), test_size=.4
    )
    # Get the model
    model = get_model()

    # Fit model on training data
    model.fit(training_images, training_labels, epochs=EPOCHS)


    # Evaluate neural network performance
    model.evaluate(testing_images,  testing_labels, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")


def get_model():
    """
    Returns a compiled model. Assume that the
    'input_shape' is (IMG_WIDTH, IMG_HEIGHT, 1)
    The output layer should have `NUM_CATEGORIES`
    """

    model = tf.keras.models.Sequential(
        [   
            #Convolutional layer
            tf.keras.layers.Conv2D(32, (2, 2), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation = "relu"),
            tf.keras.layers.Conv2D(64, (2, 2), activation = "relu"),
            #Poolin
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(.25),


            #Convolutional layer
            tf.keras.layers.Conv2D(32, (2, 2), activation = "relu"),
            tf.keras.layers.Conv2D(64, (2, 2), activation = "relu"),
            #Poolin
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(.25),

            #Flat as a pancake
            tf.keras.layers.Flatten(),

            #Hidden layer with 128 nodes, ReLU activation
            tf.keras.layers.Dense(1024, activation = "relu"),
            tf.keras.layers.Dropout(.5),
            
            #Output layer
            tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
        ]
    )

    #Compile and return the model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0005), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model


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
    j = 0
    for i in range(ord('A'), ord('Z') + 1):
        os.system('cls')
        print(f"Accessing subdirectory {chr(i)}/Z")
        sub_dir = os.path.join(data_dir, chr(i))
        for filename in os.listdir(sub_dir):
            #Load the image as a 'numpy.ndarry'
            img = cv2.imread(os.path.join(sub_dir, filename))
            #Resize the image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            #Add the image and its label to the data lists
            images.append(img)
            labels.append(j)
        j += 1
    return (images, labels)


if __name__ == "__main__":
    main()


