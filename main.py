import cv2 as cv
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker

import torch

from Model import Model

camera = cv.VideoCapture(0)

# Load landmark detection model
detector = HandLandmarker.create_from_options(
    HandLandmarkerOptions(BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=1))

# Load pretrained ASL model
model = Model(input_features=63, output_classes=26, dropout=0.15)
model.loadWeights("models/Landmark v1.0.pt")

while True:
    success, image = camera.read()

    if success:
        # Convert image from BGR to RGB
        image[:, :, [0, -1]] = image[:, :, [-1, 0]]

        # Convert to mediapipe image format
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Detect hand landmarks
        detection_results = detector.detect(image)

        if detection_results.handedness:
            landmarks = []
            for landmark in detection_results.hand_landmarks[0]:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Convert to tensor
            landmarks = torch.tensor(landmarks)

            landmarks = torch.unsqueeze(landmarks, dim=0)

            # Predict
            predicted_logits = model(landmarks)

            # Choose the most probable label
            predicted_label = torch.argmax(predicted_logits, dim=-1)

            print(chr(predicted_label + 65))

