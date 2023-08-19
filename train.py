import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from Model import Model


class ASLDataset(Dataset):
    """
    American Sign Language Alphabet Dataset. Contains 200x200 RGB images of 26 different classes.
    Each class corresponds to a different letter of the alphabet.
    """

    def __init__(self, path: str) -> None:
        # Initialize empty arrays to store images and labels
        self.images = []
        self.labels = []

        for directory in tqdm(os.listdir(path), desc="ASL"):
            # Convert directory name into a label
            label = ord(directory) - 65

            # Create a path to the directory
            directory_path = os.path.join(path, directory)

            for file in os.listdir(directory_path):
                # Create a path to the file
                file_path = os.path.join(directory_path, file)

                # Load the image using Pillow
                with Image.open(file_path) as f:
                    # Convert image to numpy array
                    image = np.asarray(f)

                    # Save the image and label together
                    self.images.append(image)
                    self.labels.append(label)

        # Convert populated lists into numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> (np.ndarray, int):
        return self.images[idx], self.labels[idx]


class LandmarkDataset(Dataset):
    """
    Collection of hand landmarks generated from other datasets using Mediapipe. Each datapoint contains 21 landmarks,
    each with an x, y, and z location.
    """

    def __init__(self, path: str):
        with h5py.File(path, 'r') as f:
            self.landmarks = f.get("landmarks")[:].astype('float32')
            self.labels = f.get("labels")[:].astype('int64')

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]


def generateLandmarkData(dataset: ASLDataset) -> (np.ndarray, np.ndarray):
    """
    Convert a generic image dataset into a set of landmark positions
    """
    # Load hand detection model
    detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(BaseOptions(model_asset_path='hand_landmarker.task'), num_hands=1))

    landmarks = []
    labels = []

    for i in tqdm(range(len(dataset)), desc="Images"):
        image, label = dataset[i]

        # Convert to mediapipe image format
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Detect hand landmarks
        detection_results = detector.detect(image)

        if detection_results.handedness:
            for landmark in detection_results.hand_landmarks[0]:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            labels.append(label)

    return np.array(landmarks), np.array(labels)


def calculateModelAccuracy(model: nn.Module, test_data: DataLoader) -> float:
    correct_predictions = 0

    model.eval()

    with torch.no_grad():
        for landmarks, gt_labels in test_data:
            # Send batch to device
            landmarks = landmarks.to(device)
            gt_labels = gt_labels.to(device)

            # Predict
            predicted_logits = model(landmarks)

            # Choose the most probable label
            predicted_labels = torch.argmax(predicted_logits, dim=-1)

            # Add the number of correct predictions
            correct_predictions += torch.count_nonzero(predicted_labels == gt_labels)

    return correct_predictions / len(test_data.dataset)


def train(model: nn.Module, train_data: DataLoader, test_data: DataLoader, epochs: int) -> None:
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize tensorboard logging
    log = SummaryWriter("./logs/Landmark V1.0")
    for epoch in tqdm(range(epochs)):
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        for landmark, gt_label in train_data:
            # Send batch to device
            landmark = landmark.to(device)
            gt_label = gt_label.to(device)

            # Predict
            predicted_label = model(landmark)

            # Calculate loss
            loss = loss_fn(predicted_label, gt_label)

            # Back-propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            num_batches += 1

        # Log epoch loss
        log.add_scalar("Loss", epoch_loss / num_batches, epoch)

        # Log epoch accuracy
        log.add_scalar("Validation Accuracy", calculateModelAccuracy(model, test_data), epoch)

    log.close()


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Data
dataset = LandmarkDataset("data/train.h5")

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create model
model = Model(input_features=63, output_classes=26, dropout=0.15)

# Initialize model weights
model.apply(model.initWeights)

# Send model to device
model.to(device)

# Train Model
train(model=model,
      train_data=DataLoader(train_dataset, batch_size=1000),
      test_data=DataLoader(test_dataset, batch_size=1000),
      epochs=3000)

# Save model weights
model.saveWeights("models/Landmark v1.0.pt")


