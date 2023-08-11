import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

import mediapipe as mp
from mediapipe.python.solutions import drawing_utils, drawing_styles, hands

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from Model import Model


class MNISTDataset(Dataset):
    """
    MNIST American Sign Language Dataset. Contains 28x28 grayscale images of 24 different classes.
    Each class corresponds to a different letter of the alphabet, with J and Z excluded due to motion being
    required in their gestures.
    """

    def __init__(self, path: str) -> None:
        data = np.loadtxt(path, delimiter=',', dtype='uint8')
        self.labels, self.images = data[:, 0], data[:, 1:].reshape((-1, 28, 28))

        # Add a single channel (N, H, W, 1)
        self.images = np.expand_dims(self.images, -1)

        # Tile to imitate RGB (N, H, W, 3)
        self.images = np.tile(self.images, (1, 1, 1, 3))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ASLDataset(Dataset):
    """
    American Sign Language Alphabet Dataset. Contains 200x200 RGB images of 26 different classes.
    Each class corresponds to a different letter of the alphabet.
    """

    def __init__(self, path: str) -> None:
        # Initialize empty arrays to store images and labels
        self.images = []
        self.labels = []

        for directory in os.listdir(path):
            # Convert directory name into a label
            label = ord(directory) - 65

            # Create a path to the directory
            directory_path = os.path.join(path, directory)

            for file in os.listdir(directory_path):
                # Create a path to the file
                file_path = os.path.join(directory_path, file)

                # Load the image
                with Image.open(file_path) as f:
                    # Convert image to numpy array
                    image = np.array(f)

                    # Save the image and label together
                    self.images.append(image)
                    self.labels.append(label)

        # Convert populated lists into numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
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


def train(model: nn.Module, data_loader: DataLoader, epochs):
    model.train()

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize tensorboard logging
    log = SummaryWriter(log_dir="./logs/Landmark V1.0")
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        num_batches = 0
        for landmark, gt_label in data_loader:
            # Send to device
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

    log.close()


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Data
dataset = LandmarkDataset("data/ASL/train.h5")

# Train Model
train(model=Model(input_features=63, output_classes=26).to(device),
      data_loader=DataLoader(dataset, batch_size=10000),
      epochs=1000)
