import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_features, output_classes, dropout):
        super(Model, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(256, output_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.head(x)
        return x

    def saveWeights(self, path: str) -> None:
        """Save model weights to a file path"""
        torch.save(self.state_dict(), path)

    def loadWeights(self, path: str) -> None:
        """Load model weights from a file path"""
        self.load_state_dict(torch.load(path))