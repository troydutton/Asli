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
