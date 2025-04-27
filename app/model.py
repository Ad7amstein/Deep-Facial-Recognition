"""Model Architecture"""
import torch
from torch import nn

class EmbeddingModel(nn.Module):
    """Embedding model"""
    def __init__(self, in_shape, out_shape):
        """Initialize the model"""
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.ReLU(),
        )

        self.linear_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=out_shape),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward method"""
        x = self.conv_block(x)
        x = self.linear_block(x)
        return x


class L1Dist(nn.Module):
    """L1 Distance Model"""
    def __init__(self):
        """Initialize the model"""
        super().__init__()

    def forward(self, in_embedding, val_embedding):
        """Calculates L1 Distance"""
        return torch.abs(in_embedding - val_embedding)


class SiameseNN(nn.Module):
    """Siamese Neural Network Model"""
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingModel(in_shape=3, out_shape=4096)
        self.l1_dist = L1Dist()
        self.classifier = nn.Linear(4096, 1)

    def forward(self, x_anc, x_val):
        """Forward method"""
        anc_embedding = self.embedding(x_anc)
        val_embedding = self.embedding(x_val)
        l1_distance = self.l1_dist(anc_embedding, val_embedding)
        return self.classifier(l1_distance)
