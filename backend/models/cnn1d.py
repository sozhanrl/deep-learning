"""
1D-CNN (Convolutional Neural Network) over Tabular Feature Vectors.
Treats feature vectors as contiguous 1D channels to extract local multi-feature spatial patterns.
"""
import torch
import torch.nn as nn
from .base import BaseTabularDLModel


class CNN1DNetwork(nn.Module):
    def __init__(self, input_dim: int, num_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        # Input shape: (Batch, 1, InputDim)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_channels * 2 * 8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to (Batch, Channels=1, SequenceLength=Features)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CNN1DClassifierDL(BaseTabularDLModel):
    """
    1D-CNN architecture applied to structured tabular feature rows.
    """
    def __init__(
        self,
        num_channels: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        super().__init__(
            name="1D-CNN (Local Feature Interactions)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.num_channels = num_channels
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return CNN1DNetwork(input_dim, num_channels=self.num_channels, dropout=self.dropout)
