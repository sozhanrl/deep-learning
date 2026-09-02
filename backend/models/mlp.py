"""
MLP / Artificial Neural Network for Tabular Clinical Data Classification.
Deep Multilayer Perceptron with Batch Normalization, Dropout, and LeakyReLU activations.
"""
import torch
import torch.nn as nn
from .base import BaseTabularDLModel


class MLPNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout: float = 0.25):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifierDL(BaseTabularDLModel):
    """
    MLP / ANN Deep Learning Classifier for Alzheimer's structured clinical data.
    """
    def __init__(
        self,
        hidden_dims=(128, 64, 32),
        dropout: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        super().__init__(
            name="ANN / MLP (Deep Neural Net)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return MLPNetwork(input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout)
