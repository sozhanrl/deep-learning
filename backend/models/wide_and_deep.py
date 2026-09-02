"""
Wide & Deep Network for Tabular Clinical Data.
Combines a Wide linear model (memorization of key direct linear feature influences)
with a Deep feed-forward network (generalization of non-linear feature interactions).
"""
import torch
import torch.nn as nn
from .base import BaseTabularDLModel


class WideAndDeepNetwork(nn.Module):
    def __init__(self, input_dim: int, deep_hidden_dims=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        # Wide component: linear connection directly from input features to output
        self.wide = nn.Linear(input_dim, 1, bias=False)

        # Deep component: multi-layer non-linear representation
        deep_layers = []
        in_dim = input_dim
        for h_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        deep_layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*deep_layers)

        # Bias term combining wide + deep
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        return wide_out + deep_out + self.bias


class WideAndDeepClassifierDL(BaseTabularDLModel):
    """
    Wide & Deep neural network classifier for tabular Alzheimer's patient records.
    """
    def __init__(
        self,
        deep_hidden_dims=(128, 64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        super().__init__(
            name="Wide & Deep (Linear + Deep DNN)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.deep_hidden_dims = deep_hidden_dims
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return WideAndDeepNetwork(
            input_dim,
            deep_hidden_dims=self.deep_hidden_dims,
            dropout=self.dropout,
        )
