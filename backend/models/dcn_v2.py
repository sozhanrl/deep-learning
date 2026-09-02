"""
Deep & Cross Network v2 (DCN-V2) for Tabular Clinical Data.
Combines an explicit bounded-degree Cross Network with a deep MLP tower in parallel,
capturing non-linear and high-order multiplicative cross-feature interactions.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseTabularDLModel


class CrossNetworkV2(nn.Module):
    """
    Cross Network layer with matrix feature crossing:
    x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
    """
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.05)
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for w, b in zip(self.weights, self.biases):
            # xl shape: (B, D)
            # w shape: (D, D) -> xl @ w + b is (B, D)
            xl_w = torch.matmul(xl, w) + b
            xl = x0 * xl_w + xl
        return xl


class DCNV2Network(nn.Module):
    """
    Parallel Deep & Cross Network v2:
    Cross Network + Deep MLP Tower -> Concatenation -> Output Linear.
    """
    def __init__(
        self,
        input_dim: int,
        num_cross_layers: int = 3,
        deep_layers: tuple = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cross_net = CrossNetworkV2(input_dim, num_layers=num_cross_layers)

        # Deep MLP Tower
        tower = []
        in_dim = input_dim
        for h_dim in deep_layers:
            tower.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.Mish(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.deep_tower = nn.Sequential(*tower)

        # Output head combines Cross output (input_dim) + Deep output (deep_layers[-1])
        combined_dim = input_dim + deep_layers[-1]
        self.output_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_out = self.cross_net(x)
        deep_out = self.deep_tower(x)
        combined = torch.cat([cross_out, deep_out], dim=1)
        return self.output_head(combined)


class DCNV2ClassifierDL(BaseTabularDLModel):
    """
    Deep & Cross Network v2 (DCN-V2) Classifier for Alzheimer's clinical features.
    """
    def __init__(
        self,
        num_cross_layers: int = 3,
        deep_layers: tuple = (128, 64, 32),
        dropout: float = 0.2,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 45,
        early_stopping_patience: int = 8,
        random_state: int = 42,
    ):
        super().__init__(
            name="Deep & Cross Network v2 (DCN-V2)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.num_cross_layers = num_cross_layers
        self.deep_layers = deep_layers
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return DCNV2Network(
            input_dim=input_dim,
            num_cross_layers=self.num_cross_layers,
            deep_layers=self.deep_layers,
            dropout=self.dropout,
        )
