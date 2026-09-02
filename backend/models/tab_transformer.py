"""
TabTransformer / Tabular Multi-Head Self-Attention Network.
Transforms feature dimensions into learned embedding tokens and applies self-attention
to model all pairwise feature-to-feature relationships.
"""
import torch
import torch.nn as nn
from .base import BaseTabularDLModel


class TabTransformerNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 24,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 48,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Vectorized feature embedder: weight shape (input_dim, d_model)
        self.weight = nn.Parameter(torch.randn(input_dim, d_model) * 0.05)
        self.bias = nn.Parameter(torch.zeros(input_dim, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

        # Positional / Feature ID Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification MLP head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * d_model, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Vectorized projection: x has shape (B, input_dim) -> (B, input_dim, 1) * (1, input_dim, d_model)
        x_expanded = x.unsqueeze(-1) # (B, input_dim, 1)
        tokens = x_expanded * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) # (B, input_dim, d_model)
        tokens = self.act(self.layer_norm(tokens))

        seq = tokens + self.pos_embedding
        attended = self.transformer_encoder(seq)
        flat = attended.view(batch_size, -1)
        return self.classifier(flat)


class TabTransformerClassifierDL(BaseTabularDLModel):
    """
    TabTransformer / Attention Network for Alzheimer's clinical feature interactions.
    """
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 8e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        super().__init__(
            name="TabTransformer (Feature Self-Attention)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return TabTransformerNetwork(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
