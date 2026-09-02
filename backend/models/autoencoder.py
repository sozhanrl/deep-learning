"""
Autoencoder Classifier (Semi-Supervised / Anomaly Reconstruction Signal).
Compresses tabular features into a bottleneck latent code, computes reconstruction
anomaly error, and concatenates latent features + reconstruction anomaly into classifier head.
"""
import torch
import torch.nn as nn
from .base import BaseTabularDLModel


class AutoencoderNetwork(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim),
        )

        # Classification head receives latent code + reconstruction error scalar
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        # Compute MSE per-sample reconstruction anomaly error
        recon_error = torch.mean((x - reconstruction) ** 2, dim=1, keepdim=True)
        # Combine latent representation with reconstruction anomaly signal
        combined = torch.cat([latent, recon_error], dim=1)
        return self.classifier(combined)


class AutoencoderClassifierDL(BaseTabularDLModel):
    """
    Autoencoder Dimensionality Reduction + Anomaly Signal Classifier.
    """
    def __init__(
        self,
        latent_dim: int = 16,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        super().__init__(
            name="Deep Autoencoder (Latent + Anomaly Signal)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.latent_dim = latent_dim
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return AutoencoderNetwork(input_dim, latent_dim=self.latent_dim, dropout=self.dropout)
