"""
Denoising Autoencoder (DAE + Supervised Classifier Head).
Applies input feature corruption noise during feature representation learning
to build robust, invariant latent representations for clinical tabular data.
"""
import torch
import torch.nn as nn
from .base import BaseTabularDLModel


class DenoisingAutoencoderNetwork(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 24, noise_factor: float = 0.2, dropout: float = 0.2):
        super().__init__()
        self.noise_factor = noise_factor

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.SiLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 48),
            nn.BatchNorm1d(48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Add Gaussian denoising noise to inputs
            noise = torch.randn_like(x) * self.noise_factor
            corrupted_x = x + noise
        else:
            corrupted_x = x

        latent = self.encoder(corrupted_x)
        # Skip connection: concatenate robust latent representation with direct features
        fused = torch.cat([latent, x], dim=1)
        return self.classifier(fused)


class DenoisingAutoencoderDL(BaseTabularDLModel):
    """
    Denoising Autoencoder architecture with feature skip-connections.
    """
    def __init__(
        self,
        latent_dim: int = 24,
        noise_factor: float = 0.2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        super().__init__(
            name="Denoising Autoencoder (Robust Invariant Latent)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return DenoisingAutoencoderNetwork(
            input_dim,
            latent_dim=self.latent_dim,
            noise_factor=self.noise_factor,
            dropout=self.dropout,
        )
