"""
NeuroPredict AI - Deep Learning Model Registry & Architecture Factory.
Provides a central registry mapping architectural model names to their PyTorch implementations.
Suite of 8 Tabular Deep Learning Architectures.
"""
from typing import Dict, Type, List
from .base import BaseTabularDLModel
from .mlp import MLPClassifierDL
from .cnn1d import CNN1DClassifierDL
from .tab_transformer import TabTransformerClassifierDL
from .autoencoder import AutoencoderClassifierDL
from .denoising_autoencoder import DenoisingAutoencoderDL
from .wide_and_deep import WideAndDeepClassifierDL
from .tabnet import TabNetClassifierDL
from .dcn_v2 import DCNV2ClassifierDL


# Canonical 8 Deep Learning Model Names
MODEL_REGISTRY: Dict[str, Type[BaseTabularDLModel]] = {
    'ANN / MLP (Deep Neural Net)': MLPClassifierDL,
    '1D-CNN (Local Feature Interactions)': CNN1DClassifierDL,
    'TabTransformer (Feature Self-Attention)': TabTransformerClassifierDL,
    'Deep Autoencoder (Latent + Anomaly Signal)': AutoencoderClassifierDL,
    'Denoising Autoencoder (Robust Invariant Latent)': DenoisingAutoencoderDL,
    'Wide & Deep (Linear + Deep DNN)': WideAndDeepClassifierDL,
    'TabNet (Attentive Tabular Learning)': TabNetClassifierDL,
    'Deep & Cross Network v2 (DCN-V2)': DCNV2ClassifierDL,
}

# Standard Model Names for API / Frontend
DEEP_LEARNING_MODEL_NAMES: List[str] = list(MODEL_REGISTRY.keys())


def create_all_models(random_state: int = 42) -> Dict[str, BaseTabularDLModel]:
    """Instantiate fresh configured instances of all 8 deep learning models."""
    return {
        'ANN / MLP (Deep Neural Net)': MLPClassifierDL(
            hidden_dims=(128, 64, 32), dropout=0.25, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        '1D-CNN (Local Feature Interactions)': CNN1DClassifierDL(
            num_channels=32, dropout=0.2, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        'TabTransformer (Feature Self-Attention)': TabTransformerClassifierDL(
            d_model=24, nhead=4, num_layers=2, dim_feedforward=48, dropout=0.2,
            batch_size=128, learning_rate=1.5e-3, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        'Deep Autoencoder (Latent + Anomaly Signal)': AutoencoderClassifierDL(
            latent_dim=16, dropout=0.15, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        'Denoising Autoencoder (Robust Invariant Latent)': DenoisingAutoencoderDL(
            latent_dim=24, noise_factor=0.2, dropout=0.2, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        'Wide & Deep (Linear + Deep DNN)': WideAndDeepClassifierDL(
            deep_hidden_dims=(128, 64, 32), dropout=0.2, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        'TabNet (Attentive Tabular Learning)': TabNetClassifierDL(
            n_d=24, n_a=24, n_steps=3, gamma=1.3, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
        'Deep & Cross Network v2 (DCN-V2)': DCNV2ClassifierDL(
            num_cross_layers=3, deep_layers=(128, 64, 32), dropout=0.2, learning_rate=2e-3,
            batch_size=128, epochs=45, early_stopping_patience=8, random_state=random_state
        ),
    }
