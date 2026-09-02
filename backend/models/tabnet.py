"""
TabNet: Attentive Interpretable Tabular Learning (Google Research / PyTorch Native).
Implements sequential attention decision steps with Sparsemax / entmax feature masks
for sparse feature selection and built-in explainability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from .base import BaseTabularDLModel


class Sparsemax(nn.Module):
    """Sparsemax activation function for sparse attention masks."""
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_dim = input_tensor.dim()
        if self.dim != -1 and self.dim != input_dim - 1:
            input_tensor = input_tensor.transpose(self.dim, -1)

        original_size = input_tensor.size()
        input_flat = input_tensor.contiguous().view(-1, input_tensor.size(-1))
        dim = input_flat.size(-1)

        # Sort input in descending order
        sorted_input, _ = torch.sort(input_flat, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_input, dim=-1)
        k_indices = torch.arange(1, dim + 1, device=input_tensor.device, dtype=input_tensor.dtype).view(1, -1)

        # Determine support size k
        support = (sorted_input * k_indices > (cumsum - 1.0)).float()
        k = torch.sum(support, dim=-1, keepdim=True)
        tau = (torch.gather(cumsum, 1, (k.long() - 1)) - 1.0) / k

        output_flat = torch.clamp(input_flat - tau, min=0.0)
        output = output_flat.view(original_size)

        if self.dim != -1 and self.dim != input_dim - 1:
            output = output.transpose(self.dim, -1)
        return output


class FeatureTransformer(nn.Module):
    """Shared and decision step-dependent GLU blocks for TabNet."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim * 2)
        self.bn1 = nn.BatchNorm1d(output_dim * 2)
        self.fc2 = nn.Linear(output_dim, output_dim * 2)
        self.bn2 = nn.BatchNorm1d(output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First GLU layer
        h1 = self.bn1(self.fc1(x))
        h1 = h1[:, :h1.size(1)//2] * torch.sigmoid(h1[:, h1.size(1)//2:])
        # Second GLU layer with residual connection
        h2 = self.bn2(self.fc2(h1))
        h2 = h2[:, :h2.size(1)//2] * torch.sigmoid(h2[:, h2.size(1)//2:])
        return (h1 + h2) * np.sqrt(0.5)


class TabNetNetwork(nn.Module):
    """
    Sequential Attention TabNet Architecture.
    Computes sparse feature masks at N decision steps and aggregates prediction outputs.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_d: int = 24,       # Dimension of prediction decision layer
        n_a: int = 24,       # Dimension of attention masking layer
        n_steps: int = 3,    # Number of sequential decision steps
        gamma: float = 1.3,  # Coefficient for feature reuse penalty
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.initial_transformer = FeatureTransformer(input_dim, n_d + n_a)

        # Step-specific feature transformers & attentive transformers
        self.step_transformers = nn.ModuleList([
            FeatureTransformer(input_dim, n_d + n_a) for _ in range(n_steps)
        ])
        self.attentive_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, input_dim),
                nn.BatchNorm1d(input_dim),
                Sparsemax(dim=-1),
            ) for _ in range(n_steps)
        ])

        self.final_classifier = nn.Linear(n_d, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.initial_bn(x)
        prior_scales = torch.ones_like(x)
        out_aggregated = torch.zeros((x.size(0), self.n_d), device=x.device)
        aggregated_masks = torch.zeros_like(x)

        # Initial representation
        init_repr = self.initial_transformer(x_norm)
        a_prev = init_repr[:, self.n_d:]

        for step in range(self.n_steps):
            # 1. Attentive Transformer produces sparse feature mask M_i
            mask_raw = self.attentive_transformers[step](a_prev)
            mask = mask_raw * prior_scales
            aggregated_masks = aggregated_masks + mask

            # Update prior scales to penalize re-using already heavily used features
            prior_scales = prior_scales * (self.gamma - mask)

            # 2. Masked input through Feature Transformer
            masked_x = mask * x_norm
            step_repr = self.step_transformers[step](masked_x)

            d = step_repr[:, :self.n_d]
            a_prev = step_repr[:, self.n_d:]

            # 3. Decision step output aggregation
            d_act = F.relu(d)
            out_aggregated = out_aggregated + d_act

        logits = self.final_classifier(out_aggregated)
        # Normalize feature importance masks
        total_mask = aggregated_masks / self.n_steps
        return logits, total_mask


class TabNetClassifierDL(BaseTabularDLModel):
    """
    TabNet (Attentive Tabular Learning) Classifier for structured clinical Alzheimer's data.
    Provides native interpretability through its sequential sparse attention masks.
    """
    def __init__(
        self,
        n_d: int = 24,
        n_a: int = 24,
        n_steps: int = 3,
        gamma: float = 1.3,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 45,
        early_stopping_patience: int = 8,
        random_state: int = 42,
    ):
        super().__init__(
            name="TabNet (Attentive Tabular Learning)",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
        )
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

    def _build_network(self, input_dim: int) -> nn.Module:
        return TabNetNetwork(
            input_dim=input_dim,
            output_dim=1,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
        )

    def explain(self, x_input, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract native sparse attention masks from TabNet for sample attribution.
        """
        if self.net is None:
            raise RuntimeError("Model has not been trained yet.")

        self.net.eval()
        if hasattr(x_input, "values"):
            x_arr = x_input.values.astype(np.float32)
        else:
            x_arr = np.asarray(x_input, dtype=np.float32)

        if len(x_arr.shape) == 1:
            x_arr = x_arr.reshape(1, -1)

        features = feature_names or self.feature_names or [f"f{i}" for i in range(x_arr.shape[1])]
        tensor_x = torch.tensor(x_arr, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, mask = self.net(tensor_x)
            mask_scores = mask.cpu().numpy()[0]

        total = np.sum(np.abs(mask_scores))
        if total == 0:
            normalized = {feat: 1.0 / len(features) for feat in features}
        else:
            normalized = {feat: float(np.abs(mask_scores[i]) / total) for i, feat in enumerate(features[:len(mask_scores)])}

        return normalized
