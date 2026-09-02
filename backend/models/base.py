"""
Base class and utilities for Tabular Deep Learning Models in NeuroPredict AI.
Provides uniform fit / predict / predict_proba / explain interface across all architectures.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Any, Optional

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


def get_device():
    """Detect GPU if available and CUDA is functional, otherwise CPU."""
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device='cuda')
            return torch.device('cuda')
        except Exception:
            return torch.device('cpu')
    return torch.device('cpu')


class BaseTabularDLModel:
    """
    Abstract Base Model for Tabular Deep Learning architectures.
    Provides standard PyTorch training loop, early stopping, prediction, and Captum explanation.
    """

    def __init__(
        self,
        name: str = "BaseDLModel",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 150,
        early_stopping_patience: int = 15,
        random_state: int = 42,
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = get_device()
        self.net: Optional[nn.Module] = None
        self.training_time_seconds: float = 0.0
        self.feature_names: List[str] = []

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    def _build_network(self, input_dim: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_network")

    def fit(self, X, y, val_data=None, feature_names: Optional[List[str]] = None):
        """
        Train the PyTorch network on (X, y) with optional validation monitoring.
        Accepts numpy arrays or pandas DataFrames/Series.
        """
        self._set_seed()
        start_time = time.time()

        if hasattr(X, "values"):
            X_arr = X.values.astype(np.float32)
            if feature_names is None and hasattr(X, "columns"):
                self.feature_names = list(X.columns)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if hasattr(y, "values"):
            y_arr = y.values.astype(np.float32)
        else:
            y_arr = np.asarray(y, dtype=np.float32)

        if feature_names is not None:
            self.feature_names = feature_names

        input_dim = X_arr.shape[1]
        self.net = self._build_network(input_dim).to(self.device)

        # Datasets & Loaders
        train_tensor_x = torch.tensor(X_arr, dtype=torch.float32)
        train_tensor_y = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_data is not None:
            val_X, val_y = val_data
            val_x_arr = val_X.values.astype(np.float32) if hasattr(val_X, "values") else np.asarray(val_X, dtype=np.float32)
            val_y_arr = val_y.values.astype(np.float32) if hasattr(val_y, "values") else np.asarray(val_y, dtype=np.float32)
            val_tensor_x = torch.tensor(val_x_arr, dtype=torch.float32)
            val_tensor_y = torch.tensor(val_y_arr, dtype=torch.float32).unsqueeze(1)
            val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=self.batch_size, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.net.train()
            total_train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.net(bx)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
                optimizer.step()
                total_train_loss += loss.item() * len(bx)

            avg_train_loss = total_train_loss / len(train_dataset)

            eval_loss = avg_train_loss
            if val_loader is not None:
                self.net.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_logits = self.net(vx)
                        if isinstance(v_logits, tuple):
                            v_logits = v_logits[0]
                        v_loss = criterion(v_logits, vy)
                        total_val_loss += v_loss.item() * len(vx)
                eval_loss = total_val_loss / len(val_data[0])

            scheduler.step(eval_loss)

            if eval_loss < best_loss - 1e-4:
                best_loss = eval_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)

        self.training_time_seconds = round(time.time() - start_time, 3)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Compute class probabilities for inputs.
        Returns shape (N, 2): [P(y=0), P(y=1)].
        """
        if self.net is None:
            raise RuntimeError("Model has not been trained yet.")

        self.net.eval()
        if hasattr(X, "values"):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if len(X_arr.shape) == 1:
            X_arr = X_arr.reshape(1, -1)

        tensor_x = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.net(tensor_x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs_pos = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        probs_neg = 1.0 - probs_pos
        return np.column_stack((probs_neg, probs_pos))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions 0 or 1."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def explain(self, x_input, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute feature attribution weights for a single sample using Captum Integrated Gradients
        or gradient-based backpropagation attribution.
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
        tensor_x = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True).to(self.device)

        class ForwardWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                return out[0] if isinstance(out, tuple) else out

        wrapper = ForwardWrapper(self.net)

        attributions = None
        if CAPTUM_AVAILABLE:
            try:
                ig = IntegratedGradients(wrapper)
                baseline = torch.zeros_like(tensor_x)
                attr = ig.attribute(tensor_x, baseline, n_steps=30)
                attributions = attr.detach().cpu().numpy()[0]
            except Exception:
                attributions = None

        if attributions is None:
            tensor_x.grad = None
            out = wrapper(tensor_x)
            out.backward(torch.ones_like(out))
            if tensor_x.grad is not None:
                attributions = (tensor_x.grad * tensor_x).detach().cpu().numpy()[0]
            else:
                attributions = np.ones(len(features))

        raw_scores = np.maximum(0, attributions)
        total = np.sum(np.abs(raw_scores))
        if total == 0:
            normalized = {feat: 1.0 / len(features) for feat in features}
        else:
            normalized = {feat: float(np.abs(raw_scores[i]) / total) for i, feat in enumerate(features[:len(raw_scores)])}

        return normalized
