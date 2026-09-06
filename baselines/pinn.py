"""
baselines.pinn
==============

Physics-Informed Neural Network (PINN) baseline for comparison against GWR.

This module provides a PyTorch-based feed-forward neural network that can
optionally enforce physics constraints such as monotonicity of the target
with respect to certain features (e.g., dG/dT ≤ 0 for Gibbs energy).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from gwr.utils import compute_all_metrics


# ======================================================================
#  PINN Model
# ======================================================================

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class PINNRegressor(nn.Module):
    """Feed-forward network whose forward pass exposes the gradient w.r.t. the
    first input feature (temperature) for physics-loss computation.

    Parameters
    ----------
    input_dim : int
    hidden_sizes : tuple[int, ...]
    activation : str
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128, 64),
        activation: str = "relu",
    ):
        super().__init__()
        act_cls = ACTIVATIONS.get(activation, nn.ReLU)
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), act_cls()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ======================================================================
#  Training loop
# ======================================================================

def train_pinn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    hidden_sizes: Tuple[int, ...] = (128, 128, 64),
    activation: str = "relu",
    lr: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 256,
    patience: int = 30,
    lambda_physics: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[PINNRegressor, Optional[np.ndarray], Optional[StandardScaler]]:
    """Train a PINN regressor with optional physics-constraint loss.

    The physics loss penalises positive dG/dT (first feature assumed to be
    temperature).  Set ``lambda_physics=0`` to disable.

    Returns
    -------
    model   : PINNRegressor
    y_pred  : np.ndarray or None
    scaler  : StandardScaler or None
    """
    torch.manual_seed(random_state)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PINNRegressor(
        input_dim=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        activation=activation,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb.requires_grad_(True)
            pred = model(xb)
            data_loss = mse_loss(pred, yb)

            # Physics loss: penalise dG/dT > 0
            physics_loss = torch.tensor(0.0, device=device)
            if lambda_physics > 0:
                grad_T = torch.autograd.grad(
                    pred.sum(), xb, create_graph=True
                )[0][:, 0]
                physics_loss = torch.mean(torch.relu(grad_T))

            loss = data_loss + lambda_physics * physics_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Predict
    y_pred = None
    if X_test is not None:
        model.eval()
        with torch.no_grad():
            X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_pred = model(X_te).cpu().numpy()

    return model, y_pred, scaler


def evaluate_pinn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **kwargs,
) -> Dict[str, float]:
    """Train a PINN and return evaluation metrics."""
    _, y_pred, _ = train_pinn(X_train, y_train, X_test, **kwargs)
    return compute_all_metrics(y_test, y_pred)
