"""LSTM forecaster, trust-weighted loss and a small training loop."""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


class FuelLSTM(nn.Module):
    """Single-layer LSTM that maps a window of monthly features to next month's return.

    The network is deliberately small: the training set has fewer than 200
    monthly samples, and larger networks only overfit.
    """

    def __init__(self, n_features: int, hidden_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features) → (batch,)
        sequence_out, _ = self.lstm(x)
        return self.head(sequence_out[:, -1, :]).squeeze(-1)


class TrustWeightedMSELoss(nn.Module):
    """Mean squared error with each sample weighted by its data-quality score.

    Weights are rescaled to average 1 within the batch, so the loss stays on
    the same scale as plain MSE. With equal trust scores it *is* plain MSE.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
        weights = trust / trust.mean()
        return (weights * (pred - target) ** 2).mean()


@dataclass
class TrainResult:
    model: FuelLSTM
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    trust_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    trust_val: np.ndarray,
    *,
    hidden_size: int = 32,
    seed: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 30,
    use_trust: bool = True,
) -> TrainResult:
    """Train one LSTM with early stopping on the validation loss.

    Inputs must already be scaled. The weights from the best validation epoch
    are restored before returning.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32)

    xt, yt, wt = tensor(x_train), tensor(y_train), tensor(trust_train)
    xv, yv, wv = tensor(x_val), tensor(y_val), tensor(trust_val)
    if not use_trust:
        wt, wv = torch.ones_like(wt), torch.ones_like(wv)

    model = FuelLSTM(n_features=xt.shape[-1], hidden_size=hidden_size)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = TrustWeightedMSELoss()
    generator = torch.Generator().manual_seed(seed)

    result = TrainResult(model=model)
    best_val, best_state, epochs_without_gain = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        batch_losses = []
        for batch in order.split(batch_size):
            optimiser.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch], wt[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv, wv).item()
        result.train_loss.append(float(np.mean(batch_losses)))
        result.val_loss.append(val_loss)

        if val_loss < best_val:
            best_val, best_state, epochs_without_gain = val_loss, copy.deepcopy(model.state_dict()), 0
            result.best_epoch = epoch
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return result


def predict(model: FuelLSTM, x: np.ndarray) -> np.ndarray:
    """Run the model on scaled windows and return a NumPy array."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()
