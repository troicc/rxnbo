"""A multilayer perceptron in plain PyTorch, with the whole training loop written out.

Nothing here is hidden in a framework: Dataset -> DataLoader -> forward -> loss ->
backward -> optimiser step, a validation set carved from the training rows, and early
stopping on the validation loss. The class follows scikit-learn's fit/predict so the
experiments can treat it like any other model.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...] = (512, 128), dropout: float = 0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPRegressor:
    def __init__(self, hidden=(512, 128), dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 batch_size=64, max_epochs=300, patience=20, val_frac=0.1, seed=0):
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.val_frac, self.seed = val_frac, seed
        self.history: list[tuple[float, float]] = []  # (train loss, validation loss) per epoch

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, np.float32)
        # standardise inputs and target with training statistics only
        # a column constant in training (a fingerprint bit no training reaction sets) keeps
        # scale 1: dividing by its ~0 spread would turn a 1 at test time into ~10^6
        sd = X.std(0)
        self.x_mu, self.x_sd = X.mean(0), np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y)) + 1e-6
        X = (X - self.x_mu) / self.x_sd
        y = (np.asarray(y, np.float32) - self.y_mu) / self.y_sd

        order = rng.permutation(len(X))
        n_val = max(1, int(self.val_frac * len(X)))
        val, tr = order[:n_val], order[n_val:]
        train = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                           batch_size=self.batch_size, shuffle=True,
                           generator=torch.Generator().manual_seed(self.seed))
        Xv, yv = torch.from_numpy(X[val]), torch.from_numpy(y[val])

        self.model = MLP(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best, best_state, waited = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            self.model.train()  # dropout on
            total = 0.0
            for xb, yb in train:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()  # gradients of the loss with respect to every weight
                opt.step()
                total += loss.item() * len(xb)
            self.model.eval()  # dropout off
            with torch.no_grad():
                v = loss_fn(self.model(Xv), yv).item()
            self.history.append((total / len(tr), v))
            if v < best - 1e-4:
                best, best_state, waited = v, copy.deepcopy(self.model.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:  # early stopping: validation has stopped improving
                    break
        self.model.load_state_dict(best_state)
        self.epochs_ = epoch + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = (np.asarray(X, np.float32) - self.x_mu) / self.x_sd
        with torch.no_grad():
            out = self.model(torch.from_numpy(X)).numpy()
        return out * self.y_sd + self.y_mu
