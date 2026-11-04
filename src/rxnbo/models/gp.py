"""Exact Gaussian-process regression, written out in PyTorch.

A GP puts a prior on functions: any finite set of outputs f(x_1..x_n) is jointly Gaussian
with mean m and covariance K_ij = k(x_i, x_j). Observing y = f(X) + noise, the posterior
at new points X* is again Gaussian:

    mean  mu*  = m + K*^T (K + s_n^2 I)^-1 (y - m)
    cov   S*   = K** - K*^T (K + s_n^2 I)^-1 K*

Both are computed through one Cholesky factor L of (K + s_n^2 I). The hyperparameters
(signal variance s_f^2, noise s_n^2, constant mean m, and a length scale for the RBF
kernel) are chosen by maximising the log marginal likelihood

    log p(y) = -1/2 (y-m)^T (K + s_n^2 I)^-1 (y-m) - sum(log diag L) - n/2 log 2 pi,

which trades data fit (first term) against complexity (second term).

Kernels
- tanimoto: k(a, b) = s_f^2 |a & b| / |a | b| on bit vectors (fingerprints, one-hot,
            DRFP). It has no length scale: two reactions sharing most bits are similar.
- rbf:      k(a, b) = s_f^2 exp(-|a - b|^2 / 2l^2) on real vectors (text embeddings).
"""
from __future__ import annotations

import math

import numpy as np
import torch

F64 = dict(dtype=torch.float64)  # Cholesky in float32 fails on near-duplicate rows


def tanimoto_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    inter = A @ B.T
    union = A.sum(1, keepdim=True) + B.sum(1)[None, :] - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-12), torch.zeros_like(inter))


def rbf_kernel(A: torch.Tensor, B: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    d2 = (A * A).sum(1, keepdim=True) + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return torch.exp(-0.5 * d2.clamp_min(0) / lengthscale ** 2)


class GP:
    def __init__(self, kernel: str = "tanimoto", steps: int = 150, lr: float = 0.05,
                 min_noise: float = 1e-4):
        if kernel not in ("tanimoto", "rbf"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.kernel, self.steps, self.lr, self.min_noise = kernel, steps, lr, min_noise

    # -- kernel --------------------------------------------------------------------
    def _k(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Kernel without the signal variance s_f^2."""
        if self.kernel == "tanimoto":
            return tanimoto_kernel(A, B)
        return rbf_kernel(A, B, self.log_l.exp())

    def _params(self):
        return [self.log_sf2, self.log_sn2, self.mean] + ([self.log_l] if self.kernel == "rbf" else [])

    def _noise(self) -> torch.Tensor:
        return self.log_sn2.exp() + self.min_noise

    # -- marginal likelihood ---------------------------------------------------------
    def _nlml(self, K0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log marginal likelihood per point; K0 = kernel without s_f^2."""
        n = len(y)
        K = self.log_sf2.exp() * K0 + self._noise() * torch.eye(n, **F64)
        L = torch.linalg.cholesky(K)
        r = (y - self.mean)[:, None]
        alpha = torch.cholesky_solve(r, L)
        return (0.5 * (r * alpha).sum() + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * math.log(2 * math.pi)) / n

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GP":
        X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        y = np.asarray(y, dtype=np.float64)
        # work on a standardised target so the same initial values suit any data set
        self.y_mu, self.y_sd = float(y.mean()), float(y.std()) + 1e-9
        y = torch.as_tensor((y - self.y_mu) / self.y_sd, **F64)
        self.log_sf2 = torch.zeros((), **F64, requires_grad=True)
        self.log_sn2 = torch.tensor(math.log(0.1), **F64, requires_grad=True)
        self.mean = torch.zeros((), **F64, requires_grad=True)
        if self.kernel == "rbf":
            d = torch.cdist(X[:200], X[:200])
            self.log_l = torch.log(d[d > 0].median()).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam(self._params(), lr=self.lr)
        if self.kernel == "tanimoto":
            # T has no length scale, so it is the same at every step. With T = Q diag(lam) Q^T,
            # K = s_f^2 T + s_n^2 I = Q diag(s_f^2 lam + s_n^2) Q^T: one eigendecomposition,
            # then every step costs O(n) instead of a Cholesky of an n x n matrix.
            lam, Q = torch.linalg.eigh(self._k(X, X))
            lam = lam.clamp_min(0)
            Qy, Q1 = Q.T @ y, Q.T @ torch.ones(len(y), **F64)

        def objective() -> torch.Tensor:
            if self.kernel == "tanimoto":
                d = self.log_sf2.exp() * lam + self._noise()
                r = Qy - self.mean * Q1  # Q^T (y - m)
                return (0.5 * (r * r / d).sum() + 0.5 * torch.log(d).sum()
                        + 0.5 * len(y) * math.log(2 * math.pi)) / len(y)
            return self._nlml(self._k(X, X), y)

        for _ in range(self.steps):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.nlml_ = float(objective())  # at the hyperparameters actually kept
        self._condition(X, y)
        return self

    def _condition(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Cache L and alpha = (K + s_n^2 I)^-1 (y - m) with the fitted hyperparameters."""
        with torch.no_grad():
            self.X, self.y_std = X, y
            K = self.log_sf2.exp() * self._k(X, X) + self._noise() * torch.eye(len(y), **F64)
            self.L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve((y - self.mean)[:, None], self.L)

    # -- posterior -----------------------------------------------------------------
    def predict(self, Xs: np.ndarray, return_std: bool = False):
        """Posterior mean (and std of the latent f, without observation noise), in yield units."""
        Xs = torch.as_tensor(np.asarray(Xs), dtype=torch.float64)
        with torch.no_grad():
            Ks = self.log_sf2.exp() * self._k(self.X, Xs)  # n x m
            mu = (self.mean + Ks.T @ self.alpha[:, 0]) * self.y_sd + self.y_mu
            if not return_std:
                return mu.numpy()
            v = torch.cholesky_solve(Ks, self.L)
            kss = self.log_sf2.exp() * torch.ones(len(Xs), **F64) if self.kernel == "rbf" else \
                self.log_sf2.exp() * (Xs.sum(1) > 0).double()  # k(x, x) = s_f^2
            var = (kss - (Ks * v).sum(0)).clamp_min(1e-12)
            return mu.numpy(), (var.sqrt() * self.y_sd).numpy()

    def sample_pool(self, pool_chol: torch.Tensor, train_idx: np.ndarray, n: int,
                    generator: torch.Generator) -> np.ndarray:
        """n joint posterior samples of f over a whole candidate pool (Matheron's rule).

        Draw a prior sample over the pool, f ~ N(m, s_f^2 T) with T = pool_chol pool_chol^T,
        then move it to agree with the data:
            f_post = f + K_pool,train (K + s_n^2 I)^-1 (y - f[train] - eps).
        Exact, and it needs the pool's Cholesky only once per BO run, not once per round.
        Only for the Tanimoto kernel, whose prior covariance does not change with the fit.
        Returns an array (n, pool) in yield units.
        """
        if self.kernel != "tanimoto":
            raise ValueError("pool sampling needs a kernel without a length scale")
        with torch.no_grad():
            sf = self.log_sf2.exp().sqrt()
            z = torch.randn(pool_chol.shape[0], n, generator=generator, **F64)
            f = self.mean + sf * (pool_chol @ z)  # pool x n prior draws
            eps = self._noise().sqrt() * torch.randn(len(train_idx), n, generator=generator, **F64)
            resid = self.y_std[:, None] - f[train_idx] - eps
            Kpt = self.log_sf2.exp() * pool_chol @ pool_chol[train_idx].T  # pool x n_train
            f_post = f + Kpt @ torch.cholesky_solve(resid, self.L)
            return (f_post.T * self.y_sd + self.y_mu).numpy()
