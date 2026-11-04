"""Batch Bayesian optimisation over a fixed pool of candidate reactions.

The question it answers: with a budget of 110 experiments out of 3955 possible ones, how
quickly does each strategy find a high-yield reaction?

One run:
    1. pick n_init reactions at random and "run" them (look up their yields);
    2. fit a GP to every yield seen so far;
    3. score every untried reaction with an acquisition function and pick a batch;
    4. repeat 2-3 for `rounds` rounds.

Acquisition functions (mu, sd = GP posterior mean and std of a candidate, y* = best so far):
- EI  (expected improvement)       E[max(f - y* - xi, 0)]  = (mu-y*-xi) Phi(z) + sd phi(z)
- PI  (probability of improvement) P[f > y* + xi]          = Phi(z),  z = (mu-y*-xi)/sd
- UCB (upper confidence bound)     mu + sqrt(beta) sd
- TS  (Thompson sampling)          draw one function from the posterior, take its argmax
EI, PI and UCB pick a batch by the kriging believer: after each pick, pretend its yield is
the posterior mean, update the GP, and pick again - so the batch spreads out instead of
choosing five near-copies. Thompson sampling spreads by itself: five independent draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import norm

from .models.gp import GP


def expected_improvement(mu, sd, best, xi=0.01):
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def probability_of_improvement(mu, sd, best, xi=0.01):
    return norm.cdf((mu - best - xi) / sd)


def upper_confidence_bound(mu, sd, beta=4.0):
    return mu + np.sqrt(beta) * sd


ACQUISITIONS = {"ei": expected_improvement, "pi": probability_of_improvement,
                "ucb": upper_confidence_bound}
STRATEGIES = ("random", "greedy", "ei", "pi", "ucb", "ts")


@dataclass
class Run:
    strategy: str
    seed: int
    picked: list[int] = field(default_factory=list)  # pool indices in the order tried

    def yields(self, y: np.ndarray) -> np.ndarray:
        return y[np.array(self.picked)]


def pool_cholesky(X: np.ndarray, jitter: float = 1e-6) -> torch.Tensor:
    """Cholesky factor of the Tanimoto matrix over the whole pool (for Thompson sampling)."""
    from .models.gp import tanimoto_kernel
    Xt = torch.as_tensor(X, dtype=torch.float64)
    T = tanimoto_kernel(Xt, Xt)
    eye = torch.eye(len(X), dtype=torch.float64)
    while True:  # raise the jitter until the factorisation succeeds (T can be near-singular)
        L, info = torch.linalg.cholesky_ex(T + jitter * eye)
        if info == 0:
            return L
        jitter *= 10


def run(X: np.ndarray, y: np.ndarray, strategy: str, seed: int, n_init: int = 10,
        batch: int = 5, rounds: int = 20, kernel: str = "tanimoto", gp_steps: int = 100,
        chol: torch.Tensor | None = None) -> Run:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    n = len(X)
    # the same random start for every strategy with this seed: a fair comparison
    out = Run(strategy, seed, list(np.random.default_rng(10_000 + seed).choice(n, n_init, replace=False)))
    for _ in range(rounds):
        tried = np.array(out.picked)
        untried = np.setdiff1d(np.arange(n), tried)
        if strategy == "random":
            out.picked += list(rng.choice(untried, batch, replace=False))
            continue
        gp = GP(kernel, steps=gp_steps).fit(X[tried], y[tried])
        if strategy == "ts":
            if chol is None:
                chol = pool_cholesky(X)
            draws = gp.sample_pool(chol, tried, batch, gen)  # batch x pool
            draws[:, tried] = -np.inf
            chosen: list[int] = []
            for d in draws:
                d[chosen] = -np.inf
                chosen.append(int(np.argmax(d)))
            out.picked += chosen
            continue
        out.picked += _believer_batch(gp, X, y[tried], tried, untried, strategy, batch)
    return out


def _believer_batch(gp: GP, X, y_tried, tried, untried, strategy, batch) -> list[int]:
    """Pick `batch` candidates one at a time, conditioning on the mean as a fake result."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    idx, ystd = list(tried), list((y_tried - gp.y_mu) / gp.y_sd)
    best = float(np.max(y_tried))
    pool = untried.copy()
    chosen = []
    for _ in range(batch):
        mu, sd = gp.predict(X[pool], return_std=True)
        if strategy == "greedy":
            score = mu
        else:
            score = ACQUISITIONS[strategy](mu, sd, best)
        k = int(np.argmax(score))
        chosen.append(int(pool[k]))
        idx.append(int(pool[k]))
        ystd.append((mu[k] - gp.y_mu) / gp.y_sd)  # believe the mean
        pool = np.delete(pool, k)
        gp._condition(Xt[idx], torch.as_tensor(ystd, dtype=torch.float64))
    return chosen


# -- what a run achieved ------------------------------------------------------------------
def best_so_far(run: Run, y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(run.yields(y))


def experiments_to_top(run: Run, y: np.ndarray, q: float = 0.01) -> float:
    """How many experiments until one in the top q of the pool was found (inf if never)."""
    threshold = np.quantile(y, 1 - q)
    hits = np.flatnonzero(run.yields(y) >= threshold)
    return float(hits[0] + 1) if len(hits) else float("inf")


def top_coverage(run: Run, y: np.ndarray, q: float = 0.05) -> float:
    """Share of the pool's top-q reactions that the run tried."""
    top = set(np.flatnonzero(y >= np.quantile(y, 1 - q)))
    return len(top & set(run.picked)) / len(top)
