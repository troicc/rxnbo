import numpy as np
import torch

from rxnbo.models.gp import GP, tanimoto_kernel


def bits(n, d, seed):
    return (np.random.default_rng(seed).random((n, d)) < 0.2).astype(np.uint8)


def test_eigen_likelihood_equals_cholesky_likelihood():
    X, y = bits(60, 64, 0), np.random.default_rng(1).normal(size=60)
    g = GP("tanimoto", steps=30).fit(X, y)
    ys = torch.as_tensor((y - g.y_mu) / g.y_sd)
    chol = g._nlml(tanimoto_kernel(*(torch.as_tensor(X, dtype=torch.float64),) * 2), ys).item()
    assert abs(chol - g.nlml_) < 1e-3  # nlml_ was computed by the eigendecomposition


def test_posterior_mean_against_numpy():
    X, y = bits(40, 32, 2), np.random.default_rng(3).normal(size=40) * 10 + 50
    g = GP("tanimoto", steps=20).fit(X, y)
    Xs = bits(7, 32, 4)
    sf2, sn2, m = g.log_sf2.exp().item(), g._noise().item(), g.mean.item()
    T = F_tan(X, X); Ts = F_tan(Xs, X)
    ys = (y - y.mean()) / y.std()
    mu = (m + sf2 * Ts @ np.linalg.solve(sf2 * T + sn2 * np.eye(40), ys - m)) * y.std() + y.mean()
    assert np.allclose(g.predict(Xs), mu, atol=1e-6)


def F_tan(A, B):  # an independent Tanimoto: count shared and total bits with sets
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        sa = set(np.flatnonzero(a))
        for j, b in enumerate(B):
            sb = set(np.flatnonzero(b))
            out[i, j] = len(sa & sb) / len(sa | sb) if sa | sb else 0
    return out


def test_pool_samples_match_the_posterior():
    pool = bits(80, 48, 5)
    y_all = np.random.default_rng(6).normal(size=80)
    tr = np.arange(0, 80, 4)
    g = GP("tanimoto", steps=40).fit(pool[tr], y_all[tr])
    from rxnbo.bo import pool_cholesky
    S = g.sample_pool(pool_cholesky(pool), tr, 4000, torch.Generator().manual_seed(0))
    mu, sd = g.predict(pool, return_std=True)
    assert np.abs(S.mean(0) - mu).max() < 0.1 * y_all.std() + 0.05
    assert np.abs(S.std(0) - sd).max() < 0.1 * y_all.std() + 0.05


def test_rbf_fits_a_smooth_function():
    x = np.linspace(0, 6, 40)[:, None]
    g = GP("rbf", steps=200).fit(x, np.sin(x[:, 0]))
    xs = np.linspace(0.2, 5.8, 9)[:, None]
    assert np.abs(g.predict(xs) - np.sin(xs[:, 0])).max() < 0.05
