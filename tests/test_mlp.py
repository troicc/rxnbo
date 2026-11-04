import numpy as np

from rxnbo.models.mlp import MLPRegressor


def test_mlp_learns_a_sum_and_stops_early():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10)).astype(np.float32)
    y = X[:, :3].sum(1) * 5 + 20
    m = MLPRegressor(hidden=(32,), max_epochs=500, patience=10, seed=0).fit(X[:500], y[:500])
    p = m.predict(X[500:])
    r2 = 1 - ((y[500:] - p) ** 2).sum() / ((y[500:] - y[500:].mean()) ** 2).sum()
    assert r2 > 0.95
    assert m.epochs_ < 500  # early stopping ended it


def test_same_seed_same_model():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 5)), rng.normal(size=200)
    a = MLPRegressor(max_epochs=5, seed=7).fit(X, y).predict(X)
    b = MLPRegressor(max_epochs=5, seed=7).fit(X, y).predict(X)
    assert np.allclose(a, b)


def test_a_bit_never_set_in_training_does_not_explode():
    rng = np.random.default_rng(2)
    X = (rng.random((300, 20)) < 0.3).astype(np.float32)
    X[:, 0] = 0  # never set in training
    y = X[:, 1:4].sum(1) * 10
    m = MLPRegressor(hidden=(16,), max_epochs=50, seed=0).fit(X, y)
    Xt = X[:20].copy()
    Xt[:, 0] = 1  # set at test time (an unseen additive's bit)
    assert np.abs(m.predict(Xt) - m.predict(X[:20])).max() < 50
