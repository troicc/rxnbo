import numpy as np
from scipy.stats import norm

from rxnbo import bo


def test_acquisition_values_by_hand():
    # at mu = best with xi = 0, EI = sd * pdf(0) and PI = 1/2
    assert np.isclose(bo.expected_improvement(5.0, 2.0, 5.0, xi=0), 2 * norm.pdf(0))
    assert np.isclose(bo.probability_of_improvement(5.0, 2.0, 5.0, xi=0), 0.5)
    assert np.isclose(bo.upper_confidence_bound(1.0, 0.5, beta=4), 2.0)


def test_ei_grows_with_uncertainty_and_is_never_negative():
    mu = np.linspace(-3, 3, 50)
    for sd in (0.1, 1, 3):
        assert (bo.expected_improvement(mu, sd, 0.0) >= 0).all()
    assert bo.expected_improvement(0, 2.0, 0.0) > bo.expected_improvement(0, 1.0, 0.0)


def _toy():
    rng = np.random.default_rng(0)
    X = (rng.random((300, 40)) < 0.25).astype(np.uint8)
    y = X[:, :5].sum(1) * 10 + rng.normal(0, 1, 300)  # yield depends on five bits
    return X, y


def test_runs_pick_each_reaction_once_from_the_same_start():
    X, y = _toy()
    runs = {s: bo.run(X, y, s, seed=3, n_init=8, batch=4, rounds=3, gp_steps=20)
            for s in ("random", "ei", "ts")}
    for r in runs.values():
        assert len(r.picked) == 8 + 4 * 3 == len(set(r.picked))
    starts = {tuple(r.picked[:8]) for r in runs.values()}
    assert len(starts) == 1


def test_bo_beats_random_on_a_learnable_toy():
    X, y = _toy()
    gain = [bo.best_so_far(bo.run(X, y, "ei", s, n_init=5, batch=3, rounds=6, gp_steps=30), y)[-1]
            - bo.best_so_far(bo.run(X, y, "random", s, n_init=5, batch=3, rounds=6), y)[-1]
            for s in range(5)]
    assert np.mean(gain) > 0


def test_metrics():
    y = np.arange(100.0)
    r = bo.Run("x", 0, [5, 50, 99, 98])
    assert bo.experiments_to_top(r, y, 0.01) == 3
    assert bo.top_coverage(r, y, 0.05) == 2 / 5
    assert list(bo.best_so_far(r, y)) == [5, 50, 99, 99]
