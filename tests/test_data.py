import numpy as np
import pytest

from rxnbo import data


@pytest.fixture(scope="module")
def df():
    return data.load()


def test_dataset_shape(df):
    assert len(df) == 3955
    assert [df[r].nunique() for r in data.ROLES] == [15, 4, 3, 22]
    assert df["yield"].between(0, 100).all()


def _check(df, tr, te):
    assert len(np.intersect1d(tr, te)) == 0
    assert len(tr) + len(te) == len(df)


@pytest.mark.parametrize("seed", [0, 1])
def test_random_split(df, seed):
    tr, te = data.split(df, "random", seed)
    _check(df, tr, te)
    assert abs(len(tr) / len(df) - 0.7) < 0.001


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_additive_split_never_shows_a_test_additive(df, k):
    tr, te = data.split(df, "additive", k)
    _check(df, tr, te)
    assert set(df.additive.iloc[tr]).isdisjoint(df.additive.iloc[te])
    assert df.additive.iloc[te].nunique() == 5
    assert 850 < len(te) < 950


def test_the_four_additive_tests_are_different(df):
    sets = data.additive_test_sets()
    assert len({tuple(v) for v in sets.values()}) == 4


@pytest.mark.parametrize("k", range(5))
def test_arene_split_holds_out_all_three_halides(df, k):
    tr, te = data.split(df, "arene", k)
    _check(df, tr, te)
    held = df.aryl_halide.iloc[te].unique()
    assert len(held) == 3  # the Cl, Br and I forms of one arene
    assert {h[:1] for h in held} == {"C", "B", "I"}  # Cl..., Br..., I...
    assert set(held).isdisjoint(df.aryl_halide.iloc[tr])
