import numpy as np

from rxnbo import data, featurize as F


def test_tanimoto_by_hand():
    a, b, c = [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]
    T = F.tanimoto(np.array([a, b]), np.array([a, b, c]))
    assert np.allclose(T, [[1, 1 / 3, 0], [1 / 3, 1, 0]])


def test_standardize_removes_salt_and_charge():
    assert F.standardize("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert F.standardize("OC(=O)c1ccccc1") == F.standardize("c1ccccc1C(O)=O")


def test_morgan_radius_zero_counts_atom_types():
    # ethanol at radius 0: three different atom environments (CH3, CH2, OH)
    assert F.morgan("CCO", radius=0, n_bits=2048).sum() == 3
    assert F.morgan("CCO", radius=2, n_bits=2048).sum() > 3


def test_shapes_and_onehot_rows():
    df = data.load().head(50)
    oh = F.onehot(data.load())
    assert oh.shape == (3955, 44) and (oh.sum(1) == 4).all()
    assert F.reaction_morgan(df, n_bits=256).shape == (50, 4 * 256)
    assert F.descriptors(df).shape == (50, 4 * len(F.DESCRIPTORS))
