from rdkit import Chem

from rxnbo import filters


def same(a, b):
    return Chem.CanonSmiles(a) == Chem.CanonSmiles(b)


def test_amination_products_written_by_hand():
    assert same(filters.amination_product("ClC1=NC=CC=C1"), "Cc1ccc(Nc2ccccn2)cc1")
    assert same(filters.amination_product("IC1=CC=C(OC)C=C1"), "COc1ccc(Nc2ccc(C)cc2)cc1")
    assert same(filters.amination_product("BrC1=CN=CC=C1"), "Cc1ccc(Nc2cccnc2)cc1")


def test_no_or_two_halides_is_an_error():
    import pytest
    with pytest.raises(ValueError):
        filters.amination_product("c1ccccc1")
    with pytest.raises(ValueError):
        filters.amination_product("Clc1ccc(Br)cc1")


def test_reaction_smiles_has_three_parts():
    r = filters.reaction_smiles("ClC1=NC=CC=C1", "C", "N", "O")
    reactants, reagents, product = r.split(">")
    assert "Cc1ccc(N)cc1" in reactants and reagents.count(".") == 3 and "Nc2ccccn2" in product


def test_pains_and_scaffold():
    assert filters.pains_alerts("O=C1C(=Cc2ccccc2)SC(=S)N1")  # an ene-rhodanine
    assert filters.pains_alerts("CCO") == []
    assert same(filters.scaffold("CCc1ccc(CCN)cc1"), "c1ccccc1")
    assert same(filters.scaffold("Cc1ccc(Nc2ccccn2)cc1"), "c1ccc(Nc2ccccn2)cc1")


def test_substructure():
    assert filters.has_substructure("Cc1ccc(N)cc1", "c[NH2]")
    assert not filters.has_substructure("c1ccccc1", "[#7]")
