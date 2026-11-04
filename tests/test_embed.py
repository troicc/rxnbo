from rdkit import Chem

from rxnbo import embed


def has(smiles, smarts):
    return Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(smarts))


ADAMANTYL = "C1C2CC3CC1CC(C2)C3"
CYCLOHEXYL_P = "P([CH1;R1]1CCCCC1)[CH1;R1]1CCCCC1"  # ring C bearing H: not adamantyl
TBU2_P = "P([C;!R](C)(C)C)[C;!R](C)(C)C"  # chain tert-butyl: not the adamantyl bridgehead
DIMETHOXY = "c(OC)ccc(OC)"


def test_ligand_names_match_structures():
    for smi, name in embed.LIGANDS.items():
        assert has(smi, ADAMANTYL) == name.startswith("Ad")
        assert has(smi, CYCLOHEXYL_P) == (name == "XPhos")
        assert has(smi, TBU2_P) == name.startswith("tBu")
        assert has(smi, DIMETHOXY) == ("Brett" in name)


def test_base_names_match_structures():
    for smi, name in embed.BASES.items():
        assert has(smi, "[P]") == name.startswith("P2Et")
        assert has(smi, "C(C)(C)(C)N=C") == name.startswith("BTMG")
        assert (Chem.MolFromSmiles(smi).GetRingInfo().NumRings() == 2) == name.startswith("MTBD")


def test_hand_named_additives_are_the_right_isomers():
    for smi, name in embed.HAND.items():
        five = has(smi, "N(Cc1ccccc1)(Cc1ccccc1)c1ccno1")  # amine on C5, next to ring O
        assert five == name.endswith("5-amine")
