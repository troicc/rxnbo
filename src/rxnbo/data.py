"""Load the Buchwald-Hartwig HTE data set and split it three ways.

Data: Ahneman et al., Science 2018, 360, 186 - 3955 Pd-catalysed C-N couplings of
15 aryl halides x 4 ligands x 3 bases x 22 isoxazole additives with p-toluidine,
as distributed in rxn4chemistry/rxn_yields (data/raw/Dreher_and_Doyle_input_data.xlsx).

Splits
- random:      a random fraction of reactions for training (interpolation).
- additive:    Test1-4 of the original paper - five additives never seen in training.
- aryl halide: one arene (its Cl, Br and I forms together) never seen in training.

Every split returns integer row positions (train, test) into the frame `load()` gives,
and never puts the same row on both sides (tests/test_data.py checks this).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .filters import amination_product, reaction_smiles

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Dreher_and_Doyle_input_data.xlsx"
PROCESSED = ROOT / "data" / "processed"

ROLES = ("aryl_halide", "ligand", "base", "additive")
_COLUMNS = {"Aryl halide": "aryl_halide", "Ligand": "ligand", "Base": "base",
            "Additive": "additive", "Output": "yield"}


def _test_additives(sheet: pd.DataFrame) -> list[str]:
    """In sheets Test1-4 the rows come in blocks of one additive: the training additives
    first, the five test additives (~900 rows) last. Return those five."""
    adds = sheet["Additive"].tolist()
    first = {a: adds.index(a) for a in set(adds)}
    last = {a: len(adds) - 1 - adds[::-1].index(a) for a in set(adds)}
    tails = [sorted(a for a in first if first[a] >= cut)
             for cut in sorted(set(first.values()) - {0})
             if all(last[a] < cut or first[a] >= cut for a in first)]
    five = [t for t in tails if len(t) == 5]
    if len(five) != 1:
        raise ValueError(f"expected one tail of five additives, found {len(five)}")
    return five[0]


def prepare(raw: Path = RAW, out: Path = PROCESSED) -> pd.DataFrame:
    """Read the Excel file once; write data/processed/bh.csv and additive_splits.json."""
    sheets = pd.read_excel(raw, sheet_name=None)
    df = sheets["FullCV_01"].rename(columns=_COLUMNS)[[*ROLES, "yield"]]
    # FullCV_01 is a shuffled copy; sort for a stable row order independent of the file
    df = df.sort_values(list(ROLES)).reset_index(drop=True)
    df["product"] = [amination_product(a) for a in df.aryl_halide]
    df["rxn_smiles"] = [reaction_smiles(*r) for r in df[list(ROLES)].itertuples(index=False)]
    splits = {f"Test{k}": _test_additives(sheets[f"Test{k}"]) for k in range(1, 5)}
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "bh.csv", index=False)
    (out / "additive_splits.json").write_text(json.dumps(splits, indent=1))
    return df


def load() -> pd.DataFrame:
    path = PROCESSED / "bh.csv"
    if not path.exists():
        return prepare()
    return pd.read_csv(path)


def additive_test_sets() -> dict[str, list[str]]:
    path = PROCESSED / "additive_splits.json"
    if not path.exists():
        prepare()
    return json.loads(path.read_text())


def random_split(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    k = int(round(train_frac * n))
    return np.sort(order[:k]), np.sort(order[k:])


def group_split(groups: pd.Series, test_groups) -> tuple[np.ndarray, np.ndarray]:
    """Every row whose group is in `test_groups` goes to test, the rest to train."""
    test = groups.isin(set(test_groups)).to_numpy()
    if not test.any() or test.all():
        raise ValueError("a group split needs rows on both sides")
    return np.flatnonzero(~test), np.flatnonzero(test)


def additive_split(df: pd.DataFrame, test: int) -> tuple[np.ndarray, np.ndarray]:
    """Test1-4 of Ahneman et al.: five additives held out."""
    return group_split(df["additive"], additive_test_sets()[f"Test{test}"])


def arene(aryl_halide: str) -> str:
    """The aryl halide with its halogen taken off - groups the Cl, Br and I forms."""
    return amination_product(aryl_halide)


def aryl_halide_split(df: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Leave one arene out (fold 0-4): the model has never seen that ring system."""
    arenes = df["aryl_halide"].map(arene)
    names = sorted(arenes.unique())
    return group_split(arenes, [names[fold]])


def split(df: pd.DataFrame, kind: str, k: int, train_frac: float = 0.7):
    """One entry point: kind = 'random' (k = seed), 'additive' (k = 1..4), 'arene' (k = 0..4)."""
    if kind == "random":
        return random_split(len(df), train_frac, k)
    if kind == "additive":
        return additive_split(df, k)
    if kind == "arene":
        return aryl_halide_split(df, k)
    raise ValueError(f"unknown split kind {kind!r}")
