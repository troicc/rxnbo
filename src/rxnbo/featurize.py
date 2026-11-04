"""Turn a reaction (four components) into a vector.

Four representations, compared in Figure 3:
- one-hot:      which of the 15/4/3/22 choices each component is (44 bits). Knows nothing
                about chemistry; cannot say anything about an additive it has not seen.
- Morgan:       a circular fingerprint of each component, concatenated by role.
- descriptors:  a handful of RDKit 2D descriptors per component (real numbers).
- DRFP:         a fingerprint of the reaction as a whole: the symmetric difference of the
                substructures of reactants and products (Probst et al. 2022).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from .data import ROLES

RDLogger.DisableLog("rdApp.*")


def standardize(smiles: str) -> str:
    """Canonical SMILES after sanitising, keeping the largest fragment and neutralising.

    The same molecule written two ways must give one string, or a fingerprint or a
    one-hot column would treat it as two molecules.
    """
    mol = Chem.MolFromSmiles(smiles)  # MolFromSmiles sanitises (valence, aromaticity)
    if mol is None:
        raise ValueError(f"unreadable SMILES: {smiles}")
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)  # drop counter-ions, salts
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=None)
def morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan (ECFP-like) bit vector. Round 0 hashes each atom alone; round r hashes the
    atom together with everything within r bonds, so radius 2 = ECFP4 (diameter 4)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(np.uint8)
    fp.setflags(write=False)  # cached: callers must not change it
    return fp


def reaction_morgan(df: pd.DataFrame, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Morgan bits of each component, side by side (role-aware: a bit set by the base
    and the same bit set by the additive land in different columns)."""
    return np.hstack([np.stack([morgan(s, radius, n_bits) for s in df[r]]) for r in ROLES])


def onehot(df: pd.DataFrame) -> np.ndarray:
    blocks = [pd.get_dummies(df[r]).sort_index(axis=1).to_numpy(np.uint8) for r in ROLES]
    return np.hstack(blocks)


DESCRIPTORS = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
               "NumRotatableBonds", "FractionCSP3", "RingCount", "NumAromaticRings",
               "LabuteASA", "MaxPartialCharge", "MinPartialCharge")


@lru_cache(maxsize=None)
def _descriptors(smiles: str) -> tuple[float, ...]:
    mol = Chem.MolFromSmiles(smiles)
    fns = dict(Descriptors.descList)
    return tuple(float(fns[name](mol)) for name in DESCRIPTORS)


def descriptors(df: pd.DataFrame) -> np.ndarray:
    """Raw descriptor values; scale them inside the model (on training rows only)."""
    X = np.hstack([np.array([_descriptors(s) for s in df[r]]) for r in ROLES])
    return np.nan_to_num(X)


def drfp(df: pd.DataFrame, n_bits: int = 2048, radius: int = 3) -> np.ndarray:
    from drfp import DrfpEncoder
    # DRFP reads reactants>>products; the reagents (ligand, base, additive, Pd) go with the
    # reactants, otherwise every reaction of one aryl halide would get the same fingerprint.
    rxns = [f"{r.split('>')[0]}.{r.split('>')[1]}>>{r.split('>')[2]}" for r in df["rxn_smiles"]]
    fps = DrfpEncoder.encode(rxns, n_folded_length=n_bits, radius=radius)
    return np.asarray(fps, dtype=np.uint8)


CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


def featurize(df: pd.DataFrame, kind: str, cache: bool = True, **kw) -> np.ndarray:
    """One entry point for the CLI and the experiments. The slow ones (DRFP, embeddings)
    are kept in data/cache, keyed by representation, settings and the reactions themselves."""
    if cache and kind in ("drfp", "embedding"):
        key = hashlib.sha1(("|".join(df["rxn_smiles"]) + repr(sorted(kw.items()))).encode()).hexdigest()[:12]
        path = CACHE / f"{kind}-{key}.npy"
        if path.exists():
            return np.load(path)
        X = featurize(df, kind, cache=False, **kw)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, X)
        return X
    if kind == "onehot":
        return onehot(df)
    if kind == "morgan":
        return reaction_morgan(df, **kw)
    if kind == "descriptors":
        return descriptors(df)
    if kind == "drfp":
        return drfp(df, **kw)
    if kind == "embedding":
        from .embed import embed_reactions
        return embed_reactions(df, **kw)
    raise ValueError(f"unknown representation {kind!r}")


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity |a & b| / |a | b| of two sets of bit vectors."""
    A, B = A.astype(np.float64), B.astype(np.float64)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
