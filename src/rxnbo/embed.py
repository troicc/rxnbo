"""Reactions as sentences, sentences as vectors (the idea of BoChemian / GOLLuM).

A reaction is written as an English sentence naming its components, and an open text
embedding model turns the sentence into a vector. The GP then uses an RBF kernel on these
vectors (they are real-valued, not bits, so Tanimoto does not apply).

Names: the ligands and bases are the four Buchwald ligands and three organic bases of
Ahneman et al., identified by structure (tests/test_embed.py checks each SMILES has the
group its name says). Aryl halides and additives get their IUPAC names from PubChem once;
the answers are kept in data/names.json so the experiment runs offline afterwards.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = Path(__file__).resolve().parents[2] / "data" / "names.json"
MODEL = "BAAI/bge-small-en-v1.5"

LIGANDS = {  # dialkylbiaryl phosphines: P substituents, and whether the top ring has two OMe
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "tBuXPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "tBuBrettPhos",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H]"
    "(C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC": "AdBrettPhos",
}
BASES = {
    "CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG (2-tert-butyl-1,1,3,3-tetramethylguanidine)",
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et (phosphazene base)",
    "CN1CCCN2C1=NCCC2": "MTBD (7-methyl-1,5,7-triazabicyclo[4.4.0]dec-5-ene)",
}

# not in PubChem: named by hand from the ring position of the amine (O1-N2-C3-C4-C5)
HAND = {
    "C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=CC=NO1": "N,N-dibenzyl-1,2-oxazol-5-amine",
    "C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=NOC=C1": "N,N-dibenzyl-1,2-oxazol-3-amine",
}


def _pubchem_name(smiles: str) -> str:
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
           f"{urllib.parse.quote(smiles, safe='')}/property/IUPACName,Title/JSON")
    with urllib.request.urlopen(url, timeout=30) as r:
        props = json.load(r)["PropertyTable"]["Properties"][0]
    # a compound PubChem does not hold comes back with a CID but no names
    return props.get("IUPACName") or props.get("Title") or f"the compound {smiles}"


def names(df: pd.DataFrame) -> dict[str, str]:
    known = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    missing = sorted((set(df.aryl_halide) | set(df.additive)) - set(known) - set(HAND))
    for s in missing:
        known[s] = _pubchem_name(s)
    if missing:
        NAMES.write_text(json.dumps(known, indent=1, sort_keys=True))
    return {**known, **HAND, **LIGANDS, **BASES}


def describe(row, nm: dict[str, str]) -> str:
    return (f"Buchwald-Hartwig amination of {nm[row.aryl_halide]} with p-toluidine, "
            f"palladium catalyst with the ligand {nm[row.ligand]}, the base {nm[row.base]}, "
            f"and the additive {nm[row.additive]}.")


def sentences(df: pd.DataFrame) -> list[str]:
    nm = names(df)
    return [describe(r, nm) for r in df.itertuples(index=False)]


def embed_reactions(df: pd.DataFrame, model: str = MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model, device="cpu")
    return np.asarray(enc.encode(sentences(df), batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False), dtype=np.float32)
