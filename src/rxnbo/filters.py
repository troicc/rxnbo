"""Reaction SMARTS, substructure checks, PAINS and Bemis-Murcko scaffolds.

The Buchwald-Hartwig data set lists the four varied components (aryl halide, ligand,
base, additive) but not the product. The coupling partner is always p-toluidine, so the
product is made here by a reaction SMARTS: replace the aryl C-X bond by C-N(H)-tolyl.
"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

AMINE = "Cc1ccc(N)cc1"  # p-toluidine, the nucleophile in every reaction
# Pd precatalyst as written in rxn4chemistry/rxn_yields (a 2-aminobiphenyl palladacycle triflate)
PD_PRECATALYST = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"

# An aromatic carbon bonded to Cl, Br or I becomes bonded to the toluidine nitrogen.
# Atom map :1 keeps the ring carbon; the halogen (unmapped) is dropped.
AMINATION_SMARTS = "[c:1][Cl,Br,I]>>[c:1][NH]c1ccc(C)cc1"


@lru_cache(maxsize=1)
def _amination():
    rxn = AllChem.ReactionFromSmarts(AMINATION_SMARTS)
    rxn.Initialize()
    return rxn


def amination_product(aryl_halide: str) -> str:
    """Canonical SMILES of the C-N coupling product of `aryl_halide` with p-toluidine.

    Raises ValueError if the molecule has no aryl halide, or more than one
    (the product would be ambiguous).
    """
    mol = Chem.MolFromSmiles(aryl_halide)
    if mol is None:
        raise ValueError(f"unreadable SMILES: {aryl_halide}")
    rxn = _amination()
    # count the sites, not the distinct products: in a symmetric dihalide (p-dibromobenzene) both sites give the
    # same product, and a set of products would let it through as if it had one
    sites = mol.GetSubstructMatches(rxn.GetReactantTemplate(0))
    if len(sites) != 1:
        raise ValueError(f"{aryl_halide}: expected one aryl halide site, found {len(sites)}")
    products = [p[0] for p in rxn.RunReactants((mol,))
                if Chem.SanitizeMol(p[0], catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE]
    if len(products) != 1:
        raise ValueError(f"{aryl_halide}: the coupling product does not sanitise")
    return Chem.MolToSmiles(products[0])


def reaction_smiles(aryl_halide: str, ligand: str, base: str, additive: str) -> str:
    """reactants>reagents>product, the form DRFP and reaction transformers read."""
    reactants = f"{aryl_halide}.{AMINE}"
    reagents = ".".join([ligand, base, additive, PD_PRECATALYST])
    return f"{reactants}>{reagents}>{amination_product(aryl_halide)}"


def has_substructure(smiles: str, smarts: str) -> bool:
    mol, patt = Chem.MolFromSmiles(smiles), Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        raise ValueError("unreadable SMILES or SMARTS")
    return mol.HasSubstructMatch(patt)


@lru_cache(maxsize=1)
def _pains() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def pains_alerts(smiles: str) -> list[str]:
    """Names of the PAINS patterns (Baell & Holloway 2010) the molecule matches."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unreadable SMILES: {smiles}")
    return [m.GetDescription() for m in _pains().GetMatches(mol)]


def scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold: ring systems plus the linkers between them."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unreadable SMILES: {smiles}")
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
