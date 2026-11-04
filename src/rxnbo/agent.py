"""A small tool-using agent (the idea of ChemCrow, at toy size) and a test of it.

The agent is a loop: send the conversation and the tool list to the model; if it answers
with tool calls, run them (here: RDKit functions) and send the results back; stop when it
answers in text. The loop is written out by hand so every step can be seen.

The test: 30 questions about molecules from the two data sets whose exact answers RDKit
computes (molecular weight, formula, H-bond donors, TPSA, rings, Tanimoto similarity,
substructure). Each question is asked twice, with and without tools. Without tools the
model must do the arithmetic in its head; the gap measures what tools buy.
"""
from __future__ import annotations

import json
import random
import re
import time

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .evaluate import RESULTS

# -- tools: plain functions, and the JSON schemas the model sees --------------------------------
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot read the SMILES {smiles!r}")
    return mol


def describe(smiles: str) -> dict:
    """Formula, weight and simple counts of one molecule."""
    m = _mol(smiles)
    return {"canonical_smiles": Chem.MolToSmiles(m), "formula": rdMolDescriptors.CalcMolFormula(m),
            "mol_weight": round(Descriptors.MolWt(m), 3), "h_bond_donors": rdMolDescriptors.CalcNumHBD(m),
            "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(m), "tpsa": round(rdMolDescriptors.CalcTPSA(m), 2),
            "rings": rdMolDescriptors.CalcNumRings(m), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
            "heavy_atoms": m.GetNumHeavyAtoms()}


def similarity(smiles_a: str, smiles_b: str) -> dict:
    """Tanimoto similarity of Morgan fingerprints (radius 2, 2048 bits)."""
    fa, fb = (_morgan.GetFingerprint(_mol(s)) for s in (smiles_a, smiles_b))
    return {"tanimoto": round(DataStructs.TanimotoSimilarity(fa, fb), 3)}


def substructure(smiles: str, smarts: str) -> dict:
    """Whether the molecule contains the SMARTS pattern, and how many times."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError(f"unreadable SMARTS {smarts!r}")
    n = len(_mol(smiles).GetSubstructMatches(patt))
    return {"matches": n, "contains": n > 0}


TOOLS = {"describe": describe, "similarity": similarity, "substructure": substructure}
SCHEMAS = [
    {"name": "describe", "description": describe.__doc__ + " Exact values computed by RDKit.",
     "input_schema": {"type": "object", "properties": {"smiles": {"type": "string"}},
                      "required": ["smiles"], "additionalProperties": False}, "strict": True},
    {"name": "similarity", "description": similarity.__doc__,
     "input_schema": {"type": "object", "properties": {"smiles_a": {"type": "string"}, "smiles_b": {"type": "string"}},
                      "required": ["smiles_a", "smiles_b"], "additionalProperties": False}, "strict": True},
    {"name": "substructure", "description": substructure.__doc__,
     "input_schema": {"type": "object", "properties": {"smiles": {"type": "string"}, "smarts": {"type": "string"}},
                      "required": ["smiles", "smarts"], "additionalProperties": False}, "strict": True},
]

SYSTEM = ("You answer questions about molecules given as SMILES. End your reply with one line "
          "'ANSWER: <value>' holding only the value (a number, a formula, or yes/no).")


def run_agent(client, model: str, question: str, use_tools: bool, max_turns: int = 8,
              effort: str = "low") -> dict:
    """The agent loop. Returns the final text, the tool calls made and the tokens used."""
    messages = [{"role": "user", "content": question}]
    calls, used = [], {"in": 0, "out": 0}
    for _ in range(max_turns):
        kw = {"tools": SCHEMAS} if use_tools else {}
        resp = client.messages.create(model=model, max_tokens=4000, system=SYSTEM, messages=messages,
                                      output_config={"effort": effort}, **kw)
        used["in"] += resp.usage.input_tokens
        used["out"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return {"text": "", "refused": True, "calls": calls, "tokens": used}
        if resp.stop_reason != "tool_use":  # the model answered in words: done
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"text": text, "calls": calls, "tokens": used}
        messages.append({"role": "assistant", "content": resp.content})  # keep its turn as sent
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out, err = TOOLS[block.name](**block.input), False
            except Exception as e:  # a bad SMILES is reported back to the model, not raised
                out, err = {"error": str(e)}, True
            calls.append({"tool": block.name, "input": block.input, "output": out})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out), "is_error": err})
        messages.append({"role": "user", "content": results})  # all results in one turn
    return {"text": "", "calls": calls, "tokens": used, "out_of_turns": True}


# -- the questions, with answers computed by RDKit ---------------------------------------------
def tasks(n: int = 30, seed: int = 0) -> list[dict]:
    from . import data
    df = data.load()
    mols = sorted(set(df.ligand) | set(df.additive) | set(df["product"]) | set(df.base))
    rng = random.Random(seed)
    kinds = ["mol_weight", "formula", "h_bond_donors", "tpsa", "rings", "similarity", "substructure"]
    smarts = {"an ester": "[CX3](=O)[OX2][#6]", "a P-C bond": "[P][#6]", "an isoxazole ring": "c1cnoc1",
              "a pyridine ring": "n1ccccc1", "a secondary aryl amine (Ar-NH-Ar)": "c[NH]c"}
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        s = rng.choice(mols)
        if kind == "similarity":
            t = rng.choice([m for m in mols if m != s])
            q, a, tol = (f"What is the Tanimoto similarity of the Morgan fingerprints (radius 2, 2048 bits) "
                         f"of {s} and {t}? Give three decimals."), similarity(s, t)["tanimoto"], 0.01
        elif kind == "substructure":
            name, patt = rng.choice(list(smarts.items()))
            q, a, tol = f"Does {s} contain {name}? Answer yes or no.", substructure(s, patt)["contains"], None
        else:
            d = describe(s)
            label = {"mol_weight": "the molecular weight (g/mol, average masses, two decimals)",
                     "formula": "the molecular formula (Hill order)", "h_bond_donors": "the number of H-bond donors",
                     "tpsa": "the topological polar surface area (TPSA, in Å²)",
                     "rings": "the number of rings (SSSR)"}[kind]
            q, a = f"What is {label} of {s}?", d[kind]
            tol = {"mol_weight": 0.1, "tpsa": 0.5}.get(kind, 0)
        out.append({"id": i, "kind": kind, "question": q, "answer": a, "tol": tol})
    return out


def check(task: dict, text: str) -> bool:
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if not m:
        return False
    got = m[-1].strip().rstrip(".")
    want = task["answer"]
    if isinstance(want, bool):
        return got.lower().startswith("yes") == want and got.lower()[:2] in ("ye", "no")
    if isinstance(want, str):
        return got.replace(" ", "") == want
    num = re.search(r"-?\d+(\.\d+)?", got)
    return num is not None and abs(float(num.group()) - want) <= task["tol"]


ARMS = (("haiku-5.5", "claude-haiku-5-5"), ("opus-5.5", "claude-opus-5-5"))


def main(model: str | None = None, n: int = 30, log=print) -> pd.DataFrame:
    import anthropic
    client = anthropic.Anthropic()
    ts = tasks(n)
    rows = []
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "agent_runs.jsonl", "w") as fh:
        for label, m in ARMS:
            if model and m != model:
                continue
            for use_tools in (False, True):
                t0 = time.time()
                for t in ts:
                    r = run_agent(client, m, t["question"], use_tools)
                    ok = check(t, r["text"])
                    rows.append(dict(model=label, tools=use_tools, kind=t["kind"], correct=ok,
                                     n_calls=len(r["calls"]), tokens_in=r["tokens"]["in"],
                                     tokens_out=r["tokens"]["out"]))
                    fh.write(json.dumps({"model": label, "tools": use_tools, "task": t, "run": r},
                                        default=str) + "\n")
                acc = pd.DataFrame(rows[-len(ts):]).correct.mean()
                log(f"agent {label} tools={use_tools}: {acc:.2f} ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "table3_agent_runs.csv", index=False)
    tab = res.pivot_table(index="kind", columns=["model", "tools"], values="correct", aggfunc="mean")
    tab.loc["all"] = res.groupby(["model", "tools"]).correct.mean()
    tab.round(3).to_csv(RESULTS / "table3_agent.csv")
    return tab
