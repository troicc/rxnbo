"""Can an LLM read a lab-notebook procedure and fill in the reaction's conditions?

Data: 750 Buchwald-Hartwig aminations from AstraZeneca's electronic lab notebooks, in the
Open Reaction Database (ord_dataset-00005539a1e04c809a9a78647bea649c, data/raw/
ord_az_buchwald_hartwig.parquet). Each record has the chemist's free-text procedure AND the
structured fields the chemist entered (aryl halide, amine, base, Pd source, ligand, solvent,
temperature). The structured fields are the answer key; the model sees only the text.

Scoring (per field):
- compounds are compared by structure, not spelling: the set of InChIKey connectivity
  blocks of the fragments (so Cs2CO3 written any way matches, E/Z of dba does not matter);
- temperature within 5 degC;
- `correct` = matches the key; `wrong` = a confident answer that does not match (the
  hallucination-like error that hurts most); `missed` = null where the key has a value;
- `invalid` = a SMILES RDKit cannot read.
Every model answer is kept in results/extract_answers.jsonl, so a score can be checked by eye.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger

from .evaluate import RESULTS

RDLogger.DisableLog("rdApp.*")
ORD = Path(__file__).resolve().parents[2] / "data" / "raw" / "ord_az_buchwald_hartwig.parquet"
FIELDS = ("aryl_halide", "amine", "base", "pd_source", "ligand", "solvent", "temperature_c")


# -- the answer key -------------------------------------------------------------------------
@dataclass
class Record:
    id: str
    text: str
    key: dict


def _smiles(component) -> str:
    return next(i.value for i in component.identifiers if i.type == 2)  # 2 = SMILES


def load_records() -> list[Record]:
    from ord_schema.proto import reaction_pb2
    out = []
    for rid, blob in pd.read_parquet(ORD).itertuples(index=False):
        r = reaction_pb2.Reaction()
        r.ParseFromString(blob)
        inputs = {k: [_smiles(c) for c in v.components] for k, v in r.inputs.items()}
        metal = inputs.get("metal and ligand", [])
        pd_ = [s for s in metal if "Pd" in s]
        lig = [s for s in metal if "Pd" not in s]
        t = r.conditions.temperature.setpoint
        key = {
            "aryl_halide": ".".join(inputs.get("aryl halide", [])) or None,
            "amine": ".".join(inputs.get("amine", [])) or None,
            "base": ".".join(inputs.get("Base", [])) or None,
            "pd_source": ".".join(pd_) or None,
            "ligand": ".".join(lig) or None,
            "solvent": ".".join(inputs.get("Solvent", [])) or None,
            "temperature_c": t.value if r.conditions.temperature.HasField("setpoint") else None,
        }
        out.append(Record(rid, r.notes.procedure_details.strip(), key))
    return out


def sample(n: int = 100, n_shots: int = 2, seed: int = 0) -> tuple[list[Record], list[Record]]:
    """n records to test on and n_shots others to show as worked examples (never both)."""
    recs = [r for r in load_records() if len(r.text) > 80]
    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs[:n], recs[n:n + n_shots]


# -- what the model must return ---------------------------------------------------------------
class Conditions(BaseModel):
    aryl_halide: str | None = Field(None, description="SMILES of the aryl (pseudo)halide")
    amine: str | None = Field(None, description="SMILES of the amine (N-H) coupling partner")
    base: str | None = Field(None, description="SMILES of the base")
    pd_source: str | None = Field(None, description="SMILES of the palladium source or precatalyst")
    ligand: str | None = Field(None, description="SMILES of a separately added ligand, if any")
    solvent: str | None = Field(None, description="SMILES of the solvent(s), dot-separated")
    temperature_c: float | None = Field(None, description="reaction temperature in degrees Celsius")


SYSTEM = """You extract reaction conditions from a chemist's lab-notebook procedure for a \
Buchwald-Hartwig amination. Give each compound as a SMILES string. If a compound is named \
only by a code or a vague term ("the bromide", "compound 12") and its structure cannot be \
worked out from the text, return null for it rather than guessing. Leave ligand null if no \
separate ligand was added (for example when a precatalyst already carries one)."""


def _prompt(rec: Record, shots: list[Record]) -> list[dict]:
    msgs = []
    for s in shots:  # few-shot: a worked example is a user turn and the answer as JSON
        msgs.append({"role": "user", "content": f"Procedure:\n{s.text}"})
        msgs.append({"role": "assistant", "content": json.dumps(s.key)})
    msgs.append({"role": "user", "content": f"Procedure:\n{rec.text}"})
    return msgs


def ask(client, model: str, rec: Record, shots: list[Record], effort: str = "low") -> dict:
    """One structured-output call. Returns the fields, or an error/refusal marker."""
    resp = client.messages.parse(
        model=model, max_tokens=4000, system=SYSTEM, messages=_prompt(rec, shots),
        output_format=Conditions, output_config={"effort": effort},
    )
    usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    if resp.stop_reason == "refusal":  # a safety decline is a result too: count it, don't retry elsewhere
        return {"_refused": True, "_usage": usage}
    if resp.parsed_output is None:
        return {"_invalid": True, "_usage": usage}
    return {**resp.parsed_output.model_dump(), "_usage": usage}


# -- scoring ---------------------------------------------------------------------------------
def _fragments(smiles: str) -> frozenset[str] | None:
    """Structure identity of a (multi-fragment) SMILES: InChIKey connectivity blocks."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    keys = set()
    for frag in Chem.GetMolFrags(mol, asMols=True):
        k = Chem.MolToInchiKey(frag)
        if k:
            keys.add(k.split("-")[0])
    return frozenset(keys)


def score_field(field: str, answer, key) -> str:
    if key is None:
        return "no_key" if answer is None else "extra"
    if answer is None:
        return "missed"
    if field == "temperature_c":
        return "correct" if abs(float(answer) - float(key)) <= 5 else "wrong"
    want = _fragments(str(key))
    if want is None:
        return "bad_key"  # the answer key itself does not parse: nothing to compare with, so not the model's error
    got = _fragments(str(answer))
    if got is None:
        return "invalid"
    return "correct" if got == want else "wrong"


def score(answers: list[dict], records: list[Record]) -> pd.DataFrame:
    rows = []
    for a, r in zip(answers, records):
        for f in FIELDS:
            outcome = "failed" if a.get("_refused") or a.get("_invalid") else score_field(f, a.get(f), r.key[f])
            rows.append(dict(id=r.id, field=f, outcome=outcome))
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """Per field: share of records (with a key) that were correct / wrong / missed / invalid."""
    keyed = scored[~scored.outcome.isin(["no_key", "extra", "bad_key"])]
    tab = keyed.groupby("field").outcome.value_counts(normalize=True).unstack(fill_value=0)
    for col in ("correct", "wrong", "missed", "invalid", "failed"):
        if col not in tab:
            tab[col] = 0.0
    return tab[["correct", "wrong", "missed", "invalid", "failed"]].reindex(FIELDS).round(3)


# -- the experiment --------------------------------------------------------------------------
CONDITIONS = (  # (label, model, shots)
    ("haiku-5.5 zero-shot", "claude-haiku-5-5", 0),
    ("haiku-5.5 few-shot", "claude-haiku-5-5", 2),
    ("opus-5.5 zero-shot", "claude-opus-5-5", 0),
)


def main(model: str | None = None, n: int = 100, log=print) -> pd.DataFrame:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    test, shots = sample(n)
    conds = [c for c in CONDITIONS if model is None or c[1] == model]
    RESULTS.mkdir(exist_ok=True)
    tables, tokens = [], {}
    with open(RESULTS / "extract_answers.jsonl", "w") as fh:
        for label, m, k in conds:
            t = time.time()
            answers = []
            for rec in test:
                try:
                    a = ask(client, m, rec, shots[:k])
                except anthropic.BadRequestError as e:
                    a = {"_invalid": True, "_error": str(e)[:200], "_usage": {"in": 0, "out": 0}}
                answers.append(a)
                fh.write(json.dumps({"condition": label, "id": rec.id, "answer": a, "key": rec.key}) + "\n")
            tab = summarise(score(answers, test))
            tab.insert(0, "condition", label)
            tables.append(tab)
            tokens[label] = {s: sum(a["_usage"][s] for a in answers) for s in ("in", "out")}
            log(f"extract {label}: mean correct {tab.correct.mean():.3f} ({time.time() - t:.0f}s, tokens {tokens[label]})")
    res = pd.concat(tables).reset_index()
    res.to_csv(RESULTS / "table2_extraction.csv", index=False)
    (RESULTS / "table2_tokens.json").write_text(json.dumps(tokens, indent=1))
    return res
