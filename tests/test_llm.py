"""The LLM parts, tested without calling any model: a fake client plays the model."""
from types import SimpleNamespace as NS

import pytest

from rxnbo import agent, llm_extract as X


@pytest.fixture(scope="module")
def records():
    return X.load_records()


def test_answer_key_parsed(records):
    assert len(records) == 750
    r = records[0]
    assert "To a solution" in r.text or len(r.text) > 50
    assert r.key["temperature_c"] is not None and r.key["aryl_halide"]


def test_test_and_shots_never_overlap():
    test, shots = X.sample(100, 2)
    assert len(test) == 100 and len(shots) == 2
    assert not {r.id for r in test} & {r.id for r in shots}


def test_compound_scoring_by_structure_not_spelling():
    cs2co3 = "C(=O)([O-])[O-].[Cs+].[Cs+]"
    assert X.score_field("base", "[Cs+].[Cs+].[O-]C([O-])=O", cs2co3) == "correct"
    assert X.score_field("base", "O=C([O-])[O-].[K+].[K+]", cs2co3) == "wrong"
    assert X.score_field("base", "not a smiles((", cs2co3) == "invalid"
    assert X.score_field("base", None, cs2co3) == "missed"
    assert X.score_field("ligand", None, None) == "no_key"
    dba = "C1=CC=C(C=C1)/C=C/C(=O)/C=C/C2=CC=CC=C2"
    assert X.score_field("pd_source", f"{dba.replace('/', '')}.[Pd]", f"{dba}.{dba}.[Pd].[Pd]") == "correct"
    assert X.score_field("temperature_c", 104, 100.0) == "correct"
    assert X.score_field("temperature_c", 80, 100.0) == "wrong"


def test_tools_give_rdkit_values():
    d = agent.describe("CCO")
    assert d["formula"] == "C2H6O" and abs(d["mol_weight"] - 46.069) < 0.01 and d["h_bond_donors"] == 1
    assert agent.similarity("CCO", "CCO")["tanimoto"] == 1.0
    assert agent.substructure("c1ccncc1", "n1ccccc1")["contains"]
    with pytest.raises(ValueError):
        agent.describe("C1CC")


def test_check_reads_the_last_answer_line():
    t = {"answer": 46.07, "tol": 0.1}
    assert agent.check(t, "thinking...\nANSWER: 46.1")
    assert not agent.check(t, "ANSWER: 47")
    assert not agent.check(t, "46.07")  # no ANSWER line
    assert agent.check({"answer": True, "tol": None}, "ANSWER: Yes")
    assert agent.check({"answer": "C2H6O", "tol": 0}, "ANSWER: C2H6O")


class FakeClient:
    """Plays a model that calls `describe` once, then answers from the tool's result."""
    def __init__(self):
        self.messages = self
        self.seen = []

    def create(self, **kw):
        self.seen.append(kw)
        usage = NS(input_tokens=10, output_tokens=5)
        last = kw["messages"][-1]["content"]
        if isinstance(last, str):  # first turn: call the tool
            block = NS(type="tool_use", id="t1", name="describe", input={"smiles": "CCO"})
            return NS(stop_reason="tool_use", content=[block], usage=usage)
        import json
        mw = json.loads(last[0]["content"])["mol_weight"]
        return NS(stop_reason="end_turn", content=[NS(type="text", text=f"ANSWER: {mw}")], usage=usage)


def test_agent_loop_runs_the_tool_and_returns_its_result():
    c = FakeClient()
    r = agent.run_agent(c, "fake", "MW of ethanol?", use_tools=True)
    assert [x["tool"] for x in r["calls"]] == ["describe"]
    assert agent.check({"answer": 46.069, "tol": 0.1}, r["text"])
    second = c.seen[1]["messages"]
    assert second[-1]["content"][0]["tool_use_id"] == "t1"  # result paired with its call
    assert r["tokens"] == {"in": 20, "out": 10}


def test_tasks_have_rdkit_answers():
    ts = agent.tasks(14)
    assert len(ts) == 14 and {t["kind"] for t in ts} == {"mol_weight", "formula", "h_bond_donors", "tpsa",
                                                         "rings", "similarity", "substructure"}
