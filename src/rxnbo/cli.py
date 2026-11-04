"""rxnbo <command> - one command per result.

    rxnbo prepare        read the Excel file, write data/processed/
    rxnbo fig1 ... fig5  run one experiment, write results/<fig>.csv and .png
    rxnbo all            every figure and table, in order
    rxnbo extract        LLM extraction evaluation (Table 2; needs ANTHROPIC_API_KEY)
    rxnbo agent          tool-use agent evaluation (Table 3; needs ANTHROPIC_API_KEY)
    rxnbo smoke          a two-minute run of everything at small size, to check it works
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rxnbo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "fig1", "fig2", "fig3", "fig4", "fig5", "all",
                                        "extract", "agent", "smoke"])
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    ap.add_argument("--model", default=None, help="Claude model for extract / agent")
    a = ap.parse_args(argv)

    from . import data, experiments as E
    seeds = {} if a.seeds is None else {"seeds": a.seeds}
    if a.command == "prepare":
        df = data.prepare()
        print(f"{len(df)} reactions; additive test sets: {list(data.additive_test_sets())}")
    elif a.command in ("fig1", "fig2", "fig3", "fig4", "fig5"):
        getattr(E, a.command)(**seeds)
    elif a.command == "all":
        for f in ("fig1", "fig2", "fig3", "fig4", "fig5"):
            getattr(E, f)(**seeds)
    elif a.command == "smoke":
        E.fig1(models=("mean", "rf", "gp"), kinds=("random",), seeds=1)
        E.fig4(strategies=("random", "ei", "ts"), seeds=2)
    elif a.command == "extract":
        from . import llm_extract
        llm_extract.main(model=a.model)
    elif a.command == "agent":
        from . import agent
        agent.main(model=a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
