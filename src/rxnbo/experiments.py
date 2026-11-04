"""The experiments behind Figures 1-5 and Table 1. Each writes a CSV to results/ and draws
its figure from that CSV."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import bo, data, evaluate
from .featurize import featurize
from .models import baselines
from .models.gp import GP
from .models.mlp import MLPRegressor

R = evaluate.RESULTS


def make_model(name: str, seed: int):
    if name == "mlp":
        return MLPRegressor(seed=seed)
    if name == "gp":
        return GP("tanimoto")
    return baselines.make(name, seed)


def _splits(df, kinds, seeds):
    for kind in kinds:
        ks = {"random": range(seeds), "additive": range(1, 5), "arene": range(5)}[kind]
        for k in ks:
            yield kind, k, *data.split(df, kind, k)


def fig1(models=("mean", "ridge", "rf", "mlp", "gp"), kinds=("random", "additive", "arene"),
         seeds: int = 5, log=print) -> pd.DataFrame:
    """Every model on Morgan fingerprints (r=2, 1024 bits per component)."""
    df = data.load()
    X, y = featurize(df, "morgan"), df["yield"].to_numpy()
    rows = []
    for kind, k, tr, te in _splits(df, kinds, seeds):
        for m in models:
            t = time.time()
            p = make_model(m, k).fit(X[tr], y[tr]).predict(X[te])
            rows.append(dict(split=kind, k=k, model=m, n_train=len(tr), r2=evaluate.r2(y[te], p),
                             mae=evaluate.mae(y[te], p), rmse=evaluate.rmse(y[te], p)))
            log(f"fig1 {kind}{k} {m:6s} R2={rows[-1]['r2']:.3f} ({time.time() - t:.0f}s)")
    res = pd.DataFrame(rows)
    R.mkdir(exist_ok=True)
    res.to_csv(R / "fig1_models.csv", index=False)
    evaluate.fig1_models(res)
    return res


def fig2(models=("rf", "mlp", "gp"), fracs=(0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
         seeds: int = 5, log=print) -> pd.DataFrame:
    df = data.load()
    X, y = featurize(df, "morgan"), df["yield"].to_numpy()
    rows = []
    for f in fracs:
        for s in range(seeds):
            tr, te = data.random_split(len(df), f, s)
            for m in models:
                p = make_model(m, s).fit(X[tr], y[tr]).predict(X[te])
                rows.append(dict(frac=f, seed=s, model=m, r2=evaluate.r2(y[te], p), mae=evaluate.mae(y[te], p)))
            log(f"fig2 {f:.3f} seed {s}: " + " ".join(f"{r['model']}={r['r2']:.3f}" for r in rows[-len(models):]))
    res = pd.DataFrame(rows)
    res.to_csv(R / "fig2_learning_curve.csv", index=False)
    evaluate.fig2_curve(res)
    return res


FEATURE_SETS = {  # label -> (representation, settings)
    "one-hot": ("onehot", {}),
    "descriptors": ("descriptors", {}),
    "Morgan r2 1024": ("morgan", {"radius": 2, "n_bits": 1024}),
    "Morgan r2 2048": ("morgan", {"radius": 2, "n_bits": 2048}),
    "Morgan r3 1024": ("morgan", {"radius": 3, "n_bits": 1024}),
    "DRFP 2048": ("drfp", {"n_bits": 2048}),
}


def fig3(kinds=("random", "additive"), seeds: int = 3, log=print) -> pd.DataFrame:
    """The same random forest on each representation."""
    df = data.load()
    y = df["yield"].to_numpy()
    rows = []
    for label, (kind, kw) in FEATURE_SETS.items():
        X = featurize(df, kind, **kw)
        for split, k, tr, te in _splits(df, kinds, seeds):
            p = baselines.make("rf", k).fit(X[tr], y[tr]).predict(X[te])
            rows.append(dict(features=label, split=split, k=k, r2=evaluate.r2(y[te], p), mae=evaluate.mae(y[te], p)))
        log(f"fig3 {label}: " + " ".join(f"{r['split']}{r['k']}={r['r2']:.2f}" for r in rows if r["features"] == label))
    res = pd.DataFrame(rows)
    res.to_csv(R / "fig3_features.csv", index=False)
    evaluate.fig3_features(res)
    return res


def _bo_one(X, y, strategy, seed, kernel, chol):
    import torch
    torch.set_num_threads(1)  # one process per seed; threads inside would fight
    return bo.run(X, y, strategy, seed, kernel=kernel, chol=chol)


def _bo_many(X, y, strategies, seeds, kernel="tanimoto"):
    chol = bo.pool_cholesky(X) if "ts" in strategies and kernel == "tanimoto" else None
    jobs = [(s, k) for s in strategies for k in range(seeds)]
    runs = Parallel(n_jobs=min(8, os.cpu_count() or 1))(
        delayed(_bo_one)(X, y, s, k, kernel, chol) for s, k in jobs)
    return runs


def fig4(strategies=("random", "greedy", "ei", "pi", "ucb", "ts"), seeds: int = 20, log=print):
    """GP (Tanimoto, Morgan) BO: 10 random starts, then 20 rounds of 5. Also Table 1."""
    df = data.load()
    X, y = featurize(df, "morgan"), df["yield"].to_numpy()
    t = time.time()
    runs = _bo_many(X, y, strategies, seeds)
    log(f"fig4 {len(runs)} runs in {time.time() - t:.0f}s")
    curves = pd.DataFrame([dict(strategy=r.strategy, seed=r.seed, experiment=i + 1, best=b, picked=int(p))
                           for r in runs for i, (b, p) in enumerate(zip(bo.best_so_far(r, y), r.picked))])
    curves.to_csv(R / "fig4_bo_curves.csv", index=False)
    table = table1(runs, y)
    evaluate.fig4_bo(curves, float(np.quantile(y, 0.99)))
    return curves, table


def table1(runs, y) -> pd.DataFrame:
    rows = [dict(strategy=r.strategy, seed=r.seed, to_top1=bo.experiments_to_top(r, y, 0.01),
                 to_top5=bo.experiments_to_top(r, y, 0.05), coverage5=bo.top_coverage(r, y, 0.05),
                 best=float(bo.best_so_far(r, y)[-1])) for r in runs]
    per = pd.DataFrame(rows)
    per.to_csv(R / "table1_bo_runs.csv", index=False)

    def found(s):  # share of seeds that reached the top 1% within the budget
        return float(np.isfinite(s).mean())
    tab = per.groupby("strategy").agg(
        median_to_top1=("to_top1", "median"), found_top1=("to_top1", found),
        median_to_top5=("to_top5", "median"), coverage5=("coverage5", "mean"),
        best=("best", "mean")).round(3)
    tab.to_csv(R / "table1_bo.csv")
    (R / "table1_bo.md").write_text(_markdown(tab.reset_index()))
    return tab


def fig5(reprs=("onehot", "morgan", "drfp", "embedding"), strategy="ei", seeds: int = 20, log=print):
    """One BO loop (EI), four representations; RBF kernel for the real-valued embedding."""
    df = data.load()
    y = df["yield"].to_numpy()
    rows = []
    for rep in reprs:
        X = featurize(df, rep)
        kernel = "rbf" if rep == "embedding" else "tanimoto"
        t = time.time()
        runs = _bo_many(X, y, (strategy,), seeds, kernel)
        rows += [dict(representation=rep, seed=r.seed, coverage=bo.top_coverage(r, y, 0.05),
                      to_top1=bo.experiments_to_top(r, y, 0.01)) for r in runs]
        log(f"fig5 {rep}: coverage {np.mean([r['coverage'] for r in rows[-seeds:]]):.3f} ({time.time() - t:.0f}s)")
    rnd = _bo_many(featurize(df, "onehot"), y, ("random",), seeds)
    rows += [dict(representation="random", seed=r.seed, coverage=bo.top_coverage(r, y, 0.05),
                  to_top1=bo.experiments_to_top(r, y, 0.01)) for r in rnd]
    res = pd.DataFrame(rows)
    res.to_csv(R / "fig5_bo_representations.csv", index=False)
    evaluate.fig5_repr(res)
    return res


def _markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.itertuples(index=False))
