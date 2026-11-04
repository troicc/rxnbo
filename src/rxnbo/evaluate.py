"""Metrics and figures. Every figure is drawn from a CSV in results/, so a figure can be
redrawn without re-running the experiment and the numbers behind it can be read."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"mean": "#9a9a9a", "ridge": "#7b8fa8", "rf": "#2f6f9f", "mlp": "#c06a2b", "gp": "#5b8c3a",
          "random": "#9a9a9a", "greedy": "#7b8fa8", "ei": "#2f6f9f", "pi": "#8a5fa8",
          "ucb": "#c06a2b", "ts": "#5b8c3a"}


def r2(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def save(fig, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig1_models(res: pd.DataFrame) -> Path:
    """R^2 and MAE per model, random vs additive split (mean +- std over seeds / tests)."""
    g = res.groupby(["split", "model"]).agg(r2=("r2", "mean"), r2_sd=("r2", "std"),
                                            mae=("mae", "mean"), mae_sd=("mae", "std")).reset_index()
    models = [m for m in ("mean", "ridge", "rf", "mlp", "gp") if m in set(g.model)]
    splits = [s for s in ("random", "additive", "arene") if s in set(g.split)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    w = 0.8 / len(models)
    for ax, metric in zip(axes, ("r2", "mae")):
        for i, m in enumerate(models):
            sub = g[g.model == m].set_index("split").reindex(splits)
            ax.bar(np.arange(len(splits)) + i * w, sub[metric], w, yerr=sub[f"{metric}_sd"],
                   color=COLORS[m], label=m, capsize=2)
        ax.set_xticks(np.arange(len(splits)) + w * (len(models) - 1) / 2, splits)
        ax.set_ylabel("$R^2$" if metric == "r2" else "MAE (yield %)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(frameon=False, ncol=len(models), fontsize=7, loc="lower left")
    fig.suptitle("Figure 1 · models on random and out-of-sample splits", fontsize=9)
    return save(fig, "fig1_models")


def fig2_curve(res: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(4, 3))
    for m, sub in res.groupby("model"):
        g = sub.groupby("frac").r2.agg(["mean", "std"])
        ax.errorbar(g.index * 100, g["mean"], g["std"], marker="o", ms=3, capsize=2,
                    color=COLORS.get(m), label=m)
    ax.set_xscale("log")
    ax.set_xlabel("training set (% of 3955 reactions)")
    ax.set_ylabel("test $R^2$")
    ax.legend(frameon=False)
    ax.set_title("Figure 2 · learning curves (random split)", fontsize=9)
    return save(fig, "fig2_learning_curve")


def fig3_features(res: pd.DataFrame) -> Path:
    g = res.groupby(["features", "split"]).r2.agg(["mean", "std"]).reset_index()
    feats = list(dict.fromkeys(res.features))
    splits = list(dict.fromkeys(res.split))
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.8 / len(splits)
    for i, s in enumerate(splits):
        sub = g[g.split == s].set_index("features").reindex(feats)
        ax.bar(np.arange(len(feats)) + i * w, sub["mean"], w, yerr=sub["std"], capsize=2,
               label=s, color=("#2f6f9f", "#c06a2b", "#5b8c3a")[i])
    ax.set_xticks(np.arange(len(feats)) + w * (len(splits) - 1) / 2, feats, rotation=20, ha="right")
    ax.set_ylabel("test $R^2$ (random forest)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("Figure 3 · which representation", fontsize=9)
    return save(fig, "fig3_features")


def fig4_bo(curves: pd.DataFrame, top1: float) -> Path:
    """Best yield found vs experiments, median and interquartile band over seeds."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for s, sub in curves.groupby("strategy"):
        piv = sub.pivot(index="experiment", columns="seed", values="best")
        q = piv.quantile([0.25, 0.5, 0.75], axis=1).T
        ax.plot(q.index, q[0.5], color=COLORS[s], label=s, lw=1.4)
        ax.fill_between(q.index, q[0.25], q[0.75], color=COLORS[s], alpha=0.15, lw=0)
    ax.axhline(top1, color="k", lw=0.6, ls=":")
    ax.text(curves.experiment.max(), top1, " top 1%", va="center", fontsize=7)
    ax.axvline(10.5, color="k", lw=0.4, alpha=0.4)
    ax.text(10.8, 61, "BO starts", fontsize=7, alpha=0.6)
    ax.set_ylim(60, 101)  # the first random picks already reach ~70-80 %; the race is above
    ax.set_xlabel("experiments run")
    ax.set_ylabel("best yield found (%)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Figure 4 · Bayesian optimisation vs random screening", fontsize=9)
    return save(fig, "fig4_bo")


def fig5_repr(summary: pd.DataFrame) -> Path:
    g = summary.groupby("representation").coverage.agg(["mean", "std"]).reindex(
        [r for r in ("onehot", "morgan", "drfp", "embedding", "random") if r in set(summary.representation)])
    fig, ax = plt.subplots(figsize=(4.2, 3))
    labels = {"onehot": "one-hot", "morgan": "Morgan", "drfp": "DRFP",
              "embedding": "text\nembedding", "random": "random\n(no model)"}
    ax.bar([labels[i] for i in g.index], g["mean"] * 100, yerr=g["std"] * 100, capsize=2,
           color=["#7b8fa8", "#2f6f9f", "#5b8c3a", "#c06a2b", "#9a9a9a"][: len(g)])
    ax.set_ylabel("top-5% reactions found (%)")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title("Figure 5 · one GP-BO loop (EI, 110 experiments), four representations", fontsize=9)
    return save(fig, "fig5_bo_representations")
