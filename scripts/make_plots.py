"""Plot generator. Reads runs/<exp>/summary.csv (+ per-iter JSON for the
training-curve overlays and confusion-matrix delta) and writes PNG+SVG
plots into runs/<exp>/plots/.

Plots produced:
  1. data_efficiency.{png,svg}       — target-test F1 vs n_target_units (mean ± std)
  2. cross_domain.{png,svg}          — source / target / other-target F1 vs n_target_units
  3. per_type_heatmap_<eval>.{png,svg}  — entity_type × iteration F1 heatmap
  4. confusion_iter0_vs_iterK_<eval>.{png,svg} — side-by-side
  5. training_curves_seed_<s>.{png,svg} — overlay across iterations (one panel per seed)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, base: str) -> None:
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_data_efficiency(df: pd.DataFrame, plots_dir: str, target_eval: str) -> None:
    sub = df[df["eval_set"] == target_eval]
    if sub.empty:
        return
    g = sub.groupby("n_target_units")["entity_f1_micro"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g["n_target_units"], g["mean"], marker="o", label=f"{target_eval} F1 (mean over seeds)")
    if (g["count"] > 1).any():
        ax.fill_between(g["n_target_units"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)
    ax.set_xlabel("n target units injected")
    ax.set_ylabel("entity F1 (micro)")
    ax.set_title(f"Data-efficiency curve — eval on {target_eval}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, os.path.join(plots_dir, "data_efficiency"))


def plot_cross_domain(df: pd.DataFrame, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for eval_set, sub in df.groupby("eval_set"):
        g = sub.groupby("n_target_units")["entity_f1_micro"].agg(["mean", "std", "count"]).reset_index()
        ax.plot(g["n_target_units"], g["mean"], marker="o", label=eval_set)
        if (g["count"] > 1).any():
            ax.fill_between(g["n_target_units"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
    ax.set_xlabel("n target units injected")
    ax.set_ylabel("entity F1 (micro)")
    ax.set_title("Cross-domain F1 — forgetting & transfer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, os.path.join(plots_dir, "cross_domain"))


def plot_per_type_heatmap(df: pd.DataFrame, plots_dir: str) -> None:
    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    if not f1_cols:
        return
    for eval_set, sub in df.groupby("eval_set"):
        pivot = sub.groupby("iteration")[f1_cols].mean()
        # drop columns that are all NaN/0 for this eval set
        pivot = pivot.loc[:, (pivot.fillna(0).abs().sum(axis=0) > 0)]
        if pivot.empty:
            continue
        types = [c[3:] for c in pivot.columns]
        mat = pivot.values.T  # rows=type, cols=iter
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[0] + 3),
                                        max(3, 0.3 * len(types) + 1)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[0]))
        ax.set_xticklabels([str(i) for i in pivot.index])
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types, fontsize=8)
        ax.set_xlabel("iteration")
        ax.set_title(f"Per-type F1 — eval on {eval_set}")
        plt.colorbar(im, ax=ax, fraction=0.025)
        save(fig, os.path.join(plots_dir, f"per_type_heatmap_{eval_set}"))


def _read_confusion(path: str):
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        rows = []
        for row in reader:
            rows.append([int(x) for x in row[1:]])
    return labels, np.array(rows, dtype=np.int64)


def plot_confusion_iter0_vs_iterK(run_dir: str, plots_dir: str, df: pd.DataFrame) -> None:
    seeds = sorted(df["seed"].unique())
    iterations = sorted(df["iteration"].unique())
    if not seeds or not iterations:
        return
    seed = seeds[0]
    k0, kN = iterations[0], iterations[-1]
    eval_sets = sorted(df["eval_set"].unique())
    for es in eval_sets:
        p0 = os.path.join(run_dir, "seeds", f"seed_{seed}", f"iter_{k0:03d}",
                          f"confusion_matrix_{es}.csv")
        pN = os.path.join(run_dir, "seeds", f"seed_{seed}", f"iter_{kN:03d}",
                          f"confusion_matrix_{es}.csv")
        l0, m0 = _read_confusion(p0)
        lN, mN = _read_confusion(pN)
        if m0 is None or mN is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mat, labels, title in [
            (axes[0], m0, l0, f"iter {k0}"),
            (axes[1], mN, lN, f"iter {kN}"),
        ]:
            row_sum = mat.sum(axis=1, keepdims=True).clip(min=1)
            normed = mat / row_sum
            im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(f"{title} — {es} (row-normalized)")
            plt.colorbar(im, ax=ax, fraction=0.04)
        save(fig, os.path.join(plots_dir, f"confusion_iter0_vs_iterK_{es}"))


def plot_training_curves(run_dir: str, plots_dir: str, df: pd.DataFrame) -> None:
    for seed in sorted(df["seed"].unique()):
        seed_dir = os.path.join(run_dir, "seeds", f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            continue
        iter_dirs = sorted(d for d in os.listdir(seed_dir) if d.startswith("iter_"))
        if not iter_dirs:
            continue
        fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5))
        for d in iter_dirs:
            log_path = os.path.join(seed_dir, d, "train_log.jsonl")
            if not os.path.exists(log_path):
                continue
            rows = [json.loads(l) for l in open(log_path, encoding="utf-8") if l.strip()]
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            ax_loss.plot(epochs, [r["train_loss"] for r in rows], label=d, alpha=0.8)
            ax_f1.plot(epochs, [r["dev_f1"] for r in rows], label=d, alpha=0.8)
        ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("train loss")
        ax_loss.set_title(f"train loss — seed {seed}"); ax_loss.grid(True, alpha=0.3); ax_loss.legend(fontsize=7)
        ax_f1.set_xlabel("epoch"); ax_f1.set_ylabel("dev F1 (target)")
        ax_f1.set_title(f"target-dev F1 — seed {seed}"); ax_f1.grid(True, alpha=0.3); ax_f1.legend(fontsize=7)
        save(fig, os.path.join(plots_dir, f"training_curves_seed_{seed}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Path to runs/<exp>/")
    p.add_argument("--target-eval", default=None,
                   help="Eval-set name to use as the target curve (defaults to "
                        "the eval set with highest f1 variance over iterations)")
    args = p.parse_args()

    summary = os.path.join(args.run_dir, "summary.csv")
    df = pd.read_csv(summary)
    plots_dir = os.path.join(args.run_dir, "plots")

    if args.target_eval:
        target_eval = args.target_eval
    else:
        # heuristic: pick the eval set whose F1 changes most across iterations
        var = df.groupby("eval_set")["entity_f1_micro"].var().sort_values(ascending=False)
        target_eval = var.index[0] if len(var) else "target_test"

    plot_data_efficiency(df, plots_dir, target_eval)
    plot_cross_domain(df, plots_dir)
    plot_per_type_heatmap(df, plots_dir)
    plot_confusion_iter0_vs_iterK(args.run_dir, plots_dir, df)
    plot_training_curves(args.run_dir, plots_dir, df)
    print(f"Plots saved under {plots_dir}")


if __name__ == "__main__":
    main()
