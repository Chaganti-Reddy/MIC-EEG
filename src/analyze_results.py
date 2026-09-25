"""
Comprehensive results analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLASS_NAMES, FIGURES_DIR, METRICS_DIR, N_CLASSES, SYNTHETIC_DIR,
)

PALETTE = sns.color_palette("tab10", 9)
PALETTE_CLASSES = sns.color_palette("Set2", N_CLASSES)
METRIC_COLOURS = {
    "accuracy":  "#2196F3",
    "precision": "#4CAF50",
    "recall":    "#FF9800",
    "f1":        "#9C27B0",
}
METRICS = ["accuracy", "precision", "recall", "f1"]


def _save(fig: plt.Figure, path: str, label: str = "") -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    name = label or Path(path).name
    print(f"  {name}")


def load_per_subject_metrics(subjects: List[int]) -> pd.DataFrame:
    rows = []
    for s in subjects:
        jpath = METRICS_DIR / f"cnn_metrics_s{s:02d}.json"
        if not jpath.exists():
            print(f"  [warn] metrics not found: S{s:02d}")
            continue
        with open(jpath) as fh:
            m = json.load(fh)
        rows.append({
            "subject":   s,
            "label":     f"S{s:02d}",
            "accuracy":  m.get("accuracy",  np.nan),
            "precision": m.get("precision", np.nan),
            "recall":    m.get("recall",    np.nan),
            "f1":        m.get("f1",        np.nan),
        })
    return pd.DataFrame(rows)


def load_classwise(subjects: List[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for s in subjects:
        jpath = METRICS_DIR / f"cnn_metrics_s{s:02d}_classwise.json"
        if jpath.exists():
            with open(jpath) as fh:
                out[s] = json.load(fh)
    return out


def load_cross_matrix(subjects: List[int], metric: str = "accuracy") -> pd.DataFrame:
    n = len(subjects)
    mat = np.full((n, n), np.nan)
    for i, ts in enumerate(subjects):
        jpath = METRICS_DIR / f"cnn_metrics_s{ts:02d}.json"
        if jpath.exists():
            with open(jpath) as fh:
                mat[i, i] = json.load(fh).get(metric, np.nan)
        for j, xs in enumerate(subjects):
            if i == j:
                continue
            cpath = METRICS_DIR / f"cross_s{ts:02d}_to_s{xs:02d}.json"
            if cpath.exists():
                with open(cpath) as fh:
                    mat[i, j] = json.load(fh).get(metric, np.nan)
    labels = [f"S{s}" for s in subjects]
    return pd.DataFrame(mat * 100, index=labels, columns=labels)


def load_history(subjects: List[int]) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for s in subjects:
        p = METRICS_DIR / f"cnn_history_s{s:02d}.csv"
        if p.exists():
            out[s] = pd.read_csv(str(p))
    return out


def load_wgan_logs(subjects: List[int]) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for s in subjects:
        p = METRICS_DIR / f"wgan_training_log_s{s:02d}.csv"
        if p.exists():
            out[s] = pd.read_csv(str(p))
    return out


def load_synthetic(subject: int) -> Optional[tuple]:
    p = SYNTHETIC_DIR / f"synthetic_combined_s{subject:02d}.npz"
    if p.exists():
        d = np.load(str(p))
        return d["X"], d["y"]
    return None


def plot_single_metric_bar(df: pd.DataFrame, metric: str, save_path: str) -> None:
    """Generic per-subject bar chart for any metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(df))
    vals = df[metric].values * 100
    bars = ax.bar(x, vals, color=PALETTE[:len(df)], width=0.6,
                  edgecolor="white", linewidth=0.8)
    mean_val = vals.mean()
    ax.axhline(mean_val, color="crimson", linewidth=1.8, linestyle="--",
               label=f"Mean = {mean_val:.2f}%")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=11)
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel(f"{metric.capitalize()} (%)", fontsize=12)
    ax.set_ylim(0, 110); ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    _save(fig, save_path, f"summary_{metric}_bar.png")


def plot_grouped_metrics(df: pd.DataFrame, save_path: str) -> None:
    n_sub, n_m = len(df), len(METRICS)
    x, w = np.arange(n_sub), 0.18
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (m, col) in enumerate(METRIC_COLOURS.items()):
        ax.bar(x + (i - n_m / 2 + 0.5) * w, df[m].values * 100, w,
               label=m.capitalize(), color=col, alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=11)
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115); ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "summary_metrics_grouped.png")


def plot_boxplots(df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    rng = np.random.RandomState(42)
    for ax, m in zip(axes, METRICS):
        vals = df[m].dropna().values * 100
        ax.boxplot(vals, patch_artist=True,
                   boxprops=dict(facecolor=METRIC_COLOURS[m], alpha=0.7),
                   medianprops=dict(color="black", linewidth=2),
                   whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
        ax.scatter(
            1 + rng.uniform(-0.12, 0.12, len(vals)),
            vals,
            color="black",
            alpha=0.6,
            s=35,
            zorder=5,
        )
        ax.set_ylim(0, 110); ax.set_xticks([1]); ax.set_xticklabels([""])
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            1.0,
            2,
            f"μ={vals.mean():.1f}\nσ={vals.std():.1f}",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
    plt.tight_layout(); _save(fig, save_path, "summary_metrics_boxplot.png")


def plot_violin(df: pd.DataFrame, save_path: str) -> None:
    long = pd.melt(df[["label"] + METRICS], id_vars="label",
                   var_name="Metric", value_name="Score")
    long["Score"] *= 100
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=long,
        x="Metric",
        y="Score",
        palette=list(METRIC_COLOURS.values()),
        inner="point",
        ax=ax,
    )
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "summary_violin.png")


def plot_mean_std_bar(df: pd.DataFrame, save_path: str) -> None:
    means = [df[m].mean() * 100 for m in METRICS]
    stds  = [df[m].std()  * 100 for m in METRICS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([m.capitalize() for m in METRICS], means, yerr=stds,
                  color=list(METRIC_COLOURS.values()), capsize=7, edgecolor="white",
                  error_kw=dict(elinewidth=2, ecolor="black"))
    for bar, mu, sd in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd + 0.5,
                f"{mu:.2f}±{sd:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "summary_mean_std.png")


def plot_ranked_accuracy(df: pd.DataFrame, save_path: str) -> None:
    ranked = df.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    cols = [PALETTE[i % 9] for i in range(len(ranked))]
    bars = ax.barh(ranked["label"], ranked["accuracy"] * 100,
                   color=cols, edgecolor="white")
    ax.axvline(25, color="grey", linewidth=1.2, linestyle=":",
               label="Random chance (25%)")
    ax.axvline(ranked["accuracy"].mean() * 100, color="crimson",
               linewidth=1.8, linestyle="--",
               label=f"Mean = {ranked['accuracy'].mean()*100:.1f}%")
    for bar, v in zip(bars, ranked["accuracy"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{v*100:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=10); ax.set_xlim(0, 110); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "summary_ranked.png")


def plot_all_metrics_heatmap(df: pd.DataFrame, save_path: str) -> None:
    heat = df.set_index("label")[METRICS] * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Metric"); ax.set_ylabel("Subject")
    plt.tight_layout(); _save(fig, save_path, "summary_heatmap_all_metrics.png")


def plot_scatter_acc_f1(df: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, row in df.iterrows():
        ax.scatter(row["accuracy"] * 100, row["f1"] * 100,
                   color=PALETTE[i % 9], s=120, zorder=5)
        ax.annotate(row["label"],
                    (row["accuracy"] * 100 + 0.3, row["f1"] * 100 + 0.3),
                    fontsize=9)
    # Diagonal guide
    lim = [20, 100]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("F1-Score (%)", fontsize=12)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    _save(fig, save_path, "summary_scatter_acc_f1.png")


def plot_scatter_prec_rec(df: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, row in df.iterrows():
        ax.scatter(row["precision"] * 100, row["recall"] * 100,
                   color=PALETTE[i % 9], s=120, zorder=5)
        ax.annotate(row["label"],
                    (row["precision"] * 100 + 0.3, row["recall"] * 100 + 0.3),
                    fontsize=9)
    lim = [20, 100]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=1, label="Precision = Recall")
    ax.set_xlabel("Precision (%)", fontsize=12)
    ax.set_ylabel("Recall (%)", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    _save(fig, save_path, "summary_scatter_prec_rec.png")


def plot_above_chance(df: pd.DataFrame, save_path: str) -> None:
    chance = 25.0
    above  = df["accuracy"] * 100 - chance
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    bars = ax.bar(x, above, color=PALETTE[:len(df)], width=0.6, edgecolor="white")
    for bar, v in zip(bars, above):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"+{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], fontsize=11)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy above random chance (%)")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    _save(fig, save_path, "summary_above_chance.png")


def plot_corr_matrix(df: pd.DataFrame, save_path: str) -> None:
    corr = df[METRICS].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 12},
    )
    plt.tight_layout(); _save(fig, save_path, "summary_corr_matrix.png")


def _radar_angles():
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    return angles + angles[:1]


def plot_radar_individual(df: pd.DataFrame, save_path: str) -> None:
    """One radar subplot per subject."""
    n = len(df)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             subplot_kw=dict(polar=True))
    axes = np.array(axes).flatten()
    angles = _radar_angles()
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        vals = [row[m] for m in METRICS] + [row[METRICS[0]]]
        ax.plot(angles, vals, "o-", linewidth=2, color=PALETTE[idx % 9])
        ax.fill(angles, vals, alpha=0.2, color=PALETTE[idx % 9])
        ax.set_thetagrids(np.degrees(angles[:-1]),
                          [m.capitalize() for m in METRICS], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout(); _save(fig, save_path, "radar_chart.png")


def plot_radar_overlay(df: pd.DataFrame, save_path: str) -> None:
    """All subjects overlaid on one radar."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = _radar_angles()
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in METRICS] + [row[METRICS[0]]]
        ax.plot(angles, vals, "o-", linewidth=1.8, color=PALETTE[i % 9],
                label=row["label"], alpha=0.85)
        ax.fill(angles, vals, alpha=0.05, color=PALETTE[i % 9])
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [m.capitalize() for m in METRICS], fontsize=12)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout(); _save(fig, save_path, "radar_chart_overlay.png")


def plot_cross_heatmap(matrix: pd.DataFrame, metric: str, save_path: str) -> None:
    if matrix.isna().all().all():
        print(f"  [skip] No cross-subject data for {metric}")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = matrix.isna()
    sns.heatmap(matrix, annot=True, fmt=".1f",
                cmap="YlOrRd" if metric == "accuracy" else "PuBuGn",
                vmin=0, vmax=100, linewidths=0.5, linecolor="white",
                mask=mask, ax=ax, annot_kws={"size": 10})
    for i in range(len(matrix)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor="blue", lw=2.5))
    ax.set_xlabel("Test Subject",  fontsize=12)
    ax.set_ylabel("Train Subject", fontsize=12)
    plt.tight_layout(); _save(fig, save_path, Path(save_path).name)


def plot_cross_row_mean(acc_matrix: pd.DataFrame, save_path: str) -> None:
    """Mean cross-subject accuracy when used as TRAIN subject"""
    mat = acc_matrix.copy()
    np.fill_diagonal(mat.values, np.nan)
    row_means = mat.mean(axis=1).dropna()
    if row_means.empty:
        print("  [skip] cross row mean — no off-diagonal data")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(row_means.index, row_means.values,
                  color=[PALETTE[i % 9] for i in range(len(row_means))],
                  edgecolor="white")
    ax.axhline(row_means.mean(), color="crimson", linestyle="--",
               label=f"Mean = {row_means.mean():.1f}%")
    for bar, v in zip(bars, row_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("Train Subject")
    ax.set_ylabel("Mean Cross-Subject Accuracy (%)")
    ax.set_ylim(0, 80); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "cross_subject_row_mean.png")


def plot_cross_col_mean(acc_matrix: pd.DataFrame, save_path: str) -> None:
    """Mean accuracy when used as TEST subject"""
    mat = acc_matrix.copy()
    np.fill_diagonal(mat.values, np.nan)
    col_means = mat.mean(axis=0).dropna()
    if col_means.empty:
        print("  [skip] cross col mean — no off-diagonal data")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(col_means.index, col_means.values,
                  color=[PALETTE[i % 9] for i in range(len(col_means))],
                  edgecolor="white")
    ax.axhline(col_means.mean(), color="crimson", linestyle="--",
               label=f"Mean = {col_means.mean():.1f}%")
    for bar, v in zip(bars, col_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("Test Subject")
    ax.set_ylabel("Mean Cross-Subject Accuracy (%)")
    ax.set_ylim(0, 80); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "cross_subject_col_mean.png")


def _classwise_df(classwise: dict[int, dict], metric_key: str) -> Optional[pd.DataFrame]:
    rows = []
    for s, cw in classwise.items():
        row = {"subject": f"S{s:02d}"}
        for cn in CLASS_NAMES:
            row[cn] = cw.get(cn, {}).get(metric_key, np.nan)
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("subject")


def plot_class_heatmap(classwise: dict[int, dict], metric_key: str,
                       title: str, save_path: str) -> None:
    df = _classwise_df(classwise, metric_key)
    if df is None:
        print(f"  [skip] {title} heatmap — no data")
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.55 + 1.5)))
    sns.heatmap(
        df * 100,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("MI Class"); ax.set_ylabel("Subject")
    plt.tight_layout(); _save(fig, save_path, Path(save_path).name)


def plot_class_bar_per_subject(classwise: dict[int, dict], subject: int,
                               save_path: str) -> None:
    cw = classwise.get(subject)
    if cw is None:
        return
    metrics_cw = {"Precision": "precision", "Recall": "recall", "F1": "f1-score"}
    x      = np.arange(N_CLASSES)
    width  = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, key) in enumerate(metrics_cw.items()):
        vals = [cw.get(cn, {}).get(key, 0) * 100 for cn in CLASS_NAMES]
        ax.bar(x + (i - 1) * width, vals, width, label=label,
               color=list(METRIC_COLOURS.values())[i + 1], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("MI Class")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, Path(save_path).name)


def plot_class_stacked_bar(classwise: dict[int, dict], save_path: str) -> None:
    """Stacked bar showing recall of each class per subject."""
    df = _classwise_df(classwise, "recall")
    if df is None:
        print("  [skip] stacked class bar — no data")
        return
    df_pct = df * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(df_pct))
    for j, cn in enumerate(CLASS_NAMES):
        vals = df_pct[cn].fillna(0).values
        ax.bar(df_pct.index, vals, bottom=bottom,
               label=cn, color=PALETTE_CLASSES[j], edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xlabel("Subject")
    ax.set_ylabel("Cumulative Recall (%)")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "class_stacked_bar.png")


def plot_history_overlay(histories: dict[int, pd.DataFrame],
                         col: str, title: str, save_path: str) -> None:
    if not histories:
        print(f"  [skip] {title} overlay — no history data")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for s, h in histories.items():
        if col in h.columns:
            ax.plot(h[col] * 100, label=f"S{s:02d}",
                    color=PALETTE[s - 1], linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{title} (%)")
    ax.legend(ncol=3, fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout(); _save(fig, save_path, Path(save_path).name)


def plot_history_loss_overlay(histories: dict[int, pd.DataFrame],
                              col: str, title: str, save_path: str) -> None:
    if not histories:
        print(f"  [skip] {title} overlay — no history data")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for s, h in histories.items():
        if col in h.columns:
            ax.plot(h[col], label=f"S{s:02d}",
                    color=PALETTE[s - 1], linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.legend(ncol=3, fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout(); _save(fig, save_path, Path(save_path).name)


def plot_history_grid(histories: dict[int, pd.DataFrame],
                      col: str, val_col: str,
                      ylabel: str, title: str, save_path: str,
                      is_pct: bool = True) -> None:
    if not histories:
        print(f"  [skip] {title} grid — no data")
        return
    subjects = sorted(histories.keys())
    n = len(subjects)
    ncols = min(n, 3); nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.50, wspace=0.38)
    for idx, s in enumerate(subjects):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        h = histories[s]
        scale = 100 if is_pct else 1
        if col in h.columns:
            ax.plot(h[col] * scale, label="Train", color=PALETTE[s - 1], linewidth=2)
        if val_col in h.columns:
            ax.plot(
                h[val_col] * scale,
                label="Val",
                color=PALETTE[s - 1],
                linewidth=2,
                linestyle="--",
                alpha=0.7,
            )
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        if is_pct:
            ax.set_ylim(0.0, 1.05)
    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)
    fig.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    print(f"  {Path(save_path).name}")


def plot_history_individual(histories: dict[int, pd.DataFrame], subject: int,
                             col: str, save_dir: Path) -> None:
    h = histories.get(subject)
    if h is None:
        return
    for key, val_key, ylabel, suffix, is_pct in [
        ("accuracy", "val_accuracy", "Accuracy (%)", "acc",  True),
        ("loss",     "val_loss",     "Loss",          "loss", False),
    ]:
        if key not in h.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        scale = 100 if is_pct else 1
        ax.plot(h[key] * scale, label="Train", color=PALETTE[subject - 1], linewidth=2)
        if val_key in h.columns:
            ax.plot(h[val_key] * scale, label="Val", color=PALETTE[subject - 1],
                    linewidth=2, linestyle="--", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
        if is_pct: ax.set_ylim(0, 1.05)
        fname = save_dir / f"history_s{subject:02d}_{suffix}.png"
        _save(fig, str(fname), fname.name)


def plot_wgan_losses_all(logs: dict[int, pd.DataFrame], save_path: str) -> None:
    if not logs:
        print("  [skip] WGAN losses overlay — no logs")
        return
    fig, axes = plt.subplots(2, N_CLASSES, figsize=(5 * N_CLASSES, 8))
    for cls in range(N_CLASSES):
        for s, log in logs.items():
            df = log[log["class_idx"] == cls]
            if df.empty:
                continue
            col = PALETTE[s - 1]
            axes[0, cls].plot(df["epoch"], df["critic_loss"],
                              label=f"S{s}", color=col, linewidth=1.2, alpha=0.8)
            axes[1, cls].plot(
                df["epoch"],
                df["generator_loss"],
                label=f"S{s}",
                color=col,
                linewidth=1.2,
                alpha=0.8,
            )
        for ax in axes[:, cls]:
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.25)
    plt.tight_layout(); _save(fig, save_path, "wgan_losses_all.png")


def plot_wgan_final_heatmap(logs: dict[int, pd.DataFrame],
                            loss_col: str, title: str, save_path: str) -> None:
    if not logs:
        print(f"  [skip] WGAN {title} heatmap — no logs")
        return
    subjects = sorted(logs.keys())
    mat = np.full((len(subjects), N_CLASSES), np.nan)
    for i, s in enumerate(subjects):
        for cls in range(N_CLASSES):
            df = logs[s][logs[s]["class_idx"] == cls]
            if not df.empty:
                mat[i, cls] = df[loss_col].iloc[-1]
    df_heat = pd.DataFrame(mat,
                           index=[f"S{s}" for s in subjects],
                           columns=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, max(4, len(subjects) * 0.55 + 1.5)))
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm_r",
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("MI Class"); ax.set_ylabel("Subject")
    plt.tight_layout(); _save(fig, save_path, Path(save_path).name)


def plot_wgan_individual(logs: dict[int, pd.DataFrame],
                         subject: int, cls: int, save_dir: Path) -> None:
    log = logs.get(subject)
    if log is None:
        return
    df = log[log["class_idx"] == cls]
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["critic_loss"], color=PALETTE[subject - 1], linewidth=1.5)
    axes[0].set_ylabel("Critic Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["epoch"], df["generator_loss"], color=PALETTE_CLASSES[cls], linewidth=1.5)
    axes[1].set_ylabel("Generator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fname = save_dir / f"wgan_s{subject:02d}_class{cls}.png"
    _save(fig, str(fname), fname.name)


def plot_synthetic_samples(subject: int, save_dir: Path) -> None:
    """3×3 grid of random CWT scalograms (3 real + 3 synthetic per class pair)."""
    data = load_synthetic(subject)
    if data is None:
        return
    X_syn, y_syn = data

    try:
        real_path = SYNTHETIC_DIR.parent.parent / "metrics" / f"cnn_metrics_s{subject:02d}.json"
        n_show = 3
        fig, axes = plt.subplots(N_CLASSES, n_show, figsize=(n_show * 3.5, N_CLASSES * 3.2))
        for cls in range(N_CLASSES):
            idx = np.where(y_syn == cls)[0]
            rng = np.random.RandomState(cls)
            chosen = rng.choice(idx, size=min(n_show, len(idx)), replace=False)
            for col, ci in enumerate(chosen):
                img = X_syn[ci, :, :, 0]  
                ax  = axes[cls, col]
                ax.imshow(
                    img,
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    interpolation="nearest",
                )
                ax.axis("off")
        plt.tight_layout()
        fname = save_dir / f"synthetic_samples_s{subject:02d}.png"
        _save(fig, str(fname), fname.name)
    except Exception as e:
        print(f"  [warn] synthetic samples S{subject}: {e}")


def plot_synthetic_class_mean(subject: int, save_dir: Path) -> None:
    """Mean CWT scalogram per class for synthetic data."""
    data = load_synthetic(subject)
    if data is None:
        return
    X_syn, y_syn = data
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(N_CLASSES * 4, 4))
    for cls in range(N_CLASSES):
        idx  = np.where(y_syn == cls)[0]
        mean = X_syn[idx].mean(axis=0)[:, :, 0]  
        ax   = axes[cls]
        im = ax.imshow(
            mean, aspect="auto", origin="lower", cmap="hot", interpolation="nearest"
        )
        ax.set_xlabel("Time"); ax.set_ylabel("Scale")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fname = save_dir / f"synthetic_class_mean_s{subject:02d}.png"
    _save(fig, str(fname), fname.name)


def plot_synthetic_distribution(subjects: List[int], save_path: str) -> None:
    """Histogram of all synthetic CWT pixel values across subjects."""
    all_vals = []
    for s in subjects:
        d = load_synthetic(s)
        if d is not None:
            all_vals.append(d[0].ravel())
    if not all_vals:
        print("  [skip] synthetic distribution — no data")
        return
    vals = np.concatenate(all_vals)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(vals, bins=100, color="#2196F3", alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("Normalised CWT magnitude"); axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(vals[::100])   # subsample for speed
    xs  = np.linspace(vals.min(), vals.max(), 300)
    axes[1].plot(xs, kde(xs), color="#9C27B0", linewidth=2)
    axes[1].fill_between(xs, kde(xs), alpha=0.3, color="#9C27B0")
    axes[1].set_xlabel("Normalised CWT magnitude"); axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); _save(fig, save_path, "synthetic_distribution.png")


def save_summary_table(df: pd.DataFrame, subjects: List[int]) -> None:
    out = df.copy()
    for m in METRICS:
        out[m] = (out[m] * 100).round(2)
    stats = pd.DataFrame([{
        "subject": "─", "label": "Mean",
        **{m: out[m].mean().round(2) for m in METRICS},
    }, {
        "subject": "─", "label": "Std",
        **{m: out[m].std().round(2)  for m in METRICS},
    }, {
        "subject": "─", "label": "Min",
        **{m: out[m].min().round(2)  for m in METRICS},
    }, {
        "subject": "─", "label": "Max",
        **{m: out[m].max().round(2)  for m in METRICS},
    }])
    full = pd.concat([out, stats], ignore_index=True)
    p = METRICS_DIR / "summary_table.csv"
    full.to_csv(str(p), index=False)
    print(f"  Summary table → {p.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 50+ aggregate figures and tables"
    )
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)))
    args = parser.parse_args()
    subjects = args.subjects

    print(f"\n{'='*65}")
    print(f"  Comprehensive Analysis — {len(subjects)} subjects")
    print(f"{'='*65}\n")

    df        = load_per_subject_metrics(subjects)
    classwise = load_classwise(subjects)
    histories = load_history(subjects)
    wgan_logs = load_wgan_logs(subjects)

    if df.empty:
        print("  No per-subject metrics found — run training first.")
        return

    fig_count = 0

    print("\n── A. Aggregate Summary Figures ──────────────────────────────")
    for m in METRICS:
        plot_single_metric_bar(df, m, str(FIGURES_DIR / f"summary_{m}_bar.png"))
        fig_count += 1
    plot_grouped_metrics(   df, str(FIGURES_DIR / "summary_metrics_grouped.png"));  fig_count += 1
    plot_boxplots(          df, str(FIGURES_DIR / "summary_metrics_boxplot.png"));  fig_count += 1
    plot_violin(            df, str(FIGURES_DIR / "summary_violin.png"));           fig_count += 1
    plot_mean_std_bar(      df, str(FIGURES_DIR / "summary_mean_std.png"));         fig_count += 1
    plot_ranked_accuracy(   df, str(FIGURES_DIR / "summary_ranked.png"));           fig_count += 1
    plot_all_metrics_heatmap(df,str(FIGURES_DIR / "summary_heatmap_all_metrics.png")); fig_count += 1
    plot_scatter_acc_f1(    df, str(FIGURES_DIR / "summary_scatter_acc_f1.png"));   fig_count += 1
    plot_scatter_prec_rec(  df, str(FIGURES_DIR / "summary_scatter_prec_rec.png")); fig_count += 1
    plot_above_chance(      df, str(FIGURES_DIR / "summary_above_chance.png"));     fig_count += 1
    if len(df) > 2:
        plot_corr_matrix(   df, str(FIGURES_DIR / "summary_corr_matrix.png"));     fig_count += 1

    print("\n── B. Radar Charts ─────────────────────────────────────────────")
    plot_radar_individual(df, str(FIGURES_DIR / "radar_chart.png"));         fig_count += 1
    if len(df) > 1:
        plot_radar_overlay(df, str(FIGURES_DIR / "radar_chart_overlay.png")); fig_count += 1

    print("\n── C. Cross-Subject Heatmaps ───────────────────────────────────")
    for m in METRICS:
        mat = load_cross_matrix(subjects, m)
        plot_cross_heatmap(mat, m,
                           str(FIGURES_DIR / f"cross_subject_heatmap_{m}.png"))
        fig_count += 1
    acc_mat = load_cross_matrix(subjects, "accuracy")
    plot_cross_row_mean(acc_mat, str(FIGURES_DIR / "cross_subject_row_mean.png")); fig_count += 1
    plot_cross_col_mean(acc_mat, str(FIGURES_DIR / "cross_subject_col_mean.png")); fig_count += 1

    print("\n── D. Per-Class Figures ────────────────────────────────────────")
    if classwise:
        for mk, title, fname in [
            ("recall",    "Recall",    "class_recall_heatmap.png"),
            ("precision", "Precision", "class_precision_heatmap.png"),
            ("f1-score",  "F1",        "class_f1_heatmap.png"),
        ]:
            plot_class_heatmap(classwise, mk, title,
                               str(FIGURES_DIR / fname)); fig_count += 1
        for s in subjects:
            plot_class_bar_per_subject(classwise, s,
                str(FIGURES_DIR / f"class_bar_s{s:02d}.png")); fig_count += 1
        plot_class_stacked_bar(classwise,
            str(FIGURES_DIR / "class_stacked_bar.png")); fig_count += 1
    else:
        print("  [skip] Per-class figures — no classwise JSON files")

    print("\n── E. CNN Training History ─────────────────────────────────────")
    if histories:
        plot_history_overlay(histories, "accuracy",     "Train Accuracy",
            str(FIGURES_DIR / "history_overlay_acc.png"));      fig_count += 1
        plot_history_overlay(histories, "val_accuracy", "Validation Accuracy",
            str(FIGURES_DIR / "history_overlay_val_acc.png"));  fig_count += 1
        plot_history_loss_overlay(histories, "loss",     "Train Loss",
            str(FIGURES_DIR / "history_overlay_loss.png"));     fig_count += 1
        plot_history_loss_overlay(histories, "val_loss", "Validation Loss",
            str(FIGURES_DIR / "history_overlay_val_loss.png")); fig_count += 1
        plot_history_grid(histories, "accuracy", "val_accuracy",
            "Accuracy", "CNN Training Accuracy — All Subjects",
            str(FIGURES_DIR / "history_grid_acc.png"), is_pct=True);  fig_count += 1
        plot_history_grid(histories, "loss", "val_loss",
            "Loss",     "CNN Training Loss — All Subjects",
            str(FIGURES_DIR / "history_grid_loss.png"), is_pct=False); fig_count += 1
        for s in subjects:
            plot_history_individual(histories, s, "accuracy", FIGURES_DIR)
            fig_count += 2  
    else:
        print("  [skip] CNN history figures — no history CSV files")

    print("\n── F. WGAN Loss Figures ────────────────────────────────────────")
    if wgan_logs:
        plot_wgan_losses_all(wgan_logs,
            str(FIGURES_DIR / "wgan_losses_all.png"));          fig_count += 1
        plot_wgan_final_heatmap(wgan_logs, "critic_loss",    "Critic Loss",
            str(FIGURES_DIR / "wgan_critic_heatmap.png"));      fig_count += 1
        plot_wgan_final_heatmap(wgan_logs, "generator_loss", "Generator Loss",
            str(FIGURES_DIR / "wgan_generator_heatmap.png"));   fig_count += 1
        for s in subjects:
            for cls in range(N_CLASSES):
                plot_wgan_individual(wgan_logs, s, cls, FIGURES_DIR)
                fig_count += 1
    else:
        print("  [skip] WGAN loss figures — no wgan_training_log CSV files")

    print("\n── G. Synthetic Data Visualisations ────────────────────────────")
    for s in subjects:
        plot_synthetic_samples(s, FIGURES_DIR);     fig_count += 1
        plot_synthetic_class_mean(s, FIGURES_DIR);  fig_count += 1
    plot_synthetic_distribution(subjects,
        str(FIGURES_DIR / "synthetic_distribution.png")); fig_count += 1

    print("\n── H. Tables ───────────────────────────────────────────────────")
    save_summary_table(df, subjects)
    cross_csv = METRICS_DIR / "cross_subject_matrix.csv"
    acc_mat.round(2).to_csv(str(cross_csv))
    print(f"  ✓ Cross-subject matrix → {cross_csv.name}")

    actual = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\n{'='*65}")
    print(f"  Analysis complete")
    print(f"  Figures generated this run : ~{fig_count}")
    print(f"  Total PNGs in outputs/figures/ : {actual}")
    print(f"\n  Per-subject results:")
    print(f"  {'Subject':<10} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'─'*30}")
    for _, row in df.sort_values("accuracy", ascending=False).iterrows():
        print(f"  {row['label']:<10} {row['accuracy']*100:>9.2f}%  {row['f1']:>7.3f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
