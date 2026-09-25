"""
Evaluation utilities — metrics, confusion matrix, and training plots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR


def compute_metrics(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Dict:
    y_pred = model.predict(X, verbose=0).argmax(axis=-1)

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score   (y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score       (y_true, y_pred, average="macro", zero_division=0)),
        "kappa":     float(cohen_kappa_score(y_true, y_pred)),
        "cm":        confusion_matrix(y_true, y_pred, normalize="true"),
        "report":    classification_report(y_true, y_pred,
                                           target_names=CLASS_NAMES,
                                           zero_division=0),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Normalised Confusion Matrix",
) -> None:
    """Save / display a seaborn heatmap of the normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0, vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title(title,              fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  Confusion matrix → {save_path}")
    plt.show()
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    title_prefix: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["accuracy"]) + 1)

    axes[0].plot(epochs, history["accuracy"],     label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_accuracy"], label="Val",   linewidth=2, linestyle="--")
    axes[0].set_title(f"{title_prefix}Model Accuracy", fontsize=12)
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(epochs, history["loss"],     label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_loss"], label="Val",   linewidth=2, linestyle="--")
    axes[1].set_title(f"{title_prefix}Model Loss", fontsize=12)
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  Training history → {save_path}")
    plt.show()
    plt.close()


def plot_wgan_losses(
    log_df: pd.DataFrame,
    subject: int,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-class WGAN-GP critic and generator losses."""
    classes  = sorted(log_df["class_idx"].unique())
    n        = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        df = log_df[log_df["class_idx"] == cls]
        ax.plot(df["epoch"], df["critic_loss"],    label="Critic",    linewidth=1.5)
        ax.plot(df["epoch"], df["generator_loss"], label="Generator", linewidth=1.5)
        ax.set_title(f"Class {cls}: {df['class_name'].iloc[0]}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"WGAN-GP Training Losses — Subject {subject}", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  WGAN loss curves → {save_path}")
    plt.show()
    plt.close()


def build_summary_table(metrics_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Aggregate per-subject JSON results into a single DataFrame.
    """
    if metrics_dir is None:
        metrics_dir = METRICS_DIR

    rows = []
    for fname in sorted(os.listdir(str(metrics_dir))):
        if fname.startswith("cnn_metrics_s") and fname.endswith(".json"):
            with open(os.path.join(str(metrics_dir), fname)) as fh:
                rows.append(json.load(fh))

    if not rows:
        print("  No per-subject result files found in metrics/")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("subject")
    summary_path = str(metrics_dir / "all_subjects_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"  Summary table → {summary_path}")
    return df
