from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_DIR, FIGURES_DIR

PALETTE = sns.color_palette("tab10", 9)


# ── Training wrapper ──────────────────────────────────────────────────────────

def run_nosyn_training(subject: int) -> bool:
    """Train real-only CNN for one subject (writes _nosyn suffixed files)."""
    # Skip if already done
    out = METRICS_DIR / f"cnn_metrics_s{subject:02d}_nosyn.json"
    if out.exists():
        print(f"    [S{subject:02d}] _nosyn metrics already exist — skipping")
        return True
    cmd = [sys.executable, str(Path(__file__).parent / "train_cnn.py"),
           "--subject", str(subject), "--no_synthetic"]
    print(f"    [S{subject:02d}] training real-only CNN …")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"    [WARN] S{subject:02d} nosyn training failed")
        return False
    return True


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_metrics(subject: int, suffix: str = "") -> dict | None:
    path = METRICS_DIR / f"cnn_metrics_s{subject:02d}{suffix}.json"
    if not path.exists():
        return None
    with open(str(path)) as fh:
        return json.load(fh)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_grouped_bar(df: pd.DataFrame) -> None:
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, df["acc_nosyn"] * 100, w,
                label="Real only",        color="#FF7043", alpha=0.85)
    b2 = ax.bar(x + w/2, df["acc_syn"]   * 100, w,
                label="Real + Synthetic", color="#42A5F5", alpha=0.85)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(25, ls=":", color="gray", lw=1.4, label="Chance (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s):02d}" for s in df["subject"]])
    ax.set_xlabel("Subject"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Real-Only vs Real + WGAN-GP Synthetic",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "ablation_accuracy_grouped.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig); print(f"  → ablation_accuracy_grouped.png")


def plot_delta(df: pd.DataFrame) -> None:
    delta = (df["acc_syn"] - df["acc_nosyn"]) * 100
    colors = ["#4CAF50" if d >= 0 else "#F44336" for d in delta]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(df)), delta, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0, color="black", lw=1.2)
    ax.axhline(delta.mean(), color="navy", ls="--", lw=1.5,
               label=f"Mean delta = {delta.mean():+.2f}%")
    for bar, d in zip(bars, delta):
        va = "bottom" if d >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.2 if d >= 0 else -0.2),
                f"{d:+.1f}%", ha="center", va=va, fontsize=9)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"S{int(s):02d}" for s in df["subject"]])
    ax.set_xlabel("Subject"); ax.set_ylabel("Accuracy improvement (%)")
    ax.set_title("Accuracy Delta: (Real + Synthetic) − Real-Only",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "ablation_delta.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  → ablation_delta.png")


def plot_f1(df: pd.DataFrame) -> None:
    x, w = np.arange(len(df)), 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, df["f1_nosyn"] * 100, w,
           label="Real only",        color="#FF7043", alpha=0.85)
    ax.bar(x + w/2, df["f1_syn"]   * 100, w,
           label="Real + Synthetic", color="#42A5F5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(s):02d}" for s in df["subject"]])
    ax.set_xlabel("Subject"); ax.set_ylabel("F1 Score (%)")
    ax.set_title("Ablation Study: F1 Score — Real-Only vs Real + Synthetic",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "ablation_f1_grouped.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig); print(f"  → ablation_f1_grouped.png")


def plot_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(df["acc_nosyn"].min(), df["acc_syn"].min()) * 100 - 5
    hi = max(df["acc_nosyn"].max(), df["acc_syn"].max()) * 100 + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="No change")
    for i, row in df.iterrows():
        r = row["acc_nosyn"] * 100
        s = row["acc_syn"]   * 100
        ax.scatter(r, s, color=PALETTE[i % 9], s=100, zorder=5)
        ax.annotate(f"S{int(row['subject']):02d}", (r + 0.3, s + 0.3), fontsize=8)
    ax.set_xlabel("Real-Only Accuracy (%)")
    ax.set_ylabel("Real + Synthetic Accuracy (%)")
    ax.set_title("Ablation: Synthetic Augmentation Effect\n(above diagonal = improvement)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "ablation_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  → ablation_scatter.png")


def print_table(df: pd.DataFrame) -> None:
    print(f"\n  {'Subject':>8}  {'RealOnly%':>10}  {'Real+Syn%':>10}  "
          f"{'Delta%':>9}  {'F1 RO':>7}  {'F1 R+S':>7}")
    print(f"  {'─'*60}")
    for _, row in df.iterrows():
        delta = (row["acc_syn"] - row["acc_nosyn"]) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  S{int(row['subject']):02d}       "
              f"{row['acc_nosyn']*100:>10.2f}  "
              f"{row['acc_syn']*100:>10.2f}  "
              f"{sign}{delta:>8.2f}%  "
              f"{row['f1_nosyn']*100:>7.2f}  "
              f"{row['f1_syn']*100:>7.2f}")
    print(f"  {'─'*60}")
    delta_all = (df["acc_syn"] - df["acc_nosyn"]) * 100
    sign = "+" if delta_all.mean() >= 0 else ""
    print(f"  {'Mean':>8}  {df['acc_nosyn'].mean()*100:>10.2f}  "
          f"{df['acc_syn'].mean()*100:>10.2f}  "
          f"{sign}{delta_all.mean():>8.2f}%  "
          f"{df['f1_nosyn'].mean()*100:>7.2f}  "
          f"{df['f1_syn'].mean()*100:>7.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study orchestrator")
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)))
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip training, regenerate figures from existing JSONs")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Ablation Study  │  {len(args.subjects)} subjects")
    print(f"  Mode: {'analyze_only' if args.analyze_only else 'train + analyze'}")
    print(f"  NOTE: existing cnn_metrics_s0N.json are NOT touched")
    print(f"{'='*60}\n")

    if not args.analyze_only:
        for s in args.subjects:
            run_nosyn_training(s)

    # Collect metrics
    rows = []
    for s in args.subjects:
        m_nosyn = load_metrics(s, suffix="_nosyn")
        m_syn   = load_metrics(s, suffix="")
        if m_nosyn is None or m_syn is None:
            missing = []
            if m_nosyn is None: missing.append(f"cnn_metrics_s{s:02d}_nosyn.json")
            if m_syn   is None: missing.append(f"cnn_metrics_s{s:02d}.json")
            print(f"  [warn] S{s:02d}: missing {', '.join(missing)} — skipping")
            continue
        rows.append({
            "subject":    s,
            "acc_nosyn":  m_nosyn["accuracy"],
            "acc_syn":    m_syn["accuracy"],
            "f1_nosyn":   m_nosyn["f1"],
            "f1_syn":     m_syn["f1"],
            "prec_nosyn": m_nosyn["precision"],
            "prec_syn":   m_syn["precision"],
            "rec_nosyn":  m_nosyn["recall"],
            "rec_syn":    m_syn["recall"],
        })

    if not rows:
        print("  No paired metrics found. Run without --analyze_only first.")
        return

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "ablation_summary.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Saved → {csv_path.name}")

    print_table(df)

    print(f"\n  Generating figures …")
    plot_grouped_bar(df)
    plot_delta(df)
    plot_f1(df)
    plot_scatter(df)
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
