"""
Fréchet Inception Distance (FID) for CWT spectrograms.

For non-image GAN outputs we use a feature-space FID:
  1. Flatten (50, 375, 5) CWT arrays → 93 750-dim vectors
  2. PCA-reduce to min(n_samples, 256) dims
  3. Compute Fréchet distance between real and synthetic distributions:
       FID = ‖μ_r − μ_s‖² + Tr(Σ_r + Σ_s − 2·√(Σ_r·Σ_s))

Usage
-----
  python src/fid_score.py                    # all 9 subjects
  python src/fid_score.py --subjects 1 2 3   # specific subjects
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLASS_NAMES, N_CLASSES, SYNTHETIC_DIR,
    METRICS_DIR, FIGURES_DIR,
)
from preprocessing import preprocess_subject


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(fd, 0.0)


def compute_fid(real: np.ndarray, synthetic: np.ndarray,
                n_components: int = 256) -> float:
    r = real.reshape(len(real), -1).astype(np.float64)
    s = synthetic.reshape(len(synthetic), -1).astype(np.float64)
    n_comp = min(n_components, r.shape[0] - 1, s.shape[0] - 1, r.shape[1])
    if n_comp < 2:
        return float("nan")
    pca = PCA(n_components=n_comp, random_state=42)
    r_proj = pca.fit_transform(r)
    s_proj = pca.transform(s)
    mu_r, sigma_r = r_proj.mean(0), np.cov(r_proj, rowvar=False)
    mu_s, sigma_s = s_proj.mean(0), np.cov(s_proj, rowvar=False)
    return frechet_distance(mu_r, sigma_r, mu_s, sigma_s)


def compute_subject_fid(subject: int) -> dict:
    print(f"  Subject {subject:02d} …", end=" ", flush=True)
    X_real, y_real = preprocess_subject(subject, session="T")
    syn_path = SYNTHETIC_DIR / f"synthetic_combined_s{subject:02d}.npz"
    if not syn_path.exists():
        print("synthetic not found — skipping")
        return {}
    data = np.load(str(syn_path))
    X_syn, y_syn = data["X"].astype(np.float32), data["y"].astype(np.int32)
    row: dict = {"subject": subject}
    fids = []
    for c in range(N_CLASSES):
        r = X_real[y_real == c]
        s = X_syn[y_syn == c]
        fid = compute_fid(r, s)
        row[f"fid_c{c}"] = round(fid, 4)
        row[CLASS_NAMES[c]] = round(fid, 4)
        fids.append(fid)
    row["fid_mean"] = round(float(np.nanmean(fids)), 4)
    print(f"mean FID = {row['fid_mean']:.2f}")
    return row


def plot_fid_per_class(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))
    width = 0.2
    palette = sns.color_palette("Set2", N_CLASSES)
    for i, cname in enumerate(CLASS_NAMES):
        if cname in df.columns:
            ax.bar(x + i * width, df[cname], width,
                   label=cname, color=palette[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"S{s:02d}" for s in df["subject"]])
    ax.set_xlabel("Subject"); ax.set_ylabel("FID (lower = better)")
    ax.set_title("Per-Class FID Score: Real vs Synthetic CWT")
    ax.legend(title="Class"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fid_per_class.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  → fid_per_class.png")


def plot_fid_mean(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = sns.color_palette("Blues_d", len(df))
    bars = ax.bar(range(len(df)), df["fid_mean"], color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"S{s:02d}" for s in df["subject"]])
    ax.set_xlabel("Subject"); ax.set_ylabel("Mean FID (lower = better)")
    for bar, val in zip(bars, df["fid_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{val:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "fid_mean_per_subject.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig); print(f"  → fid_mean_per_subject.png")


def plot_fid_heatmap(df: pd.DataFrame) -> None:
    heat = df.set_index("subject")[[c for c in CLASS_NAMES if c in df.columns]]
    heat.index = [f"S{s:02d}" for s in heat.index]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax)
    ax.set_title("FID Score Heatmap (lower = better synthetic quality)")
    ax.set_xlabel("Class"); ax.set_ylabel("Subject")
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "fid_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"  → fid_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID scores")
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)))
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  FID Score Computation  │  {len(args.subjects)} subjects")
    print(f"{'='*55}")

    rows = []
    for s in args.subjects:
        row = compute_subject_fid(s)
        if row:
            rows.append(row)

    if not rows:
        print("  No data found."); return

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "fid_scores.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Saved → {csv_path.name}")

    print(f"\n  Generating figures …")
    plot_fid_per_class(df)
    plot_fid_mean(df)
    plot_fid_heatmap(df)

    print(f"\n  Summary:")
    print(f"  {'Subject':>10}  {'Mean FID':>10}")
    print(f"  {'─'*22}")
    for _, row in df.iterrows():
        print(f"  S{int(row['subject']):02d}        {row['fid_mean']:>10.2f}")
    print(f"  {'─'*22}")
    print(f"  {'Overall':>10}  {df['fid_mean'].mean():>10.2f}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
