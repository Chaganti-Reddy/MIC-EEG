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
import tensorflow as tf
from scipy.signal import decimate

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLASS_NAMES, N_CLASSES, METRICS_DIR, FIGURES_DIR,
    CNN_EPOCHS, CNN_BATCH_SIZE, RANDOM_SEED, SAMPLING_RATE,
)
from preprocessing import (
    load_raw, extract_epochs, _get_or_create_split_indices,
    _complex_morlet_cwt, normalise_dataset, _N_RAW_SAMPLES,
)
from models.cnn import build_cnn
from evaluate import compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary band-limited discriminability analysis (Sec. 3.7.2), independent
# of the primary 0.5–20 Hz CWT pipeline used elsewhere in the study
# (Sec. 2.2.6 / config.CWT_FREQS). To represent Beta/Gamma content up to
# 45 Hz without aliasing, this analysis decimates less aggressively than the
# main pipeline (factor 2 instead of 4 → fs' = 125 Hz, Nyquist = 62.5 Hz).
# ─────────────────────────────────────────────────────────────────────────────
BAND_DOWNSAMPLE_FACTOR = 2
BAND_FS = SAMPLING_RATE / BAND_DOWNSAMPLE_FACTOR   # 125 Hz
BAND_N_FREQS = 40

BANDS = {
    "Delta (0.5-4 Hz)": (0.5, 4.0),
    "Theta (4-8 Hz)":   (4.0, 8.0),
    "Alpha (8-13 Hz)":  (8.0, 13.0),
    "Beta (13-30 Hz)":  (13.0, 30.0),
    "Gamma (30-45 Hz)": (30.0, 45.0),
}


def _band_scalogram(trial: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Band-limited complex Morlet scalogram for one raw trial (real part only)."""
    trial = trial[:, :_N_RAW_SAMPLES]
    trial_sub = decimate(trial, BAND_DOWNSAMPLE_FACTOR, ftype="fir",
                         zero_phase=True, axis=-1)
    freqs = np.linspace(lo, hi, BAND_N_FREQS)
    channels = []
    for ch in range(trial_sub.shape[0]):
        coeffs = _complex_morlet_cwt(trial_sub[ch], freqs, BAND_FS)
        channels.append(np.real(coeffs).astype(np.float32))
    return np.stack(channels, axis=-1).astype(np.float32)   # (BAND_N_FREQS, T_band, C)


def band_dataset(X_raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    imgs = [_band_scalogram(t, lo, hi) for t in X_raw]
    return normalise_dataset(np.stack(imgs, axis=0))


def probe_band_accuracy(X_raw: np.ndarray, y: np.ndarray,
                        idx_train: np.ndarray, idx_test: np.ndarray,
                        lo: float, hi: float) -> float:
    """
    Train a CNN on the band-limited scalogram using the SAME within-subject
    80/20 partitioning and model-selection protocol described in Sec. 3.1
    (held-out partition monitors validation loss for early stopping/LR decay).
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    X_train = band_dataset(X_raw[idx_train], lo, hi)
    X_test  = band_dataset(X_raw[idx_test],  lo, hi)
    y_train, y_test = y[idx_train], y[idx_test]

    model = build_cnn(input_shape=X_train.shape[1:])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )
    return compute_metrics(model, X_test.astype(np.float32), y_test)["accuracy"]


def analyse_subject(subject: int) -> dict:
    print(f"  Subject {subject:02d} …", flush=True)
    raw = load_raw(subject, session="T")
    X_raw, y = extract_epochs(raw)      # (N, 5, 1500) @ 250 Hz, before any CWT
    idx_train, idx_test = _get_or_create_split_indices(subject, "T", y)

    row: dict = {"subject": subject}
    for band_name, (lo, hi) in BANDS.items():
        acc = probe_band_accuracy(X_raw, y, idx_train, idx_test, lo, hi)
        print(f"    {band_name:22s}  freqs={BAND_N_FREQS:2d}  acc={acc:.3f}")
        row[band_name] = round(acc, 4)
    return row


def plot_band_accuracy_heatmap(df: pd.DataFrame) -> None:
    band_cols = [c for c in df.columns if c != "subject"]
    heat = df.set_index("subject")[band_cols]
    heat.index = [f"S{s:02d}" for s in heat.index]
    heat.columns = [c.split(" ")[0] for c in heat.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGn",
                vmin=0.25, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_title("Band Discriminability — CNN Held-Out Accuracy\n(chance = 0.25)")
    ax.set_xlabel("Frequency Band"); ax.set_ylabel("Subject")
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "band_accuracy_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig); print(f"  → band_accuracy_heatmap.png")


def plot_band_accuracy_lines(df: pd.DataFrame) -> None:
    band_cols = [c for c in df.columns if c != "subject"]
    short = [c.split(" ")[0] for c in band_cols]
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(short, [row[b] for b in band_cols],
                marker="o", label=f"S{int(row['subject']):02d}",
                color=palette[i], alpha=0.8)
    ax.axhline(0.25, ls="--", color="gray", alpha=0.5, label="Chance")
    ax.set_xlabel("Frequency Band"); ax.set_ylabel("Held-Out Accuracy (CNN)")
    ax.set_title("Frequency Band Discriminability per Subject")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "band_accuracy_lines.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig); print(f"  → band_accuracy_lines.png")


def plot_mean_band_bar(df: pd.DataFrame) -> None:
    band_cols = [c for c in df.columns if c != "subject"]
    means = df[band_cols].mean()
    stds  = df[band_cols].std()
    short = [c.split(" ")[0] for c in band_cols]
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("viridis", len(band_cols))
    ax.bar(short, means, yerr=stds, capsize=4, color=palette, alpha=0.85, ecolor="black")
    ax.axhline(0.25, ls="--", color="gray", alpha=0.6, label="Chance (25%)")
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Mean Held-Out Accuracy ± std (9 subjects)")
    ax.set_title("Average Discriminability by Frequency Band")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "band_accuracy_mean.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig); print(f"  → band_accuracy_mean.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency band analysis")
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)))
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Frequency Band Analysis  │  {len(args.subjects)} subjects")
    print(f"  Auxiliary analysis, independent of the primary 0.5-20 Hz CWT")
    print(f"  pipeline. Decimation factor {BAND_DOWNSAMPLE_FACTOR} "
          f"(fs' = {BAND_FS:.1f} Hz), {BAND_N_FREQS} freq. bins per band:")
    for band_name, (lo, hi) in BANDS.items():
        print(f"    {band_name:22s}  range {lo}-{hi} Hz")
    print(f"{'='*55}\n")

    rows = []
    for s in args.subjects:
        row = analyse_subject(s)
        if row:
            rows.append(row)

    if not rows:
        print("  No data found."); return

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "band_analysis.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Saved → {csv_path.name}")

    print(f"\n  Generating figures …")
    plot_band_accuracy_heatmap(df)
    plot_band_accuracy_lines(df)
    plot_mean_band_bar(df)
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
