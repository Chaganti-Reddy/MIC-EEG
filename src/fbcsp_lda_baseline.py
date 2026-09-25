"""
Filter Bank Common Spatial Patterns (FBCSP) + LDA baseline, evaluated under
the same leakage-controlled 80/20 within-subject split used throughout the
study (Sec. 3.1). Used as a conventional-classifier sanity check for the
real-only CNN baseline (Sec. 3.6, Table 8 discussion: "38.4% ± 4.1%").

CSP is fit on the training partition only, per sub-band; the held-out
partition is transformed with the fitted filters and never seen during
CSP or LDA fitting.

Usage
-----
  python src/fbcsp_lda_baseline.py --subjects 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).parent))

from config import METRICS_DIR, SAMPLING_RATE, RANDOM_SEED
from preprocessing import load_raw, extract_epochs, _get_or_create_split_indices

mne.set_log_level("WARNING")

FREQ_BANDS = [(lo, lo + 4) for lo in range(4, 40, 4)]   # 4-8, 8-12, ..., 36-40 Hz
N_CSP_COMPONENTS = 4
N_TOP_FEATURES = 16


def band_features(X: np.ndarray, y_train: np.ndarray,
                  idx_train: np.ndarray, idx_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one CSP filter bank per sub-band on the training partition only,
    then transform both partitions into concatenated log-variance features."""
    train_feats, test_feats = [], []

    for lo, hi in FREQ_BANDS:
        X_filt = mne.filter.filter_data(
            X.astype(np.float64), sfreq=SAMPLING_RATE, l_freq=lo, h_freq=hi,
            method="fir", verbose=False,
        )
        csp = CSP(n_components=N_CSP_COMPONENTS, reg="ledoit_wolf",
                 log=True, norm_trace=False)
        csp.fit(X_filt[idx_train], y_train)
        train_feats.append(csp.transform(X_filt[idx_train]))
        test_feats.append(csp.transform(X_filt[idx_test]))

    return np.concatenate(train_feats, axis=1), np.concatenate(test_feats, axis=1)


def evaluate_subject(subject: int) -> dict:
    print(f"  Subject {subject:02d} …", flush=True)
    raw = load_raw(subject, session="T")
    X, y = extract_epochs(raw)   # (N, 5, 1500) @ 250 Hz, before any CWT
    idx_train, idx_test = _get_or_create_split_indices(subject, "T", y)
    y_train, y_test = y[idx_train], y[idx_test]

    F_train, F_test = band_features(X, y_train, idx_train, idx_test)

    # Feature selection (mutual information, fit on training partition only)
    mi = mutual_info_classif(F_train, y_train, random_state=RANDOM_SEED)
    top_idx = np.argsort(mi)[::-1][:min(N_TOP_FEATURES, F_train.shape[1])]
    F_train_sel, F_test_sel = F_train[:, top_idx], F_test[:, top_idx]

    clf = LinearDiscriminantAnalysis()
    clf.fit(F_train_sel, y_train)
    acc = clf.score(F_test_sel, y_test)
    print(f"    accuracy = {acc*100:.2f}%")
    return {"subject": subject, "accuracy": round(acc, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="FBCSP + LDA baseline")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  FBCSP + LDA Baseline  │  {len(args.subjects)} subjects")
    print(f"  {len(FREQ_BANDS)} sub-bands × {N_CSP_COMPONENTS} CSP components, "
          f"top-{N_TOP_FEATURES} MI-selected features → LDA")
    print(f"{'='*55}\n")

    rows = [evaluate_subject(s) for s in args.subjects]
    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "fbcsp_lda_baseline.csv"
    df.to_csv(str(csv_path), index=False)

    mean_acc = df["accuracy"].mean() * 100
    std_acc = df["accuracy"].std() * 100
    print(f"\n  Mean accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%  (Table 8 discussion)")
    print(f"  Saved → {csv_path.name}")


if __name__ == "__main__":
    main()
