"""
Repeats the 80/20 within-subject split with 5 different random seeds and
re-runs WGAN-GP + CNN training for each, to check the stability claim in
Sec. 3.1 ("mean within-subject accuracy varied by less than 2%, with a
standard deviation of 1.4%"). Each seed produces its own train/test
partition (independent of the cached split used for the main results),
its own WGAN-GP generator, and its own CNN — this is expensive (5 seeds x
9 subjects x full WGAN-GP + CNN training) and is provided as a script to
run explicitly, not part of the default pipeline.

Usage
-----
  python src/split_stability_check.py --subjects 1 2 3 --seeds 42 1 7 123 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from config import TEST_SPLIT, N_CLASSES, WGAN_BATCH_SIZE, WGAN_EPOCHS, METRICS_DIR
from preprocessing import load_epochs, _to_cwt
from compare_gan_architectures import train_architecture, train_eval_cnn, set_seeds

DEFAULT_SEEDS = [42, 1, 7, 123, 2024]


def run_one_seed(subject: int, seed: int) -> float:
    set_seeds(seed)
    X_raw, y = load_epochs(subject, session="T", verbose=False)

    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SPLIT, random_state=seed, stratify=y,
    )

    X_train = _to_cwt(X_raw[idx_train])
    X_test = _to_cwt(X_raw[idx_test])
    y_train, y_test = y[idx_train], y[idx_test]

    X_syn, y_syn, _elapsed = train_architecture(
        "wgan_gp", X_train, y_train, WGAN_BATCH_SIZE, WGAN_EPOCHS
    )
    return train_eval_cnn(X_train, y_train, X_syn, y_syn, X_test, y_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="80/20 split stability check (Sec. 3.1)")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Split Stability Check  │  {len(args.subjects)} subjects  │  seeds={args.seeds}")
    print(f"  WARNING: re-trains WGAN-GP + CNN for every (subject, seed) pair — expensive.")
    print(f"{'='*60}\n")

    rows = []
    for s in args.subjects:
        for seed in args.seeds:
            print(f"  subject={s:02d}  seed={seed} …", end=" ", flush=True)
            acc = run_one_seed(s, seed)
            print(f"acc={acc*100:.2f}%")
            rows.append({"subject": s, "seed": seed, "accuracy": round(acc, 4)})

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "split_stability_check.csv"
    df.to_csv(str(csv_path), index=False)

    per_subject = df.groupby("subject")["accuracy"].agg(["mean", "std"])
    per_subject_range_pct = (df.groupby("subject")["accuracy"].max()
                             - df.groupby("subject")["accuracy"].min()) * 100

    print(f"\n  Per-subject mean/std across seeds:")
    print(per_subject * 100)
    print(f"\n  Per-subject range (max-min) across seeds, in percentage points:")
    print(per_subject_range_pct)
    print(f"\n  Mean of per-subject std (Sec. 3.1 '1.4% SD' claim): "
          f"{per_subject['std'].mean()*100:.2f}%")
    print(f"  Max of per-subject range (Sec. 3.1 '<2% variation' claim): "
          f"{per_subject_range_pct.max():.2f}%")
    print(f"\n  Saved → {csv_path.name}")


if __name__ == "__main__":
    main()
