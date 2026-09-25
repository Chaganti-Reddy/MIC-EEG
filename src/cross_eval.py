"""
Cross-subject evaluation — load trained CNN for train_subject, evaluate
on a different test_subject, and save results to metrics/.

Usage
-----
  python src/cross_eval.py --train_subject 1 --test_subject 2

Prerequisites
  train_cnn.py must have been run for train_subject.

Output
------
  metrics/cross_s01_to_s02.json
  outputs/figures/cm_s01_to_s02.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODELS_DIR, METRICS_DIR, FIGURES_DIR,
    CLASS_NAMES, N_CLASSES, RANDOM_SEED,
)
from preprocessing import preprocess_subject
from models.cnn import build_cnn
from evaluate import compute_metrics, plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-subject evaluation: train on one subject, test on another"
    )
    parser.add_argument("--train_subject", type=int, required=True,
                        help="Subject whose trained CNN weights to load (1-9)")
    parser.add_argument("--test_subject",  type=int, required=True,
                        help="Subject to evaluate on (1-9)")
    args = parser.parse_args()

    ts = args.train_subject
    xs = args.test_subject

    print(f"\n  Cross-subject: S{ts} (train) → S{xs} (test)")

    weights_path = MODELS_DIR / f"cnn_s{ts:02d}_best.weights.h5"
    if not weights_path.exists():
        print(f"  Weights not found: {weights_path}")
        print(f"  Run  python src/train_cnn.py --subject {ts}  first.")
        sys.exit(1)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = build_cnn()
    dummy = np.zeros((1, 50, 375, 5), dtype=np.float32)
    model(dummy, training=False)
    model.load_weights(str(weights_path))
    print(f"  Loaded weights from {weights_path.name}")

    X_test, y_test = preprocess_subject(xs, session="T")
    print(f"  Test trials: {X_test.shape[0]}")

    metrics = compute_metrics(model, X_test.astype(np.float32), y_test)

    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Kappa     : {metrics['kappa']:.4f}")
    print(f"\n{metrics['report']}")

    plot_confusion_matrix(
        metrics["cm"], CLASS_NAMES,
        save_path=str(FIGURES_DIR / f"cm_s{ts:02d}_to_s{xs:02d}.png"),
        title=f"Cross-Subject S{ts}→S{xs}  (acc={metrics['accuracy']*100:.1f}%)",
    )

    result = {
        "train_subject": ts,
        "test_subject":  xs,
        "accuracy":      metrics["accuracy"],
        "precision":     metrics["precision"],
        "recall":        metrics["recall"],
        "f1":            metrics["f1"],
        "kappa":         metrics["kappa"],
    }
    out_path = METRICS_DIR / f"cross_s{ts:02d}_to_s{xs:02d}.json"
    with open(str(out_path), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved → {out_path.name}")


if __name__ == "__main__":
    main()
