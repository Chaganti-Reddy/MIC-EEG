from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CNN_EPOCHS, CNN_BATCH_SIZE, CNN_VAL_SPLIT,
    SYNTHETIC_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
    N_CLASSES, CLASS_NAMES, RANDOM_SEED,
)
from preprocessing import preprocess_subject, preprocess_subject_split
from models.cnn import build_cnn
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history,
)

def load_synthetic(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Load combined synthetic data produced by train_wgan.py."""
    path = SYNTHETIC_DIR / f"synthetic_combined_s{subject:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found: {path}\n"
            f"Run `python src/train_wgan.py --subject {subject}` first."
        )
    data = np.load(str(path))
    return data["X"].astype(np.float32), data["y"].astype(np.int32)


def main(args: argparse.Namespace) -> None:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    subject = args.subject
    no_syn = getattr(args, "no_synthetic", False)
    suffix = "_nosyn" if no_syn else ""

    print(f"\n{'='*60}")
    print(
        f"  CNN Training  │  Subject {subject:02d}  "
        f"[{'real-only' if no_syn else 'real + synthetic'}]"
    )
    print(f"  Epochs: {args.epochs}   Batch: {CNN_BATCH_SIZE}")
    print(f"{'='*60}")

    X_train_real, y_train_real, X_test, y_test = preprocess_subject_split(subject, session="T")

    if no_syn:
        X_train, y_train = X_train_real, y_train_real
        print(f"  Real only (train) : {X_train_real.shape[0]} trials")
    else:
        X_syn, y_syn = load_synthetic(subject)
        X_train = np.concatenate([X_train_real, X_syn], axis=0)
        y_train = np.concatenate([y_train_real, y_syn], axis=0)
        print(f"  Real (train)  : {X_train_real.shape[0]} trials")
        print(
            f"  Synthetic     : {X_syn.shape[0]}  samples  "
            f"({X_syn.shape[0] // N_CLASSES} per class)"
        )
        print(f"  Combined train: {X_train.shape[0]} total  shape={X_train.shape[1:]}")

    print(f"  Train: {len(X_train)}  Test (held-out, real-only): {len(X_test)}")
    X_val, y_val = X_test, y_test

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_cnn()

    model.summary(line_length=80)

    ckpt = MODELS_DIR / f"cnn_s{subject:02d}{suffix}_best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n  ── In-subject evaluation (Subject {subject}) ──")
    metrics = compute_metrics(model, X_val.astype(np.float32), y_val)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Kappa     : {metrics['kappa']:.4f}")
    print(f"\n{metrics['report']}")

    plot_confusion_matrix(
        metrics["cm"],
        CLASS_NAMES,
        save_path=str(FIGURES_DIR / f"cm_s{subject:02d}{suffix}.png"),
        title=f"Subject {subject} — Confusion Matrix",
    )
    plot_training_history(
        history.history,
        save_path=str(FIGURES_DIR / f"history_s{subject:02d}{suffix}.png"),
        title_prefix=f"Subject {subject}{' (no synthetic)' if suffix else ''} — ",
    )

    # ── Cross-subject evaluation ──────────────────────────────────────────────
    cross_metrics: dict = {}
    if args.cross_test and args.cross_test != subject:
        ts = args.cross_test
        print(f"\n  ── Cross-subject: train on S{subject} → test on S{ts} ──")
        X_cross, y_cross = preprocess_subject(ts, session="T")
        cross_metrics = compute_metrics(model, X_cross.astype(np.float32), y_cross)
        print(f"  Cross Accuracy : {cross_metrics['accuracy']:.4f}  "
              f"({cross_metrics['accuracy']*100:.2f}%)")
        print(f"  Cross F1-Score : {cross_metrics['f1']:.4f}")
        plot_confusion_matrix(
            cross_metrics["cm"], CLASS_NAMES,
            save_path=str(FIGURES_DIR / f"cm_s{subject:02d}_to_s{ts:02d}.png"),
            title=f"S{subject}→S{ts} Cross-Subject Confusion Matrix",
        )

    # ── Persist metrics ───────────────────────────────────────────────────────
    results = {
        "subject":   subject,
        "accuracy":  metrics["accuracy"],
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "f1":        metrics["f1"],
        "kappa":     metrics["kappa"],
    }
    if cross_metrics:
        results["cross_subject"] = args.cross_test
        results["cross_accuracy"] = cross_metrics["accuracy"]
        results["cross_f1"]       = cross_metrics["f1"]
        results["cross_kappa"]    = cross_metrics["kappa"]

    json_path = METRICS_DIR / f"cnn_metrics_s{subject:02d}{suffix}.json"
    with open(str(json_path), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Metrics JSON → {json_path.name}")

    from sklearn.metrics import classification_report as _cr
    import sklearn.metrics as _skm
    y_pred_all = model.predict(X_val.astype(np.float32), verbose=0).argmax(axis=-1)
    cr_dict    = _cr(y_val, y_pred_all, target_names=CLASS_NAMES,
                     output_dict=True, zero_division=0)
    classwise  = {cn: cr_dict[cn] for cn in CLASS_NAMES}
    cw_path = METRICS_DIR / f"cnn_metrics_s{subject:02d}{suffix}_classwise.json"
    with open(str(cw_path), "w") as fh:
        json.dump(classwise, fh, indent=2)
    print(f"  Class-wise JSON → {cw_path.name}")

    hist_csv = METRICS_DIR / f"cnn_history_s{subject:02d}{suffix}.csv"
    pd.DataFrame(history.history).to_csv(str(hist_csv), index=False)
    print(f"  History CSV  → {hist_csv.name}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN classifier on real + synthetic MI-EEG data"
    )
    parser.add_argument("--subject",    type=int, default=1,
                        help="Subject to train on (1–9)")
    parser.add_argument("--epochs",     type=int, default=CNN_EPOCHS,
                        help=f"Max training epochs (default: {CNN_EPOCHS})")
    parser.add_argument("--cross_test", type=int, default=None,
                        help="Subject index for cross-subject evaluation")
    parser.add_argument(
        "--no_synthetic",
        action="store_true",
        help="Train on real data only (ablation baseline)",
    )
    args = parser.parse_args()
    main(args)
