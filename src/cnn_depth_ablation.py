"""
CNN depth ablation (Sec. 3.6.2, Table 9): compares 1, 2, and 3 convolutional
layers with WGAN-GP augmentation, keeping every other hyperparameter fixed
at the values in config.py. Requires that train_wgan.py has already been
run for each subject (synthetic_combined_s##.npz present in outputs/synthetic/).

Usage
-----
  python src/cnn_depth_ablation.py --subjects 1 2 3 4 5 6 7 8 9 --depths 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TRIAL_SHAPE, CNN_FILTERS, CNN_KERNEL_SIZE, CNN_DROPOUT, CNN_L2,
    CNN_DENSE_UNITS, CNN_LR, CNN_EPOCHS, CNN_BATCH_SIZE, N_CLASSES,
    METRICS_DIR, RANDOM_SEED,
)
from preprocessing import preprocess_subject_split
from train_cnn import load_synthetic
from evaluate import compute_metrics


def build_cnn_depth(n_conv_layers: int, input_shape: tuple = TRIAL_SHAPE,
                    n_classes: int = N_CLASSES) -> Model:
    """Same block design as models.cnn.build_cnn, repeated n_conv_layers times."""
    reg = regularizers.l2(CNN_L2)
    inp = layers.Input(shape=input_shape, name="cwt_image")
    x = inp
    for i in range(n_conv_layers):
        x = layers.Conv2D(
            CNN_FILTERS, CNN_KERNEL_SIZE, padding="valid", activation="relu",
            kernel_regularizer=reg, name=f"conv{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i+1}")(x)
        x = layers.Dropout(CNN_DROPOUT, name=f"drop{i+1}")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(CNN_DENSE_UNITS, activation="relu",
                     kernel_regularizer=reg, name="fc1")(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name=f"cnn_depth{n_conv_layers}")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=CNN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_eval(subject: int, n_conv_layers: int) -> float:
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_train_real, y_train_real, X_test, y_test = preprocess_subject_split(subject, session="T", verbose=False)
    X_syn, y_syn = load_synthetic(subject)
    X_train = np.concatenate([X_train_real, X_syn], axis=0)
    y_train = np.concatenate([y_train_real, y_syn], axis=0)

    model = build_cnn_depth(n_conv_layers)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN depth ablation (Table 9)")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  CNN Depth Ablation  │  {len(args.subjects)} subjects  │  depths={args.depths}")
    print(f"{'='*55}\n")

    rows = []
    for depth in args.depths:
        for s in args.subjects:
            print(f"  depth={depth}  subject={s:02d} …", end=" ", flush=True)
            acc = train_eval(s, depth)
            print(f"acc={acc*100:.2f}%")
            rows.append({"depth": depth, "subject": s, "accuracy": round(acc, 4)})

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "cnn_depth_ablation.csv"
    df.to_csv(str(csv_path), index=False)

    print(f"\n  Mean accuracy by depth (Table 9):")
    summary = df.groupby("depth")["accuracy"].agg(["mean", "std"]) * 100
    print(summary)
    print(f"\n  Saved → {csv_path.name}")


if __name__ == "__main__":
    main()
