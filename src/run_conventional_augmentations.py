from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CNN_EPOCHS, CNN_BATCH_SIZE,
    N_CLASSES, RANDOM_SEED, METRICS_DIR,
)
from preprocessing import preprocess_subject_split
from augmentation import rotate_180, shift, add_gaussian_noise
from models.cnn import build_cnn
from evaluate import compute_metrics


def train_augmented_model(subject: int, aug_type: str) -> float:
    """Train CNN on real + conventionally augmented data and return test accuracy."""
    X_train_real, y_train_real, X_test, y_test = preprocess_subject_split(
        subject, session="T", verbose=False
    )

    X_aug_list = []
    y_aug_list = []
    for c in range(N_CLASSES):
        X_class = X_train_real[y_train_real == c]
        n_samples = len(X_class)
        if n_samples == 0:
            continue
        indices = np.random.choice(n_samples, 100, replace=True)
        class_samples = X_class[indices]
        if aug_type == "noise":
            aug_samples = np.stack([add_gaussian_noise(s) for s in class_samples])
        elif aug_type == "rotate":
            aug_samples = np.stack([rotate_180(s) for s in class_samples])
        elif aug_type == "shift":
            aug_samples = []
            for s in class_samples:
                dy = int(np.random.randint(-10, 11))
                dx = int(np.random.randint(-20, 21))
                aug_samples.append(shift(s, dy, dx))
            aug_samples = np.stack(aug_samples)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
        X_aug_list.append(aug_samples)
        y_aug_list.append(np.full((100,), c, dtype=np.int32))
    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    X_train_combined = np.concatenate([X_train_real, X_aug], axis=0)
    y_train_combined = np.concatenate([y_train_real, y_aug], axis=0)
    model = build_cnn()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=0,
        ),
    ]
    model.fit(
        X_train_combined, y_train_combined,
        validation_data=(X_test, y_test),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    metrics = compute_metrics(model, X_test.astype(np.float32), y_test)
    return float(metrics["accuracy"])

def main() -> None:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    subjects = list(range(1, 10))
    methods = ["noise", "rotate", "shift"]
    
    results = {}
    for m in methods:
        results[m] = []
        
    print("Starting conventional data augmentation training...")
    for s in subjects:
        print(f"\n--- Subject {s:02d} ---")
        for m in methods:
            acc = train_augmented_model(s, m)
            results[m].append(acc)
            print(f"  {m:<10}: {acc*100:.2f}%")
            
    summary = {}
    for m in methods:
        accs = np.array(results[m]) * 100
        summary[m] = {
            "raw": accs.tolist(),
            "mean": float(accs.mean()),
            "std": float(accs.std()),
        }
        print(f"\nMethod {m} across subjects: Mean={accs.mean():.2f}%, Std={accs.std():.2f}%")
        
    out_path = METRICS_DIR / "conventional_ablation.json"
    with open(str(out_path), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
