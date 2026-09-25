from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
tf.config.optimizer.set_jit(False)
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"  GPU enabled: {[g.name for g in _gpus]}  |  mixed_float16 on")
else:
    print("  No GPU found — running on CPU")

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    WGAN_EPOCHS, WGAN_BATCH_SIZE, LATENT_DIM,
    WGAN_LR, WGAN_BETA1, WGAN_BETA2,
    N_CRITIC, GP_LAMBDA,
    N_SYNTHETIC_PER_CLASS, N_CLASSES, CLASS_NAMES,
    SYNTHETIC_DIR, MODELS_DIR, METRICS_DIR,
    RANDOM_SEED,
)
from preprocessing import preprocess_subject_split, split_by_class
from models.wgan_gp import build_generator, build_critic, WGANGP


def set_seeds(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def train_wgan_class(
    X_class:   np.ndarray,
    class_idx: int,
    subject:   int,
    epochs:    int,
    log_rows:  list,
) -> np.ndarray:
    """
    Train WGAN-GP on CWT images from a single MI class.

    Parameters
    ----------
    X_class   : ndarray  (N, 50, 375, 5)   real samples for this class
    class_idx : int      0-indexed class
    subject   : int      subject number
    epochs    : int      number of training epochs
    log_rows  : list     accumulates per-epoch metric dicts

    Returns
    -------
    synthetic : ndarray  (N_SYNTHETIC_PER_CLASS, 50, 375, 5)
    """
    cname = CLASS_NAMES[class_idx]
    print(f"\n{'─'*60}")
    print(f"  Subject {subject:02d}  |  Class {class_idx}: {cname}")
    print(f"  Real samples: {len(X_class)}   Epochs: {epochs}")
    print(f"{'─'*60}")

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(X_class.astype(np.float32))
        .shuffle(buffer_size=len(X_class) * 3, seed=RANDOM_SEED)
        .batch(min(WGAN_BATCH_SIZE, len(X_class)), drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    gen    = build_generator(LATENT_DIM)
    critic = build_critic()
    wgan   = WGANGP(gen, critic,
                    latent_dim=LATENT_DIM,
                    n_critic=N_CRITIC,
                    gp_lambda=GP_LAMBDA)
    wgan.compile(
        g_optimizer=tf.keras.optimizers.legacy.Adam(WGAN_LR, WGAN_BETA1, WGAN_BETA2),
        c_optimizer=tf.keras.optimizers.legacy.Adam(WGAN_LR, WGAN_BETA1, WGAN_BETA2),
    )

    train_step_fn = tf.function(wgan.train_step)

    class_t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        c_vals, g_vals = [], []

        for batch in dataset:
            metrics = train_step_fn(batch)
            c_vals.append(float(metrics["critic_loss"]))
            g_vals.append(float(metrics["generator_loss"]))

        mean_c = float(np.mean(c_vals))
        mean_g = float(np.mean(g_vals))
        elapsed = time.perf_counter() - t0

        if epoch % 50 == 0 or epoch == 1:
            time_per_epoch = (time.perf_counter() - class_t0) / epoch
            remaining = time_per_epoch * (epochs - epoch)
            eta_min, eta_sec = divmod(int(remaining), 60)
            now = time.strftime("%H:%M:%S")
            print(
                f"  [{now}] Epoch {epoch:3d}/{epochs} │ "
                f"C_loss: {mean_c:+9.4f}  G_loss: {mean_g:+9.4f}  "
                f"({elapsed:.1f}s/ep)  ETA: {eta_min}m{eta_sec:02d}s"
            )

        log_rows.append({
            "subject":        subject,
            "class_idx":      class_idx,
            "class_name":     cname,
            "epoch":          epoch,
            "critic_loss":    mean_c,
            "generator_loss": mean_g,
        })

    z         = tf.random.normal([N_SYNTHETIC_PER_CLASS, LATENT_DIM])
    synthetic = gen(z, training=False).numpy()    

    w_path = MODELS_DIR / f"generator_s{subject:02d}_c{class_idx}.weights.h5"
    gen.save_weights(str(w_path))
    print(f"  Generator weights → {w_path.name}")

    return synthetic


def main(args: argparse.Namespace) -> None:
    set_seeds()

    print(f"\n{'='*60}")
    print(f"  WGAN-GP Training  │  Subject {args.subject:02d}")
    print(f"  Epochs: {args.epochs}   Batch: {WGAN_BATCH_SIZE}")
    print(f"  λ={GP_LAMBDA}  n_critic={N_CRITIC}  "
          f"lr={WGAN_LR}  β₁={WGAN_BETA1}  β₂={WGAN_BETA2}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = preprocess_subject_split(args.subject, session="T")
    print(f"  Train/test split: {len(y_train)} train / {len(y_test)} test "
          f"(test partition held out, NOT used for WGAN-GP training)")

    class_data = split_by_class(X_train, y_train)
    print(f"  Class distribution (training partition only): "
          + "  ".join(f"C{c}={len(v)}" for c, v in class_data.items()))

    all_synthetic: dict[int, np.ndarray] = {}
    log_rows: list[dict] = []

    for cls_idx in range(N_CLASSES):
        if cls_idx not in class_data or len(class_data[cls_idx]) == 0:
            print(f"\n  No data for class {cls_idx}, skipping.")
            continue

        synthetic = train_wgan_class(
            X_class   = class_data[cls_idx],
            class_idx = cls_idx,
            subject   = args.subject,
            epochs    = args.epochs,
            log_rows  = log_rows,
        )
        all_synthetic[cls_idx] = synthetic

        per_class_path = SYNTHETIC_DIR / f"synthetic_s{args.subject:02d}_c{cls_idx}.npy"
        np.save(str(per_class_path), synthetic)
        print(f"  {N_SYNTHETIC_PER_CLASS} synthetic samples → {per_class_path.name}")

    # Training log CSV
    log_df   = pd.DataFrame(log_rows)
    log_path = METRICS_DIR / f"wgan_training_log_s{args.subject:02d}.csv"
    log_df.to_csv(str(log_path), index=False)
    print(f"\n  Training log → {log_path.name}")

    syn_X = np.concatenate(
        [all_synthetic[c] for c in sorted(all_synthetic)], axis=0
    )
    syn_y = np.concatenate(
        [np.full(N_SYNTHETIC_PER_CLASS, c) for c in sorted(all_synthetic)]
    ).astype(np.int32)

    combined = SYNTHETIC_DIR / f"synthetic_combined_s{args.subject:02d}.npz"
    np.savez(str(combined), X=syn_X, y=syn_y)
    print(f"  Combined synthetic ({len(syn_X)} samples) → {combined.name}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train per-class WGAN-GP for one subject"
    )
    parser.add_argument("--subject", type=int, default=1,
                        help="Subject index 1–9 (default: 1)")
    parser.add_argument("--epochs",  type=int, default=WGAN_EPOCHS,
                        help=f"Training epochs (default: {WGAN_EPOCHS})")
    parser.add_argument("--gpu",     action="store_true",
                        help="Enable GPU; default runs on CPU")
    args = parser.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    main(args)
