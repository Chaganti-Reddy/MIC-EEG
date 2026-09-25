"""
Compare vanilla GAN, DCGAN, and WGAN-GP on training time, feature-space FID,
and downstream CNN accuracy (Table 6). Run on a single representative subject.

All three architectures share the same generator (models.wgan_gp.build_generator)
so that the comparison isolates the effect of the discriminator/critic
objective (BCE vs. BCE-with-conv-discriminator vs. Wasserstein + gradient
penalty), consistent with Sec. 3.5.

Usage
-----
  python src/compare_gan_architectures.py --subject 1 --epochs 300 --batch_sizes 8 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LATENT_DIM, N_CLASSES, CLASS_NAMES, TRIAL_SHAPE,
    N_SYNTHETIC_PER_CLASS, CNN_EPOCHS, CNN_BATCH_SIZE,
    METRICS_DIR, MODELS_DIR, RANDOM_SEED, WGAN_LR, WGAN_BETA1, WGAN_BETA2, GP_LAMBDA, N_CRITIC,
)
from preprocessing import preprocess_subject_split, split_by_class
from models.wgan_gp import build_generator, build_critic, WGANGP
from models.cnn import build_cnn
from evaluate import compute_metrics
from fid_score import compute_fid


def set_seeds(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─── Vanilla GAN discriminator (fully-connected, sigmoid + BCE) ──────────────
def build_vanilla_discriminator(input_shape: tuple = TRIAL_SHAPE) -> Model:
    x_in = layers.Input(shape=input_shape, name="cwt_image")
    x = layers.Flatten()(x_in)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="real_prob")(x)
    return Model(x_in, out, name="vanilla_discriminator")


# ─── DCGAN discriminator (convolutional, sigmoid + BCE) ──────────────────────
def build_dcgan_discriminator(input_shape: tuple = TRIAL_SHAPE) -> Model:
    x_in = layers.Input(shape=input_shape, name="cwt_image")
    x = layers.Conv2D(64, (5, 5), strides=(2, 3), padding="same")(x_in)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, (5, 5), strides=(1, 5), padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="real_prob")(x)
    return Model(x_in, out, name="dcgan_discriminator")


class BCEGAN(tf.keras.Model):
    """Shared training loop for vanilla GAN and DCGAN (single critic update per
    generator update, binary cross-entropy objective, no gradient penalty)."""

    def __init__(self, generator: Model, discriminator: Model, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self._d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self._g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [self._d_loss_metric, self._g_loss_metric]

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch = tf.shape(real_images)[0]

        z = tf.random.normal([batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator(z, training=True)
            real_pred = self.discriminator(real_images, training=True)
            fake_pred = self.discriminator(fake, training=True)
            d_loss = self.bce(tf.ones_like(real_pred), real_pred) + \
                     self.bce(tf.zeros_like(fake_pred), fake_pred)
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        z = tf.random.normal([batch, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator(z, training=True)
            fake_pred = self.discriminator(fake, training=True)
            g_loss = self.bce(tf.ones_like(fake_pred), fake_pred)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self._d_loss_metric.update_state(d_loss)
        self._g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}


def train_architecture(arch: str, X_train: np.ndarray, y_train: np.ndarray,
                       batch_size: int, epochs: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Train one GAN architecture per-class on the training partition; return
    (synthetic_X, synthetic_y, elapsed_seconds)."""
    class_data = split_by_class(X_train, y_train)
    all_synth, all_y = [], []
    t0 = time.perf_counter()

    for cls_idx in range(N_CLASSES):
        if cls_idx not in class_data or len(class_data[cls_idx]) == 0:
            continue
        X_class = class_data[cls_idx].astype(np.float32)

        gen = build_generator(LATENT_DIM)

        dataset = (
            tf.data.Dataset.from_tensor_slices(X_class)
            .shuffle(buffer_size=len(X_class) * 3, seed=RANDOM_SEED)
            .batch(min(batch_size, len(X_class)), drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        if arch == "wgan_gp":
            critic = build_critic()
            model = WGANGP(gen, critic, latent_dim=LATENT_DIM, n_critic=N_CRITIC, gp_lambda=GP_LAMBDA)
            model.compile(
                g_optimizer=tf.keras.optimizers.legacy.Adam(WGAN_LR, WGAN_BETA1, WGAN_BETA2),
                c_optimizer=tf.keras.optimizers.legacy.Adam(WGAN_LR, WGAN_BETA1, WGAN_BETA2),
            )
        elif arch == "dcgan":
            disc = build_dcgan_discriminator()
            model = BCEGAN(gen, disc, latent_dim=LATENT_DIM)
            model.compile(
                g_optimizer=tf.keras.optimizers.legacy.Adam(2e-4, 0.5, 0.999),
                d_optimizer=tf.keras.optimizers.legacy.Adam(2e-4, 0.5, 0.999),
            )
        elif arch == "vanilla_gan":
            disc = build_vanilla_discriminator()
            model = BCEGAN(gen, disc, latent_dim=LATENT_DIM)
            model.compile(
                g_optimizer=tf.keras.optimizers.legacy.Adam(2e-4, 0.5, 0.999),
                d_optimizer=tf.keras.optimizers.legacy.Adam(2e-4, 0.5, 0.999),
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        train_step_fn = tf.function(model.train_step)
        for _ in range(epochs):
            for batch in dataset:
                train_step_fn(batch)

        z = tf.random.normal([N_SYNTHETIC_PER_CLASS, LATENT_DIM])
        synth = gen(z, training=False).numpy()
        all_synth.append(synth)
        all_y.append(np.full(N_SYNTHETIC_PER_CLASS, cls_idx))

    elapsed = time.perf_counter() - t0
    return np.concatenate(all_synth, axis=0), np.concatenate(all_y, axis=0).astype(np.int32), elapsed


def train_eval_cnn(X_train_real, y_train_real, X_syn, y_syn, X_test, y_test) -> float:
    X_train = np.concatenate([X_train_real, X_syn], axis=0)
    y_train = np.concatenate([y_train_real, y_syn], axis=0)

    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    model = build_cnn()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "_gan_arch_cmp_best.weights.h5"), monitor="val_accuracy",
            save_best_only=True, save_weights_only=True, verbose=0),
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
    parser = argparse.ArgumentParser(
        description="Compare vanilla GAN, DCGAN, and WGAN-GP (Table 6)"
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs per class, per architecture")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32],
                        help="Batch sizes to benchmark; the fastest is reported "
                             "as the main result, consistent with Sec. 3.5")
    args = parser.parse_args()

    set_seeds()
    X_train, y_train, X_test, y_test = preprocess_subject_split(args.subject, session="T")
    print(f"  Subject {args.subject:02d}: {len(y_train)} train / {len(y_test)} test")

    rows = []
    for arch in ["vanilla_gan", "dcgan", "wgan_gp"]:
        best_row = None
        for bs in args.batch_sizes:
            print(f"\n  [{arch}] batch_size={bs} …")
            set_seeds()
            X_syn, y_syn, elapsed = train_architecture(arch, X_train, y_train, bs, args.epochs)
            fid = compute_fid(X_train.astype(np.float32), X_syn.astype(np.float32))
            acc = train_eval_cnn(X_train, y_train, X_syn, y_syn, X_test, y_test)
            row = {
                "architecture": arch, "batch_size": bs,
                "train_time_min": round(elapsed / 60.0, 2),
                "mean_fid": round(fid, 2),
                "cnn_accuracy": round(acc * 100, 2),
            }
            print(f"    time={row['train_time_min']:.2f} min  "
                  f"FID={row['mean_fid']:.1f}  acc={row['cnn_accuracy']:.2f}%")
            rows.append(row)
            if best_row is None or elapsed < best_row["train_time_min"] * 60.0:
                best_row = row
        print(f"  [{arch}] fastest batch_size={best_row['batch_size']} "
              f"→ time={best_row['train_time_min']:.2f} min")

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / f"gan_architecture_comparison_s{args.subject:02d}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Saved → {csv_path.name}")

    summary_path = METRICS_DIR / f"gan_architecture_comparison_s{args.subject:02d}_summary.json"
    fastest = df.loc[df.groupby("architecture")["train_time_min"].idxmin()]
    with open(str(summary_path), "w") as fh:
        json.dump(fastest.to_dict(orient="records"), fh, indent=2)
    print(f"  Fastest-batch-size summary (Table 6) → {summary_path.name}")


if __name__ == "__main__":
    main()
