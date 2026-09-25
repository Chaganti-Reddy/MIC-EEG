import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from preprocessing import preprocess_subject_split
from config import (
    N_CLASSES, CLASS_NAMES, RANDOM_SEED,
    CNN_EPOCHS, CNN_BATCH_SIZE, CNN_DROPOUT, CNN_L2,
    CNN_DENSE_UNITS, CNN_FILTERS, CNN_KERNEL_SIZE,
    N_SYNTHETIC_PER_CLASS, TRIAL_SHAPE
)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def build_cnn(input_shape=TRIAL_SHAPE):
    inp = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(CNN_FILTERS, CNN_KERNEL_SIZE, activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(CNN_L2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CNN_DROPOUT)(x)
    x = layers.Conv2D(CNN_FILTERS, CNN_KERNEL_SIZE, activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(CNN_L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CNN_DROPOUT)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(CNN_DENSE_UNITS, activation='relu')(x)
    out = layers.Dense(N_CLASSES, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_eval_cnn(X_train, y_train, X_test, y_test, subject, method):
    print(f"    [CNN] Training on {len(X_train)} samples ({method}) ...")
    model = build_cnn(input_shape=X_train.shape[1:])
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True,
        verbose=0)

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1:3d}/{CNN_EPOCHS} — "
                      f"loss={logs['loss']:.4f}  acc={logs['accuracy']:.4f}  "
                      f"val_acc={logs.get('val_accuracy', 0):.4f}")

    model.fit(X_train, y_train,
              epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
              validation_split=0.1,
              callbacks=[cb, EpochLogger()],
              verbose=0)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"    [CNN] Test accuracy: {acc*100:.2f}%")
    tf.keras.backend.clear_session()
    return round(acc * 100, 2)


def smote_augment(X_train, y_train, n_components=200):
    orig_shape = X_train.shape[1:]
    flat_dim   = int(np.prod(orig_shape))
    X_flat     = X_train.reshape(len(X_train), -1).astype(np.float32)

    print(f"    [SMOTE] PCA compression {flat_dim}D → {n_components}D ...")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_flat)
    print(f"    [SMOTE] Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    k_nn = min(4, min(np.bincount(y_train)) - 1)
    print(f"    [SMOTE] Running SMOTE (k={k_nn}) on {X_pca.shape} ...")
    sm = SMOTE(k_neighbors=k_nn, random_state=RANDOM_SEED)
    X_sm_pca, y_sm = sm.fit_resample(X_pca, y_train)

    n_new = len(X_sm_pca) - len(X_pca)
    print(f"    [SMOTE] Generated {n_new} synthetic samples, reconstructing ...")
    X_sm_flat = pca.inverse_transform(X_sm_pca).astype(np.float32)
    X_sm = X_sm_flat.reshape(-1, *orig_shape)
    print(f"    [SMOTE] Final augmented set: {X_sm.shape}")
    return X_sm, y_sm


def build_vae(input_dim: int, latent_dim: int = 64):
    # Encoder
    enc_inp   = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    h         = layers.Dense(256, activation='relu')(enc_inp)
    z_mean    = layers.Dense(latent_dim, name='z_mean')(h)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(h)
    eps       = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    z         = z_mean + tf.exp(0.5 * z_log_var) * eps
    encoder   = Model(enc_inp, [z_mean, z_log_var, z], name='encoder')
    dec_inp = tf.keras.Input(shape=(latent_dim,), dtype=tf.float32)
    h       = layers.Dense(256, activation='relu')(dec_inp)
    dec_out = layers.Dense(input_dim, activation='tanh')(h)
    decoder = Model(dec_inp, dec_out, name='decoder')
    z_out    = encoder(enc_inp)[2]
    rec_loss = tf.reduce_mean(tf.square(enc_inp - decoder(z_out)))
    kl_loss  = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae = Model(enc_inp, decoder(z_out), name='vae')
    vae.add_loss(rec_loss + 1e-3 * kl_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    return vae, decoder


def vae_augment(X_train, y_train,
                n_synth=N_SYNTHETIC_PER_CLASS,
                latent_dim=64,
                pca_components=200):
    orig_shape = X_train.shape[1:]
    flat_dim   = int(np.prod(orig_shape))
    X_flat     = X_train.reshape(len(X_train), -1).astype(np.float32)

    print(f"    [VAE] PCA compression {flat_dim}D → {pca_components}D ...")
    pca = PCA(n_components=pca_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_flat)
    print(f"    [VAE] Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    synth_X, synth_y = [], []

    for cls in range(N_CLASSES):
        cls_data = X_pca[y_train == cls].astype(np.float32)
        mn, mx   = cls_data.min(), cls_data.max()
        cls_norm = (2.0 * (cls_data - mn) / (mx - mn + 1e-8) - 1.0)

        print(f"    [VAE] Class {cls} ({CLASS_NAMES[cls]}): "
              f"training on {len(cls_data)} samples ...")

        vae, decoder = build_vae(pca_components, latent_dim)

        class VAELogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 25 == 0:
                    print(f"      VAE epoch {epoch+1}/100 — loss={logs['loss']:.6f}")

        vae.fit(cls_norm, cls_norm,
                epochs=100, batch_size=32,
                callbacks=[VAELogger()], verbose=0)

        z_sample = np.random.normal(0, 1, (n_synth, latent_dim)).astype(np.float32)
        synth_pca = decoder.predict(z_sample, verbose=0)
        synth_pca = (synth_pca + 1.0) / 2.0 * (mx - mn) + mn

        synth_orig = pca.inverse_transform(synth_pca).astype(np.float32)
        synth_X.append(synth_orig.reshape(n_synth, *orig_shape))
        synth_y.extend([cls] * n_synth)
        print(f"    [VAE] Class {cls}: generated {n_synth} samples ✓")
        tf.keras.backend.clear_session()

    synth_X = np.vstack(synth_X)
    synth_y = np.array(synth_y)
    X_aug   = np.concatenate([X_train, synth_X], axis=0)
    y_aug   = np.concatenate([y_train, synth_y], axis=0)
    print(f"    [VAE] Final augmented set: {X_aug.shape}")
    return X_aug, y_aug

WGAN_GP_RESULTS = {
    1: 78.26, 2: 76.09, 3: 72.46, 4: 78.99, 5: 74.64,
    6: 72.46, 7: 73.91, 8: 75.36, 9: 78.99
}
REAL_ONLY = {
    1: 25.86, 2: 31.03, 3: 27.59, 4: 24.14, 5: 29.31,
    6: 24.14, 7: 32.76, 8: 13.79, 9: 58.62
}

smote_results = {}
vae_results   = {}
total_start   = time.time()

for subj in range(1, 10):
    subj_start = time.time()
    print(f"\n{'='*60}")
    print(f"SUBJECT {subj:02d}  ({time.strftime('%H:%M:%S')})")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = preprocess_subject_split(subj, session='T', verbose=True)
    orig_shape = X_train.shape[1:]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Class dist (train): {np.bincount(y_train).tolist()}")

    print(f"\n  --- SMOTE ---")
    t0 = time.time()
    X_sm, y_sm = smote_augment(X_train, y_train, n_components=200)
    smote_acc = train_eval_cnn(X_sm, y_sm, X_test, y_test, subj, 'SMOTE')
    smote_results[subj] = smote_acc
    print(f"  SMOTE done in {time.time()-t0:.0f}s → accuracy: {smote_acc}%")

    print(f"\n  --- VAE ---")
    t0 = time.time()
    X_vae, y_vae = vae_augment(X_train, y_train,
                               n_synth=N_SYNTHETIC_PER_CLASS,
                               latent_dim=64, pca_components=200)
    vae_acc = train_eval_cnn(X_vae, y_vae, X_test, y_test, subj, 'VAE')
    vae_results[subj] = vae_acc
    print(f"  VAE done in {time.time()-t0:.0f}s → accuracy: {vae_acc}%")

    print(f"\n  Subject {subj:02d} total: {time.time()-subj_start:.0f}s")
    print(f"  Running totals → SMOTE: {list(smote_results.values())}  "
          f"VAE: {list(vae_results.values())}")

print(f"\n\nTotal elapsed: {(time.time()-total_start)/60:.1f} min")
print("\n" + "="*72)
print("FINAL RESULTS — Augmentation Method Comparison")
print("="*72)
print(f"{'Subject':<10} {'Real Only':>10} {'SMOTE':>10} {'VAE':>10} {'WGAN-GP':>10}")
print("-"*55)
for subj in range(1, 10):
    print(f"S{subj:02d}       "
          f"{REAL_ONLY[subj]:>10.2f} "
          f"{smote_results[subj]:>10.2f} "
          f"{vae_results[subj]:>10.2f} "
          f"{WGAN_GP_RESULTS[subj]:>10.2f}")
print("-"*55)
print(f"{'Mean':<10} "
      f"{np.mean(list(REAL_ONLY.values())):>10.2f} "
      f"{np.mean(list(smote_results.values())):>10.2f} "
      f"{np.mean(list(vae_results.values())):>10.2f} "
      f"{np.mean(list(WGAN_GP_RESULTS.values())):>10.2f}")
print(f"{'Std':<10} "
      f"{np.std(list(REAL_ONLY.values())):>10.2f} "
      f"{np.std(list(smote_results.values())):>10.2f} "
      f"{np.std(list(vae_results.values())):>10.2f} "
      f"{np.std(list(WGAN_GP_RESULTS.values())):>10.2f}")
print("="*72)