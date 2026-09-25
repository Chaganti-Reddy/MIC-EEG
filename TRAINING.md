# Training Guide — Motor Imagery Classification Pipeline

Complete reference for environment setup, step-by-step training, hyperparameters, expected outputs, and troubleshooting.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Stage 1 — WGAN-GP Training](#3-stage-1--wgan-gp-training)
4. [Stage 2 — CNN Training](#4-stage-2--cnn-training)
5. [Stage 3 — Cross-Subject Evaluation](#5-stage-3--cross-subject-evaluation)
6. [Stage 4 — Figure Generation](#6-stage-4--figure-generation)
7. [Additional Analyses](#7-additional-analyses)
8. [Full Pipeline (One Command)](#8-full-pipeline-one-command)
9. [Hyperparameters Reference](#9-hyperparameters-reference)
10. [Expected Results](#10-expected-results)
11. [Outputs Reference](#11-outputs-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Pre-Commit Checklist](#13-pre-commit-checklist)

---

## 1. Environment Setup

### System Requirements

| Component | Required | Tested |
|-----------|---------|--------|
| Python | 3.10+ | 3.11 |
| TensorFlow | 2.13.x | 2.13.1 |
| CUDA | 11.x | 11.8 |
| cuDNN | 8.x | 8.6 |
| RAM | 8 GB+ | 16 GB |
| VRAM | 2 GB+ | 4 GB (1766 MB accessible in WSL2) |

### Install

```bash
pip install -r requirements.txt
```

Key packages:
```
tensorflow==2.13.1
mne>=1.4
pywt>=1.4
scipy>=1.10
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
```

### GPU Environment Variables (WSL2 / Linux)

These must be set **before every training run**. TensorFlow and MNE conflict during library loading if set too late.

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/miniconda/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
```

Replace `/path/to/miniconda` with your actual conda environment path (e.g., `/home/user/miniconda3/envs/myenv`).

### Verify GPU is visible

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### CPU-only fallback

```bash
export CUDA_VISIBLE_DEVICES=-1
```
Training will be ~6–10× slower but produces identical results.

---

## 2. Dataset Preparation

Download the **BCI Competition IV Dataset 2a** from [bbci.de/competition/iv](https://www.bbci.de/competition/iv/).

Place all GDF files in the `dataset/` directory:
```
dataset/
  A01T.gdf  A01E.gdf
  A02T.gdf  A02E.gdf
  ...
  A09T.gdf  A09E.gdf
```

| File | Labels? | Use |
|------|---------|-----|
| `A0{N}T.gdf` | Yes (events 769–772) | Training and evaluation |
| `A0{N}E.gdf` | No (event 783) | **Not used** |

**Do not modify any files in `dataset/`.**

### What the preprocessor does automatically

Preprocessing runs on-the-fly inside `train_wgan.py` and `train_cnn.py` — no separate preprocessing step is needed.

```
A0{N}T.gdf
  → MNE read_raw_gdf
  → Bandpass FIR [0.5–100 Hz] + Notch 50 Hz
  → Select 5 channels by index: [0, 7, 9, 11, 21]
     (EEG-Fz, EEG-C3, EEG-Cz, EEG-C4, EEG-Oz)
  → Extract epochs: tmin=0s, tmax=6s (full trial, Sec. 2.1)
     Drop duplicate event positions
     Exclude artifact events (1023, 1025)
  → scipy.signal.decimate (FIR anti-alias filter, r=4): 1500 → 375 time points
     (250 Hz → 62.5 Hz effective sampling rate, Sec. 2.2.6)
  → Complex Morlet CWT (Eq. 20, n=6): 50 frequencies, 0.5–20 Hz, REAL part kept
  → Stack 5 channels → (50, 375, 5)
  → Normalise EACH TRIAL independently to [−1, 1] (no cross-trial/cross-split leakage)
  → Leakage-controlled 80/20 split (preprocess_subject_split): performed on RAW trial
     indices, cached per subject in outputs/splits/, BEFORE WGAN-GP ever sees the data
```

Output: `(n_trials, 50, 375, 5)` float32 — typically 288 trials per subject.

---

## 3. Stage 1 — WGAN-GP Training

**Script**: `src/train_wgan.py`

Trains one WGAN-GP per MI class per subject. Each GAN learns the CWT distribution for its class and generates 100 synthetic samples after training.

### Single subject

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/conda/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
python3 src/train_wgan.py --subject 1
```

### All 9 subjects (sequential, lid-close safe)

```bash
nohup bash -c '
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/conda/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
for s in 1 2 3 4 5 6 7 8 9; do
    echo "=== Subject $s ==="
    python3 src/train_wgan.py --subject $s
done' > /tmp/wgan_all.log 2>&1 &

echo "Running in background. Monitor with: tail -f /tmp/wgan_all.log"
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject` | required | Subject number (1–9) |
| `--epochs` | 300 | Training epochs per class |

### Training Details

For each of the 4 classes:
1. Build Generator + Critic (see architectures in README)
2. Compile with `legacy.Adam(lr=1e-4, β₁=0, β₂=0.9)` on both
3. Run 300 epochs:
   - Each epoch: `n_critic=5` critic updates per 1 generator update
   - Gradient penalty (λ=10) computed in **fp32** (not fp16) for numerical stability
4. Generate 100 synthetic samples: `G(z)` where `z ~ N(0, I₁₀₀)`

### Progress Output

```
[14:32:17] Subject 01  Class 0/4 (Left Hand)
[14:32:17] Epoch   50/300 │ C_loss: -89.84  G_loss: +49.40  │ 0.6s/ep  ETA: 2m54s
[14:35:12] Epoch  300/300 │ C_loss: -71.23  G_loss: +88.12  │ 0.5s/ep  ETA: 0m00s
[14:35:12] Generating 100 synthetic samples …  shape=(100, 50, 375, 5)
```

**Healthy loss ranges:**
- `C_loss`: −50 to −100 (negative = critic correctly scores real > fake)
- `G_loss`: +30 to +150 (positive = generator improving)
- NaN or extreme values (>±1000) indicate a problem — see Troubleshooting

### Timing Estimates

| Scope | Time |
|-------|------|
| Warmup (epoch 1) | 10–15s (XLA trace) |
| Steady state | ~0.5s/epoch |
| Per class | ~2.5 min |
| Per subject (4 classes) | ~10–12 min |
| All 9 subjects | ~90–110 min |

### Outputs per Subject

```
outputs/synthetic/
  synthetic_s0{N}_c0.npy   # Left Hand  — shape (100, 50, 375, 5)
  synthetic_s0{N}_c1.npy   # Right Hand
  synthetic_s0{N}_c2.npy   # Both Feet
  synthetic_s0{N}_c3.npy   # Tongue
  synthetic_combined_s0{N}.npz   # X: (400, 50, 375, 5)  y: (400,)

outputs/models/
  generator_s0{N}_c0.weights.h5  …  generator_s0{N}_c3.weights.h5

metrics/
  wgan_training_log_s0{N}.csv    # columns: epoch, class, critic_loss, generator_loss
```

---

## 4. Stage 2 — CNN Training

**Script**: `src/train_cnn.py`
**Prerequisite**: `outputs/synthetic/synthetic_combined_s0{N}.npz` must exist

### Single subject

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/conda/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
python3 src/train_cnn.py --subject 1
```

### With cross-subject evaluation

```bash
python3 src/train_cnn.py --subject 1 --cross_test 2
# Trains on S01, then evaluates the trained model on S02
```

### All 9 subjects

```bash
for s in 1 2 3 4 5 6 7 8 9; do
  LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/conda/lib:$LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=0 \
  python3 src/train_cnn.py --subject $s
done
```

### Ablation

```bash
python3 src/train_cnn.py --subject 1 --no_synthetic
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject` | required | Subject number (1–9) |
| `--epochs` | 70 | Max training epochs |
| `--cross_test` | None | Also evaluate on this subject |
| `--no_synthetic` | False | Train on real data only (ablation) |

### Training Details

1. Load the leakage-controlled 80/20 split via `preprocess_subject_split()` (~230 real
   train trials / ~58 real held-out test trials, cached in `outputs/splits/`)
2. Load synthetic from `synthetic_combined_s0{N}.npz` (400 trials, generated ONLY from the
   training partition's WGAN-GP, see Stage 1)
3. Concatenate: ~230 real train + 400 synthetic = **~630 training samples**
4. Evaluate on the ~58 held-out real test trials — never used in WGAN-GP training,
   synthetic-sample generation, or CNN training
5. Train with callbacks (monitored on the held-out test partition):
   - `ModelCheckpoint`: save best weights by `val_accuracy`
   - `EarlyStopping`: patience=15 on `val_loss`
   - `ReduceLROnPlateau`: factor=0.5, patience=7

### Timing Estimates

| Scope | Time |
|-------|------|
| Per epoch | ~5s |
| Full 70 epochs | ~6 min |
| With early stopping | ~3–5 min typical |
| All 9 subjects | ~30–45 min |

### Outputs per Subject

```
outputs/models/
  cnn_s0{N}_best.weights.h5       # best val_accuracy checkpoint

outputs/figures/
  cm_s0{N}.png                    # confusion matrix (4×4)
  history_s0{N}.png               # accuracy + loss curves

metrics/
  cnn_metrics_s0{N}.json          # accuracy, precision, recall, f1
  cnn_metrics_s0{N}_classwise.json  # per-class precision/recall/f1
  cnn_history_s0{N}.csv           # epoch-level training log
```

---

## 5. Stage 3 — Cross-Subject Evaluation

**Script**: `src/cross_eval.py`
**Prerequisite**: CNN trained for the train subject

Loads the CNN trained on subject A and evaluates it on subject B without any fine-tuning.

### Single pair

```bash
python3 src/cross_eval.py --train_subject 1 --test_subject 2
```

### Full 9×9 matrix (72 pairs)

```bash
for a in 1 2 3 4 5 6 7 8 9; do
  for b in 1 2 3 4 5 6 7 8 9; do
    [ $a -ne $b ] && python3 src/cross_eval.py \
      --train_subject $a --test_subject $b
  done
done
```

### Outputs

```
metrics/cross_s0{A}_to_s0{B}.json       # accuracy, f1
outputs/figures/cm_s0{A}_to_s0{B}.png  # confusion matrix
```

---

## 6. Stage 4 — Figure Generation

**Script**: `src/analyze_results.py`

Generates 200+ publication-quality figures from all saved metrics.

```bash
python3 src/analyze_results.py
```

### Figure Sections

| Section | Count | Description |
|---------|-------|-------------|
| A. Aggregate summary | ~12 | Bar charts, box plots, violin plots, scatter, heatmap |
| B. Radar charts | 2 | Per-subject multi-metric radar |
| C. Cross-subject heatmaps | 6 | 9×9 generalisation matrix per metric |
| D. Per-class figures | ~45 | Class-wise recall/precision/F1 per subject |
| E. CNN training history | ~22 | Loss + accuracy curves per subject |
| F. WGAN loss figures | ~40 | Critic + Generator curves per class per subject |
| G. Synthetic visualisations | ~20 | Real vs synthetic CWT, distribution plots |
| H. Statistical significance | 3 | Bootstrap CI, t-test, distribution plots |
| I. Tables | 2 | CSV summary table + cross-subject matrix |

All figures saved to `outputs/figures/`.

---

## 7. Additional Analyses

These scripts read existing trained models and data — **no retraining required**.

### FID Score (Synthetic Quality)

```bash
python3 src/fid_score.py
```

Computes Fréchet distance between real and synthetic CWT distributions per class per subject. Uses PCA (256 components) as a feature extractor since no domain-specific neural encoder exists for EEG scalograms.

Outputs: `metrics/fid_scores.csv`, `outputs/figures/fid_*.png`

### Frequency Band Analysis

```bash
python3 src/band_analysis.py
```

Maps the 50 CWT analysis frequencies (config.CWT_FREQS, 0.5–20 Hz, complex Morlet with
n_cycles=6, Sec. 2.2.6) to EEG frequency bands, then evaluates how much discriminative
information each band contributes by training logistic probe classifiers on band-restricted
CWT features.

Band mapping (complex Morlet, fs_eff = 62.5 Hz, CWT_FREQS = linspace(0.5, 20, 50) Hz):
```
Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–20 Hz)
```
Each of the 50 CWT rows already has a known frequency (config.CWT_FREQS), so bands are
selected directly by frequency value — no scale-to-frequency conversion is needed.

Outputs: `metrics/band_analysis.csv`, `outputs/figures/band_*.png`

### Ablation Study

```bash
python3 src/ablation.py
```

Trains CNN with `--no_synthetic` for all subjects and compares to original real+synthetic results.

- Existing `cnn_metrics_s0{N}.json` files are **never modified**
- Real-only results saved as `cnn_metrics_s0{N}_nosyn.json`

If `_nosyn` models already exist, regenerate figures only:
```bash
python3 src/ablation.py --analyze_only
```

Outputs: `metrics/ablation_summary.csv`, `outputs/figures/ablation_*.png`

---

## 8. Full Pipeline 

### Complete run from scratch

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/conda/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 \
python3 run_all.py
```

### Skip WGAN (synthetic already exists)

```bash
LD_LIBRARY_PATH=... CUDA_VISIBLE_DEVICES=0 python3 run_all.py --skip_wgan
```

### Analysis only

```bash
python3 run_all.py --analyze_only
```

### Extras only (FID + band analysis + ablation, no main training)

```bash
python3 run_all.py --extras_only
```

### `run_all.py` Flag Reference

| Flag | Description |
|------|-------------|
| `--subjects 1 2 3` | Run only specified subjects (default: 1–9) |
| `--skip_wgan` | Skip WGAN-GP training |
| `--skip_cnn` | Skip CNN training |
| `--analyze_only` | Jump directly to figure generation |
| `--extras_only` | Run FID + band + ablation only |
| `--skip_fid` | Skip FID computation in extras |
| `--skip_band` | Skip band analysis in extras |
| `--skip_ablation` | Skip ablation study in extras |
| `--wgan_epochs N` | Override WGAN epoch count |
| `--cnn_epochs N` | Override CNN epoch count |

---

## 9. Hyperparameters Reference

All values live in `src/config.py`. Edit there only — all scripts import from it.

### WGAN-GP

| Parameter | Paper Value | Notes |
|-----------|------------|-------|
| `LATENT_DIM` | 100 | Noise vector dimension |
| `WGAN_BATCH_SIZE` | **100** | Reduce to 64 if OOM on <2GB VRAM |
| `WGAN_EPOCHS` | 300 | Per class |
| `N_CRITIC` | 5 | Critic updates per generator step |
| `GP_LAMBDA` | 10 | Gradient penalty coefficient λ |
| `WGAN_LR` | 1e-4 | Adam learning rate (both networks) |
| `WGAN_BETA1` | 0.0 | Adam β₁ (no momentum — standard for WGAN) |
| `WGAN_BETA2` | 0.9 | Adam β₂ |
| `N_SYNTHETIC_PER_CLASS` | 100 | Samples to generate after training |

### CNN

| Parameter | Value | Notes |
|-----------|-------|-------|
| `CNN_LR` | 1e-4 | Adam learning rate |
| `CNN_EPOCHS` | 70 | Max epochs (EarlyStopping may stop earlier) |
| `CNN_BATCH_SIZE` | 32 | |
| `CNN_DROPOUT` | 0.5 | Applied after each Conv block |
| `CNN_L2` | 0.01 | L2 regularisation on Conv + Dense |
| `CNN_DENSE_UNITS` | 750 | Penultimate dense layer |
| `CNN_FILTERS` | 32 | Conv2D filter count (both blocks) |
| `CNN_KERNEL_SIZE` | (7, 7) | Conv2D kernel |
| `CNN_VAL_SPLIT` | 0.20 | 20% validation fraction |

### Preprocessing

| Parameter | Value |
|-----------|-------|
| `TRIAL_SHAPE` | `(50, 375, 5)` |
| `N_CWT_SCALES` | 50 |
| `CWT_FREQS` | linspace(0.5, 20, 50) Hz, complex Morlet, n_cycles=6 |
| `DOWNSAMPLE_FACTOR` | 4 (anti-aliased decimation, 250 Hz → 62.5 Hz) |
| `N_TIME_SUBSAMPLE` | 375 |
| `TMIN` / `TMAX` | 0.0 / 6.0 s |
| `CHANNELS` | `[0, 7, 9, 11, 21]` |
| `RANDOM_SEED` | 42 |

---

## 10. Expected Results

### Per-Subject Classification Accuracy (Real + Synthetic)

| Subject | Accuracy | F1 |
|---------|----------|----|
| S01 | 78.26% | 0.779 |
| S02 | 76.09% | 0.760 |
| S03 | 72.46% | 0.735 |
| S04 | 78.99% | 0.789 |
| S05 | 74.64% | 0.751 |
| S06 | 72.46% | 0.727 |
| S07 | 73.91% | 0.739 |
| S08 | 75.36% | 0.753 |
| S09 | 78.99% | 0.794 |
| **Mean** | **75.68%** | **0.759** |

### WGAN-GP Loss Convergence (Healthy)

```
Epoch  50: C_loss ≈ −85 to −70  │  G_loss ≈ +40 to +60
Epoch 150: C_loss ≈ −75 to −60  │  G_loss ≈ +70 to +100
Epoch 300: C_loss ≈ −65 to −50  │  G_loss ≈ +80 to +150
```

C_loss becoming less negative over time = generator improving = expected.

### FID Scores (lower = better)

Mean FID across all subjects: **5,714** (range: 4,466–7,015)

### Ablation

| Condition | Mean Accuracy |
|-----------|--------------|
| Real only | 29.69% |
| Real + Synthetic | 75.68% |
| Delta | +45.99% |

---

## 11. Outputs Reference

```
outputs/
├── synthetic/
│   ├── synthetic_s0{N}_c{0-3}.npy          # per-class, shape (100, 50, 375, 5)
│   └── synthetic_combined_s0{N}.npz         # all classes, X:(400,50,375,5) y:(400,)
│
├── models/
│   ├── generator_s0{N}_c{0-3}.weights.h5   # 36 WGAN generator checkpoints
│   ├── cnn_s0{N}_best.weights.h5           # 9 CNN best-val checkpoints
│   └── cnn_s0{N}_nosyn_best.weights.h5     # 9 ablation (real-only) checkpoints
│
└── figures/                                 # 200+ PNG files
    ├── cm_s0{N}.png                         # confusion matrix
    ├── history_s0{N}.png                    # training curves
    ├── cm_s0{A}_to_s0{B}.png               # cross-subject confusion
    ├── fid_*.png                            # FID analysis (3 files)
    ├── band_*.png                           # band analysis (4 files)
    ├── ablation_*.png                       # ablation study (4 files)
    ├── significance_*.png                   # statistical tests (3 files)
    └── ...  (~200 total)

metrics/
├── wgan_training_log_s0{N}.csv             # WGAN loss per epoch
├── cnn_metrics_s0{N}.json                  # accuracy, f1, precision, recall
├── cnn_metrics_s0{N}_classwise.json        # per-class breakdown
├── cnn_metrics_s0{N}_nosyn.json            # ablation real-only results
├── cnn_history_s0{N}.csv                   # epoch-level CNN log
├── cross_s0{A}_to_s0{B}.json              # cross-subject pairs (72 files)
├── cross_subject_matrix.csv                # 9×9 accuracy matrix
├── fid_scores.csv                          # per-subject per-class FID
├── band_analysis.csv                       # per-subject per-band accuracy
├── ablation_summary.csv                    # real-only vs real+syn comparison
└── summary_table.csv                       # per-subject aggregate
```

---

## 12. Troubleshooting

### `libdevice not found at ./libdevice.10.bc`

XLA JIT compilation failure — libdevice is missing or on a non-standard path.

**Fix** (already applied in `train_wgan.py`):
```python
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
tf.config.optimizer.set_jit(False)
```

### `OOM when allocating tensor`

GPU memory exhausted.

**Fix**: Reduce `WGAN_BATCH_SIZE` in `src/config.py`:
```python
WGAN_BATCH_SIZE = 64   # or 32 for very limited VRAM
```
Restore to 100 before committing.

### `InvalidArgumentError: was expected to be a half tensor but is a float tensor`

Mixed-precision dtype mismatch. The gradient penalty interpolates between real (fp32) and fake (fp16) tensors.

**Fix** (already applied in `wgan_gp.py`):
```python
real_f = tf.cast(real, tf.float32)
fake_f = tf.cast(fake, tf.float32)
x_hat  = real_f + eps * (fake_f - real_f)
```

### `NaN losses`

Two common causes:

1. **BatchNorm in critic** — creates batch-level dependencies that corrupt the gradient penalty. **Fix**: ensure critic has no `BatchNorm` layers (already removed).
2. **Empty batches** (`drop_remainder=True` with small datasets) — **Fix**: set `drop_remainder=False` (already set).

### Channel names not found

```
KeyError: "Channel 'EEG-Oz' not found"
```

MNE appends numbers to duplicate channel names (e.g., `EEG` → `EEG-0`, `EEG-1`). The dataset has non-unique channel names.

**Fix** (already applied in `preprocessing.py`): select channels by integer index, not by name.
```python
raw.pick(picks=[0, 7, 9, 11, 21])
```

### `E` file has no labels

```
Warning: No events with codes {769, 770, 771, 772} found
```

Never use `A0{N}E.gdf`. Use only `T` files.

### `CUDA_ERROR_NO_DEVICE` / GPU not found

MNE loads `libgomp` early, conflicting with the NVIDIA driver loader.

**Fix**: set `LD_LIBRARY_PATH` with `/usr/lib/wsl/lib` **first** in the path, and call `tf.config.experimental.set_memory_growth` before any other TF operation (already done at the top of `train_wgan.py`).

### Training very slow (CPU speeds)

Confirm GPU is being used:
```python
tf.config.list_physical_devices('GPU')  # must return a GPU device
```
If empty, `CUDA_VISIBLE_DEVICES` may not be set, or the `LD_LIBRARY_PATH` is wrong.

### `tf.function` retracing every epoch

Caused by Python-side loop variables changing shape/dtype. The `train_step` is wrapped with `@tf.function` (already applied). If retracing persists, add `input_signature` to the decorator.
