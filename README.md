<div align="center">

# EEG Motor Imagery Classification via WGAN-GP Data Augmentation

**4-class Brain-Computer Interface using Wasserstein GAN with Gradient Penalty**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![BCI IV 2a](https://img.shields.io/badge/Dataset-BCI%20IV%202a-green.svg)](https://www.bbci.de/competition/iv/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22870348.svg)](https://doi.org/10.5281/zenodo.22870348)

*Motor Imagery Classification Performance Enhancement with EEG Data Augmentation via Wasserstein Generative Adversarial Networks*

</div>

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Method Overview](#2-method-overview)
3. [Dataset](#3-dataset)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Model Architectures](#5-model-architectures)
6. [Training Procedure](#6-training-procedure)
7. [Data Augmentation Strategies](#7-data-augmentation-strategies)
8. [Results](#8-results)
9. [Additional Analyses](#9-additional-analyses)
10. [Project Structure](#10-project-structure)
11. [Installation & Quick Start](#11-installation--quick-start)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [Citation](#13-citation)

---

## 1. Background & Motivation

Motor Imagery (MI) Brain-Computer Interfaces (BCIs) decode a user's imagined limb movements from electroencephalography (EEG) signals — enabling communication and control for individuals with paralysis or neuromuscular disorders.

The central challenge is **data scarcity**: collecting high-quality, labelled EEG data is expensive, time-consuming, and cognitively demanding for participants. A typical BCI session yields only 288 labelled trials per subject (72 per class), which is insufficient to train deep neural networks without severe overfitting.

This work addresses the scarcity problem using **Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP)** to synthesise realistic CWT scalogram representations of EEG signals. The synthesised samples augment the training set, improving CNN classifier accuracy from near-chance (≈30%) to **75.68% mean accuracy** across 9 subjects — a +46 percentage-point improvement.

### Why WGAN-GP?

Standard GANs suffer from **mode collapse** and **training instability**, particularly problematic with small datasets like MI-EEG. WGAN-GP solves both issues:

- The **Wasserstein distance** (Earth Mover's distance) provides a smooth, meaningful loss that correlates with sample quality
- The **gradient penalty** enforces the Lipschitz constraint on the critic without weight clipping, preventing vanishing gradients
- One WGAN-GP is trained **per class**, allowing the generator to specialise on each MI class's unique spectro-temporal pattern

---

## 2. Method Overview

The pipeline has three sequential stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0 — Leakage-Controlled Split       │
│                                                                  │
│  Raw trials → stratified 80/20 split on RAW indices             │
│  → train partition (~230 trials) / held-out test (~58 trials)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Stage 1 — Feature Extraction (Preprocessing)                   │
│                                                                  │
│  Raw EEG (250 Hz) → Bandpass + Notch → 5-channel subset        │
│  → Epoch extraction (0–6 s) → anti-aliased decimation (r=4,     │
│  62.5 Hz) → complex Morlet CWT, real part → (50, 375, 5) tensor │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Stage 2 — WGAN-GP Synthesis (one GAN per class)                │
│                                                                  │
│  TRAINING-partition CWT trials only → train Critic + Generator  │
│  → sample 100 synthetic CWT trials per class                    │
│  → 400 synthetic + ~230 real train = ~630 total training samples│
│  (held-out test partition never touches the generator)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Stage 3 — CNN Classification                                   │
│                                                                  │
│  ~630 training samples → CNN → 4-class softmax                  │
│  → evaluated ONLY on the ~58 held-out real test trials           │
│  → per-subject accuracy, F1, kappa, cross-subject generalisation │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset

**BCI Competition IV Dataset 2a** — publicly available at [bbci.de/competition/iv](https://www.bbci.de/competition/iv/).

### Recording Protocol

- **9 subjects**, 2 sessions each (training `T`, evaluation `E`)
- **22 Ag/AgCl EEG electrodes** in 10-20 system (3.5 cm spacing) + 3 EOG channels
- **Sampling rate**: 250 Hz
- **Pre-filtered**: bandpass [0.5–100 Hz], 50 Hz notch (applied at recording)
- **288 trials per session**: 6 runs × 48 trials, balanced 12 per class × 4 classes

### Trial Structure

```
─2s  cue    ─── imagery window ───     end
  │ fixation │←──── cue + imagery ────→│
  0          2                         6  (seconds)
```

The classifier operates on the full 6-second trial (`tmin=0`, `tmax=6 s`), spanning fixation,
cue, and the complete motor-imagery/feedback period, consistent with the BCI IV 2a trial
timeline described in the manuscript (Section 2.1).

### Class Labels

| Label | Class | GDF Event Code |
|-------|-------|---------------|
| 0 | Left Hand | 769 |
| 1 | Right Hand | 770 |
| 2 | Both Feet | 771 |
| 3 | Tongue | 772 |

### Electrode Subset

Only 5 of the 22 channels are used (RAM constraint during CWT computation):

| Channel | Index | Location | Relevance |
|---------|-------|----------|-----------|
| EEG-Fz | 0 | Frontal midline | Supplementary motor area |
| EEG-C3 | 7 | Left central | Right-hand motor cortex |
| EEG-Cz | 9 | Central midline | Bilateral motor activity |
| EEG-C4 | 11 | Right central | Left-hand motor cortex |
| EEG-Oz | 21 | Occipital | Visual/feet imagery |

> **Note**: Only `A0{N}T.gdf` (training) files are used. Evaluation files (`A0{N}E.gdf`) carry event code 783 (unknown class) and cannot be used for supervised training.

---

## 4. Preprocessing Pipeline

```
A0{N}T.gdf
   │
   ├─ MNE read_raw_gdf  ──────────── loads 25 channels (22 EEG + 3 EOG)
   │
   ├─ FIR Bandpass [0.5 – 100 Hz]  ─ removes DC drift + high-freq noise
   ├─ Notch filter @ 50 Hz  ──────── removes power-line interference
   │
   ├─ Channel selection by index ─── picks [0,7,9,11,21] → 5 channels
   │   (by index, not name — MNE renames duplicate channel names)
   │
   ├─ Event extraction  ──────────── events {769,770,771,772} → labels {0,1,2,3}
   │   Drop duplicate positions (affects S04, S06)
   │   Exclude artifact events {1023, 1025}
   │
   ├─ Epoch extraction ───────────── tmin=0 s, tmax=6 s, no baseline correction
   │   → shape: (n_trials, 5_channels, ~1501_samples)
   │
   ├─ Leakage-controlled split ────── stratified 80/20 on RAW trial indices,
   │   cached per subject (outputs/splits/), BEFORE anything below runs
   │
   ├─ Anti-aliased decimation ─────── scipy.signal.decimate(r=4, FIR, zero_phase)
   │   → 250 Hz → 62.5 Hz, shape: (n_trials, 5, 375)
   │
   ├─ Complex Morlet CWT (per channel) ─ custom implementation of Eq. (20), n=6,
   │   │   50 frequencies linearly spaced 0.5–20 Hz
   │   → REAL part kept, shape per channel: (50, 375)
   │
   ├─ Stack channels ─────────────── → shape: (50, 375, 5)
   │
   └─ Normalise to [−1, 1] ─────── per-trial min-max scaling (no cross-trial
       or cross-split leakage, since stats are computed per trial)
```

**Output per trial**: `(50, 375, 5)` float32 tensor
- Axis 0: 50 CWT frequencies, linearly spaced 0.5–20 Hz
- Axis 1: 375 time points (62.5 Hz effective rate, Nyquist = 31.25 Hz)
- Axis 2: 5 electrode channels

### Frequency Axis

Unlike a scale-based wavelet transform, the 50 rows of the scalogram correspond directly to a
known frequency (`config.CWT_FREQS = linspace(0.5, 20, 50)` Hz) — no scale-to-frequency
conversion is needed. Band groupings used by `band_analysis.py`:

| Band | Range |
|------|-------|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–20 Hz |

(Gamma, ≥30 Hz, falls outside the 0.5–20 Hz analysis band used by the primary CWT pipeline.)

---

## 5. Model Architectures

### 5.1 WGAN-GP Generator

The generator maps a 100-dimensional Gaussian noise vector to a `(50, 375, 5)` CWT scalogram.

```
Input: z ~ N(0, I₁₀₀)

Dense(8192) + BatchNorm + ReLU
Reshape → (4, 4, 512)

ConvTranspose2D(256, 5×5, stride=(2,2), padding=valid) + BN + ReLU  → (11, 11, 256)
ConvTranspose2D(128, 5×5, stride=(2,2), padding=valid) + BN + ReLU  → (25, 25, 128)
ConvTranspose2D( 64, 5×5, stride=(2,5), padding=same)  + BN + ReLU  → (50, 125,  64)
ConvTranspose2D(  5, 5×5, stride=(1,3), padding=same)  + tanh        → (50, 375,   5)

Output: x̂ ∈ [-1, 1]^(50×375×5)
```

**Parameters**: ~8.2M trainable

### 5.2 WGAN-GP Critic

The critic scores the realism of a CWT input, outputting a scalar (no sigmoid — Wasserstein distance is unbounded).

```
Input: x ∈ ℝ^(50×375×5)

Conv2D( 64, 5×5, strides=(2,3), padding=same) + LeakyReLU(α=0.2)  → (25, 125, 64)
Conv2D(128, 5×5, strides=(1,5), padding=same) + LeakyReLU(α=0.2)  → (25,  25, 128)
Dropout(0.2)
Flatten → Dense(1)   [NO sigmoid, NO BatchNorm]

Output: scalar ∈ ℝ  (Wasserstein score)
```

> **No BatchNorm in the critic** — required for valid gradient penalty computation. BatchNorm introduces dependency between samples in a batch, violating the per-sample Lipschitz constraint enforcement.

### 5.3 CNN Classifier

```
Input: (50, 375, 5)

Block 1: Conv2D(32, 7×7, valid) + BatchNorm + ReLU
         MaxPool(2×2) + Dropout(0.5) + L2(0.01)  → (22, 184, 32)

Block 2: Conv2D(32, 7×7, valid) + BatchNorm + ReLU
         MaxPool(2×2) + Dropout(0.5) + L2(0.01)  → (8, 89, 32)

Flatten → 22 784 units
Dense(750, relu, L2=0.01)
Dense(4, softmax)

Loss:      Sparse Categorical Cross-Entropy
Optimizer: Adam(lr=1×10⁻⁴)
```

**Parameters**: ~17.5M trainable

---

## 6. Training Procedure

### 6.1 WGAN-GP Training (Algorithm 1)

The WGAN-GP objective replaces the Jensen-Shannon divergence with the **Wasserstein-1 distance**, stabilising training. The gradient penalty enforces the 1-Lipschitz constraint on the critic.

**Critic loss** (maximise Wasserstein distance):

$$\mathcal{L}_C = \underbrace{\mathbb{E}_{\tilde{x} \sim P_g}[C(\tilde{x})]}_{\text{fake score}} - \underbrace{\mathbb{E}_{x \sim P_r}[C(x)]}_{\text{real score}} + \underbrace{\lambda \cdot \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1)^2]}_{\text{gradient penalty}}$$

where $\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}$, $\epsilon \sim U[0,1]$, and $\lambda = 10$.

**Generator loss** (minimise Wasserstein distance):

$$\mathcal{L}_G = -\mathbb{E}_{\tilde{x} \sim P_g}[C(\tilde{x})]$$

**Training loop** per epoch:
- For each batch: update critic **5 times** for every **1** generator update
- Gradient penalty computed in **fp32** regardless of mixed-precision policy

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Latent dimension | 100 |
| Batch size | 100 |
| Epochs | 300 |
| n_critic | 5 |
| λ (gradient penalty) | 10 |
| Optimiser (both) | Adam(lr=1e-4, β₁=0, β₂=0.9) |

### 6.2 CNN Training

- **Split**: stratified 80/20 on raw trial indices, performed BEFORE WGAN-GP training
  (`preprocess_subject_split`) — the held-out 20% is never used for WGAN-GP training,
  synthetic-sample generation, or CNN training
- **Input**: ~230 real training trials + 400 synthetic (100/class) = ~630 training samples
- **Evaluation**: the held-out ~58 real test trials
- **Callbacks**: EarlyStopping (patience=15), ReduceLROnPlateau (factor=0.5, patience=7), ModelCheckpoint (best val_accuracy)

---

## 7. Data Augmentation Strategies

Three augmentation strategies are compared:

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Rotation (GT)** | Rotate CWT image 180° along frequency and time axes | `np.flip(X, axis=(0,1))` |
| **Shifting (GT)** | Translate image by random offset, zero-pad | `np.roll` + mask |
| **Noise Addition** | $X'(t) = X(t) + \alpha \cdot \mathcal{N}(0, \sigma^2)$, $\alpha \sim U(0.1, 0.5)$ | `augmentation.py` |
| **WGAN-GP** | Train GAN per class, sample $G(z)$, $z \sim \mathcal{N}(0, I)$ | `train_wgan.py` |

Noise addition alone degraded classification performance. WGAN-GP produced the best results by learning the full data manifold rather than applying fixed transformations.

*Direct comparison of CNN accuracy trained on real data only vs. real + WGAN-GP synthetic data shows accuracy collapsing to near-chance (25%) without augmentation. Run `python src/ablation.py` to regenerate this figure locally (`outputs/figures/ablation_accuracy_grouped.png`).*

---

## 8. Results

> **Note**: Figures and result files under `outputs/` and `metrics/` are generated locally by
> running the pipeline (`run_all.py` or the individual `src/*.py` scripts) and are not
> distributed with this repository. The tables below reproduce the values reported in the
> manuscript; regenerate the figures locally to reproduce the plots.

### 8.1 Per-Subject Classification Accuracy

| Subject | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| S01 | 78.26% | 80.23% | 78.38% | 0.779 |
| S02 | 76.09% | 76.56% | 76.20% | 0.760 |
| S03 | 72.46% | 81.95% | 72.58% | 0.735 |
| S04 | 78.99% | 79.27% | 78.99% | 0.789 |
| S05 | 74.64% | 77.29% | 74.71% | 0.751 |
| S06 | 72.46% | 75.26% | 72.44% | 0.727 |
| S07 | 73.91% | 74.27% | 73.97% | 0.739 |
| S08 | 75.36% | 76.19% | 75.48% | 0.753 |
| S09 | 78.99% | 81.04% | 78.99% | 0.794 |
| **Mean** | **75.68%** | **78.01%** | **75.75%** | **0.759** |
| **Std** | ±2.44% | ±2.55% | ±2.44% | ±0.023 |

**Statistical significance vs 25% chance level:** t(8) = 58.70, p < 0.001 (one-sample t-test)

**95% Bootstrap CI (accuracy):** [74.15%, 77.29%] (n=2000 resamples)

*Per-subject accuracy with WGAN-GP augmentation (`outputs/figures/summary_accuracy_bar.png`, dashed line = 75.68% mean across 9 subjects) and the per-subject multi-metric radar chart (`outputs/figures/radar_chart.png`, Precision/Recall/F1/Accuracy) are generated locally — see the Note above.*

### 8.2 Ablation Study — Effect of Synthetic Augmentation

Removing synthetic data collapses performance to near-chance, demonstrating that WGAN-GP augmentation is **essential** — not merely incremental.

| Condition | Mean Accuracy | Mean F1 | vs Chance |
|-----------|--------------|---------|-----------|
| Real data only (288 trials) | 29.69% | 0.207 | +4.7% |
| Real + Synthetic (688 trials) | **75.68%** | **0.759** | **+50.7%** |
| **Δ (augmentation benefit)** | **+45.99%** | **+0.552** | — |

The CNN cannot generalise from only 72 trials per class. Synthetic augmentation provides the critical mass required for deep feature learning.

*`outputs/figures/ablation_accuracy_grouped.png` (regenerate via `src/ablation.py`) shows real-only CNN (orange) never exceeding ~32%, while real+synthetic CNN (blue) reaches 72–79% per subject.*

### 8.3 Synthetic Quality (FID Scores)

Fréchet Distance between real and synthetic CWT distributions (lower = better):

| Subject | Mean FID | Classification Accuracy |
|---------|----------|------------------------|
| S02 | 4,466 | 76.09% |
| S01 | 4,648 | 78.26% |
| S09 | 4,945 | 78.99% |
| S08 | 5,899 | 75.36% |
| S03 | 5,453 | 72.46% |
| S07 | 6,072 | 73.91% |
| S04 | 6,395 | 78.99% |
| S06 | 6,537 | 72.46% |
| S05 | 7,015 | 74.64% |
| **Mean** | **5,714** | **75.68%** |

> FID is computed in PCA feature space (256 components) rather than Inception-v3 space, as no domain-specific neural feature extractor exists for EEG scalograms. Values are comparable within this dataset but not to image-domain FID benchmarks.

*`outputs/figures/fid_mean_per_subject.png` and `fid_heatmap.png` (regenerate via `src/fid_score.py`) show mean and per-class Fréchet distance between real and synthetic CWT distributions per subject (lower = better). S02 has the lowest FID (4,466) and ranks 2nd in classification accuracy.*

### 8.4 Frequency Band Discriminability

Logistic regression probes trained on band-restricted CWT features (5-fold CV):

| Band | Frequency Range | Mean CV Accuracy |
|------|----------------|------------------|
| **Beta** | **13–20 Hz** | **~0.48** |
| **Alpha** | **8–13 Hz** | **~0.45** |
| Theta | 4–8 Hz | ~0.38 |
| Delta | 0.5–4 Hz | ~0.31 |

Beta and Alpha bands show the highest discriminability, consistent with the known **mu/beta Event-Related Desynchronisation (ERD)** phenomenon: motor imagery suppresses 8–30 Hz oscillations contralateral to the imagined movement.

*`outputs/figures/band_accuracy_heatmap.png` (regenerate via `src/band_analysis.py`) shows logistic regression probe accuracy per frequency band per subject. Alpha and Beta bands are consistently above chance (0.25).*

### 8.5 Cross-Subject Generalisation

Training on one subject and testing on another yields substantially lower accuracy (~25–35%), indicating strong subject-specificity of EEG patterns. Per-subject training is the recommended approach.

*`outputs/figures/cross_subject_heatmap_accuracy.png` (regenerate via `src/cross_eval.py`) shows the 9×9 generalisation matrix. Diagonal = per-subject accuracy (72–79%). Off-diagonal = cross-subject transfer, mostly near 25% chance.*

### 8.6 Visualisations

All figures below are regenerated locally by `run_all.py` / `src/analyze_results.py` into `outputs/figures/`:

- `synthetic_class_mean_s01.png` — mean synthetic CWT scalogram per MI class (Subject S01, Channel Fz): average of 100 generator-produced scalograms per class, colour-encoding normalised CWT magnitude [−1, 1]. Class-specific spectral patterns validate that the generator learned distinct per-class distributions.
- `synthetic_samples_s01.png` — three individual generator samples per MI class (Subject S01, Channel Fz), each column an independent draw from z ~ N(0, I₁₀₀).
- `synthetic_distribution.png` — histogram and KDE of synthetic CWT coefficient magnitudes across all 9 subjects and 4 classes, showing the generator reproduces the [−1, 1]-normalised EEG scalogram value distribution.
- `wgan_losses_all.png` — critic and generator loss curves for all 9 subjects across all 4 MI classes. Convergence visible by epoch ~150 for most subjects.

## 9. Project Structure

```
Motor-Imagery-Classification/
├── dataset/                  
│   ├── A01T.gdf … A09T.gdf       
│   └── A01E.gdf … A09E.gdf       
│
├── src/
│   ├── config.py                    
│   ├── preprocessing.py              
│   ├── augmentation.py              
│   ├── evaluate.py                 
│   ├── train_wgan.py               
│   ├── train_cnn.py                 
│   ├── cross_eval.py                
│   ├── analyze_results.py           
│   ├── fid_score.py                
│   ├── band_analysis.py              
│   ├── ablation.py                  
│   └── models/
│       ├── wgan_gp.py               
│       └── cnn.py                   
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  
│   ├── 02_wgan_training.ipynb     
│   └── 03_classification.ipynb      
│
├── outputs/
│   ├── synthetic/                
│   ├── models/                     
│   └── figures/                     
│
├── metrics/                        
├── run_all.py                       
├── TRAINING.md                     
└── requirements.txt
```

---

## 10. Installation & Quick Start

### Requirements

```
Python     3.11
TensorFlow 2.13.1
CUDA       11.8    (GPU)
cuDNN      8.x     (GPU)
```

### Install

```bash
git clone https://github.com/Chaganti-Reddy/Motor-Imagery-Classification
cd Motor-Imagery-Classification
pip install -r requirements.txt
```

Download the [BCI Competition IV 2a dataset](https://www.bbci.de/competition/iv/) and place `A01T.gdf … A09T.gdf` in `dataset/`.

### Run (GPU — WSL2 / Linux)

```bash
# Set GPU environment 
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/path/to/conda/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Full pipeline — all 9 subjects end-to-end
python run_all.py

# Or step-by-step for subject 1
python src/train_wgan.py --subject 1
python src/train_cnn.py  --subject 1
python src/analyze_results.py

# Additional analyses
python src/fid_score.py
python src/band_analysis.py
python src/ablation.py     
```

> See [TRAINING.md](TRAINING.md) for full GPU setup, common errors (XLA/libdevice, mixed-precision dtype mismatches, OOM), timing estimates, and troubleshooting guide.

### Notebooks

```
01_data_exploration.ipynb  →  02_wgan_training.ipynb  →  03_classification.ipynb
```

---

## 12. Limitations & Future Work

- **5-channel constraint**: Only EEG-Fz, C3, Cz, C4, Oz are used due to RAM constraints during CWT computation. Using all 22 electrodes would likely improve spatial resolution and accuracy.
- **Subject specificity**: Cross-subject accuracy remains low (~25–35%). Domain adaptation (e.g., domain-adversarial neural networks, subject-invariant feature learning) could improve generalisation.
- **FID-gated augmentation**: Some subjects (S05, S07 in ablation) show marginal benefit or slight degradation from synthetic augmentation. Per-subject GAN quality gating via FID thresholds could selectively apply augmentation only where beneficial.
- **Fixed augmentation count**: 100 synthetic samples per class is fixed; no sensitivity sweep over this ratio was performed. Tuning the amount of synthetic data per class (e.g., via a validation-based sweep) remains unexplored.
- **Temporal features**: The CNN operates on 2D CWT images. Architectures that explicitly model temporal dynamics (e.g., EEGNet, ShallowConvNet, Transformers) may capture complementary information.
- **No independent-day (session E) evaluation**: only session `T` is used; session `E` is excluded because its GDF files carry only an unknown-class event code, and the separate true-label file distributed by the BCI Competition IV organizers is not included in this dataset copy. Evaluating cross-session (different-day) generalization on session E, once true labels are obtained, remains an important direction for future work.

---

<!-- ## 13. Citation

```bibtex
@article{mann2025motor,
  title   = {Motor Imagery Classification Performance Enhancement with EEG
             Data Augmentation via Wasserstein Generative Adversarial Networks},
  author  = {Mukesh Mann and Rakesh P. Badoni and Alok Sharma and Chaganti Venkatarami Reddy
             and Predrag S. Stanimirovi{\'{c}} and Deepak Panwar},
  year    = {2025}
}
```
-->

See [CITATION.cff](CITATION.cff) to cite this repository/code directly. The archived,
citable code release (v1.0.0) is on Zenodo: [10.5281/zenodo.22870348](https://doi.org/10.5281/zenodo.22870348).

## Contributors

1. Mukesh Mann — IIIT Sonepat

2. Rakesh P. Badoni — Mahindra University

3. Alok Sharma — IIIT Sonepat

4. Chaganti Venkatarami Reddy — IIIT Sonepat

5. Predrag S. Stanimirović — University of Niš

6. Deepak Panwar — Manipal University Jaipur


## License

This project is licensed under the MIT License © [Chaganti Reddy](https://github.com/Chaganti-Reddy/Motor-Imagery-Classification/blob/main/LICENSE).
