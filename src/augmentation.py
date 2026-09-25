"""
Data augmentation methods .

Implemented methods
-------------------
1. Geometric Transformation (GT)
   - Rotation by 180°      
   - Shifting (up/down/left/right) 

2. Noise Addition (NA)
   - Gaussian noise with scaling factor α ~ U(0.1, 0.5) 

All functions accept and return CWT images of shape (..., 50, 375, 5).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import NOISE_ALPHA_LOW, NOISE_ALPHA_HIGH


def rotate_180(sample: np.ndarray) -> np.ndarray:
    """
    Rotate CWT image by 180°. Equivalent to simultaneous flip along both spatial axes.

    Parameters
    ----------
    sample : ndarray  (50, 375, 5)

    Returns
    -------
    rotated : ndarray  same shape
    """
    return sample[::-1, ::-1, :].copy()


def shift(
    sample: np.ndarray,
    delta_y: int = 5,
    delta_x: int = 10,
) -> np.ndarray:
    """
    Translate CWT image by (delta_y, delta_x), padding with zeros.

    Parameters
    ----------
    sample  : ndarray  (H, W, C)
    delta_y : vertical shift along frequency axis (positive = down)
    delta_x : horizontal shift along time axis    (positive = right)

    Returns
    -------
    shifted : ndarray  same shape, out-of-bounds positions filled with 0
    """
    H, W, C  = sample.shape
    shifted  = np.zeros_like(sample)

    src_y0 = max(0, -delta_y);  src_y1 = min(H, H - delta_y)
    dst_y0 = max(0,  delta_y);  dst_y1 = min(H, H + delta_y)
    src_x0 = max(0, -delta_x);  src_x1 = min(W, W - delta_x)
    dst_x0 = max(0,  delta_x);  dst_x1 = min(W, W + delta_x)

    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1, :] = \
            sample[src_y0:src_y1, src_x0:src_x1, :]

    return shifted


def add_gaussian_noise(
    sample: np.ndarray,
    alpha: Optional[float] = None,
    sigma: float = 0.1,
) -> np.ndarray:
    """
    X'(t) = X(t) + α · N(0, σ²)   (Eq. 6)

    α ~ U(0.1, 0.5) if not specified.

    Parameters
    ----------
    sample : ndarray  (50, 375, 5)   values expected in [-1, 1]
    alpha  : noise scaling factor; sampled from U(0.1, 0.5) if None
    sigma  : standard deviation of Gaussian noise

    Returns
    -------
    noisy : ndarray  same shape, clipped to [-1, 1]
    """
    if alpha is None:
        alpha = float(np.random.uniform(NOISE_ALPHA_LOW, NOISE_ALPHA_HIGH))
    noise = np.random.normal(0.0, sigma, size=sample.shape).astype(sample.dtype)
    return np.clip(sample + alpha * noise, -1.0, 1.0)


def augment_batch(X: np.ndarray, method: str = "rotate") -> np.ndarray:
    """
    Apply an augmentation to every sample in a batch.

    Parameters
    ----------
    X      : ndarray  (N, 50, 375, 5)
    method : 'rotate' | 'shift' | 'noise'

    Returns
    -------
    augmented : ndarray  same shape
    """
    if method == "rotate":
        return np.stack([rotate_180(s) for s in X])
    elif method == "noise":
        return np.stack([add_gaussian_noise(s) for s in X])
    elif method == "shift":
        dy = int(np.random.randint(-10, 11))
        dx = int(np.random.randint(-20, 21))
        return np.stack([shift(s, dy, dx) for s in X])
    else:
        raise ValueError(f"Unknown augmentation method '{method}'. "
                         f"Choose from: 'rotate', 'shift', 'noise'")


def generate_augmented_set(
    X: np.ndarray,
    methods: list[str] | None = None,
) -> np.ndarray:
    """
    Generate one augmented copy per method and concatenate.

    Parameters
    ----------
    X       : ndarray  (N, 50, 375, 5)  real samples
    methods : list of method names; defaults to ['rotate', 'shift', 'noise']

    Returns
    -------
    X_aug : ndarray  (N * len(methods), 50, 375, 5)
    """
    if methods is None:
        methods = ["rotate", "shift", "noise"]
    return np.concatenate([augment_batch(X, m) for m in methods], axis=0)
