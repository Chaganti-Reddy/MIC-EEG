from __future__ import annotations

import numpy as np
import mne
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATASET_DIR, ELECTRODE_NAMES, TMIN, TMAX,
    N_TIME_SUBSAMPLE, DOWNSAMPLE_FACTOR, EFFECTIVE_SAMPLING_RATE,
    N_CWT_SCALES, CWT_FREQS, CWT_N_CYCLES,
    EVENT_IDS, TRIAL_SHAPE, N_CLASSES,
    TEST_SPLIT, RANDOM_SEED, SYNTHETIC_DIR,
)

mne.set_log_level("WARNING")

_N_RAW_SAMPLES = N_TIME_SUBSAMPLE * DOWNSAMPLE_FACTOR   # 375 * 4 = 1500 samples (6 s @ 250 Hz)

SPLIT_DIR = SYNTHETIC_DIR.parent / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def gdf_path(subject: int, session: str = "T") -> Path:
    """Return path to A0<subject><session>.gdf."""
    return DATASET_DIR / f"A{subject:02d}{session}.gdf"


def load_raw(subject: int, session: str = "T") -> mne.io.BaseRaw:
    path = gdf_path(subject, session)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)
    raw.filter(l_freq=0.5, h_freq=100.0, method="fir", verbose=False)
    raw.notch_filter(freqs=50.0, method="fir", verbose=False)

    # BCI Competition IV 2a: 22 EEG channels + 3 EOG.
    # MNE renames duplicate bare "EEG" channels to EEG-0 … EEG-16, so the
    # standard names (EEG-Fz, EEG-C3, EEG-Cz, EEG-C4, EEG-Oz) may not all
    # be present.  Pick by fixed index within the EEG channels instead.
    #
    # BCI IV 2a channel order (0-based within EEG-only channels):
    #   0=Fz, 1=FC3, 2=FC1, 3=FCz, 4=FC2, 5=FC4,
    #   6=C5,  7=C3,  8=C1,  9=Cz, 10=C2, 11=C4,
    #  12=C6, 13=CP3, 14=CP1, 15=CPz, 16=CP2, 17=CP4,
    #  18=P1, 19=Pz,  20=P2,  21=Oz
    EEG_TARGET_INDICES = [0, 7, 9, 11, 21]   # Fz, C3, Cz, C4, Oz

    missing = set(ELECTRODE_NAMES) - set(raw.ch_names)
    if not missing:
        raw.pick_channels(ELECTRODE_NAMES, ordered=True)
    else:
        eeg_chs = [ch for ch in raw.ch_names if ch.startswith("EEG")]
        if len(eeg_chs) < 22:
            raise RuntimeError(
                f"Expected ≥22 EEG channels, found {len(eeg_chs)}.\n"
                f"Available: {raw.ch_names}"
            )
        selected = [eeg_chs[i] for i in EEG_TARGET_INDICES]
        raw.pick_channels(selected, ordered=True)
        rename_map = dict(zip(selected, ELECTRODE_NAMES))
        raw.rename_channels(rename_map)

    return raw


def extract_epochs(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray]:
    events, event_id_map = mne.events_from_annotations(raw, verbose=False)

    mi_event_id = {
        k: v for k, v in event_id_map.items()
        if int(k) in EVENT_IDS
    }
    if not mi_event_id:
        raise RuntimeError(
            "No MI class events (769-772) found in annotations.\n"
            f"Available annotations: {list(event_id_map.keys())}"
        )

    artifact_codes = {v for k, v in event_id_map.items() if int(k) == 1023}
    artifact_positions: set[int] = set()
    if artifact_codes:
        for code in artifact_codes:
            positions = events[events[:, 2] == code, 0]
            artifact_positions.update(int(p) for p in positions)

    mi_mask = np.isin(events[:, 2], list(mi_event_id.values()))
    mi_evs  = events[mi_mask]

    if artifact_positions:
        clean_mask = ~np.isin(mi_evs[:, 0], list(artifact_positions))
        n_removed  = int(np.sum(~clean_mask))
        mi_evs     = mi_evs[clean_mask]
        if n_removed:
            print(f"    ↳ Removed {n_removed} artifact-marked trial(s)")

    _, uniq = np.unique(mi_evs[:, 0], return_index=True)
    mi_evs  = mi_evs[uniq]

    epochs = mne.Epochs(
        raw, mi_evs,
        event_id=mi_event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epochs.drop_bad(verbose=False)

    X = epochs.get_data()                               

    inv_map = {v: int(k) for k, v in mi_event_id.items()}
    y = np.array([EVENT_IDS[inv_map[code]] for code in epochs.events[:, 2]])

    return X, y


def _decimate_trial(trial: np.ndarray) -> np.ndarray:
    trial = trial[:, :_N_RAW_SAMPLES]
    trial_sub = decimate(trial, DOWNSAMPLE_FACTOR, ftype="fir", zero_phase=True, axis=-1)
    return trial_sub[:, :N_TIME_SUBSAMPLE]


def _complex_morlet_cwt(signal: np.ndarray, freqs: np.ndarray, fs: float,
                         n_cycles: int = CWT_N_CYCLES) -> np.ndarray:
    """
    Complex Morlet CWT following Eq. (20):
        psi(t) = exp(2i*pi*f*t) * exp(-t^2 / (2*sigma^2)),   sigma = n_cycles / (2*pi*f)

    Parameters
    ----------
    signal   : ndarray  (n_times,)   single-channel, real-valued, sampled at `fs` Hz
    freqs    : ndarray  (n_freqs,)   analysis frequencies in Hz
    fs       : float                 sampling rate of `signal`, in Hz
    n_cycles : int                   Morlet "n" parameter (bandwidth/resolution trade-off)

    Returns
    -------
    coeffs : ndarray complex  (n_freqs, n_times)
    """
    n_times = signal.shape[-1]
    coeffs = np.empty((len(freqs), n_times), dtype=np.complex128)
    for i, f in enumerate(freqs):
        sigma = n_cycles / (2.0 * np.pi * f)
        half_len = max(1, int(np.ceil(4.0 * sigma * fs)))
        t = np.arange(-half_len, half_len + 1) / fs
        wavelet = np.exp(2j * np.pi * f * t) * np.exp(-(t ** 2) / (2.0 * sigma ** 2))
        wavelet = wavelet / np.sqrt(sigma * np.sqrt(np.pi))     # unit-energy normalisation
        conv = np.convolve(signal, wavelet, mode="same")
        coeffs[i] = conv
    return coeffs


def cwt_trial(trial: np.ndarray) -> np.ndarray:
    """
    Compute the complex Morlet CWT for a single trial and keep the real part (Sec. 2.2.6).

    Parameters
    ----------
    trial : ndarray  (n_channels, n_times)   raw trial at the original 250 Hz sampling rate

    Returns
    -------
    cwt_image : ndarray  (50, 375, 5)   float32
    """
    n_channels, _ = trial.shape
    trial_sub = _decimate_trial(trial)   # (C, 375) @ 62.5 Hz

    channels = []
    for ch in range(n_channels):
        coeffs = _complex_morlet_cwt(trial_sub[ch], CWT_FREQS, EFFECTIVE_SAMPLING_RATE)
        channels.append(np.real(coeffs).astype(np.float32))

    cwt_image = np.stack(channels, axis=-1).astype(np.float32)  # (50, 375, 5)
    return cwt_image


def normalise_dataset(X: np.ndarray) -> np.ndarray:
    """
    Per-sample min-max normalisation to [-1, 1].

    Normalization is required for the tanh generator output to be comparable
    with real data during WGAN-GP training.
    """
    mn = X.min(axis=(1, 2, 3), keepdims=True)
    mx = X.max(axis=(1, 2, 3), keepdims=True)
    return (2.0 * (X - mn) / (mx - mn + 1e-8) - 1.0).astype(np.float32)


def load_epochs(
    subject: int,
    session: str = "T",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw trials and labels for one subject/session, before any CWT
    processing. This is the pre-split representation used by
    `preprocess_subject_split` so that partitioning happens before CWT
    (Sec. 3.1).
    """
    if verbose:
        print(f"  Loading subject {subject} session {session} …", end=" ", flush=True)

    raw  = load_raw(subject, session)
    X, y = extract_epochs(raw)

    if verbose:
        print(f"{X.shape[0]} trials")

    return X, y


def _to_cwt(X: np.ndarray) -> np.ndarray:
    """Compute per-trial complex Morlet CWT + [-1, 1] normalisation for a batch of raw trials."""
    cwt_list = [cwt_trial(t) for t in X]
    X_cwt    = np.stack(cwt_list, axis=0)   # (N, 50, 375, 5)
    return normalise_dataset(X_cwt)


def preprocess_subject(
    subject: int,
    session: str = "T",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for one subject/session (no train/test split).

    Parameters
    ----------
    subject : int   1–9
    session : str   'T' (training) or 'E' (evaluation)

    Returns
    -------
    X_cwt : ndarray  (n_trials, 50, 375, 5)   normalised to [-1, 1]
    y     : ndarray  (n_trials,)               0-indexed class labels
    """
    X, y  = load_epochs(subject, session, verbose=False)
    X_cwt = _to_cwt(X)

    if verbose:
        print(f"  Subject {subject} session {session}: "
              f"{X_cwt.shape[0]} trials  shape={X_cwt.shape[1:]}")

    return X_cwt, y


def _get_or_create_split_indices(subject: int, session: str, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    split_path = SPLIT_DIR / f"split_s{subject:02d}_{session}.npz"
    if split_path.exists():
        cached = np.load(split_path)
        if len(cached["idx_train"]) + len(cached["idx_test"]) == len(y):
            return cached["idx_train"], cached["idx_test"]

    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y,
    )
    np.savez(split_path, idx_train=idx_train, idx_test=idx_test)
    return idx_train, idx_test


def preprocess_subject_split(
    subject: int,
    session: str = "T",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition raw trials into training/held-out partitions BEFORE CWT
    processing (Sec. 3.1). CWT and per-trial [-1, 1] normalisation are then
    computed separately for each partition using the same fixed Morlet
    wavelet parameters.
    """
    X, y = load_epochs(subject, session=session, verbose=verbose)
    idx_train, idx_test = _get_or_create_split_indices(subject, session, y)

    X_train_cwt = _to_cwt(X[idx_train])
    X_test_cwt  = _to_cwt(X[idx_test])

    if verbose:
        print(f"    ↳ leakage-controlled split: {len(idx_train)} train / {len(idx_test)} test "
              f"(seed={RANDOM_SEED}); CWT computed separately per partition")

    return X_train_cwt, y[idx_train], X_test_cwt, y[idx_test]


def split_by_class(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """Return dict mapping class index → subset of X."""
    return {cls: X[y == cls] for cls in range(N_CLASSES)}
