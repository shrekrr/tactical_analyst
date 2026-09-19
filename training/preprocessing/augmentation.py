"""
Data augmentation for formation sequences.

All augmentations preserve formation labels.

augmentation     |  changes                    | label-preserving?
─────────────────|─────────────────────────────|──────────────────
flip_horizontal  |  y coordinate mirrored      |  YES (symmetric)
add_gaussian_noise | small random noise on x,y  |  YES
random_time_crop |  sub-window of sequence     |  YES
"""
from __future__ import annotations

import numpy as np

N_PLAYERS = 11
N_FEATURES = 4  # x, y, vx, vy


def flip_horizontal(seq: np.ndarray) -> np.ndarray:
    """
    Mirror the pitch left-right: y -> (1 - y), vy -> -vy.

    Formation labels are invariant to horizontal flip (a 4-3-3 mirrored is still 4-3-3).

    Parameters
    ----------
    seq : np.ndarray shape (T, N*F)
    """
    seq = seq.copy()
    for i in range(N_PLAYERS):
        base = i * N_FEATURES
        seq[:, base + 1] = 1.0 - seq[:, base + 1]   # y → 1-y
        seq[:, base + 3] = -seq[:, base + 3]          # vy → -vy
    return seq


def add_gaussian_noise(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    Add small Gaussian noise to position and velocity features.

    Simulates GPS/tracking jitter. Helps prevent overfitting to exact cluster
    positions produced by the rule-based labeler.
    """
    noise = np.random.randn(*seq.shape).astype(np.float32) * sigma
    noisy = seq + noise
    # Re-clip normalized coords to [0, 1] and velocities to [-1, 1]
    for i in range(N_PLAYERS):
        base = i * N_FEATURES
        noisy[:, base]     = np.clip(noisy[:, base], 0.0, 1.0)      # x
        noisy[:, base + 1] = np.clip(noisy[:, base + 1], 0.0, 1.0)  # y
        noisy[:, base + 2] = np.clip(noisy[:, base + 2], -1.0, 1.0) # vx
        noisy[:, base + 3] = np.clip(noisy[:, base + 3], -1.0, 1.0) # vy
    return noisy


def random_time_crop(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Randomly crop a sub-window of length target_len from the sequence.

    If seq is shorter than target_len, it is zero-padded on the right.
    """
    T = seq.shape[0]
    if T <= target_len:
        pad = np.zeros((target_len - T, seq.shape[1]), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)
    start = np.random.randint(0, T - target_len)
    return seq[start:start + target_len].copy()
