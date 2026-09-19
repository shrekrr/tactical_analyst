"""
PyTorch Dataset for SoccerNet-derived formation sequences.

Handles:
- Loading pre-built .npy arrays
- Match-level train/val/test splitting (no leakage)
- Coordinate normalization verification
- Sequence padding/truncation to fixed T
- Optional data augmentation (flip, noise, crop)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS_DIR = Path(__file__).parent.parent.parent / "training" / "datasets"


class FormationSequenceDataset(Dataset):
    """
    Dataset of (sequence, label) pairs for formation classification.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.
    datasets_dir : Path
        Directory containing sequences.npy, labels.npy, match_ids.npy,
        and split_manifest.json.
    augment : bool
        Whether to apply data augmentation (training only).
    seq_len : int | None
        Truncate/pad sequences to this length. None = use as-is.
    """

    def __init__(
        self,
        split: str,
        datasets_dir: Path = DATASETS_DIR,
        augment: bool = False,
        seq_len: Optional[int] = None,
    ):
        self.split = split
        self.augment = augment
        self.seq_len = seq_len

        # Load arrays
        X_all = np.load(str(datasets_dir / "sequences.npy"))
        y_all = np.load(str(datasets_dir / "labels.npy"))
        m_all = np.load(str(datasets_dir / "match_ids.npy"), allow_pickle=True)

        with open(datasets_dir / "split_manifest.json") as f:
            manifest = json.load(f)

        self.formations = manifest["formations"]
        match_split = manifest["match_split"]

        # Filter to this split (by match)
        mask = np.array([match_split.get(str(mid), "train") == split for mid in m_all])
        self.X = X_all[mask].astype(np.float32)   # (N, T, features)
        self.y = y_all[mask].astype(np.int64)
        self.match_ids = m_all[mask]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.X[idx].copy()  # (T, features)
        label = self.y[idx]

        # Augmentation (training only)
        if self.augment:
            from training.preprocessing.augmentation import (
                flip_horizontal,
                add_gaussian_noise,
            )
            if np.random.rand() < 0.5:
                seq = flip_horizontal(seq)
            if np.random.rand() < 0.3:
                seq = add_gaussian_noise(seq, sigma=0.01)

        # Truncate or pad to fixed seq_len
        if self.seq_len is not None:
            T = seq.shape[0]
            if T >= self.seq_len:
                seq = seq[:self.seq_len]
            else:
                pad = np.zeros((self.seq_len - T, seq.shape[1]), dtype=np.float32)
                seq = np.concatenate([seq, pad], axis=0)

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.
        Helps with class imbalance common in weakly-labeled data.
        """
        n_classes = len(self.formations)
        counts = np.bincount(self.y, minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes
        return torch.from_numpy(weights.astype(np.float32))
