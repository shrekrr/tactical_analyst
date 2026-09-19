"""
MLP baseline for formation classification.

No temporal modeling — treats each sequence as a flattened feature vector.
This is the weakest baseline, used to measure how much temporal structure matters.

Input:  (batch, T * N * F) — the entire sequence flattened
Output: (batch, n_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.analytics.formation import KNOWN_FORMATIONS


class MLPFormationClassifier(nn.Module):
    """
    Multi-Layer Perceptron baseline for formation classification.

    Architecture
    ------------
    Flatten(T, N*F) → [T*N*F]
    Linear + BatchNorm + GELU + Dropout
    Linear + BatchNorm + GELU + Dropout
    Linear + BatchNorm + GELU
    Linear → logits

    The flatten operation discards all temporal ordering.
    Compare its results against LSTM/Transformer to quantify temporal benefit.
    """

    def __init__(
        self,
        seq_len: int = 20,
        n_players: int = 11,
        n_features: int = 4,
        hidden_dim: int = 256,
        n_classes: int = len(KNOWN_FORMATIONS),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_players = n_players
        self.n_features = n_features
        input_dim = seq_len * n_players * n_features

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),

            nn.Linear(hidden_dim // 4, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (batch, T, N*F)
        Returns logits (batch, n_classes)
        """
        return self.net(x)
