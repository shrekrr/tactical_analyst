"""
Temporal Transformer for formation / tactical-phase classification.

Architecture
------------
Player Coordinates (T, N, F)
        ↓
Linear Feature Projection → d_model
        ↓
Positional Encoding (sinusoidal)
        ↓
Transformer Encoder (L layers, H heads)
        ↓
Global Average Pooling over time
        ↓
Classification Head (MLP)
        ↓
Formation label + confidence

Input tensor shape: (batch, T, N * F)
    T = sequence length (number of frames)
    N = max players per team (padded, default 11)
    F = features per player (x, y, vx, vy)

The model is architecture-correct but requires training data to be useful.
Without trained weights, the system falls back to the rule-based baseline.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from app.analytics.formation import KNOWN_FORMATIONS


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the time dimension."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalFormationTransformer(nn.Module):
    """
    Transformer-based formation classifier.

    Parameters
    ----------
    n_players : int
        Maximum number of players per team (padded if fewer detected).
    n_features : int
        Features per player: x, y, vx, vy (default 4).
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of encoder layers.
    n_classes : int
        Number of formation classes.
    seq_len : int
        Input sequence length (number of frames).
    dropout : float
    """

    def __init__(
        self,
        n_players: int = 11,
        n_features: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = len(KNOWN_FORMATIONS),
        seq_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes
        self.seq_len = seq_len

        input_dim = n_players * n_features

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, N * F).

        Returns
        -------
        logits : torch.Tensor
            Shape (batch, n_classes).
        """
        # Project input
        x = self.input_proj(x)             # (batch, T, d_model)
        x = self.pos_enc(x)               # add positional encoding
        x = self.transformer(x)           # (batch, T, d_model)
        x = x.mean(dim=1)                 # global average pooling over T
        logits = self.classifier(x)       # (batch, n_classes)
        return logits


class TransformerFormationClassifier:
    """
    Inference-only wrapper around TemporalFormationTransformer.

    Usage
    -----
    clf = TransformerFormationClassifier.load("models/transformer.pt")
    formation, confidence, explanation = clf.predict(sequence_array)
    """

    def __init__(self, model: TemporalFormationTransformer, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.labels = KNOWN_FORMATIONS

    @classmethod
    def load(cls, path: str | Path) -> "TransformerFormationClassifier":
        import inspect
        path = Path(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(str(path), map_location=device)
        config = state.get("config", {})
        valid_keys = set(inspect.signature(TemporalFormationTransformer.__init__).parameters.keys())
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        model = TemporalFormationTransformer(**filtered_config)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded Temporal Transformer from {}", path)
        return cls(model, device)

    def predict(
        self, sequence: np.ndarray
    ) -> Tuple[str, float, Dict]:
        """
        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, N * F) — single sequence (no batch dim).

        Returns
        -------
        (formation, confidence, explanation)
        """
        x = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())
        formation = self.labels[idx] if idx < len(self.labels) else "unknown"

        explanation = {
            "method": "temporal_transformer",
            "formation": formation,
            "confidence": conf,
            "top_3": [
                {"formation": self.labels[i], "probability": float(probs[i].item())}
                for i in probs.topk(min(3, len(self.labels))).indices.tolist()
            ],
        }
        return formation, conf, explanation
