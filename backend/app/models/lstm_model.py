"""
Bidirectional LSTM baseline for formation classification.

Captures sequential temporal structure (recurrent inductive bias).
Compare against MLP (no temporal) and Transformer (attention-based temporal).

Input:  (batch, T, N * F)
Output: (batch, n_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.analytics.formation import KNOWN_FORMATIONS


class LSTMFormationClassifier(nn.Module):
    """
    Bidirectional LSTM formation classifier.

    Architecture
    ------------
    (batch, T, N*F)
        ↓
    Linear input projection → lstm_input_dim
        ↓
    Bidirectional LSTM (n_layers layers)
        ↓
    Concat (forward_last_hidden, backward_last_hidden)
        ↓
    MLP classification head
        ↓
    logits (n_classes)

    Bidirectional captures both past context (useful for defensive shape)
    and future context (useful for detecting build-up → attack transitions).
    """

    def __init__(
        self,
        n_players: int = 11,
        n_features: int = 4,
        lstm_input_dim: int = 64,
        lstm_hidden_dim: int = 128,
        n_layers: int = 2,
        n_classes: int = len(KNOWN_FORMATIONS),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_features = n_features
        input_dim = n_players * n_features

        # Project input to LSTM input dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, lstm_input_dim),
            nn.LayerNorm(lstm_input_dim),
            nn.GELU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # After LSTM: concat both directions → 2 * lstm_hidden_dim
        lstm_out_dim = lstm_hidden_dim * 2

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (batch, T, N*F)
        Returns logits (batch, n_classes)
        """
        x = self.input_proj(x)          # (batch, T, lstm_input_dim)
        out, (h_n, _) = self.lstm(x)    # h_n: (n_layers * 2, batch, hidden)

        # Take the last layer's hidden states for both directions
        # h_n[-2] = forward  last layer
        # h_n[-1] = backward last layer
        forward_h = h_n[-2]             # (batch, hidden)
        backward_h = h_n[-1]            # (batch, hidden)
        combined = torch.cat([forward_h, backward_h], dim=-1)  # (batch, 2*hidden)

        return self.classifier(combined)
