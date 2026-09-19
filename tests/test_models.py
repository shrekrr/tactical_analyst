"""
Unit tests for model forward passes.

Verifies correct output shapes for MLP, LSTM, and Transformer.
No training required — just checks that models accept input without errors.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.analytics.formation import KNOWN_FORMATIONS
from app.models.mlp_model import MLPFormationClassifier
from app.models.lstm_model import LSTMFormationClassifier
from app.models.temporal_model import TemporalFormationTransformer

N_CLASSES = len(KNOWN_FORMATIONS)
BATCH = 4
SEQ_LEN = 20
N_PLAYERS = 11
N_FEATURES = 4
INPUT_DIM = N_PLAYERS * N_FEATURES


def make_random_input(batch=BATCH, seq_len=SEQ_LEN, dim=INPUT_DIM) -> torch.Tensor:
    """Create a random batch of sequences with normalised values."""
    return torch.rand(batch, seq_len, dim)  # [0, 1) matches normalised coords


class TestMLPForwardPass:
    def test_output_shape(self):
        model = MLPFormationClassifier(
            seq_len=SEQ_LEN, n_players=N_PLAYERS, n_features=N_FEATURES,
            n_classes=N_CLASSES,
        )
        x = make_random_input()
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_batch_size_one(self):
        model = MLPFormationClassifier(seq_len=SEQ_LEN, n_players=N_PLAYERS,
                                       n_features=N_FEATURES, n_classes=N_CLASSES)
        x = make_random_input(batch=1)
        logits = model(x)
        assert logits.shape == (1, N_CLASSES)

    def test_no_nan_output(self):
        model = MLPFormationClassifier(seq_len=SEQ_LEN, n_players=N_PLAYERS,
                                       n_features=N_FEATURES, n_classes=N_CLASSES)
        x = make_random_input()
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_padded_zeros_ok(self):
        """Models must handle padding (zero-filled player slots)."""
        model = MLPFormationClassifier(seq_len=SEQ_LEN, n_players=N_PLAYERS,
                                       n_features=N_FEATURES, n_classes=N_CLASSES)
        x = torch.zeros(BATCH, SEQ_LEN, INPUT_DIM)
        x[:, :, :3 * N_FEATURES] = torch.rand(BATCH, SEQ_LEN, 3 * N_FEATURES)
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)


class TestLSTMForwardPass:
    def test_output_shape(self):
        model = LSTMFormationClassifier(n_players=N_PLAYERS, n_features=N_FEATURES,
                                        n_classes=N_CLASSES)
        x = make_random_input()
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_single_timestep(self):
        model = LSTMFormationClassifier(n_players=N_PLAYERS, n_features=N_FEATURES,
                                        n_classes=N_CLASSES)
        x = make_random_input(seq_len=1)
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_no_nan_output(self):
        model = LSTMFormationClassifier(n_players=N_PLAYERS, n_features=N_FEATURES,
                                        n_classes=N_CLASSES)
        x = make_random_input()
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_long_sequence(self):
        """LSTM should handle sequences longer than training length."""
        model = LSTMFormationClassifier(n_players=N_PLAYERS, n_features=N_FEATURES,
                                        n_classes=N_CLASSES)
        x = make_random_input(seq_len=100)
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)


class TestTransformerForwardPass:
    def test_output_shape(self):
        model = TemporalFormationTransformer(
            n_players=N_PLAYERS, n_features=N_FEATURES,
            n_classes=N_CLASSES, seq_len=SEQ_LEN,
        )
        x = make_random_input()
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_no_nan_output(self):
        model = TemporalFormationTransformer(
            n_players=N_PLAYERS, n_features=N_FEATURES,
            n_classes=N_CLASSES, seq_len=SEQ_LEN,
        )
        x = make_random_input()
        logits = model(x)
        assert not torch.isnan(logits).any()

    def test_padded_zeros_ok(self):
        model = TemporalFormationTransformer(
            n_players=N_PLAYERS, n_features=N_FEATURES,
            n_classes=N_CLASSES, seq_len=SEQ_LEN,
        )
        x = torch.zeros(BATCH, SEQ_LEN, INPUT_DIM)
        logits = model(x)
        assert logits.shape == (BATCH, N_CLASSES)

    def test_batch_size_one(self):
        model = TemporalFormationTransformer(
            n_players=N_PLAYERS, n_features=N_FEATURES,
            n_classes=N_CLASSES, seq_len=SEQ_LEN,
        )
        x = make_random_input(batch=1)
        logits = model(x)
        assert logits.shape == (1, N_CLASSES)
