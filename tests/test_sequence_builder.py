"""
Unit tests for sequence building and match-level split integrity.

Tests: shape, normalization range, no leakage, augmentation validity.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

N_FEATURES = 4   # x, y, vx, vy per player

from create_sequences import (
    build_team_sequence,
    split_by_match,
    label_window,
)
from app.analytics.formation import KNOWN_FORMATIONS


sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class TestBuildTeamSequence:
    def _make_frames(self, n_frames=25, n_players=8):
        """Generate fake frames dict."""
        frames = {}
        for f in range(n_frames):
            frames[f] = [
                {"track_id": i, "x": float(10 + i * 5), "y": float(20 + i * 3)}
                for i in range(n_players)
            ]
        return frames

    def test_output_shape(self):
        frames = self._make_frames(25, 8)
        frame_ids = list(range(20))
        seq = build_team_sequence(frames, frame_ids, fps=5.0)
        assert seq.shape == (20, 11 * 4)  # T=20, N=11, F=4

    def test_normalized_x_range(self):
        frames = self._make_frames(25, 8)
        frame_ids = list(range(20))
        seq = build_team_sequence(frames, frame_ids, fps=5.0)
        x_values = seq[:, 0::4]   # every 4th feature starting at 0 is x
        assert x_values.min() >= 0.0
        assert x_values.max() <= 1.01   # small float tolerance

    def test_normalized_y_range(self):
        frames = self._make_frames(25, 8)
        frame_ids = list(range(20))
        seq = build_team_sequence(frames, frame_ids, fps=5.0)
        y_values = seq[:, 1::4]
        assert y_values.min() >= 0.0
        assert y_values.max() <= 1.01

    def test_velocity_clipped(self):
        frames = self._make_frames(25, 8)
        frame_ids = list(range(20))
        seq = build_team_sequence(frames, frame_ids, fps=5.0)
        vx = seq[:, 2::4]
        vy = seq[:, 3::4]
        assert vx.min() >= -1.0
        assert vx.max() <= 1.0
        assert vy.min() >= -1.0
        assert vy.max() <= 1.0

    def test_missing_frame_zero_padded(self):
        """Frames missing from dict should produce zero-position rows (padding).
        Velocities for those rows may be non-zero (derived from prev position)."""
        frames = {0: [{"track_id": 1, "x": 50.0, "y": 34.0}]}
        frame_ids = [0, 1, 2]  # frames 1 and 2 don't exist
        seq = build_team_sequence(frames, frame_ids, fps=5.0)
        assert seq.shape == (3, 11 * 4)
        # Player positions (x, y) in rows 1 and 2 should be zero (no players detected)
        # Velocities may be non-zero (negative, from prev_pos - 0 transition)
        for i in range(11):
            base = i * N_FEATURES
            assert seq[1, base] == 0.0,    f"x for player {i} in frame 1 should be 0"
            assert seq[1, base + 1] == 0.0, f"y for player {i} in frame 1 should be 0"
            assert seq[2, base] == 0.0,    f"x for player {i} in frame 2 should be 0"
            assert seq[2, base + 1] == 0.0, f"y for player {i} in frame 2 should be 0"


class TestSplitByMatch:
    def test_no_leakage(self):
        match_ids = [f"MATCH_{i:03d}" for i in range(20)]
        assignment = split_by_match(match_ids, seed=42)
        train = {m for m, s in assignment.items() if s == "train"}
        val   = {m for m, s in assignment.items() if s == "val"}
        test  = {m for m, s in assignment.items() if s == "test"}
        assert len(train & val) == 0,  "Match appears in both train and val"
        assert len(train & test) == 0, "Match appears in both train and test"
        assert len(val & test) == 0,   "Match appears in both val and test"

    def test_all_matches_assigned(self):
        match_ids = [f"MATCH_{i:03d}" for i in range(10)]
        assignment = split_by_match(match_ids, seed=0)
        assert len(assignment) == 10

    def test_min_one_per_split(self):
        match_ids = [f"M_{i}" for i in range(5)]
        assignment = split_by_match(match_ids, seed=42)
        splits = list(assignment.values())
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_deterministic(self):
        match_ids = [f"X_{i}" for i in range(15)]
        a1 = split_by_match(match_ids, seed=99)
        a2 = split_by_match(match_ids, seed=99)
        assert a1 == a2

    def test_different_seed_different_split(self):
        match_ids = [f"X_{i}" for i in range(15)]
        a1 = split_by_match(match_ids, seed=1)
        a2 = split_by_match(match_ids, seed=2)
        assert a1 != a2


class TestAugmentation:
    def test_flip_preserves_shape(self):
        from training.preprocessing.augmentation import flip_horizontal
        seq = np.random.rand(20, 44).astype(np.float32)
        flipped = flip_horizontal(seq)
        assert flipped.shape == seq.shape

    def test_flip_y_mirrored(self):
        from training.preprocessing.augmentation import flip_horizontal
        seq = np.zeros((5, 44), dtype=np.float32)
        seq[:, 1] = 0.3   # player 0 y = 0.3
        flipped = flip_horizontal(seq)
        assert abs(flipped[0, 1] - 0.7) < 1e-5  # 1 - 0.3 = 0.7

    def test_noise_within_bounds(self):
        from training.preprocessing.augmentation import add_gaussian_noise
        seq = np.random.rand(20, 44).astype(np.float32)
        noisy = add_gaussian_noise(seq, sigma=0.01)
        x_vals = noisy[:, 0::4]
        y_vals = noisy[:, 1::4]
        assert x_vals.min() >= 0.0 and x_vals.max() <= 1.0
        assert y_vals.min() >= 0.0 and y_vals.max() <= 1.0
