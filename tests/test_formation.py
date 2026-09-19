"""
Unit tests for the rule-based formation detector.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.analytics.formation import (
    KNOWN_FORMATIONS,
    detect_formation_rule_based,
    _formation_distance,
    _infer_formation_from_lines,
)


class TestFormationDetectorRuleBased:
    def _make_433_positions(self):
        """11 players in a clear 4-3-3 layout."""
        # 4 defenders, 3 midfielders, 3 forwards + 1 GK (deepest)
        positions = []
        # GK
        positions.append((5.0, 34.0))
        # 4 defenders around x=25
        for y in [14, 26, 42, 54]:
            positions.append((25.0, float(y)))
        # 3 midfielders around x=50
        for y in [20, 34, 48]:
            positions.append((50.0, float(y)))
        # 3 forwards around x=75
        for y in [20, 34, 48]:
            positions.append((75.0, float(y)))
        return positions

    def test_detects_433(self):
        positions = self._make_433_positions()
        # Pass outfield players only (exclude GK at deepest x)
        formation, conf, explanation = detect_formation_rule_based(positions)
        assert formation in KNOWN_FORMATIONS
        assert 0.0 <= conf <= 1.0

    def test_returns_unknown_too_few_players(self):
        formation, conf, _ = detect_formation_rule_based([(10.0, 20.0), (20.0, 30.0)])
        assert formation == "unknown"
        assert conf == 0.0

    def test_returns_known_formation(self):
        positions = self._make_433_positions()
        formation, _, _ = detect_formation_rule_based(positions)
        assert formation in KNOWN_FORMATIONS

    def test_empty_positions(self):
        formation, conf, _ = detect_formation_rule_based([])
        assert formation == "unknown"

    def test_confidence_range(self):
        positions = self._make_433_positions()
        _, conf, _ = detect_formation_rule_based(positions)
        assert 0.0 <= conf <= 1.0

    def test_explanation_contains_method(self):
        positions = self._make_433_positions()
        _, _, explanation = detect_formation_rule_based(positions)
        assert "method" in explanation


class TestFormationDistance:
    def test_exact_match(self):
        assert _formation_distance([4, 3, 3], "4-3-3") == 0

    def test_one_off(self):
        assert _formation_distance([4, 4, 2], "4-3-3") == 2  # |4-3|+|2-3|

    def test_different_lengths(self):
        # 4-2-3-1 vs [4, 3, 3]
        d = _formation_distance([4, 3, 3], "4-2-3-1")
        assert d >= 0


class TestInferFormationFromLines:
    def test_direct_match_433(self):
        assert _infer_formation_from_lines([4, 3, 3]) == "4-3-3"

    def test_direct_match_442(self):
        assert _infer_formation_from_lines([4, 4, 2]) == "4-4-2"

    def test_falls_back_to_closest(self):
        # [4, 3, 4] doesn't exist → should return closest known
        result = _infer_formation_from_lines([4, 3, 4])
        assert result in KNOWN_FORMATIONS
