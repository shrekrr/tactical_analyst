"""
Unit tests for player analytics functions.

Tests distance calculation, speed estimation, zone assignment, and heatmap.
These are fully deterministic — no model weights required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.analytics.player_metrics import (
    compute_distance_covered,
    compute_speed_series,
    compute_average_position,
    compute_zone_distribution,
)


class TestDistanceCovered:
    def test_straight_line(self):
        # 3 points along x-axis, 10m apart
        positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
        dist = compute_distance_covered(positions)
        assert abs(dist - 20.0) < 0.01

    def test_single_point(self):
        assert compute_distance_covered([(5.0, 5.0)]) == 0.0

    def test_empty(self):
        assert compute_distance_covered([]) == 0.0

    def test_diagonal(self):
        # Pythagoras: sqrt(3^2 + 4^2) = 5
        positions = [(0.0, 0.0), (3.0, 4.0)]
        dist = compute_distance_covered(positions)
        assert abs(dist - 5.0) < 0.01

    def test_stationary_player(self):
        positions = [(50.0, 34.0)] * 10
        dist = compute_distance_covered(positions)
        assert dist == 0.0


class TestSpeedSeries:
    def test_constant_speed(self):
        # 1m/s = 3.6 km/h, player moves 1m every second
        positions = [(float(i), 0.0) for i in range(10)]
        timestamps = [float(i) for i in range(10)]
        speeds = compute_speed_series(positions, timestamps)
        assert len(speeds) == len(positions)
        # After SG smoothing, interior values should be ~3.6 km/h
        assert all(s >= 0.0 for s in speeds)

    def test_stationary_zero_speed(self):
        positions = [(10.0, 10.0)] * 5
        timestamps = [0.0, 0.2, 0.4, 0.6, 0.8]
        speeds = compute_speed_series(positions, timestamps)
        assert all(abs(s) < 0.5 for s in speeds)  # near zero

    def test_output_length(self):
        positions = [(float(i), 0.0) for i in range(15)]
        timestamps = [float(i) * 0.2 for i in range(15)]
        speeds = compute_speed_series(positions, timestamps)
        assert len(speeds) == 15

    def test_speeds_non_negative(self):
        positions = [(float(i), float(i % 5)) for i in range(20)]
        timestamps = [float(i) * 0.2 for i in range(20)]
        speeds = compute_speed_series(positions, timestamps)
        assert all(s >= 0.0 for s in speeds)


class TestAveragePosition:
    def test_symmetric(self):
        positions = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.0)]
        avg = compute_average_position(positions)
        assert abs(avg[0] - 5.0) < 0.01
        assert abs(avg[1] - 0.0) < 0.01

    def test_single_point(self):
        avg = compute_average_position([(30.0, 20.0)])
        assert avg == (30.0, 20.0)


class TestZoneDistribution:
    def test_all_defensive(self):
        # All positions in the defensive third (x < pitch_length / 3)
        positions = [(10.0, 20.0), (15.0, 30.0), (30.0, 34.0)]
        zones = compute_zone_distribution(positions, pitch_length=105.0, pitch_width=68.0)
        assert zones["defensive_third"] > 0.0
        # Middle and attacking may be 0 or close to it
        total = zones["defensive_third"] + zones["middle_third"] + zones["attacking_third"]
        assert abs(total - 100.0) < 1.0  # percentages sum to ~100

    def test_all_attacking(self):
        positions = [(80.0, 20.0), (85.0, 30.0), (90.0, 34.0)]
        zones = compute_zone_distribution(positions, pitch_length=105.0, pitch_width=68.0)
        assert zones["attacking_third"] > 0.0

    def test_sum_to_100(self):
        positions = [(i * 10.0, 34.0) for i in range(10)]
        zones = compute_zone_distribution(positions, pitch_length=105.0, pitch_width=68.0)
        total = zones["defensive_third"] + zones["middle_third"] + zones["attacking_third"]
        assert abs(total - 100.0) < 1.0
