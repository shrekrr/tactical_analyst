"""
Player-level metrics computation.

All metrics are derived from stored TrackingPoint data —
nothing is hardcoded or fabricated.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from loguru import logger

from app.config import settings


def compute_distance_covered(
    positions: List[Tuple[float, float]],  # [(pitch_x, pitch_y), ...]
) -> float:
    """
    Total Euclidean distance covered along the trajectory.

    Returns metres.
    """
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def compute_speed_series(
    positions: List[Tuple[float, float]],
    timestamps: List[float],  # seconds
    smooth: bool = True,
) -> List[float]:
    """
    Compute instantaneous speed (km/h) between consecutive positions.

    Applies Savitzky–Golay smoothing to reduce frame-to-frame noise.
    Returns a list of the same length as positions (first entry is 0).
    """
    if len(positions) < 2:
        return [0.0] * len(positions)

    speeds = [0.0]
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            speeds.append(0.0)
            continue
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist_m = math.sqrt(dx * dx + dy * dy)
        # Convert m/s → km/h
        speed_kmh = (dist_m / dt) * 3.6
        speeds.append(speed_kmh)

    if smooth and len(speeds) >= 7:
        window = min(11, len(speeds) if len(speeds) % 2 == 1 else len(speeds) - 1)
        window = window if window % 2 == 1 else window - 1
        window = max(window, 5)
        try:
            speeds = savgol_filter(speeds, window_length=window, polyorder=2).tolist()
        except Exception:
            pass  # fallback to unsmoothed

    # Clamp to reasonable football range
    speeds = [max(0.0, min(s, 45.0)) for s in speeds]
    return speeds


def compute_zone_distribution(
    positions: List[Tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Dict[str, float]:
    """
    Compute what fraction of time a player spent in each pitch zone.

    Longitudinal zones: defensive_third | middle_third | attacking_third
    Lateral zones: left | center | right

    Returns percentages (0–100).
    """
    if not positions:
        return {
            "defensive_third": 0.0,
            "middle_third": 0.0,
            "attacking_third": 0.0,
            "left": 0.0,
            "center": 0.0,
            "right": 0.0,
        }

    counts = {
        "defensive_third": 0,
        "middle_third": 0,
        "attacking_third": 0,
        "left": 0,
        "center": 0,
        "right": 0,
    }

    third = pitch_length / 3.0
    for px, py in positions:
        # Longitudinal
        if px < third:
            counts["defensive_third"] += 1
        elif px < 2 * third:
            counts["middle_third"] += 1
        else:
            counts["attacking_third"] += 1

        # Lateral
        if py < pitch_width / 3.0:
            counts["left"] += 1
        elif py < 2 * pitch_width / 3.0:
            counts["center"] += 1
        else:
            counts["right"] += 1

    n = len(positions)
    return {k: round(v / n * 100, 1) for k, v in counts.items()}


def compute_average_position(
    positions: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Return the mean (x, y) position."""
    if not positions:
        return (0.0, 0.0)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return (float(np.mean(xs)), float(np.mean(ys)))


def build_heatmap_grid(
    positions: List[Tuple[float, float]],
    grid_x: int = 26,
    grid_y: int = 17,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> np.ndarray:
    """
    Return a 2D numpy array of normalised visit counts.
    Shape: (grid_y, grid_x).
    """
    grid = np.zeros((grid_y, grid_x), dtype=np.float32)
    for px, py in positions:
        xi = int(np.clip(px / pitch_length * grid_x, 0, grid_x - 1))
        yi = int(np.clip(py / pitch_width * grid_y, 0, grid_y - 1))
        grid[yi, xi] += 1.0
    if grid.max() > 0:
        grid /= grid.max()
    return grid
