"""
Tactical phase classification using positional heuristics.

Phases:
    attacking          — team centroid in opponent half
    defending          — team centroid in own half (deep)
    transition_attack  — rapid forward movement after ball recovery
    transition_defense — rapid backward movement after ball loss

Initial implementation is purely heuristic.
The architecture allows replacement with a Temporal Transformer classifier.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


PITCH_LENGTH_M = 105.0


def classify_phase(
    team_centroid_x: float,
    prev_centroid_x: Optional[float],
    ball_x: Optional[float],
    possession_team: Optional[str],
    own_half_max_x: float = 52.5,
    transition_threshold_m: float = 8.0,
) -> str:
    """
    Classify the current tactical phase for a team.

    Parameters
    ----------
    team_centroid_x : float
        Current centroid X (metres).
    prev_centroid_x : float | None
        Centroid X from previous window (for transition detection).
    ball_x : float | None
        Current ball X position.
    possession_team : str | None
        Label of team in possession (or None).
    own_half_max_x : float
        X boundary of the team's own half (default 52.5 = centre).
    transition_threshold_m : float
        Minimum centroid movement to classify as a transition phase.

    Returns
    -------
    phase : str
    """
    # Determine movement direction
    delta_x = 0.0
    if prev_centroid_x is not None:
        delta_x = team_centroid_x - prev_centroid_x

    in_own_half = team_centroid_x < own_half_max_x

    # Transition detection (high centroid movement between windows)
    if abs(delta_x) > transition_threshold_m:
        if delta_x > 0:
            return "transition_attack"
        else:
            return "transition_defense"

    if in_own_half:
        return "defending"
    else:
        return "attacking"


def classify_phases_sequence(
    centroid_xs: List[float],
    ball_xs: List[Optional[float]],
    possession_labels: List[Optional[str]],
    window: int = 5,
) -> List[str]:
    """
    Classify tactical phases for a sequence of frames.

    Parameters
    ----------
    centroid_xs : List[float]
        Team centroid X per frame.
    ball_xs : List[float | None]
        Ball X per frame.
    possession_labels : List[str | None]
        Possession team label per frame.
    window : int
        Smoothing window for centroid comparison.

    Returns
    -------
    phases : List[str]
        One phase label per frame.
    """
    phases = []
    n = len(centroid_xs)

    for i in range(n):
        prev_x = centroid_xs[max(0, i - window)]
        phase = classify_phase(
            team_centroid_x=centroid_xs[i],
            prev_centroid_x=prev_x if i > 0 else None,
            ball_x=ball_xs[i] if i < len(ball_xs) else None,
            possession_team=possession_labels[i] if i < len(possession_labels) else None,
        )
        phases.append(phase)

    return phases
