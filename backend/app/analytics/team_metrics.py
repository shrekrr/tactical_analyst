"""
Team-level metrics computation.

Derived entirely from player tracking points — no hardcoded values.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Match, Player, Team, TrackingPoint
from app.schemas import TeamMetrics


def compute_team_width(positions: List[Tuple[float, float]]) -> float:
    """Lateral spread: max(y) - min(y) in metres."""
    if len(positions) < 2:
        return 0.0
    ys = [p[1] for p in positions]
    return float(max(ys) - min(ys))


def compute_team_depth(positions: List[Tuple[float, float]]) -> float:
    """Longitudinal spread: max(x) - min(x) in metres."""
    if len(positions) < 2:
        return 0.0
    xs = [p[0] for p in positions]
    return float(max(xs) - min(xs))


def compute_team_centroid(
    positions: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Mean position of all outfield players."""
    if not positions:
        return (0.0, 0.0)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return (float(np.mean(xs)), float(np.mean(ys)))


def compute_compactness(positions: List[Tuple[float, float]]) -> float:
    """
    Mean distance from each player to the team centroid (metres).
    Lower = more compact.
    """
    if len(positions) < 2:
        return 0.0
    cx, cy = compute_team_centroid(positions)
    dists = [
        float(np.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2))
        for p in positions
    ]
    return float(np.mean(dists))


def compute_defensive_line(
    positions: List[Tuple[float, float]],
    n_defenders: int = 4,
    attacking_direction: str = "right",
) -> Dict[str, float]:
    """
    Estimate defensive line height from the deepest outfield players.

    ``attacking_direction = "right"`` means higher X = closer to goal;
    defensive players have lower X values.

    Returns
    -------
    dict with keys: mean, min, max (all in metres along X axis)
    """
    if not positions:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    xs = sorted([p[0] for p in positions])
    deepest = xs[:n_defenders]  # lowest X = deepest defenders

    return {
        "mean": float(np.mean(deepest)),
        "min": float(np.min(deepest)),
        "max": float(np.max(deepest)),
    }


async def compute_team_summary(
    db: AsyncSession,
    match: Match,
    team: Team,
) -> TeamMetrics:
    """
    Load all tracking points for a team and compute aggregate metrics.
    """
    # Fetch all players in this team
    player_result = await db.execute(
        select(Player).where(
            Player.match_id == match.id,
            Player.team_id == team.id,
            Player.is_referee == False,
        )
    )
    players = player_result.scalars().all()

    if not players:
        return TeamMetrics(
            team_label=team.label,
            display_name=team.display_name,
            color_hex=team.color_hex,
            avg_width_m=None,
            avg_depth_m=None,
            avg_defensive_line_m=None,
            avg_compactness_m=None,
            total_distance_m=None,
            estimated_possession_pct=None,
            current_formation=None,
        )

    # Aggregate player distances
    total_dist = sum(
        p.total_distance_m for p in players if p.total_distance_m is not None
    )

    # Per-frame aggregate positions (sample up to 500 frames for speed)
    player_ids = [p.id for p in players]
    tp_result = await db.execute(
        select(TrackingPoint)
        .where(
            TrackingPoint.player_id.in_(player_ids),
            TrackingPoint.pitch_x != None,
        )
        .order_by(TrackingPoint.frame)
    )
    tps = tp_result.scalars().all()

    if not tps:
        return TeamMetrics(
            team_label=team.label,
            display_name=team.display_name,
            color_hex=team.color_hex,
            avg_width_m=None,
            avg_depth_m=None,
            avg_defensive_line_m=None,
            avg_compactness_m=None,
            total_distance_m=total_dist,
            estimated_possession_pct=None,
            current_formation=None,
        )

    # Group by frame
    from collections import defaultdict

    frame_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for tp in tps:
        frame_positions[tp.frame].append((tp.pitch_x, tp.pitch_y))

    widths, depths, compactnesses, def_lines = [], [], [], []
    for frame, positions in frame_positions.items():
        if len(positions) < 3:
            continue
        widths.append(compute_team_width(positions))
        depths.append(compute_team_depth(positions))
        compactnesses.append(compute_compactness(positions))
        def_lines.append(compute_defensive_line(positions)["mean"])

    return TeamMetrics(
        team_label=team.label,
        display_name=team.display_name or team.label,
        color_hex=team.color_hex,
        avg_width_m=float(np.mean(widths)) if widths else None,
        avg_depth_m=float(np.mean(depths)) if depths else None,
        avg_defensive_line_m=float(np.mean(def_lines)) if def_lines else None,
        avg_compactness_m=float(np.mean(compactnesses)) if compactnesses else None,
        total_distance_m=total_dist if total_dist else None,
        estimated_possession_pct=None,  # filled by possession module
        current_formation=None,          # filled by formation module
    )
