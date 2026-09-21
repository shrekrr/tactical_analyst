"""
Possession estimation.

Basic approach:
    Ball position → nearest outfield player → that player's team → possession.

Temporal smoothing (rolling window) is applied to avoid rapid flipping
caused by detection noise.

This is an AI-estimated metric, not professional GPS tracking.
"""
from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BallTrackingPoint, Player, Team, TrackingPoint


def nearest_team(
    ball_x: float,
    ball_y: float,
    player_positions: List[Tuple[float, float, str]],  # (x, y, team_label)
) -> Optional[str]:
    """
    Return the team label of the player nearest to the ball.

    Parameters
    ----------
    ball_x, ball_y : float
        Ball pitch coordinates.
    player_positions : [(pitch_x, pitch_y, team_label), ...]
        All outfield player positions for this frame.

    Returns
    -------
    team_label : str | None
    """
    best_dist = float("inf")
    best_team: Optional[str] = None

    for px, py, team in player_positions:
        if team in (None, "unknown", "referee"):
            continue
        dist = ((ball_x - px) ** 2 + (ball_y - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_team = team

    # If ball is very far from any player, call it contested
    if best_dist > 10.0:  # 10 metres threshold
        return None
    return best_team


def smooth_possession(
    raw_sequence: List[Optional[str]],
    window: int = 10,
) -> List[Optional[str]]:
    """
    Apply a rolling majority-vote window to smooth possession labels.
    """
    smoothed = []
    q: deque = deque(maxlen=window)

    for label in raw_sequence:
        q.append(label)
        counts: dict = defaultdict(int)
        for v in q:
            if v is not None:
                counts[v] += 1
        if counts:
            smoothed.append(max(counts, key=counts.__getitem__))
        else:
            smoothed.append(None)

    return smoothed


async def build_possession_timeline(
    db: AsyncSession,
    match_id: str,
) -> List[Dict]:
    """
    Build the full possession timeline for a match from stored data.

    Returns a list of dicts: [{timestamp_s, frame, team_label, confidence}]
    """
    # Load ball positions
    ball_result = await db.execute(
        select(BallTrackingPoint)
        .where(BallTrackingPoint.match_id == match_id)
        .order_by(BallTrackingPoint.frame)
    )
    ball_points = ball_result.scalars().all()

    # Load all outfield player positions
    tp_result = await db.execute(
        select(TrackingPoint, Player, Team)
        .join(Player, TrackingPoint.player_id == Player.id)
        .outerjoin(Team, Player.team_id == Team.id)
        .where(
            Player.match_id == match_id,
            Player.is_referee == False,
            TrackingPoint.pitch_x != None,
        )
        .order_by(TrackingPoint.frame)
    )
    rows = tp_result.all()

    if not rows:
        return []

    player_by_frame: dict = defaultdict(list)
    frame_ts: dict = {}
    for tp, player, team in rows:
        if tp.pitch_x is not None and tp.pitch_y is not None:
            player_by_frame[tp.frame].append(
                (tp.pitch_x, tp.pitch_y, team.label if team else "unknown")
            )
            frame_ts[tp.frame] = tp.timestamp_s

    all_frames = sorted(player_by_frame.keys())
    if not all_frames:
        return []

    raw: List[Optional[str]] = []
    timeline_frames = []

    if ball_points:
        ball_by_frame = {bp.frame: bp for bp in ball_points}
        frames = sorted({bp.frame for bp in ball_points})
        for frame in frames:
            bp = ball_by_frame.get(frame)
            if bp is None or bp.pitch_x is None or bp.pitch_y is None:
                raw.append(None)
            else:
                raw.append(nearest_team(bp.pitch_x, bp.pitch_y, player_by_frame.get(frame, [])))
            timeline_frames.append(frame)
    else:
        # Fallback when ball is not detected: territorial field tilt / player centroid
        for frame in all_frames:
            pts = player_by_frame[frame]
            pts_a = [p[0] for p in pts if p[2] == "team_a"]
            pts_b = [p[0] for p in pts if p[2] == "team_b"]
            if pts_a and pts_b:
                raw.append("team_a" if np.mean(pts_a) >= 52.5 else "team_b")
            elif pts_a:
                raw.append("team_a")
            elif pts_b:
                raw.append("team_b")
            else:
                raw.append(None)
            timeline_frames.append(frame)

    smoothed = smooth_possession(raw, window=10)

    timeline = []
    for frame, label in zip(timeline_frames, smoothed):
        ts = frame_ts.get(frame, float(frame) / 25.0)
        timeline.append(
            {
                "frame": frame,
                "timestamp_s": round(ts, 2),
                "team_label": label,
            }
        )

    return timeline



def aggregate_possession(
    timeline: List[Dict],
) -> Dict[str, float]:
    """
    Summarise possession percentages from a possession timeline.

    Returns
    -------
    dict: {team_a: 57.3, team_b: 38.1, unknown: 4.6}
    """
    counts: dict = defaultdict(int)
    total = 0
    for entry in timeline:
        lbl = entry.get("team_label") or "unknown"
        counts[lbl] += 1
        total += 1

    if total == 0:
        return {"team_a": 0.0, "team_b": 0.0, "unknown": 0.0}

    return {k: round(v / total * 100, 1) for k, v in counts.items()}
