"""
Player-specific analytics endpoints.
"""
import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Player, TrackingPoint, Team
from app.schemas import PlayerDetailResponse, PlayerSummary, TrackingPointOut
from app.config import settings

router = APIRouter()


@router.get("/{match_id}/players", response_model=list[PlayerSummary])
async def list_players(match_id: str, db: AsyncSession = Depends(get_db)):
    """List all tracked players for a match."""
    result = await db.execute(
        select(Player)
        .options(selectinload(Player.team))
        .where(Player.match_id == match_id)
        .order_by(Player.tracking_id)
    )
    players = result.scalars().all()
    return [
        PlayerSummary(
            id=p.id,
            tracking_id=p.tracking_id,
            team_label=p.team.label if p.team else None,
            team_color=p.team.color_hex if p.team else "#888888",
            total_distance_m=p.total_distance_m,
            avg_speed_kmh=p.avg_speed_kmh,
            max_speed_kmh=p.max_speed_kmh,
            avg_x=p.avg_x,
            avg_y=p.avg_y,
            pct_defensive_third=p.pct_defensive_third,
            pct_middle_third=p.pct_middle_third,
            pct_attacking_third=p.pct_attacking_third,
            is_goalkeeper=p.is_goalkeeper,
            is_referee=p.is_referee,
        )
        for p in players
    ]


@router.get("/{match_id}/players/{player_id}", response_model=PlayerDetailResponse)
async def get_player_detail(
    match_id: str,
    player_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Full player analytics including trajectory and heatmap."""
    result = await db.execute(
        select(Player)
        .options(selectinload(Player.team), selectinload(Player.tracking_points))
        .where(Player.id == player_id, Player.match_id == match_id)
    )
    player = result.scalar_one_or_none()
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found.")

    trajectory = [
        TrackingPointOut(
            frame=tp.frame,
            timestamp_s=tp.timestamp_s,
            pitch_x=tp.pitch_x,
            pitch_y=tp.pitch_y,
            speed_kmh=tp.speed_kmh,
        )
        for tp in sorted(player.tracking_points, key=lambda t: t.frame)
    ]

    heatmap = _build_heatmap(trajectory)
    speed_series = [
        {"timestamp_s": tp.timestamp_s, "speed_kmh": tp.speed_kmh or 0.0}
        for tp in trajectory
        if tp.speed_kmh is not None
    ]

    summary = PlayerSummary(
        id=player.id,
        tracking_id=player.tracking_id,
        team_label=player.team.label if player.team else None,
        team_color=player.team.color_hex if player.team else "#888888",
        total_distance_m=player.total_distance_m,
        avg_speed_kmh=player.avg_speed_kmh,
        max_speed_kmh=player.max_speed_kmh,
        avg_x=player.avg_x,
        avg_y=player.avg_y,
        pct_defensive_third=player.pct_defensive_third,
        pct_middle_third=player.pct_middle_third,
        pct_attacking_third=player.pct_attacking_third,
        is_goalkeeper=player.is_goalkeeper,
        is_referee=player.is_referee,
    )

    return PlayerDetailResponse(
        player=summary,
        trajectory=trajectory,
        heatmap_data=heatmap,
        speed_series=speed_series,
    )


def _build_heatmap(
    trajectory: list[TrackingPointOut],
    grid_x: int = 26,
    grid_y: int = 17,
) -> list[list[float]]:
    """Generate a 2D normalized heatmap grid from trajectory points."""
    grid = np.zeros((grid_y, grid_x), dtype=np.float32)
    pitch_l = settings.pitch_length_m
    pitch_w = settings.pitch_width_m

    for tp in trajectory:
        if tp.pitch_x is None or tp.pitch_y is None:
            continue
        xi = int(np.clip(tp.pitch_x / pitch_l * grid_x, 0, grid_x - 1))
        yi = int(np.clip(tp.pitch_y / pitch_w * grid_y, 0, grid_y - 1))
        grid[yi, xi] += 1.0

    if grid.max() > 0:
        grid /= grid.max()

    return grid.tolist()
