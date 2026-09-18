"""
Match data retrieval endpoints.
"""
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Match, Player, TacticalEvent, Team, TrackingPoint, BallTrackingPoint
from app.schemas import (
    FrameSnapshot,
    MatchAnalyticsResponse,
    MatchInfoResponse,
    PlayerSummary,
    TacticalEventOut,
    TeamMetrics,
)
from app.analytics.possession import build_possession_timeline
from app.analytics.team_metrics import compute_team_summary

router = APIRouter()


@router.get("/", response_model=List[MatchInfoResponse])
async def list_matches(db: AsyncSession = Depends(get_db)):
    """List all uploaded matches."""
    result = await db.execute(select(Match).order_by(Match.created_at.desc()))
    matches = result.scalars().all()
    return [MatchInfoResponse.model_validate(m) for m in matches]


@router.get("/{match_id}", response_model=MatchInfoResponse)
async def get_match(match_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    return MatchInfoResponse.model_validate(match)


@router.get("/{match_id}/analytics", response_model=MatchAnalyticsResponse)
async def get_analytics(match_id: str, db: AsyncSession = Depends(get_db)):
    """Return full analytics for a completed match."""
    result = await db.execute(
        select(Match)
        .options(
            selectinload(Match.teams),
            selectinload(Match.players).selectinload(Player.team),
            selectinload(Match.tactical_events),
        )
        .where(Match.id == match_id)
    )
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    if match.status != "completed":
        raise HTTPException(status_code=202, detail=f"Match is still {match.status}.")

    team_metrics = [
        await compute_team_summary(db, match, team) for team in match.teams
    ]
    players_out = [
        PlayerSummary.model_validate(p) for p in match.players
    ]
    events_out = [
        TacticalEventOut.model_validate(e) for e in match.tactical_events
    ]

    # Possession timeline
    poss_tl = await build_possession_timeline(db, match_id)

    # Formation timeline from events
    formation_tl = [
        {
            "timestamp_s": e.timestamp_s,
            "team_label": e.team_label,
            "formation": e.value,
        }
        for e in match.tactical_events
        if e.event_type == "formation_change"
    ]

    # Tactical summary
    summary = _build_summary(match, team_metrics)

    return MatchAnalyticsResponse(
        match_id=match_id,
        teams=team_metrics,
        players=players_out,
        tactical_events=events_out,
        possession_timeline=poss_tl,
        formation_timeline=formation_tl,
        summary=summary,
    )


@router.get("/{match_id}/frame/{frame_no}", response_model=FrameSnapshot)
async def get_frame_snapshot(
    match_id: str,
    frame_no: int,
    db: AsyncSession = Depends(get_db),
):
    """Return all tracked positions for a specific frame."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")

    # Load player positions for this frame
    tp_result = await db.execute(
        select(TrackingPoint, Player, Team)
        .join(Player, TrackingPoint.player_id == Player.id)
        .outerjoin(Team, Player.team_id == Team.id)
        .where(Player.match_id == match_id, TrackingPoint.frame == frame_no)
    )
    rows = tp_result.all()

    players_snap = []
    for tp, player, team in rows:
        players_snap.append(
            {
                "player_id": player.id,
                "tracking_id": player.tracking_id,
                "pitch_x": tp.pitch_x,
                "pitch_y": tp.pitch_y,
                "team_label": team.label if team else None,
                "team_color": team.color_hex if team else "#888888",
                "is_goalkeeper": player.is_goalkeeper,
                "is_referee": player.is_referee,
            }
        )

    # Ball position
    ball_result = await db.execute(
        select(BallTrackingPoint).where(
            BallTrackingPoint.match_id == match_id,
            BallTrackingPoint.frame == frame_no,
        )
    )
    ball_tp = ball_result.scalar_one_or_none()
    ball = (
        {"pitch_x": ball_tp.pitch_x, "pitch_y": ball_tp.pitch_y}
        if ball_tp
        else None
    )

    # Latest formation event at or before this frame
    events_result = await db.execute(
        select(TacticalEvent)
        .where(
            TacticalEvent.match_id == match_id,
            TacticalEvent.event_type == "formation_change",
            TacticalEvent.frame <= frame_no,
        )
        .order_by(TacticalEvent.frame.desc())
        .limit(2)
    )
    form_events = events_result.scalars().all()
    formation_a = next(
        (e.value for e in form_events if e.team_label == "team_a"), None
    )
    formation_b = next(
        (e.value for e in form_events if e.team_label == "team_b"), None
    )

    fps = match.fps or 25.0
    return FrameSnapshot(
        frame=frame_no,
        timestamp_s=frame_no / fps,
        players=players_snap,
        ball=ball,
        formation_a=formation_a,
        formation_b=formation_b,
        tactical_phase=None,
        possession=None,
    )


def _build_summary(match: Match, team_metrics: list[TeamMetrics]) -> dict:
    lines = []
    for tm in team_metrics:
        if tm.current_formation:
            lines.append(f"{tm.display_name or tm.team_label}: {tm.current_formation}")
    return {
        "text": "\n".join(lines) if lines else "Analysis in progress.",
        "teams": [tm.model_dump() for tm in team_metrics],
    }
