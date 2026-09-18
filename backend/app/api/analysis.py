"""
Analysis trigger + status endpoints.
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Match
from app.schemas import (
    AnalysisRequest,
    CalibrationRequest,
    CalibrationResponse,
    MatchStatusResponse,
)
from app.pitch.calibration import compute_homography_from_points

router = APIRouter()

# Global executor for CPU-bound processing
_executor = ProcessPoolExecutor(max_workers=1)


async def _run_pipeline(match_id: str, sample_fps: int, yolo_model: str | None) -> None:
    """Run the CV pipeline in a separate process."""
    from app.processing.pipeline import run_pipeline

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            _executor,
            run_pipeline,
            match_id,
            sample_fps,
            yolo_model or settings.yolo_model,
        )
    except Exception as exc:
        logger.exception("Pipeline failed for match {}", match_id)


@router.post("/{match_id}/analyze", response_model=MatchStatusResponse)
async def start_analysis(
    match_id: str,
    body: AnalysisRequest = AnalysisRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Start the analysis pipeline for an uploaded match."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()

    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    if match.status in ("processing", "completed"):
        raise HTTPException(
            status_code=409,
            detail=f"Match is already {match.status}.",
        )

    match.status = "queued"
    match.progress = 0
    match.sample_fps = body.sample_fps
    await db.flush()

    background_tasks.add_task(
        _run_pipeline, match_id, body.sample_fps, body.yolo_model
    )

    logger.info("Queued analysis for match {} @ {}fps", match_id, body.sample_fps)
    return MatchStatusResponse(match_id=match_id, status="queued", progress=0)


@router.get("/{match_id}/status", response_model=MatchStatusResponse)
async def get_status(match_id: str, db: AsyncSession = Depends(get_db)):
    """Poll processing status."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    return MatchStatusResponse(
        match_id=match_id,
        status=match.status,
        progress=match.progress,
        error_message=match.error_message,
    )


@router.post("/{match_id}/calibrate", response_model=CalibrationResponse)
async def submit_calibration(
    match_id: str,
    body: CalibrationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit pitch calibration keypoint correspondences."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    try:
        h_matrix = compute_homography_from_points(body.points)
    except Exception as exc:
        logger.warning("Calibration failed for match {}: {}", match_id, exc)
        return CalibrationResponse(success=False, message=str(exc))

    import json

    match.homography_matrix = json.dumps(h_matrix)
    match.calibration_points = json.dumps(
        [p.model_dump() for p in body.points]
    )
    await db.flush()

    return CalibrationResponse(
        success=True,
        message="Calibration saved.",
        homography_matrix=h_matrix,
    )
