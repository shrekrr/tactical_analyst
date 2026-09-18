"""
Video upload endpoint.
"""
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Match
from app.schemas import MatchUploadResponse
from app.utils.video import probe_video

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/upload", response_model=MatchUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a football match video for analysis."""
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    match_id = str(uuid.uuid4())
    dest_path = settings.videos_dir / f"{match_id}{ext}"

    # Stream file to disk
    total_bytes = 0
    try:
        async with aiofiles.open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_BYTES:
                    await f.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
                    )
                await f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail="Failed to save video file.") from exc

    # Probe video metadata
    meta = await probe_video(dest_path)

    # Persist to DB
    match = Match(
        id=match_id,
        filename=file.filename or dest_path.name,
        video_path=str(dest_path),
        file_size_bytes=total_bytes,
        duration_s=meta.get("duration_s"),
        fps=meta.get("fps"),
        total_frames=meta.get("total_frames"),
        width=meta.get("width"),
        height=meta.get("height"),
        status="uploaded",
    )
    db.add(match)
    await db.flush()

    logger.info("Match {} uploaded: {} ({:.1f} MB)", match_id, file.filename, total_bytes / 1e6)

    return MatchUploadResponse(
        match_id=match_id,
        status="uploaded",
        filename=file.filename or "",
        file_size_bytes=total_bytes,
    )
