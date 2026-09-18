"""
Video utility helpers.
"""
import asyncio
from pathlib import Path
from typing import Any, Dict

import cv2
from loguru import logger


async def probe_video(path: Path) -> Dict[str, Any]:
    """Extract basic metadata from a video file using OpenCV (async wrapper)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _probe_sync, path)


def _probe_sync(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("Could not open video for probing: {}", path)
        return {}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else None
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_s": duration_s,
        }
    finally:
        cap.release()
