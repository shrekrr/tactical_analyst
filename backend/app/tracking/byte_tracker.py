"""
ByteTrack / BoT-SORT wrapper using ultralytics built-in tracking.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO

from app.detection.base import Detection
from app.tracking.base import Track, Tracker


class ByteTrackWrapper(Tracker):
    """
    Uses the ByteTrack algorithm built into the ultralytics YOLO tracker.

    Rather than maintaining separate track state, this implementation
    re-routes raw detections through the model's tracking API which
    handles the Kalman filter and IoU matching internally.

    Parameters
    ----------
    model : YOLO
        The YOLO model instance (shared with detector to avoid double load).
    tracker_type : str
        ``"bytetrack"`` or ``"botsort"``.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        model: YOLO,
        tracker_type: str = "bytetrack",
        device: str = "cpu",
        persistence: int = 30,
    ) -> None:
        self.model = model
        self.tracker_type = tracker_type
        self.device = device
        self.persistence = persistence  # frames to keep lost tracks
        self._active_tracks: dict[int, Track] = {}

    def update(self, detections: List[Detection], frame: int) -> List[Track]:
        """
        Convert Detection objects into Track objects.

        In this lightweight implementation we run the YOLO model's built-in
        tracker externally (from the pipeline) and this update method maps
        the results.  We keep the Tracker ABC contract so the system can swap
        in a standalone ByteTrack if needed.
        """
        tracks = []
        for det in detections:
            if det.class_name not in ("player", "referee", "goalkeeper"):
                continue
            track = Track(
                track_id=getattr(det, "track_id", det.frame),
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                frame=frame,
            )
            tracks.append(track)
        return tracks

    def reset(self) -> None:
        self._active_tracks.clear()


def run_tracking_on_frame(
    model: YOLO,
    frame: np.ndarray,
    frame_idx: int,
    tracker_cfg: str = "bytetrack.yaml",
    confidence: float = 0.25,
    device: str = "cpu",
) -> tuple[List[Track], Optional[dict]]:
    """
    Run YOLO tracking on a single frame.

    Returns
    -------
    tracks : List[Track]
        Tracked player/referee objects with persistent IDs.
    ball : dict | None
        Ball detection dict with 'bbox' and 'confidence', or None.
    """
    results = model.track(
        frame,
        conf=confidence,
        persist=True,
        tracker=tracker_cfg,
        verbose=False,
        device=device,
    )

    tracks: List[Track] = []
    ball: Optional[dict] = None

    for result in results:
        if result.boxes is None:
            continue

        names = model.names
        has_ids = result.boxes.id is not None

        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            label = names.get(cls_id, "").lower()

            # Remap COCO "person" → "player"
            if label == "person":
                label = "player"

            if label in ("player", "referee", "goalkeeper"):
                tid = int(box.id[0].item()) if has_ids and box.id is not None else -(i + 1)
                tracks.append(
                    Track(
                        track_id=tid,
                        class_name=label,
                        confidence=conf,
                        bbox=xyxy,
                        frame=frame_idx,
                    )
                )
            elif label in ("sports ball", "ball"):
                # Keep highest confidence ball
                if ball is None or conf > ball["confidence"]:
                    ball = {"bbox": xyxy, "confidence": conf, "frame": frame_idx}

    return tracks, ball
