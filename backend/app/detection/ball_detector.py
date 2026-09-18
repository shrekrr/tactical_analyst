"""
Ball-specialised detector.
Wraps YOLODetector and filters to ball detections only,
applying additional heuristics (size, aspect ratio) to reduce false positives.
"""
from __future__ import annotations

from typing import List

import numpy as np

from app.detection.base import Detection, ObjectDetector
from app.detection.yolo_detector import YOLODetector


class BallDetector(ObjectDetector):
    """
    Thin wrapper around YOLODetector that returns only ball detections,
    with additional filtering to suppress false positives.

    Parameters
    ----------
    base_detector : YOLODetector
        Shared detector instance (avoids double model loading).
    max_size_px : int
        Reject detections whose bounding box diagonal exceeds this value.
        Helps filter large circular sponsor logos.
    min_confidence : float
        Minimum confidence for ball detections (often higher than player
        threshold because the ball is small and frequently occluded).
    """

    def __init__(
        self,
        base_detector: YOLODetector,
        max_size_px: int = 80,
        min_confidence: float = 0.35,
    ) -> None:
        self._detector = base_detector
        self.max_size_px = max_size_px
        self.min_confidence = min_confidence

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        all_dets = self._detector.detect(frame, frame_idx)
        ball_dets = []
        for d in all_dets:
            if d.class_name != "ball":
                continue
            if d.confidence < self.min_confidence:
                continue
            diag = (d.width**2 + d.height**2) ** 0.5
            if diag > self.max_size_px:
                continue
            # Ball should be roughly circular
            aspect = d.width / max(d.height, 1)
            if not (0.5 <= aspect <= 2.0):
                continue
            ball_dets.append(d)

        # Return the highest-confidence ball detection (or empty list)
        if ball_dets:
            ball_dets.sort(key=lambda d: d.confidence, reverse=True)
            return ball_dets[:1]
        return []
