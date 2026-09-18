"""
Abstract base class for object detectors.
Concrete implementations can be swapped without changing the pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Detection:
    """A single detection result from one frame."""
    frame: int
    class_name: str        # "player" | "ball" | "referee" | "goalkeeper"
    confidence: float
    bbox: List[float]      # [x1, y1, x2, y2] in image pixels
    class_id: int = 0

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def bottom_center_x(self) -> float:
        """Approximate ground contact point — center of bottom edge."""
        return self.center_x

    @property
    def bottom_center_y(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def crop(self, frame_img: np.ndarray) -> np.ndarray:
        """Return the cropped image region for this detection."""
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        h, w = frame_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame_img[y1:y2, x1:x2]


class ObjectDetector(ABC):
    """
    Abstract interface for object detectors.
    All concrete detectors must implement :py:meth:`detect`.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.
        frame_idx : int
            Frame number (used for logging / storing results).

        Returns
        -------
        List[Detection]
            All detections above the confidence threshold.
        """
        ...

    def detect_batch(
        self, frames: List[np.ndarray], start_idx: int = 0
    ) -> List[List[Detection]]:
        """
        Default batched detection — calls :py:meth:`detect` sequentially.
        Subclasses may override for true GPU batch inference.
        """
        return [
            self.detect(frame, start_idx + i) for i, frame in enumerate(frames)
        ]
