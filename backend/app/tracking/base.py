"""
Abstract tracker interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from app.detection.base import Detection


@dataclass
class Track:
    """A tracked object with a persistent ID."""
    track_id: int
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2]
    frame: int

    @property
    def bottom_center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def bottom_center_y(self) -> float:
        return self.bbox[3]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


class Tracker(ABC):
    """
    Abstract interface for multi-object trackers.
    Concrete implementations must implement :py:meth:`update`.
    """

    @abstractmethod
    def update(
        self, detections: List[Detection], frame: int
    ) -> List[Track]:
        """
        Update the tracker with new detections and return active tracks.

        Parameters
        ----------
        detections : List[Detection]
            Raw detections from the detector for the current frame.
        frame : int
            Current frame number.

        Returns
        -------
        List[Track]
            All currently active tracks with persistent IDs.
        """
        ...

    def reset(self) -> None:
        """Reset tracker state between videos."""
        pass
