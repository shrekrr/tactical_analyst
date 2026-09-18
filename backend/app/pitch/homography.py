"""
Homography transformation utilities.

Converts image-space coordinates (pixels) to pitch-space coordinates (metres)
using the 3×3 homography matrix computed from calibration.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class HomographyTransformer:
    """
    Wraps a 3×3 perspective homography matrix for point transformation.

    Parameters
    ----------
    H : np.ndarray | None
        3×3 homography matrix (image → pitch).
        If None, transform() returns None for all points.
    pitch_length_m : float
        Length of the pitch in metres (for bounds clamping).
    pitch_width_m : float
        Width of the pitch in metres (for bounds clamping).
    """

    def __init__(
        self,
        H: Optional[np.ndarray],
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        self.H = H
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m

    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    def transform(
        self, img_x: float, img_y: float, clamp: bool = True
    ) -> Optional[Tuple[float, float]]:
        """
        Transform a single image-space point to pitch-space coordinates.

        Parameters
        ----------
        img_x, img_y : float
            Image coordinates (pixels).
        clamp : bool
            If True, clamp output to valid pitch bounds.

        Returns
        -------
        (pitch_x, pitch_y) in metres, or None if not calibrated.
        """
        if self.H is None:
            return None

        pt = np.array([[[img_x, img_y]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(pt, self.H)
        px, py = float(dst[0][0][0]), float(dst[0][0][1])

        if clamp:
            px = float(np.clip(px, 0.0, self.pitch_length_m))
            py = float(np.clip(py, 0.0, self.pitch_width_m))

        return px, py

    def transform_batch(
        self,
        points: list[Tuple[float, float]],
        clamp: bool = True,
    ) -> list[Optional[Tuple[float, float]]]:
        """Transform a list of (img_x, img_y) pairs."""
        if self.H is None:
            return [None] * len(points)

        if not points:
            return []

        pts = np.array([[[p[0], p[1]] for p in points]], dtype=np.float64)
        dst = cv2.perspectiveTransform(pts, self.H)[0]

        results = []
        for px, py in dst:
            px, py = float(px), float(py)
            if clamp:
                px = float(np.clip(px, 0.0, self.pitch_length_m))
                py = float(np.clip(py, 0.0, self.pitch_width_m))
            results.append((px, py))
        return results

    def update(self, H: np.ndarray) -> None:
        """Update the homography matrix (e.g., after recalibration)."""
        self.H = H
