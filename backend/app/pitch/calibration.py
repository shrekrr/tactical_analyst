"""
Pitch calibration module.

Computes a homography matrix from image↔pitch point correspondences.

Standard pitch keypoints (metres, origin = top-left of pitch):
    Corner top-left      (0, 0)
    Corner top-right     (105, 0)
    Corner bottom-left   (0, 68)
    Corner bottom-right  (105, 68)
    Center spot          (52.5, 34)
    Penalty spot left    (11, 34)
    Penalty spot right   (94, 34)
"""
from __future__ import annotations

import json
from typing import List

import cv2
import numpy as np

from app.schemas import CalibrationPoint


# Standard pitch dimensions in metres
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Suggested calibration keypoints (shown in UI for user to click)
REFERENCE_KEYPOINTS = [
    {"name": "Top-left corner",        "pitch_x": 0.0,   "pitch_y": 0.0},
    {"name": "Top-right corner",       "pitch_x": 105.0, "pitch_y": 0.0},
    {"name": "Bottom-left corner",     "pitch_x": 0.0,   "pitch_y": 68.0},
    {"name": "Bottom-right corner",    "pitch_x": 105.0, "pitch_y": 68.0},
    {"name": "Centre spot",            "pitch_x": 52.5,  "pitch_y": 34.0},
    {"name": "Left penalty spot",      "pitch_x": 11.0,  "pitch_y": 34.0},
    {"name": "Right penalty spot",     "pitch_x": 94.0,  "pitch_y": 34.0},
    {"name": "Left penalty box TL",    "pitch_x": 0.0,   "pitch_y": 13.84},
    {"name": "Left penalty box TR",    "pitch_x": 16.5,  "pitch_y": 13.84},
    {"name": "Left penalty box BL",    "pitch_x": 0.0,   "pitch_y": 54.16},
    {"name": "Left penalty box BR",    "pitch_x": 16.5,  "pitch_y": 54.16},
]


def compute_homography_from_points(
    points: List[CalibrationPoint],
) -> List[List[float]]:
    """
    Compute homography matrix H such that:
        pitch_coord = H @ img_coord (homogeneous)

    Parameters
    ----------
    points : List[CalibrationPoint]
        At least 4 image↔pitch correspondences.

    Returns
    -------
    H : List[List[float]]
        3×3 homography matrix as nested list (JSON-serialisable).

    Raises
    ------
    ValueError
        If fewer than 4 points are provided or homography cannot be computed.
    """
    if len(points) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    img_pts = np.array([[p.img_x, p.img_y] for p in points], dtype=np.float32)
    pitch_pts = np.array([[p.pitch_x, p.pitch_y] for p in points], dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError(
            "Could not compute homography. Ensure the calibration points "
            "span the pitch area and are not collinear."
        )

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < 4:
        raise ValueError(
            f"Only {inliers} inlier points after RANSAC. "
            "Provide more accurate correspondences."
        )

    return H.tolist()


def load_homography(matrix_json: str | None) -> np.ndarray | None:
    """Deserialise a stored JSON homography matrix."""
    if not matrix_json:
        return None
    try:
        return np.array(json.loads(matrix_json), dtype=np.float64)
    except Exception:
        return None
