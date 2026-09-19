"""
Unit tests for pitch homography transform.

Tests coordinate transformations, boundary handling, and invalid matrix fallback.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.pitch.homography import HomographyTransformer
from app.pitch.calibration import compute_homography_from_points
from app.schemas import CalibrationPoint


def _make_calibration_points(img_pts, pitch_pts):
    return [
        CalibrationPoint(img_x=float(i[0]), img_y=float(i[1]),
                         pitch_x=float(p[0]), pitch_y=float(p[1]))
        for i, p in zip(img_pts, pitch_pts)
    ]


def compute_homography(img_pts, pitch_pts):
    """Test helper that wraps compute_homography_from_points."""
    points = _make_calibration_points(img_pts, pitch_pts)
    try:
        H_list = compute_homography_from_points(points)
        import numpy as np
        return np.array(H_list, dtype=np.float64)
    except ValueError:
        return None


class TestHomographyTransformer:
    def _make_identity_transformer(self) -> HomographyTransformer:
        """Create a transformer with known homography (identity-like mapping)."""
        # 4 known correspondences: image pixels → pitch metres
        # Simple 1:1 scaled mapping for testability
        img_points = np.array([
            [0.0, 0.0],
            [105.0, 0.0],
            [105.0, 68.0],
            [0.0, 68.0],
        ], dtype=np.float32)

        pitch_points = np.array([
            [0.0, 0.0],
            [105.0, 0.0],
            [105.0, 68.0],
            [0.0, 68.0],
        ], dtype=np.float32)

        H = compute_homography(img_points, pitch_points)
        return HomographyTransformer(H, pitch_length_m=105.0, pitch_width_m=68.0)

    def test_corner_top_left(self):
        t = self._make_identity_transformer()
        result = t.transform(0.0, 0.0)
        assert result is not None
        assert abs(result[0] - 0.0) < 0.5
        assert abs(result[1] - 0.0) < 0.5

    def test_corner_bottom_right(self):
        t = self._make_identity_transformer()
        result = t.transform(105.0, 68.0)
        assert result is not None
        assert abs(result[0] - 105.0) < 0.5
        assert abs(result[1] - 68.0) < 0.5

    def test_centre_circle(self):
        t = self._make_identity_transformer()
        result = t.transform(52.5, 34.0)
        assert result is not None
        assert abs(result[0] - 52.5) < 0.5
        assert abs(result[1] - 34.0) < 0.5

    def test_returns_none_without_calibration(self):
        t = HomographyTransformer(None, pitch_length_m=105.0, pitch_width_m=68.0)
        result = t.transform(50.0, 25.0)
        assert result is None

    def test_is_calibrated_flag(self):
        t_none = HomographyTransformer(None, pitch_length_m=105.0, pitch_width_m=68.0)
        assert not t_none.is_calibrated

        t_real = self._make_identity_transformer()
        assert t_real.is_calibrated


class TestComputeHomography:
    def test_requires_four_points(self):
        """homography requires >= 4 correspondences."""
        img = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 68.0], [0.0, 68.0]])
        pitch = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
        H = compute_homography(img, pitch)
        assert H is not None
        assert H.shape == (3, 3)

    def test_returns_none_for_fewer_than_four(self):
        img = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 68.0]])
        pitch = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0]])
        H = compute_homography(img, pitch)
        assert H is None
