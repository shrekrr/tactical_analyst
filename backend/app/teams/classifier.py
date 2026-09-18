"""
Team classifier using K-Means clustering on HSV color histograms.

Algorithm:
1. For each tracked player, crop the jersey area (upper ~40% of bbox).
2. Convert to HSV and compute a colour histogram.
3. Collect histograms across multiple frames.
4. Run K-Means(k=2) to separate two teams.
5. Assign a persistent team label (team_a / team_b) based on cluster membership.
6. Derive a representative hex color for each team.

Does NOT hardcode any EPL team colors.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from loguru import logger


def _jersey_crop(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """Return the upper-body crop (jersey region) for a player bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    # Use top 40% as jersey area, avoiding the head (skip top 10%)
    jersey_y1 = y1 + int(h * 0.10)
    jersey_y2 = y1 + int(h * 0.50)
    if jersey_y2 <= jersey_y1 or x2 <= x1:
        return None
    crop = frame[jersey_y1:jersey_y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _hsv_histogram(crop: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compute a normalised HSV H+S 2D histogram flattened to 1D."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [bins, bins],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


class TeamClassifier:
    """
    Accumulates player colour features over multiple frames, then clusters
    into Team A / Team B using K-Means.

    Usage
    -----
    classifier = TeamClassifier()
    # During processing: collect features
    classifier.add_sample(track_id, frame, bbox)
    # After collecting enough samples:
    classifier.fit()
    # Query assignment:
    label = classifier.get_team(track_id)   # "team_a" | "team_b" | "referee"
    """

    def __init__(self, n_clusters: int = 2) -> None:
        self.n_clusters = n_clusters
        # track_id → list of HSV histograms
        self._features: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._assignments: Dict[int, str] = {}
        self._cluster_colors: Dict[str, str] = {}
        self._fitted = False

    def add_sample(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: List[float],
    ) -> None:
        """Add a colour observation for a tracked player."""
        crop = _jersey_crop(frame, bbox)
        if crop is None:
            return
        hist = _hsv_histogram(crop)
        self._features[track_id].append(hist)

    def fit(self) -> None:
        """
        Cluster collected features into two team groups.
        Silently fails if there are not enough samples.
        """
        if len(self._features) < self.n_clusters:
            logger.warning("Not enough player samples to classify teams ({} players)", len(self._features))
            return

        # Average histogram per player
        track_ids = list(self._features.keys())
        avg_hists = np.array(
            [np.mean(self._features[tid], axis=0) for tid in track_ids]
        )

        try:
            km = KMeans(
                n_clusters=self.n_clusters,
                n_init=10,
                random_state=42,
            )
            labels = km.fit_predict(avg_hists)
        except Exception as exc:
            logger.error("K-Means clustering failed: {}", exc)
            return

        # Assign team labels
        for tid, cluster in zip(track_ids, labels):
            self._assignments[tid] = f"team_{chr(ord('a') + cluster)}"

        # Compute representative BGR color per cluster
        for c in range(self.n_clusters):
            indices = [i for i, lbl in enumerate(labels) if lbl == c]
            cluster_hists = avg_hists[indices]
            # Find the dominant H bin
            mean_hist = cluster_hists.mean(axis=0).reshape(16, 16)
            h_idx, s_idx = np.unravel_index(mean_hist.argmax(), mean_hist.shape)
            h_val = int(h_idx * 180 / 16)
            s_val = int(s_idx * 256 / 16)
            bgr = cv2.cvtColor(
                np.array([[[h_val, s_val, 200]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR,
            )[0][0]
            hex_color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))
            self._cluster_colors[f"team_{chr(ord('a') + c)}"] = hex_color

        self._fitted = True
        logger.info(
            "Team classifier fitted. Colors: {}", self._cluster_colors
        )

    def get_team(self, track_id: int) -> str:
        """Return the team label for a track ID, or 'unknown'."""
        return self._assignments.get(track_id, "unknown")

    def get_color(self, team_label: str) -> str:
        """Return the hex color for a team label."""
        defaults = {"team_a": "#3b82f6", "team_b": "#ef4444"}
        return self._cluster_colors.get(team_label, defaults.get(team_label, "#888888"))

    def override_assignment(self, track_id: int, team_label: str) -> None:
        """Allow manual correction from the UI."""
        self._assignments[track_id] = team_label

    @property
    def fitted(self) -> bool:
        return self._fitted
