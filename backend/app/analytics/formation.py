"""
Formation detection.

Two layers:
1. Rule-based baseline — clusters player X positions into rows and counts
   players per row to infer the formation string.
2. ML hook — if a trained Temporal Transformer exists at the configured
   path, it is used for classification instead.

The rule-based approach is the default for MVP.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from loguru import logger


# Supported formations (outfield players per line, back-to-front)
KNOWN_FORMATIONS = [
    "4-3-3",
    "4-2-3-1",
    "4-4-2",
    "3-4-3",
    "3-5-2",
    "5-3-2",
    "5-4-1",
    "4-1-4-1",
    "3-4-1-2",
]


def _infer_formation_from_lines(line_counts: List[int]) -> str:
    """
    Convert a list of player counts per line (defensive → attacking)
    into a formation string.
    """
    s = "-".join(str(c) for c in line_counts)
    # Direct match
    if s in KNOWN_FORMATIONS:
        return s
    # Find closest known formation by character similarity
    best = min(
        KNOWN_FORMATIONS,
        key=lambda f: _formation_distance(line_counts, f),
    )
    return best


def _formation_distance(counts: List[int], formation_str: str) -> int:
    """Simple edit-distance proxy between two formations."""
    parts = [int(x) for x in formation_str.split("-")]
    # Pad to same length
    n = max(len(counts), len(parts))
    a = counts + [0] * (n - len(counts))
    b = parts + [0] * (n - len(parts))
    return sum(abs(x - y) for x, y in zip(a, b))


def detect_formation_rule_based(
    positions: List[Tuple[float, float]],
    n_lines: int = 4,
) -> Tuple[str, float, Dict]:
    """
    Estimate formation from player pitch positions using K-Means line clustering.

    Parameters
    ----------
    positions : List[(pitch_x, pitch_y)]
        Outfield player positions (exclude GK).
    n_lines : int
        Number of lines to cluster into (usually 3 or 4).

    Returns
    -------
    formation : str
        e.g. "4-3-3"
    confidence : float
        0–1 confidence estimate based on cluster separation.
    explanation : dict
        Breakdown used for explainability UI.
    """
    # Remove goalkeeper (assume the deepest player is GK)
    if len(positions) < 6:
        return "unknown", 0.0, {"reason": "Insufficient players detected"}

    xs = np.array([p[0] for p in positions]).reshape(-1, 1)

    # Try different line counts and pick best silhouette
    best_formation = "4-3-3"
    best_conf = 0.3
    best_explanation: Dict = {}

    for k in range(3, min(5, len(positions) - 1)):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(xs)
        except Exception:
            continue

        # Sort clusters by mean X (defensive → attacking)
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        counts = [int((labels == c).sum()) for c in order]

        # Skip lines with 0 players
        counts = [c for c in counts if c > 0]
        if not counts:
            continue

        formation = _infer_formation_from_lines(counts)

        # Confidence: intra-cluster tightness relative to inter-cluster distance
        inertia = km.inertia_
        spread = float(np.std(centers))
        conf = float(np.clip(spread / (inertia + 1e-6) * 10, 0.4, 0.95))

        if conf > best_conf:
            best_conf = conf
            best_formation = formation
            best_explanation = {
                "lines": k,
                "players_per_line": counts,
                "cluster_centers_m": [round(centers[o], 1) for o in order],
            }

    explanation = {
        **best_explanation,
        "method": "rule_based",
        "description": (
            f"Players clustered into {best_explanation.get('lines', '?')} lines. "
            f"Counts per line (def→atk): {best_explanation.get('players_per_line', [])}."
        ),
    }

    return best_formation, best_conf, explanation


def detect_formation(
    positions: List[Tuple[float, float]],
    model_path: Optional[str] = None,
    sequence: Optional[np.ndarray] = None,
) -> Tuple[str, float, Dict]:
    """
    Main formation detection entry point.

    If a trained model path is provided and the model file exists, the
    Temporal Transformer is used.  Otherwise falls back to rule-based.
    """
    if model_path:
        from pathlib import Path
        if Path(model_path).exists():
            try:
                from app.models.temporal_model import TransformerFormationClassifier
                clf = TransformerFormationClassifier.load(model_path)
                if sequence is not None:
                    return clf.predict(sequence)
            except Exception as exc:
                logger.warning("Temporal model inference failed ({}), using rule-based.", exc)

    return detect_formation_rule_based(positions)
