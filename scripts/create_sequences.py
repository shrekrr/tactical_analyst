"""
Build sliding-window sequences from processed tracking data.

Reads:    data/processed/{match_id}/frames.jsonl + meta.json
Writes:   training/datasets/
              sequences.npy       shape (N, T, n_players * n_features)
              labels.npy          shape (N,)  formation class index
              match_ids.npy       shape (N,)  string match_id per sequence
              split_manifest.json  maps match_id -> train|val|test

Split strategy: BY MATCH (not by frame) to prevent leakage.
  80% of matches → train
  10% of matches → val
  10% of matches → test

Label strategy: Weakly supervised via rule-based formation detector.
  Labels are heuristic, not human-annotated.

Sequence format per timestep:
  [x_1, y_1, vx_1, vy_1, ..., x_11, y_11, vx_11, vy_11]
  Players are sorted by pitch_x (deepest defender first) and padded with
  zeros if fewer than n_players are detected in a frame.

Coordinates are normalized:
  x / pitch_length_m  → [0, 1]
  y / pitch_width_m   → [0, 1]
  velocities clipped to [-1, 1] (units: pitch_lengths per frame)

Usage
-----
    python scripts/create_sequences.py --window 20 --stride 5
    python scripts/create_sequences.py --window 20 --stride 5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

DATA_PROCESSED = ROOT / "data" / "processed"
DATASETS_OUT = ROOT / "training" / "datasets"

from app.analytics.formation import KNOWN_FORMATIONS, detect_formation_rule_based

PITCH_L = 105.0
PITCH_W = 68.0
N_PLAYERS = 11      # players per team (padded with zeros if fewer)
N_FEATURES = 4      # x, y, vx, vy per player


def load_match(match_id: str) -> Tuple[Dict[int, List], float, str]:
    """
    Load one match's JSONL into a per-frame, per-team dictionary.

    Returns
    -------
    frames_home : Dict[frame -> List[dict]]
    fps : float
    source : str
    """
    match_dir = DATA_PROCESSED / match_id
    meta = json.loads((match_dir / "meta.json").read_text())
    fps = meta.get("fps", 5.0)
    source = meta.get("source", "unknown")

    frames_home: Dict[int, List] = defaultdict(list)
    frames_away: Dict[int, List] = defaultdict(list)

    with open(match_dir / "frames.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["class"] != "player":
                continue
            if r["pitch_x"] is None or r["pitch_y"] is None:
                continue
            entry = {
                "track_id": r["track_id"],
                "x": float(r["pitch_x"]),
                "y": float(r["pitch_y"]),
            }
            if r.get("team") == "home":
                frames_home[r["frame"]].append(entry)
            elif r.get("team") == "away":
                frames_away[r["frame"]].append(entry)

    return frames_home, frames_away, fps, source


def build_team_sequence(
    frames: Dict[int, List],
    frame_ids: List[int],
    fps: float,
) -> np.ndarray:
    """
    Build a (T, N_PLAYERS * N_FEATURES) array for one team across frame_ids.

    Player slot assignment: sorted by x-coordinate each frame (deepest first).
    Velocity: finite difference between consecutive frames.
    """
    seq = np.zeros((len(frame_ids), N_PLAYERS * N_FEATURES), dtype=np.float32)
    prev_positions: Optional[np.ndarray] = None

    for t, fid in enumerate(frame_ids):
        players = frames.get(fid, [])
        # Sort by x-coordinate (defensive → offensive)
        players = sorted(players, key=lambda p: p["x"])[:N_PLAYERS]

        positions = np.zeros((N_PLAYERS, 2), dtype=np.float32)
        for i, p in enumerate(players):
            positions[i, 0] = p["x"] / PITCH_L   # normalised x
            positions[i, 1] = p["y"] / PITCH_W   # normalised y

        # Velocity
        if prev_positions is not None:
            velocities = np.clip((positions - prev_positions) * fps, -1.0, 1.0)
        else:
            velocities = np.zeros_like(positions)

        prev_positions = positions.copy()

        # Flatten: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        for i in range(N_PLAYERS):
            base = i * N_FEATURES
            seq[t, base]     = positions[i, 0]
            seq[t, base + 1] = positions[i, 1]
            seq[t, base + 2] = velocities[i, 0]
            seq[t, base + 3] = velocities[i, 1]

    return seq


def label_window(frames: Dict[int, List], frame_ids: List[int]) -> int:
    """
    Apply rule-based formation detector to the middle frame of a window.
    Returns the class index (into KNOWN_FORMATIONS) or -1 if unknown.

    Label origin: WEAKLY SUPERVISED — rule-based heuristic, not human annotated.
    """
    mid = frame_ids[len(frame_ids) // 2]
    players = frames.get(mid, [])
    if len(players) < 6:
        return -1

    positions = [(p["x"], p["y"]) for p in players]
    formation, conf, _ = detect_formation_rule_based(positions)

    if formation not in KNOWN_FORMATIONS:
        return -1
    return KNOWN_FORMATIONS.index(formation)


def process_match_to_sequences(
    match_id: str,
    window: int,
    stride: int,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Extract all sliding-window sequences from one match.

    Returns
    -------
    sequences : List of (T, N*F) arrays
    labels    : List of int (formation class index)
    ids       : List of str (match_id, one per sequence for split tracking)
    """
    frames_home, frames_away, fps, source = load_match(match_id)

    all_frame_ids = sorted(set(frames_home.keys()) | set(frames_away.keys()))
    if len(all_frame_ids) < window:
        return [], [], []

    sequences = []
    labels = []
    ids = []

    for start in range(0, len(all_frame_ids) - window + 1, stride):
        window_frames = all_frame_ids[start:start + window]

        # Home team sequence
        home_label = label_window(frames_home, window_frames)
        if home_label >= 0:
            home_seq = build_team_sequence(frames_home, window_frames, fps)
            sequences.append(home_seq)
            labels.append(home_label)
            ids.append(match_id)

        # Away team sequence
        away_label = label_window(frames_away, window_frames)
        if away_label >= 0:
            away_seq = build_team_sequence(frames_away, window_frames, fps)
            sequences.append(away_seq)
            labels.append(away_label)
            ids.append(match_id)

    return sequences, labels, ids


def split_by_match(
    match_ids: List[str],
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, str]:
    """
    Assign each match to train/val/test.

    Returns Dict[match_id -> 'train'|'val'|'test'].
    NEVER splits individual matches across sets.
    """
    rng = random.Random(seed)
    shuffled = sorted(set(match_ids))
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    assignment = {}
    for i, m in enumerate(shuffled):
        if i < n_test:
            assignment[m] = "test"
        elif i < n_test + n_val:
            assignment[m] = "val"
        else:
            assignment[m] = "train"

    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training sequences from SoccerNet tracking data")
    parser.add_argument("--window", type=int, default=20, help="Frames per sequence (default: 20)")
    parser.add_argument("--stride", type=int, default=5, help="Sliding window stride (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split (default: 42)")
    args = parser.parse_args()

    if not DATA_PROCESSED.exists() or not any(DATA_PROCESSED.iterdir()):
        print(f"ERROR: No processed data found at {DATA_PROCESSED}")
        print("  Run first: python scripts/prepare_tracking_data.py")
        sys.exit(1)

    match_ids = [d.name for d in sorted(DATA_PROCESSED.iterdir()) if d.is_dir()]
    print(f"Found {len(match_ids)} processed matches")

    all_sequences: List[np.ndarray] = []
    all_labels: List[int] = []
    all_match_ids: List[str] = []

    for match_id in match_ids:
        seqs, lbls, ids = process_match_to_sequences(match_id, args.window, args.stride)
        all_sequences.extend(seqs)
        all_labels.extend(lbls)
        all_match_ids.extend(ids)
        print(f"  {match_id}: {len(seqs)} sequences")

    if not all_sequences:
        print("ERROR: No sequences generated. Check processed data.")
        sys.exit(1)

    # Build match-level split
    split_assignment = split_by_match(all_match_ids, seed=args.seed)

    # Save numpy arrays
    DATASETS_OUT.mkdir(parents=True, exist_ok=True)

    X = np.stack(all_sequences).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    m = np.array(all_match_ids)

    np.save(str(DATASETS_OUT / "sequences.npy"), X)
    np.save(str(DATASETS_OUT / "labels.npy"), y)
    np.save(str(DATASETS_OUT / "match_ids.npy"), m)

    # Save split manifest
    manifest = {
        "seed": args.seed,
        "window": args.window,
        "stride": args.stride,
        "n_sequences": len(all_sequences),
        "n_matches": len(set(all_match_ids)),
        "label_source": "weakly_supervised_rule_based",
        "label_note": (
            "Formation labels are derived from the rule-based K-Means detector. "
            "They are NOT human annotated. Treat as weakly supervised."
        ),
        "formations": KNOWN_FORMATIONS,
        "match_split": split_assignment,
        "class_distribution": {
            KNOWN_FORMATIONS[i]: int((y == i).sum())
            for i in range(len(KNOWN_FORMATIONS))
        },
    }
    with open(DATASETS_OUT / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"\n{'='*55}")
    print(f"Dataset built: {len(all_sequences)} sequences")
    print(f"Shape: {X.shape}")
    print(f"\nSplit (by match):")
    counts = {"train": 0, "val": 0, "test": 0}
    for mid in all_match_ids:
        counts[split_assignment[mid]] += 1
    for split_name, cnt in counts.items():
        print(f"  {split_name:6s}: {cnt} sequences")

    print(f"\nClass distribution:")
    for cls, cnt in manifest["class_distribution"].items():
        bar = "█" * min(30, cnt)
        print(f"  {cls:<10} {cnt:4d}  {bar}")

    print(f"\nLabel origin: WEAKLY SUPERVISED (rule-based heuristic)")
    print(f"Output: {DATASETS_OUT}")
    print(f"Manifest: {DATASETS_OUT}/split_manifest.json")


if __name__ == "__main__":
    main()
