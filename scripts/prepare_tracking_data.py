"""
Convert SoccerNet Tracking annotations → unified per-frame JSONL.

Handles two source formats automatically:
  1. Mock format  — simple list with {frame, id, class, x_pitch, y_pitch, team}
  2. Real SoccerNet GSR-2024 COCO format — {info, images, annotations, categories}
     with bbox_pitch.x_bottom_middle / y_bottom_middle in centre-origin coords

Reads:
    data/raw/soccernet/tracking/{split}/{match_id}/Labels-GameState.json

Writes:
    data/processed/{split}__{match_id}/
        frames.jsonl      — one JSON line per detection
        meta.json         — match metadata + source info

Usage
-----
    python scripts/prepare_tracking_data.py          # real + mock data
    python scripts/prepare_tracking_data.py --mock   # mock only

Output format (one JSON object per line in frames.jsonl)
---------------------------------------------------------
    {
        "frame": 42,
        "timestamp_s": 8.4,
        "track_id": 7,
        "class": "player",         # player | ball | referee | goalkeeper
        "team": "home",            # home | away | null
        "pitch_x": 48.3,          # metres, [0, 105], left-to-right
        "pitch_y": 32.1,          # metres, [0, 68],  bottom-to-top
        "source": "soccernet_gt"  # soccernet_gt | synthetic_mock
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw" / "soccernet" / "tracking"
DATA_PROCESSED = ROOT / "data" / "processed"

# Standard SoccerNet pitch dimensions
PITCH_LENGTH = 105.0   # metres, x-axis
PITCH_WIDTH  = 68.0    # metres, y-axis

# SoccerNet COCO category_id → class string
CATEGORY_MAP = {
    1: "player",
    2: "goalkeeper",
    3: "referee",
    4: "ball",
    7: "other",
}

# SoccerNet "left"/"right" team → canonical "home"/"away"
# "left" = team attacking left→right = home (convention; flipped per sequence)
TEAM_MAP = {
    "left":  "home",
    "right": "away",
}


def _sn_to_corner(x_centre: float, y_centre: float) -> tuple[float, float]:
    """
    Convert SoccerNet pitch coords (centre = 0,0) to corner-origin.

    SoccerNet:  x ∈ [-52.5, +52.5],  y ∈ [-34, +34]
    Output:     x ∈ [0, 105],         y ∈ [0, 68]
    """
    return x_centre + PITCH_LENGTH / 2.0, y_centre + PITCH_WIDTH / 2.0


def _process_coco(data: dict, fps: float, match_id: str) -> list[dict]:
    """Parse real SoccerNet COCO-style Labels-GameState.json."""
    # Build image_id → frame_number map
    image_id_to_frame: dict[str, int] = {}
    for img in data.get("images", []):
        fname = img.get("file_name", "")
        try:
            frame_num = int(Path(fname).stem)
        except ValueError:
            frame_num = 0
        image_id_to_frame[str(img["image_id"])] = frame_num

    records: list[dict] = []
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id", ""))
        frame = image_id_to_frame.get(image_id, 0)

        # Category → class string
        cat_id = ann.get("category_id", 0)
        cls = CATEGORY_MAP.get(cat_id, "other")

        # Skip pitch / camera annotations (category 5,6)
        if cat_id in (5, 6):
            continue

        # Pitch position from bbox_pitch bottom-middle foot point
        bbox_pitch = ann.get("bbox_pitch") or {}
        x_c = bbox_pitch.get("x_bottom_middle")
        y_c = bbox_pitch.get("y_bottom_middle")

        if x_c is None or y_c is None:
            continue  # no calibrated position → skip

        pitch_x, pitch_y = _sn_to_corner(x_c, y_c)

        # Clamp to pitch bounds (calibration errors can push slightly outside)
        pitch_x = max(0.0, min(PITCH_LENGTH, pitch_x))
        pitch_y = max(0.0, min(PITCH_WIDTH, pitch_y))

        # Team
        attrs = ann.get("attributes") or {}
        raw_team = attrs.get("team", None)
        team = TEAM_MAP.get(raw_team, None) if raw_team else None

        # Track ID
        track_id = ann.get("track_id")

        records.append({
            "frame": frame,
            "timestamp_s": round(frame / fps, 3),
            "track_id": track_id,
            "class": cls,
            "team": team,
            "pitch_x": round(pitch_x, 4),
            "pitch_y": round(pitch_y, 4),
            "source": "soccernet_gt",
        })

    return records


def _process_mock(data: dict, fps: float) -> list[dict]:
    """Parse mock / simple annotation format with x_pitch / y_pitch directly."""
    records = []
    for ann in data.get("annotations", []):
        x = ann.get("x_pitch")
        y = ann.get("y_pitch")
        if x is None or y is None:
            continue
        records.append({
            "frame":        ann.get("frame", 0),
            "timestamp_s":  round(ann.get("frame", 0) / fps, 3),
            "track_id":     ann.get("id"),
            "class":        ann.get("class", "player"),
            "team":         ann.get("team"),
            "pitch_x":      round(float(x), 4),
            "pitch_y":      round(float(y), 4),
            "source":       "synthetic_mock",
        })
    return records


def process_match(labels_path: Path, match_id: str) -> int:
    """
    Convert one match's Labels-GameState.json → frames.jsonl.
    Returns number of frames written.
    """
    out_dir = DATA_PROCESSED / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)

    # Auto-detect format: real COCO format has "images" and "categories" keys
    is_coco = "images" in data and "categories" in data

    if is_coco:
        info = data.get("info", {})
        fps = float(info.get("frame_rate", 25))
        records = _process_coco(data, fps, match_id)
        pitch_length = PITCH_LENGTH
        pitch_width = PITCH_WIDTH
        source_label = "soccernet_gt"
    else:
        fps = float(data.get("fps", 5.0))
        records = _process_mock(data, fps)
        pitch_length = data.get("pitch_length_m", PITCH_LENGTH)
        pitch_width = data.get("pitch_width_m", PITCH_WIDTH)
        source_label = "synthetic_mock"

    # Sort by frame then track for deterministic ordering
    records.sort(key=lambda r: (r["frame"], r.get("track_id") or 0))

    with open(out_dir / "frames.jsonl", "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec) + "\n")

    # Metadata
    meta = {
        "match_id":       match_id,
        "fps":            fps,
        "pitch_length_m": pitch_length,
        "pitch_width_m":  pitch_width,
        "n_records":      len(records),
        "source":         source_label,
        "format":         "coco_gsr" if is_coco else "mock_simple",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SoccerNet tracking data")
    parser.add_argument("--mock", action="store_true",
                        help="Process only mock data directories")
    args = parser.parse_args()

    if not DATA_RAW.exists():
        print(f"ERROR: Raw data not found at {DATA_RAW}")
        print("  Run: python scripts/download_soccernet.py --mock")
        sys.exit(1)

    total_matches = 0
    total_records = 0

    for split_dir in sorted(DATA_RAW.iterdir()):
        if not split_dir.is_dir() or split_dir.name.startswith("__"):
            continue
        split = split_dir.name

        for match_dir in sorted(split_dir.iterdir()):
            if not match_dir.is_dir():
                continue

            # --mock flag: only process directories whose name starts with MOCK
            if args.mock and not match_dir.name.startswith("MOCK"):
                continue

            labels_path = match_dir / "Labels-GameState.json"
            if not labels_path.exists():
                continue

            match_id = f"{split}__{match_dir.name}"
            n = process_match(labels_path, match_id)
            print(f"  {split}/{match_dir.name} -> {match_id} ({n:,} records)")
            total_matches += 1
            total_records += n

    print(f"\nProcessed {total_matches} matches, {total_records:,} tracking records")
    print(f"Output: {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
