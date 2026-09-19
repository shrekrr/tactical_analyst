"""
Convert SoccerNet Tracking annotations → unified per-frame JSONL.

Reads:
    data/raw/soccernet/tracking/{split}/{match_id}/Labels-GameState.json

Writes:
    data/processed/{match_id}/
        frames.jsonl      — one JSON line per detection
        meta.json         — match metadata + source info

Usage
-----
    python scripts/prepare_tracking_data.py          # real data
    python scripts/prepare_tracking_data.py --mock   # mock data

Output format (one JSON object per line in frames.jsonl)
---------------------------------------------------------
    {
        "frame": 42,
        "timestamp_s": 8.4,
        "track_id": 7,
        "class": "player",         # player | ball | referee
        "team": "home",            # home | away | null
        "pitch_x": 48.3,          # metres from left touchline
        "pitch_y": 32.1,          # metres from bottom byline
        "source": "soccernet_gt"   # provenance label
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


def process_match(labels_path: Path, match_id: str) -> int:
    """
    Convert one match's Labels-GameState.json → frames.jsonl.

    Returns number of frames written.
    """
    out_dir = DATA_PROCESSED / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path) as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    fps = data.get("fps", 5.0)
    source_label = "soccernet_gt" if data.get("source") != "mock" else "synthetic_mock"

    lines_written = 0
    with open(out_dir / "frames.jsonl", "w") as out_f:
        for ann in annotations:
            frame = ann.get("frame", 0)
            record = {
                "frame": frame,
                "timestamp_s": round(frame / fps, 3),
                "track_id": ann.get("id"),
                "class": ann.get("class"),
                "team": ann.get("team"),
                "pitch_x": ann.get("x_pitch"),
                "pitch_y": ann.get("y_pitch"),
                "source": source_label,
            }
            # Skip records with no position
            if record["pitch_x"] is None or record["pitch_y"] is None:
                continue
            out_f.write(json.dumps(record) + "\n")
            lines_written += 1

    # Write metadata
    meta = {
        "match_id": match_id,
        "fps": fps,
        "pitch_length_m": data.get("pitch_length_m", 105.0),
        "pitch_width_m": data.get("pitch_width_m", 68.0),
        "n_records": lines_written,
        "source": source_label,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return lines_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SoccerNet tracking data")
    parser.add_argument("--mock", action="store_true",
                        help="Process mock data (same code path, different source label)")
    args = parser.parse_args()

    if not DATA_RAW.exists():
        print(f"ERROR: Raw data not found at {DATA_RAW}")
        print("  Run: python scripts/download_soccernet.py --mock")
        sys.exit(1)

    total_matches = 0
    total_records = 0

    for split_dir in sorted(DATA_RAW.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "__pycache__":
            continue
        split = split_dir.name

        for match_dir in sorted(split_dir.iterdir()):
            if not match_dir.is_dir():
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
