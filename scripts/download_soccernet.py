"""
Download SoccerNet Tracking data.

Usage
-----
    # Mock mode (no account needed — generates synthetic data for pipeline testing)
    python scripts/download_soccernet.py --mock

    # Real download (requires SoccerNet password)
    export SOCCERNET_PASSWORD="your_password_here"
    python scripts/download_soccernet.py --subset dev      # 5 sequences ~500 MB
    python scripts/download_soccernet.py --subset full     # all sequences

Output
------
    data/raw/soccernet/tracking/
        {split}/  (train / valid / test)
            {match_id}/
                Labels-GameState.json     # game state annotations (positions + calibration)
                gameinfo.json

    data/raw/soccernet/calibration/
        {split}/{match_id}/
            calibration.json              # camera calibration per frame


Environment Variables
---------------------
    SOCCERNET_PASSWORD  — required for real download (set in .env or shell)
    SOCCERNET_DATA_DIR  — override default data root (default: data/raw/soccernet)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

DATA_DIR_DEFAULT = ROOT / "data" / "raw" / "soccernet"

TRACKING_SPLITS = ["train", "valid", "test"]

# Small dev subset: 5 SoccerNet tracking sequences
DEV_SUBSET_IDS = [
    "SNGS-001",
    "SNGS-002",
    "SNGS-003",
    "SNGS-004",
    "SNGS-005",
]


def download_real(data_dir: Path, subset: str) -> None:
    """
    Download SoccerNet GameState-2024 via HuggingFace.

    Dataset: SoccerNet/SN-GSR-2024 on HuggingFace
    Contains: Labels-GameState.json (player tracks + pitch calibration per sequence)

    Access: Request at https://huggingface.co/datasets/SoccerNet/SN-GSR-2024
    Then set HF_TOKEN environment variable with your HuggingFace access token.
    """
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("ERROR: SoccerNet package not installed.  pip install SoccerNet")
        sys.exit(1)

    # HuggingFace token (optional — public if access already granted)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print(f"Logged in to HuggingFace.")
        except ImportError:
            print("huggingface_hub not installed — skipping HF login.")

    data_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"] if subset == "full" else ["train", "valid"]

    print(f"Downloading SoccerNet GameState-2024 ({subset}: {splits}) ...")
    print("  Source: HuggingFace — SoccerNet/SN-GSR-2024")
    print("  (No password needed — access controlled via HuggingFace repo permissions)")

    dl = SoccerNetDownloader(LocalDirectory=str(data_dir))

    # The correct API: downloadDataTask with 'gamestate-2024'
    # Downloads .zip files from HuggingFace and extracts to data_dir/gamestate-2024/
    dl.downloadDataTask(
        task="gamestate-2024",
        split=splits,
        verbose=True,
        source="HuggingFace",
    )

    # The data lands in data_dir/gamestate-2024/{split}.zip — extract it
    import zipfile
    gsr_dir = data_dir / "gamestate-2024"
    tracking_dir = data_dir / "tracking"
    tracking_dir.mkdir(exist_ok=True)

    for spl in splits:
        zip_path = gsr_dir / f"{spl}.zip"
        if zip_path.exists():
            print(f"  Extracting {zip_path.name} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tracking_dir / spl)
            print(f"    -> {tracking_dir / spl}")
        else:
            print(f"  WARNING: {zip_path} not found — skipping.")

    print(f"\nDownload complete. Tracking data at: {tracking_dir}")
    print("You can now run: python scripts/prepare_tracking_data.py")


def generate_mock(data_dir: Path) -> None:
    """
    Generate synthetic SoccerNet-like annotations for pipeline testing.

    Produces the same directory structure and JSON format as real SoccerNet
    Tracking data so that downstream scripts work without modification.

    NOT suitable for measuring real model performance.
    """
    print("Generating synthetic SoccerNet-like mock data...")
    print("NOTE: This data is random — it cannot be used to train a meaningful model.")

    random.seed(42)

    PITCH_L = 105.0
    PITCH_W = 68.0

    for split in ["train", "valid", "test"]:
        n_seqs = 4 if split == "train" else 1
        for seq_idx in range(n_seqs):
            match_id = f"MOCK-{split[0].upper()}{seq_idx+1:03d}"
            seq_dir = data_dir / "tracking" / split / match_id
            seq_dir.mkdir(parents=True, exist_ok=True)

            # Build synthetic annotations
            n_frames = random.randint(150, 300)  # ~30–60 seconds at 5 FPS
            n_players_a = 10
            n_players_b = 10
            n_refs = 1

            # Random starting positions for each player
            start_positions = {}
            for pid in range(n_players_a + n_players_b + n_refs):
                start_positions[pid] = {
                    "x": random.uniform(5, PITCH_L - 5),
                    "y": random.uniform(5, PITCH_W - 5),
                }

            annotations = []
            for f in range(n_frames):
                t = f * 0.2  # 5 FPS → 0.2s per frame

                # Ball
                annotations.append({
                    "gameTime": f"{int(t//60):02d}:{t%60:05.2f}",
                    "frame": f,
                    "id": 0,
                    "class": "ball",
                    "x_pitch": float(PITCH_L / 2 + 10 * random.gauss(0, 1)),
                    "y_pitch": float(PITCH_W / 2 + 5 * random.gauss(0, 1)),
                    "team": None,
                })

                # Players
                for pid in range(n_players_a + n_players_b):
                    team = "home" if pid < n_players_a else "away"
                    sp = start_positions[pid]
                    x = float(sp["x"] + random.gauss(0, 0.3))
                    y = float(sp["y"] + random.gauss(0, 0.3))
                    x = max(0.0, min(PITCH_L, x))
                    y = max(0.0, min(PITCH_W, y))
                    sp["x"] = x
                    sp["y"] = y

                    annotations.append({
                        "gameTime": f"{int(t//60):02d}:{t%60:05.2f}",
                        "frame": f,
                        "id": pid + 1,
                        "class": "player",
                        "x_pitch": x,
                        "y_pitch": y,
                        "team": team,
                    })

                # Referee
                ref_x = float(start_positions[n_players_a + n_players_b]["x"] + random.gauss(0, 0.5))
                ref_y = float(start_positions[n_players_a + n_players_b]["y"] + random.gauss(0, 0.5))
                start_positions[n_players_a + n_players_b]["x"] = ref_x
                start_positions[n_players_a + n_players_b]["y"] = ref_y
                annotations.append({
                    "gameTime": f"{int(t//60):02d}:{t%60:05.2f}",
                    "frame": f,
                    "id": n_players_a + n_players_b + 1,
                    "class": "referee",
                    "x_pitch": ref_x,
                    "y_pitch": ref_y,
                    "team": None,
                })

            labels = {
                "match_id": match_id,
                "split": split,
                "source": "mock",
                "fps": 5.0,
                "pitch_length_m": PITCH_L,
                "pitch_width_m": PITCH_W,
                "annotations": annotations,
            }

            with open(seq_dir / "Labels-GameState.json", "w") as f_out:
                json.dump(labels, f_out, indent=2)

            # Minimal gameinfo
            with open(seq_dir / "gameinfo.json", "w") as f_out:
                json.dump({
                    "match_id": match_id,
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "source": "mock",
                }, f_out)

            print(f"  Created {split}/{match_id} ({n_frames} frames)")

    # Write a manifest
    manifest = {
        "source": "mock",
        "note": (
            "Synthetic data generated for pipeline validation only. "
            "Model metrics on this data are meaningless."
        ),
        "splits": {
            "train": ["MOCK-T001", "MOCK-T002", "MOCK-T003", "MOCK-T004"],
            "valid": ["MOCK-V001"],
            "test":  ["MOCK-E001"],
        },
    }
    with open(data_dir / "tracking" / "manifest.json", "w") as f_out:
        json.dump(manifest, f_out, indent=2)

    print(f"\nMock data created at: {data_dir}/tracking/")
    print("You can now run: python scripts/prepare_tracking_data.py --mock")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or mock SoccerNet Tracking data")
    parser.add_argument(
        "--subset",
        choices=["dev", "full"],
        default="dev",
        help="Download subset size (dev=5 sequences, full=all)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic data instead of downloading (no account needed)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR_DEFAULT),
        help=f"Root directory for data (default: {DATA_DIR_DEFAULT})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.mock:
        generate_mock(data_dir)
    else:
        download_real(data_dir, args.subset)


if __name__ == "__main__":
    main()
