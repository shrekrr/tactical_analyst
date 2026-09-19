"""
Inspect the generated training dataset.

Usage: python scripts/inspect_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATASETS_OUT = ROOT / "training" / "datasets"


def main() -> None:
    manifest_path = DATASETS_OUT / "split_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: No dataset found at {DATASETS_OUT}")
        print("  Run: python scripts/create_sequences.py")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    X = np.load(str(DATASETS_OUT / "sequences.npy"))
    y = np.load(str(DATASETS_OUT / "labels.npy"))
    m = np.load(str(DATASETS_OUT / "match_ids.npy"), allow_pickle=True)

    split = manifest["match_split"]
    formations = manifest["formations"]

    print("=" * 60)
    print("EPL TACTICAL ANALYST — DATASET INSPECTOR")
    print("=" * 60)

    print(f"\nSequence array: {X.shape}  (N, T, features)")
    print(f"Label array:    {y.shape}")
    print(f"Feature dim:    {X.shape[2]} = 11 players × 4 features")
    print(f"Window length:  {manifest['window']} frames")
    print(f"Stride:         {manifest['stride']} frames")
    print(f"Total matches:  {manifest['n_matches']}")

    print(f"\nLabel source: {manifest.get('label_source', '?')}")
    print(f"Note: {manifest.get('label_note', '')}")

    # Split breakdown
    print("\n── Split Breakdown (by match) " + "─" * 30)
    split_seq_counts = {"train": 0, "val": 0, "test": 0}
    split_match_sets = {"train": set(), "val": set(), "test": set()}

    for mid, s in split.items():
        split_match_sets[s].add(mid)

    for i, mid in enumerate(m):
        s = split.get(mid, "train")
        split_seq_counts[s] += 1

    for s in ["train", "val", "test"]:
        n_matches = len(split_match_sets[s])
        n_seqs = split_seq_counts[s]
        print(f"  {s:6s}:  {n_matches} matches  |  {n_seqs} sequences")

    # Check for leakage
    train_matches = split_match_sets["train"]
    val_matches = split_match_sets["val"]
    test_matches = split_match_sets["test"]
    overlap_tv = train_matches & val_matches
    overlap_tt = train_matches & test_matches
    overlap_vt = val_matches & test_matches
    if overlap_tv or overlap_tt or overlap_vt:
        print("\n⚠  LEAKAGE DETECTED!")
        print(f"  Train ∩ Val:  {overlap_tv}")
        print(f"  Train ∩ Test: {overlap_tt}")
        print(f"  Val ∩ Test:   {overlap_vt}")
    else:
        print("\n  ✓ No match-level leakage detected")

    # Class distribution
    print("\n── Class Distribution " + "─" * 38)
    max_cnt = max((y == i).sum() for i in range(len(formations)))
    for i, f in enumerate(formations):
        cnt = int((y == i).sum())
        pct = cnt / len(y) * 100
        bar = "█" * int(cnt / max(max_cnt, 1) * 30)
        print(f"  {f:<12}  {cnt:5d}  ({pct:5.1f}%)  {bar}")

    # Value ranges
    print("\n── Feature Value Ranges " + "─" * 36)
    print(f"  x coord:  [{X[:, :, 0::4].min():.3f}, {X[:, :, 0::4].max():.3f}]  (expect [0,1])")
    print(f"  y coord:  [{X[:, :, 1::4].min():.3f}, {X[:, :, 1::4].max():.3f}]  (expect [0,1])")
    print(f"  vx:       [{X[:, :, 2::4].min():.3f}, {X[:, :, 2::4].max():.3f}]  (expect [-1,1])")
    print(f"  vy:       [{X[:, :, 3::4].min():.3f}, {X[:, :, 3::4].max():.3f}]  (expect [-1,1])")

    print(f"\nOutput directory: {DATASETS_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
