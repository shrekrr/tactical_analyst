"""
Standalone inference script.

Usage
-----
    python training/inference.py \
        --model models/best_model.pt \
        --sequence path/to/sequence.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.models.temporal_model import TransformerFormationClassifier
from app.analytics.formation import KNOWN_FORMATIONS


def main():
    parser = argparse.ArgumentParser(description="Temporal Transformer — Formation Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt weight file")
    parser.add_argument("--sequence", type=str, required=True,
                        help="Path to .npy sequence (T, N*F)")
    args = parser.parse_args()

    model_path = Path(args.model)
    seq_path = Path(args.sequence)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    if not seq_path.exists():
        print(f"ERROR: Sequence not found: {seq_path}")
        sys.exit(1)

    sequence = np.load(str(seq_path)).astype(np.float32)
    print(f"Loaded sequence: shape={sequence.shape}")

    clf = TransformerFormationClassifier.load(str(model_path))
    formation, confidence, explanation = clf.predict(sequence)

    print("\n── Formation Prediction ─────────────────────")
    print(f"  Formation:  {formation}")
    print(f"  Confidence: {confidence * 100:.1f}%")
    print("\n── Top-3 ────────────────────────────────────")
    for entry in explanation.get("top_3", []):
        print(f"  {entry['formation']:<10} {entry['probability']*100:.1f}%")
    print("\n── Full explanation ─────────────────────────")
    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
