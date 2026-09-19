"""
Compare MLP vs LSTM vs Transformer on formation classification.

Trains all three models sequentially with shared config, then
produces a comparison table + models/comparison.json.

Usage
-----
    python training/compare_models.py --config training/configs/transformer_config.yaml

This reuses the base config for all three models.
Model-specific overrides (hidden_dim, lstm_hidden_dim) are embedded below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))           # makes 'training' importable
sys.path.insert(0, str(ROOT / "backend"))  # makes 'app' importable

from training.train import train


MODEL_OVERRIDES = {
    "mlp":         {"hidden_dim": 256},
    "lstm":        {"lstm_input_dim": 64, "lstm_hidden_dim": 128, "n_layers": 2},
    "transformer": {},   # uses base config as-is
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare MLP vs LSTM vs Transformer")
    parser.add_argument("--config", type=str,
                        default="training/configs/transformer_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    all_metrics = {}

    for model_type in ["mlp", "lstm", "transformer"]:
        logger.info("\n{'='*60}")
        logger.info("Training: {}", model_type.upper())
        logger.info("{'='*60}\n")

        config = {**base_config, **MODEL_OVERRIDES[model_type]}
        try:
            metrics = train(model_type, config)
            all_metrics[model_type] = metrics
        except Exception as e:
            logger.error("Training {} failed: {}", model_type, e)
            all_metrics[model_type] = {"error": str(e)}

    # Save comparison
    out_path = Path(base_config["output_dir"]) / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — FORMATION CLASSIFICATION")
    print("=" * 70)
    header = f"{'Model':<14} {'Accuracy':>10} {'Macro F1':>10} {'Params':>10} {'Train(s)':>10}"
    print(header)
    print("-" * 70)

    for model_type in ["mlp", "lstm", "transformer"]:
        m = all_metrics.get(model_type, {})
        if "error" in m:
            print(f"{model_type:<14}  ERROR: {m['error']}")
            continue
        acc  = f"{m.get('accuracy', 0)*100:.1f}%"
        f1   = f"{m.get('macro_f1', 0):.3f}"
        par  = f"{m.get('n_params', 0):,}"
        secs = f"{m.get('train_time_s', 0):.0f}s"
        print(f"{model_type:<14} {acc:>10} {f1:>10} {par:>10} {secs:>10}")

    print("=" * 70)
    print(f"\nNote: Labels are WEAKLY SUPERVISED (rule-based heuristic).")
    print(f"These metrics measure consistency with the rule-based detector,")
    print(f"not accuracy against human-annotated ground truth.")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
