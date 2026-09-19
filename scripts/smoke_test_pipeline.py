"""
End-to-end pipeline smoke test (no model weights needed).
Runs mock data generation → prepare → sequence building → inspect.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def run(cmd, cwd=ROOT):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode

def main():
    py = sys.executable

    steps = [
        ([py, "scripts/download_soccernet.py", "--mock"],         "Step 1: Generate mock data"),
        ([py, "scripts/prepare_tracking_data.py"],                 "Step 2: Prepare tracking data"),
        ([py, "scripts/create_sequences.py", "--window", "20", "--stride", "5"], "Step 3: Create sequences"),
        ([py, "scripts/inspect_dataset.py"],                       "Step 4: Inspect dataset"),
    ]

    for cmd, label in steps:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print('='*60)
        rc = run(cmd)
        if rc != 0:
            print(f"\nFAILED at: {label} (exit code {rc})")
            sys.exit(rc)

    print(f"\n{'='*60}")
    print("  ✅ All pipeline steps completed successfully!")
    print("  You can now train models with:")
    print("    python training/train.py --model mlp         --config training/configs/mlp_config.yaml")
    print("    python training/train.py --model lstm        --config training/configs/lstm_config.yaml")
    print("    python training/train.py --model transformer --config training/configs/transformer_config.yaml")
    print('='*60)

if __name__ == "__main__":
    main()
