"""
Training script for the Temporal Formation Transformer.

Usage
-----
    python training/train.py --config training/configs/transformer_config.yaml

The training loop includes:
- Train / validation / test split
- Early stopping
- TensorBoard logging
- Checkpoint saving
- Learning-rate scheduling
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.models.temporal_model import TemporalFormationTransformer
from app.analytics.formation import KNOWN_FORMATIONS


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(data_dir: Path, seq_len: int, n_players: int, n_features: int):
    """
    Load pre-processed sequences from .npy files.

    Expected files:
        data/sequences.npy   shape (N, T, n_players * n_features)
        data/labels.npy      shape (N,)  integer class indices
    """
    seq_path = data_dir / "sequences.npy"
    lbl_path = data_dir / "labels.npy"

    if not seq_path.exists() or not lbl_path.exists():
        logger.warning("No dataset found at {}. Using synthetic data for demonstration.", data_dir)
        X, y = _generate_synthetic(n_samples=200, seq_len=seq_len,
                                   n_players=n_players, n_features=n_features)
    else:
        X = np.load(str(seq_path)).astype(np.float32)
        y = np.load(str(lbl_path)).astype(np.int64)
        logger.info("Loaded dataset: X={} y={}", X.shape, y.shape)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def _generate_synthetic(
    n_samples: int, seq_len: int, n_players: int, n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random synthetic sequences for architecture validation."""
    logger.warning("Using synthetic data — model will not produce meaningful results.")
    X = np.random.randn(n_samples, seq_len, n_players * n_features).astype(np.float32)
    y = np.random.randint(0, len(KNOWN_FORMATIONS), n_samples).astype(np.int64)
    return X, y


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: {}", device)

    # ── Hyperparameters ────────────────────────────────────────────────────────
    seq_len     = config["seq_len"]
    n_players   = config["n_players"]
    n_features  = config["n_features"]
    d_model     = config["d_model"]
    n_heads     = config["n_heads"]
    n_layers    = config["n_layers"]
    dropout     = config["dropout"]
    batch_size  = config["batch_size"]
    lr          = config["lr"]
    max_epochs  = config["max_epochs"]
    patience    = config["patience"]
    output_dir  = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    data_dir = Path(config.get("data_dir", "training/datasets"))
    X, y = load_dataset(data_dir, seq_len, n_players, n_features)
    dataset = TensorDataset(X, y)

    n_total = len(dataset)
    n_test  = max(1, int(n_total * 0.1))
    n_val   = max(1, int(n_total * 0.15))
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────────
    model_config = dict(
        n_players=n_players,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=len(KNOWN_FORMATIONS),
        seq_len=seq_len,
        dropout=dropout,
    )
    model = TemporalFormationTransformer(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: {:,}", n_params)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve = 0
    best_ckpt = output_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += len(Xb)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(Xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(Xb)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        logger.info(
            "Epoch {:3d}/{} | train loss={:.4f} acc={:.3f} | val loss={:.4f} acc={:.3f}",
            epoch, max_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": model_config},
                str(best_ckpt),
            )
            logger.info("  ✓ Checkpoint saved (val_loss={:.4f})", val_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch {}", epoch)
                break

    writer.close()

    # ── Test evaluation ───────────────────────────────────────────────────────
    from training.evaluate import evaluate_model
    logger.info("Running test evaluation...")
    model.load_state_dict(torch.load(str(best_ckpt), map_location=device)["model_state_dict"])
    evaluate_model(model, test_dl, device, KNOWN_FORMATIONS, output_dir)

    logger.info("Training complete. Best model saved to {}", best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/transformer_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
