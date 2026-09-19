"""
Unified training script for formation classifiers.

Supports: mlp | lstm | transformer

Usage
-----
    python training/train.py --model transformer --config training/configs/transformer_config.yaml
    python training/train.py --model lstm        --config training/configs/lstm_config.yaml
    python training/train.py --model mlp         --config training/configs/mlp_config.yaml

TensorBoard logs:
    tensorboard --logdir models/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))            # makes 'training' importable
sys.path.insert(0, str(ROOT / "backend"))  # makes 'app' importable

from app.analytics.formation import KNOWN_FORMATIONS
from training.preprocessing.soccernet_loader import FormationSequenceDataset


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type: str, config: dict) -> nn.Module:
    """Return the correct model class given model_type."""
    n_classes = len(KNOWN_FORMATIONS)
    seq_len = config["seq_len"]
    n_players = config["n_players"]
    n_features = config["n_features"]
    dropout = config.get("dropout", 0.1)

    if model_type == "mlp":
        from app.models.mlp_model import MLPFormationClassifier
        return MLPFormationClassifier(
            seq_len=seq_len,
            n_players=n_players,
            n_features=n_features,
            hidden_dim=config.get("hidden_dim", 256),
            n_classes=n_classes,
            dropout=dropout,
        )
    elif model_type == "lstm":
        from app.models.lstm_model import LSTMFormationClassifier
        return LSTMFormationClassifier(
            n_players=n_players,
            n_features=n_features,
            lstm_input_dim=config.get("lstm_input_dim", 64),
            lstm_hidden_dim=config.get("lstm_hidden_dim", 128),
            n_layers=config.get("n_layers", 2),
            n_classes=n_classes,
            dropout=dropout,
        )
    elif model_type == "transformer":
        from app.models.temporal_model import TemporalFormationTransformer
        return TemporalFormationTransformer(
            n_players=n_players,
            n_features=n_features,
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 3),
            n_classes=n_classes,
            seq_len=seq_len,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose mlp | lstm | transformer")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model_type: str, config: dict) -> dict:
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}  (CUDA: {})", device, torch.cuda.is_available())
    if device == "cuda":
        logger.info("GPU: {}", torch.cuda.get_device_name(0))

    # ── Config ────────────────────────────────────────────────────────────────
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]
    output_dir = Path(config["output_dir"]) / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_dir = Path(config.get("data_dir", "training/datasets"))

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets from {}", datasets_dir)
    train_ds = FormationSequenceDataset("train", datasets_dir, augment=True, seq_len=seq_len)
    val_ds   = FormationSequenceDataset("val",   datasets_dir, augment=False, seq_len=seq_len)
    test_ds  = FormationSequenceDataset("test",  datasets_dir, augment=False, seq_len=seq_len)

    logger.info("Sequences — train: {}  val: {}  test: {}", len(train_ds), len(val_ds), len(test_ds))

    if len(train_ds) == 0:
        raise RuntimeError("Training set is empty. Run scripts/create_sequences.py first.")

    # Class-weighted loss for imbalanced labels
    class_weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=(device == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_type, config).to(device)
    n_params = count_params(model)
    logger.info("Model: {}  Params: {:,}", model_type, n_params)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve = 0
    best_ckpt = output_dir / "best_model.pt"
    model_config = {k: v for k, v in config.items() if k not in ("output_dir", "data_dir")}

    train_start = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train epoch
        model.train()
        t_loss = t_correct = t_total = 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * len(Xb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += len(Xb)

        # Validation epoch
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * len(Xb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += len(Xb)

        t_loss /= max(t_total, 1)
        v_loss /= max(v_total, 1)
        t_acc = t_correct / max(t_total, 1)
        v_acc = v_correct / max(v_total, 1)

        scheduler.step()

        writer.add_scalar(f"{model_type}/Loss/train", t_loss, epoch)
        writer.add_scalar(f"{model_type}/Loss/val",   v_loss, epoch)
        writer.add_scalar(f"{model_type}/Acc/train",  t_acc,  epoch)
        writer.add_scalar(f"{model_type}/Acc/val",    v_acc,  epoch)

        logger.info(
            "Epoch {:3d}/{} | train loss={:.4f} acc={:.3f} | val loss={:.4f} acc={:.3f}",
            epoch, max_epochs, t_loss, t_acc, v_loss, v_acc,
        )

        # Early stopping
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "model_type": model_type,
                "epoch": epoch,
                "val_loss": v_loss,
                "val_acc": v_acc,
            }, str(best_ckpt))
            logger.info("  ✓ Saved checkpoint (val_loss={:.4f})", v_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch {}", epoch)
                break

    train_time_s = time.time() - train_start
    writer.close()

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Running test evaluation for {} ...", model_type)
    ckpt = torch.load(str(best_ckpt), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    from training.evaluate import evaluate_model
    metrics = evaluate_model(model, test_dl, device, KNOWN_FORMATIONS, output_dir)
    metrics["model_type"] = model_type
    metrics["n_params"] = n_params
    metrics["train_time_s"] = round(train_time_s, 1)
    metrics["best_val_loss"] = round(best_val_loss, 5)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete for {}. Best model: {}", model_type, best_ckpt)
    return metrics


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train formation classifier")
    parser.add_argument("--model", choices=["mlp", "lstm", "transformer"],
                        required=True, help="Model architecture to train")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(args.model, config)
