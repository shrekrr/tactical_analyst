"""
Evaluation script for the Temporal Formation Transformer.

Outputs:
- Confusion matrix (saved as PNG)
- Per-class Precision, Recall, F1
- Macro-averaged metrics
- JSON metrics file
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
from loguru import logger
import json


def evaluate_model(
    model: nn.Module,
    test_dl,
    device: str,
    class_names: List[str],
    output_dir: Path,
) -> dict:
    """
    Run evaluation on the test set.

    Parameters
    ----------
    model : nn.Module
        Trained model (already loaded with best weights).
    test_dl : DataLoader
        Test dataloader.
    device : str
    class_names : List[str]
        Formation labels in class index order.
    output_dir : Path
        Where to save confusion matrix and metrics.

    Returns
    -------
    metrics : dict
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb = Xb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy = float((all_preds == all_labels).mean())
    macro_p = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names[:len(set(all_labels))],
        zero_division=0,
    )

    logger.info("Test Accuracy: {:.4f}", accuracy)
    logger.info("Macro Precision: {:.4f}", macro_p)
    logger.info("Macro Recall: {:.4f}", macro_r)
    logger.info("Macro F1: {:.4f}", macro_f1)
    logger.info("\n{}", report)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    n_classes = len(class_names)
    used_labels = sorted(set(all_labels.tolist()))
    used_names = [class_names[i] for i in used_labels if i < len(class_names)]
    cm = confusion_matrix(all_labels, all_preds, labels=used_labels)
    _plot_confusion_matrix(cm, used_names, output_dir / "confusion_matrix.png")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "n_test_samples": len(all_labels),
        "note": (
            "Metrics are on held-out test data. "
            "If synthetic data was used, results are not meaningful."
        ),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Formation Classifier — Confusion Matrix")
    plt.colorbar(im, ax=ax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to {}", save_path)
