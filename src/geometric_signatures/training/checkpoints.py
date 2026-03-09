"""Checkpoint save/load for model state persistence.

Handles saving and loading of model weights, optimizer state, training
epoch, and metrics. Checkpoints are stored as standard PyTorch .pt files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    path: Path,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to checkpoint.
        optimizer: Optimizer with current state.
        epoch: Current epoch number.
        metrics: Dict of metrics to store alongside checkpoint.
        path: Output file path (.pt).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint at epoch %d to %s", epoch, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path (.pt).
        model: Model to load state into.
        optimizer: Optional optimizer to restore state.

    Returns:
        Dict with "epoch" and "metrics" from the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint["epoch"])
    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
    }
