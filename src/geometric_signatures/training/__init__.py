"""Training loop and checkpoint management.

Provides config-driven multi-task training for constrained RNNs with:
- Single-seed training with reproducible seeding
- Multi-seed batch execution for statistical analysis
- Checkpoint save/load with optimizer state
- Epoch metrics tracking (loss, accuracy per task)
"""

from __future__ import annotations

from .checkpoints import load_checkpoint, save_checkpoint
from .trainer import (
    EpochMetrics,
    TrainResult,
    train_multi_seed,
    train_single_seed,
)

__all__ = [
    "EpochMetrics",
    "TrainResult",
    "train_single_seed",
    "train_multi_seed",
    "save_checkpoint",
    "load_checkpoint",
]
