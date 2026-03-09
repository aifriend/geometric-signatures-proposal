"""Config-driven multi-task training for constrained RNNs.

Provides:
- ``train_single_seed``: Train one variant with one seed (fully reproducible).
- ``train_multi_seed``: Train all seeds for one variant.
- ``EpochMetrics``, ``TrainResult``: Frozen result dataclasses.

Training strategy:
- Interleaved multi-task batching (random task per step).
- Configurable ``steps_per_epoch`` (default 50 — **not** derived from batch_size).
- Train/validation split via separate RNG streams.
- MSE loss on masked response period.
- Accuracy: fraction of trials where output sign matches target sign.
- LR scheduling: cosine annealing or reduce-on-plateau.
- Early stopping based on validation loss (patience configurable).
- Best-model checkpointing (saves lowest val_loss checkpoint).
- Gradient norm tracking for stability diagnostics.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..config import ExperimentConfig
from ..models.constrained_rnn import ConstrainedRNN
from ..motifs import MotifSwitches
from ..population import NeuralPopulationData
from ..reproducibility import capture_environment, resolve_device, set_all_seeds
from ..tasks_data import get_task
from ..tasks_data.base import TaskBatch
from ..tracking import (
    ExperimentCatalog,
    RunRecord,
    dataclass_payload,
    stable_config_hash,
    write_run_manifest,
)

from .checkpoints import save_checkpoint


def _pad_to_model(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_input_dim: int,
    model_output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad inputs/targets when task dims < model dims.

    Multi-task training uses a single model whose ``input_dim`` equals
    the **maximum** across all tasks.  Tasks with fewer channels are
    right-padded with zeros so the projection layer sees the correct
    width.  The same applies to targets/``output_dim``.
    """
    if inputs.shape[-1] < model_input_dim:
        pad_size = model_input_dim - inputs.shape[-1]
        inputs = torch.nn.functional.pad(inputs, (0, pad_size))
    if targets.shape[-1] < model_output_dim:
        pad_size = model_output_dim - targets.shape[-1]
        targets = torch.nn.functional.pad(targets, (0, pad_size))
    return inputs, targets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics for a single training epoch.

    Attributes:
        epoch: Epoch number (0-indexed).
        loss: Total training loss averaged across steps.
        task_losses: Per-task average training loss.
        task_accuracies: Per-task training accuracy (sign match).
        val_loss: Validation loss.
        val_task_accuracies: Per-task validation accuracy.
        grad_norm: Average gradient norm (pre-clip) across steps.
        lr: Learning rate at the end of this epoch.
    """

    epoch: int
    loss: float
    task_losses: dict[str, float]
    task_accuracies: dict[str, float]
    val_loss: float
    val_task_accuracies: dict[str, float]
    grad_norm: float = 0.0
    lr: float = 0.0


@dataclass(frozen=True)
class TrainResult:
    """Result of training a single model (one seed, one variant).

    Attributes:
        metrics: Tuple of per-epoch metrics.
        config_hash: Deterministic hash of the experiment config.
        seed: Random seed used for this run.
        checkpoint_path: Path to final model checkpoint.
        population_data: Neural population data from recorded states.
    """

    metrics: tuple[EpochMetrics, ...]
    config_hash: str
    seed: int
    checkpoint_path: str
    population_data: NeuralPopulationData


def _compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute sign-match accuracy in the response period.

    Args:
        outputs: Model outputs, shape (batch, time, output_dim).
        targets: Target outputs, shape (batch, time, output_dim).
        mask: Loss mask, shape (batch, time).

    Returns:
        Fraction of response-period timesteps with correct sign.
    """
    # Expand mask to match output dims
    mask_expanded = mask.unsqueeze(-1).expand_as(outputs)
    # Only count where mask is active
    pred_sign = (outputs > 0).float()
    tgt_sign = (targets > 0).float()
    correct = ((pred_sign == tgt_sign) & (mask_expanded > 0)).float()
    total = mask_expanded.sum()
    if total == 0:
        return 0.0
    return float((correct.sum() / total).item())


def _train_one_epoch(
    model: ConstrainedRNN,
    optimizer: torch.optim.Optimizer,
    task_generators: dict[str, Any],
    task_names: list[str],
    batch_size: int,
    steps_per_epoch: int,
    train_rng: torch.Generator,
    device: str,
) -> tuple[float, dict[str, float], dict[str, float], float]:
    """Run one training epoch with interleaved multi-task batching.

    Returns:
        Tuple of (avg_loss, task_losses, task_accuracies, avg_grad_norm).
    """
    model.train()
    total_loss = 0.0
    task_loss_accum: dict[str, float] = {t: 0.0 for t in task_names}
    task_acc_accum: dict[str, float] = {t: 0.0 for t in task_names}
    task_count: dict[str, int] = {t: 0 for t in task_names}

    criterion = nn.MSELoss(reduction="none")

    grad_norm_accum = 0.0
    for _step in range(steps_per_epoch):
        # Random task selection
        task_name = random.choice(task_names)
        gen = task_generators[task_name]
        batch = gen.generate_batch(batch_size, train_rng)

        inputs = batch.inputs.to(device)
        targets = batch.targets.to(device)
        mask = batch.mask.to(device)

        # Zero-pad when task dims < model dims (multi-task)
        model_in = model.config.input_dim
        model_out = model.config.output_dim
        inputs, targets = _pad_to_model(inputs, targets, model_in, model_out)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Masked MSE loss
        loss_per_elem = criterion(outputs, targets)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss_per_elem)
        loss = (loss_per_elem * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)

        loss.backward()
        # Gradient clipping for stability — returns pre-clip norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_accum += float(grad_norm)
        optimizer.step()

        step_loss = loss.item()
        total_loss += step_loss
        task_loss_accum[task_name] += step_loss
        task_count[task_name] += 1

        with torch.no_grad():
            acc = _compute_accuracy(outputs, targets, mask)
            task_acc_accum[task_name] += acc

    avg_loss = total_loss / steps_per_epoch
    avg_grad_norm = grad_norm_accum / steps_per_epoch

    task_losses = {}
    task_accuracies = {}
    for t in task_names:
        if task_count[t] > 0:
            task_losses[t] = task_loss_accum[t] / task_count[t]
            task_accuracies[t] = task_acc_accum[t] / task_count[t]
        else:
            task_losses[t] = 0.0
            task_accuracies[t] = 0.0

    return avg_loss, task_losses, task_accuracies, avg_grad_norm


def _validate(
    model: ConstrainedRNN,
    task_generators: dict[str, Any],
    task_names: list[str],
    batch_size: int,
    val_rng: torch.Generator,
    device: str,
) -> tuple[float, dict[str, float]]:
    """Run validation on all tasks.

    Returns:
        Tuple of (avg_val_loss, task_val_accuracies).
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    total_loss = 0.0
    task_accs: dict[str, float] = {}

    with torch.no_grad():
        for task_name in task_names:
            gen = task_generators[task_name]
            batch = gen.generate_batch(batch_size, val_rng)

            inputs = batch.inputs.to(device)
            targets = batch.targets.to(device)
            mask = batch.mask.to(device)

            # Zero-pad when task dims < model dims (multi-task)
            model_in = model.config.input_dim
            model_out = model.config.output_dim
            inputs, targets = _pad_to_model(inputs, targets, model_in, model_out)

            outputs = model(inputs)
            loss_per_elem = criterion(outputs, targets)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss_per_elem)
            loss = (loss_per_elem * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)

            total_loss += loss.item()
            task_accs[task_name] = _compute_accuracy(outputs, targets, mask)

    avg_loss = total_loss / len(task_names)
    return avg_loss, task_accs


def train_single_seed(
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    device: str = "cpu",
    catalog: ExperimentCatalog | None = None,
    progress_callback: Callable[[EpochMetrics], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> TrainResult:
    """Train one variant with one seed.

    Fully reproducible: calls ``set_all_seeds(seed)`` and uses separate
    ``torch.Generator`` instances for training and validation data.

    Args:
        config: Full experiment configuration.
        seed: Random seed for this run.
        output_dir: Directory for checkpoints and manifests.
        device: Device to train on — "auto", "cpu", "cuda", "cuda:N",
            or "mps" (Apple Silicon). Default: "cpu".
        catalog: Optional experiment catalog to register the run.
        progress_callback: Optional callback invoked after each epoch with
            the epoch's ``EpochMetrics``. Exceptions are caught and logged.
        cancel_event: Optional ``threading.Event`` checked at the start of
            each epoch. If set, training stops early and returns a partial
            ``TrainResult`` with only the completed epochs.

    Returns:
        TrainResult with metrics, checkpoint path, and population data.
    """
    device = resolve_device(device)
    set_all_seeds(seed)

    # Config hash for tracking
    config_payload = dataclass_payload(config)
    config_hash = stable_config_hash(config_payload)
    variant_name = _variant_name(config.motifs)

    logger.info(
        "Training seed=%d variant=%s hash=%s",
        seed, variant_name, config_hash[:8],
    )

    # Register run as "running"
    if catalog is not None:
        env = capture_environment()
        import datetime

        record = RunRecord(
            config_hash=config_hash,
            variant_name=variant_name,
            seed=seed,
            timestamp=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            manifest_path=str(output_dir / "manifest.json"),
            status="running",
            environment=env,
        )
        catalog.register_run(record)

    # Build model
    assert config.model is not None, "ModelConfig required for training"
    model = ConstrainedRNN(config.model, config.motifs, constraint_seed=seed)
    model.to(device)

    # Build optimizer
    lr = config.training.lr
    if config.training.optimizer == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Build LR scheduler
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    sched_name = config.training.lr_scheduler
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs, eta_min=lr * 0.01,
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )

    # Build task generators
    task_generators = {
        name: get_task(name)() for name in config.tasks
    }
    task_names = list(config.tasks)

    # Separate RNG streams for train and validation
    train_rng = torch.Generator().manual_seed(seed)
    val_rng = torch.Generator().manual_seed(seed + 10000)

    batch_size = config.training.batch_size
    steps_per_epoch = config.training.steps_per_epoch

    # Early stopping state
    patience = config.training.patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Training loop
    all_metrics: list[EpochMetrics] = []
    for epoch in range(config.training.epochs):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Training cancelled at epoch %d", epoch)
            break

        train_loss, task_losses, task_accs, grad_norm = _train_one_epoch(
            model, optimizer, task_generators, task_names,
            batch_size, steps_per_epoch, train_rng, device,
        )
        val_loss, val_accs = _validate(
            model, task_generators, task_names, batch_size, val_rng, device,
        )

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = EpochMetrics(
            epoch=epoch,
            loss=train_loss,
            task_losses=task_losses,
            task_accuracies=task_accs,
            val_loss=val_loss,
            val_task_accuracies=val_accs,
            grad_norm=grad_norm,
            lr=current_lr,
        )
        all_metrics.append(metrics)

        if progress_callback is not None:
            try:
                progress_callback(metrics)
            except Exception:
                logger.warning("progress_callback raised; ignoring", exc_info=True)

        # Step LR scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Logging — every 10 epochs or first/last
        log_interval = max(1, config.training.epochs // 10)
        if epoch % log_interval == 0 or epoch == config.training.epochs - 1:
            avg_train_acc = (
                sum(task_accs.values()) / len(task_accs) if task_accs else 0.0
            )
            avg_val_acc = (
                sum(val_accs.values()) / len(val_accs) if val_accs else 0.0
            )
            logger.info(
                "Epoch %d/%d — loss=%.4f val_loss=%.4f "
                "train_acc=%.3f val_acc=%.3f grad_norm=%.3f lr=%.2e",
                epoch + 1, config.training.epochs,
                train_loss, val_loss,
                avg_train_acc, avg_val_acc,
                grad_norm, current_lr,
            )

        # Best-model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best checkpoint
            run_dir = output_dir / variant_name / f"seed_{seed}"
            best_ckpt_path = run_dir / "checkpoint_best.pt"
            save_checkpoint(
                model, optimizer, epoch,
                {"best_val_loss": best_val_loss},
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d, best_val_loss=%.4f)",
                epoch + 1, patience, best_val_loss,
            )
            break

    # Save final checkpoint
    run_dir = output_dir / variant_name / f"seed_{seed}"
    checkpoint_path = run_dir / "checkpoint.pt"
    last_epoch = all_metrics[-1].epoch if all_metrics else -1
    save_checkpoint(
        model, optimizer, last_epoch,
        {"final_val_loss": all_metrics[-1].val_loss if all_metrics else float("nan")},
        checkpoint_path,
    )

    # Record states for analysis
    record_rng = torch.Generator().manual_seed(seed + 20000)
    record_batches_raw = {
        name: gen.generate_batch(batch_size, record_rng)
        for name, gen in task_generators.items()
    }
    # Pad inputs to model dimensions and move to device
    model_in = config.model.input_dim
    model_out = config.model.output_dim
    record_batches = {}
    for name, batch in record_batches_raw.items():
        padded_inputs, padded_targets = _pad_to_model(
            batch.inputs, batch.targets, model_in, model_out,
        )
        record_batches[name] = TaskBatch(
            inputs=padded_inputs.to(device),
            targets=padded_targets.to(device),
            mask=batch.mask.to(device),
            task_name=name,
            conditions=batch.conditions,
            epoch_boundaries=batch.epoch_boundaries,
        )
    population_data = model.record_states(record_batches)

    # Write manifest
    final_m = all_metrics[-1] if all_metrics else None
    final_avg_acc = (
        sum(final_m.val_task_accuracies.values()) / len(final_m.val_task_accuracies)
        if final_m and final_m.val_task_accuracies else float("nan")
    )
    manifest_payload = {
        **config_payload,
        "seed": seed,
        "variant": variant_name,
        "final_loss": final_m.loss if final_m else float("nan"),
        "final_val_loss": final_m.val_loss if final_m else float("nan"),
        "best_val_loss": best_val_loss,
        "final_val_accuracy": final_avg_acc,
        "epochs_completed": len(all_metrics),
        "early_stopped": patience > 0 and epochs_without_improvement >= patience,
        "steps_per_epoch": steps_per_epoch,
    }
    manifest_path = run_dir / "manifest.json"
    write_run_manifest(manifest_path, manifest_payload)

    # Update catalog
    if catalog is not None:
        catalog.update_status(config_hash, variant_name, seed, "completed")

    result = TrainResult(
        metrics=tuple(all_metrics),
        config_hash=config_hash,
        seed=seed,
        checkpoint_path=str(checkpoint_path),
        population_data=population_data,
    )

    logger.info("Completed seed=%d variant=%s", seed, variant_name)
    return result


def train_multi_seed(
    config: ExperimentConfig,
    output_dir: Path,
    device: str = "cpu",
    catalog: ExperimentCatalog | None = None,
) -> list[TrainResult]:
    """Train all seeds for one variant.

    Reads ``n_seeds`` from ``config.seeds`` (falls back to single seed
    from ``config.experiment.seed`` if seeds config is None).

    Args:
        config: Full experiment configuration.
        output_dir: Base output directory.
        device: Device to train on.
        catalog: Optional experiment catalog.

    Returns:
        List of TrainResult, one per seed.
    """
    if config.seeds is not None:
        seeds = list(config.seeds.seed_list())
    else:
        seeds = [config.experiment.seed]

    results: list[TrainResult] = []
    for seed in seeds:
        try:
            result = train_single_seed(config, seed, output_dir, device, catalog)
            results.append(result)
        except Exception as e:
            logger.error("Failed seed=%d: %s", seed, e)
            # Register failure in catalog
            if catalog is not None:
                config_payload = dataclass_payload(config)
                config_hash = stable_config_hash(config_payload)
                variant_name = _variant_name(config.motifs)
                catalog.update_status(config_hash, variant_name, seed, "failed")
            continue

    return results


def _variant_name(motifs: MotifSwitches) -> str:
    """Derive variant name from motif switches."""
    if all([
        motifs.normalization_gain_modulation,
        motifs.attractor_dynamics,
        motifs.selective_gating,
        motifs.expansion_recoding,
    ]):
        return "complete"

    ablated = []
    if not motifs.normalization_gain_modulation:
        ablated.append("normalization")
    if not motifs.attractor_dynamics:
        ablated.append("attractor")
    if not motifs.selective_gating:
        ablated.append("gating")
    if not motifs.expansion_recoding:
        ablated.append("expansion")

    return "ablate_" + "_".join(ablated)
