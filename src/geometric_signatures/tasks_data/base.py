"""Base classes for synthetic task data generation.

Defines the ``TaskDataset`` abstract base class and ``TaskBatch`` frozen
dataclass — the contract that all task generators must satisfy.

Each task generator produces batches of (inputs, targets, mask) tensors
along with trial metadata (conditions, epoch boundaries) that flow
through to ``NeuralPopulationData`` for analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TaskBatch:
    """A batch of task trials with associated metadata.

    Attributes:
        inputs: Input stimulus tensor, shape (batch, time, input_dim).
        targets: Target output tensor, shape (batch, time, output_dim).
        mask: Loss mask tensor, shape (batch, time). 1.0 where loss should
            be computed, 0.0 otherwise (e.g., during fixation periods).
        task_name: Name of the task that generated this batch.
        conditions: Per-trial stimulus parameters. Keys are condition names,
            values are tensors of shape (batch,) or (batch, ...).
        epoch_boundaries: Per-trial epoch onset indices. Keys are epoch names
            (e.g., "stimulus_onset", "delay", "response"), values are int
            tensors of shape (batch,).
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    task_name: str
    conditions: dict[str, torch.Tensor]
    epoch_boundaries: dict[str, torch.Tensor]


class TaskDataset(ABC):
    """Abstract base class for synthetic task generators.

    Each subclass must define:
        - ``task_name``: Identifier matching ``REQUIRED_TASKS``.
        - ``input_dim``: Number of input channels.
        - ``output_dim``: Number of output channels.
        - ``generate_batch()``: Produce a ``TaskBatch`` deterministically.

    The ``rng`` parameter (``torch.Generator``) ensures deterministic generation
    when combined with ``set_all_seeds()``.
    """

    task_name: str
    input_dim: int
    output_dim: int

    @abstractmethod
    def generate_batch(
        self,
        batch_size: int,
        rng: torch.Generator,
    ) -> TaskBatch:
        """Generate a batch of task trials.

        Args:
            batch_size: Number of trials in the batch.
            rng: PyTorch random generator for deterministic sampling.

        Returns:
            TaskBatch with inputs, targets, mask, and trial metadata.
        """
        ...

    @property
    def n_timepoints(self) -> int:
        """Number of timesteps per trial (fixed for each task)."""
        raise NotImplementedError("Subclasses should define n_timepoints")
