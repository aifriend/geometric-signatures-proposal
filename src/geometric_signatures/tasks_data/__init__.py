"""Synthetic task data generators with registry.

Provides four cognitive task generators matching the proposal's task battery:

- ``ContextDependentIntegration``: Two-stimulus, two-context integration.
- ``EvidenceAccumulation``: Random dot motion discrimination.
- ``WorkingMemory``: Delayed match-to-sample.
- ``PerceptualDiscrimination``: Go/no-go categorization.

All generators produce ``TaskBatch`` instances with deterministic generation
via ``torch.Generator``, including trial metadata (conditions, epoch
boundaries) that flow through to ``NeuralPopulationData``.

Usage::

    from geometric_signatures.tasks_data import get_task, TASK_REGISTRY

    TaskClass = get_task("evidence_accumulation")
    generator = TaskClass()
    rng = torch.Generator().manual_seed(42)
    batch = generator.generate_batch(batch_size=32, rng=rng)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import TaskDataset

from .base import TaskBatch
from .context_dependent_integration import ContextDependentIntegration
from .evidence_accumulation import EvidenceAccumulation
from .perceptual_discrimination import PerceptualDiscrimination
from .working_memory import WorkingMemory

TASK_REGISTRY: dict[str, type[TaskDataset]] = {
    "context_dependent_integration": ContextDependentIntegration,
    "evidence_accumulation": EvidenceAccumulation,
    "working_memory": WorkingMemory,
    "perceptual_discrimination": PerceptualDiscrimination,
}


def get_task(name: str) -> type[TaskDataset]:
    """Look up a task generator class by name.

    Args:
        name: Task identifier (must match a key in ``TASK_REGISTRY``).

    Returns:
        The task generator class.

    Raises:
        ValueError: If the task name is not in the registry.
    """
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: '{name}'. Available: {sorted(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]


__all__ = [
    "TaskBatch",
    "TaskDataset",
    "TASK_REGISTRY",
    "get_task",
    "ContextDependentIntegration",
    "EvidenceAccumulation",
    "WorkingMemory",
    "PerceptualDiscrimination",
]
