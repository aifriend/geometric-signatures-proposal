"""Evidence accumulation (random dot motion) task generator.

Implements a classical dot-motion discrimination task where the network must
integrate noisy evidence over time to determine the direction of motion.
Inspired by Shadlen & Newsome (1996), Gold & Shadlen (2007).

Input channels: [evidence, fixation] = 2 channels.
Output channels: [decision] = 1 channel (direction sign).

Epoch structure:
    fixation → stimulus_onset → response
"""

from __future__ import annotations

import torch

from .base import TaskBatch, TaskDataset


class EvidenceAccumulation(TaskDataset):
    """Random dot motion evidence accumulation task."""

    task_name = "evidence_accumulation"
    input_dim = 2   # evidence, fixation
    output_dim = 1   # decision

    def __init__(
        self,
        n_timesteps: int = 100,
        fixation_steps: int = 10,
        stimulus_steps: int = 70,
        noise_std: float = 1.0,
    ) -> None:
        self._n_timesteps = n_timesteps
        self._fixation_steps = fixation_steps
        self._stimulus_steps = stimulus_steps
        self._noise_std = noise_std

    @property
    def n_timepoints(self) -> int:
        return self._n_timesteps

    def generate_batch(
        self,
        batch_size: int,
        rng: torch.Generator,
    ) -> TaskBatch:
        n_t = self._n_timesteps
        inputs = torch.zeros(batch_size, n_t, self.input_dim)
        targets = torch.zeros(batch_size, n_t, self.output_dim)
        mask = torch.zeros(batch_size, n_t)

        # Motion coherence: uniform in [-1, 1]
        coherence = torch.rand(batch_size, generator=rng) * 2 - 1

        stim_start = self._fixation_steps
        resp_start = stim_start + self._stimulus_steps

        # Fixation on until response
        inputs[:, :resp_start, 1] = 1.0

        # Evidence: coherence + noise at each timestep
        for i in range(batch_size):
            noise = torch.randn(self._stimulus_steps, generator=rng) * self._noise_std
            inputs[i, stim_start:resp_start, 0] = coherence[i] + noise

            # Target: sign of coherence
            decision = 1.0 if coherence[i] > 0 else -1.0
            targets[i, resp_start:, 0] = decision

        # Loss mask: only during response
        mask[:, resp_start:] = 1.0

        conditions = {
            "coherence": coherence,
        }
        epoch_boundaries = {
            "stimulus_onset": torch.full((batch_size,), stim_start, dtype=torch.long),
            "response": torch.full((batch_size,), resp_start, dtype=torch.long),
        }

        return TaskBatch(
            inputs=inputs,
            targets=targets,
            mask=mask,
            task_name=self.task_name,
            conditions=conditions,
            epoch_boundaries=epoch_boundaries,
        )
