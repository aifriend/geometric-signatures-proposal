"""Perceptual discrimination task generator.

Implements a simple go/no-go perceptual discrimination task where the
network must categorize a noisy stimulus into one of two categories based
on its magnitude relative to a threshold.

Input channels: [stimulus, fixation] = 2 channels.
Output channels: [category] = 1 channel (+1 above threshold, -1 below).

Epoch structure:
    fixation → stimulus_onset → response
"""

from __future__ import annotations

import torch

from .base import TaskBatch, TaskDataset


class PerceptualDiscrimination(TaskDataset):
    """Perceptual discrimination (categorization) task."""

    task_name = "perceptual_discrimination"
    input_dim = 2   # stimulus, fixation
    output_dim = 1   # category

    def __init__(
        self,
        n_timesteps: int = 80,
        fixation_steps: int = 10,
        stimulus_steps: int = 40,
        noise_std: float = 0.2,
        threshold: float = 0.0,
    ) -> None:
        self._n_timesteps = n_timesteps
        self._fixation_steps = fixation_steps
        self._stimulus_steps = stimulus_steps
        self._noise_std = noise_std
        self._threshold = threshold

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

        # Stimulus magnitude: uniform in [-1, 1]
        stimulus_value = torch.rand(batch_size, generator=rng) * 2 - 1

        stim_start = self._fixation_steps
        resp_start = stim_start + self._stimulus_steps

        # Fixation on until response
        inputs[:, :resp_start, 1] = 1.0

        for i in range(batch_size):
            noise = torch.randn(self._stimulus_steps, generator=rng) * self._noise_std
            inputs[i, stim_start:resp_start, 0] = stimulus_value[i] + noise

            # Category: above or below threshold
            category = 1.0 if stimulus_value[i] > self._threshold else -1.0
            targets[i, resp_start:, 0] = category

        # Loss mask: only during response
        mask[:, resp_start:] = 1.0

        conditions = {
            "stimulus_value": stimulus_value,
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
