"""Working memory (delayed match-to-sample) task generator.

Implements a delayed match-to-sample task where the network must remember
a stimulus presented during a sample period, maintain it through a delay,
and compare it to a test stimulus. Inspired by Romo et al. (1999).

Input channels: [stimulus, fixation] = 2 channels.
Output channels: [match_decision] = 1 channel (+1 match, -1 non-match).

Epoch structure:
    fixation → stimulus_onset (sample) → delay → test_onset → response
"""

from __future__ import annotations

import torch

from .base import TaskBatch, TaskDataset


class WorkingMemory(TaskDataset):
    """Delayed match-to-sample working memory task."""

    task_name = "working_memory"
    input_dim = 2   # stimulus, fixation
    output_dim = 1   # match decision

    def __init__(
        self,
        n_timesteps: int = 120,
        fixation_steps: int = 10,
        sample_steps: int = 20,
        delay_steps: int = 40,
        test_steps: int = 20,
        noise_std: float = 0.1,
    ) -> None:
        self._n_timesteps = n_timesteps
        self._fixation_steps = fixation_steps
        self._sample_steps = sample_steps
        self._delay_steps = delay_steps
        self._test_steps = test_steps
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

        # Sample stimulus value: uniform in [0.2, 1.0] with random sign
        sample_mag = torch.rand(batch_size, generator=rng) * 0.8 + 0.2
        sample_sign = (torch.randint(0, 2, (batch_size,), generator=rng) * 2 - 1).float()
        sample_val = sample_mag * sample_sign

        # Match probability: 50%
        is_match = torch.randint(0, 2, (batch_size,), generator=rng)

        # Test stimulus: same as sample (match) or different
        test_val = torch.zeros(batch_size)
        for i in range(batch_size):
            if is_match[i] == 1:
                test_val[i] = sample_val[i]
            else:
                # Different value: flip sign
                test_val[i] = -sample_val[i]

        # Epoch boundaries
        sample_start = self._fixation_steps
        delay_start = sample_start + self._sample_steps
        test_start = delay_start + self._delay_steps
        resp_start = test_start + self._test_steps

        # Fixation signal: on until response
        inputs[:, :resp_start, 1] = 1.0

        for i in range(batch_size):
            # Sample period
            noise_s = torch.randn(self._sample_steps, generator=rng) * self._noise_std
            inputs[i, sample_start:delay_start, 0] = sample_val[i] + noise_s

            # Test period
            noise_t = torch.randn(self._test_steps, generator=rng) * self._noise_std
            inputs[i, test_start:resp_start, 0] = test_val[i] + noise_t

            # Target: +1 match, -1 non-match
            decision = 1.0 if is_match[i] == 1 else -1.0
            targets[i, resp_start:, 0] = decision

        # Loss mask: only during response
        mask[:, resp_start:] = 1.0

        conditions = {
            "sample_value": sample_val,
            "test_value": test_val,
            "is_match": is_match.float(),
        }
        epoch_boundaries = {
            "stimulus_onset": torch.full((batch_size,), sample_start, dtype=torch.long),
            "delay": torch.full((batch_size,), delay_start, dtype=torch.long),
            "test_onset": torch.full((batch_size,), test_start, dtype=torch.long),
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
