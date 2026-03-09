"""Context-dependent integration task generator.

Implements a two-stimulus, two-context integration task (inspired by Mante et al. 2013).
On each trial, two noisy stimulus channels are presented simultaneously. A context
signal indicates which stimulus to integrate for the decision. The network must
report the sign of the integrated (attended) stimulus.

Input channels: [stimulus_1, stimulus_2, context_signal] + fixation = 4 channels.
Output channels: [decision] = 1 channel (sign of integrated stimulus).

Epoch structure:
    fixation → stimulus_onset → delay → response
"""

from __future__ import annotations

import torch

from .base import TaskBatch, TaskDataset


class ContextDependentIntegration(TaskDataset):
    """Context-dependent sensory integration task."""

    task_name = "context_dependent_integration"
    input_dim = 4   # stim1, stim2, context, fixation
    output_dim = 1   # decision

    def __init__(
        self,
        n_timesteps: int = 100,
        fixation_steps: int = 10,
        stimulus_steps: int = 50,
        delay_steps: int = 20,
        noise_std: float = 0.1,
    ) -> None:
        self._n_timesteps = n_timesteps
        self._fixation_steps = fixation_steps
        self._stimulus_steps = stimulus_steps
        self._delay_steps = delay_steps
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

        # Stimulus coherences: uniform in [-1, 1]
        coh1 = torch.rand(batch_size, generator=rng) * 2 - 1
        coh2 = torch.rand(batch_size, generator=rng) * 2 - 1

        # Context: 0 = attend stim1, 1 = attend stim2
        context = torch.randint(0, 2, (batch_size,), generator=rng)

        # Epoch boundaries
        stim_start = self._fixation_steps
        delay_start = stim_start + self._stimulus_steps
        resp_start = delay_start + self._delay_steps

        # Fixation signal (on during fixation and stimulus, off during response)
        inputs[:, :resp_start, 3] = 1.0

        # Stimulus signals with noise
        noise1 = torch.randn(batch_size, self._stimulus_steps, generator=rng) * self._noise_std
        noise2 = torch.randn(batch_size, self._stimulus_steps, generator=rng) * self._noise_std

        for i in range(batch_size):
            inputs[i, stim_start:delay_start, 0] = coh1[i] + noise1[i]
            inputs[i, stim_start:delay_start, 1] = coh2[i] + noise2[i]
            inputs[i, stim_start:delay_start, 2] = 1.0 if context[i] == 0 else -1.0

            # Target: sign of attended stimulus
            attended = coh1[i] if context[i] == 0 else coh2[i]
            decision = 1.0 if attended > 0 else -1.0
            targets[i, resp_start:, 0] = decision

        # Loss mask: only during response period
        mask[:, resp_start:] = 1.0

        # Outcomes: whether the correct decision is "obvious" (|coherence| > 0.1)
        attended_coh = torch.where(context == 0, coh1, coh2)

        conditions = {
            "coherence_1": coh1,
            "coherence_2": coh2,
            "context": context.float(),
            "attended_coherence": attended_coh,
        }
        epoch_boundaries = {
            "stimulus_onset": torch.full((batch_size,), stim_start, dtype=torch.long),
            "delay": torch.full((batch_size,), delay_start, dtype=torch.long),
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
