"""Unified neural population data contract for RNN and biological sources.

NeuralPopulationData is the central data structure that flows through the entire
pipeline — from RNN state recording and biological data loading, through
preprocessing and analysis, to statistical testing and figure generation.

Every analysis method receives NeuralPopulationData, ensuring a consistent
interface regardless of data source (constrained RNN, IBL Neuropixels, Allen
Brain Observatory calcium imaging).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TrialMetadata:
    """Per-trial metadata beyond task labels.

    Enables condition-specific analysis (e.g., "only correct trials during
    stimulus epoch for context-dependent integration"). Without this, analysis
    methods mix signal with noise from error trials and irrelevant time periods.

    Attributes:
        conditions: Stimulus parameters or context labels per trial.
            Keys are condition names (e.g., "coherence", "context_id"),
            values are arrays of shape (n_trials,).
        outcomes: Per-trial outcome codes. 0=error, 1=correct, -1=miss.
            Shape: (n_trials,).
        epoch_boundaries: Named time boundaries per trial. Keys are epoch
            names (e.g., "stimulus_onset", "delay", "response"), values are
            arrays of shape (n_trials,) giving the time index of that boundary.
    """

    conditions: dict[str, np.ndarray]
    outcomes: np.ndarray
    epoch_boundaries: dict[str, np.ndarray]


@dataclass(frozen=True)
class NeuralPopulationData:
    """Unified contract for neural population activity.

    This frozen dataclass validates shape consistency at construction time
    and provides selection methods for subsetting along trials, time, and units.

    Attributes:
        activity: Neural activity tensor of shape (n_trials, n_timepoints, n_units).
        trial_labels: Task name per trial. Length must equal n_trials.
        time_axis: Timestamps in seconds. Length must equal n_timepoints.
        unit_labels: Neuron/unit identifiers. Length must equal n_units.
        source: Data origin identifier — "rnn", "ibl", or "allen".
        metadata: Source-specific metadata (frozen via MappingProxyType).
        trial_metadata: Optional rich per-trial metadata for condition-specific analysis.
    """

    activity: np.ndarray
    trial_labels: tuple[str, ...]
    time_axis: np.ndarray
    unit_labels: tuple[str, ...]
    source: str
    metadata: Any  # MappingProxyType at runtime; Any for mypy compatibility
    trial_metadata: TrialMetadata | None = None

    def __post_init__(self) -> None:
        """Validate shapes and freeze mutable containers."""
        # Freeze metadata dict → MappingProxyType (immutable view)
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", MappingProxyType(self.metadata))

        # Validate activity is 3D
        if self.activity.ndim != 3:
            raise ValueError(
                f"activity must be 3D (n_trials, n_timepoints, n_units), "
                f"got {self.activity.ndim}D with shape {self.activity.shape}"
            )

        n_trials, n_time, n_units = self.activity.shape

        if len(self.trial_labels) != n_trials:
            raise ValueError(
                f"trial_labels length ({len(self.trial_labels)}) "
                f"!= n_trials ({n_trials})"
            )
        if len(self.time_axis) != n_time:
            raise ValueError(
                f"time_axis length ({len(self.time_axis)}) "
                f"!= n_timepoints ({n_time})"
            )
        if len(self.unit_labels) != n_units:
            raise ValueError(
                f"unit_labels length ({len(self.unit_labels)}) "
                f"!= n_units ({n_units})"
            )

        # Validate source identifier
        valid_sources = ("rnn", "ibl", "allen")
        if self.source not in valid_sources:
            raise ValueError(
                f"source must be one of {valid_sources}, got '{self.source}'"
            )

        # Validate trial_metadata shapes if present
        if self.trial_metadata is not None:
            tm = self.trial_metadata
            if tm.outcomes.shape != (n_trials,):
                raise ValueError(
                    f"trial_metadata.outcomes shape ({tm.outcomes.shape}) "
                    f"!= (n_trials,) = ({n_trials},)"
                )
            for cond_name, cond_arr in tm.conditions.items():
                if cond_arr.shape[0] != n_trials:
                    raise ValueError(
                        f"trial_metadata.conditions['{cond_name}'] "
                        f"first dim ({cond_arr.shape[0]}) != n_trials ({n_trials})"
                    )
            for epoch_name, epoch_arr in tm.epoch_boundaries.items():
                if epoch_arr.shape != (n_trials,):
                    raise ValueError(
                        f"trial_metadata.epoch_boundaries['{epoch_name}'] "
                        f"shape ({epoch_arr.shape}) != (n_trials,) = ({n_trials},)"
                    )

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return int(self.activity.shape[0])

    @property
    def n_timepoints(self) -> int:
        """Number of timepoints per trial."""
        return int(self.activity.shape[1])

    @property
    def n_units(self) -> int:
        """Number of neural units."""
        return int(self.activity.shape[2])

    def select_trials(self, task: str) -> NeuralPopulationData:
        """Return a new NeuralPopulationData with only trials matching the given task."""
        mask = np.array([label == task for label in self.trial_labels])
        return self._subset_trials(mask)

    def select_correct_trials(self) -> NeuralPopulationData:
        """Return a new NeuralPopulationData with only correct trials (outcome == 1).

        Raises:
            ValueError: If trial_metadata is None (no outcome information).
        """
        if self.trial_metadata is None:
            raise ValueError("Cannot select correct trials: trial_metadata is None")
        mask = self.trial_metadata.outcomes == 1
        return self._subset_trials(mask)

    def select_epoch(self, epoch: str) -> NeuralPopulationData:
        """Return a new NeuralPopulationData restricted to a named time epoch.

        Uses epoch_boundaries to determine the time window. Requires at least
        two epoch boundary names to define start and end — the specified epoch
        is the start, and the next epoch (in sorted order) is the end.

        For simplicity, this method takes a start epoch and an end epoch boundary
        from the trial_metadata. Since epoch boundaries are per-trial, this method
        uses the minimum start and maximum end across all trials to define a
        single time window (conservative approach preserving all relevant data).

        Raises:
            ValueError: If trial_metadata is None or epoch not found.
        """
        if self.trial_metadata is None:
            raise ValueError("Cannot select epoch: trial_metadata is None")
        if epoch not in self.trial_metadata.epoch_boundaries:
            available = list(self.trial_metadata.epoch_boundaries.keys())
            raise ValueError(
                f"Epoch '{epoch}' not found. Available: {available}"
            )

        # Use the epoch boundary as start index (minimum across trials)
        start_indices = self.trial_metadata.epoch_boundaries[epoch]
        start_idx = int(np.min(start_indices))

        # Find the next epoch boundary (sorted by mean index) as end
        sorted_epochs = sorted(
            self.trial_metadata.epoch_boundaries.keys(),
            key=lambda e: float(np.mean(self.trial_metadata.epoch_boundaries[e])),  # type: ignore[union-attr]
        )
        epoch_pos = sorted_epochs.index(epoch)
        if epoch_pos + 1 < len(sorted_epochs):
            next_epoch = sorted_epochs[epoch_pos + 1]
            end_indices = self.trial_metadata.epoch_boundaries[next_epoch]
            end_idx = int(np.max(end_indices))
        else:
            # Last epoch: go to end of time axis
            end_idx = self.n_timepoints

        # Clamp
        start_idx = max(0, start_idx)
        end_idx = min(self.n_timepoints, end_idx)

        sliced_activity = self.activity[:, start_idx:end_idx, :]
        sliced_time = self.time_axis[start_idx:end_idx]

        # Update epoch boundaries relative to the new time window
        new_epoch_boundaries: dict[str, np.ndarray] = {}
        if self.trial_metadata is not None:
            for ep_name, ep_arr in self.trial_metadata.epoch_boundaries.items():
                new_epoch_boundaries[ep_name] = np.clip(
                    ep_arr - start_idx, 0, end_idx - start_idx
                )

        new_trial_metadata: TrialMetadata | None = None
        if self.trial_metadata is not None:
            new_trial_metadata = TrialMetadata(
                conditions=self.trial_metadata.conditions,
                outcomes=self.trial_metadata.outcomes,
                epoch_boundaries=new_epoch_boundaries,
            )

        return NeuralPopulationData(
            activity=sliced_activity,
            trial_labels=self.trial_labels,
            time_axis=sliced_time,
            unit_labels=self.unit_labels,
            source=self.source,
            metadata=dict(self.metadata),  # unfreeze for reconstruction
            trial_metadata=new_trial_metadata,
        )

    def select_units(self, indices: Sequence[int]) -> NeuralPopulationData:
        """Return a new NeuralPopulationData with only the specified units.

        Args:
            indices: Unit indices to keep.
        """
        idx = list(indices)
        sliced_activity = self.activity[:, :, idx]
        sliced_labels = tuple(self.unit_labels[i] for i in idx)

        return NeuralPopulationData(
            activity=sliced_activity,
            trial_labels=self.trial_labels,
            time_axis=self.time_axis.copy(),
            unit_labels=sliced_labels,
            source=self.source,
            metadata=dict(self.metadata),  # unfreeze for reconstruction
            trial_metadata=self.trial_metadata,
        )

    def _subset_trials(self, mask: np.ndarray) -> NeuralPopulationData:
        """Internal helper: subset trials by boolean mask."""
        indices = np.where(mask)[0]
        sliced_activity = self.activity[indices, :, :]
        sliced_labels = tuple(self.trial_labels[i] for i in indices)

        new_trial_metadata: TrialMetadata | None = None
        if self.trial_metadata is not None:
            new_conditions = {
                k: v[indices] for k, v in self.trial_metadata.conditions.items()
            }
            new_trial_metadata = TrialMetadata(
                conditions=new_conditions,
                outcomes=self.trial_metadata.outcomes[indices],
                epoch_boundaries={
                    k: v[indices]
                    for k, v in self.trial_metadata.epoch_boundaries.items()
                },
            )

        return NeuralPopulationData(
            activity=sliced_activity,
            trial_labels=sliced_labels,
            time_axis=self.time_axis.copy(),
            unit_labels=self.unit_labels,
            source=self.source,
            metadata=dict(self.metadata),  # unfreeze for reconstruction
            trial_metadata=new_trial_metadata,
        )
