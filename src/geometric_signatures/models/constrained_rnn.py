"""Constrained RNN model with motif-specific layers and state recording.

The ``ConstrainedRNN`` is the core model for Aim 1. It implements a
vanilla RNN cell augmented with four computational motif layers that
can be independently toggled via ``MotifSwitches``. The model supports:

- Forward pass for training (returns outputs only).
- State recording for analysis (returns ``NeuralPopulationData`` with
  full trial metadata extracted from ``TaskBatch``).
- Biological constraints (Dale's law, sparse connectivity).

Architecture::

    input → [input_proj] → h_raw
    h_raw + [recurrent] → h_recurrent
    h_recurrent → [normalization] → [attractor] → [gating] → [expansion] → h_new
    h_new → [readout] → output

Each motif layer is identity when disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from ..motifs import MotifSwitches

from .constraints import (
    apply_sparse_mask,
    create_excitatory_mask,
    create_sparse_mask,
    enforce_dale_law,
)
from .layers import (
    AttractorRecurrence,
    DivisiveNormalization,
    ExpansionRecoding,
    SelectiveGating,
)

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ..population import NeuralPopulationData
    from ..tasks_data.base import TaskBatch


class ConstrainedRNN(nn.Module):
    """Constrained recurrent neural network with toggleable motif layers.

    Args:
        config: Model configuration (hidden_size, num_layers, etc.).
        motifs: Which computational motifs are active.
        constraint_seed: Seed for generating fixed constraint masks
            (Dale's law neuron assignments, sparsity masks).
    """

    def __init__(
        self,
        config: ModelConfig,
        motifs: MotifSwitches,
        constraint_seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.motifs = motifs
        self.hidden_size = config.hidden_size

        # Time constant for continuous-time dynamics (standard in comp neuro RNNs)
        # h_new = (1 - alpha) * h_old + alpha * activation(...)
        # alpha = dt / tau;  dt=1, tau=5 → alpha=0.2
        self._alpha = 0.2

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)

        # Recurrent weight (vanilla RNN style)
        self.W_rec = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Recurrent bias
        self.h_bias = nn.Parameter(torch.zeros(config.hidden_size))

        # Readout layer — small init so output starts near zero
        self.readout = nn.Linear(config.hidden_size, config.output_dim)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.1)
        nn.init.zeros_(self.readout.bias)

        # --- Motif layers (identity when disabled) ---
        self.normalization: nn.Module
        self.attractor: nn.Module
        self.gating: nn.Module
        self.expansion: nn.Module

        if motifs.normalization_gain_modulation:
            self.normalization = DivisiveNormalization(config.hidden_size)
        else:
            self.normalization = nn.Identity()

        if motifs.attractor_dynamics:
            self.attractor = AttractorRecurrence(config.hidden_size)
        else:
            self.attractor = nn.Identity()

        if motifs.selective_gating:
            self.gating = SelectiveGating(config.hidden_size)
        else:
            self.gating = nn.Identity()

        if motifs.expansion_recoding:
            self.expansion = ExpansionRecoding(config.hidden_size)
        else:
            self.expansion = nn.Identity()

        # --- Biological constraints ---
        rng = torch.Generator().manual_seed(constraint_seed)

        # Dale's law
        self._dale_law = config.dale_law
        if config.dale_law:
            exc_mask = create_excitatory_mask(
                config.hidden_size, excitatory_fraction=0.8, rng=rng
            )
            self.register_buffer("excitatory_mask", exc_mask)
        else:
            self.register_buffer("excitatory_mask", None)

        # Sparse connectivity
        self._sparse = config.sparse_connectivity > 0.0
        if self._sparse:
            sparse_mask = create_sparse_mask(
                config.hidden_size, config.sparse_connectivity, rng=rng
            )
            self.register_buffer("sparse_mask", sparse_mask)
        else:
            self.register_buffer("sparse_mask", None)

        # Activation function
        self._activation = torch.tanh

        # --- Spectral radius initialization (critical for constrained RNNs) ---
        # Standard approach: Sompolinsky et al. 1988, Sussillo & Abbott 2009
        # After applying Dale's law + sparsity, rescale W_rec so that the
        # spectral radius equals g ≈ 1.2 (slightly chaotic, good for learning).
        self._initialize_recurrent_weights(target_spectral_radius=1.2)

    @torch.no_grad()
    def _initialize_recurrent_weights(self, target_spectral_radius: float = 1.2) -> None:
        """Rescale recurrent weights so spectral radius matches target.

        After applying Dale's law and sparsity constraints, the effective
        spectral radius can be far from 1.0.  This rescales the raw weights
        so that the *constrained* matrix has the desired spectral radius,
        placing the network at the edge of chaos — the optimal regime for
        computation and learning (Sussillo & Abbott 2009).
        """
        W = self._get_constrained_W_rec()
        # Compute current spectral radius (largest singular value is an
        # upper bound; eigenvalue magnitude is exact but may need complex ops)
        try:
            svs = torch.linalg.svdvals(W)
            current_radius = float(svs[0].item())
        except Exception:
            current_radius = float(torch.norm(W, p=2).item())

        if current_radius > 1e-6:
            scale = target_spectral_radius / current_radius
            self.W_rec.weight.data.mul_(scale)

    def _get_constrained_W_rec(self) -> torch.Tensor:
        """Get recurrent weights with biological constraints applied."""
        W = self.W_rec.weight

        if self._dale_law and self.excitatory_mask is not None:
            W = enforce_dale_law(W, self.excitatory_mask)

        if self._sparse and self.sparse_mask is not None:
            W = apply_sparse_mask(W, self.sparse_mask)

        return W

    def _step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Single RNN timestep with continuous-time dynamics.

        Uses the standard computational neuroscience formulation:
            h_new = (1 - alpha) * h + alpha * f(W_in @ x + W_rec @ h + b)
        where alpha = dt/tau is the time constant ratio.

        Args:
            x_t: Input at current timestep, shape (batch, input_dim).
            h: Previous hidden state, shape (batch, hidden_size).

        Returns:
            New hidden state, shape (batch, hidden_size).
        """
        # Input contribution
        h_input = self.input_proj(x_t)

        # Recurrent contribution with constraints
        W_rec = self._get_constrained_W_rec()
        h_rec = h @ W_rec.t()

        # Nonlinear activation of total drive
        h_raw = self._activation(h_input + h_rec + self.h_bias)

        # Apply motif layers in sequence
        h_motif = self.normalization(h_raw)
        h_motif = self.attractor(h_motif)
        h_motif = self.gating(h_motif)
        h_motif = self.expansion(h_motif)

        # Continuous-time leak: smooth integration prevents instant saturation
        # and provides better gradient flow through the (1-alpha) bypass
        h_new = (1.0 - self._alpha) * h + self._alpha * h_motif

        result: torch.Tensor = h_new
        return result

    def forward(
        self,
        inputs: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — returns outputs only (for training).

        Args:
            inputs: Input tensor, shape (batch, time, input_dim).
            h0: Optional initial hidden state, shape (batch, hidden_size).

        Returns:
            Output tensor, shape (batch, time, output_dim).
        """
        batch_size, n_time, _ = inputs.shape

        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h = h0

        outputs = []
        for t in range(n_time):
            h = self._step(inputs[:, t, :], h)
            out = self.readout(h)
            outputs.append(out)

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def forward_with_states(
        self,
        inputs: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also records hidden states (for analysis).

        Args:
            inputs: Input tensor, shape (batch, time, input_dim).
            h0: Optional initial hidden state, shape (batch, hidden_size).

        Returns:
            Tuple of:
                - outputs: shape (batch, time, output_dim)
                - states: hidden states, shape (batch, time, hidden_size)
        """
        batch_size, n_time, _ = inputs.shape

        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h = h0

        outputs = []
        states = []
        for t in range(n_time):
            h = self._step(inputs[:, t, :], h)
            out = self.readout(h)
            outputs.append(out)
            states.append(h)

        return torch.stack(outputs, dim=1), torch.stack(states, dim=1)

    @torch.no_grad()
    def record_states(
        self,
        task_batches: dict[str, TaskBatch],
    ) -> NeuralPopulationData:
        """Run forward on task batches and capture hidden states.

        Produces a ``NeuralPopulationData`` with trial metadata extracted
        from the ``TaskBatch`` objects. This is the bridge between the
        model and the analysis pipeline.

        Args:
            task_batches: Mapping from task name to generated TaskBatch.
                Multiple tasks are concatenated along the trial dimension.

        Returns:
            NeuralPopulationData with source="rnn" and full trial metadata.
        """
        from ..population import NeuralPopulationData, TrialMetadata

        all_states: list[np.ndarray] = []
        all_labels: list[str] = []
        all_outcomes: list[np.ndarray] = []
        # Track per-task batch sizes and per-task condition/epoch arrays
        task_batch_sizes: list[int] = []
        task_conditions: list[dict[str, np.ndarray]] = []
        task_epoch_boundaries: list[dict[str, np.ndarray]] = []

        n_time: int | None = None

        for task_name, batch in task_batches.items():
            _, states = self.forward_with_states(batch.inputs)
            batch_size = states.shape[0]
            cur_n_time = states.shape[1]

            if n_time is None:
                n_time = cur_n_time

            # Pad or truncate to common timepoints if tasks differ
            if cur_n_time != n_time:
                # Use max and zero-pad shorter ones
                if cur_n_time < n_time:
                    pad = torch.zeros(
                        batch_size, n_time - cur_n_time, self.hidden_size,
                        device=states.device,
                    )
                    states = torch.cat([states, pad], dim=1)
                else:
                    states = states[:, :n_time, :]

            all_states.append(states.cpu().numpy())
            all_labels.extend([task_name] * batch_size)
            task_batch_sizes.append(batch_size)

            # Collect conditions for this task
            conds: dict[str, np.ndarray] = {}
            for key, val in batch.conditions.items():
                conds[key] = val.cpu().numpy()
            task_conditions.append(conds)

            # Collect epoch boundaries for this task
            epochs: dict[str, np.ndarray] = {}
            for key, val in batch.epoch_boundaries.items():
                epochs[key] = val.cpu().numpy()
            task_epoch_boundaries.append(epochs)

            # Compute outcomes: compare model output sign to target sign
            # during the response epoch for each trial
            full_outputs = self.readout(states)  # (batch, time, output_dim)
            resp_start = batch.epoch_boundaries.get("response")
            if resp_start is not None:
                outcomes = np.zeros(batch_size, dtype=np.float64)
                for i in range(batch_size):
                    r = int(resp_start[i].item())
                    pred = full_outputs[i, r, 0].item()
                    tgt = batch.targets[i, r, 0].item()
                    outcomes[i] = 1.0 if (pred > 0) == (tgt > 0) else 0.0
            else:
                # No response epoch defined — mark unknown
                outcomes = -np.ones(batch_size, dtype=np.float64)
            all_outcomes.append(outcomes)

        assert n_time is not None, "No task batches provided"

        # Concatenate across tasks
        activity = np.concatenate(all_states, axis=0)
        n_trials = activity.shape[0]
        n_units = activity.shape[2]

        # Build time axis (arbitrary dt=1 for RNN steps)
        time_axis = np.arange(n_time, dtype=np.float64)

        # Build unit labels
        unit_labels = tuple(f"unit_{i}" for i in range(n_units))

        # Collect all condition keys and epoch boundary keys across tasks
        all_cond_keys: set[str] = set()
        all_epoch_keys: set[str] = set()
        for conds in task_conditions:
            all_cond_keys.update(conds.keys())
        for epochs in task_epoch_boundaries:
            all_epoch_keys.update(epochs.keys())

        # Build merged conditions — pad with NaN for tasks missing a key
        merged_conditions: dict[str, np.ndarray] = {}
        for key in sorted(all_cond_keys):
            arrays: list[np.ndarray] = []
            for idx, conds in enumerate(task_conditions):
                bs = task_batch_sizes[idx]
                if key in conds:
                    arrays.append(conds[key])
                else:
                    arrays.append(np.full(bs, np.nan, dtype=np.float64))
            merged_conditions[key] = np.concatenate(arrays, axis=0)

        # Build merged epoch boundaries — pad with NaN for tasks missing a key
        merged_epoch_boundaries: dict[str, np.ndarray] = {}
        for key in sorted(all_epoch_keys):
            arrays_ep: list[np.ndarray] = []
            for idx, epochs in enumerate(task_epoch_boundaries):
                bs = task_batch_sizes[idx]
                if key in epochs:
                    arrays_ep.append(epochs[key])
                else:
                    arrays_ep.append(np.full(bs, np.nan, dtype=np.float64))
            merged_epoch_boundaries[key] = np.concatenate(arrays_ep, axis=0)

        merged_outcomes = np.concatenate(all_outcomes, axis=0)

        trial_metadata = TrialMetadata(
            conditions=merged_conditions,
            outcomes=merged_outcomes,
            epoch_boundaries=merged_epoch_boundaries,
        )

        return NeuralPopulationData(
            activity=activity,
            trial_labels=tuple(all_labels),
            time_axis=time_axis,
            unit_labels=unit_labels,
            source="rnn",
            metadata={
                "model_config": {
                    "hidden_size": self.config.hidden_size,
                    "num_layers": self.config.num_layers,
                    "cell_type": self.config.cell_type,
                    "dale_law": self.config.dale_law,
                    "sparse_connectivity": self.config.sparse_connectivity,
                },
                "motifs": {
                    "normalization_gain_modulation": self.motifs.normalization_gain_modulation,
                    "attractor_dynamics": self.motifs.attractor_dynamics,
                    "selective_gating": self.motifs.selective_gating,
                    "expansion_recoding": self.motifs.expansion_recoding,
                },
            },
            trial_metadata=trial_metadata,
        )
