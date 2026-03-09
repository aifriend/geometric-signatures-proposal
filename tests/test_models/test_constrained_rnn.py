"""Tests for the ConstrainedRNN model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from geometric_signatures.config import ModelConfig
from geometric_signatures.models.constrained_rnn import ConstrainedRNN
from geometric_signatures.models.layers import (
    AttractorRecurrence,
    DivisiveNormalization,
    ExpansionRecoding,
    SelectiveGating,
)
from geometric_signatures.motifs import MotifSwitches
from geometric_signatures.tasks_data.base import TaskBatch


def _make_config(
    hidden: int = 16,
    dale: bool = False,
    sparsity: float = 0.0,
    input_dim: int = 2,
    output_dim: int = 1,
) -> ModelConfig:
    return ModelConfig(
        hidden_size=hidden,
        num_layers=1,
        cell_type="constrained_rnn",
        dale_law=dale,
        sparse_connectivity=sparsity,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _all_motifs() -> MotifSwitches:
    return MotifSwitches(
        normalization_gain_modulation=True,
        attractor_dynamics=True,
        selective_gating=True,
        expansion_recoding=True,
    )


def _no_motifs() -> MotifSwitches:
    return MotifSwitches(
        normalization_gain_modulation=False,
        attractor_dynamics=False,
        selective_gating=False,
        expansion_recoding=False,
    )


class TestForwardPass:
    """Tests for the forward pass output shapes."""

    def test_output_shape(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        x = torch.randn(4, 10, 2)
        y = model(x)
        assert y.shape == (4, 10, 1)

    def test_output_shape_no_motifs(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _no_motifs())
        x = torch.randn(4, 10, 2)
        y = model(x)
        assert y.shape == (4, 10, 1)

    def test_with_initial_state(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        x = torch.randn(4, 10, 2)
        h0 = torch.randn(4, 16)
        y = model(x, h0=h0)
        assert y.shape == (4, 10, 1)

    def test_different_dims(self) -> None:
        cfg = _make_config(hidden=32, input_dim=4, output_dim=2)
        model = ConstrainedRNN(cfg, _all_motifs())
        x = torch.randn(8, 20, 4)
        y = model(x)
        assert y.shape == (8, 20, 2)


class TestMotifToggling:
    """Tests that motif layers are correctly toggled."""

    def test_all_motifs_active(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        assert isinstance(model.normalization, DivisiveNormalization)
        assert isinstance(model.attractor, AttractorRecurrence)
        assert isinstance(model.gating, SelectiveGating)
        assert isinstance(model.expansion, ExpansionRecoding)

    def test_no_motifs_identity(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _no_motifs())
        assert isinstance(model.normalization, torch.nn.Identity)
        assert isinstance(model.attractor, torch.nn.Identity)
        assert isinstance(model.gating, torch.nn.Identity)
        assert isinstance(model.expansion, torch.nn.Identity)

    def test_single_motif_ablation(self) -> None:
        """Ablating one motif should make only that layer identity."""
        cfg = _make_config()
        motifs = MotifSwitches(
            normalization_gain_modulation=True,
            attractor_dynamics=False,  # ablated
            selective_gating=True,
            expansion_recoding=True,
        )
        model = ConstrainedRNN(cfg, motifs)
        assert isinstance(model.normalization, DivisiveNormalization)
        assert isinstance(model.attractor, torch.nn.Identity)
        assert isinstance(model.gating, SelectiveGating)
        assert isinstance(model.expansion, ExpansionRecoding)


class TestConstraints:
    """Tests for biological constraints (Dale's law, sparsity)."""

    def test_dale_law_applied(self) -> None:
        cfg = _make_config(dale=True)
        model = ConstrainedRNN(cfg, _all_motifs())
        W = model._get_constrained_W_rec()
        exc_mask = model.excitatory_mask
        # Excitatory columns should be non-negative
        assert (W[:, exc_mask] >= 0).all()
        # Inhibitory columns should be non-positive
        assert (W[:, ~exc_mask] <= 0).all()

    def test_no_dale_law(self) -> None:
        cfg = _make_config(dale=False)
        model = ConstrainedRNN(cfg, _all_motifs())
        assert model.excitatory_mask is None

    def test_sparse_connectivity(self) -> None:
        cfg = _make_config(sparsity=0.5)
        model = ConstrainedRNN(cfg, _all_motifs())
        W = model._get_constrained_W_rec()
        # Should have zeros where mask is 0
        zero_frac = (W == 0).float().mean().item()
        assert zero_frac > 0.3  # some zeros from sparsity

    def test_no_sparsity(self) -> None:
        cfg = _make_config(sparsity=0.0)
        model = ConstrainedRNN(cfg, _all_motifs())
        assert model.sparse_mask is None

    def test_combined_constraints(self) -> None:
        cfg = _make_config(dale=True, sparsity=0.3)
        model = ConstrainedRNN(cfg, _all_motifs())
        W = model._get_constrained_W_rec()
        exc_mask = model.excitatory_mask
        # Non-zero excitatory entries should be positive
        exc_cols = W[:, exc_mask]
        nonzero_exc = exc_cols[exc_cols != 0]
        if nonzero_exc.numel() > 0:
            assert (nonzero_exc > 0).all()


class TestForwardWithStates:
    """Tests for forward_with_states (analysis mode)."""

    def test_states_shape(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        x = torch.randn(4, 10, 2)
        outputs, states = model.forward_with_states(x)
        assert outputs.shape == (4, 10, 1)
        assert states.shape == (4, 10, 16)

    def test_no_grad(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        x = torch.randn(4, 10, 2)
        outputs, states = model.forward_with_states(x)
        assert not outputs.requires_grad
        assert not states.requires_grad

    def test_outputs_match_forward(self) -> None:
        """forward_with_states outputs should match forward outputs."""
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        model.eval()
        x = torch.randn(4, 10, 2)
        with torch.no_grad():
            y_forward = model(x)
        y_states, _ = model.forward_with_states(x)
        assert torch.allclose(y_forward, y_states)


class TestRecordStates:
    """Tests for record_states (produces NeuralPopulationData)."""

    def _make_task_batch(
        self, task_name: str, batch_size: int = 8, n_time: int = 20
    ) -> TaskBatch:
        return TaskBatch(
            inputs=torch.randn(batch_size, n_time, 2),
            targets=torch.randn(batch_size, n_time, 1),
            mask=torch.ones(batch_size, n_time),
            task_name=task_name,
            conditions={"coherence": torch.rand(batch_size)},
            epoch_boundaries={
                "stimulus_onset": torch.full((batch_size,), 5, dtype=torch.long),
                "response": torch.full((batch_size,), 15, dtype=torch.long),
            },
        )

    def test_returns_neural_population_data(self) -> None:
        from geometric_signatures.population import NeuralPopulationData

        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        batches = {"task_a": self._make_task_batch("task_a")}
        data = model.record_states(batches)
        assert isinstance(data, NeuralPopulationData)

    def test_activity_shape(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        batches = {"task_a": self._make_task_batch("task_a", batch_size=8)}
        data = model.record_states(batches)
        assert data.activity.shape == (8, 20, 16)

    def test_multi_task_concatenation(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        batches = {
            "task_a": self._make_task_batch("task_a", batch_size=4),
            "task_b": self._make_task_batch("task_b", batch_size=6),
        }
        data = model.record_states(batches)
        assert data.n_trials == 10
        assert data.trial_labels[:4] == ("task_a",) * 4
        assert data.trial_labels[4:] == ("task_b",) * 6

    def test_source_is_rnn(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        batches = {"task_a": self._make_task_batch("task_a")}
        data = model.record_states(batches)
        assert data.source == "rnn"

    def test_trial_metadata_present(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        batches = {"task_a": self._make_task_batch("task_a")}
        data = model.record_states(batches)
        assert data.trial_metadata is not None
        assert "coherence" in data.trial_metadata.conditions
        assert "response" in data.trial_metadata.epoch_boundaries
        assert len(data.trial_metadata.outcomes) == 8

    def test_metadata_has_model_info(self) -> None:
        cfg = _make_config()
        model = ConstrainedRNN(cfg, _all_motifs())
        batches = {"task_a": self._make_task_batch("task_a")}
        data = model.record_states(batches)
        assert "model_config" in data.metadata
        assert data.metadata["model_config"]["hidden_size"] == 16
        assert "motifs" in data.metadata
