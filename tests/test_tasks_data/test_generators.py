"""Tests for all four task data generators.

Covers shape, dtype, determinism, epoch structure, conditions, and registry.
All tests skip gracefully when torch is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geometric_signatures.tasks_data import TASK_REGISTRY, get_task
from geometric_signatures.tasks_data.base import TaskBatch, TaskDataset
from geometric_signatures.tasks_data.context_dependent_integration import (
    ContextDependentIntegration,
)
from geometric_signatures.tasks_data.evidence_accumulation import EvidenceAccumulation
from geometric_signatures.tasks_data.perceptual_discrimination import (
    PerceptualDiscrimination,
)
from geometric_signatures.tasks_data.working_memory import WorkingMemory


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
ALL_TASKS: list[tuple[str, type[TaskDataset]]] = [
    ("context_dependent_integration", ContextDependentIntegration),
    ("evidence_accumulation", EvidenceAccumulation),
    ("working_memory", WorkingMemory),
    ("perceptual_discrimination", PerceptualDiscrimination),
]


def _make_batch(cls: type[TaskDataset], seed: int = 42) -> TaskBatch:
    gen = cls()
    rng = torch.Generator().manual_seed(seed)
    return gen.generate_batch(BATCH_SIZE, rng)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------
class TestRegistry:
    """Tests for TASK_REGISTRY and get_task()."""

    def test_all_four_tasks_registered(self) -> None:
        assert set(TASK_REGISTRY) == {
            "context_dependent_integration",
            "evidence_accumulation",
            "working_memory",
            "perceptual_discrimination",
        }

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_get_task_returns_correct_class(
        self, name: str, cls: type[TaskDataset]
    ) -> None:
        assert get_task(name) is cls

    def test_get_task_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")


# ---------------------------------------------------------------------------
# Parametrized tests across all generators
# ---------------------------------------------------------------------------
class TestAllGenerators:
    """Shape, dtype, and structural tests that apply to every generator."""

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_output_shapes(self, name: str, cls: type[TaskDataset]) -> None:
        gen = cls()
        batch = _make_batch(cls)
        n_t = gen.n_timepoints
        assert batch.inputs.shape == (BATCH_SIZE, n_t, gen.input_dim)
        assert batch.targets.shape == (BATCH_SIZE, n_t, gen.output_dim)
        assert batch.mask.shape == (BATCH_SIZE, n_t)

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_dtypes_float(self, name: str, cls: type[TaskDataset]) -> None:
        batch = _make_batch(cls)
        assert batch.inputs.dtype == torch.float32
        assert batch.targets.dtype == torch.float32
        assert batch.mask.dtype == torch.float32

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_task_name_matches(self, name: str, cls: type[TaskDataset]) -> None:
        batch = _make_batch(cls)
        assert batch.task_name == name

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_mask_binary(self, name: str, cls: type[TaskDataset]) -> None:
        batch = _make_batch(cls)
        unique = torch.unique(batch.mask)
        for v in unique:
            assert v.item() in (0.0, 1.0)

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_has_conditions(self, name: str, cls: type[TaskDataset]) -> None:
        batch = _make_batch(cls)
        assert isinstance(batch.conditions, dict)
        assert len(batch.conditions) > 0
        for key, val in batch.conditions.items():
            assert isinstance(key, str)
            assert val.shape[0] == BATCH_SIZE

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_has_epoch_boundaries(self, name: str, cls: type[TaskDataset]) -> None:
        batch = _make_batch(cls)
        assert isinstance(batch.epoch_boundaries, dict)
        assert len(batch.epoch_boundaries) > 0
        # All must have "response"
        assert "response" in batch.epoch_boundaries
        for key, val in batch.epoch_boundaries.items():
            assert val.shape == (BATCH_SIZE,)
            assert val.dtype == torch.long

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_response_epoch_has_mask(self, name: str, cls: type[TaskDataset]) -> None:
        """Mask should be 1.0 from response onset onward."""
        batch = _make_batch(cls)
        resp_start = batch.epoch_boundaries["response"][0].item()
        gen = cls()
        n_t = gen.n_timepoints
        # All trials should have mask=1 in response period
        assert (batch.mask[:, resp_start:] == 1.0).all()
        # At least some timesteps before response should have mask=0
        if resp_start > 0:
            assert (batch.mask[:, :resp_start] == 0.0).all()

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_deterministic_with_same_seed(
        self, name: str, cls: type[TaskDataset]
    ) -> None:
        batch1 = _make_batch(cls, seed=123)
        batch2 = _make_batch(cls, seed=123)
        assert torch.equal(batch1.inputs, batch2.inputs)
        assert torch.equal(batch1.targets, batch2.targets)
        assert torch.equal(batch1.mask, batch2.mask)

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_different_seed_different_data(
        self, name: str, cls: type[TaskDataset]
    ) -> None:
        batch1 = _make_batch(cls, seed=1)
        batch2 = _make_batch(cls, seed=999)
        # Inputs should differ (noise is different)
        assert not torch.equal(batch1.inputs, batch2.inputs)

    @pytest.mark.parametrize("name,cls", ALL_TASKS)
    def test_n_timepoints_property(self, name: str, cls: type[TaskDataset]) -> None:
        gen = cls()
        assert isinstance(gen.n_timepoints, int)
        assert gen.n_timepoints > 0


# ---------------------------------------------------------------------------
# Task-specific tests
# ---------------------------------------------------------------------------
class TestContextDependentIntegration:
    """Tests specific to the context-dependent integration task."""

    def test_input_dim(self) -> None:
        gen = ContextDependentIntegration()
        assert gen.input_dim == 4  # stim1, stim2, context, fixation

    def test_output_dim(self) -> None:
        gen = ContextDependentIntegration()
        assert gen.output_dim == 1

    def test_conditions_keys(self) -> None:
        batch = _make_batch(ContextDependentIntegration)
        expected = {"coherence_1", "coherence_2", "context", "attended_coherence"}
        assert set(batch.conditions.keys()) == expected

    def test_epoch_boundaries_keys(self) -> None:
        batch = _make_batch(ContextDependentIntegration)
        expected = {"stimulus_onset", "delay", "response"}
        assert set(batch.epoch_boundaries.keys()) == expected

    def test_context_is_binary(self) -> None:
        batch = _make_batch(ContextDependentIntegration)
        ctx = batch.conditions["context"]
        assert set(ctx.unique().tolist()).issubset({0.0, 1.0})

    def test_fixation_signal(self) -> None:
        batch = _make_batch(ContextDependentIntegration)
        resp_start = batch.epoch_boundaries["response"][0].item()
        # Fixation (channel 3) should be 1 before response
        assert (batch.inputs[:, :resp_start, 3] == 1.0).all()

    def test_custom_timesteps(self) -> None:
        gen = ContextDependentIntegration(n_timesteps=50, fixation_steps=5)
        assert gen.n_timepoints == 50


class TestEvidenceAccumulation:
    """Tests specific to the evidence accumulation task."""

    def test_input_dim(self) -> None:
        gen = EvidenceAccumulation()
        assert gen.input_dim == 2  # evidence, fixation

    def test_output_dim(self) -> None:
        gen = EvidenceAccumulation()
        assert gen.output_dim == 1

    def test_conditions_keys(self) -> None:
        batch = _make_batch(EvidenceAccumulation)
        assert set(batch.conditions.keys()) == {"coherence"}

    def test_epoch_boundaries_keys(self) -> None:
        batch = _make_batch(EvidenceAccumulation)
        assert set(batch.epoch_boundaries.keys()) == {"stimulus_onset", "response"}

    def test_coherence_range(self) -> None:
        batch = _make_batch(EvidenceAccumulation)
        coh = batch.conditions["coherence"]
        assert (coh >= -1.0).all() and (coh <= 1.0).all()

    def test_target_is_sign_of_coherence(self) -> None:
        batch = _make_batch(EvidenceAccumulation)
        resp_start = batch.epoch_boundaries["response"][0].item()
        for i in range(BATCH_SIZE):
            coh = batch.conditions["coherence"][i].item()
            expected_sign = 1.0 if coh > 0 else -1.0
            actual = batch.targets[i, resp_start, 0].item()
            assert actual == expected_sign


class TestWorkingMemory:
    """Tests specific to the working memory task."""

    def test_input_dim(self) -> None:
        gen = WorkingMemory()
        assert gen.input_dim == 2  # stimulus, fixation

    def test_output_dim(self) -> None:
        gen = WorkingMemory()
        assert gen.output_dim == 1

    def test_conditions_keys(self) -> None:
        batch = _make_batch(WorkingMemory)
        expected = {"sample_value", "test_value", "is_match"}
        assert set(batch.conditions.keys()) == expected

    def test_epoch_boundaries_keys(self) -> None:
        batch = _make_batch(WorkingMemory)
        expected = {"stimulus_onset", "delay", "test_onset", "response"}
        assert set(batch.epoch_boundaries.keys()) == expected

    def test_is_match_binary(self) -> None:
        batch = _make_batch(WorkingMemory)
        is_match = batch.conditions["is_match"]
        assert set(is_match.unique().tolist()).issubset({0.0, 1.0})

    def test_match_target_positive(self) -> None:
        """Match trials should have +1 target."""
        batch = _make_batch(WorkingMemory)
        resp_start = batch.epoch_boundaries["response"][0].item()
        for i in range(BATCH_SIZE):
            is_match = batch.conditions["is_match"][i].item()
            target = batch.targets[i, resp_start, 0].item()
            if is_match == 1.0:
                assert target == 1.0
            else:
                assert target == -1.0

    def test_epoch_order(self) -> None:
        """Epochs should be ordered: stimulus < delay < test < response."""
        batch = _make_batch(WorkingMemory)
        stim = batch.epoch_boundaries["stimulus_onset"][0].item()
        delay = batch.epoch_boundaries["delay"][0].item()
        test = batch.epoch_boundaries["test_onset"][0].item()
        resp = batch.epoch_boundaries["response"][0].item()
        assert stim < delay < test < resp


class TestPerceptualDiscrimination:
    """Tests specific to the perceptual discrimination task."""

    def test_input_dim(self) -> None:
        gen = PerceptualDiscrimination()
        assert gen.input_dim == 2  # stimulus, fixation

    def test_output_dim(self) -> None:
        gen = PerceptualDiscrimination()
        assert gen.output_dim == 1

    def test_conditions_keys(self) -> None:
        batch = _make_batch(PerceptualDiscrimination)
        assert set(batch.conditions.keys()) == {"stimulus_value"}

    def test_epoch_boundaries_keys(self) -> None:
        batch = _make_batch(PerceptualDiscrimination)
        assert set(batch.epoch_boundaries.keys()) == {"stimulus_onset", "response"}

    def test_stimulus_range(self) -> None:
        batch = _make_batch(PerceptualDiscrimination)
        stim = batch.conditions["stimulus_value"]
        assert (stim >= -1.0).all() and (stim <= 1.0).all()

    def test_target_is_category(self) -> None:
        """Target should be +1 (above threshold) or -1 (below threshold)."""
        batch = _make_batch(PerceptualDiscrimination)
        resp_start = batch.epoch_boundaries["response"][0].item()
        for i in range(BATCH_SIZE):
            stim_val = batch.conditions["stimulus_value"][i].item()
            expected = 1.0 if stim_val > 0.0 else -1.0  # threshold=0.0
            actual = batch.targets[i, resp_start, 0].item()
            assert actual == expected

    def test_custom_threshold(self) -> None:
        gen = PerceptualDiscrimination(threshold=0.5)
        rng = torch.Generator().manual_seed(42)
        batch = gen.generate_batch(BATCH_SIZE, rng)
        resp_start = batch.epoch_boundaries["response"][0].item()
        for i in range(BATCH_SIZE):
            stim_val = batch.conditions["stimulus_value"][i].item()
            expected = 1.0 if stim_val > 0.5 else -1.0
            actual = batch.targets[i, resp_start, 0].item()
            assert actual == expected
