"""Tests for NeuralPopulationData and TrialMetadata.

Validates:
- Shape consistency at construction time.
- Rejection of mismatched dimensions.
- Metadata immutability (MappingProxyType).
- Trial, epoch, and unit selection methods.
- Correct trial filtering with TrialMetadata.
- Source validation.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from geometric_signatures.population import NeuralPopulationData, TrialMetadata


# ── Construction & validation ──────────────────────────────────


class TestNeuralPopulationDataConstruction:
    """Test __post_init__ validation logic."""

    def test_valid_construction(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        data = synthetic_population_data
        assert data.n_trials == 4
        assert data.n_timepoints == 10
        assert data.n_units == 8
        assert data.source == "rnn"

    def test_valid_construction_with_metadata(
        self, synthetic_population_data_with_metadata: NeuralPopulationData
    ) -> None:
        data = synthetic_population_data_with_metadata
        assert data.n_trials == 8
        assert data.n_timepoints == 20
        assert data.n_units == 6
        assert data.trial_metadata is not None
        assert len(data.trial_metadata.outcomes) == 8

    def test_rejects_2d_activity(self) -> None:
        with pytest.raises(ValueError, match="activity must be 3D"):
            NeuralPopulationData(
                activity=np.zeros((4, 10)),
                trial_labels=("a", "b", "c", "d"),
                time_axis=np.arange(10, dtype=float),
                unit_labels=("u0",),
                source="rnn",
                metadata={},
            )

    def test_rejects_mismatched_trial_labels(self) -> None:
        with pytest.raises(ValueError, match="trial_labels length"):
            NeuralPopulationData(
                activity=np.zeros((4, 10, 3)),
                trial_labels=("a", "b"),  # only 2, need 4
                time_axis=np.arange(10, dtype=float),
                unit_labels=("u0", "u1", "u2"),
                source="rnn",
                metadata={},
            )

    def test_rejects_mismatched_time_axis(self) -> None:
        with pytest.raises(ValueError, match="time_axis length"):
            NeuralPopulationData(
                activity=np.zeros((4, 10, 3)),
                trial_labels=("a", "b", "c", "d"),
                time_axis=np.arange(5, dtype=float),  # 5, need 10
                unit_labels=("u0", "u1", "u2"),
                source="rnn",
                metadata={},
            )

    def test_rejects_mismatched_unit_labels(self) -> None:
        with pytest.raises(ValueError, match="unit_labels length"):
            NeuralPopulationData(
                activity=np.zeros((4, 10, 3)),
                trial_labels=("a", "b", "c", "d"),
                time_axis=np.arange(10, dtype=float),
                unit_labels=("u0",),  # 1, need 3
                source="rnn",
                metadata={},
            )

    def test_rejects_invalid_source(self) -> None:
        with pytest.raises(ValueError, match="source must be one of"):
            NeuralPopulationData(
                activity=np.zeros((2, 5, 3)),
                trial_labels=("a", "b"),
                time_axis=np.arange(5, dtype=float),
                unit_labels=("u0", "u1", "u2"),
                source="invalid_source",
                metadata={},
            )

    def test_accepts_valid_sources(self) -> None:
        for source in ("rnn", "ibl", "allen"):
            data = NeuralPopulationData(
                activity=np.zeros((2, 5, 3)),
                trial_labels=("a", "b"),
                time_axis=np.arange(5, dtype=float),
                unit_labels=("u0", "u1", "u2"),
                source=source,
                metadata={},
            )
            assert data.source == source

    def test_rejects_mismatched_trial_metadata_outcomes(self) -> None:
        tm = TrialMetadata(
            conditions={},
            outcomes=np.array([1, 0]),  # 2 outcomes, but 4 trials
            epoch_boundaries={},
        )
        with pytest.raises(ValueError, match="trial_metadata.outcomes shape"):
            NeuralPopulationData(
                activity=np.zeros((4, 10, 3)),
                trial_labels=("a", "b", "c", "d"),
                time_axis=np.arange(10, dtype=float),
                unit_labels=("u0", "u1", "u2"),
                source="rnn",
                metadata={},
                trial_metadata=tm,
            )

    def test_rejects_mismatched_trial_metadata_conditions(self) -> None:
        tm = TrialMetadata(
            conditions={"coh": np.array([0.5, 0.6])},  # 2, need 4
            outcomes=np.array([1, 0, 1, 1]),
            epoch_boundaries={},
        )
        with pytest.raises(ValueError, match="trial_metadata.conditions"):
            NeuralPopulationData(
                activity=np.zeros((4, 10, 3)),
                trial_labels=("a", "b", "c", "d"),
                time_axis=np.arange(10, dtype=float),
                unit_labels=("u0", "u1", "u2"),
                source="rnn",
                metadata={},
                trial_metadata=tm,
            )

    def test_rejects_mismatched_trial_metadata_epoch_boundaries(self) -> None:
        tm = TrialMetadata(
            conditions={},
            outcomes=np.array([1, 0, 1, 1]),
            epoch_boundaries={"stim": np.array([2, 3])},  # 2, need 4
        )
        with pytest.raises(ValueError, match="trial_metadata.epoch_boundaries"):
            NeuralPopulationData(
                activity=np.zeros((4, 10, 3)),
                trial_labels=("a", "b", "c", "d"),
                time_axis=np.arange(10, dtype=float),
                unit_labels=("u0", "u1", "u2"),
                source="rnn",
                metadata={},
                trial_metadata=tm,
            )


# ── Metadata immutability ──────────────────────────────────────


class TestMetadataImmutability:
    """Test that metadata dict is frozen via MappingProxyType."""

    def test_metadata_is_frozen(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        assert isinstance(synthetic_population_data.metadata, MappingProxyType)

    def test_metadata_is_read_only(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        with pytest.raises(TypeError):
            synthetic_population_data.metadata["new_key"] = "value"  # type: ignore[index]

    def test_metadata_values_accessible(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        assert synthetic_population_data.metadata["seed"] == 42
        assert synthetic_population_data.metadata["model_config"] == "test"


# ── Selection methods ──────────────────────────────────────────


class TestSelectionMethods:
    """Test trial, epoch, and unit subsetting."""

    def test_select_trials_by_task(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        subset = synthetic_population_data.select_trials(
            "context_dependent_integration"
        )
        assert subset.n_trials == 2
        assert all(
            label == "context_dependent_integration"
            for label in subset.trial_labels
        )
        assert subset.n_timepoints == synthetic_population_data.n_timepoints
        assert subset.n_units == synthetic_population_data.n_units

    def test_select_trials_empty_result(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        subset = synthetic_population_data.select_trials("nonexistent_task")
        assert subset.n_trials == 0

    def test_select_trials_preserves_metadata(
        self,
        synthetic_population_data_with_metadata: NeuralPopulationData,
    ) -> None:
        data = synthetic_population_data_with_metadata
        subset = data.select_trials("evidence_accumulation")
        assert subset.n_trials == 2
        assert subset.trial_metadata is not None
        assert len(subset.trial_metadata.outcomes) == 2
        assert "coherence" in subset.trial_metadata.conditions
        assert len(subset.trial_metadata.conditions["coherence"]) == 2

    def test_select_correct_trials(
        self,
        synthetic_population_data_with_metadata: NeuralPopulationData,
    ) -> None:
        data = synthetic_population_data_with_metadata
        # outcomes = [1, 0, 1, 1, 0, 1, 1, 1] → 6 correct
        correct = data.select_correct_trials()
        assert correct.n_trials == 6
        assert correct.trial_metadata is not None
        assert all(correct.trial_metadata.outcomes == 1)

    def test_select_correct_trials_raises_without_metadata(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        with pytest.raises(ValueError, match="trial_metadata is None"):
            synthetic_population_data.select_correct_trials()

    def test_select_epoch(
        self,
        synthetic_population_data_with_metadata: NeuralPopulationData,
    ) -> None:
        data = synthetic_population_data_with_metadata
        # Epoch boundaries: stimulus_onset=2, delay=8, response=14
        # Selecting "stimulus_onset" → indices [2, 8) → 6 timepoints
        epoch_data = data.select_epoch("stimulus_onset")
        assert epoch_data.n_timepoints == 6  # from index 2 to 8
        assert epoch_data.n_trials == data.n_trials
        assert epoch_data.n_units == data.n_units

    def test_select_epoch_last(
        self,
        synthetic_population_data_with_metadata: NeuralPopulationData,
    ) -> None:
        data = synthetic_population_data_with_metadata
        # "response" is the last epoch → indices [14, 20) → 6 timepoints
        epoch_data = data.select_epoch("response")
        assert epoch_data.n_timepoints == 6

    def test_select_epoch_raises_without_metadata(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        with pytest.raises(ValueError, match="trial_metadata is None"):
            synthetic_population_data.select_epoch("stimulus")

    def test_select_epoch_raises_for_unknown_epoch(
        self,
        synthetic_population_data_with_metadata: NeuralPopulationData,
    ) -> None:
        with pytest.raises(ValueError, match="Epoch 'nonexistent' not found"):
            synthetic_population_data_with_metadata.select_epoch("nonexistent")

    def test_select_units(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        subset = synthetic_population_data.select_units([0, 2, 5])
        assert subset.n_units == 3
        assert subset.unit_labels == ("unit_0", "unit_2", "unit_5")
        assert subset.n_trials == synthetic_population_data.n_trials
        assert subset.n_timepoints == synthetic_population_data.n_timepoints

    def test_select_units_preserves_activity_values(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        data = synthetic_population_data
        subset = data.select_units([1, 3])
        np.testing.assert_array_equal(subset.activity[:, :, 0], data.activity[:, :, 1])
        np.testing.assert_array_equal(subset.activity[:, :, 1], data.activity[:, :, 3])


# ── Properties ─────────────────────────────────────────────────


class TestProperties:
    """Test convenience properties."""

    def test_n_trials(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        assert synthetic_population_data.n_trials == 4

    def test_n_timepoints(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        assert synthetic_population_data.n_timepoints == 10

    def test_n_units(
        self, synthetic_population_data: NeuralPopulationData
    ) -> None:
        assert synthetic_population_data.n_units == 8
