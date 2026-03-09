"""Shared test fixtures for the geometric-signatures test suite.

Provides reusable fixtures that eliminate boilerplate across test modules:

- ``tiny_motifs``: All-true MotifSwitches for quick tests.
- ``tiny_config``: Minimal ExperimentConfig from baseline YAML.
- ``synthetic_population_data``: Small NeuralPopulationData for analysis tests.
- ``synthetic_population_data_with_metadata``: Same but with TrialMetadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geometric_signatures.config import ExperimentConfig, load_experiment_config
from geometric_signatures.motifs import MotifSwitches
from geometric_signatures.population import NeuralPopulationData, TrialMetadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG = PROJECT_ROOT / "config" / "experiment.baseline.yaml"


@pytest.fixture
def tiny_motifs() -> MotifSwitches:
    """All-true MotifSwitches for quick tests."""
    return MotifSwitches(
        normalization_gain_modulation=True,
        attractor_dynamics=True,
        selective_gating=True,
        expansion_recoding=True,
    )


@pytest.fixture
def tiny_config() -> ExperimentConfig:
    """Minimal ExperimentConfig loaded from the baseline YAML."""
    return load_experiment_config(BASELINE_CONFIG)


@pytest.fixture
def synthetic_population_data() -> NeuralPopulationData:
    """Small synthetic NeuralPopulationData without trial metadata.

    Shape: 4 trials, 10 timepoints, 8 units.
    Trial labels: 2x context_dependent_integration, 2x evidence_accumulation.
    """
    rng = np.random.default_rng(42)
    n_trials, n_time, n_units = 4, 10, 8

    return NeuralPopulationData(
        activity=rng.standard_normal((n_trials, n_time, n_units)),
        trial_labels=(
            "context_dependent_integration",
            "context_dependent_integration",
            "evidence_accumulation",
            "evidence_accumulation",
        ),
        time_axis=np.linspace(0.0, 0.9, n_time),
        unit_labels=tuple(f"unit_{i}" for i in range(n_units)),
        source="rnn",
        metadata={"model_config": "test", "seed": 42},
    )


@pytest.fixture
def synthetic_population_data_with_metadata() -> NeuralPopulationData:
    """Small synthetic NeuralPopulationData with full TrialMetadata.

    Shape: 8 trials, 20 timepoints, 6 units.
    Trial labels: 4 tasks x 2 trials each.
    Includes conditions (coherence), outcomes, and epoch boundaries.
    """
    rng = np.random.default_rng(123)
    n_trials, n_time, n_units = 8, 20, 6

    trial_labels = (
        "context_dependent_integration",
        "context_dependent_integration",
        "evidence_accumulation",
        "evidence_accumulation",
        "working_memory",
        "working_memory",
        "perceptual_discrimination",
        "perceptual_discrimination",
    )

    trial_metadata = TrialMetadata(
        conditions={
            "coherence": rng.uniform(0.0, 1.0, size=(n_trials,)),
            "context_id": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        },
        outcomes=np.array([1, 0, 1, 1, 0, 1, 1, 1]),  # 1=correct, 0=error
        epoch_boundaries={
            "stimulus_onset": np.full(n_trials, 2),
            "delay": np.full(n_trials, 8),
            "response": np.full(n_trials, 14),
        },
    )

    return NeuralPopulationData(
        activity=rng.standard_normal((n_trials, n_time, n_units)),
        trial_labels=trial_labels,
        time_axis=np.linspace(0.0, 1.9, n_time),
        unit_labels=tuple(f"neuron_{i}" for i in range(n_units)),
        source="rnn",
        metadata={"model_config": "test_with_metadata", "seed": 123},
        trial_metadata=trial_metadata,
    )
