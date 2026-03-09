"""Tests for composable pipeline stages."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.analysis.base import AnalysisResult
from geometric_signatures.config import ExperimentConfig
from geometric_signatures.pipeline.stages import (
    stage_aggregate,
    stage_analyze,
    stage_compare,
    stage_generate_variants,
    stage_preprocess,
)
from geometric_signatures.population import NeuralPopulationData


@pytest.fixture()
def tiny_config(tiny_config: ExperimentConfig) -> ExperimentConfig:
    """Use the conftest tiny_config fixture."""
    return tiny_config


@pytest.fixture()
def sample_population() -> NeuralPopulationData:
    """Small synthetic data for pipeline tests."""
    rng = np.random.default_rng(42)
    return NeuralPopulationData(
        activity=rng.standard_normal((8, 10, 6)),
        trial_labels=("task_a", "task_a", "task_b", "task_b",
                       "task_a", "task_a", "task_b", "task_b"),
        time_axis=np.arange(10, dtype=np.float64),
        unit_labels=tuple(f"u{i}" for i in range(6)),
        source="rnn",
        metadata={},
    )


class TestStageGenerateVariants:
    """Tests for variant generation stage."""

    def test_generates_five_variants(self, tiny_config: ExperimentConfig) -> None:
        variants = stage_generate_variants(tiny_config)
        # 1 complete + 4 single ablations
        assert len(variants) == 5

    def test_contains_complete(self, tiny_config: ExperimentConfig) -> None:
        variants = stage_generate_variants(tiny_config)
        assert "complete" in variants

    def test_ablation_names(self, tiny_config: ExperimentConfig) -> None:
        variants = stage_generate_variants(tiny_config)
        names = set(variants.keys())
        assert "ablate_normalization" in names or any(
            k.startswith("ablate_") for k in names
        )


class TestStagePreprocess:
    """Tests for preprocessing stage."""

    def test_returns_population_data(
        self, sample_population: NeuralPopulationData
    ) -> None:
        result = stage_preprocess(sample_population, method="population_geometry")
        assert isinstance(result, NeuralPopulationData)

    def test_no_method_returns_original(
        self, sample_population: NeuralPopulationData
    ) -> None:
        """Without method, applies default preprocessing."""
        result = stage_preprocess(sample_population)
        assert result.activity.shape[0] == sample_population.activity.shape[0]

    def test_rsa_trial_averaging(
        self, sample_population: NeuralPopulationData
    ) -> None:
        result = stage_preprocess(sample_population, method="rsa")
        # RSA default: trial_average=True → fewer trials (unique labels)
        n_unique = len(set(sample_population.trial_labels))
        assert result.activity.shape[0] == n_unique


class TestStageAnalyze:
    """Tests for analysis stage."""

    def test_runs_single_method(
        self, sample_population: NeuralPopulationData
    ) -> None:
        results = stage_analyze(sample_population, ["population_geometry"])
        assert "population_geometry" in results
        assert isinstance(results["population_geometry"], AnalysisResult)

    def test_runs_multiple_methods(
        self, sample_population: NeuralPopulationData
    ) -> None:
        results = stage_analyze(sample_population, ["cka", "rsa"])
        assert "cka" in results
        assert "rsa" in results


class TestStageAggregate:
    """Tests for aggregation stage."""

    def test_aggregates_multiple_seeds(self) -> None:
        results = {
            "population_geometry": [
                AnalysisResult(
                    method="population_geometry",
                    config_hash="abc",
                    seed=i,
                    variant="complete",
                    scalars={"pr": float(i)},
                )
                for i in range(5)
            ]
        }
        agg = stage_aggregate(results, n_bootstrap=100,
                              rng=np.random.default_rng(42))
        assert "population_geometry" in agg
        assert agg["population_geometry"].n_seeds == 5

    def test_skips_empty_methods(self) -> None:
        results: dict[str, list[AnalysisResult]] = {"cka": []}
        agg = stage_aggregate(results, rng=np.random.default_rng(42))
        assert "cka" not in agg


class TestStageCompare:
    """Tests for comparison stage."""

    def test_compares_against_reference(self) -> None:
        rng = np.random.default_rng(42)

        def make_results(variant: str, offset: float) -> list[AnalysisResult]:
            return [
                AnalysisResult(
                    method="pg",
                    config_hash="abc",
                    seed=i,
                    variant=variant,
                    scalars={"pr": offset + rng.standard_normal() * 0.1},
                )
                for i in range(10)
            ]

        variant_results = {
            "complete": {"pg": make_results("complete", 5.0)},
            "ablate_x": {"pg": make_results("ablate_x", 2.0)},
        }

        comparisons = stage_compare(
            variant_results,
            reference_variant="complete",
            n_permutations=200,
            rng=np.random.default_rng(42),
        )

        assert "ablate_x" in comparisons
        assert "pg.pr" in comparisons["ablate_x"]

    def test_missing_reference_raises(self) -> None:
        with pytest.raises(ValueError, match="not in results"):
            stage_compare({}, reference_variant="nonexistent")

    def test_skips_reference_variant(self) -> None:
        """Reference variant should not be compared against itself."""
        variant_results = {
            "complete": {"pg": [
                AnalysisResult(
                    method="pg", config_hash="abc", seed=i,
                    variant="complete", scalars={"pr": 3.0},
                )
                for i in range(5)
            ]},
        }
        comparisons = stage_compare(
            variant_results,
            reference_variant="complete",
            rng=np.random.default_rng(42),
        )
        assert "complete" not in comparisons
