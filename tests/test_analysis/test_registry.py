"""Tests for the analysis registry and run_analysis function."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.analysis import (
    get_analysis_method,
    run_analysis,
)
from geometric_signatures.analysis.base import AnalysisMethod
from geometric_signatures.analysis.results import load_results, save_results
from geometric_signatures.population import NeuralPopulationData


@pytest.fixture()
def sample_data() -> NeuralPopulationData:
    """Small synthetic data for registry tests."""
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


class TestGetAnalysisMethod:
    """Tests for get_analysis_method."""

    def test_cka(self) -> None:
        method = get_analysis_method("cka")
        assert isinstance(method, AnalysisMethod)
        assert method.name == "cka"

    def test_rsa(self) -> None:
        method = get_analysis_method("rsa")
        assert isinstance(method, AnalysisMethod)
        assert method.name == "rsa"

    def test_population_geometry(self) -> None:
        method = get_analysis_method("population_geometry")
        assert isinstance(method, AnalysisMethod)
        assert method.name == "population_geometry"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown analysis method"):
            get_analysis_method("nonexistent_method")


class TestRunAnalysis:
    """Tests for run_analysis function."""

    def test_runs_single_method(
        self, sample_data: NeuralPopulationData
    ) -> None:
        results = run_analysis(sample_data, methods=["cka"])
        assert len(results) == 1
        assert "cka" in results

    def test_runs_multiple_methods(
        self, sample_data: NeuralPopulationData
    ) -> None:
        results = run_analysis(
            sample_data, methods=["cka", "population_geometry"]
        )
        assert len(results) == 2
        assert "cka" in results
        assert "population_geometry" in results

    def test_no_preprocess(self, sample_data: NeuralPopulationData) -> None:
        results = run_analysis(
            sample_data, methods=["cka"], preprocess=False
        )
        assert "cka" in results


class TestResultsPersistence:
    """Tests for save_results and load_results."""

    def test_round_trip(
        self, sample_data: NeuralPopulationData, tmp_path: object
    ) -> None:
        from pathlib import Path

        output_dir = Path(str(tmp_path)) / "analysis_results"

        results = run_analysis(
            sample_data, methods=["cka", "population_geometry"]
        )
        save_results(results, output_dir)
        loaded = load_results(output_dir)

        assert set(loaded.keys()) == set(results.keys())
        for method_name in results:
            assert loaded[method_name].method == results[method_name].method
            assert loaded[method_name].scalars == pytest.approx(
                results[method_name].scalars
            )

    def test_load_specific_methods(
        self, sample_data: NeuralPopulationData, tmp_path: object
    ) -> None:
        from pathlib import Path

        output_dir = Path(str(tmp_path)) / "results"

        results = run_analysis(
            sample_data, methods=["cka", "population_geometry"]
        )
        save_results(results, output_dir)
        loaded = load_results(output_dir, methods=["cka"])

        assert len(loaded) == 1
        assert "cka" in loaded
