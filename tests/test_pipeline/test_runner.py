"""Tests for pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geometric_signatures.analysis.base import AnalysisResult
from geometric_signatures.analysis.results import save_results
from geometric_signatures.pipeline.runner import (
    PipelineOptions,
    PipelineResult,
    _save_comparison_summary,
    run_pipeline,
)
from geometric_signatures.population import NeuralPopulationData
from geometric_signatures.statistics.permutation import PermutationTestResult


class TestPipelineOptions:
    """Tests for PipelineOptions defaults."""

    def test_defaults(self) -> None:
        opts = PipelineOptions()
        assert opts.skip_training is False
        assert opts.skip_analysis is False
        assert opts.skip_statistics is False
        assert opts.variants is None
        assert opts.analysis_methods is None
        assert opts.device == "auto"

    def test_custom_options(self) -> None:
        opts = PipelineOptions(
            skip_training=True,
            variants=("complete",),
            device="cuda",
        )
        assert opts.skip_training is True
        assert opts.variants == ("complete",)
        assert opts.device == "cuda"


class TestSaveComparisonSummary:
    """Tests for comparison summary persistence."""

    def test_saves_json(self, tmp_path: Path) -> None:
        comparisons = {
            "ablate_x": {
                "pg.pr": PermutationTestResult(
                    observed_statistic=2.5,
                    p_value=0.01,
                    null_distribution=np.zeros(100),
                    n_permutations=100,
                ),
            },
        }
        path = tmp_path / "comparisons.json"
        _save_comparison_summary(comparisons, path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert "ablate_x" in data
        assert data["ablate_x"]["pg.pr"]["p_value"] == 0.01
        assert data["ablate_x"]["pg.pr"]["observed_statistic"] == 2.5

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "summary.json"
        _save_comparison_summary({}, path)
        assert path.exists()

    def test_empty_comparisons(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        _save_comparison_summary({}, path)
        data = json.loads(path.read_text())
        assert data == {}


class TestPipelineResult:
    """Tests for PipelineResult structure."""

    def test_construction(self) -> None:
        result = PipelineResult(
            analysis_results={},
            aggregated_results={},
            comparisons={},
            n_variants=5,
            n_seeds=10,
        )
        assert result.n_variants == 5
        assert result.n_seeds == 10
