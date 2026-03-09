"""Tests for analysis base: AnalysisMethod protocol and AnalysisResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geometric_signatures.analysis.base import AnalysisMethod, AnalysisResult


class TestAnalysisResult:
    """Tests for AnalysisResult frozen dataclass."""

    def test_construction(self) -> None:
        result = AnalysisResult(
            method="test",
            config_hash="abc123",
            seed=42,
            variant="complete",
            arrays={"embedding": np.zeros((3, 2))},
            scalars={"metric_a": 1.5},
        )
        assert result.method == "test"
        assert result.config_hash == "abc123"
        assert result.seed == 42
        assert result.variant == "complete"
        assert "embedding" in result.arrays
        assert result.scalars["metric_a"] == 1.5

    def test_default_empty_dicts(self) -> None:
        result = AnalysisResult(
            method="test", config_hash="", seed=0, variant=""
        )
        assert result.arrays == {}
        assert result.scalars == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        original = AnalysisResult(
            method="test_method",
            config_hash="hash123",
            seed=42,
            variant="ablate_attractor",
            arrays={
                "embedding": np.random.randn(5, 3),
                "diagram": np.array([[0.0, 1.0], [0.5, 2.0]]),
            },
            scalars={"distance": 1.23, "dimension": 5.0},
        )
        base_path = tmp_path / "results" / "test"
        original.save(base_path)

        # Verify files exist
        assert (Path(str(base_path) + ".json")).exists()
        assert (Path(str(base_path) + ".npz")).exists()

        # Load and compare
        loaded = AnalysisResult.load(base_path)
        assert loaded.method == original.method
        assert loaded.config_hash == original.config_hash
        assert loaded.seed == original.seed
        assert loaded.variant == original.variant
        assert loaded.scalars == original.scalars
        for key in original.arrays:
            np.testing.assert_array_almost_equal(
                loaded.arrays[key], original.arrays[key]
            )

    def test_save_without_arrays(self, tmp_path: Path) -> None:
        result = AnalysisResult(
            method="scalar_only",
            config_hash="h",
            seed=0,
            variant="v",
            scalars={"val": 42.0},
        )
        base_path = tmp_path / "no_arrays"
        result.save(base_path)

        loaded = AnalysisResult.load(base_path)
        assert loaded.scalars["val"] == 42.0
        assert loaded.arrays == {}

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AnalysisResult.load(tmp_path / "nonexistent")

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        result = AnalysisResult(
            method="test", config_hash="h", seed=0, variant="v"
        )
        base_path = tmp_path / "deep" / "nested" / "path"
        result.save(base_path)
        assert (Path(str(base_path) + ".json")).exists()


class TestAnalysisMethodProtocol:
    """Tests for AnalysisMethod Protocol."""

    def test_class_satisfies_protocol(self) -> None:
        """A class with name and compute method satisfies AnalysisMethod."""

        class MockMethod:
            name: str = "mock"

            def compute(self, data: object) -> AnalysisResult:
                return AnalysisResult(
                    method=self.name, config_hash="", seed=0, variant=""
                )

        assert isinstance(MockMethod(), AnalysisMethod)

    def test_missing_name_fails_protocol(self) -> None:
        """A class without name attr does not satisfy AnalysisMethod."""

        class BadMethod:
            def compute(self, data: object) -> AnalysisResult:
                return AnalysisResult(
                    method="", config_hash="", seed=0, variant=""
                )

        assert not isinstance(BadMethod(), AnalysisMethod)

    def test_missing_compute_fails_protocol(self) -> None:
        """A class without compute method does not satisfy AnalysisMethod."""

        class NoCompute:
            name: str = "no_compute"

        assert not isinstance(NoCompute(), AnalysisMethod)
