"""Tests for Allen Brain Observatory data loader (mocked — no real API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestLoadAllenExperiment:
    """Tests for load_allen_experiment with mocked AllenSDK."""

    @pytest.fixture()
    def mock_cache(self) -> MagicMock:
        """Create a mock AllenSDK cache with realistic return data."""
        rng = np.random.default_rng(42)
        n_cells = 10
        n_timepoints = 5000
        dt = 1.0 / 30.0  # ~30 Hz

        # Mock experiment object
        experiment = MagicMock()

        # dff_traces: DataFrame with cell IDs as index, timestamps as columns
        dff_data = rng.standard_normal((n_cells, n_timepoints))
        cell_ids = list(range(100, 100 + n_cells))
        experiment.dff_traces = pd.DataFrame(
            dff_data, index=cell_ids
        )

        # ophys_timestamps
        experiment.ophys_timestamps = np.arange(n_timepoints) * dt

        # stimulus_presentations
        n_stim = 30
        start_times = np.sort(rng.uniform(5, 150, n_stim))
        stim_names = rng.choice(
            ["natural_scenes", "gabors", "static_gratings"], n_stim
        )
        experiment.stimulus_presentations = pd.DataFrame({
            "start_time": start_times,
            "stimulus_name": stim_names,
        })

        # Mock cache
        cache = MagicMock()
        cache.get_behavior_ophys_experiment.return_value = experiment

        # Experiment table for listing
        exp_table = pd.DataFrame(
            {"targeted_structure": ["VISp", "VISl", "VISp"]},
            index=[111, 222, 333],
        )
        cache.get_ophys_experiment_table.return_value = exp_table

        return cache

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_returns_neural_population_data(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Loader returns NeuralPopulationData with correct source."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment
        from geometric_signatures.population import NeuralPopulationData

        result = load_allen_experiment(123, cache_dir=tmp_path)

        assert isinstance(result, NeuralPopulationData)
        assert result.source == "allen"

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_activity_shape(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Activity has shape (n_trials, n_timepoints, n_units)."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment

        result = load_allen_experiment(123, cache_dir=tmp_path)

        assert result.activity.ndim == 3
        n_trials, n_time, n_units = result.activity.shape
        assert n_trials > 0
        assert n_time > 0
        assert n_units > 0

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_metadata_contents(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Metadata contains expected keys."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment

        result = load_allen_experiment(123, cache_dir=tmp_path)

        assert "experiment_id" in result.metadata
        assert "stimulus_name" in result.metadata
        assert "n_cells" in result.metadata
        assert "window" in result.metadata

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_stimulus_filter(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Stimulus name filter reduces trial count."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment

        result_all = load_allen_experiment(123, cache_dir=tmp_path)
        result_filtered = load_allen_experiment(
            123, cache_dir=tmp_path, stimulus_name="natural_scenes"
        )

        # Filtered should have fewer or equal trials
        assert result_filtered.activity.shape[0] <= result_all.activity.shape[0]
        assert result_filtered.metadata["stimulus_name"] == "natural_scenes"

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_invalid_stimulus_raises(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Invalid stimulus name raises ValueError."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache

        # Override stimulus_presentations with a name that won't match
        exp = mock_cache.get_behavior_ophys_experiment.return_value
        exp.stimulus_presentations = pd.DataFrame({
            "start_time": [1.0],
            "stimulus_name": ["gabors"],
        })

        from geometric_signatures.data.allen import load_allen_experiment

        with pytest.raises(ValueError, match="No presentations found"):
            load_allen_experiment(
                123, cache_dir=tmp_path, stimulus_name="nonexistent_stim"
            )

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_unit_labels_format(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Unit labels follow cell_<id> format."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment

        result = load_allen_experiment(123, cache_dir=tmp_path)

        for label in result.unit_labels:
            assert label.startswith("cell_")

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_custom_window(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Custom window affects time axis and metadata."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment

        result = load_allen_experiment(
            123, cache_dir=tmp_path, window=(-1.0, 3.0)
        )

        assert result.time_axis[0] == pytest.approx(-1.0)
        assert result.metadata["window"] == [-1.0, 3.0]

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_no_normalize(
        self,
        mock_cache_cls: MagicMock,
        mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        """normalize='none' skips normalization."""
        mock_cache_cls.from_s3_cache.return_value = mock_cache
        from geometric_signatures.data.allen import load_allen_experiment

        # Should not raise
        result = load_allen_experiment(
            123, cache_dir=tmp_path, normalize="none"
        )

        assert result.activity.ndim == 3


class TestListAllenExperiments:
    """Tests for Allen experiment listing."""

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_returns_list_of_ints(
        self, mock_cache_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Returns a list of experiment ID integers."""
        mock_cache = MagicMock()
        exp_table = pd.DataFrame(
            {"targeted_structure": ["VISp", "VISl"]},
            index=[111, 222],
        )
        mock_cache.get_ophys_experiment_table.return_value = exp_table
        mock_cache_cls.from_s3_cache.return_value = mock_cache

        from geometric_signatures.data.allen import list_allen_experiments

        result = list_allen_experiments(cache_dir=tmp_path)

        assert isinstance(result, list)
        assert all(isinstance(i, (int, np.integer)) for i in result)
        assert 111 in result
        assert 222 in result

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", True)
    @patch(
        "geometric_signatures.data.allen.VisualBehaviorOphysProjectCache",
        create=True,
    )
    def test_area_filter(
        self, mock_cache_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Area filter reduces experiment list."""
        mock_cache = MagicMock()
        exp_table = pd.DataFrame(
            {"targeted_structure": ["VISp", "VISl", "VISp"]},
            index=[111, 222, 333],
        )
        mock_cache.get_ophys_experiment_table.return_value = exp_table
        mock_cache_cls.from_s3_cache.return_value = mock_cache

        from geometric_signatures.data.allen import list_allen_experiments

        result = list_allen_experiments(area="VISp", cache_dir=tmp_path)

        assert 111 in result
        assert 333 in result
        assert 222 not in result

    @patch("geometric_signatures.data.allen.ALLENSDK_AVAILABLE", False)
    def test_unavailable_raises_import_error(self) -> None:
        """Raises ImportError when AllenSDK is not installed."""
        from geometric_signatures.data.allen import list_allen_experiments

        with pytest.raises(ImportError, match="allensdk"):
            list_allen_experiments()
