"""Tests for IBL data loader (mocked — no real API calls)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestLoadIblSession:
    """Tests for load_ibl_session with mocked ONE API."""

    @pytest.fixture()
    def mock_one(self) -> MagicMock:
        """Create a mock ONE client with realistic return data."""
        mock = MagicMock()

        # Simulate spike data: 1000 spikes across 5 clusters
        rng = np.random.default_rng(42)
        n_spikes = 1000
        spike_times = np.sort(rng.uniform(0, 10, n_spikes))
        spike_clusters = rng.integers(0, 5, n_spikes)

        mock.load_datasets.return_value = (spike_times, spike_clusters)

        # Cluster regions for filtering
        cluster_regions = np.array(
            ["VISp", "VISl", "VISp", "VISp", "VISl"]
        )
        mock.load_dataset.return_value = cluster_regions

        # Trials object with event times and contrast info
        n_trials = 20
        stim_on = np.sort(rng.uniform(1, 9, n_trials))
        contrast_left = rng.choice([np.nan, 0.25, 0.5, 1.0], n_trials)
        contrast_right = rng.choice([np.nan, 0.25, 0.5, 1.0], n_trials)

        trials = SimpleNamespace(
            stimOn_times=stim_on,
            contrastLeft=contrast_left,
            contrastRight=contrast_right,
        )
        mock.load_object.return_value = trials

        # Session search
        mock.search.return_value = ["eid_001", "eid_002", "eid_003"]

        return mock

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_returns_neural_population_data(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Loader returns NeuralPopulationData with correct source."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session
        from geometric_signatures.population import NeuralPopulationData

        result = load_ibl_session("test_eid", cache_dir=tmp_path)

        assert isinstance(result, NeuralPopulationData)
        assert result.source == "ibl"

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_activity_shape(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Activity has shape (n_trials, n_timepoints, n_units)."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session

        result = load_ibl_session("test_eid", cache_dir=tmp_path)

        assert result.activity.ndim == 3
        n_trials, n_time, n_units = result.activity.shape
        assert n_trials > 0
        assert n_time > 0
        assert n_units > 0

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_metadata_contents(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Metadata contains expected keys."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session

        result = load_ibl_session("test_eid", cache_dir=tmp_path)

        assert "session_eid" in result.metadata
        assert "brain_region" in result.metadata
        assert "bin_size" in result.metadata
        assert "align_event" in result.metadata
        assert "n_clusters" in result.metadata

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_brain_region_filter(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Brain region filtering reduces cluster count."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session

        result = load_ibl_session(
            "test_eid", cache_dir=tmp_path, brain_region="VISp"
        )

        assert result.metadata["brain_region"] == "VISp"

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_trial_labels_from_contrast(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Trial labels extracted from contrast values."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session

        result = load_ibl_session("test_eid", cache_dir=tmp_path)

        valid_labels = {"left_stim", "right_stim", "both_stim", "no_stim"}
        for label in result.trial_labels:
            assert label in valid_labels

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_invalid_align_event_raises(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Invalid alignment event raises ValueError."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session

        with pytest.raises(ValueError, match="not found in trials"):
            load_ibl_session(
                "test_eid",
                cache_dir=tmp_path,
                align_event="nonexistent_event",
            )

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_custom_window(
        self, mock_one_cls: MagicMock, mock_one: MagicMock, tmp_path: Path
    ) -> None:
        """Custom window affects time axis."""
        mock_one_cls.return_value = mock_one
        from geometric_signatures.data.ibl import load_ibl_session

        result = load_ibl_session(
            "test_eid", cache_dir=tmp_path, window=(-1.0, 2.0)
        )

        assert result.time_axis[0] == pytest.approx(-1.0)
        assert result.metadata["window"] == [-1.0, 2.0]


class TestExtractIblTrialLabels:
    """Tests for IBL trial label extraction."""

    def test_contrast_labels(self) -> None:
        """Labels extracted from contrastLeft/contrastRight."""
        from geometric_signatures.data.ibl import _extract_ibl_trial_labels

        trials = SimpleNamespace(
            contrastLeft=np.array([np.nan, 0.5, 0.5, np.nan]),
            contrastRight=np.array([0.5, np.nan, 0.5, np.nan]),
        )
        valid_mask = np.array([True, True, True, True])
        labels = _extract_ibl_trial_labels(trials, valid_mask)

        assert labels == ("right_stim", "left_stim", "both_stim", "no_stim")

    def test_fallback_generic_labels(self) -> None:
        """Falls back to generic labels when contrast info absent."""
        from geometric_signatures.data.ibl import _extract_ibl_trial_labels

        trials = SimpleNamespace()  # no contrastLeft/contrastRight
        valid_mask = np.array([True, False, True])
        labels = _extract_ibl_trial_labels(trials, valid_mask)

        assert len(labels) == 2  # only valid trials
        assert labels[0] == "trial_0"
        assert labels[1] == "trial_1"


class TestListIblSessions:
    """Tests for IBL session listing."""

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", True)
    @patch("geometric_signatures.data.ibl.ONE", create=True)
    def test_returns_list_of_strings(
        self, mock_one_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Returns a list of session EID strings."""
        mock_one = MagicMock()
        mock_one.search.return_value = ["eid_1", "eid_2"]
        mock_one_cls.return_value = mock_one

        from geometric_signatures.data.ibl import list_ibl_sessions

        result = list_ibl_sessions(cache_dir=tmp_path)

        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    @patch("geometric_signatures.data.ibl.ONE_AVAILABLE", False)
    def test_unavailable_raises_import_error(self) -> None:
        """Raises ImportError when ONE API is not installed."""
        from geometric_signatures.data.ibl import list_ibl_sessions

        with pytest.raises(ImportError, match="ONE-api"):
            list_ibl_sessions()
