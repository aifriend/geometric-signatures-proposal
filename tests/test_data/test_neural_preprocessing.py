"""Tests for neural preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.data.neural_preprocessing import (
    align_trials,
    bin_spikes,
    compute_delta_f_over_f,
    normalize_population,
)


class TestBinSpikes:
    """Tests for spike binning."""

    def test_output_shapes(self) -> None:
        """Rates, bin_centers, cluster_ids have consistent shapes."""
        rng = np.random.default_rng(42)
        n_spikes = 500
        spike_times = np.sort(rng.uniform(0, 10, n_spikes))
        spike_clusters = rng.integers(0, 5, n_spikes)
        bin_size = 0.1

        rates, bin_centers, cluster_ids = bin_spikes(
            spike_times, spike_clusters, bin_size
        )

        assert rates.ndim == 2
        assert rates.shape[0] == len(bin_centers)
        assert rates.shape[1] == len(cluster_ids)

    def test_cluster_ids_sorted_unique(self) -> None:
        """Cluster IDs are sorted and unique."""
        spike_times = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
        spike_clusters = np.array([3, 1, 3, 1, 2])
        _, _, cluster_ids = bin_spikes(spike_times, spike_clusters, 0.5)

        assert list(cluster_ids) == [1, 2, 3]

    def test_rates_in_hz(self) -> None:
        """Rates should be counts / bin_size (Hz)."""
        # 10 spikes in cluster 0, all at t=0.05 (within first bin [0, 0.1))
        spike_times = np.full(10, 0.05)
        spike_clusters = np.zeros(10, dtype=int)
        bin_size = 0.1

        rates, _, _ = bin_spikes(spike_times, spike_clusters, bin_size)

        # 10 spikes in 0.1s bin = 100 Hz
        assert rates[0, 0] == pytest.approx(100.0)

    def test_custom_time_range(self) -> None:
        """t_start and t_end control the binning range."""
        spike_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spike_clusters = np.zeros(5, dtype=int)
        bin_size = 1.0

        rates, bin_centers, _ = bin_spikes(
            spike_times, spike_clusters, bin_size, t_start=0.0, t_end=6.0
        )

        assert bin_centers[0] == pytest.approx(0.5)
        assert bin_centers[-1] >= 5.0

    def test_no_spikes_in_bin_gives_zero(self) -> None:
        """Bins with no spikes have rate 0."""
        spike_times = np.array([5.0])
        spike_clusters = np.array([0])
        bin_size = 1.0

        rates, _, _ = bin_spikes(
            spike_times, spike_clusters, bin_size, t_start=0.0, t_end=10.0
        )

        # Most bins should be zero
        n_nonzero = np.count_nonzero(rates)
        assert n_nonzero == 1

    def test_multiple_clusters(self) -> None:
        """Each cluster gets its own column."""
        spike_times = np.array([0.5, 0.5, 0.5])
        spike_clusters = np.array([0, 1, 2])
        bin_size = 1.0

        rates, _, cluster_ids = bin_spikes(
            spike_times, spike_clusters, bin_size, t_start=0.0, t_end=1.0
        )

        assert len(cluster_ids) == 3
        assert rates.shape[1] == 3
        # Each cluster has 1 spike in 1s bin = 1 Hz
        np.testing.assert_allclose(rates[0], [1.0, 1.0, 1.0])


class TestComputeDeltaFOverF:
    """Tests for ΔF/F computation."""

    def test_output_shape_matches_input(self) -> None:
        """Output shape equals input shape."""
        fluorescence = np.random.default_rng(42).uniform(100, 200, (5, 100))
        dff = compute_delta_f_over_f(fluorescence)
        assert dff.shape == fluorescence.shape

    def test_constant_signal_gives_zero(self) -> None:
        """Constant fluorescence → ΔF/F ≈ 0."""
        fluorescence = np.full((3, 50), 150.0)
        dff = compute_delta_f_over_f(fluorescence)
        np.testing.assert_allclose(dff, 0.0, atol=1e-8)

    def test_baseline_percentile(self) -> None:
        """Lower percentile gives higher baseline estimate for noisy data."""
        rng = np.random.default_rng(42)
        fluorescence = rng.uniform(100, 200, (1, 1000))

        dff_10 = compute_delta_f_over_f(fluorescence, baseline_percentile=10.0)
        dff_50 = compute_delta_f_over_f(fluorescence, baseline_percentile=50.0)

        # 10th percentile baseline is lower → higher ΔF/F
        assert dff_10.mean() > dff_50.mean()

    def test_baseline_window(self) -> None:
        """baseline_window uses first N timepoints as baseline."""
        # Signal: 100 for first 10 timepoints, then 200
        fluorescence = np.concatenate([
            np.full((1, 10), 100.0),
            np.full((1, 90), 200.0),
        ], axis=1)

        dff = compute_delta_f_over_f(fluorescence, baseline_window=10)

        # First 10 timepoints: (100 - 100) / 100 = 0
        np.testing.assert_allclose(dff[0, :10], 0.0, atol=1e-8)
        # Remaining: (200 - 100) / 100 = 1.0
        np.testing.assert_allclose(dff[0, 10:], 1.0, atol=1e-8)

    def test_near_zero_baseline_handling(self) -> None:
        """Near-zero baseline doesn't produce inf/nan."""
        fluorescence = np.full((1, 50), 1e-15)
        dff = compute_delta_f_over_f(fluorescence)
        assert np.all(np.isfinite(dff))

    def test_multidimensional(self) -> None:
        """Works with higher-dimensional inputs."""
        fluorescence = np.random.default_rng(42).uniform(100, 200, (3, 5, 100))
        dff = compute_delta_f_over_f(fluorescence)
        assert dff.shape == (3, 5, 100)


class TestAlignTrials:
    """Tests for trial-aligned extraction."""

    def test_output_shape(self) -> None:
        """Output shape is (n_trials, n_window_bins, n_units)."""
        rng = np.random.default_rng(42)
        n_time = 1000
        n_units = 10
        dt = 0.01  # 10ms bins

        neural_data = rng.standard_normal((n_time, n_units))
        time_axis = np.arange(n_time) * dt
        event_times = np.array([1.0, 2.0, 3.0])
        window = (-0.5, 1.5)

        aligned, trial_time = align_trials(
            neural_data, time_axis, event_times, window
        )

        assert aligned.ndim == 3
        assert aligned.shape[0] == 3  # n_trials
        assert aligned.shape[2] == n_units
        assert len(trial_time) == aligned.shape[1]

    def test_trial_time_relative(self) -> None:
        """Trial time axis starts near window[0] and ends near window[1]."""
        neural_data = np.zeros((1000, 5))
        time_axis = np.arange(1000) * 0.01
        event_times = np.array([5.0])
        window = (-0.5, 1.5)

        _, trial_time = align_trials(
            neural_data, time_axis, event_times, window
        )

        assert trial_time[0] == pytest.approx(-0.5)
        assert trial_time[-1] < 1.5
        assert trial_time[-1] > 1.4  # close to window end

    def test_alignment_extracts_correct_data(self) -> None:
        """Data at event time should align to t=0 in trial time."""
        n_time = 100
        n_units = 2
        dt = 0.1

        # Create data with a spike at t=5.0
        neural_data = np.zeros((n_time, n_units))
        neural_data[50] = [10.0, 20.0]  # t = 50 * 0.1 = 5.0

        time_axis = np.arange(n_time) * dt
        event_times = np.array([5.0])
        window = (-1.0, 1.0)

        aligned, trial_time = align_trials(
            neural_data, time_axis, event_times, window
        )

        # Find the bin closest to t=0 (event onset)
        zero_idx = np.argmin(np.abs(trial_time))
        np.testing.assert_allclose(aligned[0, zero_idx], [10.0, 20.0])

    def test_multiple_events(self) -> None:
        """Multiple events produce multiple trials."""
        neural_data = np.ones((1000, 3))
        time_axis = np.arange(1000) * 0.01
        event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = (-0.2, 0.5)

        aligned, _ = align_trials(
            neural_data, time_axis, event_times, window
        )

        assert aligned.shape[0] == 5

    def test_clipping_at_boundaries(self) -> None:
        """Events near data boundaries don't raise — clips indices."""
        neural_data = np.ones((100, 2))
        time_axis = np.arange(100) * 0.01  # 0 to 0.99s
        event_times = np.array([0.0])  # Event at start
        window = (-0.5, 0.5)  # Goes before data start

        aligned, _ = align_trials(
            neural_data, time_axis, event_times, window
        )

        assert aligned.shape[0] == 1
        assert np.all(np.isfinite(aligned))


class TestNormalizePopulation:
    """Tests for population normalization."""

    def test_zscore_mean_zero(self) -> None:
        """Z-scored data should have near-zero mean per unit."""
        rng = np.random.default_rng(42)
        activity = rng.standard_normal((20, 50, 10)) * 5 + 3

        normed = normalize_population(activity, method="zscore")

        # Mean across non-unit axes should be ~0
        means = normed.mean(axis=(0, 1))
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_zscore_unit_std(self) -> None:
        """Z-scored data should have unit std per unit."""
        rng = np.random.default_rng(42)
        activity = rng.standard_normal((20, 50, 10)) * 5 + 3

        normed = normalize_population(activity, method="zscore")

        stds = normed.std(axis=(0, 1))
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_max_normalization(self) -> None:
        """Max-normalized data has max absolute value 1.0 per unit."""
        rng = np.random.default_rng(42)
        activity = rng.standard_normal((20, 50, 10)) * 5

        normed = normalize_population(activity, method="max")

        max_abs = np.max(np.abs(normed), axis=(0, 1))
        np.testing.assert_allclose(max_abs, 1.0, atol=1e-10)

    def test_range_normalization(self) -> None:
        """Range-normalized data is in [0, 1] per unit."""
        rng = np.random.default_rng(42)
        activity = rng.standard_normal((20, 50, 10)) * 5 + 3

        normed = normalize_population(activity, method="range")

        mins = normed.min(axis=(0, 1))
        maxs = normed.max(axis=(0, 1))
        np.testing.assert_allclose(mins, 0.0, atol=1e-10)
        np.testing.assert_allclose(maxs, 1.0, atol=1e-10)

    def test_unknown_method_raises(self) -> None:
        """Unknown normalization method raises ValueError."""
        activity = np.ones((5, 10, 3))
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_population(activity, method="invalid")

    def test_constant_unit_handling(self) -> None:
        """Constant units (zero variance) don't produce inf/nan."""
        activity = np.ones((5, 10, 3))
        activity[:, :, 1] = 5.0  # Different constant

        for method in ("zscore", "max", "range"):
            normed = normalize_population(activity, method=method)
            assert np.all(np.isfinite(normed)), f"Non-finite values for {method}"

    def test_output_shape_preserved(self) -> None:
        """Output shape matches input shape."""
        activity = np.random.default_rng(42).standard_normal((8, 20, 5))

        for method in ("zscore", "max", "range"):
            normed = normalize_population(activity, method=method)
            assert normed.shape == activity.shape
