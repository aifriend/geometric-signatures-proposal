"""Tests for multiple comparison correction methods."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.statistics.correction import (
    bonferroni_correction,
    fdr_correction,
    holm_correction,
)


class TestFDRCorrection:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_no_rejections_for_large_p_values(self) -> None:
        p = np.array([0.5, 0.6, 0.7, 0.8])
        rejected, _ = fdr_correction(p)
        assert not np.any(rejected)

    def test_rejects_small_p_values(self) -> None:
        p = np.array([0.001, 0.002, 0.8, 0.9])
        rejected, _ = fdr_correction(p, alpha=0.05)
        assert rejected[0] and rejected[1]
        assert not rejected[2] and not rejected[3]

    def test_corrected_p_in_range(self) -> None:
        p = np.array([0.01, 0.04, 0.08, 0.5])
        _, corrected = fdr_correction(p)
        assert np.all(corrected >= 0.0)
        assert np.all(corrected <= 1.0)

    def test_corrected_p_monotone(self) -> None:
        """Corrected p-values should be monotonized."""
        p = np.array([0.01, 0.03, 0.04, 0.05])
        _, corrected = fdr_correction(p)
        sorted_idx = np.argsort(p)
        sorted_corrected = corrected[sorted_idx]
        # Non-decreasing when sorted by original p-values
        for i in range(len(sorted_corrected) - 1):
            assert sorted_corrected[i] <= sorted_corrected[i + 1] + 1e-10

    def test_single_p_value(self) -> None:
        p = np.array([0.03])
        rejected, corrected = fdr_correction(p, alpha=0.05)
        assert rejected[0]
        np.testing.assert_allclose(corrected[0], 0.03)

    def test_empty_input(self) -> None:
        rejected, corrected = fdr_correction(np.array([]))
        assert len(rejected) == 0
        assert len(corrected) == 0

    def test_all_significant(self) -> None:
        p = np.array([0.001, 0.002, 0.003])
        rejected, _ = fdr_correction(p, alpha=0.05)
        assert np.all(rejected)

    def test_corrected_ge_original(self) -> None:
        """Corrected p-values should be >= original."""
        p = np.array([0.01, 0.03, 0.04, 0.5])
        _, corrected = fdr_correction(p)
        assert np.all(corrected >= p - 1e-10)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p-values"):
            fdr_correction(np.array([-0.1, 0.5]))
        with pytest.raises(ValueError, match="p-values"):
            fdr_correction(np.array([0.5, 1.5]))


class TestBonferroniCorrection:
    """Tests for Bonferroni FWER correction."""

    def test_multiplies_by_n(self) -> None:
        p = np.array([0.01, 0.02, 0.03])
        _, corrected = bonferroni_correction(p)
        np.testing.assert_allclose(corrected, [0.03, 0.06, 0.09])

    def test_capped_at_one(self) -> None:
        p = np.array([0.01, 0.5, 0.8])
        _, corrected = bonferroni_correction(p)
        assert np.all(corrected <= 1.0)

    def test_more_conservative_than_fdr(self) -> None:
        """Bonferroni should reject fewer hypotheses than FDR."""
        rng = np.random.default_rng(42)
        # Mix of significant and non-significant
        p = np.concatenate([
            rng.uniform(0.001, 0.01, size=5),
            rng.uniform(0.05, 0.5, size=15),
        ])
        fdr_rejected, _ = fdr_correction(p, alpha=0.05)
        bonf_rejected, _ = bonferroni_correction(p, alpha=0.05)
        assert bonf_rejected.sum() <= fdr_rejected.sum()

    def test_single_p_value_unchanged(self) -> None:
        p = np.array([0.03])
        _, corrected = bonferroni_correction(p)
        np.testing.assert_allclose(corrected[0], 0.03)

    def test_empty_input(self) -> None:
        rejected, corrected = bonferroni_correction(np.array([]))
        assert len(rejected) == 0
        assert len(corrected) == 0

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p-values"):
            bonferroni_correction(np.array([-0.1, 0.5]))


class TestHolmCorrection:
    """Tests for Holm step-down FWER correction."""

    def test_less_conservative_than_bonferroni(self) -> None:
        """Holm should reject at least as many as Bonferroni."""
        rng = np.random.default_rng(42)
        p = np.concatenate([
            rng.uniform(0.001, 0.005, size=5),
            rng.uniform(0.1, 0.5, size=15),
        ])
        bonf_rejected, _ = bonferroni_correction(p, alpha=0.05)
        holm_rejected, _ = holm_correction(p, alpha=0.05)
        assert holm_rejected.sum() >= bonf_rejected.sum()

    def test_corrected_p_in_range(self) -> None:
        p = np.array([0.01, 0.02, 0.03, 0.5])
        _, corrected = holm_correction(p)
        assert np.all(corrected >= 0.0)
        assert np.all(corrected <= 1.0)

    def test_corrected_ge_original(self) -> None:
        p = np.array([0.01, 0.02, 0.5, 0.8])
        _, corrected = holm_correction(p)
        assert np.all(corrected >= p - 1e-10)

    def test_single_p_value_unchanged(self) -> None:
        p = np.array([0.03])
        _, corrected = holm_correction(p)
        np.testing.assert_allclose(corrected[0], 0.03)

    def test_empty_input(self) -> None:
        rejected, corrected = holm_correction(np.array([]))
        assert len(rejected) == 0
        assert len(corrected) == 0

    def test_all_significant(self) -> None:
        p = np.array([0.001, 0.002, 0.003])
        rejected, _ = holm_correction(p, alpha=0.05)
        assert np.all(rejected)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p-values"):
            holm_correction(np.array([0.5, 1.5]))

    def test_monotone_corrected(self) -> None:
        """Corrected p-values (in sorted order) should be non-decreasing."""
        p = np.array([0.01, 0.02, 0.03, 0.05])
        _, corrected = holm_correction(p)
        sorted_idx = np.argsort(p)
        sorted_corrected = corrected[sorted_idx]
        for i in range(len(sorted_corrected) - 1):
            assert sorted_corrected[i] <= sorted_corrected[i + 1] + 1e-10
