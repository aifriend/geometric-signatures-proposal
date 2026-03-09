"""Tests for biological constraints (Dale's law, sparsity)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geometric_signatures.models.constraints import (
    apply_sparse_mask,
    create_excitatory_mask,
    create_sparse_mask,
    enforce_dale_law,
)

HIDDEN = 16


class TestDaleLaw:
    """Tests for enforce_dale_law."""

    def test_excitatory_columns_non_negative(self) -> None:
        W = torch.randn(HIDDEN, HIDDEN)
        exc_mask = torch.ones(HIDDEN, dtype=torch.bool)  # all excitatory
        constrained = enforce_dale_law(W, exc_mask)
        assert (constrained >= 0).all()

    def test_inhibitory_columns_non_positive(self) -> None:
        W = torch.randn(HIDDEN, HIDDEN)
        exc_mask = torch.zeros(HIDDEN, dtype=torch.bool)  # all inhibitory
        constrained = enforce_dale_law(W, exc_mask)
        assert (constrained <= 0).all()

    def test_mixed_population(self) -> None:
        W = torch.randn(HIDDEN, HIDDEN)
        exc_mask = torch.tensor([True, False] * (HIDDEN // 2))
        constrained = enforce_dale_law(W, exc_mask)
        # Excitatory columns (True) should be non-negative
        assert (constrained[:, exc_mask] >= 0).all()
        # Inhibitory columns (False) should be non-positive
        assert (constrained[:, ~exc_mask] <= 0).all()

    def test_preserves_magnitude(self) -> None:
        W = torch.randn(HIDDEN, HIDDEN)
        exc_mask = torch.ones(HIDDEN, dtype=torch.bool)
        constrained = enforce_dale_law(W, exc_mask)
        assert torch.allclose(constrained.abs(), W.abs())


class TestSparseMask:
    """Tests for create_sparse_mask and apply_sparse_mask."""

    def test_zero_sparsity_fully_connected(self) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = create_sparse_mask(HIDDEN, 0.0, rng)
        assert (mask == 1.0).all()

    def test_high_sparsity_mostly_zero(self) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = create_sparse_mask(HIDDEN, 0.9, rng)
        fraction_nonzero = mask.sum().item() / mask.numel()
        # Should be approximately 10% nonzero (with some variance)
        assert 0.01 < fraction_nonzero < 0.3

    def test_sparsity_rejects_invalid(self) -> None:
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="Sparsity must be"):
            create_sparse_mask(HIDDEN, 1.0, rng)
        with pytest.raises(ValueError, match="Sparsity must be"):
            create_sparse_mask(HIDDEN, -0.1, rng)

    def test_mask_deterministic(self) -> None:
        rng1 = torch.Generator().manual_seed(42)
        rng2 = torch.Generator().manual_seed(42)
        mask1 = create_sparse_mask(HIDDEN, 0.5, rng1)
        mask2 = create_sparse_mask(HIDDEN, 0.5, rng2)
        assert torch.equal(mask1, mask2)

    def test_apply_sparse_mask(self) -> None:
        W = torch.ones(HIDDEN, HIDDEN)
        mask = torch.zeros(HIDDEN, HIDDEN)
        mask[0, 0] = 1.0
        result = apply_sparse_mask(W, mask)
        assert result[0, 0] == 1.0
        assert result.sum() == 1.0


class TestExcitatoryMask:
    """Tests for create_excitatory_mask."""

    def test_fraction_approximately_correct(self) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = create_excitatory_mask(1000, 0.8, rng)
        frac = mask.sum().item() / 1000
        assert 0.7 < frac < 0.9  # ~80% ± some variance

    def test_deterministic(self) -> None:
        rng1 = torch.Generator().manual_seed(42)
        rng2 = torch.Generator().manual_seed(42)
        mask1 = create_excitatory_mask(HIDDEN, 0.8, rng1)
        mask2 = create_excitatory_mask(HIDDEN, 0.8, rng2)
        assert torch.equal(mask1, mask2)

    def test_rejects_invalid_fraction(self) -> None:
        rng = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="excitatory_fraction"):
            create_excitatory_mask(HIDDEN, 0.0, rng)
        with pytest.raises(ValueError, match="excitatory_fraction"):
            create_excitatory_mask(HIDDEN, 1.0, rng)

    def test_returns_bool_tensor(self) -> None:
        rng = torch.Generator().manual_seed(0)
        mask = create_excitatory_mask(HIDDEN, 0.8, rng)
        assert mask.dtype == torch.bool
