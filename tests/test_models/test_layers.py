"""Tests for motif-specific neural network layers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geometric_signatures.models.layers import (
    AttractorRecurrence,
    DivisiveNormalization,
    ExpansionRecoding,
    SelectiveGating,
)

HIDDEN = 16
BATCH = 4


class TestDivisiveNormalization:
    """Tests for DivisiveNormalization layer."""

    def test_output_shape(self) -> None:
        layer = DivisiveNormalization(HIDDEN)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, HIDDEN)

    def test_preserves_sign(self) -> None:
        """Normalization should preserve sign of activations."""
        layer = DivisiveNormalization(HIDDEN)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        # Check sign is preserved for non-zero elements
        nonzero = x.abs() > 0.01
        assert ((x[nonzero] > 0) == (y[nonzero] > 0)).all()

    def test_reduces_magnitude(self) -> None:
        """Normalization should reduce magnitude of large activations."""
        layer = DivisiveNormalization(HIDDEN)
        x = torch.ones(1, HIDDEN) * 10.0
        y = layer(x)
        assert y.abs().max() < x.abs().max()

    def test_learnable_parameters(self) -> None:
        layer = DivisiveNormalization(HIDDEN)
        params = list(layer.parameters())
        assert len(params) == 2  # pool_weights, sigma

    def test_3d_input(self) -> None:
        """Should work with (batch, time, hidden) tensors."""
        layer = DivisiveNormalization(HIDDEN)
        x = torch.randn(BATCH, 10, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, 10, HIDDEN)


class TestAttractorRecurrence:
    """Tests for AttractorRecurrence layer."""

    def test_output_shape(self) -> None:
        layer = AttractorRecurrence(HIDDEN, rank=4)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, HIDDEN)

    def test_residual_connection(self) -> None:
        """With alpha=0, output should equal input."""
        layer = AttractorRecurrence(HIDDEN, rank=4)
        with torch.no_grad():
            layer.alpha.fill_(0.0)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert torch.allclose(y, x)

    def test_modifies_input(self) -> None:
        """With non-zero alpha, output should differ from input."""
        layer = AttractorRecurrence(HIDDEN, rank=4)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert not torch.equal(y, x)

    def test_custom_rank(self) -> None:
        layer = AttractorRecurrence(HIDDEN, rank=2)
        assert layer.U.shape == (HIDDEN, 2)

    def test_3d_input(self) -> None:
        layer = AttractorRecurrence(HIDDEN)
        x = torch.randn(BATCH, 10, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, 10, HIDDEN)


class TestSelectiveGating:
    """Tests for SelectiveGating layer."""

    def test_output_shape(self) -> None:
        layer = SelectiveGating(HIDDEN)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, HIDDEN)

    def test_output_bounded(self) -> None:
        """Gating with sigmoid should keep output bounded."""
        layer = SelectiveGating(HIDDEN)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        # |y| should not exceed |x| since gate is in [0, 1]
        assert (y.abs() <= x.abs() + 1e-6).all()

    def test_learnable_parameters(self) -> None:
        layer = SelectiveGating(HIDDEN)
        params = list(layer.parameters())
        assert len(params) == 2  # gate.weight, gate.bias

    def test_3d_input(self) -> None:
        layer = SelectiveGating(HIDDEN)
        x = torch.randn(BATCH, 10, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, 10, HIDDEN)


class TestExpansionRecoding:
    """Tests for ExpansionRecoding layer."""

    def test_output_shape(self) -> None:
        layer = ExpansionRecoding(HIDDEN, expansion_factor=2)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, HIDDEN)

    def test_residual_connection(self) -> None:
        """Output should be close to input when compress weights are small."""
        layer = ExpansionRecoding(HIDDEN, expansion_factor=2)
        with torch.no_grad():
            layer.compress.weight.fill_(0.0)
            layer.compress.bias.fill_(0.0)
        x = torch.randn(BATCH, HIDDEN)
        y = layer(x)
        assert torch.allclose(y, x)

    def test_custom_expansion_factor(self) -> None:
        layer = ExpansionRecoding(HIDDEN, expansion_factor=4)
        assert layer.expand.out_features == HIDDEN * 4
        assert layer.compress.in_features == HIDDEN * 4

    def test_3d_input(self) -> None:
        layer = ExpansionRecoding(HIDDEN)
        x = torch.randn(BATCH, 10, HIDDEN)
        y = layer(x)
        assert y.shape == (BATCH, 10, HIDDEN)
