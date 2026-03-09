"""Motif-specific neural network layers.

Each computational motif from the proposal is implemented as an ``nn.Module``
that transforms hidden state activations. When a motif is disabled (via
``MotifSwitches``), the corresponding layer acts as an identity function.

Motif layers:
    - ``DivisiveNormalization``: Gain modulation via divisive normalization.
    - ``AttractorRecurrence``: Attractor dynamics through recurrent amplification.
    - ``SelectiveGating``: Selective gating of information flow.
    - ``ExpansionRecoding``: Dimensionality expansion and nonlinear recoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DivisiveNormalization(nn.Module):
    """Divisive normalization / gain modulation motif.

    Implements a learnable divisive normalization where each unit's
    activity is divided by a weighted sum of pool activity plus a
    semi-saturation constant. This captures the normalization and
    gain modulation motif from the proposal.

    .. math::
        y_i = \\frac{x_i}{\\sigma + \\sum_j w_{ij} x_j^2}

    where :math:`\\sigma` is a learnable semi-saturation constant.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Learnable normalization pool weights
        self.pool_weights = nn.Parameter(
            torch.ones(hidden_size) / hidden_size
        )
        # Semi-saturation constant (learnable, initialized to prevent division by zero)
        self.sigma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply divisive normalization.

        Args:
            x: Hidden state tensor, shape (..., hidden_size).

        Returns:
            Normalized tensor, same shape as input.
        """
        # Pool activity: weighted sum of squared activations
        pool = torch.sum(self.pool_weights * x * x, dim=-1, keepdim=True)
        # Divisive normalization with semi-saturation constant
        denominator = torch.abs(self.sigma) + pool
        return x / denominator


class AttractorRecurrence(nn.Module):
    """Attractor dynamics motif.

    Implements a low-rank recurrent interaction that creates attractor
    states in the hidden activity space. The attractor dynamics are
    modeled as a symmetric low-rank perturbation to the recurrent
    connectivity.

    .. math::
        y = x + \\alpha \\cdot U U^T x

    where :math:`U` is a low-rank basis and :math:`\\alpha` controls
    the attractor strength.
    """

    def __init__(self, hidden_size: int, rank: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        # Low-rank attractor basis
        self.U = nn.Parameter(
            torch.randn(hidden_size, rank) / (hidden_size ** 0.5)
        )
        # Attractor strength
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attractor dynamics.

        Args:
            x: Hidden state tensor, shape (..., hidden_size).

        Returns:
            Tensor with attractor perturbation, same shape as input.
        """
        # Project to low-rank space and back: U @ U^T @ x
        proj = x @ self.U  # (..., rank)
        reconstruction = proj @ self.U.t()  # (..., hidden_size)
        result: torch.Tensor = x + self.alpha * reconstruction
        return result


class SelectiveGating(nn.Module):
    """Selective gating motif.

    Implements a learned gating mechanism that selectively amplifies
    or suppresses activity in different hidden units. This captures
    the selective gating motif from the proposal.

    .. math::
        y = x \\odot \\sigma(W_{gate} x + b_{gate})

    where :math:`\\sigma` is the sigmoid function and :math:`\\odot`
    denotes element-wise multiplication.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Linear(hidden_size, hidden_size)
        # Initialize gate bias to positive values so gates start ~open
        nn.init.constant_(self.gate.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selective gating.

        Args:
            x: Hidden state tensor, shape (..., hidden_size).

        Returns:
            Gated tensor, same shape as input.
        """
        gate_values = torch.sigmoid(self.gate(x))
        return x * gate_values


class ExpansionRecoding(nn.Module):
    """Expansion and nonlinear recoding motif.

    Implements a transient dimensionality expansion followed by
    nonlinear compression back to the original dimensionality.
    This captures mixed selectivity and expansion recoding.

    .. math::
        y = W_{down} \\cdot \\text{ReLU}(W_{up} x + b_{up}) + x

    The expansion factor controls the ratio of expanded to original
    dimensionality. A residual connection preserves the input.
    """

    def __init__(self, hidden_size: int, expansion_factor: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        expanded = hidden_size * expansion_factor
        self.expand = nn.Linear(hidden_size, expanded)
        self.compress = nn.Linear(expanded, hidden_size)
        # Initialize compress to small values so residual dominates initially
        nn.init.xavier_uniform_(self.compress.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply expansion recoding with residual connection.

        Args:
            x: Hidden state tensor, shape (..., hidden_size).

        Returns:
            Recoded tensor, same shape as input.
        """
        expanded = torch.relu(self.expand(x))
        compressed = self.compress(expanded)
        result: torch.Tensor = x + compressed
        return result
