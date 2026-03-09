"""Biological constraints for RNN connectivity.

Provides functions to enforce biologically plausible constraints on
recurrent weight matrices:

- ``enforce_dale_law``: Ensures each neuron is either excitatory or
  inhibitory (not both), following Dale's principle.
- ``apply_sparse_mask``: Creates and applies a fixed binary sparsity
  mask to recurrent weights, simulating sparse biological connectivity.
"""

from __future__ import annotations

import torch


def enforce_dale_law(
    weight: torch.Tensor,
    excitatory_mask: torch.Tensor,
) -> torch.Tensor:
    """Enforce Dale's law on a weight matrix.

    Dale's law states that each neuron releases the same neurotransmitter
    at all its synapses — it is either excitatory or inhibitory, not both.

    This is enforced by taking the absolute value of weights and then
    applying sign constraints: excitatory neurons have non-negative
    outgoing weights, inhibitory neurons have non-positive outgoing weights.

    Args:
        weight: Recurrent weight matrix, shape (hidden_size, hidden_size).
            Columns correspond to presynaptic (source) neurons.
        excitatory_mask: Boolean tensor, shape (hidden_size,). True for
            excitatory neurons, False for inhibitory neurons.

    Returns:
        Constrained weight matrix with Dale's law enforced.
    """
    # Take absolute value to get magnitude
    abs_weight = torch.abs(weight)
    # Create sign mask: +1 for excitatory, -1 for inhibitory
    sign = torch.where(excitatory_mask, torch.ones_like(excitatory_mask, dtype=weight.dtype),
                       -torch.ones_like(excitatory_mask, dtype=weight.dtype))
    # Apply sign to columns (presynaptic neuron determines sign)
    return abs_weight * sign.unsqueeze(0)


def create_sparse_mask(
    hidden_size: int,
    sparsity: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Create a fixed binary sparsity mask for recurrent weights.

    Args:
        hidden_size: Number of hidden units.
        sparsity: Fraction of connections to remove (0.0 = fully connected,
            1.0 = no connections). Values in [0.0, 1.0).
        rng: PyTorch random generator for deterministic mask creation.

    Returns:
        Binary mask tensor, shape (hidden_size, hidden_size). 1.0 where
        connections exist, 0.0 where pruned.

    Raises:
        ValueError: If sparsity is not in [0.0, 1.0).
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"Sparsity must be in [0.0, 1.0), got {sparsity}")

    if sparsity == 0.0:
        return torch.ones(hidden_size, hidden_size)

    # Generate uniform random values and threshold
    rand_vals = torch.rand(hidden_size, hidden_size, generator=rng)
    mask = (rand_vals >= sparsity).float()
    return mask


def apply_sparse_mask(
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a binary sparsity mask to a weight matrix.

    Args:
        weight: Weight matrix, shape (hidden_size, hidden_size).
        mask: Binary mask, shape (hidden_size, hidden_size).

    Returns:
        Masked weight matrix with zeroed-out connections.
    """
    return weight * mask


def create_excitatory_mask(
    hidden_size: int,
    excitatory_fraction: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Create a random excitatory/inhibitory neuron assignment.

    Args:
        hidden_size: Number of hidden units.
        excitatory_fraction: Fraction of neurons that are excitatory.
            Typical biological value: ~0.8 (80% excitatory).
        rng: PyTorch random generator for deterministic assignment.

    Returns:
        Boolean tensor, shape (hidden_size,). True for excitatory neurons.

    Raises:
        ValueError: If excitatory_fraction is not in (0.0, 1.0).
    """
    if not 0.0 < excitatory_fraction < 1.0:
        raise ValueError(
            f"excitatory_fraction must be in (0.0, 1.0), got {excitatory_fraction}"
        )
    rand_vals = torch.rand(hidden_size, generator=rng)
    return rand_vals < excitatory_fraction
