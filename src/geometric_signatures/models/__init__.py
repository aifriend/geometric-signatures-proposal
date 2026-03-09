"""Constrained RNN models with motif-specific layers.

Provides a modular RNN architecture where each computational motif
(normalization, attractor dynamics, selective gating, expansion recoding)
is implemented as a separate ``nn.Module`` that can be toggled on/off
via ``MotifSwitches``.

Usage::

    from geometric_signatures.models import ConstrainedRNN
    from geometric_signatures.config import ModelConfig
    from geometric_signatures.motifs import MotifSwitches

    config = ModelConfig(hidden_size=128, ...)
    motifs = MotifSwitches(...)
    model = ConstrainedRNN(config, motifs)
"""

from __future__ import annotations

from .constrained_rnn import ConstrainedRNN
from .constraints import apply_sparse_mask, enforce_dale_law
from .layers import (
    AttractorRecurrence,
    DivisiveNormalization,
    ExpansionRecoding,
    SelectiveGating,
)

__all__ = [
    "ConstrainedRNN",
    "enforce_dale_law",
    "apply_sparse_mask",
    "DivisiveNormalization",
    "AttractorRecurrence",
    "SelectiveGating",
    "ExpansionRecoding",
]
