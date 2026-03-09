"""Cross-system comparison with statistical testing.

Compares geometric signatures between any two systems (RNN variants,
biological datasets, or RNN vs. biological) using aggregated multi-seed
results with permutation tests and effect sizes.

Typical workflow::

    from geometric_signatures.comparison import compare_across_systems
    result = compare_across_systems(
        results_a=rnn_aggregated,
        results_b=bio_aggregated,
        methods=("persistent_homology", "cka"),
    )
    # result.significance contains per-metric permutation test results
    # result.effect_sizes contains Cohen's d per metric
"""

from .cross_system import (
    CrossSystemResult,
    compare_across_systems,
    identify_shared_signatures,
)

__all__ = [
    "CrossSystemResult",
    "compare_across_systems",
    "identify_shared_signatures",
]
