"""Cross-system comparison of geometric signatures.

Compares aggregated analysis results between any two systems (e.g.,
RNN variants, biological recordings, or RNN vs. biological) with
statistical testing and effect size estimation.

The comparison is metric-by-metric: for each scalar metric that exists
in both systems' aggregated results, a permutation test is run on the
per-seed values, and Cohen's d effect size is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..statistics.aggregation import AggregatedResult
from ..statistics.bootstrap import effect_size_cohens_d
from ..statistics.permutation import PermutationTestResult, permutation_test


@dataclass(frozen=True)
class CrossSystemResult:
    """Result of comparing geometric signatures across two systems.

    Attributes:
        system_a: Identifier for the first system (e.g., "rnn_complete").
        system_b: Identifier for the second system (e.g., "ibl_VISp").
        methods: Analysis methods compared.
        metric_similarities: Per-metric similarity score (1 - |effect_size|,
            clamped to [0, 1]).
        significance: Per-metric permutation test results.
        effect_sizes: Per-metric Cohen's d effect sizes.
        shared_signatures: Metrics where the two systems are statistically
            similar (p >= alpha).
        divergent_signatures: Metrics where the two systems differ
            significantly (p < alpha).
    """

    system_a: str
    system_b: str
    methods: tuple[str, ...]
    metric_similarities: dict[str, float]
    significance: dict[str, PermutationTestResult]
    effect_sizes: dict[str, float]
    shared_signatures: tuple[str, ...]
    divergent_signatures: tuple[str, ...]


def compare_across_systems(
    results_a: dict[str, AggregatedResult],
    results_b: dict[str, AggregatedResult],
    methods: Sequence[str] | None = None,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    system_a_name: str = "system_a",
    system_b_name: str = "system_b",
    rng: np.random.Generator | None = None,
) -> CrossSystemResult:
    """Compare geometric signatures between two systems.

    For each analysis method and each scalar metric, runs a permutation
    test on the per-seed values and computes Cohen's d effect size.

    Args:
        results_a: Aggregated results for system A, keyed by method name.
        results_b: Aggregated results for system B, keyed by method name.
        methods: Analysis methods to compare. If None, uses all methods
            present in both result dicts.
        n_permutations: Number of permutations for statistical tests.
        alpha: Significance threshold for classifying shared vs. divergent.
        system_a_name: Label for system A in the result.
        system_b_name: Label for system B in the result.
        rng: Numpy random generator for reproducibility.

    Returns:
        CrossSystemResult with per-metric statistics.

    Raises:
        ValueError: If no common methods or metrics are found.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine methods to compare
    if methods is not None:
        method_list = list(methods)
    else:
        common = set(results_a.keys()) & set(results_b.keys())
        if not common:
            raise ValueError(
                "No common methods between systems. "
                f"System A has: {list(results_a.keys())}. "
                f"System B has: {list(results_b.keys())}."
            )
        method_list = sorted(common)

    metric_similarities: dict[str, float] = {}
    significance: dict[str, PermutationTestResult] = {}
    effect_sizes: dict[str, float] = {}

    for method in method_list:
        agg_a = results_a.get(method)
        agg_b = results_b.get(method)

        if agg_a is None or agg_b is None:
            continue

        # Find common scalar metrics
        common_metrics = set(agg_a.seed_values.keys()) & set(
            agg_b.seed_values.keys()
        )

        for metric in sorted(common_metrics):
            values_a = np.array(agg_a.seed_values[metric])
            values_b = np.array(agg_b.seed_values[metric])

            # Need at least 2 seeds per system for meaningful comparison
            if len(values_a) < 2 or len(values_b) < 2:
                continue

            metric_key = f"{method}.{metric}"

            # Permutation test
            perm_result = permutation_test(
                values_a,
                values_b,
                n_permutations=n_permutations,
                rng=rng,
            )
            significance[metric_key] = perm_result

            # Effect size
            d = effect_size_cohens_d(values_a, values_b)
            effect_sizes[metric_key] = d

            # Similarity: 1 - |d|, clamped to [0, 1]
            similarity = max(0.0, 1.0 - abs(d))
            metric_similarities[metric_key] = similarity

    if not significance:
        raise ValueError(
            "No comparable metrics found between systems. "
            "Check that both have at least 2 seeds and share metric names."
        )

    # Classify as shared or divergent
    shared, divergent = identify_shared_signatures(
        significance, alpha=alpha
    )

    return CrossSystemResult(
        system_a=system_a_name,
        system_b=system_b_name,
        methods=tuple(method_list),
        metric_similarities=metric_similarities,
        significance=significance,
        effect_sizes=effect_sizes,
        shared_signatures=shared,
        divergent_signatures=divergent,
    )


def identify_shared_signatures(
    significance: dict[str, PermutationTestResult],
    alpha: float = 0.05,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Classify metrics as shared or divergent based on significance.

    Args:
        significance: Per-metric permutation test results.
        alpha: Significance threshold. Metrics with p >= alpha are shared;
            metrics with p < alpha are divergent.

    Returns:
        Tuple of (shared_signatures, divergent_signatures).
    """
    shared: list[str] = []
    divergent: list[str] = []

    for metric, result in sorted(significance.items()):
        if result.p_value < alpha:
            divergent.append(metric)
        else:
            shared.append(metric)

    return tuple(shared), tuple(divergent)
