"""Two-sample permutation tests for geometric signature comparison.

Used to answer: "Is the geometric signature of variant A significantly
different from variant B?" — e.g., does ablating attractor dynamics
degrade participation ratio relative to the complete model?

The test shuffles group labels N times, computes the test statistic on
each permutation, and derives a p-value from the null distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a two-sample permutation test.

    Attributes:
        observed_statistic: The test statistic computed on the real data.
        p_value: Two-sided p-value from the permutation distribution.
        null_distribution: Array of test statistics under the null.
        n_permutations: Number of permutations performed.
    """

    observed_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_permutations: int


def _default_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Default test statistic: difference in means."""
    return float(a.mean() - b.mean())


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Two-sample permutation test.

    Compares two groups by shuffling their labels ``n_permutations`` times
    and computing the test statistic on each permutation to build a null
    distribution.

    Args:
        group_a: Observations from group A (1-D array).
        group_b: Observations from group B (1-D array).
        statistic_fn: Function ``(a, b) -> float`` that computes the test
            statistic. Defaults to difference in means.
        n_permutations: Number of random permutations (default 1000).
        rng: Numpy random generator for reproducibility.

    Returns:
        PermutationTestResult with observed statistic, p-value, and
        null distribution.

    Raises:
        ValueError: If either group is empty.
    """
    group_a = np.asarray(group_a).ravel()
    group_b = np.asarray(group_b).ravel()

    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError("Both groups must have at least one observation.")

    if rng is None:
        rng = np.random.default_rng()

    if statistic_fn is None:
        statistic_fn = _default_statistic

    # Observed statistic
    observed = statistic_fn(group_a, group_b)

    # Pool and permute
    pooled = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    null_stats = np.empty(n_permutations)

    for i in range(n_permutations):
        rng.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        null_stats[i] = statistic_fn(perm_a, perm_b)

    # Two-sided p-value: fraction of null statistics at least as extreme
    p_value = float(np.mean(np.abs(null_stats) >= np.abs(observed)))

    # Ensure p-value is at least 1/(n_permutations+1) — never exactly 0
    p_value = max(p_value, 1.0 / (n_permutations + 1))

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=p_value,
        null_distribution=null_stats,
        n_permutations=n_permutations,
    )
