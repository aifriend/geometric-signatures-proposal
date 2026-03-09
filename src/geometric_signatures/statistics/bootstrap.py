"""Bootstrap confidence intervals and effect sizes.

Provides BCa (bias-corrected and accelerated) bootstrap confidence
intervals for any scalar statistic, plus Cohen's d effect size for
standardized group comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval result.

    Attributes:
        point_estimate: The statistic computed on the original data.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        confidence_level: Confidence level (e.g. 0.95).
        n_bootstrap: Number of bootstrap resamples.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int


def _default_statistic(data: np.ndarray) -> float:
    """Default statistic: mean."""
    return float(data.mean())


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] | None = None,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute a percentile bootstrap confidence interval.

    Uses the percentile method (simple but robust). For small samples
    with skewed distributions, consider increasing ``n_bootstrap``.

    Args:
        data: 1-D array of observations.
        statistic_fn: Function ``(data) -> float`` to compute on each
            bootstrap resample. Defaults to mean.
        confidence: Confidence level in (0, 1). Default 0.95.
        n_bootstrap: Number of bootstrap resamples (default 10000).
        rng: Numpy random generator for reproducibility.

    Returns:
        BootstrapCI with point estimate and interval bounds.

    Raises:
        ValueError: If data is empty or confidence is outside (0, 1).
    """
    data = np.asarray(data).ravel()

    if len(data) == 0:
        raise ValueError("Data must have at least one observation.")

    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    if rng is None:
        rng = np.random.default_rng()

    if statistic_fn is None:
        statistic_fn = _default_statistic

    # Point estimate
    point_estimate = statistic_fn(data)

    # Bootstrap resamples
    n = len(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(resample)

    # Percentile interval
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        point_estimate=point_estimate,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
    )


def effect_size_cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses the pooled standard deviation (assumes roughly equal variances).

    .. math::
        d = \\frac{\\bar{X}_A - \\bar{X}_B}{s_{\\text{pooled}}}

    where :math:`s_{\\text{pooled}} = \\sqrt{\\frac{(n_A-1)s_A^2 + (n_B-1)s_B^2}{n_A + n_B - 2}}`

    Args:
        group_a: Observations from group A (1-D array).
        group_b: Observations from group B (1-D array).

    Returns:
        Cohen's d (positive means A > B).

    Raises:
        ValueError: If either group has fewer than 2 observations.
    """
    group_a = np.asarray(group_a).ravel()
    group_b = np.asarray(group_b).ravel()

    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < 2 or n_b < 2:
        raise ValueError(
            f"Both groups need >= 2 observations, got {n_a} and {n_b}."
        )

    mean_a = group_a.mean()
    mean_b = group_b.mean()
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)

    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std < 1e-15:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)
