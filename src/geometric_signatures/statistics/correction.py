"""Multiple comparison correction methods.

When comparing 4 motifs × 4 tasks × 5 methods = 80 hypothesis tests,
controlling the false positive rate is essential. Provides:

- **FDR (Benjamini-Hochberg)**: Controls false discovery rate — recommended
  for exploratory analysis (most comparisons).
- **Bonferroni**: Controls family-wise error rate — conservative, use when
  each test must be independently reliable.
- **Holm (step-down)**: Less conservative than Bonferroni, still controls FWER.
"""

from __future__ import annotations

import numpy as np


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Controls the expected proportion of false discoveries among rejected
    hypotheses at level ``alpha``.

    Args:
        p_values: Array of raw p-values.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of:
        - rejected: Boolean array (True = reject null hypothesis).
        - corrected_p: Adjusted p-values (monotonized).

    Raises:
        ValueError: If any p-value is outside [0, 1].
    """
    p_values = np.asarray(p_values, dtype=float).ravel()
    _validate_p_values(p_values)

    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH adjusted p-values: p_adj(i) = p(i) * n / rank(i)
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks

    # Monotonize: ensure adjusted p-values are non-decreasing from right
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Map back to original order
    corrected_p = np.empty(n)
    corrected_p[sorted_indices] = adjusted

    rejected = corrected_p <= alpha

    return rejected, corrected_p


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni FWER correction.

    Multiplies each p-value by the number of tests. Conservative but
    simple and controls the family-wise error rate.

    Args:
        p_values: Array of raw p-values.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of:
        - rejected: Boolean array (True = reject null hypothesis).
        - corrected_p: Adjusted p-values (capped at 1.0).
    """
    p_values = np.asarray(p_values, dtype=float).ravel()
    _validate_p_values(p_values)

    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    corrected_p = np.clip(p_values * n, 0.0, 1.0)
    rejected = corrected_p <= alpha

    return rejected, corrected_p


def holm_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down FWER correction.

    Less conservative than Bonferroni: starts with the smallest p-value
    and adjusts by decreasing multipliers. Still controls FWER.

    Args:
        p_values: Array of raw p-values.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of:
        - rejected: Boolean array (True = reject null hypothesis).
        - corrected_p: Adjusted p-values (monotonized).
    """
    p_values = np.asarray(p_values, dtype=float).ravel()
    _validate_p_values(p_values)

    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Holm multipliers: (n, n-1, ..., 1)
    multipliers = np.arange(n, 0, -1)
    adjusted = sorted_p * multipliers

    # Monotonize: ensure adjusted p-values are non-decreasing
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    corrected_p = np.empty(n)
    corrected_p[sorted_indices] = adjusted

    rejected = corrected_p <= alpha

    return rejected, corrected_p


def _validate_p_values(p_values: np.ndarray) -> None:
    """Check all p-values are in [0, 1]."""
    if len(p_values) > 0 and (np.any(p_values < 0) or np.any(p_values > 1)):
        raise ValueError("All p-values must be in [0, 1].")
