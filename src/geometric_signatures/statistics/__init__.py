"""Statistical testing framework for geometric signature analysis.

Provides:
- **Permutation tests**: Non-parametric hypothesis tests for comparing
  geometric signatures between variants.
- **Bootstrap confidence intervals**: BCa bootstrap CIs for any scalar
  statistic, plus Cohen's d effect sizes.
- **Multiple comparison correction**: FDR (Benjamini-Hochberg), Bonferroni,
  and Holm step-down procedures.
- **Multi-seed aggregation**: Combine analysis results across seeds into
  mean ± SEM with bootstrap CIs.
"""

from .aggregation import AggregatedResult, aggregate_across_seeds, compare_variants
from .bootstrap import BootstrapCI, bootstrap_confidence_interval, effect_size_cohens_d
from .correction import bonferroni_correction, fdr_correction, holm_correction
from .permutation import PermutationTestResult, permutation_test

__all__ = [
    # Permutation
    "PermutationTestResult",
    "permutation_test",
    # Bootstrap
    "BootstrapCI",
    "bootstrap_confidence_interval",
    "effect_size_cohens_d",
    # Correction
    "fdr_correction",
    "bonferroni_correction",
    "holm_correction",
    # Aggregation
    "AggregatedResult",
    "aggregate_across_seeds",
    "compare_variants",
]
