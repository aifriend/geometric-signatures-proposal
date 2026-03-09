"""Composable pipeline for end-to-end experiments.

Stages:
1. **generate_variants**: Create motif switch configurations.
2. **train**: Train RNN with multi-seed repetition.
3. **preprocess**: PCA denoising / normalization / trial averaging.
4. **analyze**: Run analysis methods (MARBLE, PH, RSA, CKA, geometry).
5. **aggregate**: Combine results across seeds with bootstrap CIs.
6. **compare**: Statistical comparison between variants.

The ``runner`` module wires stages together and supports partial execution
(e.g., skip training when checkpoints already exist).
"""

from .runner import PipelineOptions, PipelineResult, run_pipeline
from .stages import (
    stage_aggregate,
    stage_analyze,
    stage_compare,
    stage_generate_variants,
    stage_preprocess,
)

__all__ = [
    # Stages
    "stage_generate_variants",
    "stage_preprocess",
    "stage_analyze",
    "stage_aggregate",
    "stage_compare",
    # Runner
    "PipelineOptions",
    "PipelineResult",
    "run_pipeline",
]
